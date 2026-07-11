"""Insertion gate — decides per scan whether the current pose is trustworthy
enough to insert sensor data into the map (mapping with known poses).

ROS-agnostic: the node feeds sensor states in via feed_*() and calls
evaluate(); all times are float seconds on one common clock (ROS time).

Modes:
  fused (default)  Map from the FUSED pose authority (TF map->base_link =
                   EKF wheel+VO+licp + RTK/rtabmap bridge). Blocks only on
                   missing/stale pose and physically impossible jumps.
                   RTK/heading quality is still measured and logged, but
                   does not block.
  rtk              Strict: additionally requires RTK FIXED (ublox NavPVT
                   carrier solution), hAcc/vAcc under thresholds, fresh
                   navsat altitude and dual-antenna heading agreement.
  off              Blocks everything.
  ('force_on' is accepted as an alias of fused for compatibility.)

NOTE on navsat: /odometry/gps lives in navsat's OWN cartesian datum (origin
= first fix after start), NOT the map frame — absolute XY comparison against
the map pose is meaningless (measured ~6 m constant offset), which is why
there is no absolute position innovation gate. The gps odometry is used
ONLY for altitude, via z_correction(): the last known (gps.z - tf.z) offset,
held frozen through GPS outages (the EKF does not estimate z, so tf.z stays
constant indoors and the frozen offset remains valid on flat ground).
"""

import math
from dataclasses import dataclass, field


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ublox NavPVT flags bits
FLAGS_GNSS_FIX_OK = 0x01
CARR_SOLN_SHIFT = 6
CARR_SOLN_MASK = 0x3
CARR_SOLN_FIXED = 2


@dataclass
class GateConfig:
    mode: str = 'fused'               # fused | rtk | off
    max_hacc_mm: float = 50.0
    max_vacc_mm: float = 80.0
    max_gps_age_s: float = 1.5
    max_gps_odom_age_s: float = 1.5
    max_heading_diff_deg: float = 8.0
    heading_offset_deg: float = 0.0
    max_heading_age_s: float = 1.5
    max_pose_age_s: float = 0.7
    max_speed_mps: float = 1.0
    jump_margin_m: float = 0.25
    jump_cooldown_s: float = 5.0
    map_corr_xy_m: float = 0.08       # map->odom step = localization correction
    map_corr_yaw_deg: float = 2.0
    rtk_pose_src: str = 'SAT'         # rtk mode: required pose owner
                                      # (mapping must NOT ride on rtabmap poses)
    max_pose_src_age_s: float = 2.0


@dataclass
class GateDecision:
    passed: bool
    reasons: list                     # empty when passed
    info: dict = field(default_factory=dict)


class InsertionGate:
    def __init__(self, config=None):
        self.cfg = config or GateConfig()
        # latest sensor states: (t, ...) or None
        self._navpvt = None           # (t, fix_ok, carr_fixed, hacc_mm, vacc_mm)
        self._heading = None          # (t, yaw_rad)
        self._heading_navpvt = None   # (t, carrier_fixed) dual-antenna health
        self._pose = None             # (t, x, y, z, yaw)  TF map->base_link
        self._gps_odom = None         # (t, x, y, z)       navsat odometry/gps
        self._z_corr = 0.0            # last RTK-grade gps.z - tf.z (frozen otherwise)
        self._z_corr_t = None         # time of last RTK-grade update
        self._cooldown_until = -1.0
        self._last_jump_m = 0.0
        self._map_odom = None         # (t, x, y, yaw) TF map->odom
        self._pose_src = None         # (t, status) from /nav_tf/odom_status
        self.corrections = 0          # localization corrections seen
        self._last_corr_m = 0.0

    @property
    def mode(self):
        return 'fused' if self.cfg.mode == 'force_on' else self.cfg.mode

    # ---- feeders -----------------------------------------------------------
    def feed_navpvt(self, t, flags, hacc_mm, vacc_mm):
        fix_ok = bool(flags & FLAGS_GNSS_FIX_OK)
        carr = (flags >> CARR_SOLN_SHIFT) & CARR_SOLN_MASK
        self._navpvt = (t, fix_ok, carr == CARR_SOLN_FIXED, float(hacc_mm), float(vacc_mm))

    def feed_heading(self, t, yaw_rad):
        self._heading = (t, yaw_rad)

    def feed_heading_navpvt(self, t, flags):
        """Dual-antenna RECEIVER health: without a carrier solution the
        heading silently degrades to IMU (nav_fused keeps publishing!) and
        the whole GPS chain stays blocked by design (heading-dependent
        navsat datum) — surface it instead of failing silently
        (2026-07-06 evening: 40+ min invisible outage)."""
        carr = (flags >> CARR_SOLN_SHIFT) & CARR_SOLN_MASK
        self._heading_navpvt = (t, carr == CARR_SOLN_FIXED)

    def feed_pose(self, t, x, y, z, yaw):
        prev = self._pose
        if prev is not None:
            dt = t - prev[0]
            if dt > 0:
                d = math.hypot(x - prev[1], y - prev[2])
                allowed = self.cfg.max_speed_mps * dt + self.cfg.jump_margin_m
                if d > allowed:
                    self._cooldown_until = t + self.cfg.jump_cooldown_s
                    self._last_jump_m = d
        self._pose = (t, x, y, z, yaw)

    def feed_map_odom(self, t, x, y, yaw):
        """Watchdog on the map->odom transform: a step here IS a
        localization correction (RTK anchor, rtabmap loop closure) — data
        inserted before it sits at the old place, so pause insertion for
        the cooldown and count it (the UI surfaces the count)."""
        prev = self._map_odom
        self._map_odom = (t, x, y, yaw)
        if prev is None:
            return
        d = math.hypot(x - prev[1], y - prev[2])
        dyaw = abs(math.degrees(wrap_angle(yaw - prev[3])))
        if d > self.cfg.map_corr_xy_m or dyaw > self.cfg.map_corr_yaw_deg:
            self.corrections += 1
            self._last_corr_m = d
            self._cooldown_until = max(self._cooldown_until,
                                       t + self.cfg.jump_cooldown_s)

    def feed_pose_src(self, t, status):
        """Who owns the map pose right now (navi_transform status: SAT =
        RTK anchor, DOCK, rtabmap...). Mapping v3 is rtabmap-free by design:
        rtk mode refuses insertion unless the owner is rtk_pose_src."""
        self._pose_src = (t, status)

    def feed_gps_odom(self, t, x, y, z):
        self._gps_odom = (t, x, y, z)
        # Update the z offset ONLY from RTK-grade altitude. Degraded GPS
        # (vAcc in meters, e.g. indoors through a door) would re-shift every
        # insertion differently and smear one floor across several voxel
        # layers — which the band classifier then reads as obstacles. A
        # frozen offset keeps insertions self-consistent instead.
        if self._pose is None or self._navpvt is None:
            return
        npv_t, fix_ok, carr_fixed, hacc, vacc = self._navpvt
        if (t - npv_t > self.cfg.max_gps_age_s
                or not (fix_ok and carr_fixed)
                or vacc > self.cfg.max_vacc_mm):
            return
        new = z - self._pose[3]
        if self._z_corr_t is None:
            self._z_corr = new
        else:
            self._z_corr += 0.3 * (new - self._z_corr)   # smooth cm jitter
        self._z_corr_t = t

    # ---- helpers -----------------------------------------------------------
    def z_correction(self):
        """What to ADD to a TF-derived z to get navsat altitude. Holds the
        last known offset through GPS outages; 0.0 until first GPS."""
        return self._z_corr

    # ---- decision ----------------------------------------------------------
    def evaluate(self, now):
        cfg = self.cfg
        mode = self.mode
        reasons = []
        info = {'mode': mode}

        if mode == 'off':
            return GateDecision(False, ['mode_off'], info)

        # ---- quality metrics: always computed, block only in rtk mode ------
        rtk_reasons = []
        if self._navpvt is None:
            rtk_reasons.append('no_gps')
        else:
            t, fix_ok, carr_fixed, hacc, vacc = self._navpvt
            gps_age = now - t
            info.update({'gps_age_s': round(gps_age, 2), 'rtk_fixed': carr_fixed,
                         'hacc_mm': hacc, 'vacc_mm': vacc})
            if gps_age > cfg.max_gps_age_s:
                rtk_reasons.append('gps_stale')
            if not (fix_ok and carr_fixed):
                rtk_reasons.append('no_rtk_fixed')
            if hacc > cfg.max_hacc_mm:
                rtk_reasons.append('hacc')
            if vacc > cfg.max_vacc_mm:
                rtk_reasons.append('vacc')

        if self._z_corr_t is None:
            info['z_source'] = 'none'
            rtk_reasons.append('no_rtk_altitude')
        elif now - self._z_corr_t <= cfg.max_gps_odom_age_s:
            info['z_source'] = 'rtk'
        else:
            info['z_source'] = 'frozen'
            rtk_reasons.append('altitude_frozen')
        info['z_corr_m'] = round(self._z_corr, 3)

        if self._pose_src is None:
            info['pose_src'] = '?'
            rtk_reasons.append('pose_src_unknown')
        else:
            t, src = self._pose_src
            stale = now - t > cfg.max_pose_src_age_s
            info['pose_src'] = src
            if stale:
                rtk_reasons.append('pose_src_stale')
            elif src != cfg.rtk_pose_src:
                rtk_reasons.append('pose_not_' + cfg.rtk_pose_src.lower())

        if self._heading_navpvt is None:
            info['heading_rtk'] = None
        else:
            t, carr_fixed = self._heading_navpvt
            fresh = now - t <= cfg.max_heading_age_s
            info['heading_rtk'] = bool(carr_fixed and fresh)
            if not info['heading_rtk']:
                rtk_reasons.append('heading_receiver_no_rtk')

        if self._heading is None:
            rtk_reasons.append('no_heading')
        else:
            t, yaw = self._heading
            age = now - t
            info['heading_age_s'] = round(age, 2)
            if age > cfg.max_heading_age_s:
                rtk_reasons.append('heading_stale')
            elif self._pose is not None:
                diff = wrap_angle(yaw + math.radians(cfg.heading_offset_deg)
                                  - self._pose[4])
                info['heading_diff_deg'] = round(math.degrees(diff), 1)
                if abs(math.degrees(diff)) > cfg.max_heading_diff_deg:
                    rtk_reasons.append('heading_diff')

        # ---- blocking gates -------------------------------------------------
        if self._pose is None:
            reasons.append('no_pose')
        else:
            pose_age = now - self._pose[0]
            info['pose_age_s'] = round(pose_age, 2)
            if pose_age > cfg.max_pose_age_s:
                reasons.append('pose_stale')
        if now < self._cooldown_until:
            reasons.append('pose_jump_cooldown')
            info['last_jump_m'] = round(max(self._last_jump_m,
                                            self._last_corr_m), 2)
        info['map_corrections'] = self.corrections

        if mode == 'rtk':
            reasons.extend(rtk_reasons)
        else:
            info['quality_flags'] = rtk_reasons   # visible, non-blocking

        return GateDecision(not reasons, reasons, info)
