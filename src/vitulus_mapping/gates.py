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
import threading
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
    # Rate-based pause (sync fix 2026-07-12): a single map->odom step rarely
    # exceeds map_corr_xy_m when the EDT tracker slews the correction gently at
    # ~5 Hz in 2-11 cm bites, so the per-step watchdog above almost never
    # fires — yet the CUMULATIVE shift over a cloud's TF-lookup window is large
    # enough to smear inserted points into ghost obstacles. Pause insertion
    # whenever the |map->odom| translation accumulated over the last
    # map_corr_window_s exceeds max_maporr_rate_mps * window (i.e. the map
    # frame is being actively corrected). GPS-owned stable driving (near-zero
    # correction rate) passes; tracker-corrected blind passes pause.
    max_maporr_rate_mps: float = 0.02  # 20 mm/s cumulative correction budget
    map_corr_window_s: float = 1.0
    rtk_pose_src: str = 'SAT'         # rtk mode: required pose owner
                                      # (mapping must NOT ride on rtabmap poses)
    max_pose_src_age_s: float = 2.0
    # z-correction mode (dataprep 2026-07-12). 'local' = use the CURRENT
    # RTK-grade (gps.z - tf.z) at sample time, position-tied, frozen on
    # outage — tracks real ground undulation as the robot drives. 'global' =
    # the old single heavily-smoothed scalar for the whole site (fine only on
    # near-flat ground). Default local.
    z_corr_mode: str = 'local'
    # per-source runtime range caps (item 3). Points beyond these (from the
    # sensor origin) are dropped by the NODE before forwarding to octomap;
    # the gate just carries the values for echo/persist.
    lidar_max_range_m: float = 8.0
    depth_max_range_m: float = 2.8
    # --- phantom elevated-geometry hardening (H-POSE-Z, 2026-07-19) -----------
    # ROOT CAUSE (novaTestovaciMapa 2026-07-19): the inserted cloud z is
    # TF-z (flat, EKF does not estimate z) + z_correction (RTK altitude - TF-z).
    # During a GPS-marginal window the RTK altitude EXCURSED ~+0.5 m (still
    # rtk_fixed, good vAcc — precision != accuracy) and z_correction faithfully
    # tracked it, so ~1300 ground clouds were painted 0.5 m too high. Where that
    # coverage meets the correctly-placed ground a phantom STEP appears on flat
    # concrete. A smaller frozen-z-while-moving error (-0.28 m) occurred too.
    # Three layered, slope-safe guards (all opt-in; 0/False == legacy):
    # (1) accept z_correction updates ONLY when the localization arbiter is
    #     trustworthy (probation_proven), not merely when raw carrier/vAcc look
    #     good — this is what distinguishes the marginal excursion. Mirrors the
    #     datum-capture gating already proven for the dock-departure residual.
    z_corr_require_trust: bool = False
    # (2) slew guard: real terrain moves z_correction at most ~speed*grade; a
    #     RTK re-fix / marginal excursion moves it faster. Clamp the accepted
    #     change to max_rate * dt. NOT flat-world: continuous slope survives.
    z_corr_max_rate_mps: float = 0.0        # 0 = off; 0.25 recommended
    # (3) block insertion when z is not confidently fresh AND the robot is
    #     MOVING (a stale/frozen z is only valid where it froze; driving to new
    #     terrain with it smears the ground). Stationary re-observation is fine.
    z_stale_block_when_moving: bool = False
    max_z_corr_age_s: float = 8.0
    # (3b) TRAILING-LAWN PHANTOM fix (treelawn 2026-07-19). On a slope the RTK
    #     vertical solution flaps (measured outages 1.2-38 s / up to 3.27 m
    #     driven); with z_corr_require_trust the accepted z updates go sparse, so
    #     z_source reads 'frozen' (age > max_gps_odom_age_s) yet age stays under
    #     the 8 s max_z_corr_age_s grace and the moving-block above REOPENS —
    #     clouds keep inserting with a frozen/lagging z while the robot climbs.
    #     The DEM (wheel-contact z) for the same cell was stamped at a different
    #     z-freeze state, so fresh lawn lands 0.10-0.20 m 'above ground' trailing
    #     the robot => a phantom obstacle ring along the just-driven uphill.
    #     A TIME grace cannot bound this (error = grade * distance, not time), so
    #     bound the DISTANCE a frozen z may be carried while moving: block once the
    #     robot has driven more than z_stale_max_dist_m from where z was last
    #     confidently updated. Slope-safe (no flat-world assumption — it caps the
    #     geographic reach of a stale z, so residual error <= grade *
    #     z_stale_max_dist_m; e.g. 5% * 0.5 m = 0.025 m, under band_min). Needs
    #     z_stale_block_when_moving=True. 0.0 = off (legacy); 0.5 recommended.
    z_stale_max_dist_m: float = 0.0
    # standstill dwell throttle (applied in the node): a near-stationary robot
    # re-inserting the same view hammers voxels and amplifies any z bias.
    standstill_speed_eps_mps: float = 0.05
    standstill_after_s: float = 8.0
    standstill_rate_hz: float = 0.5


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
        self._z_corr_xy = None        # pose (x, y) at last accepted z update
                                      # (treelawn: bound frozen-z carry distance)
        self._cooldown_until = -1.0
        self._last_jump_m = 0.0
        self._map_odom = None         # (t, x, y, yaw) TF map->odom
        self._pose_src = None         # (t, status) from /nav_tf/odom_status
        self.corrections = 0          # localization corrections seen
        self._last_corr_m = 0.0
        # sliding window of (t, step_m) map->odom translation deltas, for the
        # cumulative correction-rate pause
        self._corr_hist = []
        self._corr_rate_mps = 0.0     # last computed windowed rate
        # H-POSE-Z hardening state
        self._speed_mps = 0.0         # EMA of |v| from feed_pose (dwell throttle)
        self._still_since = None      # t since speed dropped below eps
        self._loc_ok = None           # arbiter trust flag from node; None=unknown
        # M2: feed_map_odom (5 Hz tick thread) and evaluate (tick thread + 3
        # concurrent cloud subscriber threads via the stamped path) both touch
        # _corr_hist/_corr_rate_mps. Guard them; only the tick path prunes.
        self._corr_lock = threading.Lock()

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
                # instantaneous speed (EMA) for the standstill dwell throttle
                # and the z-stale-while-moving insertion gate.
                self._speed_mps += 0.5 * (d / dt - self._speed_mps)
                if self._speed_mps < self.cfg.standstill_speed_eps_mps:
                    if self._still_since is None:
                        self._still_since = t
                else:
                    self._still_since = None
                allowed = self.cfg.max_speed_mps * dt + self.cfg.jump_margin_m
                if d > allowed:
                    self._cooldown_until = t + self.cfg.jump_cooldown_s
                    self._last_jump_m = d
        self._pose = (t, x, y, z, yaw)

    def set_loc_trust(self, ok):
        """Node feeds the localization arbiter's trust (probation_proven etc.).
        None = unknown (older navi_transform) -> legacy behaviour."""
        self._loc_ok = ok

    def speed_mps(self):
        return self._speed_mps

    def standstill_s(self, now):
        """Seconds below standstill_speed_eps (0.0 if moving/unknown)."""
        if self._still_since is None:
            return 0.0
        return max(0.0, now - self._still_since)

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
        # cumulative correction-rate tracking: accumulate every step (however
        # small) over a sliding window; the sum / window is the rate the
        # evaluate() pause gates on. This catches the gentle multi-Hz tracker
        # slew that never trips the per-step watchdog above. (M2: guarded.)
        with self._corr_lock:
            self._corr_hist.append((t, d))
            cutoff = t - self.cfg.map_corr_window_s
            while self._corr_hist and self._corr_hist[0][0] < cutoff:
                self._corr_hist.pop(0)
            win = max(self.cfg.map_corr_window_s, 1e-3)
            self._corr_rate_mps = sum(s for _tt, s in self._corr_hist) / win

    def corr_rate_mps(self):
        """Cumulative |map->odom| translation over the last map_corr_window_s,
        expressed as m/s. Non-zero while localization is actively correcting
        the map frame (tracker slew / RTK anchor pull-in)."""
        return self._corr_rate_mps

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
        # H-POSE-Z guard (1): accept the RTK altitude into z_corr ONLY when the
        # localization arbiter is trustworthy. A marginal-window excursion
        # (rtk_fixed + good vAcc but WRONG absolute altitude) is what smeared
        # the ground +0.5 m on 2026-07-19; probation_proven is False in exactly
        # those windows. None (unknown) keeps legacy behaviour.
        if self.cfg.z_corr_require_trust and self._loc_ok is False:
            return                       # freeze z_corr through untrusted window
        new = z - self._pose[3]
        gain = 0.6 if self.cfg.z_corr_mode == 'local' else 0.3
        if self._z_corr_t is None:
            self._z_corr = new
        else:
            target = self._z_corr + gain * (new - self._z_corr)
            # H-POSE-Z guard (2): slew-limit the accepted change. Real slope
            # moves z at speed*grade (slow); an altitude re-fix teleports.
            if self.cfg.z_corr_max_rate_mps > 0.0:
                dt = max(1e-3, t - self._z_corr_t)
                max_step = self.cfg.z_corr_max_rate_mps * dt
                dz = target - self._z_corr
                if abs(dz) > max_step:
                    dz = math.copysign(max_step, dz)
                self._z_corr += dz
            else:
                self._z_corr = target
        self._z_corr_t = t
        # remember WHERE this z was confirmed, so evaluate() can block once the
        # robot has carried the (soon-frozen) z too far over new terrain.
        self._z_corr_xy = (self._pose[1], self._pose[2])

    # ---- helpers -----------------------------------------------------------
    def z_correction(self):
        """What to ADD to a TF-derived z to get navsat altitude. Holds the
        last known offset through GPS outages; 0.0 until first GPS."""
        return self._z_corr

    # ---- decision ----------------------------------------------------------
    def evaluate(self, now, stamped_pose=None, wall_now=None):
        """Evaluate the insertion decision at time `now`.

        stamped_pose (item 7, gate-at-cloud-stamp): when the caller has looked
        up TF map->base_link at a SPECIFIC cloud timestamp, it passes that pose
        as (t, x, y, z, yaw) and `now`=that stamp. The pose-quality blocking
        gates (pose freshness, heading agreement) then judge the pose that
        actually produced the cloud, not the latest 5 Hz tick — so a cloud
        stamped mid-correction is rejected on ITS own pose, not a newer good
        one. When None, the latest fed pose is used (5 Hz tick path).

        wall_now (M11): the sensor-QUALITY feeds (navpvt / heading / gps-odom /
        pose-src) are timestamped on the RECEIPT (wall) clock, so their
        staleness must be judged against wall-clock now — not the cloud stamp,
        which on the stamped path lies in the past and yields spuriously
        NEGATIVE ages that never trip the rtk-mode staleness gates. The stamped
        path passes wall_now=Time.now(); the 5 Hz tick path leaves it None (so
        wall_now==now and behaviour is byte-identical). Pose freshness keeps
        using `now` because the pose IS the stamped pose."""
        cfg = self.cfg
        mode = self.mode
        reasons = []
        info = {'mode': mode}
        pose = stamped_pose if stamped_pose is not None else self._pose
        if wall_now is None:
            wall_now = now

        if mode == 'off':
            return GateDecision(False, ['mode_off'], info)

        # ---- quality metrics: always computed, block only in rtk mode ------
        rtk_reasons = []
        if self._navpvt is None:
            rtk_reasons.append('no_gps')
        else:
            t, fix_ok, carr_fixed, hacc, vacc = self._navpvt
            gps_age = wall_now - t          # M11: receipt-clock age
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
        elif wall_now - self._z_corr_t <= cfg.max_gps_odom_age_s:   # M11
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
            stale = wall_now - t > cfg.max_pose_src_age_s    # M11
            info['pose_src'] = src
            if stale:
                rtk_reasons.append('pose_src_stale')
            elif src != cfg.rtk_pose_src:
                rtk_reasons.append('pose_not_' + cfg.rtk_pose_src.lower())

        if self._heading_navpvt is None:
            info['heading_rtk'] = None
        else:
            t, carr_fixed = self._heading_navpvt
            fresh = wall_now - t <= cfg.max_heading_age_s    # M11
            info['heading_rtk'] = bool(carr_fixed and fresh)
            if not info['heading_rtk']:
                rtk_reasons.append('heading_receiver_no_rtk')

        if self._heading is None:
            rtk_reasons.append('no_heading')
        else:
            t, yaw = self._heading
            age = wall_now - t              # M11: receipt-clock age
            info['heading_age_s'] = round(age, 2)
            if age > cfg.max_heading_age_s:
                rtk_reasons.append('heading_stale')
            elif pose is not None:
                diff = wrap_angle(yaw + math.radians(cfg.heading_offset_deg)
                                  - pose[4])
                info['heading_diff_deg'] = round(math.degrees(diff), 1)
                if abs(math.degrees(diff)) > cfg.max_heading_diff_deg:
                    rtk_reasons.append('heading_diff')

        # ---- blocking gates -------------------------------------------------
        if pose is None:
            reasons.append('no_pose')
        else:
            pose_age = now - pose[0]
            info['pose_age_s'] = round(pose_age, 2)
            if pose_age > cfg.max_pose_age_s:
                reasons.append('pose_stale')
        if now < self._cooldown_until:
            reasons.append('pose_jump_cooldown')
            info['last_jump_m'] = round(max(self._last_jump_m,
                                            self._last_corr_m), 2)
        info['map_corrections'] = self.corrections

        # H-POSE-Z guard (3): block insertion when the ground-z reference is not
        # confidently fresh AND the robot is MOVING. A stale/frozen z_corr is
        # only valid where it froze; driving to new terrain with it inserts the
        # ground at the wrong height (the -0.28 m frozen-window error, and the
        # tail of the excursion). Stationary re-observation of the same spot is
        # allowed (speed below eps). Slope-safe: it gates on z FRESHNESS, never
        # on z being non-zero. Opt-in (default off).
        if self.cfg.z_stale_block_when_moving:
            moving = self._speed_mps >= self.cfg.standstill_speed_eps_mps
            z_age = (float('inf') if self._z_corr_t is None
                     else wall_now - self._z_corr_t)
            # treelawn (3b): also block when the frozen z has been CARRIED too
            # far over new terrain (distance bound), even if still within the
            # time grace — this is the slope hole the 8 s time grace leaves open.
            z_far = False
            z_carry_m = None
            if (cfg.z_stale_max_dist_m > 0.0 and moving
                    and self._z_corr_xy is not None and pose is not None):
                z_carry_m = math.hypot(pose[1] - self._z_corr_xy[0],
                                       pose[2] - self._z_corr_xy[1])
                z_far = z_carry_m > cfg.z_stale_max_dist_m
            if moving and (z_age > cfg.max_z_corr_age_s or z_far):
                reasons.append('z_stale_moving')
                info['z_corr_age_s'] = (None if self._z_corr_t is None
                                        else round(z_age, 2))
                if z_carry_m is not None:
                    info['z_carry_m'] = round(z_carry_m, 2)

        # rate-based pause: don't map while the map frame is actively being
        # corrected. Recompute the windowed rate against `now` so it decays to
        # 0 once feed_map_odom stops delivering corrections (stale samples age
        # out of the window even without a new feed).
        # M2: only the 5 Hz tick path (stamped_pose is None) prunes the shared
        # window; the concurrent cloud (stamped) threads compute the rate from
        # a read-only snapshot under the lock, mutating nothing.
        cutoff = now - cfg.map_corr_window_s
        win = max(cfg.map_corr_window_s, 1e-3)
        with self._corr_lock:
            if stamped_pose is None:
                while self._corr_hist and self._corr_hist[0][0] < cutoff:
                    self._corr_hist.pop(0)
                rate = sum(s for _tt, s in self._corr_hist) / win
                self._corr_rate_mps = rate
            else:
                rate = sum(s for _tt, s in self._corr_hist
                           if _tt >= cutoff) / win
        info['map_corr_rate_mmps'] = round(rate * 1000.0, 1)
        if rate > cfg.max_maporr_rate_mps:
            reasons.append('map_corr_rate')

        if mode == 'rtk':
            reasons.extend(rtk_reasons)
        else:
            info['quality_flags'] = rtk_reasons   # visible, non-blocking

        return GateDecision(not reasons, reasons, info)
