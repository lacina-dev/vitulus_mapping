"""Mapping profiles (shared, ROS-free; decisions of the owner 2026-11-01).

One named set for mapping, chosen in the Map tab, per site:

  camera   the mapping profile `wide` of vitulus_depth_obstacles (partial: a key
           that is not named keeps the default of the node)
  mapper   direct_raster: the lidar for mapping, the sources, the evidence model

Files, all written by mapping_manager only:

  <base>/mapping_profiles.json        custom profiles, global (one vocabulary)
                                      {version, profiles: {name: {camera, mapper}}}
  <base>/<site>/mapping_profile.json  {version, active, camera, mapper}: the
                                      profile of this site + what differs from it
  <base>/mapping_profile.json         the same while no map is served; a new
                                      site starts as a copy of the current state
  <base>/<site>/direct_overrides.json generated: effective mapper values, read by
                                      direct_raster at its start

Nothing here reads ROS: mapping_manager brings the live node in as plain values.
Replaces direct_presets.json and hand-kept direct_overrides.json; both are
converted on first start without losing a value (migrate_all).
"""

import json
import os

from vitulus_mapping import presets

VERSION = 1
SITE_FILE = 'mapping_profile.json'
CUSTOM_FILE = 'mapping_profiles.json'
OVERRIDES_FILE = 'direct_overrides.json'
PENDING_FILE = 'direct_pending.json'
PRE_SUFFIX = '.pre_profiles'
WIDE_STORE = '~/.vitulus/depth_obstacles/settings.json'
DEFAULT_PROFILE = 'Default (yaml)'
EPS = 1e-6

LIDAR_KEYS = ('use_lidar', 'lidar_max_range_m', 'lidar_free_no_hit',
              'lidar_min_intensity', 'lidar_free_dec')
SOURCE_KEYS = ('use_cam_obstacle', 'use_cam_free')
EVIDENCE_KEYS = ('min_hits', 'hit_inc', 'miss_dec', 'occ_erode_factor',
                 'occ_thresh', 'free_thresh', 'lo_clamp', 'hi_clamp')
MAPPER_KEYS = LIDAR_KEYS + SOURCE_KEYS + EVIDENCE_KEYS

# = mapping_v3.yaml direct_raster (test_profiles.py compares them)
MAPPER_DEFAULTS = {
    'use_lidar': True, 'lidar_max_range_m': 8.0, 'lidar_free_no_hit': True,
    'lidar_min_intensity': 0.0, 'lidar_free_dec': 0.12,
    'use_cam_obstacle': True, 'use_cam_free': True,
    'min_hits': 3, 'hit_inc': 0.5, 'miss_dec': 0.4, 'occ_erode_factor': 0.5,
    'occ_thresh': 0.85, 'free_thresh': -0.4, 'lo_clamp': -2.0, 'hi_clamp': 3.5,
}
# = vitulus_depth_obstacles config/d435_ros1.yaml profile wide; the live node's
# defaults win as soon as it answers
CAMERA_DEFAULTS = {
    'max_depth': 4.0, 'range_max': 0.0, 'max_ground_height': 0.15,
    'min_obstacle_height': -1.0, 'max_obstacle_height': 2.0,
    'min_cluster_size': 2, 'cluster_radius': 0.15,
}
# what the removed second filter of direct_raster stood at by default
LEGACY_DEFAULTS = dict(MAPPER_DEFAULTS, obstacle_max_range_m=2.8,
                       free_max_range_m=2.8, cam_obstacle_min_h=0.15,
                       cam_obstacle_max_h=0.5, cam_min_cluster_pts=1,
                       cam_cluster_radius_m=0.15)


def validate_mapper(v):
    """A complete mapper dict -> (clean dict, None) or (None, reason). The only
    copy of these limits: direct_raster and mapping_manager both call it."""
    try:
        c = {k: bool(v[k]) for k in ('use_lidar', 'lidar_free_no_hit',
                                     'use_cam_obstacle', 'use_cam_free')}
        c['min_hits'] = int(v['min_hits'])
        for k in ('lidar_max_range_m', 'lidar_min_intensity', 'lidar_free_dec',
                  'hit_inc', 'miss_dec', 'occ_erode_factor', 'occ_thresh',
                  'free_thresh', 'lo_clamp', 'hi_clamp'):
            c[k] = float(v[k])
    except (KeyError, TypeError, ValueError) as e:
        return None, 'non-numeric value: %s' % e
    for key, lo, hi in (('lidar_max_range_m', 0.5, 20.0),
                        ('hit_inc', 0.05, 5.0), ('miss_dec', 0.05, 5.0),
                        ('lidar_free_dec', 0.0, 5.0),
                        ('occ_erode_factor', 0.0, 1.0),
                        ('min_hits', 1, 50),
                        ('lidar_min_intensity', 0.0, 1000.0)):
        if not lo <= c[key] <= hi:          # NaN fails too
            return None, '%s out of range %s..%s: %s' % (key, lo, hi, c[key])
    if not c['lo_clamp'] < 0.0 <= c['hi_clamp']:
        return None, ('need lo_clamp<0<=hi_clamp, got [%s, %s]'
                      % (c['lo_clamp'], c['hi_clamp']))
    if not c['free_thresh'] < c['occ_thresh']:
        return None, ('need free_thresh<occ_thresh, got [%s, %s]'
                      % (c['free_thresh'], c['occ_thresh']))
    return c, None


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < EPS
    except (TypeError, ValueError):
        return a == b


def from_legacy(vals, wide=None):
    """An old preset / overrides dict -> {camera, mapper}, keeping what the two
    filters in a row let through: the second filter of direct_raster becomes
    part of the chain, its effect stays. `wide` = values already set on the
    chain (the old global store)."""
    w = dict(CAMERA_DEFAULTS, **(wide or {}))
    cam = dict(wide or {})
    floor = max(w['max_ground_height'], w['min_obstacle_height'])
    hmin = vals.get('cam_obstacle_min_h')
    if hmin is not None and hmin > floor + EPS:
        cam['min_obstacle_height'] = floor = min(1.0, float(hmin))
    hmax = vals.get('cam_obstacle_max_h')
    top = w['max_obstacle_height']
    if hmax is not None and (top <= 0.0 or hmax < top - EPS):
        cam['max_obstacle_height'] = top = max(0.2, float(hmax))
    if 'max_obstacle_height' in cam and floor >= top:
        cam['max_obstacle_height'] = floor + 0.05
    pts = vals.get('cam_min_cluster_pts')
    if pts is not None and pts > w['min_cluster_size']:
        cam['min_cluster_size'] = min(500, int(pts))
    radius = vals.get('cam_cluster_radius_m')
    if pts is not None and pts > 1 and radius is not None:
        cam['cluster_radius'] = max(0.02, min(1.0, float(radius)))
    reach = []
    if vals.get('use_cam_obstacle', True) and 'obstacle_max_range_m' in vals:
        reach.append(float(vals['obstacle_max_range_m']))
    if vals.get('use_cam_free', True) and 'free_max_range_m' in vals:
        reach.append(float(vals['free_max_range_m']))
    far = min([x for x in (w['range_max'], w['max_depth']) if x > 0] or [99.0])
    if reach and max(reach) < far - EPS:
        cam['range_max'] = max(0.5, min(6.0, max(reach)))
    return {'camera': cam,
            'mapper': {k: vals[k] for k in MAPPER_KEYS if k in vals}}


# the old presets did not carry the range: it stood at the default of the mapper
BUILTIN_PROFILES = {
    n: from_legacy(dict(v, obstacle_max_range_m=2.8, free_max_range_m=2.8))
    for n, v in presets.BUILTIN_PRESETS.items()}


def read_json(path):
    """{} when missing, unreadable or not an object."""
    try:
        with open(path) as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _section(doc, key):
    v = doc.get(key)
    return dict(v) if isinstance(v, dict) else {}


class Store:
    """All reading and writing of the files above."""

    def __init__(self, base_dir, wide_store=WIDE_STORE):
        self.base = base_dir
        self.wide_store_path = os.path.expanduser(wide_store)
        self.camera_defaults = dict(CAMERA_DEFAULTS)
        self.custom = self._load_custom()

    # ---- files -------------------------------------------------------------
    def _custom_path(self):
        return os.path.join(self.base, CUSTOM_FILE)

    def _load_custom(self):
        doc = read_json(self._custom_path()).get('profiles')
        out = {}
        for name, p in (doc.items() if isinstance(doc, dict) else ()):
            if isinstance(p, dict):
                out[name] = {'camera': _section(p, 'camera'),
                             'mapper': _section(p, 'mapper')}
        return out

    def _save_custom(self):
        write_json(self._custom_path(),
                   {'version': VERSION, 'profiles': self.custom})

    def site_dir(self, site):
        return os.path.join(self.base, site) if site else self.base

    def state_path(self, site):
        return os.path.join(self.site_dir(site), SITE_FILE)

    def load_state(self, site):
        """{active, camera, mapper}; None when this site has none yet."""
        path = self.state_path(site)
        if not os.path.exists(path):
            return None
        doc = read_json(path)
        active = doc.get('active')
        return {'active': active if isinstance(active, str) and active
                else DEFAULT_PROFILE,
                'camera': _section(doc, 'camera'),
                'mapper': _section(doc, 'mapper')}

    def save_state(self, site, st):
        write_json(self.state_path(site), dict(st, version=VERSION))

    def wide_store(self):
        p = read_json(self.wide_store_path).get('profiles')
        w = p.get('wide') if isinstance(p, dict) else None
        return dict(w) if isinstance(w, dict) else {}

    # ---- values ------------------------------------------------------------
    def profile(self, name):
        """The stored values of a profile; an unknown name is the default one."""
        p = self.custom.get(name) or BUILTIN_PROFILES.get(name) \
            or BUILTIN_PROFILES[DEFAULT_PROFILE]
        return {'camera': dict(p['camera']),
                'mapper': dict(MAPPER_DEFAULTS, **p['mapper'])}

    def effective(self, st):
        p = self.profile(st['active'])
        p['camera'].update(st['camera'])
        p['mapper'].update(st['mapper'])
        return p

    def camera_default(self, key):
        return self.camera_defaults.get(key)

    def diff(self, values, base):
        """What must be kept on top of `base` to get `values`."""
        cam, mp = {}, {}
        vc, bc = values['camera'], base['camera']
        for k in set(vc) | set(bc):
            a = vc.get(k, self.camera_default(k))
            if not same(a, bc.get(k, self.camera_default(k))):
                cam[k] = a
        for k, v in values['mapper'].items():
            if not same(v, base['mapper'].get(k)):
                mp[k] = v
        return {'camera': cam, 'mapper': mp}

    def names(self):
        return list(BUILTIN_PROFILES) + [n for n in self.custom
                                         if n not in BUILTIN_PROFILES]

    def listing(self):
        return [{'name': n, 'builtin': n in BUILTIN_PROFILES,
                 'custom': n in self.custom} for n in self.names()]

    def write_overrides(self, site, st):
        """direct_raster reads this at its start."""
        if not site:
            return
        doc = dict(self.effective(st)['mapper'])
        doc['_profile'] = {'name': st['active'],
                           'camera': self.effective(st)['camera']}
        write_json(os.path.join(self.site_dir(site), OVERRIDES_FILE), doc)

    # ---- changes -----------------------------------------------------------
    def set_mapper(self, st, values):
        """Partial mapper values on top of the state; (ok, reason)."""
        unknown = [k for k in values if k not in MAPPER_KEYS]
        if unknown:
            return False, 'unknown setting: %s' % ', '.join(sorted(unknown))
        eff = dict(self.effective(st)['mapper'], **values)
        clean, err = validate_mapper(eff)
        if err:
            return False, err
        base = self.profile(st['active'])['mapper']
        for k in values:
            if same(clean[k], base.get(k)):
                st['mapper'].pop(k, None)
            else:
                st['mapper'][k] = clean[k]
        return True, None

    def set_camera(self, st, values):
        """Values the chain took (the Camera card); the node validated them."""
        base = self.profile(st['active'])['camera']
        for k, v in values.items():
            if same(v, base.get(k, self.camera_default(k))):
                st['camera'].pop(k, None)
            else:
                st['camera'][k] = v

    def discard(self, st, camera=None, mapper=None):
        """Back to the profile: everything, or only the named keys."""
        if camera is None and mapper is None:
            st['camera'], st['mapper'] = {}, {}
            return
        for k in camera or ():
            st['camera'].pop(k, None)
        for k in mapper or ():
            st['mapper'].pop(k, None)

    def save_profile(self, st, name):
        """The effective values become the profile `name` and the site uses it."""
        name = (name or st['active']).strip()
        if not name or len(name) > 40 or any(ord(c) < 32 for c in name):
            return False, 'bad name'
        self.custom[name] = self.effective(st)
        self._save_custom()
        st.update(active=name, camera={}, mapper={})
        return True, None

    def delete_profile(self, name, keep=None):
        """A custom profile goes; a built-in of that name comes back. A site that
        used a profile which no longer exists stays on its values: they become
        changes of the default profile. `keep` = the state in memory."""
        if name not in self.custom:
            return False, 'not a custom profile'
        users = []
        if name not in BUILTIN_PROFILES:
            for site in [None] + self.sites():
                st = self.load_state(site)
                if st and st['active'] == name:
                    users.append((site, st))
            if keep and keep['active'] == name:
                eff = self.effective(keep)
                keep.update(self.diff(eff, self.profile(DEFAULT_PROFILE)))
                keep['active'] = DEFAULT_PROFILE
            for site, st in users:
                eff = self.effective(st)
                st.update(self.diff(eff, self.profile(DEFAULT_PROFILE)))
                st['active'] = DEFAULT_PROFILE
                self.save_state(site, st)
        del self.custom[name]
        self._save_custom()
        return True, None

    # ---- migration ---------------------------------------------------------
    def sites(self):
        try:
            return sorted(n for n in os.listdir(self.base)
                          if os.path.isdir(os.path.join(self.base, n)))
        except OSError:
            return []

    def migrate_all(self):
        """direct_presets.json -> custom profiles, then every site without a
        state gets one. Idempotent. Returns the sites converted."""
        old = os.path.join(self.base, 'direct_presets.json')
        if os.path.exists(old):
            try:
                for name, vals in presets.load_custom(old).items():
                    if name not in self.custom and isinstance(vals, dict):
                        p = from_legacy(dict(vals, obstacle_max_range_m=2.8,
                                             free_max_range_m=2.8))
                        p['mapper'] = dict(MAPPER_DEFAULTS, **p['mapper'])
                        self.custom[name] = p
                self._save_custom()
                os.replace(old, old + PRE_SUFFIX)
            except ValueError:          # broken file: stays for a human
                pass
        done = []
        for site in self.sites():
            if self.load_state(site) is None:
                self.save_state(site, self._migrated(site))
                done.append(site)
        return done

    def _migrated(self, site):
        path = os.path.join(self.site_dir(site), OVERRIDES_FILE)
        ov = read_json(path)
        if ov:
            write_json(path + PRE_SUFFIX, ov)
        legacy = dict(LEGACY_DEFAULTS)
        legacy.update({k: v for k, v in ov.items() if k in LEGACY_DEFAULTS})
        now = from_legacy(legacy, self.wide_store())
        now['mapper'] = {k: legacy[k] for k in MAPPER_KEYS}
        best = None
        for name in [DEFAULT_PROFILE] + self.names():
            d = self.diff(now, self.profile(name))
            n = len(d['camera']) + len(d['mapper'])
            if best is None or n < best[0]:
                best = (n, name, d)
        return {'active': best[1], 'camera': best[2]['camera'],
                'mapper': best[2]['mapper']}

    def take_pending(self, st):
        """The old direct_pending.json (lidar values waiting for the next
        recording) -> changes of this state. Returns True when it took one."""
        path = os.path.join(self.base, PENDING_FILE)
        vals = {k: v for k, v in read_json(path).items() if k in MAPPER_KEYS}
        if not os.path.exists(path):
            return False
        ok = True
        if vals:
            ok, _err = self.set_mapper(st, vals)
        if ok:
            os.remove(path)
        return ok and bool(vals)
