"""composite.py -- the COMBINED map of a site: every saved mapping session
(raster version) of one site merged into a single occupancy grid, NEWER
SESSION WINS where they overlap.

Why (2026-09-13, Robert): each mapping session builds a brand-new raster from
scratch and only the newest one was served, so a partial re-mapping (one
corner of the garden) THREW AWAY everything mapped earlier. The combined map
keeps all knowledge; a newer session only overrides the cells it actually
OBSERVED (free or obstacle) -- its unknown cells never erase older data.
The single sessions stay on disk untouched as "layers" and any of them can be
excluded from the combination (<site>/combined.json, key "exclude").

Frames: every version lives in its OWN session map frame, georeferenced by
the datum captured in that session (utm <- map, see geo.py). The combined
grid is built in the frame of the site's CURRENT datum (<site>/datum.yaml,
i.e. the latest session), so waypoints / zones / edits / geofence stored in
UTM keep working unchanged. Each version is resampled (nearest neighbour,
inverse mapping, rotation included) through UTM into that frame.

Per-version datum: <version>/<version>_datum.yaml. mapping_manager writes it
at Save & finish; for versions saved before this existed it is backfilled
from the site's datum_final_<ts>.yaml archive whose timestamp is closest to
the version name (the gate writes it ~1 s after the raster), else from the
site datum.yaml.

Layout produced:
    <site>/rasters/combined/combined.pgm
    <site>/rasters/combined/combined.yaml
    <site>/rasters/combined/combined_meta.json   {source: 'composite', ...}

Pure numpy + yaml. No ROS imports (usable offline / in tests).
"""

import json
import math
import os
import re
import time

import numpy as np
import yaml

from vitulus_mapping import geo
from vitulus_mapping.pgmio import load_occupancy_pgm, save_occupancy_pgm

COMBINED_NAME = 'combined'
CONFIG_FILE = 'combined.json'
_TS_RE = re.compile(r'(\d{8})_(\d{6})')
_BACKFILL_MAX_DT_S = 180.0


# --------------------------------------------------------------------------
# helpers: naming / datum lookup
# --------------------------------------------------------------------------
def is_combined(raster):
    return raster == COMBINED_NAME


def version_timestamp(name):
    """time.time()-style epoch parsed from a 'direct_YYYYmmdd_HHMMSS' name
    (or any name containing that pattern), else None."""
    m = _TS_RE.search(name or '')
    if not m:
        return None
    try:
        return time.mktime(time.strptime(m.group(1) + m.group(2),
                                         '%Y%m%d%H%M%S'))
    except ValueError:
        return None


def session_versions(site_dir):
    """Names of the session rasters (dirs under rasters/ holding <name>.pgm),
    the combined raster excluded, sorted OLDEST FIRST by the timestamp in the
    name (fallback: pgm mtime)."""
    rdir = os.path.join(site_dir, 'rasters')
    if not os.path.isdir(rdir):
        return []
    out = []
    for name in os.listdir(rdir):
        if is_combined(name):
            continue
        pgm = os.path.join(rdir, name, name + '.pgm')
        if not os.path.exists(pgm):
            continue
        ts = version_timestamp(name)
        if ts is None:
            ts = os.path.getmtime(pgm)
        out.append((ts, name))
    out.sort()
    return [n for _, n in out]


def load_config(site_dir):
    """<site>/combined.json -> dict (always has 'exclude': [list])."""
    cfg = {}
    try:
        with open(os.path.join(site_dir, CONFIG_FILE)) as f:
            d = json.load(f)
        if isinstance(d, dict):
            cfg = d
    except (OSError, ValueError):
        pass
    ex = cfg.get('exclude')
    cfg['exclude'] = [str(x) for x in ex] if isinstance(ex, list) else []
    return cfg


def save_config(site_dir, cfg):
    path = os.path.join(site_dir, CONFIG_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, path)


def set_layer_enabled(site_dir, raster, enabled):
    """Include/exclude one session raster from the combination. Returns the
    new exclude list."""
    cfg = load_config(site_dir)
    ex = [x for x in cfg['exclude'] if x != raster]
    if not enabled:
        ex.append(raster)
    cfg['exclude'] = ex
    save_config(site_dir, cfg)
    return ex


def _read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def version_datum_path(site_dir, raster):
    return os.path.join(site_dir, 'rasters', raster, raster + '_datum.yaml')


def load_site_datum(site_dir):
    """Canonical datum dict of <site>/datum.yaml, or None."""
    try:
        d = _read_yaml(os.path.join(site_dir, 'datum.yaml'))
        if 'utm_e' not in d:
            return None
        return geo.from_datum_yaml(d)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def record_version_datum(site_dir, raster, source='site datum at Save & finish'):
    """Copy the site's current datum.yaml next to the raster as the version's
    own datum (called by mapping_manager right after the session's
    finalize_georef). Returns the written path or None (no datum)."""
    src = os.path.join(site_dir, 'datum.yaml')
    try:
        d = _read_yaml(src)
    except (OSError, ValueError):
        return None
    if 'utm_e' not in d:
        return None
    d = dict(d)
    d['datum_source'] = source
    dst = version_datum_path(site_dir, raster)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.tmp'
    with open(tmp, 'w') as f:
        yaml.safe_dump(d, f, default_flow_style=False)
    os.replace(tmp, dst)
    return dst


def _nearest_final_datum(site_dir, raster):
    """datum_final_<ts>.yaml closest in time to the version name (within
    _BACKFILL_MAX_DT_S), else None."""
    ts = version_timestamp(raster)
    if ts is None:
        return None
    best, best_dt = None, None
    for name in os.listdir(site_dir):
        if not (name.startswith('datum_final_') and name.endswith('.yaml')):
            continue
        fts = version_timestamp(name)
        if fts is None:
            continue
        dt = abs(fts - ts)
        if dt <= _BACKFILL_MAX_DT_S and (best_dt is None or dt < best_dt):
            best, best_dt = name, dt
    return os.path.join(site_dir, best) if best else None


def resolve_version_datum(site_dir, raster, backfill=True):
    """Canonical datum dict for one session raster.

    Order: <version>_datum.yaml -> nearest datum_final_<ts>.yaml (backfilled
    into <version>_datum.yaml when `backfill`) -> site datum.yaml (backfilled
    too, marked as such). Returns (datum|None, source_str)."""
    vpath = version_datum_path(site_dir, raster)
    try:
        d = _read_yaml(vpath)
        if 'utm_e' in d:
            return geo.from_datum_yaml(d), 'version'
    except (OSError, ValueError, KeyError, TypeError):
        pass
    fin = _nearest_final_datum(site_dir, raster)
    if fin is not None:
        try:
            d = _read_yaml(fin)
            if 'utm_e' in d:
                if backfill:
                    _write_backfill(vpath, d, 'backfilled from ' +
                                    os.path.basename(fin))
                return geo.from_datum_yaml(d), os.path.basename(fin)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        d = _read_yaml(os.path.join(site_dir, 'datum.yaml'))
        if 'utm_e' in d:
            if backfill:
                _write_backfill(vpath, d, 'backfilled from site datum.yaml '
                                '(no session archive matched)')
            return geo.from_datum_yaml(d), 'site'
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None, 'none'


def _write_backfill(vpath, d, note):
    try:
        d = dict(d)
        d['datum_source'] = note
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        tmp = vpath + '.tmp'
        with open(tmp, 'w') as f:
            yaml.safe_dump(d, f, default_flow_style=False)
        os.replace(tmp, vpath)
    except OSError:
        pass


# --------------------------------------------------------------------------
# the merge
# --------------------------------------------------------------------------
def _affine_src_from_dst(dst_datum, src_datum):
    """2x3 affine A such that p_src = A[:, :2] @ p_dst + A[:, 2]  (metres),
    composing dst-map -> UTM -> src-map (geo.py convention)."""
    cd, sd = math.cos(dst_datum['yaw_rad']), math.sin(dst_datum['yaw_rad'])
    cs, ss = math.cos(src_datum['yaw_rad']), math.sin(src_datum['yaw_rad'])
    r_dst = np.array([[cd, -sd], [sd, cd]])          # map_dst -> utm
    r_src_inv = np.array([[cs, ss], [-ss, cs]])      # utm -> map_src
    t = np.array([dst_datum['utm_e'] - src_datum['utm_e'],
                  dst_datum['utm_n'] - src_datum['utm_n']])
    rot = r_src_inv @ r_dst
    trans = r_src_inv @ t
    return rot, trans


def build_combined(site_dir, resolution=None, log=None):
    """Merge every enabled session raster of the site into
    rasters/combined/combined.{pgm,yaml,_meta.json}.

    Returns a stats dict (also stored as the meta json) or None when the site
    has no usable session raster. Raises on I/O errors of the write itself.
    `log(msg)` optional progress callback."""
    log = log or (lambda m: None)
    site = os.path.basename(site_dir.rstrip('/'))
    versions = session_versions(site_dir)
    cfg = load_config(site_dir)
    excluded = set(cfg['exclude'])
    enabled = [v for v in versions if v not in excluded]
    dst_datum = load_site_datum(site_dir)

    sources = []
    for name in enabled:
        base = os.path.join(site_dir, 'rasters', name, name)
        try:
            data, res, origin = load_occupancy_pgm(base)
        except Exception as e:
            log('combined %s: skipping %s (load failed: %s)' % (site, name, e))
            continue
        datum, dsrc = resolve_version_datum(site_dir, name)
        sources.append({'name': name, 'data': data, 'res': float(res),
                        'origin': (float(origin[0]), float(origin[1])),
                        'datum': datum, 'datum_source': dsrc})
    if not sources:
        return None

    if resolution is None:
        resolution = sources[-1]['res']
    resolution = float(resolution)

    # No site datum (never had RTK): every version is assumed to share the
    # frame -> identity placement.
    if dst_datum is None:
        dst_datum = {'utm_e': 0.0, 'utm_n': 0.0, 'yaw_rad': 0.0}
        for s in sources:
            s['datum'] = None
    for s in sources:
        if s['datum'] is None:
            s['datum'] = dict(dst_datum)      # identity
            s['datum_source'] = s['datum_source'] + ' (identity)'

    # ---- target extent: union of every version's corners in the dst frame
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for s in sources:
        nx, ny = s['data'].shape
        ox, oy = s['origin']
        corners = np.array([[ox, oy], [ox + nx * s['res'], oy],
                            [ox, oy + ny * s['res']],
                            [ox + nx * s['res'], oy + ny * s['res']]])
        rot, trans = _affine_src_from_dst(dst_datum, s['datum'])
        # inverse: p_dst = rot^-1 (p_src - trans); rot is orthonormal
        pd = (corners - trans) @ rot          # (rot^T)^T ... rot^-1 = rot^T
        s['dst_bbox'] = (pd.min(axis=0), pd.max(axis=0))
        lo = np.minimum(lo, pd.min(axis=0))
        hi = np.maximum(hi, pd.max(axis=0))
    pad = 2 * resolution
    origin_x = math.floor((lo[0] - pad) / resolution) * resolution
    origin_y = math.floor((lo[1] - pad) / resolution) * resolution
    nx = int(math.ceil((hi[0] + pad - origin_x) / resolution))
    ny = int(math.ceil((hi[1] + pad - origin_y) / resolution))
    nx, ny = max(nx, 1), max(ny, 1)
    if nx * ny > 40_000_000:
        raise ValueError('combined grid too large: %dx%d' % (nx, ny))

    out = np.full((nx, ny), -1, np.int8)
    meta_sources = []
    # oldest -> newest: newer OBSERVED cells overwrite older ones
    for s in sources:
        rot, trans = _affine_src_from_dst(dst_datum, s['datum'])
        bmin, bmax = s['dst_bbox']
        ix0 = max(int(math.floor((bmin[0] - origin_x) / resolution)) - 1, 0)
        iy0 = max(int(math.floor((bmin[1] - origin_y) / resolution)) - 1, 0)
        ix1 = min(int(math.ceil((bmax[0] - origin_x) / resolution)) + 1, nx)
        iy1 = min(int(math.ceil((bmax[1] - origin_y) / resolution)) + 1, ny)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        xs = origin_x + (np.arange(ix0, ix1) + 0.5) * resolution
        ys = origin_y + (np.arange(iy0, iy1) + 0.5) * resolution
        gx, gy = np.meshgrid(xs, ys, indexing='ij')      # [ix, iy]
        sx = rot[0, 0] * gx + rot[0, 1] * gy + trans[0]
        sy = rot[1, 0] * gx + rot[1, 1] * gy + trans[1]
        sres = s['res']
        jx = np.floor((sx - s['origin'][0]) / sres).astype(np.int64)
        jy = np.floor((sy - s['origin'][1]) / sres).astype(np.int64)
        snx, sny = s['data'].shape
        inside = (jx >= 0) & (jx < snx) & (jy >= 0) & (jy < sny)
        vals = np.full(gx.shape, -1, np.int8)
        vals[inside] = s['data'][jx[inside], jy[inside]]
        observed = vals != -1
        block = out[ix0:ix1, iy0:iy1]
        block[observed] = vals[observed]
        applied = int(observed.sum())
        meta_sources.append({
            'name': s['name'], 'res': s['res'],
            'datum': {k: s['datum'][k] for k in ('utm_e', 'utm_n', 'yaw_rad')},
            'datum_source': s['datum_source'],
            'cells_applied': applied,
            'offset_m': [round(float(trans[0]), 3), round(float(trans[1]), 3)],
            'yaw_deg': round(math.degrees(
                geo.wrap_pi(dst_datum['yaw_rad'] - s['datum']['yaw_rad'])), 3),
        })
        log('combined %s: + %s (%d observed cells, datum %s)'
            % (site, s['name'], applied, s['datum_source']))

    cdir = os.path.join(site_dir, 'rasters', COMBINED_NAME)
    base = os.path.join(cdir, COMBINED_NAME)
    save_occupancy_pgm(base, out, resolution, (origin_x, origin_y))
    stats = {
        'source': 'composite',
        'built': time.strftime('%Y-%m-%d %H:%M:%S'),
        'rule': 'oldest->newest, newer observed cells win',
        'datum': {k: dst_datum[k] for k in ('utm_e', 'utm_n', 'yaw_rad')},
        'res': resolution,
        'shape': [nx, ny],
        'origin': [origin_x, origin_y],
        'occ_cells': int((out == 100).sum()),
        'free_cells': int((out == 0).sum()),
        'observed_cells': int((out != -1).sum()),
        'sources': meta_sources,
        'excluded': sorted(excluded & set(versions)),
    }
    tmp = base + '_meta.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(stats, f, indent=1)
    os.replace(tmp, base + '_meta.json')
    return stats


def combined_info(site_dir):
    """Small status record for the UI, or None when no combined raster."""
    base = os.path.join(site_dir, 'rasters', COMBINED_NAME, COMBINED_NAME)
    if not os.path.exists(base + '.pgm'):
        return None
    info = {'name': COMBINED_NAME,
            'mtime': int(os.path.getmtime(base + '.pgm')),
            'sources': [], 'excluded': [], 'cells_free': None,
            'cells_obstacle': None, 'built': None}
    try:
        with open(base + '_meta.json') as f:
            m = json.load(f)
        info['sources'] = [s['name'] for s in m.get('sources', [])]
        info['excluded'] = m.get('excluded', [])
        info['cells_free'] = m.get('free_cells')
        info['cells_obstacle'] = m.get('occ_cells')
        info['built'] = m.get('built')
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return info


def is_stale(site_dir):
    """True when the combined raster is missing or older than any session
    raster / the combined.json config."""
    base = os.path.join(site_dir, 'rasters', COMBINED_NAME, COMBINED_NAME)
    if not os.path.exists(base + '.pgm'):
        return bool(session_versions(site_dir))
    cm = os.path.getmtime(base + '.pgm')
    for f in (CONFIG_FILE, 'datum.yaml'):     # layer toggle / new site datum
        p = os.path.join(site_dir, f)
        if os.path.exists(p) and os.path.getmtime(p) > cm:
            return True
    for name in session_versions(site_dir):
        pgm = os.path.join(site_dir, 'rasters', name, name + '.pgm')
        if os.path.getmtime(pgm) > cm:
            return True
    return False
