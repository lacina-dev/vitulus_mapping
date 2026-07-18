"""map_server-compatible pgm/yaml raster export (edit in GIMP/QGIS, serve with
map_server/nav2_map_server unchanged)."""

import os
import tempfile

import numpy as np
import yaml

# map_server pixel convention
_PX_OCC = 0        # black
_PX_FREE = 254     # white
_PX_UNKNOWN = 205  # gray


def save_occupancy_pgm(path_base, raster, resolution, origin_xy):
    """raster: int8 [ix, iy] natural axes (100/0/-1). Writes <base>.pgm +
    <base>.yaml. PGM rows top->bottom = +y max -> min, columns = +x.

    M7: atomic + ORDERED. Both files are written to temp names in the same dir
    and renamed into place with the .yaml FIRST, then the .pgm. Readers gate on
    the .pgm existing (mapping_manager) and then need the sibling .yaml, so
    'pgm exists' now guarantees a complete .yaml — no torn/half-written reads."""
    d = os.path.dirname(path_base)
    os.makedirs(d, exist_ok=True)
    nx, ny = raster.shape
    img = np.full((ny, nx), _PX_UNKNOWN, np.uint8)
    flipped = raster.T[::-1, :]                 # rows: y desc, cols: x asc
    img[flipped == 100] = _PX_OCC
    img[flipped == 0] = _PX_FREE

    pgm_path = path_base + '.pgm'
    yaml_path = path_base + '.yaml'
    meta = {
        'image': os.path.basename(pgm_path),
        'resolution': float(resolution),
        'origin': [float(origin_xy[0]), float(origin_xy[1]), 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }

    fd, tmp_pgm = tempfile.mkstemp(dir=d, suffix='.pgm')
    with os.fdopen(fd, 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (nx, ny))
        f.write(img.tobytes())
    fd, tmp_yaml = tempfile.mkstemp(dir=d, suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(meta, f, default_flow_style=False)
    # publish the .yaml FIRST, then the .pgm -> pgm-exists implies yaml-complete
    os.replace(tmp_yaml, yaml_path)
    os.replace(tmp_pgm, pgm_path)
    return pgm_path


def _read_pgm(pgm_path):
    """Minimal P5 (binary) / P2 (ascii) PGM reader -> uint8 2-D array
    [row, col] (row 0 = top, matching save_occupancy_pgm's image layout).
    Defensive: map_server-style PGMs are P5, but tolerate P2 (e.g. hand
    edited in a text editor) too."""
    with open(pgm_path, 'rb') as f:
        raw = f.read()

    # tokenize header (magic, width, height, maxval), skipping '#' comments,
    # then hand the remaining bytes off as pixel data (P5) or parse them as
    # whitespace-separated ints (P2).
    pos = 0
    tokens = []
    while len(tokens) < 4:
        while pos < len(raw) and raw[pos:pos + 1].isspace():
            pos += 1
        if pos < len(raw) and raw[pos:pos + 1] == b'#':
            while pos < len(raw) and raw[pos:pos + 1] not in (b'\n', b'\r'):
                pos += 1
            continue
        start = pos
        while pos < len(raw) and not raw[pos:pos + 1].isspace():
            pos += 1
        if start == pos:
            raise ValueError('%s: truncated PGM header' % pgm_path)
        tokens.append(raw[start:pos])
    # exactly one whitespace byte separates the header from binary data (P5)
    if raw[pos:pos + 1].isspace():
        pos += 1

    magic = tokens[0]
    nx, ny, maxval = (int(t) for t in tokens[1:4])
    if magic == b'P5':
        expect = nx * ny
        body = np.frombuffer(raw, dtype=np.uint8, count=expect, offset=pos)
        if body.size != expect:
            raise ValueError('%s: expected %d bytes of pixel data, got %d'
                              % (pgm_path, expect, body.size))
        img = body.reshape(ny, nx)
    elif magic == b'P2':
        vals = raw[pos:].split()
        img = np.array([int(v) for v in vals[:nx * ny]],
                        dtype=np.uint16).reshape(ny, nx)
        img = img.astype(np.uint8) if maxval == 255 else \
            np.round(img.astype(np.float64) * 255.0 / maxval).astype(np.uint8)
    else:
        raise ValueError('%s: unsupported PGM magic %r' % (pgm_path, magic))
    return img


def load_occupancy_pgm(path_pgm):
    """Inverse of save_occupancy_pgm. path_pgm may point at the .pgm or at
    its basename (with or without extension) — the sibling .yaml is always
    used for resolution/origin/thresholds.

    Returns (raster, resolution, origin_xy):
      raster: int8 [ix, iy] natural axes (100 occ / 0 free / -1 unknown),
              exact inverse of the row/flip save_occupancy_pgm applies.
      resolution: float
      origin_xy: (ox, oy) float tuple
    """
    if path_pgm.endswith('.pgm'):
        base = path_pgm[:-len('.pgm')]
    else:
        base = path_pgm
        path_pgm = base + '.pgm'
    yaml_path = base + '.yaml'

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    resolution = float(meta['resolution'])
    origin = meta.get('origin', [0.0, 0.0, 0.0])
    origin_xy = (float(origin[0]), float(origin[1]))
    occupied_thresh = float(meta.get('occupied_thresh', 0.65))
    free_thresh = float(meta.get('free_thresh', 0.196))
    negate = int(meta.get('negate', 0))

    img = _read_pgm(path_pgm)          # [row, col], row 0 = top (y max)
    if negate:
        img = 255 - img

    flipped = img[::-1, :]             # row 0 -> y min (undo the save flip)
    # flipped is [iy, ix]; transpose back to the natural [ix, iy] axes that
    # save_occupancy_pgm started from (flipped_save = raster.T[::-1, :]).
    px = flipped.T                     # [ix, iy], uint8 pixel values

    out = np.full(px.shape, -1, np.int8)
    out[px == _PX_OCC] = 100
    out[px == _PX_FREE] = 0
    out[px == _PX_UNKNOWN] = -1
    other = (px != _PX_OCC) & (px != _PX_FREE) & (px != _PX_UNKNOWN)
    if other.any():
        # nearest-by-threshold fallback for pixels that don't match one of
        # the three canonical values exactly (e.g. externally edited PGM) —
        # occupied_thresh/free_thresh are map_server "occupied-probability"
        # style thresholds on the *inverse* 0..1 (white=free) scale used by
        # map_server: p = (255 - px) / 255.
        p = (255.0 - px[other].astype(np.float64)) / 255.0
        vals = np.full(p.shape, -1, np.int8)
        vals[p >= occupied_thresh] = 100
        vals[p <= free_thresh] = 0
        out[other] = vals

    return out, resolution, origin_xy
