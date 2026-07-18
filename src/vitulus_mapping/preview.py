"""PNG previews for the web UI (published as latched CompressedImage, same
pattern as /gloc/heatmap). Natural [ix, iy] arrays -> image rows top = max y."""

import cv2
import numpy as np

_MAX_PX = 1400          # keep UI payloads light


def _orient(arr):
    """[ix, iy] -> image[r, c] with r = y desc, c = x asc."""
    return arr.T[::-1, :]


def _shrink(img):
    h, w = img.shape[:2]
    scale = max(h, w) / float(_MAX_PX)
    if scale > 1.0:
        img = cv2.resize(img, (int(w / scale), int(h / scale)),
                         interpolation=cv2.INTER_NEAREST)
    return img


def raster_to_png(raster):
    """Occupancy raster (100/0/-1) -> PNG bytes: white free, black obstacle,
    gray unknown (map_server look)."""
    img = np.full(raster.shape[::-1], 205, np.uint8)
    o = _orient(raster)
    img[o == 0] = 254
    img[o == 100] = 0
    ok, buf = cv2.imencode('.png', _shrink(img))
    if not ok:
        raise RuntimeError('png encode failed')
    return buf.tobytes(), None


def dem_to_png(elev, source):
    """Elevation -> TURBO colormap PNG bytes (BGRA); no-data cells are fully
    TRANSPARENT (alpha 0) so the 3D terrain plane shows only real data.
    Returns (bytes, (zmin, zmax)) with the 2-98 percentile scaling used."""
    valid = source != 0
    e = _orient(elev)
    v = _orient(valid)
    img = np.zeros(e.shape + (4,), np.uint8)
    zr = None
    if v.any():
        vals = e[v]
        zmin, zmax = np.percentile(vals, [2, 98])
        if zmax - zmin < 0.05:
            zmax = zmin + 0.05
        norm = np.nan_to_num(np.clip((e - zmin) / (zmax - zmin), 0, 1))
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
        img[v, :3] = colored[v]
        img[v, 3] = 255
        zr = (float(zmin), float(zmax))
    ok, buf = cv2.imencode('.png', _shrink(img))
    if not ok:
        raise RuntimeError('png encode failed')
    return buf.tobytes(), zr
