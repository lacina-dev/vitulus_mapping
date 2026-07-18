#!/usr/bin/env python3
"""editmask.py -- composite human map edits onto an OccupancyGrid raster.

PURE PYTHON (numpy + cv2 only). NO rospy / ROS. Unit-testable offline and
reusable. cv2 is already a hard dependency of the mapping nodes
(mapping_manager, preview.py), so we reuse it for polygon fill / thick
polyline stamping rather than re-implementing scanline fill.

--------------------------------------------------------------------------
GRID CONVENTION (must match mapping_manager._publish_site_map / pgmio /
band_projector.publish_grid)
--------------------------------------------------------------------------
The occupancy array handed around inside mapping_manager is int8 in NATURAL
axes ``grid[ix, iy]`` (shape ``(nx, ny)``), value 100=occupied, 0=free,
-1=unknown -- exactly what ``pgmio.load_occupancy_pgm`` returns. The ROS
message is built as ``grid.T.ravel()`` with ``width=nx, height=ny`` and
``origin=(ox, oy)``; cell ``(ix, iy)`` covers world x in
``[ox+ix*res, ox+(ix+1)*res)`` and y in ``[oy+iy*res, oy+(iy+1)*res)``.

A MAP-frame point ``(x, y)`` maps to the integer cell that CONTAINS it::

        ix = floor((x - ox) / res)
        iy = floor((y - oy) / res)

This is exactly the cell-index convention ``band_projector.compare()`` uses
(``np.floor((gx-origin)/res)``), so edits land on the same cells the raster
was built on. cv2 fills a pixel when its integer index is inside the polygon
(boundary-inclusive), so a rectangle whose vertices map to indices [i0..i1]
fills cells i0..i1 inclusive.

We build the cv2 mask in image axes ``[iy, ix]`` (shape ``(ny, nx)``, cv2
point = (col=ix, row=iy)) and apply it back to ``grid[ix, iy]`` via the
transpose, so the natural-axes array the caller holds is what gets stamped.

--------------------------------------------------------------------------
EDIT SEMANTICS
--------------------------------------------------------------------------
Each edit is ``{name, kind, vertices:[(x, y), ...], width_m?}`` with vertices
in the MAP frame (metres). kind:
    'free'     -> polygon filled with 0   (clear / carve out)
    'obstacle' -> polygon filled with 100 (add obstacle)
    'wall'     -> polyline stamped 100 with thickness = width_m (virtual wall)
Edits are applied in LIST ORDER; a later edit overwrites earlier ones on the
cells they share (last-writer-wins). Compositing never touches -1 vs 0/100
selectively -- it stamps the target value on every covered cell.
"""

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except Exception:                                    # pragma: no cover
    cv2 = None
    _HAVE_CV2 = False

__all__ = [
    "KIND_VALUE", "rasterize_edits", "edits_to_map", "map_points_to_utm",
]

# kind -> occupancy value stamped onto the grid
KIND_VALUE = {"free": 0, "obstacle": 100, "wall": 100}


def _cell_coords(vertices, resolution, origin_xy):
    """MAP-frame [(x, y), ...] -> int32 cv2 points [[cx, cy], ...] in
    cell-centre coordinate space (see module docstring). cv2 wants (col=ix,
    row=iy) which is exactly (cx, cy) here."""
    ox, oy = origin_xy
    pts = np.empty((len(vertices), 2), dtype=np.float64)
    for i, (x, y) in enumerate(vertices):
        pts[i, 0] = (x - ox) / resolution           # cx  (col = ix axis)
        pts[i, 1] = (y - oy) / resolution           # cy  (row = iy axis)
    # floor -> the cell that CONTAINS the point (matches band_projector).
    return np.floor(pts).astype(np.int32)


def _stamp_polygon(mask, vertices, resolution, origin_xy):
    if len(vertices) < 3:
        return False
    pts = _cell_coords(vertices, resolution, origin_xy)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
    return True


def _stamp_wall(mask, vertices, width_m, resolution, origin_xy):
    """Stamp a virtual wall: a 1-px 8-connected centreline, then dilate by a
    disk of radius (span-1)//2 so the rendered perpendicular width is exactly
    ``span = round(width_m/res)`` cells (odd widths land exact; even widths
    round down to the nearest odd cell count). Dilation (vs raw cv2.polylines
    thickness, whose span<->thickness map is irregular) keeps the width
    predictable and diagonals fully connected."""
    if len(vertices) < 1:
        return False
    pts = _cell_coords(vertices, resolution, origin_xy)
    if len(vertices) == 1:
        mask[int(pts[0, 1]), int(pts[0, 0])] = 1     # [iy, ix]
    else:
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 1, thickness=1)
    span = max(1, int(round(float(width_m) / float(resolution))))
    r = (span - 1) // 2
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        cv2.dilate(mask, k, dst=mask)
    return True


def rasterize_edits(grid, edits, resolution, origin_xy):
    """Composite MAP-frame edits onto ``grid`` IN PLACE (last-writer-wins).

    grid       : int8 numpy array, NATURAL axes [ix, iy] (shape (nx, ny)),
                 values 100/0/-1 -- mutated in place.
    edits      : iterable of {name, kind, vertices:[(x, y), ...] MAP metres,
                 width_m? (walls)}; applied in order.
    resolution : float metres/cell.
    origin_xy  : (ox, oy) world coords of the grid origin corner.

    Returns the number of edits actually stamped (skips malformed / unknown).
    """
    if not _HAVE_CV2:
        raise RuntimeError("editmask.rasterize_edits requires cv2")
    nx, ny = grid.shape
    applied = 0
    for e in edits:
        kind = e.get("kind")
        if kind not in KIND_VALUE:
            continue
        verts = e.get("vertices") or []
        # one mask per edit so list order = compositing order (each edit
        # overwrites earlier ones on the cells it covers).
        mask = np.zeros((ny, nx), dtype=np.uint8)   # image axes [iy, ix]
        if kind == "wall":
            ok = _stamp_wall(mask, verts, e.get("width_m", 0.10),
                             resolution, origin_xy)
        else:
            ok = _stamp_polygon(mask, verts, resolution, origin_xy)
        if not ok:
            continue
        sel = mask.T.astype(bool)                   # back to [ix, iy]
        grid[sel] = KIND_VALUE[kind]
        applied += 1
    return applied


# --------------------------------------------------------------------------
# geo helpers (lazy import of the pure-python geo module; kept here so the
# core rasterizer above stays geo-free and unit-testable)
# --------------------------------------------------------------------------
def edits_to_map(edits, datum):
    """Return a copy of ``edits`` with vertices converted UTM -> MAP frame
    using the canonical ``datum`` (any schema geo.normalize_datum accepts)."""
    from vitulus_mapping import geo
    out = []
    for e in edits:
        verts = [geo.utm_to_map(float(vx), float(vy), datum)
                 for (vx, vy) in e.get("vertices", [])]
        ne = dict(e)
        ne["vertices"] = verts
        out.append(ne)
    return out


def map_points_to_utm(points, datum):
    """MAP-frame [(x, y), ...] -> UTM [(e, n), ...] using ``datum``."""
    from vitulus_mapping import geo
    return [geo.map_to_utm(float(x), float(y), datum) for (x, y) in points]
