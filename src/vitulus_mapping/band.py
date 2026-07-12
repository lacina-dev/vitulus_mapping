"""Band classifier/projector — obstacle(x, y) = evidence of occupied voxels
with z - DEM(x, y) in [band_min, band_max] (default 0.10-0.60 m ABOVE LOCAL
GROUND, not absolute z; correct on slopes).

Output raster values follow nav_msgs/OccupancyGrid convention:
  100 obstacle, 0 free, -1 unknown.
"""

import numpy as np
from scipy import ndimage

from .demgrid import SRC_INPAINT, SRC_NONE

OCC = 100
FREE = 0
UNKNOWN = -1


def project_band(dem, pts_xyz, band_min=0.10, band_max=0.60, min_evidence=6,
                 min_cluster_cells=3, ground_ref_radius_m=1.0,
                 borrowed_min_evidence=2):
    """dem: DemGrid (already filled/inpainted copy), pts_xyz: Nx3 occupied
    voxel centers in map frame. Returns (raster int8 [ix, iy], info dict).

    min_cluster_cells: obstacle clusters (8-connected) smaller than this are
    dropped — depth-camera speckles and grass tufts are isolated cells, real
    obstacles (walls, fences, trunks) are contiguous. Tuned on field data
    2026-07-06: min_evidence 3 + cluster 3 halves false positives while the
    large real clusters survive intact.

    ground_ref_radius_m (lidarsync 2026-07-12, ISSUE A fix): a vertical
    obstacle seen only by the 2D lidar in an UNDRIVEN column (garage wall,
    fence, trunk at the dock) has NO occupied voxel at ground level in its own
    column, so fill_from_columns cannot give that column ground and the voxel
    was dropped as 'unreferenced' — the lidar's hard geometry never reached the
    obstacle raster. FIX: when a voxel's own column lacks real ground, reference
    it against the NEAREST real-ground cell within this radius (ground is
    locally continuous; the robot drives on it right next to the wall). Still
    anchored ONLY on measured ground (trajectory/percentile-fill via the EDT
    over `real_ground`), never on inpaint-invented ground — honesty preserved.
    Set 0.0 to restore the strict same-column-only behaviour.

    borrowed_min_evidence (lidarsync 2026-07-12, ISSUE A part 2): the 2D lidar
    paints a single thin scan plane, so a solid wall yields only ~1-2 occupied
    voxels per 5 cm cell — far below the depth-tuned min_evidence=6 (dense depth
    clouds stack many voxels/cell). Applying min_evidence=6 to lidar cells
    silently deletes real walls. So cells whose in-band evidence came from a
    BORROWED-ground (lidar-only) column use this lower floor instead; own-ground
    (depth-dense) cells keep the full min_evidence. The spatial cluster filter
    (min_cluster_cells) then removes any sparse lidar speckle, so lowering the
    per-cell floor here does not admit isolated noise — a wall is a large
    contiguous cluster (measured at the dock: garage wall = ~450-cell cluster)."""
    nx, ny = dem.elev.shape
    evidence = np.zeros((nx, ny), np.int32)
    evidence_borrowed = np.zeros((nx, ny), np.int32)
    unref = 0
    borrowed = 0
    # Inpainted DEM cells are INVENTED ground (diffused from neighbours, no
    # measurement). They must never serve as a ground reference: a voxel over
    # an inpainted cell cannot be honestly classified obstacle-vs-free, because
    # the "ground" it is compared against is fabricated. Treat such columns as
    # unreferenced (audit 2026-07-12).
    real_ground = (dem.source != SRC_NONE) & (dem.source != SRC_INPAINT)
    if len(pts_xyz):
        ix = np.floor((pts_xyz[:, 0] - dem.ox) / dem.res).astype(np.int64)
        iy = np.floor((pts_xyz[:, 1] - dem.oy) / dem.res).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, zs = ix[ok], iy[ok], pts_xyz[:, 2][ok]
        ground = dem.elev[ix, iy]
        # a column is referenceable only if its ground cell is REAL (traj/fill),
        # not NaN and not inpaint-invented
        has_ground = ~np.isnan(ground) & real_ground[ix, iy]
        # ISSUE A fix: for columns WITHOUT own real ground, borrow the nearest
        # real-ground elevation within ground_ref_radius_m (single EDT over the
        # real-ground mask). This recovers lidar-only vertical obstacles that
        # sit next to driven ground but have no ground voxel in their own column.
        own_ground = has_ground.copy()          # depth-dense columns
        can_borrow = np.zeros_like(has_ground)
        if ground_ref_radius_m and ground_ref_radius_m > 0.0 \
                and real_ground.any() and (~has_ground).any():
            dist, (jx, jy) = ndimage.distance_transform_edt(
                ~real_ground, return_indices=True, sampling=dem.res)
            near_dist = dist[ix, iy]
            near_ground = dem.elev[jx[ix, iy], jy[ix, iy]]
            can_borrow = (~has_ground) & (near_dist <= ground_ref_radius_m) \
                & ~np.isnan(near_ground)
            ground = np.where(can_borrow, near_ground, ground)
            has_ground = has_ground | can_borrow
            borrowed = int(can_borrow.sum())
        dz = zs - ground
        in_band = has_ground & (dz >= band_min) & (dz <= band_max)
        # split evidence by ground source so each gets its own per-cell floor
        own_band = in_band & own_ground
        bor_band = in_band & can_borrow
        np.add.at(evidence, (ix[own_band], iy[own_band]), 1)
        np.add.at(evidence_borrowed, (ix[bor_band], iy[bor_band]), 1)
        # occupied columns we cannot classify (no real ground reference)
        unref = int((~has_ground).sum())

    raster = np.full((nx, ny), UNKNOWN, np.int8)
    # inpainted ground is shown as FREE (it is a plausible driveable surface for
    # display/nav continuity) but, per above, it never anchors obstacle claims
    known_ground = dem.source != SRC_NONE
    raster[known_ground] = FREE
    # own-ground (depth) cells use the full floor; borrowed-ground (sparse 2D
    # lidar) cells use the lower floor. A cell reaching EITHER floor is obstacle.
    obstacle = (evidence >= min_evidence) \
        | (evidence_borrowed >= max(1, borrowed_min_evidence))
    speckles = 0
    if min_cluster_cells > 1 and obstacle.any():
        lab, _n = ndimage.label(obstacle, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_cluster_cells
        keep[0] = False
        kept = keep[lab]
        speckles = int(obstacle.sum() - kept.sum())
        obstacle = kept
    raster[obstacle] = OCC

    info = {
        'cells_obstacle': int(obstacle.sum()),
        'speckles_dropped': speckles,
        'cells_free': int((raster == FREE).sum()),
        'cells_unknown': int((raster == UNKNOWN).sum()),
        'points_unreferenced': unref,
        'points_ground_borrowed': borrowed,
        'cells_obstacle_lidar': int((evidence_borrowed
                                     >= max(1, borrowed_min_evidence)).sum()),
        'band': [band_min, band_max],
        'min_evidence': min_evidence,
        'borrowed_min_evidence': borrowed_min_evidence,
    }
    return raster, info


def compare_rasters(a, b):
    """IoU / agreement stats between two equally-shaped int8 rasters (A/B vs
    e.g. the rtabmap grid resampled onto the same grid)."""
    both_known = (a != UNKNOWN) & (b != UNKNOWN)
    ao = a == OCC
    bo = b == OCC
    inter = int((ao & bo).sum())
    union = int((ao | bo).sum())
    return {
        'iou_obstacles': round(inter / union, 3) if union else None,
        'a_only_obstacles': int((ao & ~bo & both_known).sum()),
        'b_only_obstacles': int((bo & ~ao & both_known).sum()),
        'agree_cells': int(((a == b) & both_known).sum()),
        'both_known_cells': int(both_known.sum()),
    }
