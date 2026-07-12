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
                 min_cluster_cells=3):
    """dem: DemGrid (already filled/inpainted copy), pts_xyz: Nx3 occupied
    voxel centers in map frame. Returns (raster int8 [ix, iy], info dict).

    min_cluster_cells: obstacle clusters (8-connected) smaller than this are
    dropped — depth-camera speckles and grass tufts are isolated cells, real
    obstacles (walls, fences, trunks) are contiguous. Tuned on field data
    2026-07-06: min_evidence 3 + cluster 3 halves false positives while the
    large real clusters survive intact."""
    nx, ny = dem.elev.shape
    evidence = np.zeros((nx, ny), np.int32)
    unref = 0
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
        dz = zs - ground
        in_band = has_ground & (dz >= band_min) & (dz <= band_max)
        np.add.at(evidence, (ix[in_band], iy[in_band]), 1)
        # occupied columns we cannot classify (no real ground reference)
        unref = int((~has_ground).sum())

    raster = np.full((nx, ny), UNKNOWN, np.int8)
    # inpainted ground is shown as FREE (it is a plausible driveable surface for
    # display/nav continuity) but, per above, it never anchors obstacle claims
    known_ground = dem.source != SRC_NONE
    raster[known_ground] = FREE
    obstacle = evidence >= min_evidence
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
        'band': [band_min, band_max],
        'min_evidence': min_evidence,
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
