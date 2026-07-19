"""Band classifier/projector — obstacle(x, y) = evidence of occupied voxels
with z - DEM(x, y) in [band_min, band_max] (default 0.10-0.60 m ABOVE LOCAL
GROUND, not absolute z; correct on slopes).

Output raster values follow nav_msgs/OccupancyGrid convention:
  100 obstacle, 0 free, -1 unknown.
"""

import numpy as np
from scipy import ndimage

from .demgrid import SRC_FILL, SRC_INPAINT, SRC_NONE, SRC_TRAJ

OCC = 100
FREE = 0
UNKNOWN = -1


def project_band(dem, pts_xyz, band_min=0.10, band_max=0.60, min_evidence=6,
                 min_cluster_cells=3, ground_ref_radius_m=1.0,
                 borrowed_min_evidence=2, free_via_borrowed_ground=True,
                 free_ground_radius_m=None,
                 free_xyz=None, free_band_min=-0.30, free_band_max=0.60,
                 free_borrow_legacy=False,
                 cluster_min_size=15, cluster_min_ztop_m=0.30,
                 cluster_traj_dist_m=0.6,
                 occ_prob_xyz=None, min_occupancy=0.0):
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
    contiguous cluster (measured at the dock: garage wall = ~450-cell cluster).

    free_via_borrowed_ground (freeserve 2026-07-12, REPORT 1 fix): FREE was
    painted ONLY on cells with real DEM ground (dem.source != SRC_NONE). After
    the DEM tightening (fill_percentile 1.5, coherence guard, fill_min_pts 6)
    the real-ground columns collapsed toward the driven trajectory, so FREE
    shrank to a thin ribbon even though the robot demonstrably drove a real
    area. FIX (mirror of the obstacle ground-borrow): a cell WITHOUT real ground
    of its own but within free_ground_radius_m of a real-ground cell, and with
    NO obstacle evidence, is plausibly driveable surrounding ground and is
    marked FREE. Same honesty rule as the obstacle borrow — anchored on the EDT
    over the REAL-ground mask (trajectory/percentile-fill, never inpaint), so
    FREE never spreads beyond free_ground_radius_m of a genuine measurement.
    free_ground_radius_m defaults to ground_ref_radius_m when None. Set
    free_via_borrowed_ground=False (or radius 0) to restore the strict
    real-ground-only FREE behaviour.

    CLASSIFIER v2 (2026-07-19, forensic redesign). Two defects were verified on
    the novaTestovaciMapa dataset and are fixed here:

    D1 — FREE rework. The old free_via_borrowed_ground painted 67.6% of FREE as
    1.0 m-radius rings around any real-ground cell IGNORING occlusion (22.4% had
    no raytraced backing) and counted INPAINT cells (invented ground) as FREE.
    The octomap FREE leaves — the actual raytraced free space — were never
    consulted. v2 default: FREE = real-ground cells (SRC_TRAJ|SRC_FILL, NOT
    inpaint) UNION an octomap-free-band mask, minus obstacles. A cell is
    free-band if some octomap FREE leaf sits in its column with
    z - ground(cell) in [free_band_min, free_band_max] (ground = own real ground
    else nearest borrowed real ground within free_ground_radius_m; no ground
    reference in range => NOT free). Pass the free leaf centres as `free_xyz`
    (Nx3, from the ot_dump helper). If free_xyz is None/empty the mask is empty
    and FREE collapses to real ground only (honest: unmapped => unknown). The old
    ring behaviour is preserved for rollback behind free_borrow_legacy=True.

    D2 — cluster z-top gate. 92.5% of obstacle cells are borrowed-path 2D-lidar
    (thin-z by nature — NOT gated on z-extent). Real clusters reach column
    z-top >= 0.4 m; false lawn patches sit <= 0.2 m. After the existing cluster
    filter, a cluster is dropped iff ALL of: size < cluster_min_size AND its
    max column z-top above (own-else-borrowed) ground < cluster_min_ztop_m AND
    its min distance to a trajectory cell > cluster_traj_dist_m. On this dataset
    that removes 262 false cells with 0 real-cluster losses.

    CLASSIFIER v3 (2026-07-19, forensic round-2). After v2 the FREE space was
    accepted but the user reported REMAINING false obstacles. Round-2 forensics
    (novaTestovaciMapa, 2833 v2 obstacle cells, 17 real clusters incl. the garage
    623-cell cluster) tested three discriminators; only octomap per-voxel
    OCCUPANCY PROBABILITY separated real from false. octomap stores log-odds per
    leaf: real structure (walls, trunks, posts) is hit by many rays and saturates
    toward the clamp (>=0.95), while transients (a walking operator, swaying
    grass, drive-by noise) accumulate few hits and their occupancy sits near the
    0.5 threshold. Per-CELL occupancy does NOT separate (real thin-lidar walls
    contain low-prob cells too — measured overlap p5=0.61 both classes); the
    per-CLUSTER MAXIMUM in-band occupancy is the clean discriminator: real
    clusters bottom out at cmax=0.927, transient clusters top out below, so a
    cluster whose BEST in-band voxel never reached min_occupancy was never
    confirmed by repeated rays and is dropped. min_occupancy=0.90 removes 226
    false cells at ZERO real-cluster loss here, INDEPENDENT of size (catches
    larger transient smears the z-top/size gate spares). H1 operator-trail
    (distance-to-trajectory, free-leaves-above-band) and H2 DEM-slope showed no
    separation (slope was reversed — real obstacles sit on the steeper wall-base
    gradient) and were rejected. Pass the occupied leaf centres+probability as
    `occ_prob_xyz` (Nx4 x,y,z,occ, from the ot_dump helper). min_occupancy=0
    disables the gate (backward compatible); a cluster with no occupancy coverage
    at all (cmax==0, e.g. occ_prob_xyz snapshot mismatch) is KEPT, never dropped
    — the gate is strictly evidence-additive."""
    nx, ny = dem.elev.shape
    evidence = np.zeros((nx, ny), np.int32)
    evidence_borrowed = np.zeros((nx, ny), np.int32)
    unref = 0
    borrowed = 0
    if free_ground_radius_m is None:
        free_ground_radius_m = ground_ref_radius_m
    # Inpainted DEM cells are INVENTED ground (diffused from neighbours, no
    # measurement). They must never serve as a ground reference: a voxel over
    # an inpainted cell cannot be honestly classified obstacle-vs-free, because
    # the "ground" it is compared against is fabricated. Treat such columns as
    # unreferenced (audit 2026-07-12).
    real_ground = (dem.source != SRC_NONE) & (dem.source != SRC_INPAINT)
    # Single EDT over the real-ground mask (per-cell distance to the nearest
    # measured ground cell, and that cell's index). Shared by BOTH the obstacle
    # ground-borrow (below) and the FREE ground-borrow (REPORT 1). Computed once
    # here so the FREE path does not need its own transform.
    # v2: the free-band mask (D1) and the z-top gate (D2) both need the borrowed
    # ground reference too, so compute the EDT whenever the grid has any
    # real-ground and any non-real-ground cell (cheap; a single transform).
    rg_dist = rg_jx = rg_jy = None
    if real_ground.any() and (~real_ground).any():
        rg_dist, (rg_jx, rg_jy) = ndimage.distance_transform_edt(
            ~real_ground, return_indices=True, sampling=dem.res)
    # near_ground_grid[c] = elevation of the nearest real-ground cell to c
    # (only meaningful where rg_dist is finite); used to borrow ground for the
    # free-band mask and the z-top gate.
    near_ground_grid = None
    if rg_dist is not None:
        near_ground_grid = dem.elev[rg_jx, rg_jy]

    # Per-cell max occupied-voxel z (all in-bounds occupied leaves, band-agnostic
    # — the z-top gate needs the true column top, not just the in-band part).
    zmax_grid = np.full((nx, ny), -np.inf, np.float32)
    if len(pts_xyz):
        ix = np.floor((pts_xyz[:, 0] - dem.ox) / dem.res).astype(np.int64)
        iy = np.floor((pts_xyz[:, 1] - dem.oy) / dem.res).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, zs = ix[ok], iy[ok], pts_xyz[:, 2][ok]
        np.maximum.at(zmax_grid, (ix, iy), zs.astype(np.float32))
        ground = dem.elev[ix, iy]
        # a column is referenceable only if its ground cell is REAL (traj/fill),
        # not NaN and not inpaint-invented
        has_ground = ~np.isnan(ground) & real_ground[ix, iy]
        # ISSUE A fix: for columns WITHOUT own real ground, borrow the nearest
        # real-ground elevation within ground_ref_radius_m (shared EDT over the
        # real-ground mask). This recovers lidar-only vertical obstacles that
        # sit next to driven ground but have no ground voxel in their own column.
        own_ground = has_ground.copy()          # depth-dense columns
        can_borrow = np.zeros_like(has_ground)
        if ground_ref_radius_m and ground_ref_radius_m > 0.0 \
                and rg_dist is not None and (~has_ground).any():
            near_dist = rg_dist[ix, iy]
            near_ground = dem.elev[rg_jx[ix, iy], rg_jy[ix, iy]]
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

    def cell_ground_within(radius_m):
        """Own-else-borrowed REAL ground elevation per cell: own real ground
        where present, else the nearest real-ground elevation within radius_m
        (shared EDT), else NaN. Never inpaint. Shared by the free-band mask (D1)
        and the z-top gate (D2)."""
        g = np.where(real_ground, dem.elev, np.nan).astype(np.float32)
        if rg_dist is not None and radius_m and radius_m > 0.0:
            borrow = (~real_ground) & (rg_dist <= radius_m) \
                & np.isfinite(near_ground_grid)
            g[borrow] = near_ground_grid[borrow]
        return g

    raster = np.full((nx, ny), UNKNOWN, np.int8)
    # v2 D1: base FREE is REAL ground only (trajectory + percentile-fill).
    real_free_base = real_ground
    free_borrowed = 0        # legacy borrow-ring count (legacy mode only)
    free_octomap = 0         # octomap-free-band count added (v2 default mode)
    if free_borrow_legacy:
        # ---- ROLLBACK PATH (free_borrow_legacy=True) --------------------------
        # Reproduce the pre-v2 free_via_borrowed_ground behaviour EXACTLY:
        # inpaint shown as FREE + free_ground_radius_m borrow rings around any
        # real-ground cell (occlusion-blind; the defect being retired). Kept only
        # so an operator can fall back if the octomap free mask is unavailable.
        known_ground = dem.source != SRC_NONE
        raster[known_ground] = FREE
        if free_via_borrowed_ground and free_ground_radius_m \
                and free_ground_radius_m > 0.0 and rg_dist is not None:
            borrow_free = (~known_ground) & (rg_dist <= free_ground_radius_m)
            free_borrowed = int(borrow_free.sum())
            raster[borrow_free] = FREE
    else:
        # ---- v2 DEFAULT PATH: real ground UNION octomap-free-band mask --------
        raster[real_free_base] = FREE
        free_band_mask = np.zeros((nx, ny), bool)
        if free_xyz is not None and len(free_xyz):
            fxyz = np.asarray(free_xyz, np.float64)
            fx, fy, fz = fxyz[:, 0], fxyz[:, 1], fxyz[:, 2]
            fxi = np.floor((fx - dem.ox) / dem.res).astype(np.int64)
            fyi = np.floor((fy - dem.oy) / dem.res).astype(np.int64)
            inb = (fxi >= 0) & (fxi < nx) & (fyi >= 0) & (fyi < ny)
            cg = cell_ground_within(free_ground_radius_m)
            fg = cg[fxi.clip(0, nx - 1), fyi.clip(0, ny - 1)]
            fdz = fz - fg
            good = inb & np.isfinite(fg) \
                & (fdz >= free_band_min) & (fdz <= free_band_max)
            free_band_mask[fxi[good], fyi[good]] = True
        free_octomap = int((free_band_mask & ~real_free_base).sum())
        raster[free_band_mask] = FREE
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

    # ---- v2 D2: cluster z-top gate ------------------------------------------
    # Drop a surviving cluster iff ALL of: small (size < cluster_min_size) AND
    # low (max column z-top above own-else-borrowed ground < cluster_min_ztop_m)
    # AND far from any driven trajectory cell (min dist > cluster_traj_dist_m).
    # These three together isolate false lawn/vegetation patches from real thin
    # 2D-lidar walls (which are large, tall, or hug the driven path).
    ztop_gate_clusters = 0
    ztop_gate_cells = 0
    if obstacle.any() and cluster_min_size and cluster_min_size > 0:
        obs_ground = cell_ground_within(ground_ref_radius_m)
        ztop = zmax_grid - obs_ground              # NaN where no ground ref
        ztop = np.where(np.isfinite(ztop), ztop, -np.inf).astype(np.float32)
        traj_mask = dem.source == SRC_TRAJ
        if traj_mask.any():
            traj_dist = ndimage.distance_transform_edt(
                ~traj_mask, sampling=dem.res)
        else:
            traj_dist = np.full((nx, ny), np.inf, np.float32)
        lab2, n2 = ndimage.label(obstacle, structure=np.ones((3, 3)))
        if n2:
            idx = np.arange(n2 + 1)
            csize = np.bincount(lab2.ravel(), minlength=n2 + 1)
            cmaxztop = ndimage.maximum(ztop, lab2, idx)
            cmintraj = ndimage.minimum(traj_dist, lab2, idx)
            drop = (csize < cluster_min_size) \
                & (cmaxztop < cluster_min_ztop_m) \
                & (cmintraj > cluster_traj_dist_m)
            drop[0] = False
            if drop.any():
                drop_mask = drop[lab2]
                ztop_gate_clusters = int(drop.sum())
                ztop_gate_cells = int(drop_mask.sum())
                obstacle = obstacle & ~drop_mask

    # ---- v3: cluster occupancy-probability gate -----------------------------
    # Drop a surviving obstacle cluster whose MAXIMUM in-band octomap occupancy
    # probability is below min_occupancy — the whole cluster was raytraced only
    # a few times (a walking operator, swaying grass, drive-by noise) and never
    # confirmed to saturation like real structure. Per-cell occupancy would kill
    # real thin walls (they contain low-prob cells too); the cluster max is the
    # discriminator. cmax==0 (no occupancy coverage, e.g. snapshot mismatch) is
    # KEPT — the gate only removes clusters it can positively see are transient.
    occ_gate_clusters = 0
    occ_gate_cells = 0
    if (min_occupancy and min_occupancy > 0.0 and obstacle.any()
            and occ_prob_xyz is not None and len(occ_prob_xyz)):
        op = np.asarray(occ_prob_xyz, np.float64)
        opx = np.floor((op[:, 0] - dem.ox) / dem.res).astype(np.int64)
        opy = np.floor((op[:, 1] - dem.oy) / dem.res).astype(np.int64)
        oki = (opx >= 0) & (opx < nx) & (opy >= 0) & (opy < ny)
        opx, opy, opz, opp = opx[oki], opy[oki], op[oki, 2], op[oki, 3]
        occg = cell_ground_within(ground_ref_radius_m)
        gsel = occg[opx, opy]
        odz = opz - gsel
        oin = np.isfinite(gsel) & (odz >= band_min) & (odz <= band_max)
        occprob = np.zeros((nx, ny), np.float32)
        if oin.any():
            np.maximum.at(occprob, (opx[oin], opy[oin]),
                          opp[oin].astype(np.float32))
        lab3, n3 = ndimage.label(obstacle, structure=np.ones((3, 3)))
        if n3:
            idx3 = np.arange(n3 + 1)
            cmaxp = ndimage.maximum(occprob, lab3, idx3)
            # cmaxp>0 guard: only drop clusters we can SEE are low-occupancy;
            # a cluster with no occupancy coverage at all is kept (honest).
            drop_o = (cmaxp < min_occupancy) & (cmaxp > 0.0)
            drop_o[0] = False
            if drop_o.any():
                drop_om = drop_o[lab3]
                occ_gate_clusters = int(drop_o.sum())
                occ_gate_cells = int(drop_om.sum())
                obstacle = obstacle & ~drop_om
    raster[obstacle] = OCC

    info = {
        'cells_obstacle': int(obstacle.sum()),
        'speckles_dropped': speckles,
        'ztop_gate_clusters_dropped': ztop_gate_clusters,
        'ztop_gate_cells_dropped': ztop_gate_cells,
        'occ_gate_clusters_dropped': occ_gate_clusters,
        'occ_gate_cells_dropped': occ_gate_cells,
        'cells_free': int((raster == FREE).sum()),
        'cells_free_realground': int(real_free_base.sum()),
        'cells_free_octomap': free_octomap,
        'cells_free_borrowed': free_borrowed,
        'free_leaves_seen': int(len(free_xyz)) if free_xyz is not None else 0,
        'free_borrow_legacy': bool(free_borrow_legacy),
        'cells_unknown': int((raster == UNKNOWN).sum()),
        'points_unreferenced': unref,
        'points_ground_borrowed': borrowed,
        'cells_obstacle_lidar': int((evidence_borrowed
                                     >= max(1, borrowed_min_evidence)).sum()),
        'band': [band_min, band_max],
        'free_band': [free_band_min, free_band_max],
        'min_evidence': min_evidence,
        'borrowed_min_evidence': borrowed_min_evidence,
        'cluster_min_size': cluster_min_size,
        'cluster_min_ztop_m': cluster_min_ztop_m,
        'cluster_traj_dist_m': cluster_traj_dist_m,
        'min_occupancy': min_occupancy,
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
