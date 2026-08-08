"""DirectGrid — a DIRECT 2D log-odds occupancy raster built straight from the
per-frame segmented sensor clouds (Mapping v3, direct-raster path, 2026-07-19).

WHY THIS EXISTS (user design, binding): the band_projector chain builds a 3D
octomap and then reasons about z relative to a DEM to classify obstacle vs
ground. That whole pipeline is exposed to GLOBAL pose-z drift (the phantom
class of bugs chased all of 2026-07-19). This grid sidesteps the problem
entirely by discarding absolute z: it accumulates 2D evidence directly from
clouds that were ALREADY segmented obstacle-vs-ground per frame against the
LOCAL ground plane (rtabmap obstacles_detection: /obstacles_cloud vs
/ground_cloud) plus the sun-immune 2D lidar. The projection to XY throws z away,
so pose-z errors cannot create phantom steps.

Model: a per-cell log-odds value with clamps. Three kinds of evidence:
  * obstacle hit   -> +hit_inc, and a per-cell hit COUNTER +1 (capped one per
                     integrate call, i.e. one per sensor frame, so `hits` counts
                     how many DISTINCT frames saw an obstacle here — temporal
                     confirmation, not a single dense frame);
  * free / miss    -> -miss_dec (erodes transients back to unknown/free);
  * lidar free ray -> the cells strictly between the sensor and the hit (or the
                     whole ray, capped at max range, for a no-return beam) get
                     the free/miss decrement (2D raycast).

A cell renders as OCCUPIED only when its log-odds >= occ_thresh AND it has been
hit in >= min_hits distinct frames; FREE when log-odds <= free_thresh; else
UNKNOWN. So a walking operator / drive-by noise (< min_hits frames, then eroded
by later free evidence) never sticks.

Axes match DemGrid / band_projector exactly: natural [ix, iy], ix grows with
+x, iy with +y; origin (ox, oy) = world (x, y) of the min corner of cell (0, 0);
resolution in metres. render() returns an int8 raster (100 occ / 0 free /
-1 unknown) ready for pgmio.save_occupancy_pgm and the OccupancyGrid publish
convention (raster.T.ravel()).

ROS-agnostic and unit-testable: no rospy import here.
"""

import math

import numpy as np

_GROW_CELLS = 128          # grow in blocks to amortize reallocation (as DemGrid)


class DirectGrid:
    def __init__(self, resolution=0.05, hit_inc=0.5, miss_dec=0.4,
                 lo_clamp=-2.0, hi_clamp=3.5, occ_thresh=0.85,
                 free_thresh=-0.4, min_hits=3):
        self.res = float(resolution)
        self.hit_inc = float(hit_inc)
        self.miss_dec = float(miss_dec)
        self.lo_clamp = float(lo_clamp)
        self.hi_clamp = float(hi_clamp)
        self.occ_thresh = float(occ_thresh)
        self.free_thresh = float(free_thresh)
        self.min_hits = int(min_hits)
        self.ox = 0.0              # world x of min corner of cell (0, 0)
        self.oy = 0.0
        self.lo = np.zeros((0, 0), np.float32)     # log-odds
        self.hits = np.zeros((0, 0), np.uint16)    # distinct-frame hit counter
        self.obs = np.zeros((0, 0), bool)          # observed-at-all mask
        # lifetime counters (for status/meta)
        self.n_obstacle_frames = 0     # integrate calls that applied a hit
        self.n_free_frames = 0         # integrate calls that applied free

    # ---- geometry ----------------------------------------------------------
    @property
    def shape(self):
        return self.lo.shape

    def world_to_idx(self, x, y):
        return (int(math.floor((x - self.ox) / self.res)),
                int(math.floor((y - self.oy) / self.res)))

    def _grow_to(self, ix, iy):
        """Ensure cell (ix, iy) exists; grow arrays + shift origin as needed."""
        nx, ny = self.lo.shape
        pad_xlo = pad_ylo = pad_xhi = pad_yhi = 0
        if ix < 0:
            pad_xlo = (-ix + _GROW_CELLS - 1) // _GROW_CELLS * _GROW_CELLS
        if iy < 0:
            pad_ylo = (-iy + _GROW_CELLS - 1) // _GROW_CELLS * _GROW_CELLS
        if ix >= nx:
            pad_xhi = (ix - nx + _GROW_CELLS) // _GROW_CELLS * _GROW_CELLS
        if iy >= ny:
            pad_yhi = (iy - ny + _GROW_CELLS) // _GROW_CELLS * _GROW_CELLS
        if not (pad_xlo or pad_ylo or pad_xhi or pad_yhi):
            return
        pads = ((pad_xlo, pad_xhi), (pad_ylo, pad_yhi))
        self.lo = np.pad(self.lo, pads)
        self.hits = np.pad(self.hits, pads)
        self.obs = np.pad(self.obs, pads)
        self.ox -= pad_xlo * self.res
        self.oy -= pad_ylo * self.res

    def _ensure_bounds(self, xs, ys):
        if len(xs) == 0:
            return
        self._grow_to(*self.world_to_idx(float(np.min(xs)), float(np.min(ys))))
        self._grow_to(*self.world_to_idx(float(np.max(xs)), float(np.max(ys))))

    def _cells(self, xs, ys):
        ix = np.floor((xs - self.ox) / self.res).astype(np.int64)
        iy = np.floor((ys - self.oy) / self.res).astype(np.int64)
        return ix, iy

    # ---- evidence updates --------------------------------------------------
    def _apply_hits(self, ix, iy):
        """Apply obstacle evidence to the given (already in-bounds) cells: one
        +hit_inc and one hit-counter tick per DISTINCT cell (so a single dense
        frame counts as ONE hit toward min_hits, not hundreds)."""
        ny = self.lo.shape[1]
        flat = np.unique(ix * ny + iy)
        if flat.size == 0:
            return
        ux, uy = flat // ny, flat % ny
        self.lo[ux, uy] = np.minimum(self.lo[ux, uy] + self.hit_inc,
                                     self.hi_clamp)
        self.hits[ux, uy] = np.minimum(
            self.hits[ux, uy].astype(np.int32) + 1, 65535).astype(np.uint16)
        self.obs[ux, uy] = True
        self.n_obstacle_frames += 1

    def _apply_free(self, ix, iy, exclude_flat=None):
        """Apply free/miss evidence (-miss_dec) to the given cells, once per
        distinct cell. `exclude_flat` (flat indices) are skipped — used so a
        lidar ray never frees its own hit cell within the same scan."""
        ny = self.lo.shape[1]
        flat = np.unique(ix * ny + iy)
        if exclude_flat is not None and exclude_flat.size:
            flat = flat[~np.isin(flat, exclude_flat)]
        if flat.size == 0:
            return
        ux, uy = flat // ny, flat % ny
        self.lo[ux, uy] = np.maximum(self.lo[ux, uy] - self.miss_dec,
                                     self.lo_clamp)
        # AUDIT P2-1 (2026-08-08): expire the historical hit counter while the
        # cell is confidently FREE — `hits` used to only ever grow, so
        # min_hits=3 meant 3 obstacle frames per cell LIFETIME: a long-erased
        # transient could re-promote to occupied from a single later noise hit
        # (its old confirmations never expired). One decrement per free frame,
        # only once log-odds is at/below free_thresh, floor 0. Genuine static
        # obstacles are unaffected (their endpoint cells are excluded from
        # same-scan freeing and carry high hit counts).
        decay = (self.lo[ux, uy] <= self.free_thresh) & (self.hits[ux, uy] > 0)
        if decay.any():
            dx_, dy_ = ux[decay], uy[decay]
            self.hits[dx_, dy_] = (self.hits[dx_, dy_].astype(np.int32)
                                   - 1).astype(np.uint16)
        self.obs[ux, uy] = True
        self.n_free_frames += 1

    def add_obstacle_points(self, xs, ys):
        """Camera OBSTACLE points (already ground-projected to XY, already
        range-capped by the caller). One frame of obstacle evidence."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if xs.size == 0:
            return
        self._ensure_bounds(xs, ys)
        ix, iy = self._cells(xs, ys)
        self._apply_hits(ix, iy)

    def add_free_points(self, xs, ys):
        """Camera GROUND / FREE points (already range-capped). One frame of
        free evidence — erodes stale obstacles under the freshly-seen ground."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if xs.size == 0:
            return
        self._ensure_bounds(xs, ys)
        ix, iy = self._cells(xs, ys)
        self._apply_free(ix, iy)

    def integrate_scan(self, sx, sy, ex, ey, hit_mask, max_range=None):
        """2D lidar integration with raycasting.

        sx, sy    sensor origin in the grid/map frame (metres).
        ex, ey    per-beam endpoint in the grid/map frame. For a HIT beam this
                  is the return point; for a no-return beam it is the point at
                  the lidar range cap along the beam.
        hit_mask  bool per beam: True => an obstacle was seen at (ex, ey).
        max_range optional cap for the free-ray sample length (defaults to the
                  longest beam); only bounds the sample count.

        A HIT beam: free the cells strictly BEFORE the endpoint, obstacle
        evidence at the endpoint cell. A no-return beam (hit_mask False): free
        the whole ray. Endpoint (hit) cells are never freed within this scan.
        """
        ex = np.asarray(ex, dtype=np.float64)
        ey = np.asarray(ey, dtype=np.float64)
        hit_mask = np.asarray(hit_mask, dtype=bool)
        if ex.size == 0:
            return
        allx = np.append(ex, float(sx))
        ally = np.append(ey, float(sy))
        self._ensure_bounds(allx, ally)
        ny = self.lo.shape[1]

        dx = ex - sx
        dy = ey - sy
        dist = np.hypot(dx, dy)
        with np.errstate(invalid='ignore', divide='ignore'):
            ux = np.where(dist > 1e-6, dx / dist, 0.0)
            uy = np.where(dist > 1e-6, dy / dist, 0.0)
        # free sample length per beam: up to one cell before a hit, whole ray
        # for a no-return beam.
        limit = np.where(hit_mask, dist - self.res, dist)
        limit = np.maximum(limit, 0.0)

        # obstacle endpoints first (so we can exclude them from the free set)
        occ_flat = np.array([], dtype=np.int64)
        if hit_mask.any():
            hix, hiy = self._cells(ex[hit_mask], ey[hit_mask])
            occ_flat = np.unique(hix * ny + hiy)

        # free raycast samples along every beam (vectorized over steps x beams)
        maxlim = float(limit.max()) if limit.size else 0.0
        if max_range is not None:
            maxlim = min(maxlim, float(max_range))
        nsteps = int(math.floor(maxlim / self.res)) if maxlim > 0 else 0
        if nsteps > 0:
            steps = np.arange(1, nsteps + 1) * self.res           # (S,)
            valid = steps[:, None] < limit[None, :]               # (S, B)
            if valid.any():
                fx = (sx + ux[None, :] * steps[:, None])[valid]
                fy = (sy + uy[None, :] * steps[:, None])[valid]
                fix, fiy = self._cells(fx, fy)
                self._apply_free(fix, fiy, exclude_flat=occ_flat)

        if occ_flat.size:
            ux2, uy2 = occ_flat // ny, occ_flat % ny
            self.lo[ux2, uy2] = np.minimum(self.lo[ux2, uy2] + self.hit_inc,
                                           self.hi_clamp)
            self.hits[ux2, uy2] = np.minimum(
                self.hits[ux2, uy2].astype(np.int32) + 1, 65535).astype(
                    np.uint16)
            self.obs[ux2, uy2] = True
            self.n_obstacle_frames += 1

    # ---- rendering ---------------------------------------------------------
    def raster(self):
        """Full-grid int8 raster: 100 occ / 0 free / -1 unknown."""
        r = np.full(self.lo.shape, -1, np.int8)
        occ = self.obs & (self.lo >= self.occ_thresh) \
            & (self.hits >= self.min_hits)
        free = self.obs & (self.lo <= self.free_thresh) & (~occ)
        r[free] = 0
        r[occ] = 100
        return r

    def valid_bbox(self):
        """(ix0, iy0, ix1, iy1) inclusive bounds of observed cells, or None."""
        if not self.obs.any():
            return None
        xs = np.flatnonzero(self.obs.any(axis=1))
        ys = np.flatnonzero(self.obs.any(axis=0))
        return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])

    def render(self, margin_cells=2):
        """(raster_int8[ix, iy], ox, oy) cropped to the observed area plus a
        small margin, so the published/saved grid is not mostly padding."""
        full = self.raster()
        bb = self.valid_bbox()
        if bb is None:
            return full, self.ox, self.oy
        ix0, iy0, ix1, iy1 = bb
        nx, ny = full.shape
        m = int(max(0, margin_cells))
        x0 = max(0, ix0 - m)
        y0 = max(0, iy0 - m)
        x1 = min(nx, ix1 + 1 + m)
        y1 = min(ny, iy1 + 1 + m)
        crop = np.ascontiguousarray(full[x0:x1, y0:y1])
        return crop, self.ox + x0 * self.res, self.oy + y0 * self.res

    def stats(self):
        r = self.raster()
        return {
            'occ_cells': int((r == 100).sum()),
            'free_cells': int((r == 0).sum()),
            'observed_cells': int(self.obs.sum()),
            'shape': list(self.shape),
            'res': self.res,
            'obstacle_frames': self.n_obstacle_frames,
            'free_frames': self.n_free_frames,
        }

    def set_params(self, hit_inc=None, miss_dec=None, lo_clamp=None,
                   hi_clamp=None, occ_thresh=None, free_thresh=None,
                   min_hits=None):
        """Live-update the log-odds tunables. Threshold/min_hits changes take
        effect on the next render(); increments affect future integration."""
        if hit_inc is not None:
            self.hit_inc = float(hit_inc)
        if miss_dec is not None:
            self.miss_dec = float(miss_dec)
        if lo_clamp is not None:
            self.lo_clamp = float(lo_clamp)
        if hi_clamp is not None:
            self.hi_clamp = float(hi_clamp)
        if occ_thresh is not None:
            self.occ_thresh = float(occ_thresh)
        if free_thresh is not None:
            self.free_thresh = float(free_thresh)
        if min_hits is not None:
            self.min_hits = int(min_hits)
