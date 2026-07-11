"""DemGrid — 2.5D digital elevation model on a dynamically growing grid.

Canonical elevation source is the robot trajectory (wheel-contact z, cm-class,
immune to vegetation). Octree-column percentile fill and inpainting are
DERIVED data: they run on a copy at raster-regeneration time, never on the
canonical grid.

Axes are 'natural': arrays indexed [ix, iy], ix grows with +x, iy with +y,
origin = world (x, y) of the min corner of cell (0, 0). All world coords are
map-frame meters, elevations are map-frame altitude (z corrected by navsat).

Sources: 0 = unknown, 1 = trajectory, 2 = octree percentile fill, 3 = inpaint.
"""

import math
import os
import tempfile

import numpy as np
from scipy import ndimage

SRC_NONE = 0
SRC_TRAJ = 1
SRC_FILL = 2
SRC_INPAINT = 3

_GROW_CELLS = 128          # grow in blocks to amortize reallocation
_WEIGHT_CAP = 50.0         # cap running-mean weight so seasonal change adapts


class DemGrid:
    def __init__(self, resolution=0.05):
        self.res = float(resolution)
        self.ox = 0.0              # world x of min corner of cell (0,0)
        self.oy = 0.0
        self.elev = np.full((0, 0), np.nan, np.float32)
        self.weight = np.zeros((0, 0), np.float32)
        self.last_seen = np.zeros((0, 0), np.float64)
        self.source = np.zeros((0, 0), np.uint8)

    # ---- geometry ----------------------------------------------------------
    @property
    def shape(self):
        return self.elev.shape

    def world_to_idx(self, x, y):
        return (int(math.floor((x - self.ox) / self.res)),
                int(math.floor((y - self.oy) / self.res)))

    def _grow_to(self, ix, iy):
        """Ensure cell (ix, iy) exists; grows arrays and shifts origin as needed."""
        nx, ny = self.elev.shape
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
            return ix, iy
        pads = ((pad_xlo, pad_xhi), (pad_ylo, pad_yhi))
        self.elev = np.pad(self.elev, pads, constant_values=np.nan)
        self.weight = np.pad(self.weight, pads)
        self.last_seen = np.pad(self.last_seen, pads)
        self.source = np.pad(self.source, pads)
        self.ox -= pad_xlo * self.res
        self.oy -= pad_ylo * self.res
        return ix + pad_xlo, iy + pad_ylo

    # ---- canonical updates (trajectory) -------------------------------------
    def stamp_disc(self, x, y, z, t, radius):
        """Insert one trajectory ground sample as a disc of cells (running
        weighted mean). Trajectory always overrides derived sources."""
        r_cells = max(0, int(math.ceil(radius / self.res)))
        cx, cy = self.world_to_idx(x, y)
        cx, cy = self._grow_to(cx - r_cells, cy - r_cells)
        cx += r_cells
        cy += r_cells
        cx2, cy2 = self._grow_to(cx + r_cells, cy + r_cells)
        cx, cy = cx2 - r_cells, cy2 - r_cells

        n = 2 * r_cells + 1
        yy, xx = np.mgrid[0:n, 0:n]
        disc = (xx - r_cells) ** 2 + (yy - r_cells) ** 2 <= r_cells ** 2
        sl = (slice(cx - r_cells, cx + r_cells + 1),
              slice(cy - r_cells, cy + r_cells + 1))

        e = self.elev[sl]
        w = self.weight[sl]
        s = self.source[sl]
        fresh = disc & (s != SRC_TRAJ)         # first traj sample wins over fills
        e[fresh] = z
        w[fresh] = 1.0
        upd = disc & ~fresh
        e[upd] = (e[upd] * w[upd] + z) / (w[upd] + 1.0)
        w[upd] = np.minimum(w[upd] + 1.0, _WEIGHT_CAP)
        s[disc] = SRC_TRAJ
        self.last_seen[sl][disc] = t

    # ---- derived updates (raster regeneration, run on a copy) ---------------
    def fill_from_columns(self, pts_xyz, percentile=5.0, min_pts=3):
        """Fill cells WITHOUT trajectory data with a per-column percentile of
        occupied-voxel z (ghost-resistant low envelope). Returns filled count."""
        if len(pts_xyz) == 0:
            return 0
        xs = pts_xyz[:, 0]
        ys = pts_xyz[:, 1]
        self._grow_to(*self.world_to_idx(xs.min(), ys.min()))
        self._grow_to(*self.world_to_idx(xs.max(), ys.max()))
        nx, ny = self.elev.shape
        ix = np.floor((xs - self.ox) / self.res).astype(np.int64)
        iy = np.floor((ys - self.oy) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, zs = ix[ok], iy[ok], pts_xyz[:, 2][ok]

        cell = ix * ny + iy
        order = np.argsort(cell, kind='stable')
        cell, zs = cell[order], zs[order]
        starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
        counts = np.diff(np.r_[starts, len(cell)])

        filled = 0
        el = self.elev.ravel()
        sr = self.source.ravel()
        for s0, cnt in zip(starts, counts):
            if cnt < min_pts:
                continue
            c = cell[s0]
            if sr[c] == SRC_TRAJ:
                continue
            el[c] = np.percentile(zs[s0:s0 + cnt], percentile)
            sr[c] = SRC_FILL
            filled += 1
        return filled

    def slope_filter_fills(self, max_slope_deg=35.0, radius_m=0.30):
        """Drop percentile fills that disagree with nearby trajectory cells
        beyond the max plausible slope (2D-lidar-mows-the-ground guard)."""
        traj = self.source == SRC_TRAJ
        fill = self.source == SRC_FILL
        if not traj.any() or not fill.any():
            return 0
        dist, (jx, jy) = ndimage.distance_transform_edt(
            ~traj, return_indices=True, sampling=self.res)
        near = fill & (dist <= radius_m * 4)    # only judge fills near evidence
        dz = np.abs(self.elev - self.elev[jx, jy])
        max_dz = np.tan(math.radians(max_slope_deg)) * np.maximum(dist, self.res)
        bad = near & (dz > max_dz)
        self.elev[bad] = np.nan
        self.source[bad] = SRC_NONE
        return int(bad.sum())

    def inpaint(self, max_dist_m=0.5, iters=None):
        """Diffuse known elevation into unknown cells up to max_dist_m away."""
        known = self.source != SRC_NONE
        if not known.any():
            return 0
        dist = ndimage.distance_transform_edt(~known, sampling=self.res)
        target = (~known) & (dist <= max_dist_m)
        if not target.any():
            return 0
        e = self.elev.copy()
        n_it = iters or int(math.ceil(max_dist_m / self.res)) + 2
        k = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        for _ in range(n_it):
            vals = np.nan_to_num(e)
            have = (~np.isnan(e)).astype(np.float32)
            num = ndimage.convolve(vals, k, mode='constant')
            den = ndimage.convolve(have, k, mode='constant')
            cand = target & np.isnan(e) & (den > 0)
            e[cand] = num[cand] / den[cand]
        done = target & ~np.isnan(e)
        self.elev[done] = e[done]
        self.source[done] = SRC_INPAINT
        return int(done.sum())

    # ---- queries -------------------------------------------------------------
    def query_many(self, xs, ys):
        """Vectorized nearest-cell elevation lookup; NaN where unknown."""
        nx, ny = self.elev.shape
        ix = np.floor((xs - self.ox) / self.res).astype(np.int64)
        iy = np.floor((ys - self.oy) / self.res).astype(np.int64)
        out = np.full(len(xs), np.nan, np.float32)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out[ok] = self.elev[ix[ok], iy[ok]]
        return out

    def valid_bbox(self):
        """(ix0, iy0, ix1, iy1) inclusive bounds of known cells, or None."""
        known = self.source != SRC_NONE
        if not known.any():
            return None
        xs = np.flatnonzero(known.any(axis=1))
        ys = np.flatnonzero(known.any(axis=0))
        return xs[0], ys[0], xs[-1], ys[-1]

    def stats(self):
        traj = int((self.source == SRC_TRAJ).sum())
        known = int((self.source != SRC_NONE).sum())
        return {'cells_traj': traj, 'cells_known': known,
                'area_traj_m2': round(traj * self.res * self.res, 1),
                'shape': list(self.shape), 'res': self.res}

    def copy(self):
        d = DemGrid(self.res)
        d.ox, d.oy = self.ox, self.oy
        d.elev = self.elev.copy()
        d.weight = self.weight.copy()
        d.last_seen = self.last_seen.copy()
        d.source = self.source.copy()
        return d

    # ---- persistence -----------------------------------------------------------
    def save_npz(self, path):
        """Atomic save (tmp + rename) so a reader never sees a torn file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.npz')
        os.close(fd)
        np.savez_compressed(tmp, res=self.res, ox=self.ox, oy=self.oy,
                            elev=self.elev, weight=self.weight,
                            last_seen=self.last_seen, source=self.source)
        os.replace(tmp, path)

    @classmethod
    def load_npz(cls, path):
        with np.load(path) as f:
            d = cls(float(f['res']))
            d.ox = float(f['ox'])
            d.oy = float(f['oy'])
            d.elev = f['elev']
            d.weight = f['weight']
            d.last_seen = f['last_seen']
            d.source = f['source']
        return d
