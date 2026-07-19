#!/usr/bin/env python3
"""Unit tests for DirectGrid (the direct-raster pure library). Run:

    python3 tools/test_directgrid.py

No ROS required. Exercises: obstacle appears only after min_hits distinct
frames, free/miss erosion of a transient, lidar free-ray carving, camera range
handling at the grid level, auto-growth + origin shift, and render() cropping.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vitulus_mapping.directgrid import DirectGrid   # noqa: E402

FAIL = []


def check(cond, msg):
    print(('  ok  ' if cond else ' FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


def cell_val(g, x, y):
    """Rendered value (full grid) at world (x, y), or None if out of grid."""
    r = g.raster()
    ix, iy = g.world_to_idx(x, y)
    if 0 <= ix < r.shape[0] and 0 <= iy < r.shape[1]:
        return int(r[ix, iy])
    return None


def test_obstacle_needs_min_hits():
    print('test_obstacle_needs_min_hits')
    g = DirectGrid(resolution=0.05, hit_inc=0.5, occ_thresh=0.85, min_hits=3)
    x, y = 2.0, 1.0
    for i in range(2):                       # 2 distinct frames
        g.add_obstacle_points([x], [y])
    check(cell_val(g, x, y) != 100,
          'below min_hits (2 frames) does NOT render occupied')
    g.add_obstacle_points([x], [y])          # 3rd frame
    check(cell_val(g, x, y) == 100,
          'reaches occupied after min_hits (3) frames')
    # a single dense frame (many points same cell) is still ONE hit -> not occ
    g2 = DirectGrid(resolution=0.05, hit_inc=0.5, occ_thresh=0.85, min_hits=3)
    g2.add_obstacle_points([x] * 500, [y] * 500)
    check(g2.hits[g2.world_to_idx(x, y)] == 1,
          'one dense frame counts as a single hit (temporal, not density)')
    check(cell_val(g2, x, y) != 100,
          'one dense frame does not render occupied (min_hits guard)')


def test_transient_erodes():
    print('test_transient_erodes')
    g = DirectGrid(resolution=0.05, hit_inc=0.5, miss_dec=0.4,
                   occ_thresh=0.85, free_thresh=-0.4, min_hits=3)
    x, y = 0.5, 0.5
    for _ in range(3):
        g.add_obstacle_points([x], [y])
    check(cell_val(g, x, y) == 100, 'transient obstacle first shows occupied')
    # ground/free now seen repeatedly over the same cell -> erodes
    for _ in range(6):
        g.add_free_points([x], [y])
    check(cell_val(g, x, y) != 100, 'obstacle erodes away under free evidence')
    check(cell_val(g, x, y) == 0, 'and eventually becomes free')


def test_lidar_free_ray():
    print('test_lidar_free_ray')
    g = DirectGrid(resolution=0.05, hit_inc=0.5, miss_dec=0.4,
                   occ_thresh=0.85, free_thresh=-0.4, min_hits=3)
    sx, sy = 0.0, 0.0
    hx, hy = 1.0, 0.0                        # hit 1 m ahead on +x
    g.integrate_scan(sx, sy, [hx], [hy], [True], max_range=8.0)
    # a cell midway on the ray is free after a single pass
    check(cell_val(g, 0.5, 0.0) == 0, 'cell mid-ray is carved FREE in one scan')
    # the hit cell itself: hits=1 < min_hits -> not yet occupied, and NOT freed
    check(cell_val(g, hx, hy) != 0, 'hit cell is not freed by its own ray')
    check(cell_val(g, hx, hy) != 100, 'hit cell needs min_hits before occupied')
    # confirm it becomes occupied after enough scans, ray staying free
    for _ in range(3):
        g.integrate_scan(sx, sy, [hx], [hy], [True], max_range=8.0)
    check(cell_val(g, hx, hy) == 100, 'hit cell occupied after min_hits scans')
    check(cell_val(g, 0.5, 0.0) == 0, 'ray cells stay FREE across scans')


def test_lidar_no_return_free():
    print('test_lidar_no_return_free')
    g = DirectGrid(resolution=0.05, miss_dec=0.4, free_thresh=-0.4)
    # a no-return beam: endpoint at the range cap, hit_mask False -> whole ray
    # (including near the cap) is free, no obstacle anywhere.
    g.integrate_scan(0.0, 0.0, [3.0], [0.0], [False], max_range=3.0)
    check(cell_val(g, 1.5, 0.0) == 0, 'no-return beam frees mid-ray')
    check(cell_val(g, 2.9, 0.0) == 0, 'no-return beam frees near the cap')
    r = g.raster()
    check(int((r == 100).sum()) == 0, 'no-return beam creates no obstacle cell')


def test_autogrow_and_origin():
    print('test_autogrow_and_origin')
    g = DirectGrid(resolution=0.05)
    check(g.shape == (0, 0), 'starts empty')
    # a point far in the negative quadrant forces growth + origin shift
    g.add_free_points([-5.3], [-7.1])
    ix, iy = g.world_to_idx(-5.3, -7.1)
    check(0 <= ix < g.shape[0] and 0 <= iy < g.shape[1],
          'negative-quadrant cell is in-bounds after growth')
    check(g.obs[ix, iy], 'the grown cell is marked observed')
    check(g.ox <= -5.3 and g.oy <= -7.1, 'origin shifted to cover the point')


def test_render_crop():
    print('test_render_crop')
    g = DirectGrid(resolution=0.05, hit_inc=0.5, occ_thresh=0.85, min_hits=2)
    # far-apart observations -> a big padded grid, small observed bbox
    g.add_obstacle_points([10.0], [10.0])    # two frames -> crosses occ_thresh
    g.add_obstacle_points([10.0], [10.0])
    g.add_free_points([10.2, 10.25, 10.3], [10.0, 10.0, 10.0])
    crop, ox, oy = g.render(margin_cells=2)
    check(crop.shape[0] < g.shape[0] and crop.shape[1] < g.shape[1],
          'render crops away the unknown padding')
    # the occupied cell must survive the crop, and its world position (cell
    # centre from the reported origin) must round-trip to ~(10, 10).
    occ = np.argwhere(crop == 100)
    check(occ.shape[0] == 1, 'exactly one occupied cell in the crop')
    if occ.shape[0] == 1:
        cix, ciy = occ[0]
        wx = ox + (cix + 0.5) * g.res
        wy = oy + (ciy + 0.5) * g.res
        check(abs(wx - 10.0) <= g.res and abs(wy - 10.0) <= g.res,
              'occupied cell world position preserved at the cropped origin')


def main():
    for t in (test_obstacle_needs_min_hits, test_transient_erodes,
              test_lidar_free_ray, test_lidar_no_return_free,
              test_autogrow_and_origin, test_render_crop):
        t()
    print()
    if FAIL:
        print('FAILED %d check(s):' % len(FAIL))
        for m in FAIL:
            print('  - ' + m)
        sys.exit(1)
    print('ALL DIRECTGRID TESTS PASSED')


if __name__ == '__main__':
    main()
