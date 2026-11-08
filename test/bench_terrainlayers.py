#!/usr/bin/env python3
"""Size and time bench of vitulus_mapping.terrainlayers (T30 D1). Not a unit test.

Run: python3 test/bench_terrainlayers.py [--area 3700] [--batches 300]

Fills a square of `area` m^2 with realistic visits (smooth lawn + 3 mm noise,
95 % free, 4 % costly, 1 % lethal, 3 visits per cell, times by mowing rows),
saves it and prints JSON: RAM per cell, disk per m^2 (zlib npz), save / load ms
per tile, and the fuse time of 120 x 120 windows driven over it.
Made-up UTM point, no real site.
"""

import argparse
import json
import resource
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
import vitulus_mapping                                              # noqa: E402
vitulus_mapping.__path__.insert(0, os.path.join(SRC, 'vitulus_mapping'))
from vitulus_mapping import terrainlayers as tl                     # noqa: E402

E0, N0 = 500000.0, 5000000.0
DATUM = {"utm_e": E0, "utm_n": N0, "yaw_rad": 0.01}
T0 = 1790000000.0


def surface(e, n):
    x, y = e - E0, n - N0
    return 0.3 * np.sin(x / 9.0) + 0.2 * np.cos(y / 7.0) + 0.01 * x


def fill(st, side_m, rng):
    res = tl.RES
    g0x, g0y = int(E0 / res), int(N0 / res)
    ncell = int(round(side_m / res))
    gx1, gy1 = g0x + ncell, g0y + ncell
    for tx in range(g0x >> 8, ((gx1 - 1) >> 8) + 1):
        for ty in range(g0y >> 8, ((gy1 - 1) >> 8) + 1):
            lx0 = max(g0x - (tx << 8), 0)
            lx1 = min(gx1 - (tx << 8), tl.TILE)
            ly0 = max(g0y - (ty << 8), 0)
            ly1 = min(gy1 - (ty << 8), tl.TILE)
            if lx0 >= lx1 or ly0 >= ly1:
                continue
            ly, lx = np.mgrid[ly0:ly1, lx0:lx1]
            ly = ly.ravel()
            lx = lx.ravel()
            e = ((tx << 8) + lx + 0.5) * res
            n = ((ty << 8) + ly + 0.5) * res
            tile = st.tile(tx, ty, create=True)
            for k in range(3):
                u = rng.random_sample(lx.size)
                reason = np.where(u < 0.95, 1, np.where(u < 0.99, 2, np.where(u < 0.995, 3, 4)))
                cost = np.where(reason == 1, rng.randint(0, 50, lx.size),
                                np.where(reason == 2, rng.randint(50, 100, lx.size), 100))
                v = {
                    "h": surface(e, n) + rng.normal(0, 0.003, lx.size),
                    "var": rng.uniform(1e-4, 3e-4, lx.size),
                    "t": T0 + k * 86400 + np.floor((n - N0) / 0.6) * 40 + (e - E0) / 0.3,
                    "cost": cost.astype(float), "reason": reason.astype(float),
                    "slope": rng.uniform(0, 0.2, lx.size), "step": rng.uniform(0, 0.03, lx.size),
                    "rough": rng.uniform(0, 0.01, lx.size), "points": rng.randint(3, 20, lx.size),
                    "sector": rng.randint(0, 8, lx.size).astype(float),
                    "session": np.full(lx.size, float(k + 1)),
                }
                tile.commit(ly, lx, v, st.p)
    return ncell * ncell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", type=float, default=3700.0)
    ap.add_argument("--batches", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.RandomState(1)
    d = tempfile.mkdtemp(prefix="t30bench_")
    try:
        root = os.path.join(d, "terrain")
        st = tl.TerrainStore(root, DATUM, params={"max_loaded_tiles": 1000})
        side = math.sqrt(a.area)
        t = time.time()
        cells = fill(st, side, rng)
        fill_s = time.time() - t
        ram = sum(x.ram_bytes() for x in st.tiles.values())
        known = sum(x.known() for x in st.tiles.values())
        ntiles = len(st.tiles)
        t = time.time()
        st.save_dirty()
        save_ms = (time.time() - t) * 1000.0 / ntiles
        disk = sum(os.path.getsize(os.path.join(root, f)) for f in os.listdir(root)
                   if f.endswith(".npz"))
        idx = os.path.getsize(os.path.join(root, "index.json"))
        st2 = tl.TerrainStore(root, DATUM, params={"max_loaded_tiles": 1000})
        t = time.time()
        for key in list(st.tiles.keys()):
            st2.tile(*key)
        load_ms = (time.time() - t) * 1000.0 / ntiles
        # fuse: windows driven along rows over the known area (overlap with the stored map)
        st3 = tl.TerrainStore(root, DATUM, params={"accept_delay": 0.0, "max_loaded_tiles": 16})
        times = []
        x = 3.0
        y = 3.0
        hd = 0.0
        for i in range(a.batches):
            xs = (np.arange(120) + 0.5) * tl.RES + x - 3.0
            ys = (np.arange(120) + 0.5) * tl.RES + y - 3.0
            X, Y = np.meshgrid(xs, ys)
            e, n = E0 + X, N0 + Y
            h = (surface(e, n) + 0.4 + rng.normal(0, 0.003, X.shape)).astype(np.float32)
            win = tl.Window(x - 3.0, y - 3.0, tl.RES, h, np.full(h.shape, 1e-4, np.float32),
                            np.full(h.shape, 5, np.float32), np.full(h.shape, 10, np.float32),
                            np.ones(h.shape, np.float32), t_fused=np.full(h.shape, T0 + 9e5 + i))
            pose = (0.0, 0.0, -0.01)        # map <- odom: odom = UTM axes here (datum yaw cancelled)
            t = time.perf_counter()
            st3.fuse(win, pose, T0 + 9e5 + i, robot=(x, y, 0.0, hd))
            times.append((time.perf_counter() - t) * 1000.0)
            x += 0.25 if hd == 0.0 else -0.25
            if x > side - 3.0 or x < 3.0:
                y += 0.6
                hd = math.pi - hd if hd else math.pi
                x = min(max(x, 3.0), side - 3.0)
        t = time.perf_counter()
        st3.close_all()
        close_ms = (time.perf_counter() - t) * 1000.0
        times = np.array(times)
        out = {
            "area_m2": round(side * side, 1), "cells": cells, "known": known, "tiles": ntiles,
            "fill_s": round(fill_s, 2),
            "ram_bytes_per_cell": round(ram / float(ntiles * tl.TILE * tl.TILE), 2),
            "ram_mb_tiles": round(ram / 1e6, 1),
            "ram_mb_per_1000m2_known": round(ram / 1e6 / (known * tl.RES ** 2 / 1000.0), 1),
            "disk_mb": round(disk / 1e6, 2), "index_kb": round(idx / 1e3, 1),
            "disk_kb_per_m2_known": round(disk / 1e3 / (known * tl.RES ** 2), 2),
            "save_ms_per_tile": round(save_ms, 1), "load_ms_per_tile": round(load_ms, 1),
            "fuse_ms_avg": round(float(times.mean()), 2),
            "fuse_ms_p95": round(float(np.percentile(times, 95)), 2),
            "fuse_ms_max": round(float(times.max()), 2),
            "close_all_ms": round(close_ms, 1),
            "batches": int(times.size),
            "z_offset": st3.z_offset,
            "fuse_store_ram_mb": round(st3.ram_bytes() / 1e6, 1),
            "max_rss_mb_whole_bench": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1),
        }
        print(json.dumps(out, indent=1))
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
