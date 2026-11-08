#!/usr/bin/env python3
"""Seed a site's persistent terrain map (T30) from an offline terrain mosaic.

    python3 tools/terrain_import_mosaic.py <mosaic dir> --datum <site datum.yaml> --out <terrain dir>

<mosaic dir> = output of vitulus_terrain_mosaic (teren_<tag>_layers.npz, teren_<tag>.json,
grid/grid.json + grid/t_last.f32), in the map frame of the site datum valid when the bags
were recorded. Every imported cell becomes ONE tentative visit (terrainlayers.import_map_grid);
cells already known in <out> are kept. Writes <out>/index.json + tiles; prints a JSON summary.
The output holds the site's UTM tile numbers: keep it on the robot / laptop, never commit it.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vitulus_mapping import geo, terrainexport, terrainlayers as tl    # noqa: E402


def load_mosaic(mdir):
    try:
        return terrainexport.load_mosaic_dir(mdir)
    except ValueError as e:
        raise SystemExit(str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mosaic")
    ap.add_argument("--datum", required=True, help="the site's datum.yaml valid for the bags")
    ap.add_argument("--out", required=True, help="terrain dir (<site>/terrain)")
    ap.add_argument("--overwrite", action="store_true", help="replace known cells too")
    a = ap.parse_args()
    datum = geo.from_datum_yaml(yaml.safe_load(open(a.datum)))
    meta, layers, t = load_mosaic(a.mosaic)
    st = tl.TerrainStore(a.out, datum, params={"max_loaded_tiles": 100000})
    n, offset = st.import_map_grid((float(meta["origin_x"]), float(meta["origin_y"])), float(meta["resolution"]),
                           layers, t, overwrite=a.overwrite)
    written = st.save_dirty()
    known = int(np.count_nonzero(np.isfinite(layers["elevation"])))
    disk = sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.out, "t_*.npz")))
    reason = layers["reason"][np.isfinite(layers["elevation"])]
    print(json.dumps({
        "mosaic": os.path.basename(os.path.normpath(a.mosaic)), "bags": meta.get("bags"),
        "mosaic_known_cells": known, "imported_cells": n, "height_offset_m": round(offset, 4), "tiles_written": written,
        "tiles_total": len(st.index["tiles"]), "disk_mb": round(disk / 1e6, 2),
        "lethal_cells_tentative": int(np.count_nonzero((reason == 3) | (reason == 4))),
        "obstacle_cells": int(np.count_nonzero(reason == 5)),
        "area_m2": round(n * tl.RES ** 2, 1),
    }, indent=1))


if __name__ == "__main__":
    main()
