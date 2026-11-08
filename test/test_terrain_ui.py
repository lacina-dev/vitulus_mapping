#!/usr/bin/env python3
"""ROS-free tests of the T31 UI interface of terrain_recorder (vitulus_mapping.terrainfuse
RecorderCore.command / state, vitulus_mapping.terrainexport).

Run: python3 test/test_terrain_ui.py   (from the package root)
Contract: ~/vitulus/predavka/t31-teren-ui-kontrakt.md. Made-up site and coordinates.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import zlib

import numpy as np
import yaml

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
import vitulus_mapping                                              # noqa: E402
vitulus_mapping.__path__.insert(0, os.path.join(SRC, 'vitulus_mapping'))
from vitulus_mapping import terrainfuse as tf                       # noqa: E402
from vitulus_mapping import terrainlayers as tl                     # noqa: E402

E0, N0 = 500000.0, 5000000.0
DATUM = {"utm_e": E0, "utm_n": N0, "yaw_rad": 0.0, "alt": 300.0, "utm_zone": 33}
T0 = 1790000000.0
RTK_FLAGS = 0x01 | (2 << 6)
LOC_OK = {"owner": "gps", "gps_good": True, "heading_ok": True, "anchor_frozen": False,
          "heading_anchor_frozen": False}


def read_png(path):
    """(array, colour type, bit depth) of a PNG written by terrainexport.write_png."""
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(raw):
        n = int.from_bytes(raw[pos:pos + 4], "big")
        tag, body = raw[pos + 4:pos + 8], raw[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h = int.from_bytes(body[:4], "big"), int.from_bytes(body[4:8], "big")
            depth, color = body[8], body[9]
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    ch = {0: 1, 6: 4}[color] * (depth // 8)
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * ch + 1)
    assert not rows[:, 0].any()                      # filter 0 everywhere
    px = rows[:, 1:]
    if color == 0 and depth == 16:
        return px.copy().view(">u2").reshape(h, w).astype(np.uint16), color, depth
    if color == 6:
        return px.reshape(h, w, 4), color, depth
    return px.reshape(h, w), color, depth


def window(x0, y0, z, t, reason=1, cost=10.0):
    shp = (120, 120)
    return tl.Window(x0, y0, 0.05, np.full(shp, z, np.float32), np.full(shp, 1e-4, np.float32),
                     np.full(shp, 5, np.float32), np.full(shp, cost, np.float32),
                     np.full(shp, reason, np.float32), t_fused=np.full(shp, t))


class Base(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="t31_")
        self.site = "testsite"
        self.sdir = os.path.join(self.root, self.site)
        os.makedirs(self.sdir)
        with open(os.path.join(self.sdir, "datum.yaml"), "w") as f:
            yaml.safe_dump(dict(DATUM, version=2), f)
        with open(os.path.join(self.root, ".last_served"), "w") as f:
            json.dump({"site": self.site}, f)
        self.settings = os.path.join(self.root, "home", "terrain_recorder.json")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def core(self):
        c = tf.RecorderCore({"flush_s": 0.0, "settings_file": self.settings}, store_params={"accept_delay": 0.0},
                            sites_root=self.root)
        c.poll(T0, force=True)
        return c

    def fill(self, c, z=0.1, reason=1, cost=10.0):
        """A visit straight into the store (no gate needed)."""
        c.store.fuse(window(-3.0, -3.0, z, T0, reason, cost), (0.0, 0.0, 0.0), T0)
        c.store.close_all()
        c.store.save_dirty()

    def cmd(self, c, **kw):
        return c.command(json.dumps(dict(kw, id="t")), T0 + 1)


class Control(Base):
    def test_default_off_and_persistent_enable(self):
        c = self.core()
        st = c.state(T0)
        self.assertEqual((st["enabled"], st["mode"], st["v"]), (False, "off", 2))
        self.assertFalse(c.want_subscription(T0))
        r = self.cmd(c, cmd="enable")
        self.assertEqual((r["ok"], r["result"], r["id"]), (True, {"enabled": True}, "t"))
        self.assertTrue(json.load(open(self.settings))["enabled"])
        self.assertEqual(self.core().state(T0)["enabled"], True)        # survives a restart
        self.cmd(c, cmd="disable")
        self.assertEqual(self.core().state(T0)["mode"], "off")

    def test_pause_is_not_stored(self):
        c = self.core()
        self.cmd(c, cmd="enable")
        r = self.cmd(c, cmd="pause")
        self.assertEqual((r["ok"], c.state(T0)["mode"]), (True, "paused"))
        self.assertFalse(c.want_subscription(T0 + 100))
        self.assertEqual(self.core().state(T0)["mode"], "waiting")       # restart: by "enabled"
        self.cmd(c, cmd="resume")
        self.assertEqual(c.state(T0)["mode"], "waiting")

    def test_disabled_recorder_fuses_nothing(self):
        c = self.core()
        for t in np.arange(T0 - 15, T0 + 0.5, 0.5):
            c.gate.feed_loc(t, LOC_OK)
            c.gate.feed_pvt(t, 3, RTK_FLAGS, 14)
        lay = {k: np.zeros((120, 120), np.float32) for k in tf.LAYERS}
        lay["variance"][:] = 1e-4
        lay["points"][:] = 5
        g = tf.build_gridmap(T0, (-3, -3), 0.05, lay, (0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(c.feed(g, T0 + 0.3), "off")

    def test_unknown_and_bad(self):
        c = self.core()
        self.assertEqual(self.cmd(c, cmd="fly")["error"], "unknown_cmd")
        self.assertEqual(c.command("{bad", T0)["error"], "unknown_cmd")
        self.assertEqual(self.cmd(c, cmd="import", source="bag")["error"], "not_available")
        self.assertEqual(self.cmd(c, cmd="import", source="mosaic", name="nope")["error"], "unknown_import")
        self.assertEqual(c.state(T0)["last_cmd"]["error"], "unknown_import")

    def test_delete_map_with_token_and_backup(self):
        c = self.core()
        self.fill(c)
        r1 = self.cmd(c, cmd="delete_map")
        self.assertTrue(r1["ok"])
        tok = r1["result"]["confirm_token"]
        self.assertGreater(r1["result"]["area_m2"], 25.0)
        self.assertTrue(os.path.exists(os.path.join(self.sdir, "terrain", "index.json")))   # nothing yet
        self.assertEqual(self.cmd(c, cmd="delete_map", confirm="wrong")["error"], "bad_token")
        r1 = self.cmd(c, cmd="delete_map")                                  # a wrong one used it up
        r2 = self.cmd(c, cmd="delete_map", confirm=r1["result"]["confirm_token"])
        self.assertTrue(r2["ok"], r2)
        name = r2["result"]["backup"]
        self.assertTrue(os.path.exists(os.path.join(self.sdir, "terrain_backup", name, "index.json")))
        self.assertEqual(c.state(T0)["map"]["known_cells"], 0)
        self.assertEqual(c.state(T0)["backups"][0]["name"], name)
        self.assertGreater(c.state(T0)["backups"][0]["area_m2"], 25.0)
        self.assertNotEqual(tok, r1["result"]["confirm_token"])

    def test_token_only_in_the_answer(self):
        c = self.core()
        self.fill(c)
        r = self.cmd(c, cmd="delete_map")
        self.assertEqual(len(r["result"]["confirm_token"]), 6)              # cmd_result carries it
        lc = c.state(T0)["last_cmd"]
        self.assertEqual((lc["result"]["confirm_token"], lc["result"]["token_sent"]), (None, True))
        self.assertIn("confirm_token", c.last_cmd["result"])                 # the answer itself unchanged

    def test_long_command_marked_running(self):
        c = self.core()
        c.mark_running(json.dumps({"id": "x", "cmd": "delete_map", "confirm": "abc"}))
        self.assertEqual(c.state(T0)["last_cmd"], {"id": "x", "cmd": "delete_map", "ok": None, "error": None,
                                                   "result": {}, "running": True})
        c.mark_running(json.dumps({"id": "y", "cmd": "delete_map"}))         # step 1 is quick: unchanged
        self.assertEqual(c.state(T0)["last_cmd"]["id"], "x")
        r = self.cmd(c, cmd="list")
        self.assertFalse(r["running"])

    def test_token_expires(self):
        c = self.core()
        self.fill(c)
        tok = c.command(json.dumps({"cmd": "delete_map"}), T0)["result"]["confirm_token"]
        r = c.command(json.dumps({"cmd": "delete_map", "confirm": tok}), T0 + 61)
        self.assertEqual(r["error"], "bad_token")

    def test_import_merge_and_replace(self):
        c = self.core()
        self.fill(c, z=0.3)
        mdir = os.path.join(self.sdir, "terrain_import", "m1")
        os.makedirs(mdir)
        h, w = 100, 160
        el = np.full((h, w), 1.0, np.float32)                 # own height reference (+0.7 m)
        np.savez_compressed(os.path.join(mdir, "teren_m1_layers.npz"), elevation=el,
                            elevation_std=np.full((h, w), 0.01, np.float32), cost=np.full((h, w), 5.0, np.float32),
                            reason=np.ones((h, w), np.uint8), slope_deg=np.full((h, w), 2.0, np.float32),
                            step=np.zeros((h, w), np.float32), roughness=np.zeros((h, w), np.float32))
        json.dump({"frame": "map", "yaw": 0.0, "resolution": 0.05, "width": w, "height": h, "origin_x": -2.0,
                   "origin_y": -2.0, "files": {"layers": "teren_m1_layers.npz"}},
                  open(os.path.join(mdir, "teren_m1.json"), "w"))
        self.assertEqual(c.state(T0)["imports"], [{"name": "m1", "path": "terrain_import/m1"}])
        r = self.cmd(c, cmd="import", source="mosaic", name="m1", mode="merge")
        self.assertTrue(r["ok"], r)
        self.assertAlmostEqual(r["result"]["height_offset_m"], -0.7, delta=0.002)
        self.assertGreater(r["result"]["imported_cells"], 1000)
        self.assertIsNone(r["result"]["backup"])
        r = self.cmd(c, cmd="import", source="mosaic", name="m1", mode="replace")
        self.assertIn("confirm_token", r["result"])
        # a token given for m1 does not replace the map with another mosaic
        os.makedirs(os.path.join(self.sdir, "terrain_import", "m9"))
        shutil.copy(os.path.join(mdir, "teren_m1.json"), os.path.join(self.sdir, "terrain_import", "m9", "teren_m9.json"))
        shutil.copy(os.path.join(mdir, "teren_m1_layers.npz"), os.path.join(self.sdir, "terrain_import", "m9"))
        bad = self.cmd(c, cmd="import", source="mosaic", name="m9", mode="replace",
                       confirm=r["result"]["confirm_token"])
        self.assertEqual(bad["error"], "bad_token")
        r = self.cmd(c, cmd="import", source="mosaic", name="m1", mode="replace")
        r = self.cmd(c, cmd="import", source="mosaic", name="m1", mode="replace",
                     confirm=r["result"]["confirm_token"])
        self.assertTrue(r["ok"], r)
        self.assertIsNotNone(r["result"]["backup"])
        self.assertEqual(r["result"]["imported_cells"], h * w)
        self.assertEqual(c.state(T0)["map"]["known_cells"], h * w)


class Export(Base):
    def test_files_like_the_mosaic(self):
        c = self.core()
        # a window 0.1 m high with a patch 0.3 m higher in its north-east
        w = window(-3.0, -3.0, 0.1, T0 + 60)
        w.elevation[90:110, 90:110] = 0.4
        c.store.fuse(w, (0.0, 0.0, 0.0), T0 + 60)
        c.store.close_all()
        meta = c.export_now()                                     # the niced child process, from the files
        self.assertTrue(meta["ok"], meta)
        out = os.path.join(self.sdir, "terrain_mosaic")
        self.assertEqual(meta["version"], 1)
        with open(os.path.join(out, "teren_testsite.json")) as f:
            j = json.load(f)
        for k in ("frame", "resolution", "width", "height", "yaw", "origin_x", "origin_y", "corner_note",
                  "north_angle_rad", "files", "encoding", "reasons_ui", "source", "version", "area_m2",
                  "generated", "generated_unix", "nogo_candidates"):
            self.assertIn(k, j)
        self.assertEqual((j["frame"], j["source"], j["resolution"]), ("map", "terrain_recorder", 0.05))
        self.assertAlmostEqual(j["north_angle_rad"], np.pi / 2)
        e16, color, depth = read_png(os.path.join(out, j["files"]["elevation16"]))
        self.assertEqual((e16.shape, color, depth), ((j["height"], j["width"]), 0, 16))
        cost, _, d8 = read_png(os.path.join(out, j["files"]["cost"]))
        rgba, c6, _ = read_png(os.path.join(out, j["files"]["cost_rgba"]))
        reason, _, _ = read_png(os.path.join(out, j["files"]["reason"]))
        self.assertEqual((d8, c6, rgba.shape[2]), (8, 6, 4))

        def px(x, y):          # the contract's pixel centre formula, inverted
            col = int((x - j["origin_x"]) / j["resolution"])
            row = j["height"] - 1 - int((y - j["origin_y"]) / j["resolution"])
            return row, col
        r, cc = px(0.0, 0.0)
        self.assertEqual(int(e16[r, cc]), 1)                     # lowest point = 1 mm
        r, cc = px(2.0, 2.0)                                      # the patch: +0.3 m
        self.assertEqual(int(e16[r, cc]), 301)
        self.assertEqual((int(cost[r, cc]), int(reason[r, cc]), int(rgba[r, cc, 3])), (10, 1, 200))
        self.assertEqual((int(cost[0, 0]), int(rgba[0, 0, 3])), (255, 0))   # margin: unknown
        self.assertAlmostEqual(j["area_m2"], c.state(T0)["map"]["area_m2"], delta=1.0)
        self.assertEqual(j["files"]["cost"], "teren_testsite_v1_cost.png")
        self.assertEqual(c.export_now()["version"], 2)
        self.assertEqual(c.state(T0)["export"]["version"], 2)

    def test_nogo_candidates_only_listed(self):
        c = self.core()
        c.store.p["confirm_sessions"] = 1                         # confirm from one session for the test
        c.store.p["confirm_dirs"] = 1
        for i in range(3):
            w = window(-3.0, -3.0, 0.1, T0 + 60 * i)
            w.reason[40:60, 40:52] = 3                            # 1.0 x 0.6 m step band
            w.cost[40:60, 40:52] = 100
            c.store.fuse(w, (0.0, 0.0, 0.0), T0 + 60 * i)
            c.store.close_all()
        c.export_now()
        cands = c.state(T0)["nogo_candidates"]
        self.assertEqual(len(cands), 1)
        k = cands[0]
        self.assertEqual((k["kind"], k["cells"]), ("step", 240))
        self.assertAlmostEqual(k["area_m2"], 0.6, places=3)
        self.assertAlmostEqual(k["centroid"][0], -3.0 + 46 * 0.05, delta=0.03)
        xs = [p[0] for p in k["polygon"]]
        self.assertAlmostEqual(min(xs), -3.0 + 40 * 0.05 - 0.15, delta=0.03)   # hull + 0.15 m
        self.assertFalse(os.path.exists(os.path.join(self.sdir, "places.geojson")))

    def test_small_cluster_and_tentative_not_listed(self):
        c = self.core()
        for i in range(3):
            w = window(-3.0, -3.0, 0.1, T0 + 60 * i)
            w.reason[40:60, 40:52] = 3
            w.cost[40:60, 40:52] = 100
            c.store.fuse(w, (0.0, 0.0, 0.0), T0 + 60 * i)
            c.store.close_all()
        c.export_now()                                            # one session, one heading: tentative
        self.assertEqual(c.state(T0)["nogo_candidates"], [])


class Guards(Base):
    def mosaic(self, name, datum=None):
        mdir = os.path.join(self.sdir, "terrain_import", name)
        os.makedirs(mdir)
        h, w = 20, 20
        np.savez_compressed(os.path.join(mdir, "teren_%s_layers.npz" % name), elevation=np.zeros((h, w), np.float32),
                            elevation_std=np.full((h, w), 0.01, np.float32), cost=np.zeros((h, w), np.float32),
                            reason=np.ones((h, w), np.uint8), slope_deg=np.zeros((h, w), np.float32),
                            step=np.zeros((h, w), np.float32), roughness=np.zeros((h, w), np.float32))
        with open(os.path.join(mdir, "teren_%s.json" % name), "w") as f:
            json.dump({"frame": "map", "yaw": 0.0, "resolution": 0.05, "width": w, "height": h, "origin_x": 0.0,
                       "origin_y": 0.0, "files": {"layers": "teren_%s_layers.npz" % name}}, f)
        if datum is not None:
            with open(os.path.join(mdir, "datum.yaml"), "w") as f:
                yaml.safe_dump(datum, f)

    def test_utm_zone_and_small_datum_delta(self):
        c = self.core()
        self.mosaic("z34", dict(DATUM, utm_zone=34))
        r = self.cmd(c, cmd="import", source="mosaic", name="z34", mode="merge")
        self.assertTrue(r["error"].startswith("utm_zone_mismatch"), r)
        self.mosaic("near", dict(DATUM, utm_e=E0 + 0.3, yaw_rad=0.01))      # 0.3 m, 0.57 deg: accepted
        r = self.cmd(c, cmd="import", source="mosaic", name="near", mode="merge")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["datum_delta"]["dE"], 0.3)
        self.mosaic("turned", dict(DATUM, yaw_rad=0.03))                     # 1.7 deg: refused
        r = self.cmd(c, cmd="import", source="mosaic", name="turned", mode="replace")
        self.assertTrue(r["error"].startswith("datum_differs"), r)          # already at the token step

    def test_export_reads_only(self):
        c = self.core()
        self.fill(c)
        tdir = os.path.join(self.sdir, "terrain")
        tile = sorted(f for f in os.listdir(tdir) if f.startswith("t_"))[0]
        with open(os.path.join(tdir, tile), "wb") as f:
            f.write(b"broken")
        before = sorted(os.listdir(tdir))
        res = c.export_now()
        self.assertTrue(res["ok"], res)
        self.assertEqual(sorted(os.listdir(tdir)), before)                  # nothing moved, no broken/
        self.assertFalse(os.path.exists(os.path.join(tdir, "broken")))

    def sleeper(self, c, age=0.0):
        import subprocess
        out = tempfile.TemporaryFile(mode="w+")
        p = subprocess.Popen(["sleep", "30"], stdout=out)
        c.export_proc = (p, out, "interval", __import__("time").time() - age)
        return p

    def test_export_timeout(self):
        c = tf.RecorderCore({"flush_s": 0.0, "settings_file": self.settings, "export_timeout_s": 5.0},
                            store_params={"accept_delay": 0.0}, sites_root=self.root)
        c.poll(T0, force=True)
        p = self.sleeper(c, age=10.0)
        c.export_tick(T0)
        self.assertIsNotNone(p.poll())                                      # killed
        self.assertEqual((c.export_info["ok"], c.export_info["error"]), (False, "timeout"))

    def test_import_replace_kills_a_running_export(self):
        c = self.core()
        self.fill(c)
        self.mosaic("m3")
        p = self.sleeper(c)
        r = self.cmd(c, cmd="import", source="mosaic", name="m3", mode="replace")
        r = self.cmd(c, cmd="import", source="mosaic", name="m3", mode="replace", confirm=r["result"]["confirm_token"])
        self.assertTrue(r["ok"], r)
        self.assertIsNotNone(p.poll())
        self.assertIsNone(c.export_proc)

    def test_versions_never_repeat_after_delete(self):
        c = self.core()
        self.fill(c)
        c.export_now()
        c.export_now()
        tok = self.cmd(c, cmd="delete_map")["result"]["confirm_token"]
        self.cmd(c, cmd="delete_map", confirm=tok)
        self.fill(c)
        self.assertEqual(c.export_now()["version"], 3)                      # not 1 again
        with open(os.path.join(self.sdir, "terrain_export.version")) as f:
            self.assertEqual(f.read().strip(), "3")


class ExportVersions(Base):
    def test_versioned_files_and_cleanup_after_the_json(self):
        c = self.core()
        self.fill(c)
        out = os.path.join(self.sdir, "terrain_mosaic")
        for _ in range(4):
            c.export_now()
        with open(os.path.join(out, "teren_testsite.json")) as f:
            j = json.load(f)
        self.assertEqual(j["version"], 4)
        pngs = sorted(p for p in os.listdir(out) if p.endswith(".png"))
        # the JSON's version and the one before stay (a reader with the old JSON finds its images)
        self.assertEqual(pngs, sorted("teren_testsite_v%d_%s.png" % (v, k) for v in (3, 4)
                                      for k in ("cost", "reason", "cost_rgba", "elevation16")))
        for k, name in j["files"].items():
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)

    def test_async_export_interval_and_commands(self):
        c = tf.RecorderCore({"flush_s": 0.0, "settings_file": self.settings, "export_s": 600.0},
                            store_params={"accept_delay": 0.0}, sites_root=self.root)
        c.poll(T0, force=True)
        self.fill(c)
        c.export_dirty = True
        c.export_tick(T0)                                     # due: no export yet -> starts
        self.assertIsNotNone(c.export_proc)
        self.assertTrue(c.state(T0)["export"]["running"])
        c._collect_export(wait=True)
        self.assertEqual((c.export_info["version"], c.export_info["reason"]), (1, "interval"))
        self.assertGreaterEqual(c.export_info["cpu_s"], 0.0)
        c.export_dirty = True
        c.export_tick(T0 + 60)                                # changed, but < export_s since the last
        self.assertIsNone(c.export_proc)
        r = self.cmd(c, cmd="rebuild_export")
        self.assertEqual(r["result"], {"queued": True})
        c.export_tick(T0 + 61)
        c._collect_export(wait=True)
        self.assertEqual((c.export_info["version"], c.export_info["reason"]), (2, "rebuild_export"))

    def test_backups_pruned_to_the_newest(self):
        c = tf.RecorderCore({"flush_s": 0.0, "settings_file": self.settings, "backup_keep": 2},
                            store_params={"accept_delay": 0.0}, sites_root=self.root)
        c.poll(T0, force=True)
        root = os.path.join(self.sdir, "terrain_backup")
        os.makedirs(root)
        for name in ("20260101-000000", "20260102-000000", "20260103-000000"):
            os.makedirs(os.path.join(root, name))
        self.fill(c)
        tok = self.cmd(c, cmd="delete_map")["result"]["confirm_token"]
        name = self.cmd(c, cmd="delete_map", confirm=tok)["result"]["backup"]
        self.assertEqual(sorted(os.listdir(root)), sorted(["20260103-000000", name]))

    def test_import_uses_the_mosaic_datum(self):
        c = self.core()
        mdir = os.path.join(self.sdir, "terrain_import", "m2")
        os.makedirs(mdir)
        h, w = 20, 20
        np.savez_compressed(os.path.join(mdir, "teren_m2_layers.npz"), elevation=np.zeros((h, w), np.float32),
                            elevation_std=np.full((h, w), 0.01, np.float32), cost=np.zeros((h, w), np.float32),
                            reason=np.ones((h, w), np.uint8), slope_deg=np.zeros((h, w), np.float32),
                            step=np.zeros((h, w), np.float32), roughness=np.zeros((h, w), np.float32))
        with open(os.path.join(mdir, "teren_m2.json"), "w") as f:
            json.dump({"frame": "map", "yaw": 0.0, "resolution": 0.05, "width": w, "height": h, "origin_x": 0.0,
                       "origin_y": 0.0, "files": {"layers": "teren_m2_layers.npz"}}, f)
        # the mosaic was made when the map origin lay 5 m further east
        with open(os.path.join(mdir, "datum.yaml"), "w") as f:
            yaml.safe_dump(dict(DATUM, utm_e=E0 + 5.0), f)
        r = self.cmd(c, cmd="import", source="mosaic", name="m2", mode="merge")
        self.assertTrue(r["error"].startswith("datum_differs: dE 5.00 m"), r)      # > 0.5 m: refused (F3)
        r = self.cmd(c, cmd="import", source="mosaic", name="m2", mode="merge", force=True)
        self.assertEqual((r["ok"], r["result"]["datum"]), (True, "mosaic"))
        self.assertEqual(r["result"]["datum_delta"], {"dE": 5.0, "dN": 0.0, "dyaw_deg": 0.0})
        self.assertTrue(np.isfinite(c.store.sample_utm(E0 + 5.5, N0 + 0.5)["h"][0]))
        self.assertFalse(np.isfinite(c.store.sample_utm(E0 + 0.5, N0 + 0.5)["h"][0]))


if __name__ == "__main__":
    unittest.main()
