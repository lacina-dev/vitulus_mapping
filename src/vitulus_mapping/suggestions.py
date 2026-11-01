#!/usr/bin/env python3
"""
suggestions.py -- robot suggestions: persistent new obstacles seen while driving.

PURE PYTHON (json + math + fcntl). NO rospy / numpy. Contract: docs/MAP_OBJECTS.md
section 7.

Two processes touch <site>/suggestions.json: map_world_recorder feeds sightings
(`observe`), map_objects_server applies accept/ignore (`accept_object`, `mark`)
and publishes `public`. Every read-modify-write therefore holds an flock on
suggestions.json.lock; the file itself is replaced atomically, so plain readers
never need the lock.

File: {"version", "site", "seq", "updated", "entries": [entry]}
Entry: {"id", "kind", "points", "size_m", "seen", "first", "last", "state",
        "c": [e, n], "spots": [[e, n], ...]}
`points` and `c` are UTM (files are UTM, contract section 2), `spots` are the
distinct robot positions a thing was seen from. State "candidate" is internal:
a thing that has not met the thresholds yet. `public` hides it and converts to
the map frame.

Ignored entries are never deleted: they are what keeps a place ignored.
"""

import fcntl
import hashlib
import json
import math
import os
import time

from . import bundle
from . import geo

__all__ = ["DEFAULTS", "path", "mtime", "load", "public", "observe",
           "accept_object", "mark", "match_object"]

FILENAME = "suggestions.json"
VERSION = 1
STATES = ("new", "accepted", "ignored")

DEFAULTS = {
    "min_seen": 3,            # contract: seen at least 3 times ...
    "min_positions": 2,       # ... from at least 2 positions ...
    "min_span_s": 120.0,      # ... over at least 2 minutes
    "min_gap_s": 20.0,        # sightings closer in time count once
    "position_sep_m": 1.0,    # robot positions this far apart are distinct
    "merge_m": 0.5,           # sighting within this of an entry = same thing
    "ignore_m": 0.7,          # ignored/accepted entry silences this radius
    "object_margin_m": 0.6,   # sighting within this of an object = known
    "protected_m": 1.0,       # keep clear of DOCK/UNDOCK/docked
    "max_extent_m": 6.0,      # larger = map shift or a wall, not a new thing
    "landmark_max_m": 0.5,    # compact thing = landmark, otherwise obstacle
    "min_outline_m": 0.3,     # smallest obstacle square (validator: 0.05 m2)
    "candidate_ttl_s": 14 * 86400.0,   # unconfirmed sightings are forgotten
    "new_ttl_s": 30 * 86400.0,         # a new one not seen for a month too
    "max_entries": 500,
}

# Types that say "this thing is already known". A zone, path or waypoint does
# not: a new stone in a zone is exactly what a suggestion is for. A `place`
# does not either: an area named "zahrada" must not swallow every suggestion.
_KNOWN_TYPES = ("obstacle", "wall", "free", "nogo", "landmark")

# type -> (family, geometry) allowed by accept
_ACCEPT_TYPES = {
    "obstacle": ("edit", "area"),
    "landmark": ("place", "point"),
    "place": ("place", "area"),
    "nogo": ("place", "area"),
}


# --------------------------------------------------------------------------
# file
# --------------------------------------------------------------------------
def path(site):
    return os.path.join(bundle.site_dir(site), FILENAME)


def mtime(site):
    """mtime of suggestions.json, 0.0 when there is none."""
    try:
        return os.path.getmtime(path(site))
    except OSError:
        return 0.0


def _empty(site):
    return {"version": VERSION, "site": site, "seq": 0, "updated": 0.0, "entries": []}


def load(site):
    """The whole file. Missing = empty. An unreadable file is set aside, never
    overwritten, because it may hold ignored places."""
    p = path(site)
    try:
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise ValueError("bad shape")
        return data
    except (IOError, OSError):
        return _empty(site)
    except ValueError:
        try:
            os.replace(p, "%s.bad.%d" % (p, int(time.time())))
        except OSError:
            pass
        return _empty(site)


def _save(site, data, now):
    data["seq"] = int(data.get("seq", 0)) + 1
    data["updated"] = float(now)
    p = path(site)
    tmp = "%s.tmp.%d" % (p, os.getpid())
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


class _Locked(object):
    """Holds the site's lock file across a read-modify-write."""

    def __init__(self, site):
        self.site = site
        self.fd = None

    def __enter__(self):
        d = bundle.site_dir(self.site)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        self.fd = open(path(self.site) + ".lock", "w")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()
        return False


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _dist_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    u = 0.0 if n <= 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / n))
    return math.hypot(px - (ax + u * dx), py - (ay + u * dy))


def _dist_line(px, py, pts, closed=False):
    if len(pts) == 1:
        return math.hypot(px - pts[0][0], py - pts[0][1])
    best = float("inf")
    for i in range(len(pts) - 1):
        best = min(best, _dist_segment(px, py, pts[i][0], pts[i][1],
                                       pts[i + 1][0], pts[i + 1][1]))
    if closed:
        best = min(best, _dist_segment(px, py, pts[-1][0], pts[-1][1],
                                       pts[0][0], pts[0][1]))
    return best


def _inside(px, py, ring):
    hit = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit


def match_object(x, y, objects, margin=None, protected_m=None):
    """The object a map-frame point belongs to, or None. `objects` are contract
    objects with map-frame points."""
    margin = DEFAULTS["object_margin_m"] if margin is None else margin
    protected_m = DEFAULTS["protected_m"] if protected_m is None else protected_m
    for obj in objects or ():
        pts = obj.get("points") or []
        if not pts:
            continue
        if obj.get("protected"):
            if _dist_line(x, y, pts) <= protected_m:
                return obj
        if obj.get("type") not in _KNOWN_TYPES:
            continue
        area = obj.get("geometry") == "area"
        if area and _inside(x, y, pts):
            return obj
        if _dist_line(x, y, pts, closed=area) <= margin:
            return obj
    return None


def _outline(entry, cfg):
    """UTM `points`: one point for a landmark, a square for an obstacle."""
    e, n = entry["c"]
    if entry["kind"] == "landmark":
        return [[round(e, 3), round(n, 3)]]
    h = max(float(entry["size_m"]), cfg["min_outline_m"]) / 2.0
    return [[round(e - h, 3), round(n - h, 3)], [round(e + h, 3), round(n - h, 3)],
            [round(e + h, 3), round(n + h, 3)], [round(e - h, 3), round(n + h, 3)]]


def _new_id(e, n, now):
    raw = "%.2f:%.2f:%.3f" % (e, n, now)
    return "s_" + hashlib.sha1(raw.encode("ascii")).hexdigest()[:8]


# --------------------------------------------------------------------------
# recorder side
# --------------------------------------------------------------------------
def _sight(entry, e, n, re_, rn, size, now, cfg):
    seen = int(entry["seen"])
    cx, cy = entry["c"]
    entry["c"] = [(cx * seen + e) / (seen + 1), (cy * seen + n) / (seen + 1)]
    entry["seen"] = seen + 1
    entry["last"] = float(now)
    entry["size_m"] = round(max(float(entry["size_m"]), size), 2)
    if len(entry["spots"]) < 8 and all(
            math.hypot(s[0] - re_, s[1] - rn) >= cfg["position_sep_m"]
            for s in entry["spots"]):
        entry["spots"].append([round(re_, 2), round(rn, 2)])
    entry["kind"] = ("landmark" if entry["size_m"] <= cfg["landmark_max_m"]
                     else "obstacle")
    entry["points"] = _outline(entry, cfg)


def _confirmed(entry, cfg):
    return (int(entry["seen"]) >= cfg["min_seen"]
            and len(entry["spots"]) >= cfg["min_positions"]
            and float(entry["last"]) - float(entry["first"]) >= cfg["min_span_s"])


def observe(site, rows, robot, datum, objects=None, now=None, cfg=None):
    """Feed one look around the robot.

    rows    world_query.obstacles() rows; only kind == "static_new" counts
    robot   {"x", "y"} map frame, where the robot stood
    datum   canonical datum of the site
    objects contract objects (map frame), None = not known right now

    Returns {"added", "confirmed", "seen", "dropped"}. The file is written only
    when something changed, so a look at nothing costs no disk write.
    """
    conf = dict(DEFAULTS)
    conf.update(cfg or {})
    now = time.time() if now is None else float(now)
    stats = {"added": 0, "confirmed": 0, "seen": 0, "dropped": 0}
    if datum is None or robot is None:
        return stats
    with _Locked(site):
        data = load(site)
        entries = data["entries"]
        changed = False

        # expiry and things that became known by other means
        keep = []
        for ent in entries:
            state = ent.get("state")
            age = now - float(ent.get("last", now))
            drop = ((state == "candidate" and age > conf["candidate_ttl_s"])
                    or (state == "new" and age > conf["new_ttl_s"]))
            if not drop and state in ("candidate", "new") and objects is not None:
                x, y = geo.utm_to_map(ent["c"][0], ent["c"][1], datum)
                drop = match_object(x, y, objects, conf["object_margin_m"],
                                    conf["protected_m"]) is not None
            if drop:
                stats["dropped"] += 1
                changed = True
            else:
                keep.append(ent)
        entries = keep

        re_, rn = geo.map_to_utm(float(robot["x"]), float(robot["y"]), datum)
        touched = set()
        for row in rows or ():
            if row.get("kind") != "static_new":
                continue
            size = float(row.get("extent_m") or 0.0)
            if size > conf["max_extent_m"]:
                continue
            x, y = float(row["centroid"][0]), float(row["centroid"][1])
            if match_object(x, y, objects, conf["object_margin_m"],
                            conf["protected_m"]) is not None:
                continue
            e, n = geo.map_to_utm(x, y, datum)
            silent = False
            near, near_d = None, conf["merge_m"]
            for ent in entries:
                d = math.hypot(ent["c"][0] - e, ent["c"][1] - n)
                if ent.get("state") in ("ignored", "accepted"):
                    silent = silent or d <= conf["ignore_m"]
                elif d <= near_d:
                    near, near_d = ent, d
            if silent:
                continue
            if near is None:
                if len(entries) >= conf["max_entries"]:
                    continue
                ent = {"id": _new_id(e, n, now), "kind": "obstacle", "points": [],
                       "size_m": round(size, 2), "seen": 0, "first": now,
                       "last": 0.0, "state": "candidate", "c": [e, n], "spots": []}
                _sight(ent, e, n, re_, rn, size, now, conf)
                entries.append(ent)
                touched.add(ent["id"])
                stats["added"] += 1
                changed = True
                continue
            if near["id"] in touched or now - float(near["last"]) < conf["min_gap_s"]:
                continue
            _sight(near, e, n, re_, rn, size, now, conf)
            touched.add(near["id"])
            stats["seen"] += 1
            changed = True
            if near["state"] == "candidate" and _confirmed(near, conf):
                near["state"] = "new"
                stats["confirmed"] += 1

        if changed:
            data["entries"] = entries
            data["site"] = site
            _save(site, data, now)
    return stats


# --------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------
def public(site, datum):
    """Payload of /map_objects/suggestions, map frame. No datum = empty list."""
    data = load(site)
    out = []
    if datum is not None:
        for ent in data["entries"]:
            if ent.get("state") not in STATES:
                continue
            out.append({"id": ent["id"], "kind": ent["kind"],
                        "points": [[round(v, 3) for v in geo.utm_to_map(p[0], p[1], datum)]
                                   for p in ent["points"]],
                        "size_m": ent["size_m"], "seen": ent["seen"],
                        "first": ent["first"], "last": ent["last"],
                        "state": ent["state"]})
    return {"site": site, "seq": int(data.get("seq", 0)),
            "stamp": float(data.get("updated", 0.0)), "suggestions": out}


def _find(data, sid):
    for ent in data["entries"]:
        if ent.get("id") == sid and ent.get("state") in STATES:
            return ent
    return None


def accept_object(site, sid, as_=None, datum=None, cfg=None):
    """Contract object for an accepted suggestion, without `id`, map frame, for
    the server's normal save. Returns (obj, reason); obj None = refused. Changes
    no state: the server calls mark(..., "accepted") after the save succeeded."""
    conf = dict(DEFAULTS)
    conf.update(cfg or {})
    as_ = as_ if isinstance(as_, dict) else {}
    if datum is None:
        return None, "no datum for site %s" % site
    ent = _find(load(site), sid)
    if ent is None:
        return None, "unknown suggestion %s" % sid
    if ent["state"] != "new":
        return None, "suggestion %s is already %s" % (sid, ent["state"])
    typ = str(as_.get("type") or ent["kind"])
    if typ not in _ACCEPT_TYPES:
        return None, "cannot accept as type %s" % typ
    family, shape = _ACCEPT_TYPES[typ]
    # the suggestion's own shape follows its kind, the requested type may differ
    wanted = dict(ent, kind="landmark" if shape == "point" else "obstacle")
    utm = _outline(wanted, conf)
    points = [[round(v, 3) for v in geo.utm_to_map(p[0], p[1], datum)] for p in utm]
    obj = {"family": family, "type": typ, "geometry": shape, "points": points,
           "name": str(as_.get("name") or "robot_" + sid[2:]),
           "yaw": None, "aliases": list(as_.get("aliases") or []),
           "note": str(as_.get("note") or ""), "tags": list(as_.get("tags") or []),
           "source": "robot", "props": dict(as_.get("props") or {})}
    return obj, ""


def mark(site, sid, state, now=None):
    """Set the state of a suggestion: "accepted", "ignored" or back to "new"."""
    if state not in STATES:
        return False, "bad state %s" % state
    with _Locked(site):
        data = load(site)
        ent = _find(data, sid)
        if ent is None:
            return False, "unknown suggestion %s" % sid
        if ent["state"] != state:
            ent["state"] = state
            _save(site, data, time.time() if now is None else now)
    return True, ""
