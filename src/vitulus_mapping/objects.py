#!/usr/bin/env python3
"""objects.py -- map objects: one view over every file of a site (docs/MAP_OBJECTS.md).

PURE PYTHON (bundle, geo, objnames; yaml + json + math). NO rospy / numpy / ROS:
the map_objects_server node, mapping_manager, the agent and the tests all read
through here, so a rule means the same thing everywhere.

An object is a plain dict (contract section 2). Objects read from files are in
UTM, topics are in the map frame: load_all(site) gives the map frame, to_frame()
converts. `active` is computed in exactly one place, is_active().

A missing file means no information: load_families() says None for that family,
which is not the same as an empty list and must never be read as "all deleted".
"""

import hashlib
import json
import math
import os
import time
import uuid

import yaml

from vitulus_mapping import bundle, geo, objnames

__all__ = [
    "FAMILIES", "RULE_TYPES", "family_of_type", "geometry_of_type",
    "new_id", "default_id", "meta_key", "is_protected", "is_active",
    "load_datum", "load_family", "load_families", "load_all", "active_rules", "load_rules",
    "rules_signature", "stampable", "nogo_stamp_edits", "find_id", "find_by_id",
    "validate", "to_frame", "same_content", "diff_family",
    "load_meta", "save_meta", "load_places", "save_places",
    "places_path", "meta_path", "history_dir", "suggestions_path",
    "program_zone_names", "zone_back_in_order", "owner_nogo_problem",
    "polygon_area", "self_crossing", "point_in_polygon", "polygons_overlap",
    "polygon_covers",
]

FAMILIES = ("zone", "waypoint", "path", "edit", "place")
RULE_TYPES = ("nogo", "slow", "noblade")

_FAMILY_OF = {
    "zone": "zone", "waypoint": "waypoint", "path": "path",
    "obstacle": "edit", "free": "edit", "wall": "edit",
    "place": "place", "landmark": "place", "nogo": "place", "slow": "place",
    "noblade": "place", "corridor": "place",
}
_GEOMETRY_OF = {
    "zone": "area", "waypoint": "point", "path": "line",
    "obstacle": "area", "free": "area", "wall": "line",
    "place": "area", "landmark": "point", "nogo": "area", "slow": "area",
    "noblade": "area", "corridor": "line",
}
SOURCES = ("owner", "agent", "robot")
# Sidecar fields. Everything else of a non-place object belongs to its owner.
META_FIELDS = ("id", "aliases", "note", "tags", "source", "created", "updated")
# Written by the owner as a by-product, not a change of the object. `effective`
# comes from the planner, a list per overriding program; never sent back to it.
# `programs` travels in the journal only: which programs held a deleted zone.
COMPUTED_PROPS = {"zone": ("area", "length", "effective", "programs"), "waypoint": ("z",)}
ZONE_DEFAULTS = {"cut_height": 65, "rpm": 3300, "type": "normal", "border_paths": 2,
                 "coverage_angle": 0, "paths_distance": 0.2, "simplify": 0.1}
ZONE_INT_PROPS = ("cut_height", "rpm", "border_paths", "coverage_angle")
CUT_HEIGHT_MM = (20, 80)
SPEED_MPS = (0.05, 0.40)
DEFAULT_SLOW_MPS = 0.20
DEFAULT_CORRIDOR_M = 0.60
MIN_AREA_M2 = 0.05
# Round trips through the owners (map -> UTM -> float32) are not exact.
POINT_TOL_M = 0.005
YAW_TOL_RAD = 0.002


def family_of_type(obj_type):
    return _FAMILY_OF.get(obj_type)


def geometry_of_type(obj_type):
    return _GEOMETRY_OF.get(obj_type)


def new_id():
    return "o_" + uuid.uuid4().hex[:8]


def default_id(family, name):
    """Id of an object nobody has given one yet. Deterministic, so a reader that
    cannot write the sidecar sees the id the server will store."""
    h = hashlib.sha1(("%s:%s" % (family, name)).encode("utf-8")).hexdigest()
    return "o_" + h[:8]


def meta_key(family, name):
    return "%s:%s" % (family, name)


def is_protected(family, name):
    """Case-insensitive: sites not migrated yet still hold `docked`."""
    return family == "waypoint" and bool(name) and objnames.is_dock_point(name)


def is_active(obj, now=None):
    """The only place that computes `active`."""
    if obj.get("type") not in RULE_TYPES:
        return True
    t = time.time() if now is None else now
    vf, vu = obj.get("valid_from"), obj.get("valid_until")
    # A bound that is no number limits nothing: a rule rather stays on.
    return (not _number(vf) or t >= vf) and (not _number(vu) or t < vu)


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def places_path(site):
    return os.path.join(bundle.site_dir(site), "places.geojson")


def meta_path(site):
    return os.path.join(bundle.site_dir(site), "objects_meta.json")


def history_dir(site):
    return os.path.join(bundle.site_dir(site), "history")


def suggestions_path(site):
    return os.path.join(bundle.site_dir(site), "suggestions.json")


def load_datum(site):
    """Canonical datum of the site, or None (no file, unreadable, no origin)."""
    try:
        with open(os.path.join(bundle.site_dir(site), "datum.yaml")) as f:
            d = yaml.safe_load(f)
        if not d or "utm_e" not in d:
            return None
        return geo.from_datum_yaml(d)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return None


def load_meta(site):
    """{key: meta}; {} when the file is missing, None when it is unreadable (the
    caller must not overwrite what it could not read)."""
    p = meta_path(site)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            m = json.load(f)
        return m if isinstance(m, dict) else None
    except (OSError, ValueError):
        return None


def save_meta(site, meta):
    bundle._write_json(meta_path(site), meta)


def _blank(family, obj_type, name, points):
    return {"id": None, "family": family, "type": obj_type, "name": name,
            "geometry": _GEOMETRY_OF[obj_type], "points": points, "yaw": None,
            "aliases": [], "note": "", "tags": [], "source": "owner",
            "created": 0.0, "updated": 0.0,
            "valid_from": None, "valid_until": None, "active": True,
            "protected": False, "props": {}}


def _feature_points(feat):
    """Points of a places feature, closing vertex dropped. ValueError if unreadable."""
    try:
        g = feat["geometry"]
        c = g["coordinates"]
        if g["type"] == "Point":
            pts = [[float(c[0]), float(c[1])]]
        elif g["type"] == "LineString":
            pts = [[float(v[0]), float(v[1])] for v in c]
        else:
            pts = [[float(v[0]), float(v[1])] for v in c[0]]
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
    except (KeyError, TypeError, IndexError, ValueError):
        raise ValueError("bad_geometry")
    if not all(math.isfinite(v) for p in pts for v in p):
        raise ValueError("bad_geometry")
    return pts


def _place_from_feature(feat):
    """One object, or None for a feature of another family. A field of the wrong
    kind is dropped here, so no reader trips over it; ValueError(reason) when the
    object cannot stand: unknown type, or no shape of its kind."""
    pr = feat.get("properties") if isinstance(feat, dict) else None
    if not isinstance(pr, dict):
        raise ValueError("bad_type")
    t = pr.get("type")
    if not isinstance(t, str) or t not in _FAMILY_OF:
        raise ValueError("bad_type")
    if _FAMILY_OF[t] != "place":
        return None
    pts = _feature_points(feat)
    if len(pts) < {"point": 1, "line": 2, "area": 3}[_GEOMETRY_OF[t]]:
        raise ValueError("bad_geometry")
    rid = pr.get("id") if isinstance(pr.get("id"), str) and pr.get("id") else None
    name = pr.get("name") if isinstance(pr.get("name"), str) and pr.get("name") else None
    o = _blank("place", t, name or rid or "unnamed", pts)
    o["id"] = rid
    for k, ok in (("yaw", _number), ("created", _number), ("updated", _number),
                  ("aliases", _is_list), ("tags", _is_list), ("props", _is_dict),
                  ("note", _is_str), ("source", _is_str)):
        if ok(pr.get(k)):
            o[k] = pr[k]
    o["valid_from"], o["valid_until"] = pr.get("valid_from"), pr.get("valid_until")
    return o


def _is_list(v):
    return isinstance(v, list)


def _is_dict(v):
    return isinstance(v, dict)


def _is_str(v):
    return isinstance(v, str)


def _bad_feature(feat, reason):
    pr = feat.get("properties") if isinstance(feat, dict) else None
    pr = pr if isinstance(pr, dict) else {}
    try:
        pts = _feature_points(feat)
    except ValueError:
        pts = []
    text = lambda v: v if isinstance(v, str) and v else None
    return {"id": text(pr.get("id")), "name": text(pr.get("name")),
            "type": text(pr.get("type")), "reason": reason, "points": pts,
            "feature": feat}


def load_places(site, bad=None):
    """List of objects in UTM, or None when the file is missing or unreadable as a
    whole. One bad feature costs only itself: it goes to the list `bad` as
    {id, name, type, reason, points, feature}. Whoever rewrites the file has to
    hand those to save_places(keep=...)."""
    p = places_path(site)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            features = json.load(f)["features"]
        if not isinstance(features, list):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    out = []
    for feat in features:
        try:
            o = _place_from_feature(feat)
        except ValueError as e:
            if bad is not None:
                bad.append(_bad_feature(feat, str(e)))
            continue
        if o is not None:
            out.append(o)
    return out


def save_places(site, places, utm_zone=None, keep=()):
    """places: objects in UTM. Atomic, same crs/note scaffold as the bundle.
    keep: raw features that could not be read, written back untouched."""
    zone = bundle._utm_zone_for(site, utm_zone)
    feats = []
    for o in places:
        pts = [[float(p[0]), float(p[1])] for p in o["points"]]
        if o["geometry"] == "point":
            g = {"type": "Point", "coordinates": pts[0]}
        elif o["geometry"] == "line":
            g = {"type": "LineString", "coordinates": pts}
        else:
            g = {"type": "Polygon", "coordinates": [pts + pts[:1]]}
        pr = {k: o.get(k) for k in ("id", "type", "name", "yaw", "aliases", "note",
                                    "tags", "source", "created", "updated",
                                    "valid_from", "valid_until", "props")}
        feats.append(bundle._feature(g, pr))
    bundle._write_json(places_path(site), bundle._fc(zone, feats + list(keep)))


def program_zone_names(site):
    """{program: [zone names, mowing order]}, None = file missing or unreadable."""
    if not os.path.exists(bundle.programs_path(site)):
        return None
    try:
        return {str(p.get("name")): list(p.get("zone_names") or [])
                for p in bundle.load_programs(site) if isinstance(p, dict)}
    except Exception:
        return None


def zone_back_in_order(saved, current, zone):
    """`current` with `zone` where `saved` had it: after the nearest earlier zone
    still there. Equals `saved` if the program has not changed since."""
    at = 0
    for name in saved[:saved.index(zone)] if zone in saved else []:
        if name in current:
            at = current.index(name) + 1
    return current[:at] + [zone] + current[at:]


def _from_bundle(family, site):
    """Owner's file -> objects without meta, or None = no information."""
    path = {"zone": bundle.zones_path, "waypoint": bundle.waypoints_path,
            "path": bundle.paths_path, "edit": bundle.edits_path}[family](site)
    if not os.path.exists(path):
        return None
    out = []
    try:
        if family == "waypoint":
            for w in bundle.load_waypoints(site):
                o = _blank(family, "waypoint", w["name"], [[w["e"], w["n"]]])
                o["yaw"] = w["yaw_rad"]
                if w.get("z") is not None:
                    o["props"] = {"z": w["z"]}
                out.append(o)
        elif family == "path":
            for p in bundle.load_paths(site):
                o = _blank(family, "path", p["name"], [list(v) for v in p["vertices"]])
                o["props"] = {"yaws": list(p.get("yaws") or [])}
                out.append(o)
        elif family == "zone":
            for z in bundle.load_zones(site):
                o = _blank(family, "zone", z["name"], [list(v) for v in z["polygon"]])
                o["props"] = dict(z.get("props") or {})
                out.append(o)
        else:
            for e in bundle.load_edits(site):
                kind = e.get("kind") or "obstacle"
                if kind not in ("obstacle", "free", "wall"):
                    continue
                o = _blank(family, kind, e.get("name"), [list(v) for v in e["vertices"]])
                if kind == "wall":
                    o["props"] = {"width_m": e.get("width_m", 0.10)}
                out.append(o)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return None
    return [o for o in out if o["name"]]


def _dress(o, meta, now):
    """Add what is computed or kept beside the owner's file."""
    if o["family"] != "place":
        m = meta.get(meta_key(o["family"], o["name"])) or {}
        for k in META_FIELDS:
            if m.get(k) is not None:
                o[k] = m[k]
    if not o.get("id"):
        o["id"] = default_id(o["family"], o["name"])
    o["protected"] = is_protected(o["family"], o["name"])
    o["active"] = is_active(o, now)
    return o


def load_family(site, family, now=None, meta=None):
    """[objects in UTM] of one family, None = file missing or unreadable."""
    if meta is None:
        meta = load_meta(site) or {}
    t = time.time() if now is None else now
    items = load_places(site) if family == "place" else _from_bundle(family, site)
    return None if items is None else [_dress(o, meta, t) for o in items]


def load_families(site, now=None, meta=None):
    """{family: load_family()}."""
    if meta is None:
        meta = load_meta(site) or {}
    return {fam: load_family(site, fam, now, meta) for fam in FAMILIES}


def to_frame(obj, datum, frame):
    """Copy of an object converted between 'utm' (files) and 'map' (topics)."""
    o = dict(obj)
    o["props"] = dict(obj.get("props") or {})
    if frame == "map":
        pt, yw = geo.utm_to_map, geo.utm_yaw_to_map
    else:
        pt, yw = geo.map_to_utm, geo.map_yaw_to_utm
    o["points"] = [list(pt(float(p[0]), float(p[1]), datum)) for p in obj.get("points", [])]
    if o.get("yaw") is not None:
        o["yaw"] = yw(float(o["yaw"]), datum)
    if o["props"].get("yaws"):
        o["props"]["yaws"] = [yw(float(y), datum) for y in o["props"]["yaws"]]
    return o


def load_all(site, now=None, frame="map"):
    """Every object of the site, all families. Map frame needs the datum: no
    datum -> [] (frame='utm' works without one)."""
    fams = load_families(site, now)
    objs = [o for f in FAMILIES for o in (fams[f] or [])]
    if frame == "utm":
        return objs
    datum = load_datum(site)
    if datum is None:
        return []
    return [to_frame(o, datum, "map") for o in objs]


def active_rules(site, now=None):
    """Active rules of the site in the map frame, sorted by id:
    [{id, type, name, points, speed_mps}]. Missing file or datum -> []."""
    return load_rules(site, now) or []


def load_rules(site, now=None, problems=None):
    """active_rules(), but None when they cannot be told at all: places file
    missing or unreadable as a whole, or no datum. Who enforces rules keeps the
    last ones for that site then; [] means there really are none.

    A bad rule never costs the others, and errs on the safe side. It is reported
    in `problems` as {id, name, reason}.
      slow with a bad speed       stays, default speed
      bad validity bound          that bound limits nothing, the rule stays on
      nogo or noblade that cannot be read, or a feature of unknown type:
                                  stays, with its points if readable (else []) and
                                  "bad": reason, which the guard takes as a nogo
                                  it cannot place; such a rule is never stamped
      slow, or a non-rule object, that cannot be read: reported only"""
    bad = []
    places = load_places(site, bad)
    datum = load_datum(site)
    if places is None or datum is None:
        return None
    t = time.time() if now is None else now
    report = (lambda rid, name, why: problems.append(
        {"id": rid, "name": name, "reason": why})) if problems is not None \
        else (lambda rid, name, why: None)
    out = []
    for o in places:
        if o["type"] not in RULE_TYPES:
            continue
        rid = o.get("id") or default_id("place", o["name"])
        if any(v is not None and not _number(v) for v in (o["valid_from"], o["valid_until"])):
            report(rid, o["name"], "bad_validity")
        if not is_active(o, t):
            continue
        speed = None
        if o["type"] == "slow":
            speed = (o.get("props") or {}).get("speed_mps")
            if not _number(speed) or not SPEED_MPS[0] <= speed <= SPEED_MPS[1]:
                report(rid, o["name"], "bad_speed")
                speed = DEFAULT_SLOW_MPS
            speed = float(speed)
        out.append({"id": rid, "type": o["type"], "name": o["name"],
                    "points": to_frame(o, datum, "map")["points"], "speed_mps": speed})
    for i, b in enumerate(bad):
        rid = b["id"] or default_id("place", b["name"] or "bad_%d" % i)
        name = b["name"] or rid
        report(rid, name, b["reason"])
        if b["type"] in _FAMILY_OF and b["type"] not in ("nogo", "noblade"):
            continue
        pts = [list(geo.utm_to_map(p[0], p[1], datum)) for p in b["points"]]
        out.append({"id": rid, "type": b["type"] if b["type"] == "noblade" else "nogo",
                    "name": name, "points": pts, "speed_mps": None, "bad": b["reason"]})
    return sorted(out, key=lambda r: r["id"])


def rules_signature(rules):
    """Comparable form of a rule list: equal <=> nothing to do for a consumer."""
    return tuple((r["id"], r["type"], r["name"], r.get("speed_mps"), r.get("bad"),
                  tuple((round(p[0], 3), round(p[1], 3)) for p in r["points"]))
                 for r in rules)


def stampable(rules):
    """The nogo rules that have an outline to put in a map."""
    return [r for r in rules if r["type"] == "nogo" and len(r["points"]) >= 3]


def nogo_stamp_edits(rules):
    """Active nogo rules as editmask edits (map frame), obstacle stamps."""
    return [{"name": r["name"], "kind": "obstacle", "vertices": r["points"]}
            for r in stampable(rules)]


def find_by_id(objects, obj_id):
    for o in objects:
        if o.get("id") == obj_id:
            return o
    return None


def find_id(site, name, family=None, objects=None):
    """Id of the object called `name`: exact name, then alias, then a tag or
    category that only one object has. Case-insensitive, None if not found."""
    objs = load_all(site, frame="utm") if objects is None else objects
    if family:
        objs = [o for o in objs if o["family"] == family]
    want = str(name).strip().casefold()
    for o in objs:
        if str(o.get("name")).casefold() == want:
            return o["id"]
    for o in objs:
        if any(str(a).casefold() == want for a in o.get("aliases") or []):
            return o["id"]
    hits = [o for o in objs if want in [str(t).casefold() for t in o.get("tags") or []]
            or str((o.get("props") or {}).get("category", "")).casefold() == want]
    return hits[0]["id"] if len(hits) == 1 else None


# --------------------------------------------------------------------------
# geometry (pure; the outlines are short)
# --------------------------------------------------------------------------
def polygon_area(pts):
    s = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _segments_touch(a, b, c, d):
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 and o2 and o3 and o4:
        return True
    return ((o1 == 0 and _on_segment(a, b, c)) or (o2 == 0 and _on_segment(a, b, d))
            or (o3 == 0 and _on_segment(c, d, a)) or (o4 == 0 and _on_segment(c, d, b)))


def self_crossing(pts):
    """True if the closed outline touches or crosses itself."""
    r = [tuple(p) for i, p in enumerate(pts) if i == 0 or tuple(p) != tuple(pts[i - 1])]
    if len(r) >= 2 and r[0] == r[-1]:
        r = r[:-1]
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_touch(r[i], r[(i + 1) % n], r[j], r[(j + 1) % n]):
                return True
    return False


def point_in_polygon(p, poly):
    """Boundary counts as inside."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if _orient(a, b, p) == 0 and _on_segment(a, b, p):
            return True
        if (a[1] > y) != (b[1] > y):
            if x < a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
                inside = not inside
    return inside


def polygons_overlap(p, q):
    if any(point_in_polygon(v, q) for v in p) or any(point_in_polygon(v, p) for v in q):
        return True
    for i in range(len(p)):
        for j in range(len(q)):
            if _segments_touch(p[i], p[(i + 1) % len(p)], q[j], q[(j + 1) % len(q)]):
                return True
    return False


def _distance_to_outline(p, poly):
    best = float("inf")
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        span = dx * dx + dy * dy
        t = 0.0 if span == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx
                                                     + (p[1] - a[1]) * dy) / span))
        best = min(best, math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy))
    return best


def _crossing_params(a, b, c, d):
    """Where on a-b the segment c-d crosses it, as t in (0, 1), or None."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    denom = rx * sy - ry * sx
    if denom == 0:
        return None
    t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / denom
    u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / denom
    return t if 0.0 < t < 1.0 and 0.0 <= u <= 1.0 else None


def polygon_covers(new, old, tol=POINT_TOL_M):
    """True if no part of `old` lies outside `new`. Simple polygons have no holes,
    so it is enough that the outline of `old` stays inside: every edge is cut
    where it crosses `new` and each piece checked, exact also for a concave `new`."""
    def inside(p):
        return point_in_polygon(p, new) or _distance_to_outline(p, new) <= tol
    if len(new) < 3:
        return False
    for i in range(len(old)):
        a, b = old[i], old[(i + 1) % len(old)]
        cuts = [0.0, 1.0]
        for j in range(len(new)):
            t = _crossing_params(a, b, new[j], new[(j + 1) % len(new)])
            if t is not None:
                cuts.append(t)
        cuts.sort()
        for t0, t1 in zip(cuts, cuts[1:]):
            for t in (t0, (t0 + t1) / 2.0):
                if not inside((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                    return False
    return True


def owner_nogo_problem(before, after, now=None):
    """Why an agent may not turn `before` into `after` (None = deleted), as a code
    with the rule's name, or None. Protected is a nogo not made by the agent: it
    may be enlarged and extended, never removed, cut down or switched off. Both
    objects in the same frame. Same rule as the agent's own first check."""
    if not before or before.get("type") != "nogo" or before.get("source") == "agent":
        return None
    name = before.get("name")
    if after is None:
        return "owner_nogo_delete:%s" % name
    if after.get("type") != "nogo":
        return "owner_nogo_type:%s" % name
    if not polygon_covers([tuple(p) for p in after["points"]],
                          [tuple(p) for p in before["points"]]):
        return "owner_nogo_shrink:%s" % name
    vf0, vf1 = before.get("valid_from"), after.get("valid_from")
    vu0, vu1 = before.get("valid_until"), after.get("valid_until")
    later = vf1 is not None and (vf0 is None or vf1 > vf0)
    earlier = vu1 is not None and (vu0 is None or vu1 < vu0)
    if later or earlier or (is_active(before, now) and not is_active(after, now)):
        return "owner_nogo_validity:%s" % name
    return None


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _bad(reason, warnings=None):
    return {"ok": False, "reason": reason, "warnings": warnings or []}


def _check_name(obj, peers):
    """Name rules of objnames against the same family; moving a protected object
    keeps its reserved name, renaming it does not."""
    name = obj.get("name")
    me = next((o for o in peers if obj.get("id") and o.get("id") == obj.get("id")), None)
    if me is not None and me.get("name") == name:
        return None
    others = [o.get("name") for o in peers if o is not me]
    if me is not None and is_protected(me["family"], me.get("name")):
        return "reserved"
    return objnames.check_name(name, others)


def _check_shape(obj):
    geom = _GEOMETRY_OF[obj["type"]]
    pts = obj.get("points") or []
    need = {"point": 1, "line": 2, "area": 3}[geom]
    if len(pts) < need:
        return "too_few_points"
    if geom == "point" and len(pts) != 1:
        return "bad_points"
    try:
        if not all(len(p) == 2 and _number(p[0]) and _number(p[1]) for p in pts):
            return "bad_points"
    except TypeError:
        return "bad_points"
    if geom == "area":
        if self_crossing(pts):       # first: a bowtie's signed area cancels out
            return "self_crossing"
        if polygon_area(pts) < MIN_AREA_M2:
            return "too_small"
    return None


def _check_props(obj):
    t, pr = obj["type"], obj.get("props") or {}
    if t == "slow" and not (_number(pr.get("speed_mps"))
                            and SPEED_MPS[0] <= pr["speed_mps"] <= SPEED_MPS[1]):
        return "bad_speed"
    if t in ("corridor", "wall") and not (_number(pr.get("width_m")) and pr["width_m"] > 0):
        return "bad_width"
    if t == "corridor" and not isinstance(pr.get("connects", []), list):
        return "bad_connects"
    if t == "zone" and "cut_height" in pr and not (
            _number(pr["cut_height"])
            and CUT_HEIGHT_MM[0] <= pr["cut_height"] <= CUT_HEIGHT_MM[1]):
        return "bad_cut_height"
    vf, vu = obj.get("valid_from"), obj.get("valid_until")
    if vf is not None or vu is not None:
        if t not in RULE_TYPES:
            return "not_temporary"
        if (vf is not None and not _number(vf)) or (vu is not None and not _number(vu)):
            return "bad_validity"
        if vf is not None and vu is not None and vf >= vu:
            return "bad_validity"
    return None


def validate(obj, site=None, objects=None):
    """Check an object in the MAP frame. {"ok", "reason", "warnings"}; reason is a
    code. `objects` = the site's objects in the map frame (default: load_all).
    Without a site only the object itself is checked. A nogo cannot be checked
    against the dock without a datum, so it is refused then."""
    t = obj.get("type")
    fam = _FAMILY_OF.get(t)
    if fam is None:
        return _bad("bad_type")
    if obj.get("family") not in (None, fam):
        return _bad("bad_family")
    reason = _check_shape(obj) or _check_props(obj)
    if reason:
        return _bad(reason)
    warnings = []
    if site is None and objects is None:
        return {"ok": True, "reason": None, "warnings": warnings}
    if objects is None:
        objects = load_all(site)
        if t == "nogo" and load_datum(site) is None:
            return _bad("no_datum")
    reason = _check_name(obj, [o for o in objects if o["family"] == fam])
    if reason:
        return _bad(reason)
    pts = [tuple(p) for p in obj["points"]]
    if t == "nogo":
        for o in objects:
            if (o["family"] == "waypoint" and is_protected("waypoint", o["name"])
                    and point_in_polygon(tuple(o["points"][0]), pts)):
                return _bad("nogo_contains_protected:" + o["name"])
        for o in objects:
            if o["family"] == "zone" and polygons_overlap(pts, [tuple(p) for p in o["points"]]):
                warnings.append("nogo_overlaps_zone:" + o["name"])
    elif fam == "waypoint" and is_protected(fam, obj.get("name")):
        # Moving stays allowed, but it should not pass unnoticed.
        for o in objects:
            if o["type"] == "nogo" and o.get("active") and point_in_polygon(
                    pts[0], [tuple(p) for p in o["points"]]):
                warnings.append("protected_in_nogo:" + o["name"])
    return {"ok": True, "reason": None, "warnings": warnings}


# --------------------------------------------------------------------------
# comparing
# --------------------------------------------------------------------------
def _close_pts(a, b):
    return len(a) == len(b) and all(
        abs(p[0] - q[0]) <= POINT_TOL_M and abs(p[1] - q[1]) <= POINT_TOL_M
        for p, q in zip(a, b))


def _close_yaw(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(geo.wrap_pi(float(a) - float(b))) <= YAW_TOL_RAD


def _close_value(a, b):
    if _number(a) and _number(b):
        return abs(a - b) <= 1e-6
    return a == b


def same_content(a, b):
    """Same object for the watcher: same shape and everything a person set, with
    the round-trip noise and the owners' computed props ignored. Name not compared."""
    if a.get("type") != b.get("type") or not _close_pts(a["points"], b["points"]):
        return False
    if not _close_yaw(a.get("yaw"), b.get("yaw")):
        return False
    skip = COMPUTED_PROPS.get(a.get("family"), ()) + ("yaws",)
    pa, pb = a.get("props") or {}, b.get("props") or {}
    for k in set(pa) | set(pb):
        if k not in skip and not _close_value(pa.get(k), pb.get(k)):
            return False
    ya, yb = pa.get("yaws"), pb.get("yaws")
    if ya and yb:
        if len(ya) != len(yb) or not all(_close_yaw(x, y) for x, y in zip(ya, yb)):
            return False
    if a.get("family") == "place":
        for k in ("aliases", "note", "tags", "valid_from", "valid_until"):
            if a.get(k) != b.get(k):
                return False
    return True


def _diff_key(o):
    """docked -> DOCKED (migration) is the same object, not a delete + create."""
    if is_protected(o.get("family"), o.get("name")):
        return o["name"].casefold()
    return o["name"]


def diff_family(old, new):
    """Changes between two lists of one family: [(before, after)], before None =
    created, after None = deleted. A removed and an added object of the same
    content are one rename."""
    old_by = {_diff_key(o): o for o in old}
    new_by = {_diff_key(o): o for o in new}
    out = []
    for key, n in new_by.items():
        o = old_by.get(key)
        if o is not None and not same_content(o, n):
            out.append((o, n))
    gone = [o for key, o in old_by.items() if key not in new_by]
    fresh = [n for key, n in new_by.items() if key not in old_by]
    for n in fresh:
        o = next((g for g in gone if same_content(g, n)), None)
        if o is not None:
            gone.remove(o)
        out.append((o, n))
    out.extend((o, None) for o in gone)
    return out
