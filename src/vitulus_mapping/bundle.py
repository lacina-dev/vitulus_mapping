#!/usr/bin/env python3
"""
bundle.py -- site-bundle read/write for the mapping-v3 unification.

PURE PYTHON (yaml + json + math + os only). NO rospy / numpy / ROS -- navi_man
and node_planner import this at load time, and offline tools use it too.

Canonical site of truth per garden:  ~/.vitulus/mapping_v3/<site>/
    manifest.yaml       site, version, utm_zone, navi_map link, created/updated
    waypoints.geojson   Point   [utm_e, utm_n] ; props name, yaw_rad, z?
    paths.geojson       LineString vertices [utm_e, utm_n] ; props name, yaws[]
    zones.geojson       Polygon rings [utm_e, utm_n] ; props name + zone props
    programs.yaml        list of program dicts referencing zone names
    edits.geojson        human map edits (WP-D1): Polygon obstacle/free +
                         LineString wall(width_m); composited over rasters

GEOMETRY IS UTM METRES (EPSG:326<zone>), portable across maps/datums and
QGIS-friendly.  GeoJSON is RFC 7946-shaped (FeatureCollection/Feature/geometry)
but coordinates are UTM eastings/northings, so each file declares a legacy
"crs" member (urn:ogc:def:crs:EPSG::326NN) -- QGIS honours it -- plus a
top-level "note" property spelling out that coordinates are UTM metres.

All writes are ATOMIC (write tmp in the same dir, then os.replace).

The API below is FROZEN -- WP-B (navi_man) and WP-C (node_planner) code against
these signatures.  Data shapes:
    waypoint : {name:str, e:float, n:float, yaw_rad:float, z:float(optional)}
    path     : {name:str, vertices:[(e,n),...], yaws:[float,...]}
    zone     : {name:str, polygon:[(e,n),...], props:{...}}
    program  : arbitrary dict per plan (name, zone_names[], speed, rpm, ...)
"""

import json
import math
import os
import time

import yaml

__all__ = [
    "SITES_ROOT", "site_dir", "site_exists", "list_sites",
    "manifest_path", "waypoints_path", "paths_path", "zones_path",
    "programs_path", "bundle_files",
    "load_manifest", "save_manifest",
    "load_waypoints", "save_waypoints",
    "load_paths", "save_paths",
    "load_zones", "save_zones",
    "load_edits", "save_edits", "edits_path", "edits_utm_zone",
    "load_programs", "save_programs",
    "epsg_for_zone", "crs_member",
    "newest_mtime", "file_mtime", "any_mtime",
]

SITES_ROOT = os.path.expanduser("~/.vitulus/mapping_v3")

_GEOJSON_NOTE = ("Coordinates are UTM metres (easting, northing), NOT lon/lat. "
                 "See crs member and the site manifest utm_zone.")

# file-kind -> basename
_FILES = {
    "manifest": "manifest.yaml",
    "waypoints": "waypoints.geojson",
    "paths": "paths.geojson",
    "zones": "zones.geojson",
    "edits": "edits.geojson",
    "programs": "programs.yaml",
}


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def site_dir(site):
    return os.path.join(SITES_ROOT, site)


def site_exists(site):
    return os.path.isdir(site_dir(site))


def list_sites():
    if not os.path.isdir(SITES_ROOT):
        return []
    return sorted(d for d in os.listdir(SITES_ROOT)
                  if os.path.isdir(os.path.join(SITES_ROOT, d)))


def _p(site, kind):
    return os.path.join(site_dir(site), _FILES[kind])


def manifest_path(site):
    return _p(site, "manifest")


def waypoints_path(site):
    return _p(site, "waypoints")


def paths_path(site):
    return _p(site, "paths")


def zones_path(site):
    return _p(site, "zones")


def edits_path(site):
    return _p(site, "edits")


def programs_path(site):
    return _p(site, "programs")


def bundle_files(site):
    """dict kind -> absolute path for every bundle file kind."""
    return {k: _p(site, k) for k in _FILES}


# --------------------------------------------------------------------------
# crs / epsg
# --------------------------------------------------------------------------
def epsg_for_zone(utm_zone):
    """Northern-hemisphere UTM EPSG code for a zone number (326NN)."""
    return 32600 + int(utm_zone)


def crs_member(utm_zone):
    """Legacy GeoJSON named-CRS member (QGIS-friendly)."""
    return {"type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::%d"
                                    % epsg_for_zone(utm_zone)}}


# --------------------------------------------------------------------------
# atomic write helpers
# --------------------------------------------------------------------------
def _atomic_write_text(path, text):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _write_yaml(path, obj):
    _atomic_write_text(path, yaml.safe_dump(obj, default_flow_style=False,
                                            sort_keys=False, allow_unicode=True))


def _write_json(path, obj):
    _atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def load_manifest(site):
    """Return the manifest dict, or None if it does not exist."""
    p = manifest_path(site)
    if not os.path.exists(p):
        return None
    return _load_yaml(p)


def save_manifest(site, manifest):
    """Write manifest.yaml (atomic).  Stamps 'updated'; sets 'created' if
    absent.  Returns the written dict."""
    m = dict(manifest)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    m.setdefault("site", site)
    m.setdefault("version", 1)
    m.setdefault("created", now)
    m["updated"] = now
    _write_yaml(manifest_path(site), m)
    return m


def _utm_zone_for(site, utm_zone):
    if utm_zone is not None:
        return int(utm_zone)
    m = load_manifest(site)
    if m and m.get("utm_zone") is not None:
        return int(m["utm_zone"])
    raise ValueError("utm_zone not given and manifest has none for site %r"
                     % site)


# --------------------------------------------------------------------------
# GeoJSON FeatureCollection scaffold
# --------------------------------------------------------------------------
def _fc(utm_zone, features):
    return {
        "type": "FeatureCollection",
        "note": _GEOJSON_NOTE,
        "crs": crs_member(utm_zone),
        "features": features,
    }


def _feature(geometry, properties):
    return {"type": "Feature", "geometry": geometry, "properties": properties}


# --------------------------------------------------------------------------
# waypoints  (Point)
# --------------------------------------------------------------------------
def load_waypoints(site):
    """Return list of {name, e, n, yaw_rad, z?}.  [] if file absent."""
    p = waypoints_path(site)
    if not os.path.exists(p):
        return []
    fc = _load_json(p)
    out = []
    for feat in fc.get("features", []):
        coords = feat["geometry"]["coordinates"]
        props = feat.get("properties", {}) or {}
        wp = {"name": props.get("name"),
              "e": float(coords[0]), "n": float(coords[1]),
              "yaw_rad": float(props.get("yaw_rad", 0.0))}
        if len(coords) > 2 and coords[2] is not None:
            wp["z"] = float(coords[2])
        elif props.get("z") is not None:
            wp["z"] = float(props["z"])
        out.append(wp)
    return out


def save_waypoints(site, waypoints, utm_zone=None):
    """waypoints: iterable of {name, e, n, yaw_rad, z?}."""
    zone = _utm_zone_for(site, utm_zone)
    feats = []
    for wp in waypoints:
        z = wp.get("z")
        coords = [float(wp["e"]), float(wp["n"])]
        if z is not None:
            coords.append(float(z))
        props = {"name": wp.get("name"),
                 "yaw_rad": float(wp.get("yaw_rad", 0.0))}
        if z is not None:
            props["z"] = float(z)
        feats.append(_feature({"type": "Point", "coordinates": coords}, props))
    _write_json(waypoints_path(site), _fc(zone, feats))


# --------------------------------------------------------------------------
# paths  (LineString)
# --------------------------------------------------------------------------
def load_paths(site):
    """Return list of {name, vertices:[(e,n),...], yaws:[...]}.  [] if absent."""
    p = paths_path(site)
    if not os.path.exists(p):
        return []
    fc = _load_json(p)
    out = []
    for feat in fc.get("features", []):
        coords = feat["geometry"]["coordinates"]
        props = feat.get("properties", {}) or {}
        verts = [(float(c[0]), float(c[1])) for c in coords]
        yaws = [float(y) for y in (props.get("yaws") or [])]
        out.append({"name": props.get("name"),
                    "vertices": verts, "yaws": yaws})
    return out


def save_paths(site, paths, utm_zone=None):
    """paths: iterable of {name, vertices:[(e,n),...], yaws:[...]}."""
    zone = _utm_zone_for(site, utm_zone)
    feats = []
    for pa in paths:
        coords = [[float(v[0]), float(v[1])] for v in pa["vertices"]]
        props = {"name": pa.get("name"),
                 "yaws": [float(y) for y in (pa.get("yaws") or [])]}
        feats.append(_feature({"type": "LineString", "coordinates": coords},
                              props))
    _write_json(paths_path(site), _fc(zone, feats))


# --------------------------------------------------------------------------
# zones  (Polygon)
# --------------------------------------------------------------------------
def load_zones(site):
    """Return list of {name, polygon:[(e,n),...], props:{...}}.  [] if absent.

    The stored ring is closed (RFC 7946); the returned 'polygon' drops the
    duplicated closing vertex so it matches the click order used upstream."""
    p = zones_path(site)
    if not os.path.exists(p):
        return []
    fc = _load_json(p)
    out = []
    for feat in fc.get("features", []):
        rings = feat["geometry"]["coordinates"]
        ring = rings[0] if rings else []
        poly = [(float(c[0]), float(c[1])) for c in ring]
        if len(poly) >= 2 and poly[0] == poly[-1]:
            poly = poly[:-1]
        props = dict(feat.get("properties", {}) or {})
        name = props.pop("name", None)
        out.append({"name": name, "polygon": poly, "props": props})
    return out


def save_zones(site, zones, utm_zone=None):
    """zones: iterable of {name, polygon:[(e,n),...], props:{...}}.  The polygon
    ring is closed automatically for RFC 7946 validity."""
    zone = _utm_zone_for(site, utm_zone)
    feats = []
    for z in zones:
        ring = [[float(v[0]), float(v[1])] for v in z["polygon"]]
        if len(ring) >= 3 and ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        props = {"name": z.get("name")}
        props.update(z.get("props", {}) or {})
        feats.append(_feature({"type": "Polygon", "coordinates": [ring]},
                              props))
    _write_json(zones_path(site), _fc(zone, feats))


# --------------------------------------------------------------------------
# edits  (WP-D1: Polygon obstacle/free + LineString wall) -- human map
# corrections composited over every raster regeneration.  Geometry is UTM
# metres like every other bundle file.  List order == compositing order.
# --------------------------------------------------------------------------
def load_edits(site):
    """Return list of {name, kind:'obstacle'|'free'|'wall',
    vertices:[(e,n),...], width_m (walls only)}.  [] if file absent.

    obstacle/free are stored as (closed) Polygon rings -- the returned
    'vertices' drops the duplicated closing vertex to match click order;
    wall is a LineString whose 'vertices' are the raw polyline points and
    whose width lives in the 'width_m' property."""
    p = edits_path(site)
    if not os.path.exists(p):
        return []
    fc = _load_json(p)
    out = []
    for feat in fc.get("features", []):
        geom = feat.get("geometry", {}) or {}
        gtype = geom.get("type")
        props = dict(feat.get("properties", {}) or {})
        name = props.get("name")
        kind = props.get("kind")
        if gtype == "Polygon":
            rings = geom.get("coordinates") or []
            ring = rings[0] if rings else []
            verts = [(float(c[0]), float(c[1])) for c in ring]
            if len(verts) >= 2 and verts[0] == verts[-1]:
                verts = verts[:-1]
            kind = kind or "obstacle"
            out.append({"name": name, "kind": kind, "vertices": verts})
        elif gtype == "LineString":
            coords = geom.get("coordinates") or []
            verts = [(float(c[0]), float(c[1])) for c in coords]
            out.append({"name": name, "kind": kind or "wall",
                        "vertices": verts,
                        "width_m": float(props.get("width_m", 0.10))})
    return out


def edits_utm_zone(site):
    """Best-effort UTM zone number parsed from an existing edits.geojson crs
    member (EPSG:326NN -> NN), or None if absent/unparseable. Lets a rewrite
    (e.g. removing one edit) preserve the file's zone when no datum is handy."""
    p = edits_path(site)
    if not os.path.exists(p):
        return None
    try:
        fc = _load_json(p)
        name = fc.get("crs", {}).get("properties", {}).get("name", "")
        epsg = int(name.rsplit(":", 1)[-1])
        if 32601 <= epsg <= 32660:
            return epsg - 32600
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def save_edits(site, edits, utm_zone=None):
    """edits: iterable of {name, kind, vertices:[(e,n),...] UTM, width_m?}.
    obstacle/free -> Polygon (ring auto-closed); wall -> LineString with a
    width_m property.  List order is preserved (== compositing order).
    Atomic write, same crs/note scaffold as save_zones."""
    zone = _utm_zone_for(site, utm_zone)
    feats = []
    for e in edits:
        kind = e.get("kind", "obstacle")
        verts = [[float(v[0]), float(v[1])] for v in e.get("vertices", [])]
        props = {"name": e.get("name"), "kind": kind}
        if kind == "wall":
            props["width_m"] = float(e.get("width_m", 0.10))
            geom = {"type": "LineString", "coordinates": verts}
        else:
            ring = list(verts)
            if len(ring) >= 3 and ring[0] != ring[-1]:
                ring = ring + [ring[0]]
            geom = {"type": "Polygon", "coordinates": [ring]}
        feats.append(_feature(geom, props))
    _write_json(edits_path(site), _fc(zone, feats))


# --------------------------------------------------------------------------
# programs  (plain yaml list)
# --------------------------------------------------------------------------
def load_programs(site):
    """Return list of program dicts.  [] if file absent."""
    p = programs_path(site)
    if not os.path.exists(p):
        return []
    data = _load_yaml(p)
    if data is None:
        return []
    if isinstance(data, dict) and "programs" in data:
        return list(data["programs"])
    return list(data)


def save_programs(site, programs):
    """programs: iterable of dicts.  Stored under a top-level 'programs' key
    with a 'note' so the yaml is self-describing."""
    doc = {"note": "Site mowing programs; geometry lives in zones.geojson "
                   "(referenced by zone_names).",
           "programs": list(programs)}
    _write_yaml(programs_path(site), doc)


# --------------------------------------------------------------------------
# mtime helpers  (for import-if-newer downstream)
# --------------------------------------------------------------------------
def file_mtime(site, kind):
    """mtime (float epoch) of one bundle file kind, or None if absent."""
    p = _p(site, kind)
    return os.path.getmtime(p) if os.path.exists(p) else None


def newest_mtime(site, kinds=None):
    """Newest mtime across the given bundle file kinds (default: all), or None
    if none exist.  kinds is a str or iterable of kind names."""
    if kinds is None:
        kinds = list(_FILES)
    elif isinstance(kinds, str):
        kinds = [kinds]
    times = [t for t in (file_mtime(site, k) for k in kinds) if t is not None]
    return max(times) if times else None


def any_mtime(site):
    """Alias for newest_mtime over all kinds (manifest-or-any newest)."""
    return newest_mtime(site)
