#!/usr/bin/env python3
"""
geo.py -- map-frame <-> UTM affine conversions for the site-bundle world.

PURE PYTHON. No rospy / numpy / ROS imports (navi_man and node_planner import
this at module load, and offline tools/QGIS helpers use it too). Only ``math``.

=====================================================================
CONVENTION / DERIVATION  (do not change without re-deriving from source)
=====================================================================
A "datum" describes where the robot MAP frame origin sits in UTM and how the
map axes are rotated relative to UTM.  The canonical relation is:

        p_utm = R(yaw) . p_map + (utm_e, utm_n)

with the STANDARD 2D rotation matrix (CCW positive):

        e = utm_e + cos(yaw)*x - sin(yaw)*y
        n = utm_n + sin(yaw)*x + cos(yaw)*y

Inverse:
        dx = e - utm_e ;  dy = n - utm_n
        x =  cos(yaw)*dx + sin(yaw)*dy
        y = -sin(yaw)*dx + cos(yaw)*dy

WHY this exact convention (two independent sources agree):

1) vitulus_mapping/nodes/insertion_gate :: _sample_utm_map()  (lines ~413-436)
   Builds the site georef datum as
        m_um = to_mat(utm<-odom) @ inv(to_mat(map<-odom))  ==  utm <- map
   then stores  (e, n, alt) = m_um[0:3,3]  and  yaw = euler_from_matrix(m_um)[2]
   into datum.yaml as utm_e / utm_n / alt / yaw_rad.
   => datum.yaml's (utm_e, utm_n, yaw_rad) IS the homogeneous transform that
      maps a MAP-frame point into UTM:  p_utm = R(yaw).p_map + t.  yaw is the
      map frame's rotation expressed in the UTM frame (ENU-like, CCW positive).

2) vitulus_navi/nodes/navi_man :: save_map_utm_pose()  (lines ~1116-1127)
   Stores  lookup_transform('utm','map')  ->  utm_x/utm_y/utm_z (translation)
   and utm_orientation_{x,y,z,w} (rotation) into the legacy MapData pickle.
   A TF 'utm'->'map' (parent utm, child map) is exactly the same utm<-map
   transform: it takes points expressed in child(map) to parent(utm).
   navi_man :: set_map_utm_transform() (lines ~1063-1104) then reuses that same
   quaternion as the map->odom rotation and (utm_odom - utm_xyz) as the
   translation, confirming utm_x/y/z is the UTM position of the map origin and
   the quaternion is the map-frame orientation relative to UTM.

Therefore the pickle datum (utm_x, utm_y + quaternion) and the datum.yaml
(utm_e, utm_n + yaw_rad) describe the SAME thing; yaw_rad = the yaw (Z) euler
component of that quaternion.  Both feed map_to_utm/utm_to_map identically.

For a quaternion (x,y,z,w):
        yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
(reduces to yaw = 2*atan2(z, w) for a pure-Z rotation, which these datums are).

A heading (yaw) that is expressed in the MAP frame converts to UTM by simply
adding the datum yaw:  yaw_utm = wrap(yaw_map + datum.yaw_rad)  (and inverse
subtracts), because the map axes are rotated by datum.yaw_rad within UTM.
"""

import math

__all__ = [
    "quat_to_yaw", "yaw_to_quat",
    "from_datum_yaml", "from_pickle_datum", "normalize_datum",
    "map_to_utm", "utm_to_map",
    "map_yaw_to_utm", "utm_yaw_to_map",
    "wrap_pi", "datum_delta",
]


# --------------------------------------------------------------------------
# small angle / quaternion helpers
# --------------------------------------------------------------------------
def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(x, y, z, w):
    """Yaw (rotation about Z) of a quaternion, radians."""
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw):
    """(x, y, z, w) for a pure yaw (Z) rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# --------------------------------------------------------------------------
# datum normalizers  -> canonical dict {utm_e, utm_n, yaw_rad, alt?, utm_zone?}
# --------------------------------------------------------------------------
def from_datum_yaml(d):
    """Normalize a site datum.yaml dict (schema: utm_e/utm_n/yaw_rad[/alt/
    utm_zone]) to the canonical datum dict used by map_to_utm/utm_to_map."""
    out = {
        "utm_e": float(d["utm_e"]),
        "utm_n": float(d["utm_n"]),
        "yaw_rad": float(d.get("yaw_rad", 0.0)),
    }
    if "alt" in d and d["alt"] is not None:
        out["alt"] = float(d["alt"])
    if "utm_zone" in d and d["utm_zone"] is not None:
        out["utm_zone"] = int(d["utm_zone"])
    return out


def from_pickle_datum(mapdata):
    """Normalize a legacy MapData pickle (navi_man mapdata.MapData) datum
    (utm_x/utm_y/utm_z + utm_orientation_{x,y,z,w}) to the canonical datum
    dict.  Accepts either an object with attributes or a plain dict."""
    def g(name, default=0.0):
        if isinstance(mapdata, dict):
            return mapdata.get(name, default)
        return getattr(mapdata, name, default)

    qx = g("utm_orientation_x", 0.0)
    qy = g("utm_orientation_y", 0.0)
    qz = g("utm_orientation_z", 0.0)
    qw = g("utm_orientation_w", 1.0)
    # a freshly-constructed MapData has w==0 (unset) -> treat as identity
    if qx == 0.0 and qy == 0.0 and qz == 0.0 and qw == 0.0:
        qw = 1.0
    out = {
        "utm_e": float(g("utm_x", 0.0)),
        "utm_n": float(g("utm_y", 0.0)),
        "yaw_rad": quat_to_yaw(qx, qy, qz, qw),
    }
    utm_z = g("utm_z", None)
    if utm_z is not None:
        out["alt"] = float(utm_z)
    return out


def normalize_datum(d):
    """Best-effort: accept a datum.yaml-style dict, a canonical dict, or a
    legacy MapData object/dict and return the canonical datum dict."""
    if not isinstance(d, dict):
        return from_pickle_datum(d)          # MapData object
    if "utm_e" in d:
        return from_datum_yaml(d)
    if "utm_x" in d or "utm_orientation_w" in d:
        return from_pickle_datum(d)
    raise ValueError("normalize_datum: unrecognized datum schema: %r"
                     % (list(d.keys()),))


# --------------------------------------------------------------------------
# the affine conversions
# --------------------------------------------------------------------------
def map_to_utm(x, y, datum):
    """MAP-frame (x, y) metres -> UTM (easting, northing) metres.

    datum: dict with utm_e/utm_n/yaw_rad (canonical) OR any schema accepted by
    normalize_datum (datum.yaml, legacy pickle object/dict)."""
    d = _canon(datum)
    c = math.cos(d["yaw_rad"])
    s = math.sin(d["yaw_rad"])
    e = d["utm_e"] + c * x - s * y
    n = d["utm_n"] + s * x + c * y
    return (e, n)


def utm_to_map(e, n, datum):
    """UTM (easting, northing) metres -> MAP-frame (x, y) metres."""
    d = _canon(datum)
    c = math.cos(d["yaw_rad"])
    s = math.sin(d["yaw_rad"])
    dx = e - d["utm_e"]
    dy = n - d["utm_n"]
    x = c * dx + s * dy
    y = -s * dx + c * dy
    return (x, y)


def map_yaw_to_utm(yaw_map, datum):
    """A heading expressed in the MAP frame -> the same heading in UTM."""
    d = _canon(datum)
    return wrap_pi(yaw_map + d["yaw_rad"])


def utm_yaw_to_map(yaw_utm, datum):
    """A heading expressed in UTM -> the same heading in the MAP frame."""
    d = _canon(datum)
    return wrap_pi(yaw_utm - d["yaw_rad"])


def datum_delta(a, b):
    """Return (dE, dN, dAlt, dYaw) = b - a between two datums (any schema).
    dYaw wrapped to (-pi, pi].  dAlt is None if either lacks alt."""
    da = _canon(a)
    db = _canon(b)
    de = db["utm_e"] - da["utm_e"]
    dn = db["utm_n"] - da["utm_n"]
    dyaw = wrap_pi(db["yaw_rad"] - da["yaw_rad"])
    if "alt" in da and "alt" in db:
        dalt = db["alt"] - da["alt"]
    else:
        dalt = None
    return (de, dn, dalt, dyaw)


def _canon(datum):
    # already canonical?
    if isinstance(datum, dict) and "utm_e" in datum and "yaw_rad" in datum:
        return datum
    return normalize_datum(datum)
