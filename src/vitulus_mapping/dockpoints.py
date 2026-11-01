"""The dock points saved when a map recording ends in the dock.

DOCKED is the pose of the seated robot. The robot reverses in, so its heading
there points out of the dock. DOCK and UNDOCK go where the intensity undock
ends: straight ahead of DOCKED, facing the same way.

The controller aims at `undock_distance` from the reflector and the seated base
is `charge_point_distance` from it (docking_controller, navigation_image), so the
robot travels the difference, 1.02 m with the shipped config. The values are read
from what the controller itself loads, never copied here.
"""

import math

import yaml

from vitulus_mapping import objnames

PARAM_NS = '/docking_controller/docking/'


def drive_out_m(undock_distance, charge_point_distance):
    """Metres from DOCKED to the end of the undock, None if that makes no sense."""
    try:
        out = float(undock_distance) - float(charge_point_distance)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def drive_out_m_from_params(get_param):
    """From the running controller's parameters. get_param(name, default)."""
    return drive_out_m(get_param(PARAM_NS + 'undock_distance', None),
                       get_param(PARAM_NS + 'charge_point_distance', None))


def drive_out_m_from_yaml(path):
    """From vitulus_dock/config/docking_controller.yaml, None if unreadable."""
    try:
        with open(path) as f:
            docking = yaml.safe_load(f)['docking']
        return drive_out_m(docking.get('undock_distance'),
                           docking.get('charge_point_distance'))
    except (OSError, TypeError, KeyError, AttributeError, yaml.YAMLError):
        return None


def points_to_save(existing, x, y, yaw, out_m):
    """save_waypoint_at payloads for a stop in the dock, DOCKED first.

    existing: names on the site, None = not known. DOCK and UNDOCK are added only
    when known to be missing, so a point the owner moved stays where it is.
    out_m: drive_out_m, None = DOCKED only."""
    points = [{'name': 'DOCKED', 'x': x, 'y': y, 'yaw': yaw}]
    if existing is None or out_m is None:
        return points
    have = {objnames.canonical_name(n) for n in existing}
    outside = {'x': x + out_m * math.cos(yaw), 'y': y + out_m * math.sin(yaw), 'yaw': yaw}
    for name in ('UNDOCK', 'DOCK'):
        if name not in have:
            points.append(dict(outside, name=name))
    return points
