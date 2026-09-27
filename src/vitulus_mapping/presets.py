# -*- coding: utf-8 -*-
"""Direct-raster environment presets, shared by direct_raster (applies/saves
them during a session) and mapping_manager (publishes the list all the time,
also outside a session - vitulus-field#23).

Pure Python, no ROS: importable and testable without a master.
"""
import json
import os

DEFAULT_PATH = '~/.vitulus/mapping_v3/direct_presets.json'

# keys a preset may carry: environment character only - never the source
# enables/ranges, those stay the user's live choice
PRESET_KEYS = ('cam_obstacle_min_h', 'cam_obstacle_max_h',
               'cam_min_cluster_pts', 'cam_cluster_radius_m',
               'lidar_min_intensity', 'hit_inc', 'miss_dec',
               'lidar_free_dec', 'occ_erode_factor',
               'occ_thresh', 'free_thresh', 'min_hits',
               'lo_clamp', 'hi_clamp')

# Environment presets (2026-08-09): named PARTIAL settings dicts applied via
# /mapping/direct_preset. Only the "environment character" keys — source
# enables and ranges are the user's own live choice and are never touched by
# a preset. Custom presets (saved from the UI) persist globally in
# direct_presets.json and shadow a builtin of the same name.
# PRESET REWORK (2026-08-09 field test zahradaDneska): the first-cut presets
# were erosion-broken — 'Lawn tall' had miss_dec 0.5 > hit_inc 0.4 with
# occ_thresh 1.0/min_hits 5, and with the lidar carving free 360° at 6 Hz only
# 96 occupied cells survived a 10-min drive. Rules now: miss_dec <= hit_inc,
# reachable thresholds, and the sceptical knobs are lidar_free_dec +
# occ_erode_factor + min_hits rather than a free/hit imbalance.
BUILTIN_PRESETS = {
    'Lawn short': {
        'cam_obstacle_min_h': 0.12, 'cam_obstacle_max_h': 0.6,
        'cam_min_cluster_pts': 1, 'cam_cluster_radius_m': 0.15,
        'lidar_min_intensity': 0.0,
        'hit_inc': 0.6, 'miss_dec': 0.4, 'occ_thresh': 0.85,
        'free_thresh': -0.4, 'min_hits': 2,
        'lidar_free_dec': 0.15, 'occ_erode_factor': 0.5,
    },
    'Lawn tall': {
        # tall grass: lift the obstacle floor above the blade tips, demand
        # more frames + bigger blobs, drop weak lidar returns (grass);
        # scepticism comes from min_hits/cluster — NOT from miss>hit
        'cam_obstacle_min_h': 0.25, 'cam_obstacle_max_h': 0.8,
        'cam_min_cluster_pts': 4, 'cam_cluster_radius_m': 0.2,
        'lidar_min_intensity': 6.0,
        'hit_inc': 0.5, 'miss_dec': 0.3, 'occ_thresh': 0.8,
        'free_thresh': -0.4, 'min_hits': 4,
        'lidar_free_dec': 0.08, 'occ_erode_factor': 0.3,
    },
    'Paved yard': {
        # hard surfaces, walls/fences matter: converge fast, take tall stuff;
        # lidar is trustworthy here so its free rays may erode a bit more
        'cam_obstacle_min_h': 0.10, 'cam_obstacle_max_h': 1.5,
        'cam_min_cluster_pts': 1, 'cam_cluster_radius_m': 0.15,
        'lidar_min_intensity': 0.0,
        'hit_inc': 0.7, 'miss_dec': 0.4, 'occ_thresh': 0.85,
        'free_thresh': -0.4, 'min_hits': 2,
        'lidar_free_dec': 0.2, 'occ_erode_factor': 0.5,
    },
    'Bushes / edge': {
        # soft moving vegetation: slow accumulation, sticky once confirmed
        'cam_obstacle_min_h': 0.20, 'cam_obstacle_max_h': 1.8,
        'cam_min_cluster_pts': 5, 'cam_cluster_radius_m': 0.2,
        'lidar_min_intensity': 4.0,
        'hit_inc': 0.4, 'miss_dec': 0.3, 'occ_thresh': 0.9,
        'free_thresh': -0.4, 'min_hits': 5,
        'lidar_free_dec': 0.08, 'occ_erode_factor': 0.3,
    },
    'Default (yaml)': {
        'cam_obstacle_min_h': 0.15, 'cam_obstacle_max_h': 0.5,
        'cam_min_cluster_pts': 1, 'cam_cluster_radius_m': 0.15,
        'lidar_min_intensity': 0.0,
        'hit_inc': 0.5, 'miss_dec': 0.4, 'occ_thresh': 0.85,
        'free_thresh': -0.4, 'min_hits': 3,
        'lidar_free_dec': 0.12, 'occ_erode_factor': 0.5,
    },
}


def load_custom(path=DEFAULT_PATH, log=None):
    """Custom presets {name: values} from the JSON file; {} when missing or
    unreadable (the problem is reported through log(fmt, *args))."""
    path = os.path.expanduser(path)
    try:
        if os.path.exists(path):
            with open(path) as f:
                p = json.load(f)
            if isinstance(p, dict):
                return p
    except Exception as e:
        if log:
            log('could not load presets: %s', e)
    return {}


def save_custom(path, custom, log=None):
    """Write custom presets to the JSON file. Returns True on success."""
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(custom, f, indent=1)
        return True
    except Exception as e:
        if log:
            log('could not save presets: %s', e)
        return False


def presets_list(custom):
    """The list published as direct_status.presets and
    /mapping_manager/direct_presets: builtins not shadowed by a custom preset
    of the same name, then the custom ones."""
    custom = custom or {}
    out = []
    for name, vals in BUILTIN_PRESETS.items():
        if name not in custom:
            out.append({'name': name, 'builtin': True, 'values': vals})
    for name, vals in custom.items():
        out.append({'name': name, 'builtin': False, 'values': vals})
    return out
