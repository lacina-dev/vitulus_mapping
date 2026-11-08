# vitulus_mapping

Mapping v3 — "mapping with known poses" for the
[Vitulus](https://github.com/lacina-dev/vitulus) autonomous mower robot.
Instead of running a SLAM system, the package trusts the robot's existing
localization (RTK GNSS + fused odometry) and builds maps from it: a 2D
occupancy raster of obstacles, a trajectory-based ground elevation model
(DEM), and a per-garden **site bundle** (waypoints, zones, programs, human map
edits — all in UTM metres, QGIS-friendly). It runs alongside the navigation
stack, publishes only under `/mapping/*` and `/mapping_manager/*`, and leaves
localization untouched.

> **Status:** personal robot project. Interfaces, topics and file formats
> change without notice; published as a reference, not a reusable component.

## Features

* **Direct 2D raster mapper** (`direct_raster`) — per-cell log-odds occupancy
  built straight from per-frame segmented sensor data: 2D lidar hits + free
  raycasts, camera obstacle points, camera ground points as free evidence.
  Absolute z is discarded entirely, making the map immune to global pose-z
  drift (the failure mode that killed the earlier octomap/band chain).
  Live-tunable per source (enable + max range) at runtime.
* **Gated insertion** (`insertion_gate`) — data enters the map only when the
  pose is trustworthy: pose freshness, jump detection, map-correction-rate
  budget; in strict mode additionally RTK FIXED + accuracy + dual-antenna
  heading agreement. Every decision is logged to CSV. Modes: `fused`
  (default), `rtk` (strict), `off`.
* **Trajectory DEM** (`trajectory_dem`) — cm-class ground elevation from where
  the wheels actually drove (navsat altitude minus wheel radius), immune to
  vegetation. Insertion altitude uses a dead-reckoned z (wheel speed ×
  IMU grade) anchored to RTK by a slow bias, so mapping continues through GPS
  outages.
* **Session lifecycle & serving** (`mapping_manager`) — permanent UI-facing
  node that starts/stops the pipeline per site, renders PNG previews of saved
  maps, serves a chosen raster as a latched `OccupancyGrid`, and manages
  sites/rasters (list, preview, remove).
* **Site bundles** (`src/vitulus_mapping/bundle.py`) — canonical per-garden
  data under `~/.vitulus/mapping_v3/<site>/`: `manifest.yaml`,
  `waypoints.geojson`, `paths.geojson`, `zones.geojson`, `programs.yaml`,
  `edits.geojson`. Geometry is UTM metres (EPSG:326xx), writes are atomic;
  pure Python so the planner and offline tools import it without ROS.
* **Human map edits** (`editmask.py`) — obstacle/free polygons and walls drawn
  in the web UI are composited over the raster at serve time.
* **map_server-compatible output** (`pgmio.py`) — rasters saved as pgm + yaml,
  editable in GIMP/QGIS and loadable by `map_server` unchanged.

## Architecture

```
/rplidar/scan_filtered ──┐
/obstacles_cloud ────────┤→ direct_raster ──→ /mapping/direct_map (OccupancyGrid)
/ground_cloud ───────────┘        ▲                └→ pgm/yaml snapshots per site
                                  │ pose-quality gate (gates.py, fused mode)
/gnss/navpvt, /odometry/gps → insertion_gate ──→ /mapping/gate, gate_status,
                                  │              z_corr, per-scan CSV log,
                                  │              site georef datum.yaml
                                  └→ trajectory_dem → /mapping/terrain (GridMap)
                                                      └→ dem.npz (autosave)

mapping_manager (permanent, started from vitulus_ui.launch)
  starts/stops the three nodes above per site (roslaunch subprocess),
  publishes /mapping_manager/status, terrain_png, raster_png, site_map, …
```

### Main interfaces

| Topic | Dir | Description |
|---|---|---|
| `/mapping_manager/start`, `stop`, `remove_site`, `serve_site`, `show_site`, `preview_raster`, `remove_raster` | in | `std_msgs/String` session & map management (used by the web UI Map tab) |
| `/mapping_manager/status` | out | latched JSON: running state, sites, rasters |
| `/mapping_manager/site_map` | out | latched `nav_msgs/OccupancyGrid` — the served map |
| `/mapping/direct_map`, `/mapping/direct_status` | out | live raster + JSON status |
| `/mapping/gate`, `/mapping/gate_status`, `/mapping/gate_mode` | out/in | gate decision, JSON detail, mode switch (`rtk`/`force_on`/`off`) |
| `/mapping/terrain` | out | `grid_map_msgs/GridMap` (ground_elevation, confidence, …) |
| `/mapping/save_direct`, `/mapping/save_dem`, `/mapping/set_direct` | in | snapshot raster / save DEM / live tuning JSON |
| `/mapping_manager/profile_cmd`, `/mapping_manager/profile_state` | in/out | the mapping profile of the site: JSON commands, latched state (`src/vitulus_mapping/profiles.py`) |

### Data layout (`~/.vitulus/mapping_v3/<site>/`)

* `dem.npz` — canonical trajectory DEM (atomic autosave)
* `mapping_profile.json` — the mapping profile of this site + what differs from it
  (custom profiles: `~/.vitulus/mapping_v3/mapping_profiles.json`, global);
  `direct_overrides.json` is generated from it for direct_raster
* `rasters/<name>/` — pgm + yaml + meta.json snapshots
* `datum.yaml` — site georef anchor (UTM pose of the map origin, captured once
  under sustained RTK quality)
* `waypoints/paths/zones/edits.geojson`, `programs.yaml`, `manifest.yaml` —
  the site bundle
* `logs/gate_*.csv` — per-scan gate decisions
* `terrain/` — persistent terrain map (below); `terrain_backup/<time>/` — maps moved aside by the UI;
  `terrain_import/<name>/` — offline mosaics ready to import; `terrain_mosaic/` — export for the UI

## Persistent terrain map (`terrain_recorder`)

Fuses the local terrain window of `traversability_local` (vitulus_traversability, `/traversability/fuse`:
`grid_map_msgs/GridMap` + layer `fuse_meta`) into UTM tiles of the served site (12.8 m, 5 cm): visits instead
of batches, height aligned onto the stored map (odometry z drift / jumps), changes over time (mowing, moved
objects), terrain bans only as tentative (cost 99) until confirmed over ≥ 3 visits, ≥ 2 sessions and ≥ 2
headings — and even then only listed as `nogo_candidates`, never written anywhere.

* Code: `src/vitulus_mapping/terrainlayers.py` (store), `terrainfuse.py` (message, localisation gate,
  smooth-correction detector, recorder logic, UI commands), `terrainexport.py` (export files, nogo candidates),
  `nodes/terrain_recorder`, `tools/terrain_import_mosaic.py`.
* Start: `launch/terrain_recorder.launch` (included by `vitulus_ui.launch`) starts the node only with
  `VITULUS_TERRAIN_GLOBAL=true`; composing is then **off until switched on from the UI** (`enable`, stored in
  `~/.vitulus/terrain_recorder.json`). `nice -n 19`.
* Fused only while: RTK FIXED + hAcc ≤ 30 mm (`/gnss/navpvt`), `/nav_tf/loc_status` owner `gps` with heading,
  both for the last 10 s; not settling (local node); a batch is fused 3 s (≥ period + 1 s) after its stamp.
  Smooth corrections pulled into odom (`/odometry/odom` vs `/odometry/ekf_wheel_odom` over 1 s, > 3 cm / 0.5°) are
  only counted by default (`corr_enforce: false`); switched on they set a freshness boundary (event + 3 s) and
  batches without a comparison of both odometries are dropped (fails closed).
* Export for the UI (`<site>/terrain_mosaic/`, versioned image names, JSON written last, version counter
  `<site>/terrain_export.version` never reused): a child process at `nice 19` reading the stored tiles read-only
  (a broken tile is skipped, nothing is moved), killed after `export_timeout_s` (300 s), at most every `export_s`
  (600 s) while the map changes, at the end of a program and on `rebuild_export` / `import` / `delete_map`;
  `terrain_backup/` keeps the newest `backup_keep` (5). A mosaic import checks the UTM zone and the distance of the
  mosaic's `datum.yaml` from the site datum (> 0.5 m / 1° only with `force`).
* Height registration of a window: Huber offset over the overlap, or with `align_tilt` a robust plane (offset +
  tilt, capped at `align_max_tilt_deg`) fitted only against closed visits of earlier passes (`test/terrain_align_compare.sh`
  compares both on bags with the consistency metric of the T10 accuracy work).
  Sessions: rising edge of `/mower_smach/busy` or 30 min without fusion (a respawn is no new session).
* Interfaces: `/terrain_recorder/state` (latched JSON for the UI), `/terrain_recorder/cmd` →
  `/terrain_recorder/cmd_result` (enable, disable, pause, resume, delete_map with a confirmation token and a
  backup, import of a mosaic, rebuild_export, list), `/terrain_recorder/stats`, `/terrain_recorder/updates`.
  Contracts: `t30-kontrakt.md` (input), `t31-teren-ui-kontrakt.md` (UI) in the project hand-over folder.
* Tests: `test/test_terrainlayers.py`, `test_terrainfuse.py`, `test_terrain_ui.py` (no ROS);
  `test/terrain_recorder_ros_check.sh` (private roscore, fake messages), `terrain_recorder_bag_check.sh` and
  `terrain_recorder_events_check.sh` (a bag through `traversability_local` → recorder; the latter checks
  settling, shifts against offline mosaics and confirmed bans on the track); `bench_terrainlayers.py`.

## Requirements

* ROS Noetic on Ubuntu 20.04, Python 3
* `grid_map_msgs`, `tf2_ros`, `laser_geometry`, `ublox_msgs`,
  `vitulus_msgs` (from the Vitulus stack)
* Python: numpy, OpenCV (`cv2`), PyYAML
* Sensor inputs from the Vitulus robot: RTK GNSS (u-blox NavPVT + navsat
  odometry), segmented clouds from rtabmap `obstacles_detection`, filtered
  RPLidar scan

## Build & run

All Vitulus packages live side by side in `<workspace>/src/vitulus/` (on the
robot: `~/catkin_ws/src/vitulus/`); see `INSTALL.md` in the
[vitulus](https://github.com/lacina-dev/vitulus) repo for the whole stack.

```bash
mkdir -p ~/catkin_ws/src/vitulus && cd ~/catkin_ws/src/vitulus
git clone https://github.com/lacina-dev/vitulus_mapping.git
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

Normal use is through the web UI
([vitulus_ui](https://github.com/lacina-dev/vitulus_ui), Map tab): type a site
name, **Start**, drive/mow the garden, **Stop** — the raster snapshot is saved
automatically. `mapping_manager` runs permanently from `vitulus_ui.launch`.

Manual start of the pipeline alone:

```bash
roslaunch vitulus_mapping mapping_v3.launch site:=my_garden
# optional per-site resolution (default 0.05 m):
roslaunch vitulus_mapping mapping_v3.launch site:=my_garden resolution:=0.10
```

Useful checks:

```bash
rostopic echo /mapping/gate_status      # gate decision + reasons, hAcc, heading
rostopic echo /mapping_manager/status   # sites, rasters, serving state
```

## Configuration

`config/mapping_v3.yaml` (loaded into the `/mapping` namespace) is heavily
commented and is the reference for all tunables: gate strictness and rate
limits, DR-z dead-reckoning parameters, per-source range caps, log-odds
increments/clamps, `min_hits` anti-transient thresholds, decay, and preview
sizes. Per-site overrides (resolution, source ranges) live in
`<site>/site_config.json` and are written by the live-tuning UI.

## License

MIT (see `package.xml`).
