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

### Data layout (`~/.vitulus/mapping_v3/<site>/`)

* `dem.npz` — canonical trajectory DEM (atomic autosave)
* `rasters/<name>/` — pgm + yaml + meta.json snapshots
* `datum.yaml` — site georef anchor (UTM pose of the map origin, captured once
  under sustained RTK quality)
* `waypoints/paths/zones/edits.geojson`, `programs.yaml`, `manifest.yaml` —
  the site bundle
* `logs/gate_*.csv` — per-scan gate decisions

## Requirements

* ROS Noetic on Ubuntu 20.04, Python 3
* `grid_map_msgs`, `tf2_ros`, `laser_geometry`, `ublox_msgs`,
  `vitulus_msgs` (from the Vitulus stack)
* Python: numpy, OpenCV (`cv2`), PyYAML
* Sensor inputs from the Vitulus robot: RTK GNSS (u-blox NavPVT + navsat
  odometry), segmented clouds from rtabmap `obstacles_detection`, filtered
  RPLidar scan

## Build & run

```bash
cd ~/catkin_ws/src && git clone https://github.com/lacina-dev/vitulus_mapping.git
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
