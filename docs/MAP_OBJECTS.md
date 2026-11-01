# Map objects contract (2026-11-01)

Binding interface for the map editor rebuild. Workers implement this exactly. A change to anything
here goes through the coordinator first. Names of topics, files and fields are normative.

Goal: a map object is a thing with meaning, not a drawing. Humans and the Vitulus agent use the same
objects, and rules attached to them bind all autonomous motion.

## 1. Files in `~/.vitulus/mapping_v3/<site>/`

Existing files keep their format and owners (`waypoints`, `paths`, `zones`, `edits`.geojson,
`programs.yaml`). Owners keep writing them: navi_man, node_planner, mapping_manager. New files,
all owned by `map_objects_server` and all written atomically (tmp + rename):

| File | Content |
|---|---|
| `places.geojson` | objects of the new types (section 2), UTM like the rest of the bundle |
| `objects_meta.json` | `{"<family>:<name>": meta}` sidecar for waypoint/path/zone/edit objects |
| `history/journal.jsonl` | one entry per change (section 4) |
| `suggestions.json` | robot suggestions (section 7) |
| `coverage.npz`, `gnss_quality.npz` | recorded layers (section 7) |

A site without these files is valid. Missing file = no information, never an error, never a wipe.

## 2. Object model

Every object, whatever file holds it, is presented as:

```json
{"id": "o_8f3a2c1d", "family": "zone|waypoint|path|edit|place",
 "type": "zone|waypoint|path|obstacle|free|wall|place|landmark|nogo|slow|noblade|corridor",
 "name": "...", "geometry": "point|line|area", "points": [[x, y]], "yaw": null,
 "aliases": [], "note": "", "tags": [], "source": "owner|agent|robot",
 "created": 0.0, "updated": 0.0,
 "valid_from": null, "valid_until": null, "active": true,
 "protected": false, "props": {}}
```

- `points` are in the map frame in topics, UTM in files. Times are unix seconds.
- `id` is stable across rename and reshape and is generated once (`o_` + 8 hex). `name` stays the key
  for existing owner topics; unique within a family, rules in `objnames.py`.
- `protected: true` for DOCK, UNDOCK, DOCKED: cannot be deleted by anyone, UI hides Delete, the
  server refuses. Moving stays allowed.
- `props` by type:
  - `zone`: the existing zone props plus `effective` when a program overrides them.
  - `slow`: `speed_mps` (0.05 to 0.40, default 0.20).
  - `corridor`: `width_m`, `connects` (zone or place names).
  - `wall`, `obstacle`, `free`: as in edits.
  - `place`, `landmark`: `category` free text (terasa, strom, poklop, bazén...).
- New types by geometry: `place` area, `landmark` point, `nogo` area, `slow` area, `noblade` area,
  `corridor` line.
- `valid_from` / `valid_until` make any `nogo`, `slow`, `noblade` temporary. `active` is computed.
- Validator refuses a `nogo` that contains DOCK, UNDOCK or DOCKED, a self-crossing outline, and an
  area under 0.05 m2. It reports, not refuses, a `nogo` that overlaps a zone.

## 3. `map_objects_server` (new node, package vitulus_mapping)

All JSON in `std_msgs/String`.

| Topic | Dir | Payload |
|---|---|---|
| `/map_objects/op` | in | `{"req": "...", "actor": "owner|agent|robot", "op": "save|update|delete|undo|redo|group", ...}` |
| `/map_objects/result` | out | `{"req", "ok", "reason", "id", "warnings": []}` |
| `/map_objects/objects` | latched | `{"site", "seq", "objects": [...]}`, all families |
| `/map_objects/rules` | latched | `{"site", "seq", "stamp", "rules": [{"id","type","name","points","speed_mps"}]}`, active rules only, map frame |
| `/map_objects/history` | latched | `{"seq", "can_undo", "can_redo", "entries": [last 100]}` |
| `/map_objects/suggestions` | latched | section 7 |

- `save` carries a full object (no `id` = create), `update` carries `id` + changed fields, `delete` carries `id`,
  `group` carries `ops: [...]` applied and undone as one step.
- `place` family is written directly. Other families forward to the owner's existing topic
  (`/navi_manager/save_waypoint_at`, `remove_point`, `save_path_at`, `remove_path`,
  `/mapping_manager/save_edit`, `remove_edit`, planner zone save/remove, rename topics) and wait for
  the file to change or the owner's result. Meta-only fields go to the sidecar.
- Existing direct UI topics keep working. The server watches bundle mtimes at 1 Hz, diffs and journals
  anything that changed without it (`actor: "unknown"`), and assigns ids. Computed zone props (`area`,
  `length`) alone are not a change.
- A site switch reloads everything. The server is safe with a missing datum: objects list empty,
  reason in `result`.
- Activation of temporary rules: 5 s timer, republish `/map_objects/rules` only when the set changes.
- Python API in `vitulus_mapping/objects.py`: `load_all(site)`, `active_rules(site, now)`,
  `find_id(...)`, `validate(obj, site)`, pure and without rospy, used by the node, mapping_manager and tests.

## 4. History and undo

Entry: `{"seq","ts","actor","site","group","op","family","id","name","before","after"}`,
`before`/`after` full object or null. Journal keeps all entries, undo reach is 200. Undo applies the
inverse through section 3 forwarding and appends a new entry, so history is never rewritten. A zone
undo uses the planner save. Undo that cannot apply (zone used by running program) returns `ok:false`
with the reason.

## 5. Rules enforcement

### Stamping (vitulus_mapping)
`nogo` polygons stamp as obstacle in `mapping_manager._composite_edits`, so global planner, local
costmap and coverage all avoid them. Re-serve only when the active set changes (`/map_objects/rules`
subscription or change-only timer, check thread safety of `_publish_site_map`). Publish also
`/mapping_manager/site_map_loc`, the same map without `nogo` stamping; `gloc_server` uses it, so a
rule never becomes a phantom wall for localization.

### Motion limit (vitulus_base `cmd_vel_mux`, vitulus_safety `map_rules_guard`)
New latched-free topic `/cmd_vel_mux/limit`, JSON at 5 Hz from the guard:
`{"source":"map_rules","stamp":t,"v_fwd":0.4,"v_back":0.25,"w":0.9,"w_left":0.9,"w_right":0.0,"rules":2,"reason":"nogo:Bazen"}`.
`w_left` (+angular.z) and `w_right` (-angular.z) are optional; `w` = max of both. Without them the mux uses `w`.
A rotation away from a `nogo` stays possible with `v_fwd = v_back = 0`, rotation into it does not.

- Mux clamps every non-human input at its single forward point: `linear.x` clamped to
  `[-v_back, v_fwd]`, `angular.z` scaled by the same factor as linear when speed is capped so the arc
  is kept. Human inputs (`ds4`, `teleop`) are never clamped.
- The clamp is directional. A robot inside or next to a `nogo` can still leave. `/cmd_vel_mux/hold` is
  NOT used for rules, because it blocks leaving and waiting states have no timeout.
- Stale limit (older than 1.0 s) while the last message had `rules > 0`: non-human output is zero, with
  a throttled warning. No message ever, or `rules == 0`: no limit. Mux status reports the active limit.
- Guard node `vitulus_safety/nodes/map_rules_guard`: TF `map->base_link` + `/map_objects/rules` at 5 Hz.
  `slow`: cap when the footprint overlaps. `nogo`: `v_fwd`/`v_back` tapering to 0 at 0.15 m footprint
  clearance along that direction, stopping-distance aware; `w` limited if rotation sweeps the footprint in.
  TF older than 1 s with rules present: `v_fwd = v_back = w = 0`. Use the real footprint from config.
- Guard publishes `/map_rules/state` latched JSON `{"in":[...],"near":[...],"limit":{...},"blade":"..."}`
  for UI and agent, and `/map_rules/warning` when a human drives into a `nogo`.
- `drive_primitive` refuses a command whose straight-line target crosses a `nogo` at start, with a reason.
- `~enabled` param, default true. With no rules the guard publishes `rules: 0` and changes nothing.

### Blade (`map_rules_guard`)
Footprint overlaps active `noblade` while a mower program runs: `/mower/set_motor_on` false, re-asserted
at most 1 Hz while inside (smach recovery can turn it on again, firmware retry too). On exit, `true` only
if the guard stopped it, the program still runs and no stop reason exists. Pattern:
`vitulus_gnss/nodes/gnss_watchdog:303-365`. Forbidden: rpm 0 (firmware ignores it), `/mower/set_power`
in either direction (true resets the MCU on a powered board, false homes the deck and powers off).
Allow 6 s soft-start after exit without treating it as a fault.

### Coverage (vitulus_planner)
`nogo` and `noblade` masks go to OBSTACLE in `np_zone` only, after node_planner:708; `np_zone_navi`
stays so `noblade` remains passable. `/map_objects/rules` change triggers the existing live refresh
worker. A running mission keeps its snapshot; the guard covers it at runtime.

### Stored paths (vitulus_navi)
`navi_man` refuses `execute_path` before publish (~:985) when the path crosses an active `nogo` or a
lethal edit, with a reason on an existing result channel. Backstop in `smach_exe_path` GET_PATH_DATA
with a `first_lethal_index`-style check; a refusal goes to WAIT_FOR_PATH, not the CHECK_PATH loop.

### Human joystick
Never blocked. Guard warns, UI shows a banner.

## 6. Planner additions

| Topic | Payload |
|---|---|
| `/web_plan/zone_check` in, `/web_plan/zone_check_result` out | `{"req","points","settings"}` -> `{"req","ok","mowable_m2","unreachable_m2","length_m","minutes","preview":[...],"problems":[...]}` |
| `/web_plan/zone_op` in, `/web_plan/zone_op_result` out | `{"req","op":"split|merge|duplicate","name(s)","line":[[x,y],[x,y]]}` |

`zone_op` updates the programs that use the zone and refuses during a running program that uses it.
Cut height limit everywhere: 20-80 mm (deck maximum), matching mower settings.

## 7. World layers and suggestions

New node `map_world_recorder` (vitulus_mapping), robot feature, no agent dependency:

- `coverage.npz`: grid at 0.10 m, last-mowed unix time per cell, written when blade rpm > 0 by the deck
  footprint. Flushed every 30 s and at program end.
- `gnss_quality.npz`: grid at 0.5 m, best and median hAcc and fix class seen per cell.
- `/map_objects/layer/<coverage|gnss|slope>` latched JSON
  `{"site","stamp","origin":[x,y],"res","w","h","png_b64","legend":{...}}`, republished at most every 30 s
  on change. `slope` is computed from `dem.npz`.
- Suggestions: persistent new static obstacles, seen at least 3 times from at least 2 positions over at
  least 2 minutes apart, not matching map or objects, from world_query's `static_new`. Entry:
  `{"id","kind":"obstacle|landmark","points","size_m","seen","first","last","state":"new|accepted|ignored"}`.
  Accept and ignore are `/map_objects/op` `{"op":"suggestion","id","action","as":{...}}`. Ignored stays
  ignored at that place.

## 8. Agent

The agent reads everything through `vitulus_mapping.objects` and writes through `/map_objects/op` with
`actor: "agent"`. Every agent write is journaled and undoable. Places resolve by name, alias, tag,
category and relation to a landmark ("u stromu", "za domem"). The `agent_` name prefix requirement goes away.

## 9. Rules for all workers

- Czech to Robert, code comments in the style of the file. UI text only through i18n, en + cs in the same step.
- Every UI or robot change updates the help (`vitulus_ui/docs/help/*.md` + `cs/`) in the same step.
- Robot features live in robot packages. Agent runtime is never a dependency of a robot function.
- Notebook first, phone must work: touch targets 40 px+, no horizontal scroll, map stays visible.
- UI panels follow one anatomy, no left stripe, no jumping layout, remember expanded state.
- Python nodes run from src via relay stubs. A new node needs `catkin_install_python` and a stub built by
  the coordinator; never cp over a stub.
- Never publish to a running ROS system from tests; use fake or pure tests.

## 10. Owner consent and additions (2026-11-01)

- Agent writes (owner's explicit consent): all objects incl. rules; delete everything except DOCK, UNDOCK,
  DOCKED; every change journaled and undoable.
- A `nogo` with `source != "agent"` cannot be deleted, shrunk, retyped or deactivated by actor `agent` on any
  path (delete, update, save with id, group, undo, redo). Enforced in `objects.owner_nogo_problem`, server is
  authoritative, the agent layer is the first line. Reason codes `owner_nogo_<delete|type|shrink|validity>:<name>`.
- `/map_objects/result` is latched, paired by `req`. `history/journal.jsonl` is append with fsync;
  `history/snapshot.json` is the watcher baseline. Undo of a zone delete restores program membership.
- `/cmd_vel_mux/limit` carries optional `w_left` and `w_right`. After a respawn the guard never turns the
  blade on for a pre-restart stop, it reports `restore_failed`.

## 11. Mow load trail (2026-11-01)

Part of `map_world_recorder`, logic in `vitulus_mapping/mowtrail.py`. A run = `/mower_smach/busy` or blade on,
closed 30 s after both ended. Load 0..1: 1.0 = the firmware cuts the motor, higher of `isense / ISENSE_MAX`
(`/mower/stall`) and rpm sag against the stall ratio 0.25. States: `cutting spinup transit blocked overload hot
rule hold`. Temperature is raw, the peak since the row before, spikes are never smoothed or dropped.

Files `<site>/mow_load/`: `<run>.jsonl` while running, `<run>.npz` finished, `index.json`, last 30 runs. Raw on
disk (26 columns, docstring of mowtrail.py, older 20-column runs load with the rest unknown); on the wire 15
columns, downsampled, the rest by a tap.

| Topic | Dir | Payload |
|---|---|---|
| `/map_objects/mow_trail` | out, about 1 Hz | `{"site","run","from","n","stride":15,"start","r":[...],"e":[...]}`; `end` when closed |
| `/map_objects/mow_trail/snapshot` | latched | current or last run, `{"site","run","active","program","start","n","ds","stride","r","limits"}`, at most 12000 rows |
| `/map_objects/mow_trail/request` | in | `{"req","run"}`; run id, `current`, `list`; with `"t"` (0.1 s) the raw row |
| `/map_objects/mow_trail/detail` | out | `{"req","ok","run","row":{...}}`, row with zone, path, chunk, owner, program, near events |
| `/map_objects/mow_trail/past` | latched | answer with its `req`, a separate topic so one client never changes another's view |
| `/map_objects/mow_runs` | latched | `{"site","active","runs":[{"id","program","start","end","stop_reason","rows","length_m","cut_m","mean_load","peak_load","peak_a","peak_temp","time_mow_s","time_transit_s","time_stop_s","stops","energy_wh","rtk_frac","rtcm_gaps","rtcm_rate","rtcm_age_max"}],"limits"}` |

Wire row: `x_cm, y_cm, load_pct, amps_ma, rpm, temp_dc, state (+16 = line break), t_ds, speed_cms, slope_dd, fix, rpm_dip,
hacc_mm, rtcm_age_ds, rtcm_rate_ds`, unknown = -9999. Event on the wire: `t_ds, x_cm, y_cm, kind index` in `blocked hot overload rule temp recovery
backup detour stall no_progress rewindow hold rtcm_gap fix_down fix_up`. `gnss_quality.npz` gained `rtcm_age_ds`
(worst age per cell), absent in older files.
`worldlayers.layer_query(site, "mow_load", points|polygon|neither, run=None)`.
