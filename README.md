# vitulus_mapping — Mapping v3 (mapování se známými pózami)

Implementace fází 0–2 z `vitulus_navi/MAPPING_ARCHITECTURE_PLAN.md`. Běží
VEDLE současného stacku, lokalizace nedotčena, vše žije pod `/mapping/*`.

## Architektura

```
/ground_cloud ─┐                              ┌─ /mapping/octomap_point_cloud_centers
/obstacles_cloud ─┤→ insertion_gate → cloud_in → octomap_server (3D archiv důkazů)
 (rtabmap nodelety)   │  RTK FIXED + hAcc/vAcc                     │
                      │  heading shoda, XY inovace,   band_projector (na vyžádání)
                      │  jump cooldown; CSV log        │  DEM kopie → 5% výplň sloupců
                      └→ gate (Bool) → trajectory_dem  │  → sklon-filtr → inpaint
                          z = /odometry/gps − 0.055 m  │  → pásmo z−DEM ∈ [0.10;0.60]
                          → terrain (GridMap) + dem.npz└→ obstacle_map + pgm/yaml
```

**Z-korekce:** nav EKF neestimuje z (TF map→base_link má z≈0). Skutečná výška
se bere z `/odometry/gps` (navsat, `zero_altitude: false`). Gate propuštěným
cloudům podvrhne alias TF frame `<frame>_mapping` s opravenou z — octomap
raytracuje ze správného počátku a voxely mají absolutní výšku konzistentní
s trajektorickou DEM.

## Návod: jak udělat mapu (po lopatě)

1. UI → tab **Map** → sekce Mapping v3: napiš jméno zahrady → **Start**.
2. Jeď (kdekoliv — mapuje se z fúzované odometrie, výchozí mód **Fused**).
   V 3D pohledu se každých ~10 s překreslí mapa: **zelená = zmapovaná zem,
   oranžová = překážky nad zemí** (band 0,10–0,60 m nad lokální zemí).
3. Projeď zahradu (klidně normální sekání; víc průjezdů = lepší mapa).
4. **Snapshot** = uloží všechno (3D archiv + terén + rastr). **Stop**.
   Mapa je v `~/.vitulus/mapping_v3/<jméno>/`; Start se stejným jménem
   pokračuje.

Módy brány: **Fused** (výchozí — mapuje vždy podle fúzované pózy; RTK
kvalita se jen loguje) · **RTK** (přísný — vkládá jen při RTK FIXED,
pro finální přesnou mapu) · **Off** (pauza).

Pozn.: absolutní porovnání navsat XY vs map póza NELZE použít jako bránu —
navsat má vlastní datum (offset ~6 m); proto brána stojí na čerstvosti pózy
+ jump detektoru a RTK kvalita je metadata.

## Ovládání z web UI (doporučené)

Tab **Map** → sekce „Mapping v3 — terrain & obstacles". Zadej jméno site
(zahrady) → **Start** (nová mapa = nové jméno; existující site pokračuje —
při stopu se uloží nový direct rastr). Sekce ukazuje stav brány (OPEN/CLOSED + důvody,
hAcc, Δheading, počty cloudů), DEM statistiky, náhledy elevace a rastru;
tlačítka Raster/Snapshot/Compare a přepínač módu brány (RTK/Force/Off).
Lifecycle řídí node `mapping_manager` (běží trvale z vitulus_ui.launch,
topicy /mapping_manager/start|stop|remove_site|status).

## Spuštění ručně (alternativa k UI)

```bash
roslaunch vitulus_mapping mapping_v3.launch site:=zahrada
# volitelně per-site rozlišení (jinak 0.05 m):
roslaunch vitulus_mapping mapping_v3.launch site:=zahrada resolution:=0.10
```

Pozn. (2026-07-19): octomap/band chain (octomap_server + band_projector) byl
vyřazen — `direct_raster` (přímý 2D log-odds rastr z cloudů, imunní vůči pose-z
driftu) JE mapper. Argumenty `octomap:=` / `octomap_archive:=` už neexistují.

Pak normálně sekat/jezdit. Brána se otevírá sama jen při RTK FIXED; stav:

```bash
rostopic echo /mapping/gate_status     # JSON: pass, reasons, hAcc, heading_diff…
```

## Ovládání

| Akce | Příkaz |
|---|---|
| Snapshot direct rastru | `rostopic pub -1 /mapping/save_direct std_msgs/String "data: '{\"name\": \"direct_manual\"}'"` |
| Garážový průjezd bez RTK | `rostopic pub -1 /mapping/gate_mode std_msgs/String "data: 'force_on'"` (zpět: `'rtk'`) |
| Uložit DEM hned | `rostopic pub -1 /mapping/save_dem std_msgs/String "data: ''"` |

## Výstupy (`~/.vitulus/mapping_v3/<site>/`)

- `dem.npz` — kanonická trajektorická DEM (autosave 60 s, atomicky)
- `garden.ot` — (legacy) octomap 3D archiv starých site; už se nezapisuje
- `rasters/<jméno>/` — pgm + yaml (map_server-kompatibilní, editovatelné) + meta.json
  (direct_<ts> = direct raster; final_<ts> = legacy band raster starých site)
- `logs/gate_*.csv` — rozhodnutí brány per scan
- `compare_*.json` — A/B statistiky vs rtabmap

## Vizualizace (rviz)

- `/mapping/terrain` — grid_map plugin, vrstva `ground_elevation`
- `/mapping/occupied_cells_vis_array` — octomap voxely
- `/mapping/obstacle_map` — výsledný ground-relative rastr

## Polní validace (fáze 1 dle plánu)

1. Projet svah, `rostopic echo /mapping/terrain --noarr` / rviz — elevace musí
   kopírovat RTK výšku (kalibrační konstanta `trajectory_dem/z_offset`).
2. `gate_*.csv` — zkontrolovat heading_diff (má být ~0°; jinak nastavit
   `insertion_gate/heading_offset_deg`) a četnost pádů brány.
3. Po sekání `regenerate` + `compare` — IoU vs rtabmap grid.

## Ladicí parametry (config/mapping_v3.yaml)

- pásmo překážek: `band_projector/band_min|band_max` (0.10–0.60 m nad zemí)
- přísnost brány: `insertion_gate/max_hacc_mm|max_heading_diff_deg|max_xy_innov_m`
- throttle vkládání: `insertion_gate/max_rate_hz` (single-thread octomap!)
- tráva vs země: `band_projector/fill_percentile`, `max_slope_deg`
