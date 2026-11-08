# Výška trávy před robotem (grass_height)

Robot měří hloubkovou kamerou D435 výšku trávy v pruhu před sebou a nesplete si
stoupání nebo klesání terénu s vysokou nebo nízkou trávou. Robotí funkce, na
agentovi nezávisí. Kód: `src/vitulus_mapping/grass.py` (logika, bez ROS),
`nodes/grass_height` (uzel), `nodes/grass_calib` (kalibrace). Testy:
`test/test_grass.py`, `test/test_grass_replay.py` (spouštět po souborech).

## Kalibrace krok za krokem (pro ownera)

Potřebuješ: pravítko nebo svinovací metr, kus rovného betonu (stačí plocha
před garáží), posekaný trávník a kus vyšší trávy. Robot vždy **stojí** (žádný
program neběží), vzorek leží **přímo před robotem**. Každý záznam trvá 10 s.

1. **Beton (nula kamery).** Postav robota na rovný beton tak, aby před ním byly
   aspoň 2 m volného betonu. Spusť
   `rosrun vitulus_mapping grass_calib record --label beton --x 1.2`.
   Pravítko tu není potřeba, beton je nula. Pak robota **otoč o 180°** na
   stejném betonu a nahraj beton znovu: z obou směrů se spočítá chyba sklonu
   IMU (sklon betonu se v průměru odečte). Bez toho `fit` skončí chybou
   (`--allow-single-direction` jen výslovně; pak buňky mapy rozlišené koly
   nesou nejistotu 1,5° a třídu, kterou nejistota neunese, nezapíšou).
2. **Posekaná tráva.** Postav robota na trávník, který právě posekal (kola na
   trávě). Pravítkem změř výšku trávy na 3 místech ve vzdálenosti 0,8 / 1,2 /
   1,6 m **od středu robota** (polovina mezi předními a zadními koly; přední
   hrana robota je asi 0,3 m od středu), v ose robota: pravítko postav na
   zem (ke kořenům, ne na drn) a odečti, kam sahá většina stébel (ne nejvyšší
   jednotlivé stéblo). Spočítej průměr v mm.
3. Pro každou vzdálenost nahraj záznam (robot stojí, nikdo před ním):
   `grass_calib record --label posekano --true-mm <průměr> --x 0.8`, totéž s
   `--x 1.2` a `--x 1.6`.
4. **Vyšší tráva.** Najdi místo, kde je tráva aspoň o 4–6 cm vyšší než posekaná
   a robot stojí na posekané trávě těsně před ní (nebo v ní). Změř stejně jako
   v kroku 2 a nahraj `--label vysoka --true-mm <průměr>` pro `--x 0.8 / 1.2 / 1.6`.
5. **Opakování.** Když měření nesedí, prostě ho nahraj znovu: `fit` bere
   jen **poslední** záznam každého štítku a vzdálenosti (beton i podle
   směru). Seznam: `grass_calib list`; smazat štítek: `grass_calib clear
   --label posekano`. Starší sadu vybereš `fit --session 20261108`, všechny
   záznamy `fit --all`.
6. **Výpočet.** `grass_calib fit`. Vypíše nulu kamery (offset, sklon), chybu
   sklonu IMU, měřítko (`true = k * raw`, beton je bod 0,0, žádný další
   posun), průměrnou odchylku každého štítku a chyby na každém vzorku (sklon
   robota proti betonu podle IMU je u každého vzorku opravený) a doporučenou
   vzdálenost koridoru podle šumu. Chyby do ±10 mm jsou dobré, do ±20 mm
   použitelné. Když posekaná a vysoká tráva nesedí na jednu přímku (víc než
   10 mm), vypíše VAROVÁNÍ: změř znovu. Uzel si novou kalibraci načte sám do
   5 s. `--dry-run` jen spočítá; předchozí kalibrace zůstane jako `.prev`.
7. **Kontrola.** `grass_calib check` vypisuje každou sekundu, co kamera vidí
   (výška, třída, zdroj terénu, drsnost). Postav robota před beton, posekanou
   a vysokou trávu a porovnej s pravítkem. Ukončíš Ctrl+C.

Záznamy (statistiky .json + surové snímky .npz) jsou v `~/.vitulus/grass/calib/`,
výsledná kalibrace v `~/.vitulus/grass/calib.json` (předchozí jako `.prev`).

## Co uzel dělá

Uzel běží jen když robot jede nebo seče (otáčky nože, nenulový příkaz
rychlosti v posledních 15 s); samotné `/mower_smach/busy` u doku bez pohybu ho
nebudí. Jinak spí: drží jen dva malé odběry. Za jízdy bere hloubku jako surové
bajty a zpracuje nejvýš `rate_hz` (4) snímků za sekundu, každý 4. pixel
(asi 4 ms numpy na snímek). Přenos byl změřen: surový proud 15 Hz do rospy
stojí 0,4 % jádra a v kameře nic; compressedDepth by v procesu kamery kódoval
PNG každého snímku, decimující nodelet by musel běžet v procesu kamery, ani
jedno se nevyplatí. Póza se bere v čase snímku (bez ní snímek nedává buňky).

Bez kalibrace (`~/.vitulus/grass/calib.json` chybí nebo je vadný) uzel varuje
v logu a `/grass/ahead` nese `reason: uncalibrated`; čísla jsou surová.

Soubor `grass.npz` se nikdy nepřepíše staršími daty: buňky od posledního
uložení drží uzel v deníku, a když se soubor mezitím změnil (jiná instance,
restart), načte ho znovu a přidá jen svoje; sloučení a uložení běží pod
zámkem souboru (fcntl, `grass.npz.lock`). Mřížku vlastní pracovní vlákno
(místo, DEM, kalibrace, ukládání, vykreslení vrstvy), smyčka snímků do ní
jen posílá buňky frontou, takže uložení ani vykreslení snímky nezdrží. Buňky
daleko od místa (DEM a dosavadní vrstva + 30 m) se ignorují, mřížka má strop
150 × 150 m (51 MB polí v plné velikosti; medián se počítá jen nad viděnými
buňkami). Náklon (roll/pitch) se bere z IMU v čase snímku (interpolace);
když IMU chybí, z TF a bez odečtu offsetu IMU.

Spuštění ručně mimo stack: `roslaunch vitulus_mapping grass_height.launch
respawn:=false`, aby si uzel při startu stacku převzal jeho launch.

**Výstupy**

- `/grass/ahead` (JSON, každý zpracovaný snímek): `canopy_mm` (vrch porostu nad
  zemí pod robotem), `ground_mm` (terén v koridoru proti zemi pod robotem),
  `height_mm` (výška trávy), `class` (`below_cut` | `+2-4cm` | `>+4cm` |
  `unknown`, proti výšce sekání z `/mower/status`), `terrain_source`
  (`wheels` | `dem` | `extrapolated` | `surface`), `slope_deg` (sklon terénu
  vpřed, + do kopce), `confidence` (pravděpodobnost, že třída sedí),
  `uncertainty_mm`, `reason` (`terrain_change`, `low_confidence`, `no_data`,
  `uncalibrated`). Na hraně posekané a neposekané trávy navíc `lane_edge`,
  `lanes_mm` (výška každého pruhu), `max_lane_mm`, `min_lane_mm`,
  `min_lane_class`; `height_mm` a `class` jsou pak vysokého pruhu.
- Vrstva mapy `grass`: `/map_objects/layer/grass` (stejný formát jako coverage/
  gnss/slope), soubor `~/.vitulus/mapping_v3/<site>/grass.npz`: mřížka 0,2 m,
  poslední výška, medián (histogram po 10 mm), počet pozorování, čas, zdroj
  terénu. Barvy jsou třídy mediánu proti výšce sekání (zelená = do výšky sečení
  + 2 cm, žlutá = +2–4 cm, oranžová = víc). Buňky jsou po 0,2 m z bodů, které
  do nich padnou, takže hrana posekané trávy zůstane v mapě ostrá (žádný
  průměr koridoru). Čtení bez ROS: `worldlayers.layer_query(site, "grass", points=…)`.
- Sloupec do záznamu sečení (mow_load) se nepřidává: řádek má pevný formát
  verze 4 a drát má vlastní formát; vrstva nese totéž čistě.

## Jak se rozliší terén a tráva

Kamera vidí jen vrch porostu („koberec“), nikdy zem pod ním. Proto:

1. **Odložené rozlišení (vrstva).** Každý snímek uloží vrch koberce každé buňky
   0,2 m s časem. Zem buňky přijde později: když přes ni přejedou kola (výška
   robota integrovaná ze sklonu IMU a pohybu po mapě; konstantní chyba sklonu
   se odečte, protože koberec i zem jsou vyjádřeny stejným sklonem), nebo po
   `pending_ttl_s` z DEM dřívějších jízd (`dem.npz`, dotyk kol). DEM se bere
   relativně k místu, kde robot stál, když se díval. Do vrstvy jde jen
   rozlišená buňka.
   Do buňky se počítá jen pohled, který z ní viděl aspoň polovinu plochy
   (v zatáčce okraj koridoru přejíždí buňky jen rohem), a buňka, jejíž body
   mají plochý spodek pod mnohem vyšším vrchem (trávník a vrch obrubníku nebo
   svahu), se zahodí; zahodí se tak i buňka rozpůlená hranou posekané trávy,
   hranu nesou buňky vedle. Při ukončení uzlu se nejdřív rozliší čekající
   buňky (DEM), pak se uloží mřížka.
   Kola rozhodují jen buňky viděné nejvýš 4 m stopy zpět (integrovaná výška
   ujíždí); odchylka sklonu IMU se odečte podle kalibrace. Chyba sklonu se
   ruší, když robot jel od pohledu k buňce rovně; nejistota buňky roste s tím,
   o kolik cesta uhnula z přímky (zatáčky, otočky), a buňka, jejíž třídu
   nejistota neunese, se do vrstvy nezapíše. Jedna epocha =
   jedna souvislá výška: skok pózy nebo mezera začne novou a staré buňky kol
   se zahodí.
2. **Okamžitý odhad.** Zem v koridoru = buňky, které kola znají nejvýš 3 m
   stopy zpět (nejistota roste s touto vzdáleností), jinak DEM, jinak terén
   protažený dopředu. Protahuje se **vyhlazený** sklon stopy (medián za
   poslední 1 m a nejvýš 3 s), ne okamžitá rovina kol, takže kývání těla na
   drnech se nepočítá jako tráva. Když ale koberec i nejnižší pruh leží s
   okamžitou rovinou těla (do 3°) a sedí s ní líp než s mediánem, tělo už je
   na novém sklonu (vjezd na vyvýšeninu) a platí rovina těla.
3. **Změna terénu.** Koberec v koridoru proti očekávané zemi:
   - sklon nejnižšího pruhu je terén **jen když je ten pruh holý** (hladký
     a na zemi): mírná rampa betonu se odečte. U drsného pruhu může jít
     o trávu, která dopředu houstne: výšku nikdy nesníží, jen zvětší
     nejistotu; prohlubeň (sklon dolů) se započte, aby vysokou trávu
     neschovala (nulou se ořezává jen výsledek, nikdy terén);
   - koberec skloněný o víc než 2,5° **a** skloněný i nejnižší pruh: zvedá se
     celý koridor souvisle, je to terén → bez kol/DEM `unknown`,
     `reason: terrain_change`;
   - pruhy po 7 cm vedle sebe tvoří ostrý schod (> 25 mm; plynulý boční
     sklon se od schodu odliší proložením přímkou): hladký vyšší pruh je
     obrubník nebo chodník a tráva se čte z nižších pruhů; vyšší pruh je
     hrana posekané a neposekané trávy jen když jeho spodek (p10) sedí na
     zemi nízkých pruhů: měřeno od spodku nízkých pruhů je do 20 mm, nebo
     leží v dolní polovině výšky od spodku nízkých pruhů po vrch (p90)
     vyššího pruhu, tj. `p10 − lo_bot ≤ 0,5·(p90 − lo_bot)` (`edge_bottom_frac`).
     Tráva má stébla všech délek, takže tenhle podíl drží pro 6 cm i 25 cm
     trávy i vedle holé cesty; obrubník nebo svah zvedne spodek s vrchem
     a podíl jde k 1. Vrch vyššího pruhu navíc nesmí být plochý. Obě strany
     schodu se posuzují stejně: pruh se dvěma výškami (leží přímo na hraně)
     se nepočítá ani k jedné straně a každá strana musí mít aspoň jeden čistý
     pruh. Hlásí se pak výška každého pruhu a odpověď je vysoký pruh.
     Zvednutý celý pruh (obrubník, okraj záhonu, kamenná obruba, svah) je
     terén → `unknown`; nerovné strany schodu (keř, předmět) taky;
   - hladký koberec (drsnost ≤ `rough_bare_mm`) blízko očekávané zemi
     (± 20 mm) je holý povrch (`surface`). O trávě rozhoduje výška nad zemí,
     drsnost jen podpůrně: hustá rovnoměrná tráva vysoko nad zemí je tráva.

## Parametry

`config/grass_height.yaml` (koridor x 0,8–1,6 m, |y| < 0,35 m, `rate_hz`,
`stride`, `pending_ttl_s`, `slope_change_deg`, `conf_min`, …). Doporučený
koridor z `grass_calib fit` má přednost před konfigurací.

## Ověřeno

- Syntetické snímky (rovina, schod betonu, svah s trávou, vysoká tráva, okraj
  posekaného pruhu, jízda přes rampu) a scénáře nezávislé kontroly (skok pózy
  po sjezdu, chyba IMU 0,7° tam a zpět, kývání ±1,5° při 0,7 Hz, rampa ±2,4°,
  obrubník 5 a 8 cm, příkop s vysokou trávou, omezení paměti):
  `test/test_grass.py`.
- Přehrání dvou záznamů z trávníku (cap1, cap2, 2026-11): jen `below_cut`
  nebo `unknown` (začátek cap2 = robot vjíždí na vyvýšeninu, DEM potvrzuje
  terén), nikdy vysoká tráva: `test/test_grass_replay.py` (data mimo git,
  `GRASS_CAPS`).

## Známá omezení v1

První verze je záměrně konzervativní: když si není jistá, hlásí `unknown`,
ne vysokou trávu. Ladit se bude na skutečných datech až po kalibraci
(`grass_calib`: beton ve 2 směrech, posekaná tráva, vyšší tráva); další
ladění na syntetických scénářích už má malý přínos.

- **Šikmé hrany.** Hrana posekané a neposekané trávy rovnoběžná s jízdou se
  pozná spolehlivě (obě strany stejně, tráva 13–25 cm vedle posekané i vedle
  holé cesty). Hrana šikmo přes koridor se rozpadne: při 10–20° se pozná
  jen část snímků, při 30° a víc skoro žádný; výsledek je `unknown`, nebo
  `below_cut`, dokud je v koridoru hlavně posekaná část.
- **Zatáčky.** Koridor míří po tečně, v zatáčce vidí hlavně vnější stranu
  oblouku. Neposekaná tráva těsně uvnitř oblouku pak může vyjít `below_cut`,
  hrana těsně vně oblouku `unknown`. Do mapy se buňka zapíše, jen když ji
  pohled viděl aspoň z poloviny.
- **Hustá a polehlá tráva.** Hustá rovnoměrná tráva má spodek vysoko pod
  vrchem jako tráva na obrubníku; hrana husté trávy proto vychází `unknown`
  (nerozliší se od zatravněného obrubníku). Celoplošná hustá nebo polehlá
  tráva blízko hranice tříd vychází `unknown` (`low_confidence`).
  Zmírnění `edge_top_rough` na tom nic nemění (ověřeno 0,15 → 0), proto
  zůstává 0,15.
- **Obrubníky a svahy vedle stopy.** Hladký holý obrubník se pozná jako
  obrubník a tráva se čte z nízkých pruhů (typicky `below_cut`); zatravněné
  obrubníky a svahy vycházejí `unknown`. Krátká tráva na obrubníku (≤ 6 cm)
  nikdy nevyjde jako vysoká tráva.
- **Nízká obruba s delší trávou.** Na obrubě nebo schodu 2,6–4 cm s trávou
  8–10 cm se hlásí výška tráva + schod, třída tak může vyjít o stupeň výš
  (`+2-4cm` → `>+4cm`). Vědomý důsledek kritéria spodku (D41: neladit na
  syntetice, ověřit na datech).
- **Svislé plochy vedle stopy** (zeď, plot, bok bedny) se mohou vyhodnotit
  jako hrana neposekané trávy (`>+4cm`): body na svislé ploše mají spodek u
  země a hrubý vrch a za plochou nejsou pruhy. Výstup `/grass/ahead` zatím
  nikdo neřídí (jen informace); ve v2 vyžadovat ≥ 2 čisté vyšší pruhy nebo
  body za vyšším pruhem.

## DEM a výška země (2026-11-08)

`trajectory_dem` odečítal od výšky base_link „poloměr kola“ 0,055 m, ale
base_link podle URDF leží v rovině dotyku kol (středy kol 0,105 m nad ním,
poloměr 0,1025 m). Všechny dosavadní `dem.npz` jsou proto o 5,5 cm níž.
Opraveno v `trajectory_dem` (`ground_offset_m: 0.0` v `mapping_v3.yaml`);
`DemGrid.load_npz` starý soubor (bez klíče `ground_ref`) při načtení zvedne
o 0,055 m a `trajectory_dem` ho při dalším uložení zapíše se značkou. Na výšku
trávy to vliv nemá: DEM se tu bere jen relativně k DEM pod robotem a posun se
odečte (test `test_demgrid_ground.py`). Sklon (vrstva slope) se nemění.
