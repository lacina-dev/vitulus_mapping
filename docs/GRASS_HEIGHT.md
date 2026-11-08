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
6. **Výpočet.** `grass_calib fit`. Záznam, při kterém IMU neukazovalo stálý
   sklon (rozptyl přes 0,3°, robot se hýbal nebo IMU zlobí), se vyřadí (nedává
   měřítko ani offset IMU); `record` i `fit` to vypíšou (VAROVÁNÍ: nahraj znovu). Vypíše nulu kamery (offset, sklon), chybu
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
  `uncalibrated`). Se zemí z kol navíc `ground_track_m` (kolik stopy zpět
  kola zem razítkovala), `drift_rate` (sklon křivky driftu, m/m) a při
  částečném pokrytí `ground_cells_frac`. Na hraně posekané a neposekané trávy navíc `lane_edge`,
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
   Kola rozhodují buňky viděné nejvýš 4 m stopy zpět (integrovaná výška
   ujíždí), se známou křivkou driftu (viz níž) až 110 m; odchylka sklonu IMU
   se odečte podle kalibrace. Buňka, kterou kamera uvidí znovu z dalšího
   pruhu (víc než 4 m stopy od prvního pohledu), začne pozorování znovu. Chyba sklonu se
   ruší, když robot jel od pohledu k buňce rovně; nejistota buňky roste s tím,
   o kolik cesta uhnula z přímky (zatáčky, otočky), a buňka, jejíž třídu
   nejistota neunese, se do vrstvy nezapíše. Jedna epocha =
   jedna souvislá výška: skok pózy nebo mezera začne novou a staré buňky kol
   se zahodí.
2. **Okamžitý odhad.** Zem v koridoru = buňky, které kola znají (bez křivky
   driftu nejvýš 3 m stopy zpět, s ní až 110 m — typicky stopa předchozího
   pruhu), jinak DEM, jinak terén protažený dopředu. Když kola znají aspoň
   polovinu buněk koridoru, bere se medián přes ně; když jen část (aspoň 2
   buňky, i v předních 70 % koridoru), proloží se rovina stopou robota (čtyři
   kontakty teď) a známými buňkami a zbytek koridoru se čte z ní — jen když
   známé buňky na rovině leží do 20 mm; nejistota roste s tím, jak daleko do
   strany od známých buněk rovina sahá (`ground_cells_frac` ve výstupu). Protahuje se **vyhlazený** sklon stopy (medián za
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

## Drift výšky z kol (2026-11-08)

Výška robota se integruje ze sklonu IMU a pohybu. Za jízdy s nožem IMU ukazuje
sklon o 1,3° jinak než v klidu (vibrace nože; v klidu kalibrace 0,94°), takže
integrovaná výška ujíždí ~2,1 cm na metr stopy (test 8. 11.: 4,7 m za 228 m,
s nožem 2,3 %, bez nože 0,8 %). Na blízko se chyba ruší (koberec i zem jsou
vyjádřeny stejným sklonem), ale zem ze stopy předchozího pruhu (10–60 m stopy
zpět) byla nepoužitelná, a proto byl okamžitý odhad z 95 % extrapolace.

Kola teď razítkují celou stopu robota (3 × 5 bodů: nápravy a střed, kola,
půl cesty, osa) do buněk 0,2 m a drží je 120 m stopy. Když kola přejedou buňku,
kterou už razítkovala (≥ 3 m stopy dřív), rozdíl výšek je drift mezi těmi dvěma
místy stopy. Z těchto návratů se prokládá **křivka driftu** D(s) (po částech
lineární, uzel každé 4 m za posledních 120 m stopy, nejmenší čtverce s penalizací
zakřivení, odlehlé hodnoty potlačené, D(teď) = 0) a zem z dřívějška se převede
na drift chvíle, kdy se kamera dívá. Kauzální ověření na záznamu 8. 11.: chyba
převodu medián 15 mm na jakékoli vzdálenosti 3–70 m (jeden lineární sklon dával
30–37 mm na 15–45 m). Nejistota země = rozptyl návratů (≥ 8 mm). Křivka platí
až po 20 návratech se součtem 60 m stopy; nová epocha ji zahodí i se zemí z kol.
Nová epocha = mezera nad 2 s, spánek, nebo **skok pózy**: krok delší než
0,15 m + 0,8 m/s × čas mezi snímky (při 4 Hz 0,35 m; robot jezdí do 0,5 m/s),
nejvýš 1 m. Dřív se skok pod 1 m nepoznal a zem z kol se porovnávala přes něj
(simulace skoku 0,5–0,9 m: falešné `>+4cm` s jistotou ~1).

Skok rozložený do víc snímků (EKF korekci GNSS dosouvá postupně, každý krok
pod limitem) se hlídá dvakrát:

- **Kola** (neshoda země): kola se vracejí na buňky, jejichž uložená zem teď
  leží jinde. Kontroluje se každý návrat z víc než 0,6 m stopy zpět: medián
  posledních 6 návratů proti křivce nad max(20 mm, 3σ křivky) → zem z dávna se
  v tomtéž snímku nepoužije a od dalšího začne nová epocha (`epochs_ground`).
  Nekontroluje se v otočce (kurz se za poslední 2 s mění rychleji než 10°/s)
  ani na čerstvé křivce (méně než 150 návratů nebo 20 m stopy): večerní
  záznamy 8. 11. tam bez kontroly dávaly falešné epochy ve víceraných otočkách
  na konci pruhu (návraty z ~10 m stopy o 30–80 mm mimo křivku s σ ~11 mm,
  map→odom pevné) a po nich DEM hlásil `>+4cm` tam, kde zem z kol říkala
  `below_cut`. Slabina: póza posunutá po svahu posune i integrovanou výšku,
  takže kola pod sebou rozdíl skoro nevidí (ruší se); vidí ho až na zakřivení
  terénu, 1,5–1,8 m po začátku posunu.
- **Koberec**: koridor 0,8–1,6 m před koly vidí buňky, které kamera viděla na
  dřívějším průjezdu. Jejich tehdejší vrch porostu (převedený křivkou driftu,
  stejné pásmo vzdálenosti 0,4 m) proti dnešnímu = přesně chyba, kterou by
  dostala výška. Nad max(30 mm, 3,5× rozptyl mediánů tohoto trávníku, nejméně
  3,5σ křivky) a jen nahoru (staré výšky příliš nízko = falešná vysoká tráva;
  sečení mezi průjezdy vrch jen snižuje). Epochu sám nezakládá: na záznamu
  2. běhu byl nad mezí 17× bez posunu (vrch porostu před a po sečení a z jiného
  úhlu se liší, rozptyl ~36 mm).
- **Pochybnost o póze** = koberec nad mezí, nebo kola nahoru na 65 % své meze
  (platí 0,5 m stopy), nebo po epoše z neshody země, dokud kola nemají novou
  křivku. Pak smí zem z dávna i DEM (bere se v póze, je mimo stejně jako ona)
  říct nejvýš `below_cut`; vyšší třída → `unknown` (`low_confidence`,
  `pose_check`) a její výška se nepočítá do vyhlazení dalších snímků. Třída
  nad výšku sečení z extrapolace potřebuje v pochybnosti jistotu 0,7 místo 0,6
  (po epoše nemá koridor žádnou změřenou zem a krtinec se čte jako tráva).
  Bez pochybnosti platí 0,6 jako dřív.

**Cena pravidel.** Na pěti záznamech 8. 11. (1. a 2. běh, 18:21, 18:41, 19:15)
je výstup snímek po snímku stejný jako v masteru (`unknown` 5 / 572 / 62 /
208 / 167, `>+4cm` 2 / 16 / 39 / 87 / 51 snímků; žádná epocha z neshody země).
Skutečná vysoká tráva v simulaci kontrolora (`k_r3.py`, první průjezd 110 /
130 / 180 mm, sečení, hrana, druhý průjezd): stejné jako master ve všech
variantách kromě jedné: druhý průjezd neposekanou 130 mm trávou na zvlněném
terénu s chybou IMU 1,3° najde 403 ze 480 snímků vysoké trávy místo 407 (4×
`pose_check`: koberec tam přes mez bez posunu). Mez koberce 3σ / 25 mm by
stála 23 snímků (384), proto 3,5σ / 30 mm. Simulace posunů pózy
(`k_adv2.py`, 39 variant: zvlněný, boční svah 7°, strmý 14°, hrbolatý s
krtincem, oblouky, šum pózy, chyba náklonu IMU; posun 0,3 / 0,5 / 0,9 m
rozložený do 1–20 snímků, boční 0,4 m, hrbolatý s oblouky 0,6 m): 0 falešných
tříd ve všech 39 variantách a bez posunu žádná epocha navíc (master: 3–27
falešných `>+4cm` u rozložených posunů).
Neplatí to obecně: kde koridor nemá buňky z dřívějška (první průjezd, okraj),
chrání jen kola a extrapolace sama. Kontrola kol ve 4σ místo 3σ by měla na
2. běhu větší rezervu (nejvyšší poměr medián/mez 0,71 místo 0,95), ale na
hrbolatém terénu s oblouky by propustila 14–19 falešných tříd. Falešná epocha
není zadarmo: zahodí zem z kol a dokud kola nemají novou křivku, odpovídá DEM
nebo extrapolace s jejich vlastními chybami (proto pochybnost po epoše a brány
kontroly kol výše).

Počty nových epoch podle příčiny jsou ve `stats` uzlu (`epochs_jump`,
`epochs_gap`, `epochs_ground`; skok a posun v témž snímku se počítají oba),
`carpet_over` = snímky s kobercem nad mezí; log při usnutí.

Výpočet je malá
soustava normálních rovnic (31 × 31, Choleského rozklad v numpy, bez BLAS):
při 4 Hz 2 % jádra jako dřív (soustava ~500 × 31 přes `lstsq` budila
vícevláknový OpenBLAS: 58 % jádra); uzel navíc nastaví `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS` na 1. Křivka se počítá nejvýš jednou za
0,5 m stopy nebo 30 návratů.

Zem z dávna (přes křivku, víc než 3 m stopy) musí sedět na změnu terénu: když
se koberec proti tělu sklání víc než 2,5° a „tráva“ nad touto zemí by byla nad
výškou sečení a podél koridoru roste strměji než 2,5°, stará zem sem nepatří →
`unknown`, `terrain_change` (8. 11.: 4 snímky 120–155 mm při sjezdu do dolíku,
vrstva tam ≤ 80 mm).

Přehrání 2. běhu 8. 11. (bag `terrain_mow_2026-11-08_15-50-05`, 2 860 snímků
hloubky, kalibrace z robota, DEM z robota), koridor v zóně při sečení
(1 428 snímků):

| | před | po |
|---|---|---|
| `unknown` | 52,8 % (terrain_change 416, low_confidence 328) | **24,8 %** (201 / 143) |
| zdroj země | extrapolace 95 %, kola 1 % | kola 62 %, extrapolace 35 % |
| `>+4cm` | 17 (kola 1) | 9 (kola 1, DEM 5, extrapolace 3) |
| výška p25/50/75 | 27 / 46 / 57 mm | 20 / 37 / 55 mm |
| chyba proti vrstvě (buňky přejeté koly hned) | extrapolace +9 mm (IQR 32) | kola −2 mm (IQR 35), mimo 2σ 2 % |

V záznamu je jeden skutečný skok pózy (mezi snímky 1292 a 1293, číslováno od
0: 0,39 m za 0,27 s, v mapě i v odometrii), od něj se zem z kol staví znovu:
před skokem 20,8 % `unknown`, po něm 27,3 %. Bez detekce skoku vycházelo
18,3 %, ale za cenu porovnání země přes skok.

Vrstva: stejné buňky beze změny (medián rozdílu 0 mm), o 14 % buněk víc.
Přehrání ukládaného záznamu `/grass/ahead` z robota sedí se starým kódem
(46 % unknown / medián 46 mm na robotu proti 53 % / 46 mm v přehrání).

**Výška proti pravítku.** Starý medián 46 mm byl nadsazený extrapolací (dolík
se započte, kopeček ne) a výběrem (rozhodnuté snímky končí u ~80 mm, zbytek
`unknown`). Proti zemi z kol má trávník zóny vrch porostu ~35 mm nad rovinou
kol; stejná metoda na stojícím robotu v místě kalibrace dává 56 mm (pravítko
72 mm). Za jízdy ani s nožem se nic neztrácí (stojící robot v zóně 46 mm,
jedoucí 45 mm, obojí extrapolací). Zóna tedy měla kratší trávu než místo
kalibrace; posun mm proti pravítku je věc kalibrace (níže).

## Parametry

`config/grass_height.yaml` (koridor x 0,8–1,6 m, |y| < 0,35 m, `rate_hz`,
`stride`, `pending_ttl_s`, `slope_change_deg`, `conf_min`, …). Doporučený
koridor z `grass_calib fit` má přednost před konfigurací.

## Ověřeno

- Návraty kol a křivka driftu: nelineární drift (2,1 cm/m + vlna) zpětně
  ze šumových návratů, pruhy tam a zpět po zvlněném terénu (sklon do 6°) s
  chybou IMU 1,3°: v dalších pruzích < 20 % `unknown`, výška 46 ± 8 mm, žádná
  falešná třída, neposekaná tráva vedle posekaných pruhů zůstane `>+4cm`,
  skok pózy 0,5 / 0,7 / 0,9 m podél pruhu → nová epocha, žádná falešná třída;
  týž skok rozložený do 3 / 5 / 10 snímků a posun na hrbolatém terénu s
  krtincem (0,5 m za 3 / 20 snímků, 0,6 m za 10) → žádná falešná třída
  (`test/test_grass.py`, třída `Drift`).

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
- **Kalibrace 8. 11. nesedí na přímku.** Posekaná tráva (pravítko 72 mm,
  kamera 57 mm nad rovinou kol) a vysoká (120 mm, kamera 117 mm, s opravou
  sklonu 136 mm) nejdou jedním měřítkem přes nulu betonu: `fit` (bez záznamu
  s nestálým IMU) dává 0,979 (posekaná −15 mm, vysoká +13 mm, VAROVÁNÍ). Pravděpodobně kola stojí na
  drnu nad nulou pravítka (posun, který model bez posunu nepřipouští) a
  pravítko u vysoké trávy čte níž než kamera. Změřit znovu v zóně, kde se
  seče; o posunu na trávníku rozhodnout (testy kalibrace ho zatím zakazují).
- **DEM starší jízdy** dává v trávníku proti zemi z kol o +26 mm vyšší trávu
  s nejistotou jen 15 mm (DEM z 1. 11., `trajectory_dem` od té doby nepsal).
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
