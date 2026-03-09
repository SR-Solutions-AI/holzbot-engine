# Performanță RasterScan și 2 vs 3 etaje

## De ce 3 etaje durează mult mai mult decât 2?

- **Timpul total** = wall-clock al celui mai lent plan per fază.
- **Phase 1** (Raster API + AI walls): rulează în paralel pe planuri (până la 3 workers).
- **Phase 2** (brute force + garage + zone + room_scales): tot în paralel. Un plan cu **multe zone** (garaj, intrare acoperită, terasă, balcon, blacklist) și multe camere face multe apeluri Gemini și BFS → un singur plan poate domina timpul (ex. 400s vs 160s pentru celelalte).

După rulare, în **Rezumat timpi** vei vedea:
- `Raster P1 plan plan_01_cluster_1`, `Raster P2 plan plan_01_cluster_1`, etc. → vezi care plan e lent.

## Variabile de mediu

| Variabilă | Efect |
|-----------|--------|
| `HOLZBOT_RASTER_WORKERS=N` | Număr maxim de planuri în paralel (default 3). Ex: `HOLZBOT_RASTER_WORKERS=2`. |
| `HOLZBOT_DEBUG=1` | Pornește toate imaginile de debug și log-urile (încetinește). |

## Ce am optimizat deja

- Phase 1 și Phase 2 rulează în paralel pe planuri (până la 3 workers).
- Tiling: overlap 64, batch 4 (CUDA), `inference_mode()`, resize 1536px pentru imagini mici.
- Fără `HOLZBOT_DEBUG`: mai puține fișiere scrise, fără print-uri verbose în raster_processing.
- Count Objects și Roof rulează în paralel după Scale.
- **OCR zone (când Gemini lipsește):** un singur pas OCR cu toți termenii (garaj, terasă, balcon, intrare, wintergarden) pe 2 regiuni (`FAST_OCR_REGIONS`), apoi atribuire rezultate per zonă. Grid fallback redus la 2×2 (în loc de 4×4). Asta reduce mult timpul când nu e disponibil Gemini.

## Unde se pierde timpul (și ce putem îmbunătăți)

Din Rezumat timpi, ce domină: **RasterScan** (~469s, 68%) = Phase 1 (AI walls) + Phase 2; **Raster P2** (Garage + interior/exterior) ~438s sum = `generate_walls_from_room_coordinates`; **Roof** ~114s; Count Objects ~34s.

**Ce face Raster P2 atât de lent (per plan):** În `generate_walls_from_room_coordinates`: (1) 1 apel Gemini zone labels; (2) **un singur apel batch** Gemini pentru toate camerele (sau fallback N apeluri per cameră); (3) 1 apel Gemini blacklist; (4) **un singur apel batch** pentru toate deschiderile (sau fallback per deschidere); (5) **find_missing_wall (BFS)** când garaj/zonă atinge marginea (max_iterations redus la 30k). Chiar cu batch-urile, timpul rămas vine din: latența API (zone labels + room batch + blacklist + door batch), **BFS-ul** pe planuri mari când există garaj/zone la margine, și procesarea imaginilor (crop-uri, măști, interior/exterior).

**De ce „Garage + interior/exterior” încă durează:** Etapa măsoară **întreaga** funcție `generate_walls_from_room_coordinates` (nu doar garajul). Include: etichetare zone (Gemini), construire crop-uri camere (paralel), **1 apel batch camere** (sau N apeluri la fallback), blacklist (Gemini), **1 apel batch uși/geamuri** (sau N apeluri), **BFS** pentru garaj și pentru fiecare zonă (intrare acoperită, terasă etc.) când flood-fill atinge marginea, plus interior/exterior. Pentru optimizare suplimentară: reduce BFS (ex. max_iterations 15k) sau salt peste find_missing_wall când zona e mică; batch pentru zone labels dacă se poate agrega mai multe imagini.

**Idei de optimizare viitoare:** ~~Batch room labels~~ (făcut: un apel batch pentru toate camerele); ~~batch door classification~~ (făcut); ~~limitare max_iterations la BFS~~ (făcut: 30k). Rămas: reduce și mai mult BFS sau skip când zona e mică; batch zone labels dacă e posibil; remediere warning-uri Shapely în holzbot-roof.
