# 📐 Documentație Detaliată: Scale Detection Workflow

## 📋 Cuprins

1. [Prezentare Generală](#prezentare-generală)
2. [Arhitectura Modulului Scale](#arhitectura-modulului-scale)
3. [Workflow Principal](#workflow-principal)
4. [Metoda 1: RasterScan (Prioritate)](#metoda-1-rasterscan-prioritate)
5. [Metoda 2: CubiCasa + Gemini (Fallback)](#metoda-2-cubicasa--gemini-fallback)
6. [Algoritmi și Procese](#algoritmi-și-procese)
7. [Fișiere Generate](#fișiere-generate)
8. [Structura Output](#structura-output)

---

## 🎯 Prezentare Generală

Modulul **Scale Detection** este responsabil pentru determinarea scării planului arhitectural, adică calcularea valorii **metri per pixel (m/px)**. Această valoare este esențială pentru toate calculele ulterioare (arie, perimetru, măsurători, prețuri).

### Obiectiv
Determinarea precisă a scării planului pentru a permite conversia corectă între pixeli și metri reali.

### Strategie Duală
1. **PRIORITATE**: Folosește scale-ul calculat de RasterScan (dacă disponibil)
2. **FALLBACK**: Folosește CubiCasa + Gemini pentru calcularea scalei

---

## 🏗️ Arhitectura Modulului Scale

### Structura Fișierelor

```
holzbot-engine/scale/
├── __init__.py          # Exportă funcțiile principale
├── jobs.py              # Workflow principal și orchestrator
└── openai_scale.py      # Implementare OpenAI (opțional, nefolosit în producție)
```

### Dependențe

- **cubicasa_detector**: Pentru detecția pereților și camerelor folosind AI
- **RasterScan API**: Pentru vectorizare și detecție automată de camere
- **Gemini API**: Pentru analiza textului și calcularea suprafețelor camerelor

---

## 🔄 Workflow Principal

### Entry Point

```python
run_scale_detection_for_run(run_id: str, max_workers: int = 4)
```

### Pași Principali

1. **Încărcare Planuri**: Se încarcă toate planurile din run-ul curent
2. **Procesare Paralelă**: Se procesează planurile în paralel (max 4 thread-uri)
3. **Detectare Scară**: Pentru fiecare plan se determină scala
4. **Salvare Rezultate**: Se salvează `scale_result.json` pentru fiecare plan

### Funcția Principală per Plan

```python
_run_for_single_plan(run_id, index, total, plan) -> ScaleJobResult
```

---

## 🚀 Metoda 1: RasterScan (Prioritate)

### Când se Folosește

Când există fișierul `room_scales.json` generat de RasterScan în:
```
scale/{plan_id}/cubicasa_steps/raster_processing/rooms/room_scales.json
```

### Workflow RasterScan

#### Pas 1: Apel RasterScan API
- **Locație**: `cubicasa_detector/detector.py` → `run_cubicasa_detection()`
- **Acțiune**: Se trimite imaginea planului către RasterScan API pentru vectorizare
- **Input**: 
  - Imagine plan (JPG/PNG)
  - Preprocesare: eliminare linii subțiri
  - Redimensionare dacă > 2048px (limită API)
- **Output**: 
  - `raster/response.json` - Date vectoriale (camere, pereți, uși, ferestre)
  - `raster/api_walls_mask.png` - Mască pereți
  - `raster/rooms.png` - Imagine cu camere colorate
  - `raster/output.svg` - Vectorizare SVG
  - `raster/output.dxf` - Vectorizare DXF

#### Pas 2: Brute Force Alignment
- **Algoritm**: Căutare exhaustivă pentru transformarea optimă între `api_walls_mask.png` și `02_ai_walls_closed.png`
- **Parametri Testați**:
  - Scale: 0.5x - 2.0x (step 0.05)
  - Position: ±200px (step 10px)
  - Direction: `api_to_orig` sau `orig_to_api`
- **Metrică**: IoU (Intersection over Union) între măști
- **Output**: 
  - `raster/brute_force_best_config.json` - Configurația optimă salvată (cache)
  - `raster/alignment_debug/` - Imagini de debug pentru fiecare configurație testată

#### Pas 3: Generare Crop
- **Algoritm**: Se generează un crop optim al planului bazat pe zona detectată de RasterScan
- **Output**: 
  - `raster/00_original_crop.png` - Crop-ul optimizat
  - `raster/crop_info.json` - Informații despre crop (dimensiuni, offset)

#### Pas 4: Generare Pereți din Coordonate
- **Funcție**: `generate_walls_from_room_coordinates()`
- **Locație**: `cubicasa_detector/raster_processing.py`
- **Algoritm**:
  1. Se transformă coordonatele camerelor din JSON la coordonatele originale
  2. Se generează contururi exterioare din coordonatele camerelor
  3. Se validează segmentele de pereți din JSON (coverage minim 70% cu `api_walls_mask.png`)
  4. Se regenerează camerele folosind flood fill limitat de pereții validați
  5. Se calculează aria fiecărei camere în pixeli
  6. Se trimite fiecare cameră către Gemini pentru estimarea suprafeței în m²
  7. Se calculează metri per pixel pentru fiecare cameră: `m_px = sqrt(area_m2 / area_px)`
  8. Se calculează media ponderată: `weighted_m_px = Σ(area_m2 * m_px) / Σ(area_m2)`

#### Pas 5: Salvare Rezultate
- **Fișier**: `raster_processing/walls_from_coords/room_scales.json`
- **Structură**:
```json
{
  "rooms": {
    "0": {
      "area_m2": 15.5,
      "area_px": 125000,
      "room_name": "Living Room",
      "m_px": 0.01114
    },
    ...
  },
  "total_area_m2": 120.5,
  "total_area_px": 9800000,
  "m_px": 0.01108,
  "weighted_average_m_px": 0.01109,
  "room_scales": {...}
}
```

### Imagini Generate în RasterScan Workflow

#### În `raster/`:
- `00_original_preprocessed.png` - Imagine preprocesată (linii subțiri eliminate)
- `input_resized.jpg` - Imagine redimensionată pentru API
- `api_walls_mask.png` - Mască pereți de la RasterScan
- `rooms.png` - Camere colorate
- `walls_overlay_on_crop.png` - Overlay pereți peste crop
- `rooms_overlay_on_crop.png` - Overlay camere peste crop
- `00_original_crop.png` - Crop optimizat
- `crop_info.json` - Informații crop

#### În `raster_processing/walls_from_coords/`:
- `01_walls_from_coords.png` - Mască pereți generată din coordonate (folosește `walls_overlay_mask` validată)
- `02_walls_thick.png` - Pereți cu grosime aplicată (dilatare morphological)
- `02b_walls_outline.png` - Outline pereți (fără interior)
- `03_walls_overlay.png` - Overlay pereți peste plan (mov)
- `04_walls_3d.png` - Randare 3D izometrică (matplotlib voxels sau fallback)
- `05_walls_outline.png` - Outline roșu pe ambele părți ale pereților
- `06_walls_separated.png` - Separare pereți interiori/exteriori (flood fill)
- `07_walls_interior.png` - Mască pereți interiori
- `08_walls_exterior.png` - Mască pereți exteriori
- `09_interior.png` - Mască interior casei (portocaliu)
- `10_flood_structure.png` - Structură flood fill (albastru/roșu)
- `11_interior_structure.png` - Structură pereți interiori
- `12_exterior_structure.png` - Structură pereți exteriori
- `room_{i}_debug.png` - Debug pentru fiecare cameră (cu pereții validați)
- `room_{i}_location.png` - Locația camerei pe plan (galben + pereți roșii)
- `room_{i}_crop.png` - Crop cameră pentru Gemini
- `room_{i}_mask.png` - Mască cameră
- `room_scales.json` - **FIȘIER CRITIC**: Conține scale-ul calculat
- `openings_measurements.json` - Măsurători uși/ferestre

#### În `raster_processing/openings/`:
- `door_{idx}_{type}.png` - Crop-uri pentru fiecare deschidere (door/window/garage_door/stairs)
- `01_openings.png` - Plan cu toate deschiderile colorate
- `02_exterior_doors.png` - Uși interioare (verde) și exterioare (roșu)

#### În `raster_processing/wall_segments_debug/`:
- `wall_segment_{idx:03d}.png` - Debug pentru fiecare segment de perete (verde=valid, roșu=invalid)

---

## 🔄 Metoda 2: CubiCasa + Gemini (Fallback)

### Când se Folosește

Când **NU** există `room_scales.json` de la RasterScan sau când RasterScan eșuează.

### Workflow CubiCasa

#### Pas 1: Încărcare Model AI
- **Model**: `hg_furukawa_original` (CubiCasa5k)
- **Weights**: `model_weights.pth`
- **Device**: MPS (Apple Silicon) / CUDA / CPU

#### Pas 2: Preprocesare Imagine
- **Filter Thin Lines**: Elimină liniile foarte subțiri (< 0.01% din imagine)
  - Eroziune + Dilatare morphological
  - Output: `00_original.png`, `filter_01_eroded.png`, `filter_02_restored.png`

#### Pas 3: Detecție AI
- **Input**: Imagine preprocesată
- **Model**: Neural network pentru segmentare semantică
- **Output**: Heatmaps pentru:
  - 13 tipuri de pereți
  - 12 tipuri de camere
  - 11 tipuri de iconuri (uși, ferestre, etc.)

#### Pas 4: Post-procesare Pereți
- **Adaptive Closing**: 
  - Imagini mari (>1000px): kernel 0.3% din dimensiune, 2 iterații
  - Imagini mici: kernel 1.0% din dimensiune, 5 iterații
- **Thinning**: Pentru imagini mari, eroziune pentru subțiere pereți
- **Output**: `01b_walls_closed_adaptive.png`, `01c_ai_walls_thinned.png`

#### Pas 5: Reparare Pereți
- **Border-Constrained Fill**: Închide goluri în peretele exterior
  - Detectează cel mai mare contur
  - Generează convex hull
  - Conectează puncte extreme cu validare (test intruziune + ghost)
- **Interval Merging**: Unește segmente de pereți folosind Binary Conflict Profiling
  - LSD (Line Segment Detector) pentru vectorizare
  - Grupare pe axe (orizontale/verticale)
  - Test intruziune: verifică pereți perpendiculari
  - Test Ghost: verifică intensitate în imaginea originală
- **Output**: `02_ai_walls_closed.png`, `02c_border_constrained_fill.png`, `02f_*_*.png`

#### Pas 6: Detectare Scări
- **Input**: `export_objects/detections.json` (detecții Roboflow)
- **Algoritm**: Flood fill în bounding boxes pentru scări
- **Output**: `02_stairs_filled.png`

#### Pas 7: Reparare Pereți Casei
- **Algoritm**: Flood fill din interiorul casei + completare goluri pe contur
- **Output**: `03_house_walls_repaired.png`

#### Pas 8: Detectare Interior/Exterior
- **Algoritm**: Flood fill din colțuri + analiză componente conectate
- **Output**: `03_indoor_mask.png`, `03_outdoor_mask.png`

#### Pas 9: Generare Pereți Interiori/Exteriori
- **Algoritm**: Separare bazată pe flood fill și outline detection
- **Output**: `04_walls_interior_1px.png`, `04_walls_exterior_1px.png`

#### Pas 10: Detectare Scale per Cameră
- **Input**: Etichete text din plan (ex: "15.5 m²")
- **Algoritm**: 
  1. OCR pentru detectarea textului
  2. Parsare regex pentru m²
  3. Calcul aria în pixeli (contur cameră)
  4. Calcul `m_px = sqrt(area_m2 / area_px)`
- **Output**: `scale_detection/room_{i}_*.png`

#### Pas 11: Optimizare Scale
- **Algoritm**: Media ponderată sau mediană a scale-urilor per cameră
- **Output**: `scale_result.json`

---

## 🧮 Algoritmi și Procese

### 1. Brute Force Alignment

**Scop**: Găsește transformarea optimă între masca RasterScan și masca CubiCasa.

**Algoritm**:
```python
for scale in [0.5, 0.55, ..., 2.0]:
    for x_pos in [-200, -190, ..., 200]:
        for y_pos in [-200, -190, ..., 200]:
            for direction in ['api_to_orig', 'orig_to_api']:
                transformed_mask = apply_transform(api_mask, scale, x_pos, y_pos, direction)
                iou = calculate_iou(transformed_mask, orig_mask)
                if iou > best_iou:
                    best_config = {scale, x_pos, y_pos, direction}
```

**Metrică**: IoU (Intersection over Union)

### 2. Binary Conflict Profiling

**Scop**: Unește segmente de pereți fragmentate.

**Pași**:
1. **Vectorizare**: LSD (Line Segment Detector) detectează segmente
2. **Grupare**: Segmentele sunt grupate pe "șine" (aceeași coordonată Y/X)
3. **Sortare**: Segmentele sunt sortate pe fiecare șină
4. **Validare Perechi Adiacente**:
   - **Test Intruziune**: Verifică dacă există pereți perpendiculari între capete
   - **Test Ghost**: Verifică dacă zona e prea albă (cameră) în original
5. **Conectare**: Se conectează doar segmentele care trec ambele teste

### 3. Border-Constrained Fill

**Scop**: Închide goluri în peretele exterior.

**Algoritm**:
1. Detectează cel mai mare contur (perimetrul casei)
2. Generează convex hull
3. Pentru fiecare pereche de puncte extreme din hull:
   - Verifică dacă există gol între ele (< 40% perete)
   - Verifică că linia nu trece prin interior (> 2 puncte în interior = invalid)
   - Verifică că există pereți în jur (neighborhood check)
   - Desenează linia dacă validă

### 4. Scale Detection per Cameră (Gemini)

**Prompt**: `GEMINI_PROMPT_CROP` - Analizează crop-ul camerei și returnează:
- `area_m2`: Suprafața în metri pătrați
- `room_name`: Numele camerei

**Calcul**:
```python
area_px = np.count_nonzero(room_mask)
m_px = np.sqrt(area_m2 / area_px)
```

### 5. Weighted Average Scale

**Algoritm**:
```python
weighted_m_px = Σ(area_m2[i] * m_px[i]) / Σ(area_m2[i])
```

Camerele mai mari au mai multă greutate în calculul final.

---

## 📁 Fișiere Generate

### Structura Output

```
output/{run_id}/scale/{plan_id}/
├── scale_result.json                    # Rezultat final scale detection
├── cubicasa_result.json                 # Cache complet CubiCasa (dacă folosit fallback)
└── cubicasa_steps/
    ├── 00_original.png                  # Imagine originală
    ├── 01b_walls_closed_adaptive.png    # Pereți după closing adaptiv
    ├── 02_ai_walls_closed.png           # Pereți după reparare
    ├── 03_indoor_mask.png               # Mască interior
    ├── 03_outdoor_mask.png              # Mască exterior
    ├── 04_walls_interior_1px.png        # Pereți interiori
    ├── 04_walls_exterior_1px.png        # Pereți exteriori
    ├── raster/                          # Output RasterScan
    │   ├── response.json                # Răspuns API RasterScan
    │   ├── api_walls_mask.png           # Mască pereți RasterScan
    │   ├── rooms.png                     # Camere RasterScan
    │   ├── walls_overlay_on_crop.png    # Overlay pereți
    │   ├── rooms_overlay_on_crop.png    # Overlay camere
    │   ├── 00_original_crop.png         # Crop optimizat
    │   └── brute_force_best_config.json # Configurație alignment
    └── raster_processing/
        ├── walls_from_coords/            # Pereți generati din coordonate
        │   ├── 01_walls_from_coords.png  # Mască pereți (walls_overlay_mask)
        │   ├── 02_walls_thick.png        # Pereți cu grosime
        │   ├── 03_walls_overlay.png      # Overlay mov
        │   ├── 04_walls_3d.png           # Randare 3D
        │   ├── 05_walls_outline.png      # Outline roșu
        │   ├── 06_walls_separated.png    # Separare interior/exterior
        │   ├── 07_walls_interior.png     # Pereți interiori
        │   ├── 08_walls_exterior.png     # Pereți exteriori
        │   ├── 09_interior.png           # Mască interior
        │   ├── 10_flood_structure.png    # Structură flood fill
        │   ├── 11_interior_structure.png # Structură interior
        │   ├── 12_exterior_structure.png # Structură exterior
        │   ├── room_{i}_debug.png        # Debug camere
        │   ├── room_{i}_location.png     # Locație camere
        │   ├── room_{i}_crop.png         # Crop camere
        │   ├── room_scales.json          # ⭐ FIȘIER CRITIC
        │   └── openings_measurements.json # Măsurători deschideri
        └── openings/                     # Deschideri (uși/ferestre)
            ├── door_{idx}_{type}.png     # Crop-uri deschideri
            ├── 01_openings.png            # Plan cu deschideri
            └── 02_exterior_doors.png      # Uși interioare/exterioare
```

### Fișiere Critice

#### `scale_result.json`
```json
{
  "meters_per_pixel": 0.01108,
  "method": "raster_scan_gemini" | "cubicasa_gemini",
  "confidence": "high" | "medium",
  "rooms_analyzed": 5,
  "optimization_info": {
    "method": "weighted_average",
    "rooms_count": 5
  },
  "per_room_details": [
    {
      "room_id": "0",
      "room_name": "Living Room",
      "area_m2": 15.5,
      "m_px": 0.01114
    }
  ],
  "meta": {
    "plan_id": "plan_01_cluster_2",
    "plan_image": "...",
    "generated_at": "2026-01-27T16:23:00Z",
    "stage": "scale"
  }
}
```

#### `room_scales.json` (RasterScan)
```json
{
  "rooms": {
    "0": {
      "area_m2": 15.5,
      "area_px": 125000,
      "room_name": "Living Room"
    }
  },
  "total_area_m2": 120.5,
  "total_area_px": 9800000,
  "m_px": 0.01108,
  "weighted_average_m_px": 0.01109,
  "room_scales": {...}
}
```

---

## 🔍 Detalii Tehnice

### Validare Segmente Perete (70% Coverage)

Pentru fiecare segment de perete din JSON-ul RasterScan:
1. Se transformă coordonatele la sistemul original
2. Se creează o mască pentru linia segmentului
3. Se calculează coverage: `coverage = (wall_pixels / total_line_pixels) * 100`
4. Se acceptă doar dacă `coverage >= 70%`

### Regenerare Camere

1. Se folosește `walls_overlay_mask` (masca validată cu 70%) ca barieră
2. Se face flood fill din centrul fiecărei camere originale
3. Se extrag contururile camerelor regenerate
4. Se validează că aria camerei regenerate este > 100px

### Calcul Scale per Cameră

```python
# 1. Aria în pixeli
area_px = np.count_nonzero(room_mask)

# 2. Aria în m² (de la Gemini)
area_m2 = gemini_result['area_m2']

# 3. Metri per pixel
m_px = np.sqrt(area_m2 / area_px)
```

### Media Ponderată

```python
total_weighted = sum(room_data['area_m2'] * room_data['m_px'] 
                     for room_data in room_scales.values())
total_area = sum(room_data['area_m2'] for room_data in room_scales.values())
weighted_m_px = total_weighted / total_area if total_area > 0 else None
```

---

## ⚠️ Erori și Fallback-uri

### Eroare: `room_scales.json` nu există

**Cauză**: RasterScan nu a generat fișierul sau workflow-ul a eșuat.

**Soluție**: 
1. Se încearcă crearea unui fișier minimal pentru compatibilitate
2. Se folosește fallback la CubiCasa + Gemini

### Eroare: Randare 3D

**Cauză**: Eroare de broadcasting în matplotlib voxels sau import error.

**Soluție**: 
1. Se prinde excepția și se continuă workflow-ul
2. Se încearcă fallback simplu (izometric manual)
3. Dacă și fallback-ul eșuează, se skip randarea 3D

### Eroare: Gemini API

**Cauză**: Rate limit, timeout, sau eroare API.

**Soluție**: 
1. Se prinde excepția pentru fiecare cameră
2. Se continuă cu celelalte camere
3. Scale-ul se calculează doar din camerele procesate cu succes

---

## 📊 Metrici și Performanță

### Confidență Scale

- **High**: ≥ 3 camere analizate cu succes
- **Medium**: < 3 camere analizate

### Acuratețe

- **RasterScan + Gemini**: ± 8-10% (bazat pe estimările Gemini)
- **CubiCasa + Gemini**: ± 10-15% (bazat pe detecția AI + estimări Gemini)

### Timp de Procesare

- **RasterScan Workflow**: ~30-60 secunde per plan
  - API Call: ~10-20s
  - Brute Force: ~5-10s
  - Generare Pereți: ~5-10s
  - Gemini Calls (paralel): ~10-20s
  
- **CubiCasa Workflow**: ~60-120 secunde per plan
  - AI Detection: ~20-40s
  - Post-procesare: ~10-20s
  - Gemini Calls: ~30-60s

---

## 🔗 Integrare cu Alte Module

### Input
- **Plan Image**: Din etapa `segmenter` sau `detections`
- **Detections JSON**: Pentru detectarea scărilor (opțional)

### Output
- **scale_result.json**: Folosit de modulele:
  - `area`: Pentru calcularea ariei
  - `perimeter`: Pentru calcularea perimetrului
  - `pricing`: Pentru calcularea prețurilor
  - `measure_objects`: Pentru măsurătorile obiectelor

---

## 📝 Note Importante

1. **Pereții Perfecti**: Toate fișierele de pereți (`01_walls_from_coords.png`, `02_walls_thick.png`, etc.) folosesc EXACT aceeași mască (`walls_overlay_mask`) ca în `room_x_debug.png` pentru consistență.

2. **Cache Brute Force**: Configurația optimă de alignment este salvată în `brute_force_best_config.json` pentru a evita recalcularea.

3. **Paralelizare Gemini**: Apelurile către Gemini pentru analiza camerelor se fac în paralel (max 4 thread-uri) pentru performanță.

4. **Validare Robustă**: Fiecare pas are fallback-uri și gestionare de erori pentru a permite workflow-ul să continue chiar dacă un pas eșuează.

5. **Compatibilitate**: Formatul `scale_result.json` este consistent indiferent de metoda folosită (RasterScan sau CubiCasa).

---

## 🎓 Concluzie

Modulul Scale Detection este un component critic al pipeline-ului Holzbot, oferind două metode complementare pentru determinarea scării planului. Metoda RasterScan (prioritate) oferă o acuratețe mai bună și un workflow mai rapid, în timp ce metoda CubiCasa (fallback) asigură funcționalitate chiar și când RasterScan nu este disponibil.
