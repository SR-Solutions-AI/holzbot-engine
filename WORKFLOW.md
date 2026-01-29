# Workflow Holzbot Engine

Documentație completă a workflow-ului de procesare a planurilor arhitecturale.

## Overview

Workflow-ul procesează planuri arhitecturale (PDF-uri) și generează oferte complete cu măsurători, prețuri și documentație PDF. Procesul este organizat în pași secvențiali, fiecare generând date necesare pentru pașii următori.

## Pași Workflow

### STEP 1-3: Multi-file Segmentation & Classification

**Modul:** `segmenter/jobs.py`

**Proces:**
1. **Conversie PDF → PNG**: Convertește paginile PDF în imagini PNG de înaltă calitate (300 DPI)
2. **Eliminare text**: Elimină textul din plan pentru a obține doar structura
3. **Eliminare hașuri**: Elimină hașurile și pattern-urile pentru a obține linii clare
4. **Detectare contururi**: Detectează contururile pereților
5. **Filtrare grosimi**: Filtrează linii subțiri (false positives)
6. **Solidificare pereți**: Solidifică pereții pentru a obține o structură clară
7. **Detectare zone pereți**: Identifică zonele de pereți
8. **Detectare clustere**: Detectează clustere de planuri (mai multe planuri într-o pagină)
9. **Clasificare**: Clasifică planurile în:
   - `house_blueprint`: Planuri de casă
   - `site_blueprint`: Planuri de amplasament
   - `side_views`: Vedere laterală
   - `text`: Doar text

**Output:**
- Imagini segmentate în `segmentation/src_*_input_*/clusters/plan_crops/`
- Planuri clasificate în `segmentation/src_*_input_*/classified/`
- UI Notification: `segmentation` (solid walls), `classification` (blueprints only)

---

### STEP 4: Floor Classification

**Modul:** `floor_classifier/`

**Proces:**
- Clasifică planurile `house_blueprint` în:
  - `Ground Floor` (parter)
  - `Top Floor` (etaj)
- Folosește GPT-4o pentru clasificare
- Salvează metadata în `plan_metadata/`

**Output:**
- `plan_metadata/_floor_classification_summary.json`
- UI Notification: `floor_classification`

---

### STEP 5: Detections

**Modul:** `detections/jobs.py`

**Proces:**
1. **Roboflow Inference**: Detectează obiecte în planuri folosind Roboflow API:
   - `door`: Uși
   - `window`: Ferestre
   - `double_door`: Uși duble
   - `double_window`: Ferestre duble
   - `stairs`: Scări
2. **Object Crops**: Generează crop-uri pentru fiecare detecție
3. Salvează detecțiile în `export_objects/detections.json`

**Output:**
- `detections/plan_*_cluster_*/export_objects/detections.json`
- Crop-uri pentru fiecare tip de obiect
- UI Notification: Trimisă din `raster_processing.py` (STEP 6) după generarea `01_openings.png`

---

### STEP 6: Scale Detection

**Modul:** `scale/jobs.py` → `cubicasa_detector/`

**Proces:**

#### 6.1. CubiCasa Detection
- Detectează pereții, camerele și deschiderile folosind modelul CubiCasa
- Generează măști pentru pereți, camere și deschideri

#### 6.2. RasterScan API
- Apelează RasterScan API pentru vectorizare
- Primește coordonate precise pentru:
  - Pereți (segmente)
  - Camere (poligoane)
  - Deschideri (uși/ferestre)

#### 6.3. Brute Force Alignment
- Aliniază măștile RasterScan cu imaginea originală
- Testează multiple scale-uri și poziții
- Generează transformarea optimă
- Cache-ază rezultatul pentru eficiență

#### 6.4. Generate Walls from Coordinates
**Modul:** `cubicasa_detector/raster_processing.py`

**Proces:**
1. **Generare pereți**: Generează pereți din coordonatele camerelor
   - `01_walls_from_coords.png`: Pereți inițiali
   - `02_walls_thick.png`: Pereți cu grosime (0.005% din lățimea imaginii, minim 5px)
   - `02b_walls_outline.png`: Outline pereți
   - `03_walls_overlay.png`: Overlay pereți peste plan

2. **Separare interior/exterior**:
   - `05_walls_outline.png`: Outline roșu
   - `06_walls_separated.png`: Separare outline-uri
   - `07_walls_interior.png`: Pereți interiori
   - `08_walls_exterior.png`: Pereți exteriori

3. **Randare 3D**:
   - `04_walls_3d.png`: Randare 3D izometrică cu matplotlib
   - Înălțime pereți: 0.9m (30% din 3m)
   - Pereți cu grosime reală (cuboiduri 3D)

4. **Flood Fill Structure**:
   - `09_interior.png`: Interiorul casei
   - `10_flood_structure.png`: Flood fill pe structură
   - `11_interior_structure.png`: Structura pereților interiori
   - `12_exterior_structure.png`: Structura pereților exteriori

5. **Clasificare deschideri**:
   - Procesează deschiderile din RasterScan
   - Template matching pentru clasificare preliminară
   - Gemini API pentru clasificare finală:
     - `door`: Ușă
     - `window`: Fereastră
     - `garage_door`: Ușă de garaj
     - `stairs`: Scări
   - Generează crop-uri pentru fiecare deschidere în `openings/`
   - `01_openings.png`: Plan cu toate deschiderile colorate
   - `02_exterior_doors.png`: Uși exterioare (roșu) și interioare (verde)

6. **Calcul metri per pixel**:
   - Pentru fiecare cameră:
     - Generează crop cu camera evidențiată
     - Trimite la Gemini pentru estimare suprafață (m²)
     - Calculează metri per pixel: `m² / pixeli`
   - Calculează metri per pixel global din toate camerele
   - Salvează în `room_scales.json`

7. **Măsurători deschideri**:
   - Pentru uși/garage_doors: lățime = `min(bbox_width, bbox_height) * meters_per_pixel`
   - Pentru ferestre/stairs: lungime = `max(bbox_width, bbox_height) * meters_per_pixel`
   - Salvează în `openings_measurements.json` cu status (exterior/interior)

**Output:**
- `raster_processing/walls_from_coords/`: Toate imaginile de procesare
- `raster_processing/openings/`: Crop-uri și imagini deschideri
- `room_scales.json`: Suprafețe camere și metri per pixel
- `openings_measurements.json`: Măsurători deschideri
- UI Notifications:
  - `scale`: `04_walls_3d.png`
  - `detections`: `01_openings.png`
  - `exterior_doors`: `02_exterior_doors.png`

---

### STEP 7: Count Objects

**Modul:** `count_objects/jobs.py`

**Proces:**
1. **Hybrid Validation**: Validează detecțiile Roboflow:
   - Template matching cu referințe
   - Gemini API pentru validare finală
   - Filtrare duplicate
   - Filtrare outdoor (dacă există mască)

2. **Clasificare obiecte**:
   - `stairs`: Scări
   - `door`: Uși
   - `double_door`: Uși duble
   - `window`: Ferestre
   - `double_window`: Ferestre duble

3. **Generare imagine finală**: `final_orange.jpg` cu toate obiectele validate

**Output:**
- `count_objects/plan_*_cluster_*/final_orange.jpg`
- UI Notification: `count_objects`

---

### STEP 10: Measure Objects

**Modul:** `measure_objects/jobs.py`

**Proces:**
1. **Încarcă măsurători**:
   - Prioritate: `openings_measurements.json` din `raster_processing` (workflow nou)
   - Fallback: `exterior_doors.json` (workflow vechi - ELIMINAT)

2. **Calculează lățimi și arii**:
   - Folosește `meters_per_pixel` din `scale_result.json`
   - Calculează lățimi pentru uși
   - Calculează arii pentru ferestre

3. **Agregare**:
   - Agregă informații deschideri
   - Folosește `openings_measurements.json` pentru status (exterior/interior)

**Output:**
- `measure_objects/plan_*_cluster_*/measurements.json`
- UI Notification: `measure_objects`

---

### STEP 11: Perimeter - ELIMINAT

**Status:** ❌ **ELIMINAT COMPLET**

**Motiv:** Perimetrul este calculat direct în STEP 6 (Scale Detection) și este inclus în `cubicasa_result.json`. Nu mai este nevoie de un pas separat.

**Date disponibile în STEP 6:**
- `walls_ext_m`: Perimetrul exterior (pentru finisaje)
- `walls_int_m`: Perimetrul interior (pentru finisaje)
- `walls_skeleton_ext_m`: Structură pereți exteriori
- `walls_skeleton_structure_int_m`: Structură pereți interiori

**Module care folosesc perimetrul:**
- `area/jobs.py`: Citește din `cubicasa_result.json` (prioritate) sau fallback la `perimeter/walls_measurements_gemini.json`
- `roof/jobs.py`: Citește din `cubicasa_result.json` (prioritate) sau fallback la `perimeter/walls_measurements_gemini.json`

---

### STEP 12: Area

**Modul:** `area/jobs.py`

**Proces:**
1. **Încarcă suprafețe**:
   - Prioritate: `room_scales.json` din `raster_processing` (workflow nou)
   - Folosește `total_area_m2` pentru `area_net_m2` și `area_gross_m2`

2. **Calculează arii**:
   - `area_net_m2`: Suprafață netă (din `room_scales.json`)
   - `area_gross_m2`: Suprafață brută (din `room_scales.json`)
   - Folosește înălțimea pereților din `frontend_data` pentru calcule 3D

3. **Măsurători pereți**:
   - Structură pereți interiori: `11_interior_structure.png` → pixeli × `meters_per_pixel`
   - Structură pereți exteriori: `12_exterior_structure.png` → pixeli × `meters_per_pixel`
   - Suprafață pereți interiori: `07_walls_interior.png` → pixeli × `meters_per_pixel`
   - Suprafață pereți exteriori: `08_walls_exterior.png` → pixeli × `meters_per_pixel`

**Output:**
- `area/plan_*_cluster_*/area.json`
- UI Notification: `area` (imagine 3D)

---

### STEP 13: Roof

**Modul:** `roof/jobs.py`

**Proces:**
1. **Detectare tip acoperiș**: Folosește GPT-4o pentru a detecta tipul acoperișului
2. **Calcul suprafață**: Calculează suprafața acoperișului
3. **Calcul înclinare**: Estimează înclinarea acoperișului

**Output:**
- `roof/plan_*_cluster_*/roof.json`
- UI Notification: `roof`

---

### STEP 14-15: Pricing

**Modul:** `pricing/jobs.py`

**Proces:**
1. **Încarcă date frontend**: Materiale, sistem constructiv, performanță, etc.
2. **Calculează prețuri**:
   - Structură
   - Finisaje
   - Ferestre și uși
   - Acoperiș
   - Instalații
   - Logistică
3. **Agregare**: Agregă prețurile pentru fiecare categorie

**Output:**
- `pricing/plan_*_cluster_*/pricing.json`
- UI Notification: `pricing`

---

### STEP 16-17: Offer Generation

**Modul:** `offer_builder.py`

**Proces:**
1. **Construiește ofertă**: Agregă toate datele:
   - Măsurători
   - Prețuri
   - Materiale
   - Configurații
2. **Salvează**: `final_offer.json`

**Output:**
- `final_offer.json`
- UI Notification: `offer_generation`

---

### STEP 18-19: PDF Generation

**Modul:** `pdf_generator.py`

**Proces:**
1. **User PDF**: Generează PDF pentru utilizator (cu branding)
2. **Admin PDF**: Generează PDF pentru admin (fără branding)
3. **Calculation Method PDF**: Generează PDF cu metodologia de calcul (în engleză)

**Output:**
- `output/{run_id}/final_offer_user.pdf`
- `output/{run_id}/final_offer_admin.pdf`
- `output/{run_id}/calculation_method.pdf`
- UI Notification: `pdf_generation`

---

## Date de Intrare

### Frontend Data
- `tenant_slug`: Slug tenant
- `calc_mode`: Mod de calcul (structure, structure_windows, full_house)
- `sistemConstructiv`: Sistem constructiv
- `materialeFinisaj`: Materiale finisaje
- `performanta`: Performanță
- `performantaEnergetica`: Performanță energetică
- `incalzire`: Încălzire
- `logistica`: Logistică
- `ferestreUsi`: Ferestre și uși
- `offer_type_id`: ID tip ofertă
- `allowed_pricing_categories`: Categorii de prețuri permise

---

## Date de Ieșire

### Fișiere JSON

1. **`room_scales.json`** (raster_processing):
   ```json
   {
     "rooms": [
       {
         "idx": 0,
         "area_m2": 7.58,
         "area_px": 119756,
         "meters_per_pixel": 0.0000632
       }
     ],
     "total_area_m2": 141.94,
     "total_area_px": 2534085,
     "meters_per_pixel": 0.007484138
   }
   ```

2. **`openings_measurements.json`** (raster_processing):
   ```json
   {
     "openings": [
       {
         "idx": 0,
         "type": "door",
         "bbox": [x1, y1, x2, y2],
         "width_m": 0.756,
         "length_m": null,
         "status": "exterior"
       }
     ]
   }
   ```

3. **`scale_result.json`** (scale):
   ```json
   {
     "meters_per_pixel": 0.007484138,
     "total_area_m2": 141.94
   }
   ```

4. **`final_offer.json`** (offer_builder):
   - Ofertă completă cu toate măsurătorile, prețurile și configurațiile

---

## UI Notifications (Livefeed)

Workflow-ul trimite notificări UI în format:
```
>>> UI:STAGE:{stage_tag}|IMG:{image_path}
```

### Tabel Complet: Când și Ce Imagini se Trimite

| Stage Tag | Când se Trimite | Imagine Trimisă | Sursă | Modul |
|-----------|----------------|-----------------|-------|-------|
| `segmentation_start` | La începutul STEP 1-3 | Preview pagină PDF convertită | `orchestrator.py` | `notify_ui()` |
| `segmentation` | După STEP 1-3 | `solid_walls/solidified.jpg` | `orchestrator.py` | `notify_ui()` |
| `classification` | După STEP 1-3 | Planuri din `classified/blueprints/` | `orchestrator.py` | `notify_ui()` |
| `floor_classification` | După STEP 4 | - | `orchestrator.py` | `notify_ui()` |
| `scale` | În timpul STEP 6 | `04_walls_3d.png` | `raster_processing.py` | `print()` direct |
| `detections` | În timpul STEP 6 | `01_openings.png` | `raster_processing.py` | `print()` direct |
| `exterior_doors` | În timpul STEP 6 | `02_exterior_doors.png` | `raster_processing.py` | `print()` direct |
| `count_objects` | După STEP 7 | `final_orange.jpg` | `count_objects/jobs.py` | `print()` direct |
| `measure_objects` | După STEP 10 | - | `orchestrator.py` | `notify_ui()` |
| `area` | După STEP 12 | `04_walls_3d.png` (prioritate) sau fallback | `orchestrator.py` | `notify_ui()` |
| `roof` | După STEP 13 | - | `orchestrator.py` | `notify_ui()` |
| `pricing` | După STEP 14-15 | - | `orchestrator.py` | `notify_ui()` |
| `offer_generation` | După STEP 16-17 | - | `orchestrator.py` | `notify_ui()` |
| `pdf_generation` | După STEP 18-19 | Path către PDF generat | `orchestrator.py` | `notify_ui()` |
| `computation_complete` | La final | Path către PDF utilizator | `orchestrator.py` | `notify_ui()` |

### Detalii per Stage

#### `segmentation_start`
- **Când**: La începutul procesării, înainte de segmentare
- **Imagine**: Preview-ul primei pagini PDF convertită în PNG
- **Locație**: `jobs/{run_id}/_previews/page_001.png`

#### `segmentation`
- **Când**: După generarea pereților solidificați
- **Imagine**: `segmentation/src_*_input_*/solid_walls/solidified.jpg`
- **Conținut**: Plan cu pereții solidificați (fără text, fără hașuri)

#### `classification`
- **Când**: După clasificarea planurilor
- **Imagine**: Toate planurile din `segmentation/src_*_input_*/classified/blueprints/`
- **Conținut**: Doar planurile clasificate ca `house_blueprint`

#### `scale`
- **Când**: În timpul STEP 6, după generarea randării 3D
- **Imagine**: `scale/plan_*_cluster_*/cubicasa_steps/raster_processing/walls_from_coords/04_walls_3d.png`
- **Conținut**: Randare 3D izometrică a pereților (înălțime 0.9m, cu grosime reală)
- **Notă**: Trimisă direct din `raster_processing.py` cu `print()`, nu prin `notify_ui()`

#### `detections`
- **Când**: În timpul STEP 6, după clasificarea deschiderilor
- **Imagine**: `scale/plan_*_cluster_*/cubicasa_steps/raster_processing/openings/01_openings.png`
- **Conținut**: Plan cu toate deschiderile colorate:
  - Albastru: Ferestre
  - Verde: Uși
  - Roz: Scări
  - Mov: Uși de garaj
- **Notă**: Trimisă direct din `raster_processing.py` cu `print()`

#### `exterior_doors`
- **Când**: În timpul STEP 6, după determinarea statusului (exterior/interior)
- **Imagine**: `scale/plan_*_cluster_*/cubicasa_steps/raster_processing/openings/02_exterior_doors.png`
- **Conținut**: Plan cu ușile:
  - Roșu: Uși exterioare
  - Verde: Uși interioare
- **Notă**: Trimisă direct din `raster_processing.py` cu `print()`

#### `count_objects`
- **Când**: După STEP 7, după validarea obiectelor
- **Imagine**: `count_objects/plan_*_cluster_*/final_orange.jpg`
- **Conținut**: Plan cu toate obiectele validate (uși, ferestre, scări) marcate în portocaliu
- **Notă**: Trimisă direct din `count_objects/jobs.py` cu `print()`

#### `area`
- **Când**: După STEP 12
- **Imagine**: 
  - **Prioritate 1**: `scale/plan_*_cluster_*/cubicasa_steps/raster_processing/walls_from_coords/04_walls_3d.png`
  - **Fallback 1**: `scale/plan_*_cluster_*/walls_3d_view.png`
  - **Fallback 2**: `scale/plan_*_cluster_*/final_viz.png`
- **Conținut**: Randare 3D sau vizualizare finală a planului

#### `pdf_generation`
- **Când**: După generarea PDF-urilor (STEP 18-19)
- **Imagine**: Path către PDF generat:
  - `output/{run_id}/final_offer_user.pdf` (pentru utilizator)
  - `output/{run_id}/final_offer_admin.pdf` (pentru admin)
  - `output/{run_id}/calculation_method.pdf` (metodologie calcul)

#### `computation_complete`
- **Când**: La finalul întregului workflow
- **Imagine**: Path către PDF utilizator (dacă există)
- **Conținut**: Semnalizare că procesarea este completă

### Notificări Fără Imagine

Următoarele stage-uri nu trimit imagini, doar semnalează progresul:
- `floor_classification`
- `measure_objects`
- `roof`
- `pricing`
- `offer_generation`

---

## Algoritmi Eliminați

### Terasă și Garaj
- **Eliminat**: `fill_terrace_room()` și `fill_garage_room()`
- **Motiv**: Nu mai sunt necesare în workflow-ul nou bazat pe RasterScan
- **Impact**: Workflow-ul folosește doar datele din RasterScan API

### Exterior Doors (Workflow Vechi)
- **Eliminat**: `run_exterior_doors_for_run()` din `orchestrator.py`
- **Motiv**: Înlocuit cu workflow nou în `raster_processing.py`
- **Impact**: Deschiderile sunt procesate direct în STEP 6 (Scale Detection)

---

## Dependențe între Pași

```
STEP 1-3 (Segmentation)
    ↓
STEP 4 (Floor Classification)
    ↓
STEP 5 (Detections) ──┐
    ↓                  │
STEP 6 (Scale) ────────┼──→ STEP 7 (Count Objects)
    ↓                  │         ↓
    ├─→ room_scales.json         │
    ├─→ openings_measurements.json│
    ├─→ cubicasa_result.json ────┘
    └─→ scale_result.json
    ↓
STEP 10 (Measure) ────┐
    ↓                  │
STEP 12 (Area) ────────┼──→ STEP 13 (Roof)
    ↓                  │         ↓
    │                  │         │ (folosește cubicasa_result.json pentru perimetru)
STEP 14-15 (Pricing) ←─┘         │
    ↓                            │
STEP 16-17 (Offer) ←─────────────┘
    ↓
STEP 18-19 (PDF)
```

**Notă:** STEP 11 (Perimeter) a fost eliminat. Perimetrul este calculat direct în STEP 6 și este disponibil în `cubicasa_result.json`.

---

## Cache și Optimizări

1. **Brute Force Alignment Cache**: Cache-ază transformarea RasterScan → Original
   - Fișier: `raster/brute_force_best_config.json`
   - Evită recalcularea pentru același plan

2. **CubiCasa Cache**: Cache-ază rezultatele CubiCasa
   - Evită re-rularea modelului pentru același plan

---

## Erori și Fallback-uri

1. **Gemini API Errors**: 
   - Dacă Gemini eșuează, workflow-ul continuă cu datele disponibile
   - Log-urile indică erorile pentru debugging

2. **Scale Detection Failure**:
   - Dacă nu se poate determina scala, workflow-ul se oprește
   - Eroare: `RuntimeError: Nu am putut determina scala!`

3. **Missing Files**:
   - Workflow-ul verifică existența fișierelor necesare
   - Folosește fallback-uri când este posibil

---

## Structură Directoare

```
output/{run_id}/
├── detections/
│   └── plan_*_cluster_*/
│       └── export_objects/
│           └── detections.json
├── scale/
│   └── plan_*_cluster_*/
│       └── cubicasa_steps/
│           ├── raster/
│           │   ├── response.json
│           │   ├── walls.png
│           │   ├── rooms.png
│           │   ├── doors.png
│           │   └── brute_force_best_config.json
│           └── raster_processing/
│               ├── walls_from_coords/
│               │   ├── 01_walls_from_coords.png
│               │   ├── 02_walls_thick.png
│               │   ├── 04_walls_3d.png
│               │   ├── 07_walls_interior.png
│               │   ├── 08_walls_exterior.png
│               │   ├── 11_interior_structure.png
│               │   ├── 12_exterior_structure.png
│               │   └── room_scales.json
│               └── openings/
│                   ├── 01_openings.png
│                   ├── 02_exterior_doors.png
│                   └── openings_measurements.json
├── count_objects/
├── measure_objects/
├── perimeter/
├── area/
├── roof/
├── pricing/
└── final_offer.json
```

---

## Note Tehnice

1. **Template Matching**: Folosit pentru clasificare preliminară a deschiderilor
2. **Gemini API**: Folosit pentru:
   - Clasificare deschideri
   - Estimare suprafață camere
   - Validare obiecte
3. **RasterScan API**: Folosit pentru vectorizare precisă
4. **CubiCasa Model**: Folosit pentru detectare pereți/camere
5. **Roboflow API**: Folosit pentru detectare obiecte (uși, ferestre, scări)

---

## Versiune

Documentație actualizată: 2026-01-26
Workflow Version: 2.0 (RasterScan-based)
