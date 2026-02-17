# DOCUMENTAȚIE: Sursa Datelor pentru Calculul Prețului Ofertei

## 📋 Prezentare Generală

Acest document descrie **tot ce conține scriptul care construiește prețul ofertei** (`pricing/jobs.py` și `pricing/calculator.py`) și **de unde este luat fiecare lucru în parte**.

**IMPORTANT:** După modificările recente, **NU mai folosim CubiCasa pentru datele de pricing**. Toate datele vin **EXCLUSIV din RasterScan** (API-ul de vectorizare).

---

## 🔄 Workflow-ul de Date

### 1. **CubiCasa** (DOAR pentru segmentare)
- **Folosit DOAR la început** pentru a împărți planul în clustere
- **NU mai este folosit deloc** pentru datele de pricing
- **NU mai există fallback** la CubiCasa în pricing

### 2. **RasterScan API** (Sursa principală pentru TOATE datele)
- **Vectorizare plan**: Detectează camere, pereți, deschideri
- **Toate datele pentru pricing** vin din RasterScan

---

## 📊 Datele Folosite în Pricing

### **A. ARIE (Suprafață)**

#### **Sursă:** `scale/{plan_id}/cubicasa_steps/raster_processing/walls_from_coords/room_scales.json`

**Structură:**
```json
{
  "total_area_m2": 86.52,
  "rooms": [
    {
      "room_id": 0,
      "area_m2": 7.58,
      "area_pixels": 121110,
      "meters_per_pixel": 0.007977
    },
    ...
  ]
}
```

**Folosit pentru:**
- `area_net_m2`: Suprafața netă totală (din `total_area_m2`)
- `area_gross_m2`: Suprafața brută totală (din `total_area_m2`)
- Calculul ariilor pentru: fundație, podea, tavan, acoperiș

**În cod:**
- `pricing/jobs.py` linia 139-214: Citește `room_scales.json`
- `area/calculator.py`: Construiește structura completă `area_data` folosind `total_area_m2`

---

### **B. PEREȚI (Lungimi și Arii)**

#### **Sursă:** `scale/{plan_id}/cubicasa_steps/raster_processing/walls_from_coords/walls_measurements.json`

**Structură:**
```json
{
  "estimations": {
    "average_result": {
      "interior_meters": 45.23,
      "exterior_meters": 38.15,
      "interior_meters_structure": 42.10
    }
  }
}
```

**Calculat din:**
- **Pereți exteriori (outline)**: Număr pixeli din `08_walls_exterior.png` × `meters_per_pixel`
- **Pereți interiori (outline)**: Număr pixeli din `07_walls_interior.png` × `meters_per_pixel`
- **Pereți interiori (skeleton)**: Număr pixeli din `11_interior_structure.png` × `meters_per_pixel`

**Folosit pentru:**
- **Lungimi pereți**: `interior_meters`, `exterior_meters`, `interior_meters_structure`
- **Arii pereți**: Lungime × Înălțime pereți (din formular: 2.50m, 2.70m, 2.85m, sau înălțime mansardă)
- **Calculul costurilor**: Structură pereți (interior + exterior) și finisaje (interior + exterior)

**În cod:**
- `cubicasa_detector/detector.py` linia 3354-3360: Calculează `walls_ext_m`, `walls_int_m`, `walls_skeleton_structure_int_m`
- `cubicasa_detector/detector.py` linia 3361-3372: **Salvează `walls_measurements.json`** (NOU - fără dependență de CubiCasa)
- `pricing/jobs.py` linia 145-175: Citește `walls_measurements.json` (ELIMINAT `cubicasa_result.json`)
- `area/calculator.py` linia 36-75: Construiește `walls_data` folosind lungimile din `walls_measurements`

---

### **C. DESCHIDERI (Uși și Ferestre)**

#### **Sursă:** `scale/{plan_id}/cubicasa_steps/raster_processing/walls_from_coords/openings_measurements.json`

**Structură:**
```json
{
  "openings": [
    {
      "id": 0,
      "type": "window",
      "width_m": 3.318,
      "status": "exterior"
    },
    {
      "id": 1,
      "type": "door",
      "width_m": 0.822,
      "status": "interior"
    },
    ...
  ]
}
```

**Detectat de:**
- **RasterScan API**: Detectează automat deschiderile (uși și ferestre) din plan
- **Clasificare tip**: Folosește Gemini AI pentru a determina dacă este "door" sau "window"
- **Clasificare status**: Determină dacă este "exterior" sau "interior" bazat pe poziție

**Folosit pentru:**
- **Număr deschideri**: Numără uși interioare, uși exterioare, ferestre
- **Arii deschideri**: `width_m × height_m` (înălțimea vine din formular)
- **Scădere din pereți**: Ariile deschiderilor se scad din ariile brute ale pereților
- **Calculul costurilor**: Preț per m² pentru fiecare tip de deschidere

**În cod:**
- `cubicasa_detector/raster_processing.py`: Procesează deschiderile din RasterScan
- `pricing/jobs.py` linia 98-135: Citește `openings_measurements.json` și normalizează formatul
- `pricing/modules/openings.py`: Calculează costurile pentru fiecare deschidere

---

### **D. TIP ETAJ (Ground Floor / Top Floor / Intermediate)**

#### **Sursă:** `jobs/{run_id}/plan_metadata/{plan_name}.json`

**Structură:**
```json
{
  "floor_classification": {
    "floor_type": "ground_floor",
    "confidence": "high"
  }
}
```

**Folosit pentru:**
- **Determinarea finisajelor**: `finisajInterior_ground`, `fatada_ground`, `finisajInterior_floor_1`, etc.
- **Calculul fundației**: Doar pentru `ground_floor`
- **Calculul acoperișului**: Doar pentru `top_floor`
- **Indexarea etajelor**: Pentru a folosi finisajele corecte per etaj

**În cod:**
- `pricing/jobs.py` linia 176-193: Citește `floor_type` din `plan_metadata`
- `pricing/calculator.py` linia 169-221: Determină cheile de finisaje bazat pe `floor_type`

---

### **E. ACOPERIȘ (Roof)**

#### **Sursă:** `output/{run_id}/roof/{plan_id}/roof_estimation.json`

**Structură:**
```json
{
  "roof_final_total_eur": 20860.88,
  "components": {
    "roof_base": {...},
    "sheet_metal": {...},
    "material": {...},
    ...
  },
  "inputs": {
    "house_area_m2": 116.8,
    "perimeter_m": 38.15
  }
}
```

**Calculat în:**
- **STEP 13: Roof** (`roof/jobs.py`)
- **Folosește suprafața din RasterScan**: `room_scales.json` → `total_area_m2`
- **Folosește perimetrul din RasterScan**: `walls_measurements.json` → `exterior_meters`

**Folosit pentru:**
- **Cost total acoperiș**: Include structură, tinichigerie, izolație, învelitoare
- **Breakdown detaliat**: Fiecare componentă cu cantitate și preț

**În cod:**
- `roof/jobs.py`: Calculează costurile acoperișului
- `pricing/jobs.py` linia 246-247: Citește `roof_estimation.json`
- `pricing/modules/roof.py`: Transformă datele din roof în format pentru pricing

---

### **F. PREFERINȚE UTILIZATOR (Frontend Data)**

#### **Sursă:** `jobs/{run_id}/frontend_data.json`

**Conține:**
- **Sistem constructiv**: `sistemConstructiv.tipSistem` (CLT, HOLZRAHMEN, MASSIVHOLZ)
- **Acces șantier**: `sistemConstructiv.accesSantier` (Leicht, Mittel, Schwierig) – factor pe structura totală
- **Teren**: `sistemConstructiv.teren` (Eben, Leichte Hanglage, Starke Hanglage) – factor pe structura totală
- **Tip fundație**: `sistemConstructiv.tipFundatie` (Placă, Piloți, Soclu)
- **Finisaje interioare**: `materialeFinisaj.finisajInterior_ground`, `finisajInterior_floor_1`, etc.
- **Finisaje exterioare**: `materialeFinisaj.fatada_ground`, `fatada_floor_1`, etc.
- **Înălțime etaje**: `sistemConstructiv.inaltimeEtaje` (2.50m, 2.70m, 2.85m)
- **Înălțime pereți mansardă**: `sistemConstructiv.inaltimePeretiMansarda`
- **Înălțime ferestre**: `ferestreUsi.bodentiefeFenster` (doar pentru calculul ariei)
- **Înălțime uși**: `ferestreUsi.turhohe` (doar pentru calculul ariei)
- **Fensterart**: `ferestreUsi.windowQuality` (2-fach, 3-fach, 3-fach Passiv) – determină prețul €/m² ferestre
- **Performanță energetică**: `performanta.nivelEnergetic` (Standard, KfW 55, KfW 40, KfW 40+)
- **Tip încălzire**: `performanta.tipIncalzire` (Gaz, Pompa de căldură, Electric)
- **Ventilație**: `performanta.ventilatie` (True/False)
- **Tip semineu**: `performanta.tipSemineu` (Klassischer Holzofen, Moderner Design-Kaminofen, etc.)

**Folosit pentru:**
- **Coeficienți de preț**: Determină prețurile unitare pentru fiecare componentă
- **Modificatori**: Aplică multiplicatori bazat pe preferințe (prefabricare, performanță energetică, etc.)
- **Calculul final**: Toate costurile sunt calculate folosind aceste preferințe

**În cod:**
- `pricing/jobs.py` linia 308-317: Încarcă `frontend_data.json`
- `pricing/calculator.py` linia 59-252: Extrage și normalizează toate preferințele
- `pricing/modules/*`: Fiecare modul folosește preferințele relevante

---

## 🧮 Modulele de Calcul

### **1. STRUCTURĂ PEREȚI** (`pricing/modules/walls.py`)

**Input:**
- `area_int_net`: Aria netă pereți interiori (m²) - **din `area_data.walls.interior.net_area_m2_structure`**
- `area_ext_net`: Aria netă pereți exteriori (m²) - **din `area_data.walls.exterior.net_area_m2`**
- `system`: Sistem constructiv (CLT, HOLZRAHMEN, MASSIVHOLZ) - **din `frontend_data.sistemConstructiv.tipSistem`**

**Calcul:**
- Preț unitar: `pricing_coeffs.system.base_unit_prices[system][interior/exterior]` (fără modificator prefabricare)
- Cost pereți: `area × preț_unit`. **Acces șantier** și **teren** se aplică ulterior pe **întreaga structură** (fundație + pereți + planșeu + acoperiș) în `calculator.py`.

**Output:**
- `total_cost`: Cost total structură pereți
- `detailed_items`: 2 items (interior + exterior)

---

### **2. FINISAJE** (`pricing/modules/finishes.py`)

**Input:**
- `area_int_net`: Aria netă pereți interiori pentru finisaje (m²) - **din `area_data.walls.interior.net_area_m2`**
- `area_ext_net`: Aria netă pereți exteriori (m²) - **din `area_data.walls.exterior.net_area_m2`**
- `type_int`: Tip finisaj interior (Tencuială, Lemn, Fibrociment, etc.) - **din `frontend_data.materialeFinisaj.finisajInterior_ground/floor_X`**
- `type_ext`: Tip finisaj exterior (Tencuială, Mix, Lemn Ars, etc.) - **din `frontend_data.materialeFinisaj.fatada_ground/floor_X`**

**Calcul:**
- Preț unitar: `pricing_coeffs.finishes.interior[type_int]` și `pricing_coeffs.finishes.exterior[type_ext]`
- Cost final: `area × preț_unit`

**Output:**
- `total_cost`: Cost total finisaje
- `detailed_items`: 2 items (interior + exterior)

---

### **3. FUNDAȚIE** (`pricing/modules/foundation.py`)

**Input:**
- `foundation_area_m2`: Suprafața fundației (m²) - **din `area_data.surfaces.foundation_m2`**
- `type_foundation`: Tip fundație (Placă, Piloți, Soclu) - **din `frontend_data.sistemConstructiv.tipFundatie`**

**Calcul:**
- Preț unitar: `pricing_coeffs.foundation.unit_price_per_m2[type_foundation]`
- Cost final: `area × preț_unit`

**Output:**
- `total_cost`: Cost total fundație
- `detailed_items`: 1 item

---

### **4. DESCHIDERI** (`pricing/modules/openings.py`)

**Input:**
- `openings_list`: Lista deschiderilor - **din `openings_measurements.json` (RasterScan)**
- `frontend_data`: Pentru înălțimi (doar arie) și Fensterart - **din `frontend_data.json`**

**Structură deschidere:**
```json
{
  "type": "door" | "window" | "double_door" | "double_window",
  "width_m": 0.822,
  "status": "interior" | "exterior"
}
```

**Calcul:**
- **Înălțime ferestre**: Din `frontend_data.ferestreUsi.bodentiefeFenster` – folosit **doar pentru arie** (width × height).
- **Înălțime uși**: Din `frontend_data.ferestreUsi.turhohe` – folosit **doar pentru arie**.
- **Arie**: `width_m × height_m`.
- **Uși**: preț €/m² din `door_interior_price_per_m2` sau `door_exterior_price_per_m2` (după status).
- **Ferestre**: preț €/m² din `windows_price_per_m2` (2-fach / 3-fach / 3-fach Passiv) conform `windowQuality`.
- **Cost final**: `arie × preț_per_m²` (fără material tâmplărie, fără modificator calitate suplimentar).

**Output:**
- `total_cost`: Cost total deschideri
- `detailed_items`: Lista cu toate deschiderile (fiecare cu cost individual; label tip: Interior/Exterior sau tip geam)

---

### **5. PODEA/TAVAN** (`pricing/modules/floors.py`)

**Input:**
- `floor_area`: Suprafața podelei (m²) - **din `area_data.surfaces.floor_m2`**
- `ceiling_area`: Suprafața tavanului (m²) - **din `area_data.surfaces.ceiling_m2`**

**Calcul:**
- Preț unitar podea: `pricing_coeffs.area.floor_coefficient_per_m2`
- Preț unitar tavan: `pricing_coeffs.area.ceiling_coefficient_per_m2`
- Cost final: `area × preț_unit`

**Output:**
- `total_cost`: Cost total podea + tavan
- `detailed_items`: 2 items (podea + tavan)

---

### **6. ACOPERIȘ** (`pricing/modules/roof.py`)

**Input:**
- `roof_result_data`: Datele complete din `roof_estimation.json`

**Conține:**
- `roof_final_total_eur`: Cost total acoperiș
- `components`: Breakdown pe componente (structură, tinichigerie, izolație, învelitoare)
- `inputs`: Suprafață și perimetru folosite

**Calcul:**
- **Nu se recalculează în pricing** - se folosește direct costul din `roof_estimation.json`
- Transformă formatul din roof în format pentru pricing

**Output:**
- `total_cost`: Cost total acoperiș (din roof)
- `detailed_items`: Lista componentelor acoperișului

---

### **7. UTILITĂȚI** (`pricing/modules/utilities.py`)

**Input:**
- `total_floor_area_m2`: Suprafața totală a tuturor etajelor (m²) - **din `area_data.surfaces.floor_m2` (sumat pentru toate etajele)**
- `energy_level`: Nivel energetic (Standard, KfW 55, KfW 40, KfW 40+) - **din `frontend_data.performanta.nivelEnergetic`**
- `heating_type`: Tip încălzire (Gaz, Pompa de căldură, Electric) - **din `frontend_data.performanta.tipIncalzire`**
- `has_ventilation`: Dacă are ventilație (True/False) - **din `frontend_data.performanta.ventilatie`**

**Calcul:**
- **Electricitate**: `area × coeff_electricity × modifier_energetic`
- **Încălzire**: `area × coeff_heating × modifier_tip × modifier_energetic`
- **Ventilație**: `area × coeff_ventilation` (dacă `has_ventilation == True`)
- **Canalizare**: `area × coeff_sewage` (implicit inclus)

**Output:**
- `total_cost`: Cost total utilități
- `detailed_items`: 4 items (electricitate, încălzire, ventilație, canalizare)

---

### **8. SEMINEU** (`pricing/modules/utilities.py` - `calculate_fireplace_details`)

**Input:**
- `fireplace_type`: Tip semineu - **din `frontend_data.performanta.tipSemineu` sau `frontend_data.incalzire.tipSemineu`**
- `total_floors`: Număr total etaje - **din numărul de planuri procesate**

**Calcul:**
- **Cost semineu**: Preț fix bazat pe tip (8500€ - 18000€)
- **Cost horn (coș)**: 4500€ bază + 1500€ per etaj

**Output:**
- `total_cost`: Cost total semineu + horn
- `detailed_items`: 2 items (semineu + horn)

---

### **9. SCĂRI** (`pricing/modules/stairs.py`)

**Input:**
- `total_floors`: Număr total etaje - **din numărul de planuri procesate**

**Calcul:**
- Număr scări: `total_floors - 1` (1 etaj = 0 scări, 2 etaje = 1 scară, etc.)
- Preț per scară: `pricing_coeffs.stairs.price_per_stair_unit + pricing_coeffs.stairs.railing_price_per_stair`
- Cost final: `num_scări × preț_per_scară`

**Output:**
- `total_cost`: Cost total scări
- `detailed_items`: 2 items (structură scară + balustradă)

---

## 📁 Fișiere JSON Folosite

### **1. `room_scales.json`** (RasterScan)
**Locație:** `scale/{plan_id}/cubicasa_steps/raster_processing/walls_from_coords/room_scales.json`

**Conține:**
- `total_area_m2`: Suprafața totală netă (m²)
- `rooms[]`: Lista camerelor cu arii individuale

**Folosit pentru:**
- `area_net_m2` și `area_gross_m2` în `area_data`

---

### **2. `walls_measurements.json`** (RasterScan) ⭐ **NOU - FĂRĂ CubiCasa**
**Locație:** `scale/{plan_id}/cubicasa_steps/raster_processing/walls_from_coords/walls_measurements.json`

**Conține:**
- `estimations.average_result.interior_meters`: Lungime pereți interiori (m)
- `estimations.average_result.exterior_meters`: Lungime pereți exteriori (m)
- `estimations.average_result.interior_meters_structure`: Lungime pereți interiori pentru structură (m)

**Folosit pentru:**
- Calculul ariilor pereți (lungime × înălțime)
- Costurile pentru structură și finisaje

---

### **3. `openings_measurements.json`** (RasterScan)
**Locație:** `scale/{plan_id}/cubicasa_steps/raster_processing/walls_from_coords/openings_measurements.json`

**Conține:**
- `openings[]`: Lista deschiderilor cu `type`, `width_m`, `status`

**Folosit pentru:**
- Numărarea și calculul costurilor pentru uși și ferestre
- Scăderea ariilor deschiderilor din ariile pereților

---

### **4. `roof_estimation.json`** (Roof Module)
**Locație:** `output/{run_id}/roof/{plan_id}/roof_estimation.json`

**Conține:**
- `roof_final_total_eur`: Cost total acoperiș
- `components`: Breakdown pe componente
- `inputs`: Suprafață și perimetru folosite

**Folosit pentru:**
- Costul total al acoperișului (nu se recalculează în pricing)

---

### **5. `frontend_data.json`** (User Preferences)
**Locație:** `jobs/{run_id}/frontend_data.json`

**Conține:**
- Toate preferințele utilizatorului (sistem constructiv, finisaje, performanță, etc.)

**Folosit pentru:**
- Determinarea coeficienților de preț
- Modificatori pentru prefabricare, performanță energetică, etc.

---

### **6. `plan_metadata/{plan_name}.json`** (Floor Classification)
**Locație:** `jobs/{run_id}/plan_metadata/{plan_name}.json`

**Conține:**
- `floor_classification.floor_type`: Tip etaj (ground_floor, top_floor, intermediate)

**Folosit pentru:**
- Determinarea finisajelor per etaj
- Calculul fundației (doar ground_floor)
- Calculul acoperișului (doar top_floor)

---

## 🚫 Date ELIMINATE (Nu mai folosite)

### **❌ `cubicasa_result.json`**
- **ELIMINAT COMPLET** din pricing
- **NU mai există fallback** la CubiCasa
- **NU mai citim** `cubicasa_result.json` în `pricing/jobs.py`

### **❌ `areas_calculated.json`** (din pasul Area)
- **ELIMINAT** - nu mai este necesar
- **Folosim direct** `room_scales.json` din RasterScan

### **❌ `openings_all.json`** (din pasul Measure Objects)
- **ELIMINAT** - nu mai este necesar
- **Folosim direct** `openings_measurements.json` din RasterScan

---

## 🔧 Modificări Recente

### **1. Salvarea `walls_measurements.json`** (NOU)
- **Locație:** `cubicasa_detector/detector.py` linia 3361-3372
- **Salvează** walls_measurements într-un fișier separat **FĂRĂ dependență de CubiCasa**
- **Format:** Identic cu formatul vechi din `cubicasa_result.json`, dar salvat separat

### **2. Eliminarea dependenței de `cubicasa_result.json`** (NOU)
- **Locație:** `pricing/jobs.py` linia 91, 146-175
- **ELIMINAT** citirea din `cubicasa_result.json`
- **FOLOSIM DOAR** `walls_measurements.json` din RasterScan

### **3. Prioritate RasterScan** (NOU)
- **Toate datele** vin **EXCLUSIV din RasterScan**
- **NU mai există fallback** la metode vechi
- **NU mai folosim** CubiCasa pentru datele de pricing

---

## 📝 Rezumat: De Unde Vine Fiecare Dată

| **Dată** | **Sursă** | **Fișier** | **Folosit Pentru** |
|----------|-----------|------------|-------------------|
| **Arie netă/brută** | RasterScan | `room_scales.json` | Fundație, podea, tavan, acoperiș |
| **Lungimi pereți** | RasterScan | `walls_measurements.json` | Arii pereți (structură + finisaje) |
| **Deschideri** | RasterScan | `openings_measurements.json` | Costuri uși/ferestre, scădere din pereți |
| **Tip etaj** | Floor Classification | `plan_metadata/*.json` | Finisaje per etaj, fundație, acoperiș |
| **Cost acoperiș** | Roof Module | `roof_estimation.json` | Cost total acoperiș |
| **Preferințe utilizator** | Frontend | `frontend_data.json` | Coeficienți, modificatori, înălțimi |
| **Coeficienți preț** | Database | `pricing_parameters` (Supabase) | Prețuri unitare pentru toate componentele |

---

## ✅ Verificare: Toate Datele din RasterScan

**DA** - Toate datele necesare pentru pricing vin **EXCLUSIV din RasterScan**:
- ✅ Arie: `room_scales.json`
- ✅ Pereți: `walls_measurements.json` (NOU - salvat separat)
- ✅ Deschideri: `openings_measurements.json`
- ✅ Acoperiș: Folosește arie și perimetru din RasterScan

**NU mai folosim:**
- ❌ `cubicasa_result.json` (ELIMINAT)
- ❌ `areas_calculated.json` (ELIMINAT)
- ❌ `openings_all.json` (ELIMINAT)

---

## 🎯 Concluzie

**Scriptul de pricing (`pricing/jobs.py` + `pricing/calculator.py`) construiește prețul ofertei folosind:**

1. **Date din RasterScan** (arie, pereți, deschideri)
2. **Date din Roof Module** (cost acoperiș)
3. **Preferințe utilizator** (sistem, finisaje, performanță)
4. **Coeficienți din Database** (prețuri unitare)

**NU mai există dependență de CubiCasa pentru datele de pricing!**
