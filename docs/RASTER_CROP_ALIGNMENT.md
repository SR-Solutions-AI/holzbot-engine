# Raster crop și aliniere

## Ce știm despre crop-ul Raster

- **API-ul Raster nu expune** niciun câmp de crop/bbox în răspuns (`response.json` conține doar `data.area`, `data.doors`, `data.rooms`, `data.walls`, `data.image`, etc.).
- **Noi trimitem** întreaga imagine (doar scale-down la max 1000px pe latura lungă), fără crop în prealabil.
- Dacă **Raster face crop intern** (ex. elimină zone din jurul casei), imaginea returnată în `data.image` poate avea **dimensiuni diferite** față de imaginea trimisă.

## Detectare crop

După fiecare apel Raster reușit:

1. **Comparăm** dimensiunile imaginii returnate (`raster_response.png`) cu dimensiunile imaginii trimise (`request_w` x `request_h`).
2. Dacă **dimensiunile diferă**, în `raster_request_info.json` se adaugă:
   - `response_image_w`, `response_image_h`
   - `raster_may_crop: true`
3. În consolă apare avertismentul: *„Raster a returnat imagine WxH (request: W'xH') – posibil crop intern; alinierea pe original poate fi incorectă.”*

## Cum ne asigurăm că alinierea este corectă

### 1. Overlay pe imaginea trimisă (mereu corect)

- **overlay_on_request.png** – pereții/camerele/ușile din `response.json` sunt desenați pe `raster_request.png` la coordonate 1:1.
- Coordonatele din JSON sunt în spațiul imaginii trimise; dacă Raster nu schimbă cadrul, acest overlay este corect.

### 2. Mască pereți în spațiul request (fără dependență de imaginea returnată)

- **api_walls_from_json.png** – mască binară (pereți) construită din `data.walls` în același spațiu ca `raster_request.png` (request_w x request_h).
- Poate fi folosită pentru aliniere în locul măştii extrase din imaginea returnată de Raster, astfel încât transformarea la original să rămână doar **scale_factor** (fără offset de crop necunoscut).

### 3. Când Raster face crop

- **Masca din imaginea returnată** (`api_walls_mask` din `raster_response.png`) este în spațiul imaginii cropate; nu știm offset-ul de crop, deci maparea la original doar cu `scale_factor` poate fi greșită.
- **Recomandare:** când `raster_request_info.json` conține `raster_may_crop: true`, pentru aliniere se poate folosi **api_walls_from_json.png** (redimensionată la original cu `scale_factor`) în locul măştii din imaginea returnată.
- **Alternativ:** pre-crop pe planul nostru (ex. detectare clădire + crop la bounding box) înainte de trimitere la Raster, astfel încât „cadrul” trimis să fie deja același cu zona de interes și să nu mai existe discrepanță de crop.

### 4. Cerințe față de Raster (opțional)

Pentru aliniere perfectă când Raster face crop, ar fi nevoie de un câmp în răspuns, de tip:

- `crop_bbox: [x_min, y_min, x_max, y_max]` în coordonatele imaginii trimise, sau  
- `crop_bbox_original: [...]` în coordonatele imaginii originale,

astfel încât să putem transforma coordonatele din răspuns în spațiul imaginii originale.
