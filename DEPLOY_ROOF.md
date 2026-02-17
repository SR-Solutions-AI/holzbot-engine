# Deploy holzbot-roof (vizualizări 3D acoperiș)

Etapa **roof** rulează `holzbot-roof/scripts/clean_workflow.py` cu wall masks din
`scale/<plan_id>/cubicasa_steps/raster_processing/walls_from_coords/01_walls_from_coords.png`.

## Structură folder (obligatoriu pe VPS)

```
holzbot-dynamic/
├── holzbot-api/
├── holzbot-engine/   ← engine-ul
├── holzbot-roof/     ← TREBUIE să fie sibling de holzbot-engine
└── holzbot-web/
```

`holzbot-roof` trebuie în același folder părinte cu `holzbot-engine`. Engine-ul rulează
`python holzbot-roof/scripts/clean_workflow.py` cu cwd=holzbot-roof.

## ⚠️ IMPORTANT: _orig_pyc

`holzbot-roof/roof_calc/_orig_pyc/` trebuie să conțină fișiere `.pyc` compilate pentru versiunea
de Python folosită de engine (ex: `visualize.cpython-311.pyc`, `flood_fill.cpython-311.pyc`, etc.).

Dacă lipsește: `FileNotFoundError: Missing cached bytecode for visualize in ..._orig_pyc`

Dacă ai „bad magic number”: bytecode-ul din _orig_pyc e pentru altă versiune Python. Trebuie
pyc compilat cu aceeași versiune ca engine-ul (ex. Python 3.11).

## Dependențe

Engine-ul are în `requirements.txt`:
- plotly>=5.18.0
- pyvista>=0.43.0
- kaleido>=0.2.0

Dacă lipsesc dependențe din holzbot-roof:

```bash
pip install -r ../holzbot-roof/requirements.txt
```

## Input

- Un etaj: `01_walls_from_coords.png` din planul respectiv
- Mai multe etaje: se copiază toate măștile într-un temp dir și se rulează workflow pe folder

## Output

- `output/<run_id>/roof/roof_3d/rectangles/floor_X/` – măști dreptunghiuri
- `output/<run_id>/roof/roof_3d/roof_types/floor_X/{1_w,2_w,4_w,4.5_w}/` – lines.png, faces.png, frame.html, unfold/
- `output/<run_id>/roof/roof_3d/entire/{1_w,2_w,4_w,4.5_w}/` – frame.html, filled.html, filled.png (pentru LiveFeed), filled.png (pentru LiveFeed)
