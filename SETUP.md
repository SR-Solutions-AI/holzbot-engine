# Setup: venv și requirements

**Important:** Folosește **Python 3.11 sau 3.12**. Python 3.14 nu este suportat de unele dependențe (ex. `inference-sdk`, `torch`). Dacă ai mai multe versiuni instalate:

```bash
python3.12 -m venv .venv
# sau
python3.11 -m venv .venv
```

## 1. Intră în folderul engine

```bash
cd holzbot-engine
```

(sau `cd holzbot-dynamic/holzbot-engine` dacă ești în rădăcina proiectului)

## 2. Creează un virtual environment

```bash
python3.12 -m venv .venv
```

(sau `python3.11` dacă ai doar 3.11; **nu** folosi `python3` dacă e 3.14 – verifică cu `python3 --version`. Poți folosi și numele `venv` în loc de `.venv`.)

## 3. Activează venv-ul

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

După activare, promptul arată ceva de genul `(.venv) ...`.

## 4. Instalează dependențele

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Pentru mediu local (inclusiv Supabase, FastAPI etc.):
```bash
pip install -r requirements_local.txt
```

## 5. Variabile de mediu (opțional)

Creează fișier `.env` în `holzbot-engine` cu cheile necesare, de exemplu:

```env
GEMINI_API_KEY=...
RASTER_API_KEY=...
```

Sau exportează în shell înainte de a rula scripturile:

```bash
export GEMINI_API_KEY=...
export RASTER_API_KEY=...
```

## Comenzi rapide (rezumat)

```bash
cd holzbot-engine
python3.12 -m venv .venv    # sau python3.11 (necesar: Python < 3.13)
source .venv/bin/activate   # pe Windows: .venv\Scripts\activate
pip install --upgrade pip && pip install -r requirements.txt
```

## Troubleshooting

- **`No matching distribution found for inference-sdk==0.62.2`** sau **`ModuleNotFoundError: No module named 'torch'`**  
  Venv-ul a fost creat cu **Python 3.14**. Șterge `.venv` și recreează cu Python 3.12 (sau 3.11):

  ```bash
  rm -rf .venv
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip && pip install -r requirements.txt
  ```

- Dacă nu ai `python3.12`: instalează-l (e.g. `brew install python@3.12` pe macOS) sau folosește [pyenv](https://github.com/pyenv/pyenv). După `brew install python@3.12`, pe Mac folosește calea completă dacă nu e în PATH: `/opt/homebrew/bin/python3.12 -m venv .venv`

## Dezactivare venv

Când ai terminat:
```bash
deactivate
```
