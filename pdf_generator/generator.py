from __future__ import annotations
import json
import io
import os
import re
from pathlib import Path
from datetime import datetime
import random

# --- NEW: OpenAI Import ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen.canvas import Canvas

from PIL import Image as PILImage, ImageEnhance, ImageOps

from config.settings import load_plan_infos, PlansListError, RUNNER_ROOT, PROJECT_ROOT
from config.frontend_loader import load_frontend_data_for_run

# ---------- 1. DICTIONAR STATIC EXTINS (CONSTANTE) ----------
# Acestea au prioritate MAXIMĂ peste AI pentru consistență și viteză.
STATIC_TRANSLATIONS = {
    # Unități
    "buc": "Stk.",
    "buc.": "Stk.",
    "bucata": "Stk.",
    "piece": "Stk.",
    "pieces": "Stk.",
    "mp": "m²",
    "m2": "m²",
    "ml": "m",
    "kg": "kg",
    "tone": "t",
    "ora": "Std.",
    "manopera": "Arbeitsleistung",
    
    # Structură & Nivele
    "Parter": "Erdgeschoss",
    "Ground Floor": "Erdgeschoss",
    "ground_floor": "Erdgeschoss",
    "Etaj": "Obergeschoss",
    "top_floor": "Obergeschoss / Dachgeschoss",
    "Etaj 1": "1. Obergeschoss",
    "Mansarda": "Dachgeschoss",
    "Acoperis": "Dach",
    "Fundație": "Fundament",
    "Placa": "Bodenplatte",
    
    # Elemente Constructive
    "Pereti": "Wände",
    "Pereti Exteriori": "Außenwände",
    "Pereti Interiori": "Innenwände",
    "Planseu": "Geschossdecke",
    "Grinda": "Holzbalken",
    "Stalp": "Stütze",
    "Scara": "Treppe",
    "Balustrada": "Geländer",
    
    # Materiale
    "Beton": "Beton",
    "Lemn": "Holz",
    "Caramida": "Ziegel",
    "Fier": "Stahl",
    "Vata": "Dämmung",
    "Rigips": "Gipskarton",
    
    # Categorii Oferta
    "Structură": "Rohbau / Konstruktion",
    "Arhitectura": "Architektur",
    "Instalatii": "Haustechnik",
    "Finisaje": "Ausbau & Oberflächen",
    "Casă completă": "Schlüsselfertig",
    "La rosu": "Rohbau",
    
    # Utilități
    "Electrice": "Elektroinstallation",
    "Sanitare": "Sanitärinstallation",
    "Termice": "Heizungstechnik",
    "Canalizare": "Abwasser",
    
    # Deschideri
    "Fereastra": "Fenster",
    "Usa": "Tür",
    "Usa intrare": "Haustür",
    
    # Diverse
    "Total": "Gesamt",
    "Pret": "Preis",
    "Cantitate": "Menge",
    "Descriere": "Beschreibung"
}

# ---------- 2. AI TRANSLATION SERVICE (AGRESIV) ----------
class GermanEnforcer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if OpenAI and api_key else None
        self.cache = STATIC_TRANSLATIONS.copy()
        self.pending_texts = set()

    def is_translatable(self, text):
        """
        Verifică dacă textul merită tradus.
        Ignoră numere, șiruri goale sau texte foarte scurte care par a fi coduri.
        """
        if not text or not isinstance(text, str):
            return False
        # Elimină numere simple "123.45", "100", "12,50"
        if re.match(r'^[\d\.,\s€%]+$', text.strip()):
            return False
        # Elimină chei interne scurte
        if len(text) < 2:
            return False
        # Dacă e deja în cache (adică e în STATIC_TRANSLATIONS), nu mai cerem la AI
        if text in self.cache:
            return False
        return True

    def collect(self, data):
        """
        Navighează recursiv prin orice structură JSON (dict/list) 
        și colectează TOT ce pare a fi text relevant.
        """
        if isinstance(data, dict):
            for k, v in data.items():
                # Colectăm valorile doar pentru cheile care conțin text vizibil utilizatorului
                if k in ["name", "details", "category", "unit", "type", "location", "material_user", "roof_type_user", "floor_type"]:
                    if self.is_translatable(v):
                        self.pending_texts.add(v)
                # Recursivitate
                self.collect(v)
        elif isinstance(data, list):
            for item in data:
                self.collect(item)

    def process_translation_queue(self):
        """Trimite tot ce s-a colectat la OpenAI într-un singur batch sau batch-uri mari."""
        if not self.client or not self.pending_texts:
            return

        text_list = list(self.pending_texts)
        print(f"🇩🇪 [GermanEnforcer] Translating {len(text_list)} unique terms via AI...")
        
        batch_size = 50
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            
            prompt = (
                "You are a professional technical translator for the German construction industry (Bauwesen). "
                "Translate the following JSON list of terms from Romanian or English to German. "
                "Keep technical precision (e.g. 'Parter' -> 'Erdgeschoss', 'Bucata' -> 'Stk.', 'Manopera' -> 'Montage/Arbeitsleistung'). "
                "Output ONLY a valid JSON object where keys are the input text and values are the German translation."
            )

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(batch)}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                translations = json.loads(content)
                self.cache.update(translations)
            except Exception as e:
                print(f"⚠️ [GermanEnforcer] AI Batch Error: {e}")
        
        # Curățăm coada după procesare
        self.pending_texts.clear()

    def get(self, text):
        """Returnează traducerea din cache. Dacă nu există, returnează originalul."""
        if not text or not isinstance(text, str): 
            return text
        return self.cache.get(text, text)

    def deep_apply(self, data):
        """
        Modifică structura de date IN-PLACE sau returnează o copie modificată,
        înlocuind textele cu traducerea lor din cache.
        """
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                # Dacă cheia e una de text, traducem valoarea
                if k in ["name", "details", "category", "unit", "type", "location", "floor_type"]:
                    if isinstance(v, str):
                        new_data[k] = self.get(v)
                    else:
                        new_data[k] = self.deep_apply(v)
                else:
                    new_data[k] = self.deep_apply(v)
            return new_data
        elif isinstance(data, list):
            return [self.deep_apply(item) for item in data]
        else:
            return data

# ---------- FONTS & ASSETS ----------
FONTS_DIR = Path(__file__).parent.parent / "pdf_assets" / "fonts"
FONT_REG = FONTS_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONTS_DIR / "DejaVuSans-Bold.ttf"
BASE_FONT, BOLD_FONT = "DejaVuSans", "DejaVuSans-Bold"

try:
    if FONT_REG.exists() and FONT_BOLD.exists():
        pdfmetrics.registerFont(TTFont(BASE_FONT, str(FONT_REG)))
        pdfmetrics.registerFont(TTFont(BOLD_FONT, str(FONT_BOLD)))
except Exception:
    BASE_FONT, BOLD_FONT = "Helvetica", "Helvetica-Bold"

COMPANY = {
    "name": "Chiemgauer Holzhaus",
    "legal": "LSP Holzbau GmbH & Co KG",
    "addr_lines": ["Seiboldsdorfer Mühle 1a", "83278 Traunstein"],
    "phone": "+49 (0) 861 / 166 192 0",
    "fax":   "+49 (0) 861 / 166 192 20",
    "email": "info@chiemgauer-holzhaus.de",
    "web":   "www.chiemgauer-holzhaus.de",
    "footer_left":  "Chiemgauer Holzhaus\nLSP Holzbau GmbH & Co KG\nRegistergericht Traunstein HRA Nr. 7311\nGeschäftsführer Bernhard Oeggl",
    "footer_mid":   "LSP Verwaltungs GmbH\nPersönlich haftende Gesellschafterin\nRegistergericht Traunstein HRB Nr. 13146",
    "footer_right": "Volksbank Raiffeisenbank Oberbayern Südost eG\nKto.Nr. 7 313 640  ·  BLZ 710 900 00\nIBAN: DE81 7109 0000 0007 3136 40   BIC: GENODEF1BGL   USt-ID: DE131544091",
}

IMG_IDENTITY = PROJECT_ROOT / "offer_identity.png"
IMG_LOGOS = PROJECT_ROOT / "offer_logos.png"

# ---------- LOGIC ----------
def _get_offer_inclusions(nivel_oferta: str) -> dict:
    INCLUSIONS = {
        "Structură": {
            "foundation": True, "structure_walls": True, "roof": True, "floors_ceilings": True,
            "openings": False, "finishes": False, "utilities": False
        },
        "Structură + ferestre": {
            "foundation": True, "structure_walls": True, "roof": True, "floors_ceilings": True,
            "openings": True, "finishes": False, "utilities": False
        },
        "Casă completă": {
            "foundation": True, "structure_walls": True, "roof": True, "floors_ceilings": True,
            "openings": True, "finishes": True, "utilities": True
        }
    }
    return INCLUSIONS.get(nivel_oferta, INCLUSIONS["Casă completă"])

# ---------- STYLES ----------
def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="H1", fontName=BOLD_FONT, fontSize=12, leading=22, spaceAfter=10, spaceBefore=6))
    s.add(ParagraphStyle(name="H2", fontName=BOLD_FONT, fontSize=11, leading=14, spaceBefore=12, spaceAfter=6))
    s.add(ParagraphStyle(name="Body", fontName=BASE_FONT, fontSize=10, leading=14, spaceAfter=4))
    s.add(ParagraphStyle(name="Small", fontName=BASE_FONT, fontSize=7.2, leading=9.2))
    s.add(ParagraphStyle(name="Disclaimer", fontName=BASE_FONT, fontSize=8.5, leading=11, textColor=colors.HexColor("#333333")))
    s.add(ParagraphStyle(name="Cell", fontName=BASE_FONT, fontSize=9.5, leading=12))
    s.add(ParagraphStyle(name="CellBold", fontName=BOLD_FONT, fontSize=9.5, leading=12))
    s.add(ParagraphStyle(name="CellSmall", fontName=BASE_FONT, fontSize=9, leading=11))
    return s

def P(text, style_name="Cell"):
    return Paragraph((str(text) or "").replace("\n", "<br/>"), _styles()[style_name])

def _money(x):
    try:
        v = float(x)
        s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} €"
    except: return "—"

def _fmt_m2(v):
    try:
        val = float(v)
        s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} m²"
    except: return "—"

def _fmt_qty(v, unit=""):
    try:
        val = float(v)
        s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} {unit}"
    except: return "—"

# ---------- PDF DRAWING ----------
def _draw_ribbon(canv: Canvas):
    canv.saveState()
    x, y = 18*mm, A4[1]-23*mm
    w, h = A4[0]-36*mm, 9*mm
    canv.setFillColor(colors.HexColor("#1c1c1c"))
    canv.rect(x, y, w, h, stroke=0, fill=1)
    canv.setFillColor(colors.white)
    canv.setFont(BOLD_FONT, 10)
    canv.drawString(x+6*mm, y+2.35*mm, "ANGEBOT – UNVERBINDLICHE KOSTENSCHÄTZUNG (RICHTWERT) ±10 %")
    canv.restoreState()

def _draw_footer(canv: Canvas):
    canv.saveState()
    y = 9*mm
    colw = (A4[0]-36*mm)/3.0
    x0 = 18*mm
    canv.setFont(BASE_FONT, 6.6)
    canv.setFillColor(colors.black)
    for i, block in enumerate((COMPANY["footer_left"], COMPANY["footer_mid"], COMPANY["footer_right"])):
        tx = x0 + i*colw
        lines = block.split("\n")
        for idx, line in enumerate(lines):
            canv.drawString(tx, y + (len(lines)-idx-1)*3.0*mm, line)
    canv.restoreState()

def _draw_firstpage_right_box(canv: Canvas, offer_no: str, handler: str):
    canv.saveState()
    box_x = A4[0]-18*mm-65*mm
    box_y = A4[1]-62*mm
    cw = 65*mm
    row_h = 8.2*mm
    rows = [("Datum", datetime.now().strftime("%d.%m.%Y")), ("Bearbeiter", handler), ("Kunden-Nr.", "—"), ("Auftrag", offer_no)]
    canv.setFont(BASE_FONT, 9)
    canv.setStrokeColor(colors.black)
    canv.rect(box_x, box_y - row_h*len(rows), cw, row_h*len(rows), stroke=1, fill=0)
    for i, (k, v) in enumerate(rows):
        y = box_y - (i+1)*row_h + 2.6*mm
        canv.drawString(box_x+3*mm, y, k)
        canv.drawRightString(box_x+cw-3*mm, y, v)
    canv.restoreState()

def _first_page_canvas(offer_no: str, handler: str):
    def _inner(canv: Canvas, doc):
        _draw_ribbon(canv)
        if IMG_IDENTITY.exists():
            canv.drawImage(str(IMG_IDENTITY), A4[0]-18*mm-85*mm, A4[1]-53*mm, 85*mm, 22*mm, preserveAspectRatio=True, mask='auto')
        if IMG_LOGOS.exists():
            canv.drawImage(str(IMG_LOGOS), 18*mm, A4[1]-55*mm, 80*mm, 26*mm, preserveAspectRatio=True, mask='auto', anchor='sw')
        _draw_firstpage_right_box(canv, offer_no, handler)
        _draw_footer(canv)
    return _inner

def _later_pages_canvas(canv: Canvas, doc):
    _draw_ribbon(canv)
    _draw_footer(canv)

def _header_block(story, styles, offer_no: str, client: dict):
    left_lines = [COMPANY["legal"], *COMPANY["addr_lines"], "", f"Tel. {COMPANY['phone']}", f"Fax {COMPANY['fax']}", "", COMPANY["email"], COMPANY["web"]]
    tbl = Table([[Paragraph("<br/>".join(left_lines), styles["Small"]), Paragraph("", styles["Small"])]], colWidths=[95*mm, A4[0]-36*mm-95*mm])
    tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(Spacer(1, 36*mm))
    story.append(tbl)
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(f"Angebot • Nr.: {offer_no}", styles["H1"]))
    story.append(Spacer(1, 3*mm))
    
    # Info Client
    name = (client.get("nume") or client.get("name") or "—").strip()
    city = (client.get("localitate") or client.get("city") or "—").strip()
    phone = (client.get("telefon") or client.get("phone") or "—").strip()
    email = (client.get("email") or "—").strip()
    proj = (client.get("referinta") or client.get("project_label") or "—").strip()
    lines = [f"<b>Bauherr / Kunde:</b> {name}", f"<b>Ort / Bauort:</b> {city}", f"<b>Telefon:</b> {phone}", f"<b>E-Mail:</b> {email}", f"<b>Bauvorhaben:</b> {proj}"]
    story.append(Paragraph("<br/>".join(lines), _styles()["Cell"]))
    story.append(Spacer(1, 6*mm))

def _intro(story, styles, client):
    story.append(Paragraph("Angebot für Ihr Chiemgauer Massivholzhaus", styles["H2"]))
    story.append(Paragraph("Sehr geehrte Damen und Herren,", styles["Body"]))
    story.append(Paragraph("vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung.", styles["Body"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("HINWEIS: Unverbindliche Kostenschätzung. Kein verbindliches Angebot.", styles["Disclaimer"]))
    story.append(Spacer(1, 6*mm))

# ---------- TABLES (NOW USING DEEP TRANSLATED DATA) ----------
# Aici datele vin deja traduse, deci nu mai apelăm translator.get()

def _table_standard(story, styles, title, data_dict):
    items = data_dict.get("items", []) or data_dict.get("detailed_items", [])
    if not items: return
    story.append(Paragraph(title, styles["H2"]))
    head = [P("Bauteil", "CellBold"), P("Fläche", "CellBold"), P("Preis/m²", "CellBold"), P("Gesamt", "CellBold")]
    data = []
    for it in items:
        # it['name'] este deja in germana datorita deep_apply
        data.append([P(it.get("name", "—")), P(_fmt_m2(it.get("area_m2", 0))), P(_money(it.get("unit_price", 0)), "CellSmall"), P(_money(it.get("cost", 0)), "CellBold")])
    data.append([P("SUMME", "CellBold"), "", "", P(_money(data_dict.get("total_cost", 0)), "CellBold")])
    tbl = Table([head] + data, colWidths=[75*mm, 40*mm, 32*mm, 38*mm])
    tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.3, colors.black), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,1), (-1,-1), "RIGHT")]))
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

def _table_roof_quantities(story, styles, pricing_data: dict):
    roof = pricing_data.get("breakdown", {}).get("roof", {})
    items = roof.get("items", []) or roof.get("detailed_items", [])
    display_items = [it for it in items if "extra_walls" not in str(it.get("category", ""))]
    if not display_items: return
    story.append(Paragraph("Dachkonstruktion – Detail", styles["H2"]))
    head = [P("Komponente", "CellBold"), P("Bemerkung", "CellBold"), P("Menge", "CellBold"), P("Preis", "CellBold")]
    data = []
    for it in display_items:
        data.append([P(it.get("name", "—")), P(it.get("details", ""), "CellSmall"), P(_fmt_qty(it.get("quantity", 0), it.get("unit", "")), "CellSmall"), P(_money(it.get("cost", 0)), "CellBold")])
    data.append([P("SUMME DACH", "CellBold"), "", "", P(_money(sum(it.get("cost", 0) for it in display_items)), "CellBold")])
    tbl = Table([head] + data, colWidths=[55*mm, 70*mm, 24*mm, 32*mm])
    tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.3, colors.black), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (2,1), (3,-1), "RIGHT")]))
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

def _table_stairs(story, styles, stairs_dict: dict):
    items = stairs_dict.get("detailed_items", [])
    if not items: return
    story.append(Paragraph("Treppenanlagen", styles["H2"]))
    head = [P("Bauteil", "CellBold"), P("Beschreibung", "CellBold"), P("Menge", "CellBold"), P("Gesamt", "CellBold")]
    data = []
    for it in items:
        data.append([P(it.get("name", "—")), P(it.get("details", "—"), "CellSmall"), P(f"{it.get('quantity',0)} {it.get('unit', 'Stk.')}", "CellSmall"), P(_money(it.get("cost", 0)), "CellBold")])
    data.append([P("SUMME TREPPEN", "CellBold"), "", "", P(_money(stairs_dict.get("total_cost", 0)), "CellBold")])
    tbl = Table([head] + data, colWidths=[70*mm, 55*mm, 25*mm, 31*mm])
    tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.3, colors.black), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (2,1), (3,-1), "RIGHT")]))
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

def _table_global_openings(story, styles, all_openings: list):
    if not all_openings: return
    story.append(PageBreak())
    story.append(Paragraph("Zusammenfassung Fenster & Türen", styles["H1"]))
    agg = {"windows": {"n": 0, "eur": 0.0}, "doors_int": {"n": 0, "eur": 0.0}, "doors_ext": {"n": 0, "eur": 0.0}}
    
    # Keyword-urile sunt acum în germană sau engleză
    keywords_window = ["fenster", "window", "glass"]
    keywords_door = ["tür", "door"]
    keywords_ext = ["aussen", "exterior", "entrance", "haustür"]
    
    for it in all_openings:
        full_text = (str(it.get("name", "")) + " " + str(it.get("type", "")) + " " + str(it.get("category", "")) + " " + str(it.get("location", ""))).lower()
        cost = float(it.get("total_cost", 0))
        if any(k in full_text for k in keywords_window):
            agg["windows"]["n"] += 1; agg["windows"]["eur"] += cost
        elif any(k in full_text for k in keywords_door):
            if any(k in full_text for k in keywords_ext): agg["doors_ext"]["n"] += 1; agg["doors_ext"]["eur"] += cost
            else: agg["doors_int"]["n"] += 1; agg["doors_int"]["eur"] += cost

    def avg(total, n): return total / n if n > 0 else 0.0
    head = [P("Kategorie", "CellBold"), P("Stückzahl", "CellBold"), P("Ø Preis/Stk.", "CellBold"), P("Gesamt", "CellBold")]
    data = []
    if agg["windows"]["n"]: data.append([P("Fensterelemente"), P(str(agg["windows"]["n"])), P(_money(avg(agg["windows"]["eur"], agg["windows"]["n"]))), P(_money(agg["windows"]["eur"]))])
    if agg["doors_ext"]["n"]: data.append([P("Außentüren / Hauseingang"), P(str(agg["doors_ext"]["n"])), P(_money(avg(agg["doors_ext"]["eur"], agg["doors_ext"]["n"]))), P(_money(agg["doors_ext"]["eur"]))])
    if agg["doors_int"]["n"]: data.append([P("Innentüren"), P(str(agg["doors_int"]["n"])), P(_money(avg(agg["doors_int"]["eur"], agg["doors_int"]["n"]))), P(_money(agg["doors_int"]["eur"]))])
    total_eur = sum(x["eur"] for x in agg.values())
    if total_eur > 0:
        data.append([P("SUMME ÖFFNUNGEN", "CellBold"), "", "", P(_money(total_eur), "CellBold")])
        tbl = Table([head] + data, colWidths=[68*mm, 26*mm, 34*mm, 40*mm])
        tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.3, colors.black), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), ("ALIGN", (1,1), (-1,-1), "RIGHT"), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
        story.append(tbl)
        story.append(Spacer(1, 8*mm))

def _table_global_utilities(story, styles, all_utilities: list):
    if not all_utilities: return
    story.append(Paragraph("Zusammenfassung Haustechnik & Installationen", styles["H1"]))
    agg = {}
    total_util = 0.0
    for it in all_utilities:
        cat = it.get("category", "Sonstiges")
        cost = it.get("total_cost", 0.0)
        total_util += cost
        agg[cat] = agg.get(cat, 0.0) + cost
    
    head = [P("Gewerk / Kategorie", "CellBold"), P("Gesamtpreis", "CellBold")]
    data = []
    for k, v in agg.items(): data.append([P(k), P(_money(v))])
    data.append([P("SUMME HAUSTECHNIK", "CellBold"), P(_money(total_util), "CellBold")])
    tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
    tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.3, colors.black), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), ("ALIGN", (1,1), (1,-1), "RIGHT"), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story.append(tbl)
    story.append(Spacer(1, 8*mm))

def _closing_blocks(story, styles):
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Annahmen & Vorbehalte", styles["H2"]))
    story.append(Paragraph("Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen.", styles["Body"]))
    story.append(Paragraph("Preisbindefrist: 4 Wochen.", styles["Body"]))

# ---------- MAIN GENERATOR ----------
def generate_complete_offer_pdf(run_id: str, output_path: Path | None = None) -> Path:
    print(f"🚀 [PDF] START: {run_id}")
    output_root = RUNNER_ROOT / "output" / run_id
    if not output_root.exists(): raise FileNotFoundError(f"Output nu există: {output_root}")
    if output_path is None:
        pdf_dir = output_root / "offer_pdf"; pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"oferta_{run_id}.pdf"

    frontend_data = load_frontend_data_for_run(run_id)
    client_data = frontend_data.get("client", frontend_data)
    nivel_oferta = frontend_data.get("materialeFinisaj", {}).get("nivelOferta", "Casă completă")
    inclusions = _get_offer_inclusions(nivel_oferta)
    
    # Load Plans
    try: plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError: plan_infos = []

    enriched_plans = []
    for plan in plan_infos:
        meta_path = output_root / "plan_metadata" / f"{plan.plan_id}.json"
        floor_type = "unknown"
        if meta_path.exists():
            try: floor_type = json.load(open(meta_path)).get("floor_classification", {}).get("floor_type", "unknown")
            except: pass
        if floor_type == "unknown":
            if "parter" in plan.plan_id.lower() or "ground" in plan.plan_id.lower(): floor_type = "ground_floor"
            elif "etaj" in plan.plan_id.lower() or "top" in plan.plan_id.lower(): floor_type = "top_floor"
        enriched_plans.append({"plan": plan, "floor_type": floor_type, "sort": 0 if floor_type == "ground_floor" else 1})
    enriched_plans.sort(key=lambda x: x["sort"])

    plans_data = []
    global_openings = []
    global_utilities = []

    for p_data in enriched_plans:
        plan = p_data["plan"]
        pricing_path = plan.stage_work_dir / "pricing_raw.json"
        if pricing_path.exists():
            with open(pricing_path, encoding="utf-8") as f: p_json = json.load(f)
            breakdown = p_json.get("breakdown", {})
            filtered_breakdown = {}
            filtered_total = 0.0
            for category_key, category_data in breakdown.items():
                if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                     filtered_breakdown[category_key] = category_data; filtered_total += category_data.get("total_cost", 0.0); continue
                if inclusions.get(category_key, False):
                    filtered_breakdown[category_key] = category_data; filtered_total += category_data.get("total_cost", 0.0)
            p_json["breakdown"] = filtered_breakdown
            p_json["total_cost_eur"] = filtered_total
            
            if inclusions.get("utilities", False): global_utilities.extend(breakdown.get("utilities", {}).get("items", []))
            if inclusions.get("openings", False): global_openings.extend(breakdown.get("openings", {}).get("items", []))
            if inclusions.get("roof", False):
                # Move extra walls logic (kept same as previous)
                roof_items = breakdown.get("roof", {}).get("items", [])
                extra_wall = next((it for it in roof_items if "extra_walls" in it.get("category", "")), None)
                if extra_wall and inclusions.get("structure_walls", False):
                    cost = extra_wall.get("cost", 0.0)
                    ws = filtered_breakdown.get("structure_walls", {})
                    target = next((it for it in ws.get("items", []) if "Außenwände" in it.get("name","") or "Exterior" in it.get("name","")), None)
                    if target: target["cost"] += cost; ws["total_cost"] += cost; filtered_breakdown.get("roof", {})["total_cost"] -= cost; filtered_total += cost

            plans_data.append({"info": plan, "type": p_data["floor_type"], "pricing": p_json})

    # --- 🔥 AGGRESSIVE GERMAN ENFORCEMENT 🔥 ---
    print("🇩🇪 [PDF] Starting Aggressive Translation...")
    enforcer = GermanEnforcer()
    
    # 1. Colectăm TOATE stringurile din TOATE structurile
    enforcer.collect(plans_data)
    enforcer.collect(global_openings)
    enforcer.collect(global_utilities)
    
    # 2. Traducem en-masse via OpenAI (un singur apel mare sau câteva batch-uri)
    enforcer.process_translation_queue()
    
    # 3. Aplicăm traducerile IN-PLACE (Rescriem structurile de date)
    plans_data = enforcer.deep_apply(plans_data)
    global_openings = enforcer.deep_apply(global_openings)
    global_utilities = enforcer.deep_apply(global_utilities)
    
    print("🇩🇪 [PDF] Translation Applied. Generating PDF...")

    # --- BUILD PDF (Using now translated data) ---
    offer_no = f"CHH-{datetime.now().strftime('%Y')}-{random.randint(1000,9999)}"
    handler = "Florian Siemer"
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=42*mm, bottomMargin=22*mm, title=f"Angebot {offer_no}", author=COMPANY["name"])
    styles = _styles()
    story = []
    
    _header_block(story, styles, offer_no, client_data)
    _intro(story, styles, client_data)

    for entry in plans_data:
        plan = entry["info"]
        pricing = entry["pricing"]
        story.append(PageBreak())
        floor_label = "Erdgeschoss" if entry["type"] == "ground_floor" else "Obergeschoss / Dachgeschoss"
        story.append(Paragraph(f"Planungsebene: {floor_label} ({plan.plan_id})", styles["H2"]))
        if plan.plan_image.exists():
            try:
                im = PILImage.open(plan.plan_image).convert("L")
                im = ImageEnhance.Brightness(im).enhance(0.9); im = ImageOps.autocontrast(im)
                width, height = im.size; aspect = width / height; target_width = A4[0]-36*mm
                if aspect < 1: target_width = (A4[0]-36*mm) * 0.65
                img_byte_arr = io.BytesIO(); im.save(img_byte_arr, format='PNG'); img_byte_arr.seek(0)
                rl_img = Image(img_byte_arr); rl_img._restrictSize(target_width, 75*mm); rl_img.hAlign = 'CENTER'
                story.append(Spacer(1, 5*mm)); story.append(rl_img); story.append(Spacer(1, 8*mm))
            except: pass
        
        bd = pricing.get("breakdown", {})
        if inclusions.get("foundation", False): _table_standard(story, styles, "Fundament / Bodenplatte", bd.get("foundation", {}))
        if bd.get("structure_walls"): _table_standard(story, styles, "Tragwerkskonstruktion – Wände", bd.get("structure_walls", {}))
        if bd.get("floors_ceilings"): _table_standard(story, styles, "Geschossdecken & Balken", bd.get("floors_ceilings", {}))
        if bd.get("stairs"): _table_stairs(story, styles, bd.get("stairs", {}))
        if bd.get("roof"): _table_roof_quantities(story, styles, pricing)
        if inclusions.get("finishes", False) and bd.get("finishes"): _table_standard(story, styles, "Oberflächen & Ausbau", bd.get("finishes", {}))

    if inclusions.get("openings", False): _table_global_openings(story, styles, global_openings)
    if inclusions.get("utilities", False): _table_global_utilities(story, styles, global_utilities)

    story.append(PageBreak())
    story.append(Paragraph("Gesamtkostenzusammenstellung", styles["H1"]))
    filtered_total = sum(e["pricing"].get("total_cost_eur", 0.0) for e in plans_data)
    org = filtered_total * 0.05; sup = filtered_total * 0.05; profit = filtered_total * 0.10
    disp_org = org + profit/2; disp_sup = sup + profit/2
    net = filtered_total + disp_org + disp_sup; vat = net * 0.19; gross = net + vat
    
    head = [P("Position", "CellBold"), P("Betrag", "CellBold")]
    data = [
        [P("Baukosten (Konstruktion, Ausbau, Technik)"), P(_money(filtered_total))],
        [P("Baustelleneinrichtung, Logistik & Planung (10%)"), P(_money(disp_org))],
        [P("Bauleitung, Koordination & Gewinn (10%)"), P(_money(disp_sup))],
        [P("<b>Nettosumme (exkl. MwSt.)</b>"), P(_money(net), "CellBold")],
        [P("MwSt. (19%)"), P(_money(vat))],
        [P("<b>GESAMTSUMME BRUTTO</b>"), P(_money(gross), "H2")],
    ]
    tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
    tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.3, colors.black), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), ("ALIGN", (1,1), (1,-1), "RIGHT"), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story.append(tbl)
    _closing_blocks(story, styles)
    
    doc.build(story, onFirstPage=_first_page_canvas(offer_no, handler), onLaterPages=_later_pages_canvas)
    print(f"✅ [PDF] Generat Final (DE): {output_path}")
    return output_path