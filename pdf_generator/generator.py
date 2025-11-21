
from __future__ import annotations
import json
import io
import os
from pathlib import Path
from datetime import datetime
import random

# --- NEW: OpenAI Import ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Fallback dacă nu e instalat

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen.canvas import Canvas

from PIL import Image as PILImage, ImageEnhance, ImageOps

from ..config.settings import load_plan_infos, PlansListError, RUNNER_ROOT, PROJECT_ROOT
from ..config.frontend_loader import load_frontend_data_for_run

# ---------- AI TRANSLATION SERVICE ----------
class TechnicalTranslator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if OpenAI and api_key else None
        self.cache = {}
        # Cache predefinit pentru termeni comuni (fallback rapid)
        self.cache.update({
            "Parter": "Erdgeschoss",
            "Etaj": "Obergeschoss",
            "Mansarda": "Dachgeschoss",
            "Fundație": "Fundament",
            "Beton": "Beton",
            "Lemn": "Holz",
            "Fereastră": "Fenster",
            "Ușă": "Tür",
            # SCĂRI - Specific
            "Scară interioară completă (Structură + Finisaj)": "Komplette Innentreppe (Konstruktion + Belag)",
            "Balustradă scară": "Treppengeländer",
            "Include trepte, contratrepte și structură rezistență": "Inklusive Stufen, Setzstufen und Tragstruktur",
            "Mână curentă și elemente siguranță": "Handlauf und Sicherheitselemente"
        })

    def collect_texts(self, plans_data, global_openings, global_utilities):
        """Colectează toate textele unice care necesită traducere."""
        texts = set()

        # 1. Din pricing breakdown
        for plan in plans_data:
            breakdown = plan.get("pricing", {}).get("breakdown", {})
            for cat_key, cat_data in breakdown.items():
                items = cat_data.get("items", []) or cat_data.get("detailed_items", [])
                for item in items:
                    if item.get("name"): texts.add(item["name"])
                    if item.get("details"): texts.add(item["details"])
                    if item.get("category"): texts.add(item["category"])

        # 2. Din global lists
        for item in global_openings:
            if item.get("type"): texts.add(str(item["type"]))
            if item.get("location"): texts.add(str(item["location"]))
        
        for item in global_utilities:
            if item.get("category"): texts.add(str(item["category"]))

        # Eliminăm ce e deja în cache sau gol sau numeric
        return [t for t in texts if t and t not in self.cache and not str(t).replace('.', '').isdigit()]

    def translate_batch(self, text_list):
        """Trimite lista la OpenAI și populează cache-ul."""
        if not self.client or not text_list:
            return

        print(f"🤖 [AI-Translator] Translating {len(text_list)} terms to German...")

        # Batching pentru a nu depăși limitele (ex: 50 termeni per request)
        batch_size = 50
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            
            prompt = (
                "You are a technical translator for construction quotes (Bauwesen). "
                "Translate the following JSON list of terms from Romanian/English to professional German. "
                "Keep the meaning precise (e.g., 'Parter' -> 'Erdgeschoss', 'Beton armat' -> 'Stahlbeton'). "
                "Return ONLY a JSON object where keys are original text and values are German translations."
            )

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o", # sau gpt-3.5-turbo
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
                print(f"⚠️ [AI-Translator] Error translating batch: {e}")

    def get(self, text):
        """Returnează traducerea sau originalul."""
        if not text: return ""
        text_str = str(text)
        return self.cache.get(text_str, text_str)

# ---------- FONTS ----------
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

# ---------- COMPANY & IMAGES ----------
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

# ---------- OFFER INCLUSIONS (LOGIC) ----------
def _get_offer_inclusions(nivel_oferta: str) -> dict:
    """
    Returnează categoriile incluse. 
    Cheile dicționarului de input rămân în română ('Casă completă') 
    pentru a se potrivi cu ce vine din frontend (formConfig.ts).
    """
    INCLUSIONS = {
        "Structură": {
            "foundation": True,
            "structure_walls": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": False,
            "finishes": False,
            "utilities": False
        },
        "Structură + ferestre": {
            "foundation": True,
            "structure_walls": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": True,
            "finishes": False,
            "utilities": False
        },
        "Casă completă": {
            "foundation": True,
            "structure_walls": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": True,
            "finishes": True,
            "utilities": True
        }
    }
    
    # Default la Casă completă dacă nu se găsește cheia
    return INCLUSIONS.get(nivel_oferta, INCLUSIONS["Casă completă"])

# ---------- STYLES & FORMATTING ----------
def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="H1", fontName=BOLD_FONT,  fontSize=12,   leading=22, spaceAfter=10, spaceBefore=6))
    s.add(ParagraphStyle(name="H2", fontName=BOLD_FONT,  fontSize=11, leading=14, spaceBefore=12, spaceAfter=6))
    s.add(ParagraphStyle(name="Body", fontName=BASE_FONT, fontSize=10,  leading=14, spaceAfter=4))
    s.add(ParagraphStyle(name="Small", fontName=BASE_FONT, fontSize=7.2, leading=9.2))
    s.add(ParagraphStyle(name="Disclaimer", fontName=BASE_FONT, fontSize=8.5, leading=11, textColor=colors.HexColor("#333333")))
    s.add(ParagraphStyle(name="Cell", fontName=BASE_FONT, fontSize=9.5, leading=12))
    s.add(ParagraphStyle(name="CellBold", fontName=BOLD_FONT, fontSize=9.5, leading=12))
    s.add(ParagraphStyle(name="CellSmall", fontName=BASE_FONT, fontSize=9, leading=11))
    return s

def P(text, style_name="Cell"):
    return Paragraph((str(text) or "").replace("\n", "<br/>"), _styles()[style_name])

def _money(x):
    """Formatare Germană: 1.234,56 €"""
    try:
        v = float(x)
        # Formatare cu comma pentru mii temporar, apoi switch
        # standard f-string is 1,234.56
        s = f"{v:,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} €"
    except:
        return "—"

def _fmt_m2(v):
    """Formatare Germană: 1.234,56 m²"""
    try:
        val = float(v)
        s = f"{val:,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} m²"
    except:
        return "—"

def _fmt_qty(v, unit=""):
    try:
        val = float(v)
        s = f"{val:,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} {unit}"
    except:
        return "—"

# ---------- CANVAS (HEADER/FOOTER) ----------

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
    rows = [
        ("Datum", datetime.now().strftime("%d.%m.%Y")),
        ("Bearbeiter", handler),
        ("Kunden-Nr.", "—"), # Fibu-Info -> Kunden-Nr.
        ("Auftrag", offer_no),
    ]
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

# ---------- CONTENT BLOCKS ----------

def _header_block(story, styles, offer_no: str, client: dict):
    left_lines = [
        COMPANY["legal"],
        *COMPANY["addr_lines"],
        "",
        f"Tel. {COMPANY['phone']}",
        f"Fax {COMPANY['fax']}",
        "",
        COMPANY["email"],
        COMPANY["web"],
    ]
    left_par = Paragraph("<br/>".join(left_lines), styles["Small"])
    right_par = Paragraph("", styles["Small"])

    tbl = Table([[left_par, right_par]], colWidths=[95*mm, A4[0]-36*mm-95*mm])
    tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))

    story.append(Spacer(1, 36*mm))
    story.append(tbl)
    story.append(Spacer(1, 6*mm))
    
    story.append(Paragraph(f"Angebot • Nr.: {offer_no}", styles["H1"]))
    story.append(Spacer(1, 3*mm))
    
    story.append(_client_info_block(client))
    story.append(Spacer(1, 6*mm))

def _client_info_block(client: dict):
    # Mapare variabile din formConfig.ts
    name = (client.get("nume") or client.get("name") or "—").strip()
    city = (client.get("localitate") or client.get("city") or "—").strip()
    phone = (client.get("telefon") or client.get("phone") or "—").strip()
    email = (client.get("email") or "—").strip()
    proj = (client.get("referinta") or client.get("project_label") or "—").strip()

    lines = [
        f"<b>Bauherr / Kunde:</b> {name}",
        f"<b>Ort / Bauort:</b> {city}",
        f"<b>Telefon:</b> {phone}",
        f"<b>E-Mail:</b> {email}",
        f"<b>Bauvorhaben:</b> {proj}",
    ]
    return Paragraph("<br/>".join(lines), _styles()["Cell"])

def _intro(story, styles, client):
    story.append(Paragraph("Angebot für Ihr Chiemgauer Massivholzhaus", styles["H2"]))
    story.append(Paragraph("Sehr geehrte Damen und Herren,", styles["Body"]))
    story.append(Paragraph("vielen Dank für Ihre Anfrage und das uns entgegengebrachte Vertrauen. Auf Basis der übermittelten Informationen haben wir für Sie die folgende Kostenschätzung ausgearbeitet.", styles["Body"]))
    story.append(Paragraph("Für Rückfragen oder eine detaillierte Besprechung stehen wir Ihnen jederzeit gerne zur Verfügung.", styles["Body"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "HINWEIS: Dieses Dokument stellt eine unverbindliche Kostenschätzung dar und ist kein verbindliches Festpreisangebot im rechtlichen Sinne.",
        styles["Disclaimer"]
    ))
    story.append(Spacer(1, 6*mm))

# ---------- TABLES (WITH TRANSLATION) ----------

def _table_standard(story, styles, title, data_dict, translator: TechnicalTranslator):
    items = data_dict.get("items", []) or data_dict.get("detailed_items", [])
    if not items:
        return

    story.append(Paragraph(title, styles["H2"]))
    # German Headers
    head = [P("Bauteil", "CellBold"), P("Fläche", "CellBold"), P("Preis/m²", "CellBold"), P("Gesamt", "CellBold")]
    data = []

    for it in items:
        # Traducere nume element
        raw_name = it.get("name", "—")
        translated_name = translator.get(raw_name)

        data.append([
            P(translated_name), 
            P(_fmt_m2(it.get("area_m2", 0))), 
            P(_money(it.get("unit_price", 0)), "CellSmall"),
            P(_money(it.get("cost", 0)), "CellBold"),
        ])
    
    data.append([P("SUMME", "CellBold"), "", "", P(_money(data_dict.get("total_cost", 0)), "CellBold")])
    
    tbl = Table([head] + data, colWidths=[75*mm, 40*mm, 32*mm, 38*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

def _table_roof_quantities(story, styles, pricing_data: dict, translator: TechnicalTranslator):
    """Tabel Acoperiș cu cantități. FĂRĂ PEREȚI SUPLIMENTARI (mutați la pereți exteriori)."""
    roof = pricing_data.get("breakdown", {}).get("roof", {})
    items = roof.get("items", []) or roof.get("detailed_items", [])
    
    display_items = [it for it in items if "extra_walls" not in it.get("category", "")]
    
    if not display_items:
        return
    
    visible_total = sum(it.get("cost", 0) for it in display_items)

    story.append(Paragraph("Dachkonstruktion – Detail", styles["H2"]))
    
    # German Headers
    head = [P("Komponente", "CellBold"), P("Bemerkung", "CellBold"), P("Menge", "CellBold"), P("Preis", "CellBold")]
    data = []
    
    for it in display_items:
        qty = it.get("quantity", 0)
        unit = it.get("unit", "")
        
        # Traducere
        t_name = translator.get(it.get("name", "—"))
        t_details = translator.get(it.get("details", ""))
        
        data.append([
            P(t_name),
            P(t_details, "CellSmall"),
            P(_fmt_qty(qty, unit), "CellSmall"),
            P(_money(it.get("cost", 0)), "CellBold"),
        ])
    
    data.append([P("SUMME DACH", "CellBold"), "", "", P(_money(visible_total), "CellBold")])
    
    tbl = Table([head] + data, colWidths=[55*mm, 70*mm, 24*mm, 32*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,1), (3,-1), "RIGHT"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

# ✨✨✨ FUNCȚIE NOUĂ: TABEL SCĂRI (CU TRADUCERI CORECTE) ✨✨✨
def _table_stairs(story, styles, stairs_dict: dict, translator: TechnicalTranslator):
    """
    Tabel dedicat pentru scări (Treppen).
    Afișează structura și balustrada.
    """
    items = stairs_dict.get("detailed_items", [])
    if not items:
        return

    # Titlu Secțiune
    story.append(Paragraph("Treppenanlagen", styles["H2"]))
    
    # Headere în Germană
    head = [P("Bauteil", "CellBold"), P("Beschreibung", "CellBold"), P("Menge", "CellBold"), P("Gesamt", "CellBold")]
    data = []
    
    for it in items:
        # Traducere dinamică (sau din cache)
        t_name = translator.get(it.get("name", "—"))
        t_details = translator.get(it.get("details", "—"))
        
        qty = it.get("quantity", 0)
        raw_unit = it.get("unit", "buc")
        
        # Mapare unitate (buc -> Stk.)
        unit = "Stk." if raw_unit in ["buc", "piece"] else raw_unit
        
        cost = it.get("cost", 0)
        
        data.append([
            P(t_name),
            P(t_details, "CellSmall"),
            P(f"{qty} {unit}", "CellSmall"),
            P(_money(cost), "CellBold"),
        ])

    data.append([P("SUMME TREPPEN", "CellBold"), "", "", P(_money(stairs_dict.get("total_cost", 0)), "CellBold")])
    
    tbl = Table([head] + data, colWidths=[70*mm, 55*mm, 25*mm, 31*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,1), (3,-1), "RIGHT"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 4*mm))


def _table_global_openings(story, styles, all_openings: list, translator: TechnicalTranslator):
    if not all_openings: return 0.0

    story.append(PageBreak())
    story.append(Paragraph("Zusammenfassung Fenster & Türen", styles["H1"]))
    
    agg = {
        "windows": {"n": 0, "eur": 0.0},
        "doors_int": {"n": 0, "eur": 0.0},
        "doors_ext": {"n": 0, "eur": 0.0}
    }
    
    for it in all_openings:
        t = str(it.get("type", "")).lower()
        cost = float(it.get("total_cost", 0))
        
        if "window" in t:
            agg["windows"]["n"] += 1
            agg["windows"]["eur"] += cost
        elif "door" in t:
            if "exterior" in t or "entrance" in t or "outside" in str(it.get("location", "")).lower():
                agg["doors_ext"]["n"] += 1
                agg["doors_ext"]["eur"] += cost
            else:
                agg["doors_int"]["n"] += 1
                agg["doors_int"]["eur"] += cost

    def avg(total, n): return total / n if n > 0 else 0.0

    # German Headers
    head = [P("Kategorie", "CellBold"), P("Stückzahl", "CellBold"), P("Ø Preis/Stk.", "CellBold"), P("Gesamt", "CellBold")]
    data = []
    
    if agg["windows"]["n"] > 0:
        data.append([P("Fensterelemente"), P(str(agg["windows"]["n"])), P(_money(avg(agg["windows"]["eur"], agg["windows"]["n"]))), P(_money(agg["windows"]["eur"]))])
    if agg["doors_ext"]["n"] > 0:
        data.append([P("Außentüren / Hauseingang"), P(str(agg["doors_ext"]["n"])), P(_money(avg(agg["doors_ext"]["eur"], agg["doors_ext"]["n"]))), P(_money(agg["doors_ext"]["eur"]))])
    if agg["doors_int"]["n"] > 0:
        data.append([P("Innentüren"), P(str(agg["doors_int"]["n"])), P(_money(avg(agg["doors_int"]["eur"], agg["doors_int"]["n"]))), P(_money(agg["doors_int"]["eur"]))])
    
    total_eur = agg["doors_int"]["eur"] + agg["doors_ext"]["eur"] + agg["windows"]["eur"]
    data.append([P("SUMME ÖFFNUNGEN", "CellBold"), "", "", P(_money(total_eur), "CellBold")])

    tbl = Table([head] + data, colWidths=[68*mm, 26*mm, 34*mm, 40*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8*mm))
    return total_eur

def _table_global_utilities(story, styles, all_utilities: list, translator: TechnicalTranslator):
    """Tabel global utilități (instalații) - toate planurile însumate."""
    if not all_utilities: return 0.0

    story.append(Paragraph("Zusammenfassung Haustechnik & Installationen", styles["H1"]))
    
    agg = {"electricity": 0.0, "sewage": 0.0, "heating": 0.0, "ventilation": 0.0}
    total_util = 0.0
    
    for it in all_utilities:
        cat = it.get("category", "")
        cost = it.get("total_cost", 0.0)
        total_util += cost
        if cat in agg:
            agg[cat] += cost
        else:
            agg.setdefault("other", 0.0)
            agg["other"] += cost
            
    head = [P("Gewerk / Kategorie", "CellBold"), P("Gesamtpreis", "CellBold")]
    data = []
    
    # German Mapping
    label_map = {
        "electricity": "Elektroinstallation",
        "sewage": "Sanitär & Abwasser",
        "heating": "Heizung & Wärmetechnik",
        "ventilation": "Lüftungstechnik",
        "other": "Sonstiges"
    }
    
    for k, v in agg.items():
        if v > 0:
            # Traducere fallback (dacă label_map nu are cheia)
            lbl = label_map.get(k, k.title())
            data.append([P(lbl), P(_money(v))])
            
    data.append([P("SUMME HAUSTECHNIK", "CellBold"), P(_money(total_util), "CellBold")])
    
    tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8*mm))
    return total_util

def _closing_blocks(story, styles):
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Annahmen & Vorbehalte", styles["H2"]))
    story.append(Paragraph(
        "Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen und Standard-Ausführungsdetails für den Massivholzbau. "
        "Spezielle geotechnische Anforderungen (Bodengutachten) oder spätere Planungsänderungen können die Massen und Kosten beeinflussen.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Die Ausführungstermine sind abhängig von der Materialverfügbarkeit und der Witterung. Preisbindefrist: 4 Wochen ab Ausstellungsdatum.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Wir hoffen, Ihnen mit diesem Angebot eine gute Entscheidungsgrundlage zu bieten und freuen uns auf Ihre Rückmeldung.",
        styles["Body"]
    ))

# ---------- MAIN GENERATOR ----------

def generate_complete_offer_pdf(run_id: str, output_path: Path | None = None) -> Path:
    print(f"🚀 [PDF] START: {run_id}")
    
    output_root = RUNNER_ROOT / "output" / run_id
    if not output_root.exists():
        raise FileNotFoundError(f"Output nu există: {output_root}")

    if output_path is None:
        pdf_dir = output_root / "offer_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"oferta_{run_id}.pdf"

    # Date client & nivel ofertă
    print(f"🔍 [PDF] Loading frontend data for run_id: {run_id}")
    frontend_data = load_frontend_data_for_run(run_id)
    
    # Variabilele din frontend sunt populate aici. Ele au cheile definite in formConfig.ts
    client_data = frontend_data.get("client", frontend_data)
    
    # Extrage nivelul de ofertă (ex: 'Casă completă' - string din frontend)
    nivel_oferta = frontend_data.get("materialeFinisaj", {}).get("nivelOferta", "Casă completă")
    inclusions = _get_offer_inclusions(nivel_oferta)
    
    print(f"📋 [PDF] Client data loaded: {list(client_data.keys())}")
    print(f"📋 [PDF] Nivel ofertă: {nivel_oferta}")
    print(f"📋 [PDF] Inclusions: {inclusions}")

    # Load Plans
    try:
        plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError:
        plan_infos = []

    # Clasificare Planuri
    enriched_plans = []
    for plan in plan_infos:
        meta_path = output_root / "plan_metadata" / f"{plan.plan_id}.json"
        floor_type = "unknown"
        if meta_path.exists():
            try:
                with open(meta_path) as f: 
                    floor_type = json.load(f).get("floor_classification", {}).get("floor_type", "unknown")
            except: 
                pass
        
        if floor_type == "unknown":
            if "parter" in plan.plan_id.lower() or "ground" in plan.plan_id.lower(): 
                floor_type = "ground_floor"
            elif "etaj" in plan.plan_id.lower() or "top" in plan.plan_id.lower(): 
                floor_type = "top_floor"

        enriched_plans.append({
            "plan": plan,
            "floor_type": floor_type,
            "sort": 0 if floor_type == "ground_floor" else 1 if floor_type == "intermediate" else 2
        })
    enriched_plans.sort(key=lambda x: x["sort"])

    # ✅ PREPROCESARE CU FILTRARE
    plans_data = []
    global_openings = []
    global_utilities = []

    for p_data in enriched_plans:
        plan = p_data["plan"]
        pricing_path = plan.stage_work_dir / "pricing_raw.json"
        if pricing_path.exists():
            with open(pricing_path, encoding="utf-8") as f:
                p_json = json.load(f)
            
            breakdown = p_json.get("breakdown", {})
            
            # ✅ FILTRARE: Aplică inclusions
            filtered_breakdown = {}
            filtered_total = 0.0
            
            for category_key, category_data in breakdown.items():
                # Gestionăm scările separat: sunt incluse dacă 'Structure + Floors' sunt incluse
                # De regulă, scările fac parte din structură
                if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                     filtered_breakdown[category_key] = category_data
                     filtered_total += category_data.get("total_cost", 0.0)
                     continue

                # Verifică dacă categoria e inclusă în nivelul de ofertă
                should_include = inclusions.get(category_key, False)
                
                if should_include:
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
            
            # Suprascrie breakdown-ul cu versiunea filtrată
            p_json["breakdown"] = filtered_breakdown
            p_json["total_cost_eur"] = filtered_total
            
            # 1. Utilități -> Global (doar dacă incluse)
            if inclusions.get("utilities", False):
                utils = breakdown.get("utilities", {}).get("items", [])
                global_utilities.extend(utils)
            
            # 2. Openings -> Global (doar dacă incluse)
            if inclusions.get("openings", False):
                ops = breakdown.get("openings", {}).get("items", [])
                global_openings.extend(ops)
            
            # 3. Mutare pereți acoperiș (dacă roof e inclus)
            if inclusions.get("roof", False):
                roof_items = breakdown.get("roof", {}).get("items", [])
                extra_wall_item = next((it for it in roof_items if "extra_walls" in it.get("category", "")), None)
                
                if extra_wall_item and inclusions.get("structure_walls", False):
                    cost_extra = extra_wall_item.get("cost", 0.0)
                    walls_struct = filtered_breakdown.get("structure_walls", {})
                    walls_items = walls_struct.get("items", [])
                    ext_wall_target = next((it for it in walls_items if "Außenwände" in it.get("name", "") or "Exterior" in it.get("name", "")), None)
                    
                    if ext_wall_target:
                        ext_wall_target["cost"] += cost_extra
                        walls_struct["total_cost"] += cost_extra
                        filtered_breakdown.get("roof", {})["total_cost"] -= cost_extra
                        filtered_total += cost_extra  # Ajustează totalul

            plans_data.append({"info": plan, "type": p_data["floor_type"], "pricing": p_json})

    # --- AI TRANSLATION PHASE ---
    translator = TechnicalTranslator()
    print("🌍 [PDF] Collecting texts for translation...")
    texts_to_translate = translator.collect_texts(plans_data, global_openings, global_utilities)
    translator.translate_batch(texts_to_translate)
    print("🌍 [PDF] Translation complete.")

    # --- BUILD PDF ---
    offer_no = f"CHH-{datetime.now().strftime('%Y')}-{random.randint(1000,9999)}"
    handler = "Florian Siemer"
    
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=42*mm, bottomMargin=22*mm,
                            title=f"Angebot {offer_no}", author=COMPANY["name"])
    
    styles = _styles()
    story = []
    
    _header_block(story, styles, offer_no, client_data)
    _intro(story, styles, client_data)

    # ✅ LOOP PLANURI CU TABELE FILTRATE
    for entry in plans_data:
        plan = entry["info"]
        pricing = entry["pricing"]
        
        story.append(PageBreak())
        
        # Titlu German pentru etaj
        floor_label = "Erdgeschoss" if entry["type"] == "ground_floor" else "Obergeschoss / Dachgeschoss"
        story.append(Paragraph(f"Planungsebene: {floor_label} ({plan.plan_id})", styles["H2"]))
        
        # Imagine plan
        if plan.plan_image.exists():
            try:
                im = PILImage.open(plan.plan_image).convert("L")
                im = ImageEnhance.Brightness(im).enhance(0.9)
                im = ImageOps.autocontrast(im)
                
                width, height = im.size
                aspect = width / height
                target_width = A4[0]-36*mm
                if aspect < 1:
                    target_width = (A4[0]-36*mm) * 0.65
                
                img_byte_arr = io.BytesIO()
                im.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                rl_img = Image(img_byte_arr)
                rl_img._restrictSize(target_width, 75*mm)
                rl_img.hAlign = 'CENTER'
                
                story.append(Spacer(1, 5*mm)) 
                story.append(rl_img)
                story.append(Spacer(1, 8*mm))
            except Exception as e:
                print(f"⚠️ Eroare imagine: {e}")

        # ✅ TABELE FILTRATE: Afișează doar ce e inclus, cu Titluri Germane
        breakdown = pricing.get("breakdown", {})
        
        # Foundation
        if inclusions.get("foundation", False) and breakdown.get("foundation"):
            _table_standard(story, styles, "Fundament / Bodenplatte", breakdown.get("foundation", {}), translator)
        
        # Pereți
        if breakdown.get("structure_walls"):
            _table_standard(story, styles, "Tragwerkskonstruktion – Wände", breakdown.get("structure_walls", {}), translator)
        
        # Planșee
        if breakdown.get("floors_ceilings"):
            _table_standard(story, styles, "Geschossdecken & Balken", breakdown.get("floors_ceilings", {}), translator)

        # ✨ TABEL SCĂRI (ADĂUGAT)
        if breakdown.get("stairs"):
            _table_stairs(story, styles, breakdown.get("stairs", {}), translator)

        # Acoperiș
        if breakdown.get("roof"):
            _table_roof_quantities(story, styles, pricing, translator)
        
        # Finisaje
        if inclusions.get("finishes", False) and breakdown.get("finishes"):
            _table_standard(story, styles, "Oberflächen & Ausbau", breakdown.get("finishes", {}), translator)

    # ✅ CENTRALIZATOARE (doar dacă incluse)
    if inclusions.get("openings", False) and global_openings:
        _table_global_openings(story, styles, global_openings, translator)
    
    if inclusions.get("utilities", False) and global_utilities:
        _table_global_utilities(story, styles, global_utilities, translator)

    # ✅ CALCUL FINAL
    story.append(PageBreak())
    story.append(Paragraph("Gesamtkostenzusammenstellung", styles["H1"]))
    
    # Recalculează totalul DOAR din categoriile incluse
    filtered_total_construction = sum(
        entry["pricing"].get("total_cost_eur", 0.0) 
        for entry in plans_data
    )
    
    print(f"💰 [PDF] Total construction (filtered): {filtered_total_construction:,.2f} EUR")
    
    org_percentage = 0.05
    sup_percentage = 0.05
    profit_percentage = 0.10
    
    real_org_cost = filtered_total_construction * org_percentage
    real_sup_cost = filtered_total_construction * sup_percentage
    real_profit_cost = filtered_total_construction * profit_percentage
    
    split_profit = real_profit_cost / 2
    
    display_org_cost = real_org_cost + split_profit
    display_sup_cost = real_sup_cost + split_profit
    
    total_net = filtered_total_construction + display_org_cost + display_sup_cost
    vat = total_net * 0.19
    total_gross = total_net + vat
    
    head = [P("Position", "CellBold"), P("Betrag", "CellBold")]
    data = [
        [P("Baukosten (Konstruktion, Ausbau, Technik)"), P(_money(filtered_total_construction))],
        [P("Baustelleneinrichtung, Logistik & Planung (10%)"), P(_money(display_org_cost))],
        [P("Bauleitung, Koordination & Gewinn (10%)"), P(_money(display_sup_cost))],
        [P("<b>Nettosumme (exkl. MwSt.)</b>"), P(_money(total_net), "CellBold")],
        [P("MwSt. (19%)"), P(_money(vat))],
        [P("<b>GESAMTSUMME BRUTTO</b>"), P(_money(total_gross), "H2")],
    ]
    
    tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(tbl)
    
    _closing_blocks(story, styles)
    
    doc.build(story, onFirstPage=_first_page_canvas(offer_no, handler), onLaterPages=_later_pages_canvas)
    print(f"✅ [PDF] Generat Final (DE): {output_path}")
    return output_path
