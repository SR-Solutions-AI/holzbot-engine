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
    "Fundament (Sockel)": "Fundament (Sockel)",
    "Placa": "Bodenplatte",
    
    # ELEMENTE SCAPATE ANTERIOR (FIX)
    "Structură Tavan": "Deckenkonstruktion",
    "Structura Tavan": "Deckenkonstruktion",
    "Bodenstruktur": "Bodenaufbau",
    "Deckenstruktur": "Deckenaufbau",
    "Structura Podea": "Bodenkonstruktion",
    
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
    "Dublu": "Zweiflügelig",
    "Simplu": "Einflügelig",
    "Double": "Zweiflügelig",
    "Single": "Einflügelig",
    
    # Diverse & Hardcoded Labels (Extins)
    "Total": "Gesamt",
    "Pret": "Preis",
    "Cantitate": "Menge",
    "Descriere": "Beschreibung",
    "Angebot": "Angebot", 
    "Nr.": "Nr.",
    "Bauherr / Kunde": "Bauherr / Kunde",
    "Ort / Bauort": "Ort / Bauort",
    "Telefon": "Telefon",
    "E-Mail": "E-Mail",
    "Bauvorhaben": "Bauvorhaben",
    "Angebot für Ihr Chiemgauer Massivholzhaus": "Angebot für Ihr Chiemgauer Massivholzhaus",
    "Sehr geehrte Damen und Herren,": "Liebe Kundschaft,", 
    "vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung.": "vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung.",
    "HINWEIS: Unverbindliche Kostenschätzung. Kein verbindliches Angebot.": "HINWEIS: Unverbindliche Kostenschätzung. Kein verbindliches Angebot.",
    "Planungsebene": "Planungsebene",
    "Gesamtkostenzusammenstellung": "Gesamtkostenzusammenstellung",
    "Baukosten (Konstruktion, Ausbau, Technik)": "Baukosten (Konstruktion, Ausbau, Technik)",
    "Baustelleneinrichtung, Logistik & Planung (10%)": "Baustelleneinrichtung, Logistik & Planung",
    "Bauleitung, Koordination & Gewinn (10%)": "Bauleitung & Koordination",
    "Nettosumme (exkl. MwSt.)": "Nettosumme (exkl. MwSt.)",
    "MwSt. (19%)": "MwSt. (19%)",
    "GESAMTSUMME BRUTTO": "GESAMTSUMME BRUTTO",
    "Position": "Position",
    "Betrag": "Betrag",
    "Annahmen & Vorbehalte": "Annahmen & Vorbehalte",
    "Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen.": "Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen.",
    "Bauteil": "Bauteil", 
    "Fläche": "Fläche", 
    "Preis/m²": "Preis/m²", 
    "Gesamt": "Gesamt", 
    "SUMME": "SUMME",
    "Dachkonstruktion – Detail": "Dachkonstruktion – Detail", 
    "Komponente": "Komponente", 
    "Bemerkung": "Bemerkung", 
    "Menge": "Menge", 
    "Preis": "Preis", 
    "SUMME DACH": "SUMME DACH",
    "Treppenanlagen": "Treppenanlagen", 
    "Beschreibung": "Beschreibung", 
    "SUMME TREPPEN": "SUMME TREPPEN", 
    "Zusammenfassung Fenster & Türen": "Zusammenfassung Fenster & Türen",
    "Stückzahl": "Stückzahl", 
    "Ø Preis/Stk.": "Ø Preis/Stk.", 
    "SUMME ÖFFNUNGEN": "SUMME ÖFFNUNGEN", 
    "Zusammenfassung Haustechnik & Installationen": "Zusammenfassung Haustechnik & Installationen",
    "Gewerk / Kategorie": "Gewerk / Kategorie", 
    "Gesamtpreis": "Gesamtpreis", 
    "SUMME HAUSTECHNIK": "SUMME HAUSTECHNIK", 
    "Außentür": "Außentür", 
    "Innentür": "Innentür", 
    "Fenster": "Fenster",
    "Zweiflügelig": "Zweiflügelig", 
    "Einflügelig": "Einflügelig",
    "Fundament / Bodenplatte": "Fundament / Bodenplatte",
    "Tragwerkskonstruktion – Wände": "Tragwerkskonstruktion – Wände",
    "Geschossdecken & Balken": "Geschossdecken & Balken",
    "Oberflächen & Ausbau": "Oberflächen & Ausbau",
    "Kategorie": "Kategorie",
    
    # NOU: Textele de extins
    "Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen.": "Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen.",
    "Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung.": "Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung.",
}

# ---------- 2. AI TRANSLATION SERVICE (AGRESIV) ----------
class GermanEnforcer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if OpenAI and api_key else None
        self.cache = STATIC_TRANSLATIONS.copy()

    def get(self, text):
        """Returnează traducerea din cache. Dacă nu există, returnează originalul."""
        if not text or not isinstance(text, str): 
            return text
        
        translated = self.cache.get(text)
        
        if translated and '(' in translated and ')' in translated and translated != text:
            translated = re.sub(r'\s*\([^)]*\)\s*', '', translated).strip()
            return translated

        return translated if translated else text

    def translate_table_batch(self, texts: list[str]) -> dict[str, str]:
        """
        Traduce un batch de texte DIRECT pentru tabele.
        Filtrează automat: > 3 caractere, nu e preț/număr.
        """
        if not self.client:
            return {t: t for t in texts}
        
        # Filtrează ce merită tradus
        to_translate = []
        for t in texts:
            if not isinstance(t, str):
                continue
            t_clean = t.strip()
            if len(t_clean) <= 3:  # ✅ Skip < 3 caractere
                continue
            if re.match(r'^[\d\.,\s€]+$', t_clean):  # ✅ Skip prețuri/numere
                continue
            to_translate.append(t_clean)
        
        if not to_translate:
            return {t: t for t in texts}
        
        print(f"🇩🇪 [TableTranslation] Translating {len(to_translate)} items...")
        
        # Verifică cache-ul întâi
        results = {}
        missing = []
        
        for text in to_translate:
            if text in self.cache:
                results[text] = self.cache[text]
            else:
                missing.append(text)
        
        # Traduce ce lipsește
        if missing:
            prompt = (
                "You are a professional technical translator for the German construction industry. "
                "Translate the following JSON list from Romanian/English to German. "
                "Keep technical precision (e.g. 'Parter' -> 'Erdgeschoss', 'Structura Tavan' -> 'Deckenkonstruktion', "
                "'Pereti Exteriori' -> 'Außenwände', 'Lemn' -> 'Holz'). "
                "Output ONLY a valid JSON object where keys are input text and values are German translations."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(missing)}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                
                translations = json.loads(response.choices[0].message.content)
                self.cache.update(translations)
                results.update(translations)
                
                print(f"✅ [TableTranslation] Cached {len(translations)} new translations")
                
            except Exception as e:
                print(f"⚠️ [TableTranslation] Error: {e}")
                for text in missing:
                    results[text] = text
        
        # Returnează dicționar complet
        final_results = {}
        for t in texts:
            if isinstance(t, str):
                t_clean = t.strip()
                final_results[t] = results.get(t_clean, t)
            else:
                final_results[t] = t
        
        return final_results

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

def _header_block(story, styles, offer_no: str, client: dict, enforcer):
    left_lines = [COMPANY["legal"], *COMPANY["addr_lines"], "", f"Tel. {COMPANY['phone']}", f"Fax {COMPANY['fax']}", "", COMPANY["email"], COMPANY["web"]]
    tbl = Table([[P("<br/>".join(left_lines), "Small"), P("", "Small")]], colWidths=[95*mm, A4[0]-36*mm-95*mm])
    tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(Spacer(1, 36*mm))
    story.append(tbl)
    story.append(Spacer(1, 6*mm))
    
    story.append(Paragraph(f"{enforcer.get('Angebot')} • {enforcer.get('Nr.')}: {offer_no}", styles["H1"]))
    story.append(Spacer(1, 3*mm))
    
    name = enforcer.get((client.get("nume") or client.get("name") or "—").strip())
    city = enforcer.get((client.get("localitate") or client.get("city") or "—").strip())
    
    lines = [
        f"<b>{enforcer.get('Bauherr / Kunde')}:</b> {name}", 
        f"<b>{enforcer.get('Ort / Bauort')}:</b> {city}", 
        f"<b>{enforcer.get('Telefon')}:</b> {client.get('telefon') or '—'}", 
        f"<b>{enforcer.get('E-Mail')}:</b> {client.get('email') or '—'}"
    ]
    story.append(Paragraph("<br/>".join(lines), _styles()["Cell"]))
    story.append(Spacer(1, 6*mm))

def _intro(story, styles, client: dict, enforcer: GermanEnforcer):
    story.append(Paragraph(enforcer.get("Angebot für Ihr Chiemgauer Massivholzhaus"), styles["H2"]))
    story.append(Paragraph(enforcer.get("Sehr geehrte Damen und Herren,"), styles["Body"]))
    
    story.append(Paragraph(
        enforcer.get("Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen."), 
        styles["Body"]
    ))
    story.append(Paragraph(
        enforcer.get("Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung."), 
        styles["Body"]
    ))
    
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(enforcer.get("HINWEIS: Unverbindliche Kostenschätzung. Kein verbindliches Angebot."), styles["Disclaimer"]))
    story.append(Spacer(1, 6*mm))

# ---------- TABLES (TRADUCERE DIRECTĂ) ----------

def _table_standard(story, styles, title: str, data_dict: dict, enforcer: GermanEnforcer):
    items = data_dict.get("items", []) or data_dict.get("detailed_items", [])
    if not items: 
        return
    
    # ✅ COLECTEAZĂ tot textul
    all_texts = [title, "Bauteil", "Fläche", "Preis/m²", "Gesamt", "SUMME"]
    for it in items:
        all_texts.append(it.get("name", ""))
    
    # ✅ TRADUCE în batch
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE tabelul
    story.append(Paragraph(translations[title], styles["H2"]))
    
    head = [
        P(translations["Bauteil"], "CellBold"), 
        P(translations["Fläche"], "CellBold"), 
        P(translations["Preis/m²"], "CellBold"), 
        P(translations["Gesamt"], "CellBold")
    ]
    
    data = []
    for it in items:
        name_original = it.get("name", "—")
        name_translated = translations.get(name_original, name_original)
        
        data.append([
            P(name_translated), 
            P(_fmt_m2(it.get("area_m2", 0))), 
            P(_money(it.get("unit_price", 0)), "CellSmall"), 
            P(_money(it.get("cost", 0)), "CellBold")
        ])
    
    data.append([
        P(translations["SUMME"], "CellBold"), 
        "", 
        "", 
        P(_money(data_dict.get("total_cost", 0)), "CellBold")
    ])
    
    tbl = Table([head] + data, colWidths=[75*mm, 40*mm, 32*mm, 38*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (1,1), (-1,-1), "RIGHT")
    ]))
    
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

def _table_roof_quantities(story, styles, pricing_data: dict, enforcer: GermanEnforcer):
    roof = pricing_data.get("breakdown", {}).get("roof", {})
    items = roof.get("items", []) or roof.get("detailed_items", [])
    display_items = [it for it in items if "extra_walls" not in str(it.get("category", ""))]
    
    if not display_items: 
        return
    
    # ✅ COLECTEAZĂ
    all_texts = ["Dachkonstruktion – Detail", "Komponente", "Bemerkung", "Menge", "Preis", "SUMME DACH"]
    for it in display_items:
        name_raw = it.get("name", "")
        name_clean = re.sub(r'\s*\([^)]*\)\s*', '', name_raw).strip()
        all_texts.append(name_clean)
        all_texts.append(it.get("details", ""))
        all_texts.append(it.get("unit", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE
    story.append(Paragraph(translations["Dachkonstruktion – Detail"], styles["H2"]))
    
    head = [
        P(translations["Komponente"], "CellBold"), 
        P(translations["Bemerkung"], "CellBold"), 
        P(translations["Menge"], "CellBold"), 
        P(translations["Preis"], "CellBold")
    ]
    
    data = []
    for it in display_items:
        name_raw = it.get("name", "")
        name_clean = re.sub(r'\s*\([^)]*\)\s*', '', name_raw).strip()
        
        data.append([
            P(translations.get(name_clean, name_clean)), 
            P(translations.get(it.get("details", ""), it.get("details", "")), "CellSmall"), 
            P(_fmt_qty(it.get("quantity", 0), translations.get(it.get("unit", ""), it.get("unit", ""))), "CellSmall"), 
            P(_money(it.get("cost", 0)), "CellBold")
        ])
    
    data.append([
        P(translations["SUMME DACH"], "CellBold"), 
        "", 
        "", 
        P(_money(sum(it.get("cost", 0) for it in display_items)), "CellBold")
    ])
    
    tbl = Table([head] + data, colWidths=[55*mm, 70*mm, 24*mm, 32*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (2,1), (3,-1), "RIGHT")
    ]))
    
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

def _table_stairs(story, styles, stairs_dict: dict, enforcer: GermanEnforcer):
    items = stairs_dict.get("detailed_items", [])
    if not items: 
        return
    
    # ✅ COLECTEAZĂ
    all_texts = ["Treppenanlagen", "Bauteil", "Beschreibung", "Menge", "Gesamt", "SUMME TREPPEN"]
    for it in items:
        all_texts.append(it.get("name", ""))
        all_texts.append(it.get("details", ""))
        all_texts.append(it.get("unit", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE
    story.append(Paragraph(translations["Treppenanlagen"], styles["H2"]))
    
    head = [
        P(translations["Bauteil"], "CellBold"), 
        P(translations["Beschreibung"], "CellBold"), 
        P(translations["Menge"], "CellBold"), 
        P(translations["Gesamt"], "CellBold")
    ]
    
    data = []
    for it in items:
        qty_text = f"{it.get('quantity', 0)} {translations.get(it.get('unit', 'Stk.'), it.get('unit', 'Stk.'))}"
        
        data.append([
            P(translations.get(it.get("name", "—"), it.get("name", "—"))), 
            P(translations.get(it.get("details", "—"), it.get("details", "—")), "CellSmall"), 
            P(qty_text, "CellSmall"), 
            P(_money(it.get("cost", 0)), "CellBold")
        ])
    
    data.append([
        P(translations["SUMME TREPPEN"], "CellBold"), 
        "", 
        "", 
        P(_money(stairs_dict.get("total_cost", 0)), "CellBold")
    ])
    
    tbl = Table([head] + data, colWidths=[70*mm, 55*mm, 25*mm, 31*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (2,1), (3,-1), "RIGHT")
    ]))
    
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

def _table_global_openings(story, styles, all_openings: list, enforcer: GermanEnforcer):
    if not all_openings: 
        return
    
    story.append(PageBreak())
    
    # ✅ COLECTEAZĂ texte pentru clasificare
    all_texts = [
        "Zusammenfassung Fenster & Türen", 
        "Kategorie", 
        "Stückzahl", 
        "Ø Preis/Stk.", 
        "Gesamt",
        "SUMME ÖFFNUNGEN",
        "Außentür",
        "Innentür",
        "Fenster",
        "Zweiflügelig",
        "Einflügelig"
    ]
    
    # Adaugă și textele din items pentru clasificare
    for it in all_openings:
        all_texts.append(it.get("name", ""))
        all_texts.append(it.get("type", ""))
        all_texts.append(it.get("category", ""))
        all_texts.append(it.get("location", ""))
        all_texts.append(it.get("details", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    story.append(Paragraph(translations["Zusammenfassung Fenster & Türen"], styles["H1"]))
    
    groups = {}
    
    kw_window = ["fenster", "window", "glass", "fereastra"]
    kw_door = ["tür", "door", "usa", "ușă"]
    kw_ext = ["aussen", "außen", "exterior", "entrance", "haustür", "main"]
    kw_double = ["doppel", "double", "dublu", "zwei", "2-"]
    
    for it in all_openings:
        full_text = (
            str(it.get("name", "")) + " " + 
            str(it.get("type", "")) + " " + 
            str(it.get("category", "")) + " " + 
            str(it.get("location", "")) + " " +
            str(it.get("details", ""))
        ).lower()
        
        cost = float(it.get("total_cost", 0))
        
        is_window = any(x in full_text for x in kw_window)
        is_door = any(x in full_text for x in kw_door)
        
        if not (is_window or is_door): 
            continue
            
        is_ext = any(x in full_text for x in kw_ext)
        loc_str = "Außentür" if is_door and is_ext else "Innentür" if is_door else "Fenster"
        
        is_double = any(x in full_text for x in kw_double)
        type_str = translations["Zweiflügelig"] if is_double else translations["Einflügelig"]
        
        category_label = translations[loc_str]
        label = f"{category_label} ({type_str})"
            
        if label not in groups:
            groups[label] = {"n": 0, "eur": 0.0}
        groups[label]["n"] += 1
        groups[label]["eur"] += cost

    def avg(total, n): 
        return total / n if n > 0 else 0.0
    
    head = [
        P(translations["Kategorie"], "CellBold"), 
        P(translations["Stückzahl"], "CellBold"), 
        P(translations["Ø Preis/Stk."], "CellBold"), 
        P(translations["Gesamt"], "CellBold")
    ]
    
    data = []
    total_eur = 0.0
    
    for label in sorted(groups.keys()):
        g = groups[label]
        count = g["n"]
        cost = g["eur"]
        total_eur += cost
        
        data.append([
            P(label),
            P(str(count)),
            P(_money(avg(cost, count))),
            P(_money(cost))
        ])

    if total_eur > 0:
        data.append([
            P(translations["SUMME ÖFFNUNGEN"], "CellBold"), 
            "", 
            "", 
            P(_money(total_eur), "CellBold")
        ])
        
        tbl = Table([head] + data, colWidths=[68*mm, 26*mm, 34*mm, 40*mm])
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, colors.black),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
            ("ALIGN", (1,1), (-1,-1), "RIGHT"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE")
        ]))
        
        story.append(tbl)
        story.append(Spacer(1, 8*mm))

def _table_global_utilities(story, styles, all_utilities: list, enforcer: GermanEnforcer):
    if not all_utilities: 
        return
    
    # ✅ COLECTEAZĂ
    all_texts = [
        "Zusammenfassung Haustechnik & Installationen",
        "Gewerk / Kategorie",
        "Gesamtpreis",
        "SUMME HAUSTECHNIK"
    ]
    
    for it in all_utilities:
        all_texts.append(it.get("category", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    story.append(Paragraph(translations["Zusammenfassung Haustechnik & Installationen"], styles["H1"]))
    
    agg = {}
    total_util = 0.0
    
    for it in all_utilities:
        cat_original = it.get("category", "Sonstiges")
        cat_translated = translations.get(cat_original, cat_original)
        cost = it.get("total_cost", 0.0)
        total_util += cost
        agg[cat_translated] = agg.get(cat_translated, 0.0) + cost
    
    head = [
        P(translations["Gewerk / Kategorie"], "CellBold"), 
        P(translations["Gesamtpreis"], "CellBold")
    ]
    
    data = []
    for k, v in agg.items(): 
        data.append([P(k), P(_money(v))])
        
    data.append([
        P(translations["SUMME HAUSTECHNIK"], "CellBold"), 
        P(_money(total_util), "CellBold")
    ])
    
    tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), 
        ("ALIGN", (1,1), (1,-1), "RIGHT"), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    
    story.append(tbl)
    story.append(Spacer(1, 8*mm))

def _closing_blocks(story, styles, enforcer: GermanEnforcer):
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(enforcer.get("Annahmen & Vorbehalte"), styles["H2"]))
    story.append(Paragraph(enforcer.get("Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen."), styles["Body"]))

# ---------- MAIN GENERATOR ----------
def generate_complete_offer_pdf(run_id: str, output_path: Path | None = None) -> Path:
    print(f"🚀 [PDF] START: {run_id}")
    
    jobs_root_base = PROJECT_ROOT / "jobs"
    output_root = jobs_root_base / run_id
    
    if not output_root.exists(): 
        output_root = RUNNER_ROOT / "output" / run_id
        if not output_root.exists():
            print(f"❌ EROARE: Nu găsesc directorul de output pentru run_id='{run_id}'")
            raise FileNotFoundError(f"Output nu există: {output_root}")
        print(f"⚠️ Folosind directorul standard: {output_root.resolve()}")
    else:
        print(f"✅ Folosind JOBS_ROOT: {output_root.resolve()}")

    if output_path is None:
        final_output_root = RUNNER_ROOT / "output" / run_id
        pdf_dir = final_output_root / "offer_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"oferta_{run_id}.pdf"

    frontend_data = load_frontend_data_for_run(run_id)
    client_data_untranslated = frontend_data.get("client", frontend_data)
    nivel_oferta = frontend_data.get("materialeFinisaj", {}).get("nivelOferta", "Casă completă")
    inclusions = _get_offer_inclusions(nivel_oferta)
    
    try: 
        plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError: 
        plan_infos = []

    enriched_plans = []
    for plan in plan_infos:
        plan_name_parts = plan.plan_id.split('_')
        
        if len(plan_name_parts) >= 2 and plan_name_parts[-2] == 'cluster':
            meta_filename = f"{plan_name_parts[-2]}_{plan_name_parts[-1]}.json"
        else:
            meta_filename = f"{plan.plan_id}.json"

        meta_path = output_root / "plan_metadata" / meta_filename 
        
        floor_type = "unknown"
        
        if meta_path.exists():
            try: 
                raw_type = json.load(open(meta_path)).get("floor_classification", {}).get("floor_type", "unknown")
                floor_type = raw_type.strip().lower()
            except Exception as e: 
                floor_type = "unknown"
        
        if floor_type == "unknown":
            if "parter" in plan.plan_id.lower() or "ground" in plan.plan_id.lower(): 
                floor_type = "ground_floor"
            elif "etaj" in plan.plan_id.lower() or "top" in plan.plan_id.lower(): 
                floor_type = "top_floor"
        
        sort_key = 0 if floor_type == "ground_floor" else 1
        enriched_plans.append({"plan": plan, "floor_type": floor_type, "sort": sort_key})
        
    enriched_plans.sort(key=lambda x: x["sort"])

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
            filtered_breakdown = {}
            filtered_total = 0.0
            
            for category_key, category_data in breakdown.items():
                if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                if inclusions.get(category_key, False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
            
            p_json["breakdown"] = filtered_breakdown
            p_json["total_cost_eur"] = filtered_total
            
            if inclusions.get("utilities", False): 
                global_utilities.extend(breakdown.get("utilities", {}).get("items", []))
            if inclusions.get("openings", False): 
                global_openings.extend(breakdown.get("openings", {}).get("items", []))
            if inclusions.get("roof", False):
                roof_items = breakdown.get("roof", {}).get("items", [])
                extra_wall = next((it for it in roof_items if "extra_walls" in it.get("category", "")), None)
                if extra_wall and inclusions.get("structure_walls", False):
                    cost = extra_wall.get("cost", 0.0)
                    ws = filtered_breakdown.get("structure_walls", {})
                    target = next((it for it in ws.get("items", []) if "Außenwände" in it.get("name","") or "Exterior" in it.get("name","")), None)
                    if target: 
                        target["cost"] += cost
                        ws["total_cost"] += cost
                        filtered_breakdown.get("roof", {})["total_cost"] -= cost
                        filtered_total += cost

            plans_data.append({"info": plan, "type": p_data["floor_type"], "pricing": p_json})

    # ✅ INIȚIALIZARE ENFORCER (fără collect & process_translation_queue)
    print("🇩🇪 [PDF] Initializing GermanEnforcer (Direct Table Translation Mode)...")
    enforcer = GermanEnforcer()

    # --- BUILD PDF ---
    offer_no = f"CHH-{datetime.now().strftime('%Y')}-{random.randint(1000,9999)}"
    handler = "Florian Siemer"
    doc = SimpleDocTemplate(
        str(output_path), 
        pagesize=A4, 
        leftMargin=18*mm, 
        rightMargin=18*mm, 
        topMargin=42*mm, 
        bottomMargin=22*mm, 
        title=f"Angebot {offer_no}", 
        author=COMPANY["name"]
    )
    
    styles = _styles()
    story = []
    
    _header_block(story, styles, offer_no, client_data_untranslated, enforcer) 
    _intro(story, styles, client_data_untranslated, enforcer) 

    for entry in plans_data:
        plan = entry["info"]
        pricing = entry["pricing"]
        story.append(PageBreak())
        
        entry_type_clean = entry["type"].strip().lower() 
        floor_label_raw = "Erdgeschoss" if entry_type_clean == "ground_floor" else "Obergeschoss / Dachgeschoss"
        
        story.append(Paragraph(f"{enforcer.get('Planungsebene')}: {enforcer.get(floor_label_raw)}", styles["H2"]))
        
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
            except: 
                pass
        
        bd = pricing.get("breakdown", {})
        
        if inclusions.get("foundation", False): 
            _table_standard(story, styles, "Fundament / Bodenplatte", bd.get("foundation", {}), enforcer)
        if bd.get("structure_walls"): 
            _table_standard(story, styles, "Tragwerkskonstruktion – Wände", bd.get("structure_walls", {}), enforcer)
        if bd.get("floors_ceilings"): 
            _table_standard(story, styles, "Geschossdecken & Balken", bd.get("floors_ceilings", {}), enforcer)
        if bd.get("stairs"): 
            _table_stairs(story, styles, bd.get("stairs", {}), enforcer)
        if bd.get("roof"): 
            _table_roof_quantities(story, styles, pricing, enforcer)
        if inclusions.get("finishes", False) and bd.get("finishes"): 
            _table_standard(story, styles, "Oberflächen & Ausbau", bd.get("finishes", {}), enforcer)

    if inclusions.get("openings", False): 
        _table_global_openings(story, styles, global_openings, enforcer)
    if inclusions.get("utilities", False): 
        _table_global_utilities(story, styles, global_utilities, enforcer)

    story.append(PageBreak())
    story.append(Paragraph(enforcer.get("Gesamtkostenzusammenstellung"), styles["H1"]))
    
    filtered_total = sum(e["pricing"].get("total_cost_eur", 0.0) for e in plans_data)
    
    cost_margin_logistics = filtered_total * 0.10
    cost_margin_oversight = filtered_total * 0.10
    
    net = filtered_total + cost_margin_logistics + cost_margin_oversight
    vat = net * 0.19
    gross = net + vat
    
    head = [
        P(enforcer.get("Position"), "CellBold"), 
        P(enforcer.get("Betrag"), "CellBold")
    ]
    
    data = [
        [P(enforcer.get("Baukosten (Konstruktion, Ausbau, Technik)")), P(_money(filtered_total))],
        [P(enforcer.get("Baustelleneinrichtung, Logistik & Planung")), P(_money(cost_margin_logistics))],
        [P(enforcer.get("Bauleitung & Koordination")), P(_money(cost_margin_oversight))],
        [P(f"<b>{enforcer.get('Nettosumme (exkl. MwSt.)')}</b>"), P(_money(net), "CellBold")],
        [P(enforcer.get("MwSt. (19%)")), P(_money(vat))],
        [P(f"<b>{enforcer.get('GESAMTSUMME BRUTTO')}</b>"), P(_money(gross), "H2")],
    ]
    
    tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")), 
        ("ALIGN", (1,1), (1,-1), "RIGHT"), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    
    story.append(tbl)
    
    _closing_blocks(story, styles, enforcer)
    
    doc.build(
        story, 
        onFirstPage=_first_page_canvas(offer_no, handler), 
        onLaterPages=_later_pages_canvas
    )
    
    print(f"✅ [PDF] Generat Final (DE): {output_path}")
    return output_path