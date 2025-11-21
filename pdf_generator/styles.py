# styles.py - Definirea stilurilor pentru PDF
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

# Încearcă să încarce fonturile DejaVu
FONTS_DIR = Path(__file__).parent.parent / "pdf_assets" / "fonts"
BASE_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"

try:
    font_regular = FONTS_DIR / "DejaVuSans.ttf"
    font_bold = FONTS_DIR / "DejaVuSans-Bold.ttf"
    if font_regular.exists() and font_bold.exists():
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(font_regular)))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(font_bold)))
        BASE_FONT = "DejaVuSans"
        BOLD_FONT = "DejaVuSans-Bold"
except Exception:
    pass

_CACHED_STYLES = None

def get_styles():
    global _CACHED_STYLES
    if _CACHED_STYLES is not None:
        return _CACHED_STYLES
    
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name="TitleMain",
        fontName=BOLD_FONT,
        fontSize=18,
        leading=22,
        spaceAfter=12,
        textColor=colors.HexColor("#1c1c1c"),
    ))
    
    styles.add(ParagraphStyle(
        name="H1",
        fontName=BOLD_FONT,
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#2c2c2c"),
    ))
    
    styles.add(ParagraphStyle(
        name="H2",
        fontName=BOLD_FONT,
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#3c3c3c"),
    ))
    
    styles.add(ParagraphStyle(
        name="H3",
        fontName=BOLD_FONT,
        fontSize=10.5,
        leading=13,
        spaceBefore=8,
        spaceAfter=3,
        textColor=colors.HexColor("#4c4c4c"),
    ))
    
    styles.add(ParagraphStyle(
        name="Body",
        fontName=BASE_FONT,
        fontSize=10,
        leading=14,
        spaceAfter=4,
    ))
    
    styles.add(ParagraphStyle(
        name="BodySmall",
        fontName=BASE_FONT,
        fontSize=9,
        leading=12,
        spaceAfter=3,
        textColor=colors.HexColor("#666666"),
    ))
    
    styles.add(ParagraphStyle(
        name="Cell",
        fontName=BASE_FONT,
        fontSize=9,
        leading=11,
    ))
    
    styles.add(ParagraphStyle(
        name="CellBold",
        fontName=BOLD_FONT,
        fontSize=9,
        leading=11,
    ))
    
    styles.add(ParagraphStyle(
        name="CellSmall",
        fontName=BASE_FONT,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#666666"),
    ))
    
    styles.add(ParagraphStyle(
        name="InfoBox",
        fontName=BASE_FONT,
        fontSize=9,
        leading=12,
        leftIndent=10,
        rightIndent=10,
        spaceAfter=6,
        textColor=colors.HexColor("#444444"),
    ))
    
    _CACHED_STYLES = styles
    return styles

COLORS = {
    "primary": colors.HexColor("#1c1c1c"),
    "secondary": colors.HexColor("#4a4a4a"),
    "accent": colors.HexColor("#0066cc"),
    "bg_light": colors.HexColor("#f7f7f7"),
    "bg_header": colors.HexColor("#e8e8e8"),
    "border": colors.HexColor("#cccccc"),
    "success": colors.HexColor("#2d7a2d"),
    "warning": colors.HexColor("#cc6600"),
}