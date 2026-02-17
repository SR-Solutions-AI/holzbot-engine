# __init__.py - Modul pentru generarea PDF-ului de ofertă

from .generator import generate_complete_offer_pdf, generate_admin_offer_pdf, generate_admin_calculation_method_pdf
from .roof_measurements_pdf import generate_roof_measurements_pdf

__all__ = [
    "generate_complete_offer_pdf",
    "generate_admin_offer_pdf",
    "generate_admin_calculation_method_pdf",
    "generate_roof_measurements_pdf",
]