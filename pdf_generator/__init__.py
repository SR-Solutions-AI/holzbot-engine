# __init__.py - PDF offer generation (lazy imports: avoid ReportLab when only submodules e.g. offer_scope are used)

from __future__ import annotations

__all__ = [
    "generate_complete_offer_pdf",
    "generate_admin_offer_pdf",
    "generate_admin_calculation_method_pdf",
    "generate_roof_measurements_pdf",
]


def __getattr__(name: str):
    if name == "generate_complete_offer_pdf":
        from .generator import generate_complete_offer_pdf

        return generate_complete_offer_pdf
    if name == "generate_admin_offer_pdf":
        from .generator import generate_admin_offer_pdf

        return generate_admin_offer_pdf
    if name == "generate_admin_calculation_method_pdf":
        from .generator import generate_admin_calculation_method_pdf

        return generate_admin_calculation_method_pdf
    if name == "generate_roof_measurements_pdf":
        from .roof_measurements_pdf import generate_roof_measurements_pdf

        return generate_roof_measurements_pdf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
