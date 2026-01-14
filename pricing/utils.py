"""
pricing/utils.py
Utilități pentru debugging, validare și raportare.
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List


def validate_pricing_data(pricing_data: dict) -> tuple[bool, list[str]]:
    """
    Validează că datele de pricing sunt complete și corecte.
    
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    required_categories = ["structure", "windows_doors", "finishes", "roof", "utilities"]
    
    for cat in required_categories:
        if cat not in pricing_data:
            errors.append(f"Missing category: {cat}")
            continue
        
        cat_data = pricing_data[cat]
        if not isinstance(cat_data, dict):
            errors.append(f"Category {cat} is not a dict")
            continue
        
        if "items" not in cat_data:
            errors.append(f"Category {cat} missing 'items' key")
        
        if "total_cost_eur" not in cat_data:
            errors.append(f"Category {cat} missing 'total_cost_eur'")
    
    if "total_cost_eur" not in pricing_data:
        errors.append("Missing total_cost_eur at root level")
    
    return len(errors) == 0, errors


def generate_pricing_summary(pricing_data: dict) -> str:
    """
    Generează un rezumat text lizibil al costurilor.
    """
    lines = []
    lines.append("\n" + "="*70)
    lines.append("💰 REZUMAT COSTURI")
    lines.append("="*70)
    
    total = 0.0
    
    for cat_key, cat_data in pricing_data.items():
        if cat_key == "total_cost_eur" or not isinstance(cat_data, dict):
            continue
        
        cat_name = cat_data.get("category", cat_key)
        cat_total = cat_data.get("total_cost_eur", 0)
        total += cat_total
        
        lines.append(f"\n📦 {cat_name}")
        lines.append("-" * 70)
        
        for item in cat_data.get("items", []):
            name = item["name"]
            qty = item["quantity"]
            unit = item["unit"]
            price = item["unit_price_eur"]
            item_total = item["total_cost_eur"]
            
            lines.append(f"   • {name:30s} {qty:8.2f} {unit:4s} × {price:8.2f} EUR = {item_total:10,.2f} EUR")
        
        lines.append(f"\n   SUBTOTAL {cat_name:20s} {cat_total:10,.2f} EUR")
    
    lines.append("\n" + "="*70)
    lines.append(f"💰 TOTAL GENERAL: {total:,.2f} EUR")
    lines.append("="*70 + "\n")
    
    return "\n".join(lines)


def compare_offers(offers: List[Dict]) -> str:
    """
    Compară mai multe oferte side-by-side.
    
    Args:
        offers: Lista de dicționare cu oferte (din build_final_offer)
    
    Returns:
        String formatat cu comparație
    """
    lines = []
    lines.append("\n" + "="*100)
    lines.append("📊 COMPARAȚIE OFERTE")
    lines.append("="*100)
    
    # Header
    header = f"{'Nivel':30s}"
    for offer in offers:
        header += f" | {offer['offer_level']:20s}"
    lines.append(header)
    lines.append("-" * 100)
    
    # Total prices
    totals = f"{'TOTAL (EUR)':30s}"
    for offer in offers:
        total = offer['summary']['total_price_eur']
        totals += f" | {total:20,.2f}"
    lines.append(totals)
    
    lines.append("-" * 100)
    
    # Categories included
    all_categories = set()
    for offer in offers:
        all_categories.update(offer['categories'].keys())
    
    for cat in sorted(all_categories):
        row = f"{cat:30s}"
        for offer in offers:
            if cat in offer['categories']:
                cost = offer['categories'][cat].get('total_cost_eur', 0)
                row += f" | {cost:20,.2f}"
            else:
                row += f" | {'—':>20s}"
        lines.append(row)
    
    lines.append("="*100 + "\n")
    
    return "\n".join(lines)


def export_pricing_to_csv(pricing_data: dict, output_path: Path):
    """
    Exportă datele de pricing în format CSV.
    """
    import csv
    
    rows = []
    rows.append(["Categorie", "Element", "Cantitate", "Unitate", "Preț unitar (EUR)", "Total (EUR)"])
    
    for cat_key, cat_data in pricing_data.items():
        if cat_key == "total_cost_eur" or not isinstance(cat_data, dict):
            continue
        
        cat_name = cat_data.get("category", cat_key)
        
        for item in cat_data.get("items", []):
            rows.append([
                cat_name,
                item["name"],
                f"{item['quantity']:.2f}",
                item["unit"],
                f"{item['unit_price_eur']:.2f}",
                f"{item['total_cost_eur']:.2f}",
            ])
    
    # Total row
    total = pricing_data.get("total_cost_eur", 0)
    rows.append(["", "", "", "", "TOTAL", f"{total:.2f}"])
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"   📊 CSV exported: {output_path}")


def calculate_price_per_sqm(pricing_data: dict, floor_area: float) -> dict:
    """
    Calculează prețul pe m² pentru fiecare categorie.
    """
    if floor_area <= 0:
        return {}
    
    result = {}
    
    for cat_key, cat_data in pricing_data.items():
        if cat_key == "total_cost_eur" or not isinstance(cat_data, dict):
            continue
        
        cat_total = cat_data.get("total_cost_eur", 0)
        price_per_sqm = cat_total / floor_area
        
        result[cat_key] = {
            "total_cost_eur": cat_total,
            "price_per_sqm_eur": round(price_per_sqm, 2),
        }
    
    total = pricing_data.get("total_cost_eur", 0)
    result["total"] = {
        "total_cost_eur": total,
        "price_per_sqm_eur": round(total / floor_area, 2),
    }
    
    return result


def generate_offer_report(
    offer: dict,
    floor_area: float,
    output_path: Path | None = None
) -> str:
    """
    Generează un raport detaliat pentru o ofertă.
    """
    lines = []
    lines.append("\n" + "="*70)
    lines.append(f"📋 RAPORT OFERTĂ: {offer['offer_level']}")
    lines.append("="*70)
    
    # Summary
    summary = offer['summary']
    lines.append(f"\n💰 Cost total: {summary['total_price_eur']:,.2f} EUR")
    
    if floor_area > 0:
        price_per_sqm = summary['total_price_eur'] / floor_area
        lines.append(f"📐 Preț per m²: {price_per_sqm:,.2f} EUR/m²")
        lines.append(f"📏 Suprafață: {floor_area:.2f} m²")
    
    if summary.get('premium_markup_eur', 0) > 0:
        lines.append(f"✨ Markup premium: +{summary['premium_markup_eur']:,.2f} EUR")
    
    # Categories
    lines.append("\n" + "-"*70)
    lines.append("📦 CATEGORII INCLUSE")
    lines.append("-"*70)
    
    for cat_key, cat_data in offer['categories'].items():
        cat_name = cat_data.get('category', cat_key)
        cat_total = cat_data.get('total_cost_eur', 0)
        percentage = (cat_total / summary['total_price_eur'] * 100) if summary['total_price_eur'] > 0 else 0
        
        lines.append(f"\n   {cat_name:30s} {cat_total:10,.2f} EUR ({percentage:5.1f}%)")
        
        for item in cat_data.get('items', []):
            lines.append(f"      • {item['name']:25s} {item['quantity']:6.2f} {item['unit']}")
    
    lines.append("\n" + "="*70 + "\n")
    
    report = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"   📄 Report saved: {output_path}")
    
    return report