# new/runner/offer_builder.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

def build_final_offer(
    pricing_data: dict, 
    offer_level: str, 
    output_path: Path
) -> dict:
    """
    Generează oferta finală detaliată (final_offer.json).
    """
    
    INCLUSIONS = {
        "Structură": {
            "foundation": True,
            "structure": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": False,
            "finishes": False,
            "utilities": False  # ✨ NOU
        },
        "Structură + ferestre": {
            "foundation": True,
            "structure": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": True,
            "finishes": False,
            "utilities": False  # ✨ NOU
        },
        "Casă completă": {
            "foundation": True,
            "structure": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": True,
            "finishes": True,
            "utilities": True  # ✨ NOU - Instalații incluse doar în pachetul complet
        }
    }
    
    selection = INCLUSIONS.get(offer_level, INCLUSIONS["Casă completă"])
    
    final_json = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "offer_level": offer_level
        },
        "summary": {
            "total_price_eur": 0.0,
            "currency": "EUR"
        },
        "detailed_breakdown": [],
        "excluded_options": []
    }
    
    total_price = 0.0
    
    categories_order = [
        "foundation", 
        "structure", 
        "floors_ceilings", 
        "roof", 
        "openings", 
        "finishes",
        "utilities"  # ✨ NOU
    ]
    
    category_names_ro = {
        "foundation": "Fundație",
        "structure": "Structură Pereți",
        "floors_ceilings": "Planșee și Tavane",
        "roof": "Acoperiș",
        "openings": "Tâmplărie (Uși/Ferestre)",
        "finishes": "Finisaje",
        "utilities": "Utilități & Instalații"  # ✨ NOU
    }
    
    breakdown = pricing_data.get("breakdown", {})
    
    for cat_key in categories_order:
        data = breakdown.get(cat_key, {})
        if not data:
            continue
        
        is_included = selection.get(cat_key, False)
        items = data.get("items") or data.get("detailed_items") or []
        
        if not items and data.get("total_cost", 0) > 0:
            items = [{
                "name": category_names_ro[cat_key],
                "details": "Cost estimat global",
                "total_cost": data["total_cost"]
            }]
        
        if is_included:
            total_price += data.get("total_cost", 0.0)
            
            final_json["detailed_breakdown"].append({
                "category": category_names_ro[cat_key],
                "total_category": data.get("total_cost", 0.0),
                "items": items
            })
        else:
            final_json["excluded_options"].append({
                "category": category_names_ro[cat_key],
                "estimated_cost": data.get("total_cost", 0.0),
                "items_count": len(items)
            })
    
    final_json["summary"]["total_price_eur"] = round(total_price, 2)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    return final_json