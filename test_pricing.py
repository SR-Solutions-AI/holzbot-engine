"""
test_pricing.py
Script pentru testarea modulului de pricing și generare oferte.
"""
from pathlib import Path
import json

# Simulăm datele care ar veni din pipeline
MOCK_AREA_DATA = {
    "floor_area_sqm": 120.0,
    "wall_area_sqm": 280.0,
    "exterior_wall_area_sqm": 150.0,
    "interior_wall_area_sqm": 130.0,
    "foundation_area_sqm": 120.0,
}

MOCK_ROOF_DATA = {
    "roof_area_sqm": 160.0,
}

MOCK_COUNT_DATA = {
    "exterior_door": {"count": 2},
    "interior_door": {"count": 8},
    "window": {"count": 12},
}

MOCK_MEASURE_DATA = {
    "measured_objects": [
        {"class_name": "window", "width_m": 1.2, "height_m": 1.5, "area_sqm": 1.8},
        {"class_name": "window", "width_m": 1.0, "height_m": 1.2, "area_sqm": 1.2},
        {"class_name": "window", "width_m": 1.5, "height_m": 1.5, "area_sqm": 2.25},
        {"class_name": "window", "width_m": 1.2, "height_m": 1.5, "area_sqm": 1.8},
        {"class_name": "window", "width_m": 1.0, "height_m": 1.2, "area_sqm": 1.2},
        {"class_name": "window", "width_m": 1.5, "height_m": 1.5, "area_sqm": 2.25},
        {"class_name": "window", "width_m": 1.2, "height_m": 1.5, "area_sqm": 1.8},
        {"class_name": "window", "width_m": 1.0, "height_m": 1.2, "area_sqm": 1.2},
        {"class_name": "window", "width_m": 1.5, "height_m": 1.5, "area_sqm": 2.25},
        {"class_name": "window", "width_m": 1.2, "height_m": 1.5, "area_sqm": 1.8},
        {"class_name": "window", "width_m": 1.0, "height_m": 1.2, "area_sqm": 1.2},
        {"class_name": "window", "width_m": 1.5, "height_m": 1.5, "area_sqm": 2.25},
    ]
}


def test_pricing_engine():
    """Testează pricing engine-ul."""
    from runner.pricing.pricing_engine import PricingEngine
    
    print("\n" + "="*70)
    print("🧪 TEST: Pricing Engine")
    print("="*70)
    
    engine = PricingEngine()
    result = engine.calculate_complete_pricing(
        area_data=MOCK_AREA_DATA,
        roof_data=MOCK_ROOF_DATA,
        count_data=MOCK_COUNT_DATA,
        measure_data=MOCK_MEASURE_DATA
    )
    
    print("\n📊 Rezultat Pricing:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print(f"\n💰 TOTAL COST: {result['total_cost_eur']:,.2f} EUR")
    
    return result


def test_offer_builder(pricing_data: dict):
    """Testează builder-ul de oferte."""
    from runner.offer_builder import build_final_offer, OFFER_LEVELS
    
    print("\n" + "="*70)
    print("🧪 TEST: Offer Builder")
    print("="*70)
    
    for level_name in OFFER_LEVELS.keys():
        print(f"\n📋 Nivel: {level_name}")
        print("-" * 70)
        
        offer = build_final_offer(
            pricing_data=pricing_data,
            offer_level=level_name,
            output_path=None  # Nu salvăm în test
        )
        
        print(f"   Categorii incluse: {', '.join(offer['metadata']['included_categories'])}")
        print(f"   Cost de bază: {offer['summary']['base_cost_eur']:,.2f} EUR")
        if offer['summary']['premium_markup_eur'] > 0:
            print(f"   Markup premium: +{offer['summary']['premium_markup_eur']:,.2f} EUR")
        print(f"   💰 TOTAL: {offer['summary']['total_price_eur']:,.2f} EUR")


def test_complete_flow():
    """Testează flow-ul complet."""
    print("\n" + "="*80)
    print("🚀 TEST COMPLET: Pricing → Offer Generation")
    print("="*80)
    
    # Step 1: Calculăm pricing
    pricing_data = test_pricing_engine()
    
    # Step 2: Generăm oferte
    test_offer_builder(pricing_data)
    
    print("\n" + "="*80)
    print("✅ TEST COMPLET FINALIZAT")
    print("="*80)


if __name__ == "__main__":
    test_complete_flow()