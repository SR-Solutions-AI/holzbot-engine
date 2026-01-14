# verify_db.py
import os
import json
from dotenv import load_dotenv

# Încărcăm variabilele din .env
load_dotenv()

try:
    print("🔌 1. Testare importuri...")
    from pricing.db_loader import fetch_pricing_parameters
    print("   ✅ Importuri OK.")

    print("\n🌍 2. Conectare la Supabase și descărcare prețuri...")
    # Încercăm să tragem prețurile pentru clientul nostru
    coeffs = fetch_pricing_parameters("chiemgauer", calc_mode="default")
    
    print("   ✅ Conexiune REUȘITĂ!")
    print(f"   📦 S-au descărcat {len(coeffs)} categorii principale de prețuri.")

    # Afișăm câteva valori ca să fim siguri
    print("\n🔍 3. Verificare date (Eșantion):")
    
    # Verificăm fundația
    foundation_price = coeffs.get('foundation', {}).get('unit_price_per_m2', {}).get('Placă')
    print(f"   - Preț Placă Beton: {foundation_price} EUR/m2 (Ar trebui să fie 210.0)")
    
    # Verificăm ferestrele
    window_price = coeffs.get('openings', {}).get('windows_unit_prices_per_m2', {}).get('PVC')
    print(f"   - Preț Fereastră PVC: {window_price} EUR/m2 (Ar trebui să fie 550.0)")

    # Verificăm CLT
    clt_price = coeffs.get('system', {}).get('base_unit_prices', {}).get('CLT', {}).get('exterior')
    print(f"   - Preț CLT Exterior: {clt_price} EUR/m2 (Ar trebui să fie 275.0)")

    print("\n🎉 SUPER! Engine-ul este conectat la baza de date.")

except Exception as e:
    print(f"\n❌ EROARE CRITICĂ: {e}")
    print("Verifică dacă ai instalat 'supabase' (pip install supabase) și dacă fișierul .env este corect.")