#!/usr/bin/env python3
"""
Script pentru a adăuga structura de formular în baza de date pentru holzbau@holzbot.com
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Adăugăm rădăcina proiectului la path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.supabase_client import get_supabase_client

def load_form_schema():
    """Încarcă schema de formular din formSchema.json"""
    schema_path = PROJECT_ROOT.parent / "holzbot-web" / "app" / "dashboard" / "formSchema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Nu găsesc formSchema.json la {schema_path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_holzbau_tenant_id(supabase):
    """Obține tenant_id pentru holzbau"""
    res = supabase.table("tenants").select("id, slug").eq("slug", "holzbau").single().execute()
    if not res.data:
        raise ValueError("Nu găsesc tenant-ul 'holzbau' în baza de date")
    return res.data["id"]

def main():
    print("🚀 Adăugare form definition pentru holzbau...")
    
    # 1. Conectare la Supabase
    supabase = get_supabase_client()
    
    # 2. Obține tenant_id pentru holzbau
    tenant_id = get_holzbau_tenant_id(supabase)
    print(f"✅ Tenant ID pentru holzbau: {tenant_id}")
    
    # 3. Încarcă schema de formular
    ui_schema = load_form_schema()
    print(f"✅ Schema de formular încărcată ({len(ui_schema.get('steps', []))} pași)")
    
    # 4. Verifică dacă există deja o definiție activă
    existing = supabase.table("form_definitions").select("id, version, is_active").eq("tenant_id", tenant_id).eq("is_active", True).execute()
    
    if existing.data:
        print(f"⚠️  Există deja {len(existing.data)} definiții active pentru holzbau:")
        for item in existing.data:
            print(f"   - ID: {item['id']}, Version: {item.get('version', 'N/A')}")
        response = input("Vrei să dezactivezi cele existente și să adaugi una nouă? (y/n): ")
        if response.lower() != 'y':
            print("❌ Operație anulată")
            return
        
        # Dezactivează toate definițiile existente
        for item in existing.data:
            supabase.table("form_definitions").update({"is_active": False}).eq("id", item["id"]).execute()
        print("✅ Definițiile existente au fost dezactivate")
    
    # 5. Determină versiunea următoare
    max_version_res = supabase.table("form_definitions").select("version").eq("tenant_id", tenant_id).order("version", desc=True).limit(1).execute()
    next_version = 1
    if max_version_res.data:
        next_version = max_version_res.data[0].get("version", 0) + 1
    
    # 6. Inserează noua definiție
    insert_data = {
        "tenant_id": tenant_id,
        "version": next_version,
        "is_active": True,
        "ui_schema": ui_schema,
        "offer_type_id": None,  # Formular generic (fără offer_type specific)
    }
    
    result = supabase.table("form_definitions").insert(insert_data).execute()
    
    if result.data:
        print(f"✅ Form definition adăugată cu succes!")
        print(f"   - ID: {result.data[0]['id']}")
        print(f"   - Version: {next_version}")
        print(f"   - Is Active: True")
    else:
        print("❌ Eroare la inserare")
        if hasattr(result, 'error') and result.error:
            print(f"   Eroare: {result.error}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Eroare: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




