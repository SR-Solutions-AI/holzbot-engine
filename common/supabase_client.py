# common/supabase_client.py
import os
from supabase import create_client, Client

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        # Fallback pentru dezvoltare locală dacă nu e setat în env
        raise ValueError("❌ EROARE: Lipsesc variabilele SUPABASE_URL sau SUPABASE_SERVICE_ROLE_KEY din .env")

    return create_client(url, key)