import requests
import os
from pathlib import Path

# Citește din .env sau config
try:
    # Opțiunea 1: Citește din .env
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
except:
    # Opțiunea 2: Pune cheia direct (temporar, pentru test)
    api_key = "AIza..."  # pune cheia ta aici

print(f"API Key: {api_key[:10]}..." if api_key else "API Key: None")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)
print(response.json())