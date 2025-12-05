# holzbot-engine/validator.py
import sys
import json
import os
import requests
import fitz  # PyMuPDF
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

# Config
MAX_DIMENSION = 1500  # Limita cerută de tine (anulăm dacă e mai mare)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def validate_plan(file_url):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # 1. Download File
        response = requests.get(file_url)
        response.raise_for_status()
        file_bytes = response.content
        content_type = response.headers.get('Content-Type', '').lower()
        
        image_data = None
        img_width = 0
        img_height = 0

        # 2. Procesare PDF sau Imagine
        if 'pdf' in content_type or file_url.endswith('.pdf'):
            # Convertim PDF (prima pagină) -> Imagine
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc.load_page(0)  # Prima pagină
                pix = page.get_pixmap()
                img_width, img_height = pix.width, pix.height
                
                # Convertim în format compatibil PIL/OpenAI
                image_data = BytesIO(pix.tobytes("png"))
            except Exception as e:
                return {"valid": False, "reason": f"PDF Corrupt: {str(e)}"}
        else:
            # Este deja imagine
            try:
                img = Image.open(BytesIO(file_bytes))
                img_width, img_height = img.size
                image_data = BytesIO(file_bytes)
            except Exception as e:
                return {"valid": False, "reason": "Invalid Image Format"}

        # 3. Verificare Dimensiuni (CRITERIUL TĂU: > 1000px ANULĂM)
        # Notă: De obicei planurile sunt mari. Ești sigur că vrei să anulezi dacă e MARE?
        # Codul de mai jos respectă cerința ta strictă.
        if img_width > MAX_DIMENSION or img_height > MAX_DIMENSION:
            return {
                "valid": False, 
                "reason": f"Plan zu groß ({img_width}x{img_height}) px. Die maximale Größe ist ({MAX_DIMENSION}x{MAX_DIMENSION}) px."
            }

        # 4. Pregătire pentru OpenAI (Base64)
        image_data.seek(0)
        base64_image = base64.b64encode(image_data.read()).decode('utf-8')
        data_url = f"data:image/png;base64,{base64_image}"

        # 5. Apel OpenAI Vision
        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an architect assistant. Verify if the image is a usable floor plan for detecting sizes.
                    Check for:
                    1. Room labels (Text inside rooms like Living, Bedroom, Bad) with square meter sizes.
                    2. Clear walls structure.
                    
                    If it is a valid plan, return valid=true.
                    If it looks like a photo, a generic sketch without text, or unrelated, valid=false.
                    
                    Return ONLY JSON: { "valid": boolean, "reason": "string" }"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Validate this plan."},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
                    ]
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        result_text = chat_response.choices[0].message.content
        return json.loads(result_text)

    except Exception as e:
        return {"valid": False, "reason": f"System Error: {str(e)}"}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"valid": False, "reason": "No URL provided"}))
        sys.exit(1)
    
    url = sys.argv[1]
    result = validate_plan(url)
    print(json.dumps(result))