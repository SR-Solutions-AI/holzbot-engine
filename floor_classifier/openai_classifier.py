# new/runner/floor_classifier/openai_classifier.py
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI


FLOOR_CLASSIFICATION_PROMPT = """
You are an expert architectural plan analyst specializing in multi-story residential buildings.

You will receive N floor plan images from the SAME building project.
Your task is to classify each plan into ONE of these floor types:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 CLASSIFICATION CATEGORIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ **ground_floor** (Parter / Erdgeschoss / Ground Floor)
   PRIMARY INDICATORS:
   ✓ Text labels: "PARTER", "GROUND FLOOR", "EG", "ERDGESCHOSS", "P", "GF"
   ✓ Main entrance door clearly visible (often thicker/double door)
   ✓ Larger overall floor area (ground floors are typically bigger)
   ✓ More rooms and corridors
   ✓ Garage/carport attached or nearby
   ✓ Outdoor terraces, patios directly accessible
   
   STAIRCASE CLUES:
   ⚠️ Stairs are PRESENT but going UP only (arrow pointing up, or "↑" symbol)
   ⚠️ Staircase starts from this floor
   
   SECONDARY INDICATORS:
   ⚠️ More exterior doors (2-4 doors vs 0-1 on upper floors)
   ⚠️ Kitchen location (ground floor kitchens are typically larger)
   ⚠️ Living/dining areas dominate
   ⚠️ Utility rooms (laundry, storage) more common

2️⃣ **top_floor** (Etaj / Obergeschoss / Upper Floor)
   PRIMARY INDICATORS:
   ✓ Text labels: "ETAJ", "MANSARDA", "OG", "OBERGESCHOSS", "ATTIC", "1. ETAJ", "FLOOR 1"
   ✓ Smaller overall floor area (often 70-85% of ground floor size)
   ✓ Sloped/angled walls indicating roof structure
   ✓ Roof windows (Velux-style) marked on plan
   ✓ Fewer rooms (typically bedrooms concentrated here)
   
   STAIRCASE CLUES:
   ⚠️ Stairs are PRESENT but ENDING here (arrow pointing down "↓", or no upward continuation)
   ⚠️ Staircase terminates at this level
   
   SECONDARY INDICATORS:
   ⚠️ Fewer or NO exterior doors (usually 0-1 doors, often to balcony)
   ⚠️ Bathrooms are smaller/simpler
   ⚠️ Bedrooms dominate the layout
   ⚠️ Master bedroom with en-suite bathroom common
   ⚠️ Less circulation space (smaller hallways)

3️⃣ **intermediate** (Etaj intermediar)
   PRIMARY INDICATORS:
   ✓ Text labels: "ETAJ 1", "ETAJ 2", "1. OG", "2. OG"
   ✓ Staircase with BOTH up and down arrows (↑↓)
   ✓ No roof elements, no foundation elements
   ✓ Medium-sized floor area (between ground and top)
   
   RARE in residential buildings (most houses are 2-story: ground + top)

4️⃣ **unknown** (Cannot determine)
   Use ONLY if:
   ✗ Plan quality too poor to read
   ✗ No text labels AND no clear architectural features
   ✗ Ambiguous indicators that contradict each other

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 CRITICAL DECISION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **BOTH ground and top floors have stairs visible!**
   → Don't rely on staircase presence alone
   → Focus on: stair direction, floor area size, door count, text labels

⚠️ **If only 1 plan total:**
   → MUST classify as "ground_floor" (single-story building)

⚠️ **If only 2 plans total:**
   → EXACTLY one MUST be "ground_floor"
   → EXACTLY one MUST be "top_floor"
   → Use area size + door count as tiebreaker:
      • Larger area + more doors = ground_floor
      • Smaller area + fewer doors = top_floor

⚠️ **If 3+ plans:**
   → EXACTLY one "ground_floor" (largest, most doors)
   → EXACTLY one "top_floor" (smallest, roof elements)
   → Others = "intermediate"

⚠️ **Door count heuristic (very reliable):**
   • Ground floor: 2-4+ exterior doors
   • Top floor: 0-1 exterior doors (often balcony access only)
   • Count doors carefully!

⚠️ **Area size heuristic (reliable):**
   • Ground floor: 100-150 m² typical
   • Top floor: 70-120 m² typical (often 10-30% smaller)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📤 OUTPUT FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return STRICTLY this JSON structure (no additional text):

{
  "building_info": {
    "total_plans": <int>,
    "floors_detected": <int>,
    "is_single_story": <bool>,
    "building_type": "residential | commercial | mixed"
  },
  "classifications": [
    {
      "plan_id": "<string>",
      "floor_type": "ground_floor | top_floor | intermediate | unknown",
      "confidence": "high | medium | low",
      "reasoning": "<string: explain why this classification, mention key indicators>",
      "indicators_found": [
        "<list of specific visual clues: text labels, door count, area estimate, etc>"
      ],
      "estimated_area_m2": <float or null>,
      "door_count_exterior": <int or null>,
      "stair_direction": "up | down | both | none"
    }
  ],
  "validation": {
    "has_ground_floor": <bool>,
    "has_top_floor": <bool>,
    "ground_floor_plan_id": "<string or null>",
    "top_floor_plan_id": "<string or null>",
    "warnings": ["<list of any inconsistencies detected>"]
  }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 ANALYSIS APPROACH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. First pass: Read ALL text labels on all plans
2. Second pass: Count exterior doors on each plan
3. Third pass: Estimate relative floor areas (rank them)
4. Fourth pass: Check staircase directions
5. Decision: Combine all indicators with weights:
   • Text labels: 40% weight
   • Door count: 30% weight
   • Floor area: 20% weight
   • Other indicators: 10% weight

Be thorough, precise, and confident in your classifications!
"""


def classify_floors_with_openai(
    plans: List[Tuple[str, Path]],  # [(plan_id, image_path), ...]
) -> dict:
    """
    Trimite toate planurile house_blueprint la GPT-4o pentru clasificare etaje.
    
    Args:
        plans: Listă de tupluri (plan_id, path_către_imagine)
    
    Returns:
        {
          "building_info": {...},
          "classifications": [...],
          "validation": {...}
        }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY lipsește din environment")
    
    client = OpenAI(api_key=api_key)
    
    print(f"  🧠 Trimit {len(plans)} planuri către GPT-4o pentru clasificare...")
    
    # Construiește mesajul cu toate imaginile
    content = [{"type": "text", "text": FLOOR_CLASSIFICATION_PROMPT}]
    
    for idx, (plan_id, img_path) in enumerate(plans, start=1):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        content.append({
            "type": "text",
            "text": f"\n{'='*60}\nPlan #{idx} - ID: {plan_id}\n{'='*60}"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "high"  # IMPORTANT: high detail pentru text recognition
            }
        })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert architectural analyst specializing in residential floor plan interpretation."
                },
                {"role": "user", "content": content}
            ],
            temperature=0,
            max_tokens=3000
        )
    except Exception as e:
        raise RuntimeError(f"Eroare la apelul OpenAI: {e}")
    
    reply = response.choices[0].message.content.strip()
    
    # Curăță JSON (elimină markdown code fences)
    if reply.startswith("```json"):
        reply = reply[7:].strip()
    elif reply.startswith("```"):
        reply = reply[3:].strip()
    
    if reply.endswith("```"):
        reply = reply[:-3].strip()
    
    try:
        result = json.loads(reply)
    except json.JSONDecodeError as e:
        print("⚠️  Răspuns invalid de la GPT-4o:")
        print(reply[:500])
        raise ValueError(f"Nu pot parsa JSON-ul returnat de GPT-4o: {e}")
    
    # Validare structură răspuns
    if "classifications" not in result:
        raise ValueError("Răspunsul GPT-4o nu conține cheia 'classifications'")
    
    print(f"  ✅ Clasificare completă: {len(result['classifications'])} planuri procesate")
    
    return result