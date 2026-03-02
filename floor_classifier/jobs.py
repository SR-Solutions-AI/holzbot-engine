# new/runner/floor_classifier/jobs.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from .openai_classifier import classify_floors_with_openai


@dataclass
class FloorClassificationResult:
    plan_id: str
    floor_type: str  # ground_floor | top_floor | intermediate | unknown
    confidence: str  # high | medium | low
    reasoning: str
    metadata_file: Path
    estimated_area_m2: float | None
    door_count_exterior: int | None


def run_floor_classification(
    job_root: Path,
    plans: List  # List[ClassifiedPlanInfo] din orchestrator
) -> List[FloorClassificationResult]:
    """
    Clasifică etajele pentru toate planurile house_blueprint.
    
    Flow:
      1. Filtrează doar house_blueprint
      2. Trimite toate imaginile la GPT-4o
      3. Salvează metadata per plan în job_root/plan_metadata/
      4. Returnează rezultatele
    
    Args:
        job_root: Rădăcina job-ului (ex: jobs/segmentation_job_20251118_...)
        plans: Listă de ClassifiedPlanInfo din orchestrator
    
    Returns:
        Listă de FloorClassificationResult
    """
    
    # Filtrează doar house_blueprint
    house_plans = [p for p in plans if p.label == "house_blueprint"]
    
    if not house_plans:
        print("\nℹ️  Niciun plan house_blueprint - sar peste clasificare etaje.\n")
        return []
    
    print(f"\n{'='*70}")
    print(f"🏢 CLASIFICARE ETAJE")
    print(f"{'='*70}")
    print(f"📋 Planuri house_blueprint: {len(house_plans)}")
    
    # Pregătește input pentru AI: (plan_id, image_path)
    plans_input = [
        (p.image_path.stem, p.image_path)
        for p in house_plans
    ]
    
    # Apel AI
    try:
        ai_result = classify_floors_with_openai(plans_input)
    except Exception as e:
        print(f"\n❌ Eroare la clasificare: {e}")
        return []
    
    # Creează folder metadata
    metadata_dir = job_root / "plan_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Salvez metadata în: {metadata_dir}")
    
    # Salvează rezultate
    results: List[FloorClassificationResult] = []
    classifications_list = ai_result.get("classifications") or []

    for idx, plan in enumerate(house_plans):
        stem = plan.image_path.stem

        # Găsește clasificarea AI: mai întâi după plan_id (stem), apoi după index (când stem e duplicat)
        classification = next(
            (c for c in classifications_list if c["plan_id"] == stem),
            None
        )
        if classification is None and idx < len(classifications_list):
            classification = classifications_list[idx]
            print(f"  ℹ️  Match după index pentru {stem} (AI a returnat plan_id={classification.get('plan_id')!r})")

        if not classification:
            print(f"  ⚠️  Nu am primit clasificare pentru {stem}")
            continue

        # Nume unic pentru fișier: plan_01_cluster_1, plan_02_cluster_1 (același format ca plan_id din load_plan_infos)
        plan_id_file = f"plan_{idx + 1:02d}_{stem}"

        # Creează metadata file
        metadata = {
            "plan_id": plan_id_file,
            "plan_image": str(plan.image_path),
            "label": plan.label,
            "floor_classification": {
                "floor_type": classification["floor_type"],
                "confidence": classification["confidence"],
                "reasoning": classification["reasoning"],
                "indicators_found": classification.get("indicators_found", []),
                "estimated_area_m2": classification.get("estimated_area_m2"),
                "door_count_exterior": classification.get("door_count_exterior"),
                "stair_direction": classification.get("stair_direction"),
                "classified_at": datetime.utcnow().isoformat() + "Z"
            },
            # Placeholder-e pentru etapele următoare
            "scale": None,
            "detections": None,
            "measurements": None,
            "area": None,
            "pricing": None
        }
        
        metadata_file = metadata_dir / f"{plan_id_file}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Emoji pentru floor type
        emoji_map = {
            "ground_floor": "🏠",
            "top_floor": "🏡",
            "intermediate": "🏢",
            "unknown": "❓"
        }
        emoji = emoji_map.get(classification["floor_type"], "📄")
        
        results.append(FloorClassificationResult(
            plan_id=plan_id_file,
            floor_type=classification["floor_type"],
            confidence=classification["confidence"],
            reasoning=classification["reasoning"],
            metadata_file=metadata_file,
            estimated_area_m2=classification.get("estimated_area_m2"),
            door_count_exterior=classification.get("door_count_exterior")
        ))
        
        print(f"  {emoji} {plan_id_file}:")
        print(f"     → {classification['floor_type'].replace('_', ' ').title()}")
        print(f"     → Confidence: {classification['confidence']}")
        if classification.get("estimated_area_m2"):
            print(f"     → Area: ~{classification['estimated_area_m2']:.0f} m²")
        if classification.get("door_count_exterior") is not None:
            print(f"     → Exterior doors: {classification['door_count_exterior']}")
    
    # Salvează și rezumatul general
    summary_file = metadata_dir / "_floor_classification_summary.json"
    summary_data = {
        **ai_result,
        "metadata_files": [str(r.metadata_file) for r in results],
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Afișare rezumat validare
    validation = ai_result.get("validation", {})
    print(f"\n{'─'*70}")
    print("📊 VALIDARE CLASIFICARE:")
    print(f"{'─'*70}")
    print(f"  ✓ Ground floor identificat: {validation.get('has_ground_floor', False)}")
    print(f"  ✓ Top floor identificat: {validation.get('has_top_floor', False)}")
    
    if validation.get("ground_floor_plan_id"):
        print(f"  🏠 Ground floor: {validation['ground_floor_plan_id']}")
    if validation.get("top_floor_plan_id"):
        print(f"  🏡 Top floor: {validation['top_floor_plan_id']}")
    
    warnings = validation.get("warnings", [])
    if warnings:
        print(f"\n  ⚠️  AVERTISMENTE:")
        for w in warnings:
            print(f"     • {w}")
    
    print(f"\n✅ Rezumat salvat în: {summary_file}")
    print(f"{'='*70}\n")
    
    return results