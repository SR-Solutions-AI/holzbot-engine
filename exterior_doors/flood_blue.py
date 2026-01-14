# new/runner/exterior_doors/flood_blue.py
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def compute_blue_mask(plan_image: Path, out_dir: Path) -> tuple[Path, Path]:
    """
    Generează masca ALBASTRĂ (EXTERIOR) prin flood fill de la margini.
    
    IMPORTANT: Flood fill-ul rămâne DOAR în exterior, NU modifică culorile din plan.
    
    Returns:
        (blue_mask_path, blue_overlay_path)
        
        blue_mask.png:
          - Alb (255) = EXTERIOR (zona accesibilă de la margini)
          - Negru (0) = INTERIOR (protejat de pereți)
        
        blue_overlay.jpg:
          - Plan original + overlay ALBASTRU pe zona exterioară
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mask = out_dir / "blue_mask.png"
    out_overlay = out_dir / "blue_overlay.jpg"
    
    # Load plan
    plan = cv2.imread(str(plan_image))
    if plan is None:
        raise RuntimeError(f"plan.jpg invalid: {plan_image}")
    
    H, W = plan.shape[:2]
    gray = cv2.cvtColor(plan, cv2.COLOR_BGR2GRAY)
    
    print(f"       🖼️  Plan: {W}×{H}px")
    
    # ==========================================
    # THRESHOLD: detectăm pereții (negru/întunecat)
    # ==========================================
    
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # ==========================================
    # FLOOD FILL de la margini (AJUSTAT)
    # ==========================================
    
    flood_work = binary.copy()
    flood_mask = np.zeros((H + 2, W + 2), np.uint8)
    
    # Seed points MAI RARI (step mai mare = mai puține puncte)
    seed_points = []
    step = 30  # ← mai rar (era 15)
    
    for x in range(0, W, step):
        seed_points.append((x, 0))
        seed_points.append((x, H - 1))
    
    for y in range(0, H, step):
        seed_points.append((0, y))
        seed_points.append((W - 1, y))
    
    flooded_count = 0
    for (x, y) in seed_points:
        if 0 <= x < W and 0 <= y < H:
            if flood_work[y, x] == 255:
                cv2.floodFill(
                    flood_work, 
                    flood_mask, 
                    seedPoint=(x, y), 
                    newVal=128,
                    loDiff=10,
                    upDiff=10
                )
                flooded_count += 1
    
    print(f"       🌊 Flood fill: {flooded_count}/{len(seed_points)} seed points active")
    
    blue_mask = (flood_work == 128).astype(np.uint8) * 255
    
    # ==========================================
    # POST-PROCESSING: Erodare MAI AGRESIVĂ
    # ==========================================
    
    # Eliminăm insulițe mici
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # era (5,5)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Erodăm MAI MULT
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # era (2,2)
    blue_mask = cv2.erode(blue_mask, kernel_erode, iterations=2)  # era iterations=1
    
    print(f"       🔧 Post-processing: eliminare zgomot + erodare agresivă")
    
    # ==========================================
    # SAVE
    # ==========================================
    
    cv2.imwrite(str(out_mask), blue_mask)
    
    overlay = plan.copy()
    overlay[blue_mask > 0] = [255, 0, 0]  # BGR: ALBASTRU
    cv2.imwrite(str(out_overlay), overlay)
    
    blue_pixels = cv2.countNonZero(blue_mask)
    blue_percent = (blue_pixels / (H * W)) * 100
    
    print(f"       ✅ Blue mask (EXTERIOR): {blue_percent:.1f}% din arie ({blue_pixels:,} px)")
    print(f"       📄 {out_mask.name}")
    print(f"       🖼️  {out_overlay.name} (ALBASTRU = exterior)")
    
    return out_mask, out_overlay