# new/runner/exterior_doors/classify.py (VERSIUNEA CORECTĂ - CU OVERLAY)
from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np


def _load_gray(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Lipsește/invalid: {path}")
    return m


def _extract_blue_mask_from_overlay(overlay_path: Path) -> np.ndarray:
    """
    Extrage masca binară a zonelor ALBASTRE din blue_overlay.jpg.
    
    Blue overlay = planul original + overlay ALBASTRU (BGR: 255,0,0) pe exterior
    
    Returns:
        Mască binară: 255 = ALBASTRU (exterior), 0 = REST
    """
    overlay = cv2.imread(str(overlay_path))
    if overlay is None:
        raise FileNotFoundError(f"Lipsește/invalid: {overlay_path}")
    
    # Extragem canalul ALBASTRU (BGR: B=255, G=0, R=0)
    # Pixelii albaștri pur au B=255, G=0, R=0
    
    b_channel = overlay[:, :, 0]  # Blue channel
    g_channel = overlay[:, :, 1]  # Green channel
    r_channel = overlay[:, :, 2]  # Red channel
    
    # Mască: ALBASTRU pur (B=255 și G<50 și R<50)
    blue_mask = ((b_channel > 200) & (g_channel < 50) & (r_channel < 50)).astype(np.uint8) * 255
    
    return blue_mask


def _bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    """Calculează lungimea diagonalei bbox."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return np.sqrt(w**2 + h**2)


def _distance_to_blue(mask_blue: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """
    Calculează distanța minimă de la bbox la cea mai apropiată zonă ALBASTRĂ.
    
    Args:
        mask_blue: Mască binară (255 = albastru/exterior, 0 = rest)
        bbox: (x1, y1, x2, y2)
    
    Returns:
        Distanța în pixeli (inf dacă nu găsește albastru)
    """
    H, W = mask_blue.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Verifică dacă bbox e valid
    if x1 >= W or x2 <= 0 or y1 >= H or y2 <= 0:
        return float('inf')
    
    # Clamp bbox
    x1 = max(0, min(x1, W-1))
    x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1))
    y2 = max(0, min(y2, H-1))
    
    if x2 <= x1 or y2 <= y1:
        return float('inf')
    
    # Expandăm bbox pentru căutare
    margin = 50
    x1_exp = max(0, x1 - margin)
    x2_exp = min(W - 1, x2 + margin)
    y1_exp = max(0, y1 - margin)
    y2_exp = min(H - 1, y2 + margin)
    
    # Extragem regiunea
    region = mask_blue[y1_exp:y2_exp+1, x1_exp:x2_exp+1]
    
    if region.size == 0:
        return float('inf')
    
    # Verificăm dacă există albastru în regiune
    if cv2.countNonZero(region) == 0:
        return float('inf')
    
    # Distance transform
    # mask_blue are 255 = ALBASTRU (destinație)
    # Inversăm: 0 = destinație, 255 = obstacol
    inv_region = cv2.bitwise_not(region)
    
    dist_transform = cv2.distanceTransform(inv_region, cv2.DIST_L2, 5)
    
    # Extragem zona bbox-ului ORIGINAL (fără margin)
    y_start = y1 - y1_exp
    y_end = y2 - y1_exp
    x_start = x1 - x1_exp
    x_end = x2 - x1_exp
    
    y_start = max(0, y_start)
    y_end = min(dist_transform.shape[0], y_end)
    x_start = max(0, x_start)
    x_end = min(dist_transform.shape[1], x_end)
    
    bbox_region = dist_transform[y_start:y_end, x_start:x_end]
    
    if bbox_region.size == 0:
        return float('inf')
    
    return float(np.min(bbox_region))


def classify_exterior_doors(
    plan_image: Path,
    blue_mask_path: Path,
    detections_all_json: Path,
    out_dir: Path,
    job_root: Path | None = None,
    original_plan_name: str | None = None
) -> tuple[Path, Path, Path]:
    """
    Clasifică ușile ca exterior/interior.
    
    REGULA:
    Pentru fiecare ușă:
      - distanță ≤ diagonală/2 → EXTERIOR
      - distanță > diagonală/2 → INTERIOR
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "exterior_doors.json"
        out_overlay = out_dir / "exterior_doors_detected.jpg"
        out_flood_marked = out_dir / "exterior_doors_flood_marked.jpg"
        
        # Load plan
        plan = cv2.imread(str(plan_image))
        if plan is None:
            raise RuntimeError(f"plan.jpg invalid: {plan_image}")
        
        H, W = plan.shape[:2]
        
        # ==========================================
        # IMPORTANT: EXTRAGEM MASCA DIN BLUE_OVERLAY.JPG
        # ==========================================
        blue_overlay_path = out_dir / "blue_overlay.jpg"
        
        if not blue_overlay_path.exists():
            raise FileNotFoundError(f"Nu găsesc blue_overlay.jpg: {blue_overlay_path}")
        
        print(f"       📐 Extrag mască ALBASTRĂ din blue_overlay.jpg...")
        mask_blue = _extract_blue_mask_from_overlay(blue_overlay_path)
        
        # Verifică câte pixeli albastre sunt
        total_blue = cv2.countNonZero(mask_blue)
        blue_percent = (total_blue / (mask_blue.shape[0] * mask_blue.shape[1])) * 100
        print(f"       📐 Pixeli albaștri: {total_blue:,} ({blue_percent:.1f}%)")
        
        if total_blue == 0:
            print(f"       ❌ ATENȚIE: Nu există zone albastre în overlay!")
        
        dets = json.loads(detections_all_json.read_text(encoding="utf-8"))
        
        # Prepare overlays
        overlay_doors = plan.copy()
        overlay_flood = cv2.imread(str(blue_overlay_path))  # Pornim de la blue_overlay
        
        # ==========================================
        # PROCESEAZĂ FIECARE UȘĂ
        # ==========================================
        
        results = []
        idx = 0
        
        for d in dets:
            typ = (d.get("type") or "").lower()
            status_det = (d.get("status") or "").lower()
            
            if "door" not in typ:
                continue
            if status_det == "rejected":
                continue
            
            idx += 1
            
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
            bbox = (x1, y1, x2, y2)
            
            # Calculează distanță și diagonală
            diagonal = _bbox_diagonal(bbox)
            distance = _distance_to_blue(mask_blue, bbox)
            
            # REGULA:
            max_allowed_distance = diagonal / 2.0
            is_exterior = (distance <= max_allowed_distance)
            
            status = "exterior" if is_exterior else "interior"
            color = (0, 0, 255) if is_exterior else (0, 200, 0)
            
            # Desenează
            cv2.rectangle(overlay_doors, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                overlay_doors,
                f"#{idx} {status[:3].upper()} d={distance:.1f}px",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA
            )
            
            cv2.rectangle(overlay_flood, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                overlay_flood,
                f"#{idx}",
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
            
            results.append({
                "id": idx,
                "type": typ,
                "status": status,
                "bbox": [x1, y1, x2, y2],
                "distance_to_blue_px": round(float(distance), 2) if distance != float('inf') else "infinity",
                "diagonal_px": round(float(diagonal), 2),
                "max_allowed_distance_px": round(float(max_allowed_distance), 2),
                "rule": f"distance ({distance:.1f}) {'<=' if is_exterior else '>'} diagonal/2 ({max_allowed_distance:.1f})"
            })
        
        # Salvează
        cv2.imwrite(str(out_overlay), overlay_doors)
        cv2.imwrite(str(out_flood_marked), overlay_flood)
        out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        
        ext_count = sum(1 for r in results if r['status']=='exterior')
        int_count = len(results) - ext_count
        
        print(f"       ✅ Exterior doors: {ext_count}/{len(results)}")
        print(f"       🏠 Interior doors: {int_count}/{len(results)}")
        print(f"       📄 {out_json.name}")
        print(f"       🖼️  {out_overlay.name}")
        print(f"       🔵 {out_flood_marked.name}")
        
        return out_json, out_overlay, out_flood_marked
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"       ❌ Eroare: {e}")
        return None, None, None