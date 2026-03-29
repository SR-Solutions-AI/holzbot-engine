#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import google.generativeai as genai

ENGINE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ENGINE_ROOT / "output"
sys.path.insert(0, str(ENGINE_ROOT))

def _gemini_model():
    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"')
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing.")
    genai.configure(api_key=api_key)
    for model_name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
        try:
            return genai.GenerativeModel(model_name)
        except Exception:
            continue
    raise RuntimeError("No Gemini flash model available")


def _best_prediction(preds: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not preds:
        return None
    return max(preds, key=lambda p: float(p.get("confidence", 0.0)))


def _bbox_out_of_bounds_count(
    doors: list[dict[str, Any]], img_w: int, img_h: int, tx: int = 0, ty: int = 0
) -> int:
    bad = 0
    for d in doors:
        bb = d.get("bbox") if isinstance(d, dict) else None
        if not (isinstance(bb, list) and len(bb) == 4):
            bad += 1
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
        x1 -= tx
        x2 -= tx
        y1 -= ty
        y2 -= ty
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x2 <= x1 or y2 <= y1:
            bad += 1
    return bad


def _normalize_class(raw_label: str) -> str:
    s = (raw_label or "").strip().lower().replace("-", " ").replace("_", " ")
    if "stair" in s:
        return "stairs"
    if "window" in s or "geam" in s:
        return "window"
    if "door" in s or "usa" in s:
        return "door"
    return "door"


def _norm_text(s: str) -> str:
    return (
        (s or "")
        .lower()
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )


def _is_garage_room_label(room_type: str, room_name: str) -> bool:
    txt = f"{_norm_text(room_type)} {_norm_text(room_name)}"
    keys = ("garage", "carport", "car por", "car port", "garaj")
    return any(k in txt for k in keys)


def _room_rect(points: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [int(round(float(p[0]))) for p in points]
    ys = [int(round(float(p[1]))) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _is_near_garage_wall(
    x1: int, y1: int, x2: int, y2: int, garage_polys: list[list[list[int]]]
) -> bool:
    def _point_in_poly(px: int, py: int, pts: list[list[int]]) -> bool:
        inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            intersects = ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / float((yj - yi) or 1) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bb = (x1, y1, x2, y2)
    for poly_pts in garage_polys:
        # Avoid heavy geometry libs; use room rect + polygon center test.
        rx1, ry1, rx2, ry2 = _room_rect(poly_pts)
        if _point_in_poly(cx, cy, poly_pts):
            return True
        expanded = (rx1 - 25, ry1 - 25, rx2 + 25, ry2 + 25)
        if _rect_overlap(bb, expanded):
            return True
    return False


def _refine_class(cls: str, conf: float, reason: str, x1: int, y1: int, x2: int, y2: int) -> tuple[str, str]:
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    ar = w / float(h)
    area = w * h
    # False-positive guard: very thin/compact symbols are much more likely windows than stairs.
    if cls == "stairs":
        if ar > 2.8 or ar < 0.35 or min(w, h) < 18 or area < 1400:
            return "window", "stairs->window heuristic (thin/small bbox)"
        if "parallel" not in _norm_text(reason) and conf < 0.97:
            return "window", "stairs->window heuristic (missing stair pattern evidence)"
    return cls, ""


def _to_gray50(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.5, gray_bgr, 0.5, 0)


def _classify_with_gemini(gemini_model, crop_path: Path, context_plan_path: Path) -> dict[str, Any]:
    prompt = (
        "You are an expert in architectural floor plans.\n"
        "You receive two images:\n"
        "1) a zoomed crop around one candidate opening symbol\n"
        "2) the full house plan with ONLY that candidate region marked\n"
        "Use both images together for context.\n"
        "Classify it into EXACTLY ONE class from: door, window, stairs.\n"
        "Door includes single-door symbols and double-door symbols.\n"
        "CRITICAL stairs rule: pick 'stairs' ONLY when you clearly see a staircase flight"
        " (multiple parallel step/tread lines plus a staircase direction pattern). "
        "If it could be either window or stairs, choose window.\n"
        "CRITICAL garage rule: if the candidate looks window-like but is on a garage/carport wall "
        "(room label Garage/Carport or nearby car drawing in the plan context), classify as door "
        "because that is a garage door.\n"
        "Return ONLY strict JSON with keys: class, confidence, reason.\n"
        'Example: {"class":"door","confidence":0.82,"reason":"arc swing door symbol"}\n'
        "No markdown, no extra text."
    )
    img_bytes = crop_path.read_bytes()
    context_bytes = context_plan_path.read_bytes()
    response = gemini_model.generate_content(
        [
            prompt,
            {"mime_type": "image/jpeg", "data": img_bytes},
            {"mime_type": "image/jpeg", "data": context_bytes},
        ]
    )
    text = (response.text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    cls = _normalize_class(str(parsed.get("class", "")))
    conf = float(parsed.get("confidence", 0.0))
    conf = max(0.0, min(1.0, conf))
    return {
        "class": cls,
        "confidence": conf,
        "reason": str(parsed.get("reason", "")),
        "raw_response": parsed,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--plans", default="", help="Comma-separated plan ids, default all plans in run/scale")
    args = ap.parse_args()

    try:
        gemini_model = _gemini_model()
    except Exception as e:
        print(str(e))
        return 2

    scale_dir = OUTPUT_ROOT / args.run_id / "scale"
    if not scale_dir.exists():
        print(f"Missing run scale dir: {scale_dir}")
        return 2

    plan_ids = [p.strip() for p in args.plans.split(",") if p.strip()] if args.plans else sorted(
        [p.name for p in scale_dir.iterdir() if p.is_dir() and p.name.startswith("plan_")]
    )

    for plan_id in plan_ids:
        raster_dir = scale_dir / plan_id / "cubicasa_steps" / "raster"
        review_json = raster_dir / "detections_review_data.json"
        plan_img_path = raster_dir / "input_resized.jpg"
        if not review_json.exists() or not plan_img_path.exists():
            print(f"[{plan_id}] skip (missing review_json or input_resized)")
            continue

        data = json.loads(review_json.read_text(encoding="utf-8"))
        doors = data.get("doors") or []
        rooms = data.get("rooms") or []
        garage_polys: list[list[list[int]]] = []
        for r in rooms:
            if not isinstance(r, dict):
                continue
            if not _is_garage_room_label(str(r.get("roomType", "")), str(r.get("roomName", ""))):
                continue
            pts = r.get("points") or []
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            clean_pts: list[list[int]] = []
            for p in pts:
                if isinstance(p, list) and len(p) >= 2:
                    clean_pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
            if len(clean_pts) >= 3:
                garage_polys.append(clean_pts)
        img = cv2.imread(str(plan_img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{plan_id}] skip (cannot read {plan_img_path})")
            continue
        img_h, img_w = img.shape[:2]

        # Some runs persist review bboxes in translated raster coords. Auto-detect the right frame.
        tx, ty = 0, 0
        tr_path = raster_dir / "brute_steps" / "translation_only_config.json"
        if tr_path.exists():
            try:
                tr = json.loads(tr_path.read_text(encoding="utf-8"))
                pos = tr.get("position") or [0, 0]
                tx = int(round(float(pos[0])))
                ty = int(round(float(pos[1])))
            except Exception:
                tx, ty = 0, 0
        raw_bad = _bbox_out_of_bounds_count(doors, img_w, img_h, tx=0, ty=0)
        translated_bad = _bbox_out_of_bounds_count(doors, img_w, img_h, tx=tx, ty=ty)
        use_translation = translated_bad < raw_bad
        # Tie-breaker: if both mappings are equally valid and translation exists,
        # prefer translated coords (some runs store review bboxes in translated frame).
        if translated_bad == raw_bad and (tx != 0 or ty != 0):
            use_translation = True
        mode = "translated" if use_translation else "raw"
        print(
            f"[{plan_id}] bbox mapping mode={mode} raw_bad={raw_bad} translated_bad={translated_bad} tx={tx} ty={ty}"
        )

        out_dir = OUTPUT_ROOT / args.run_id / "roboflow_bbox_review" / plan_id
        crops_dir = out_dir / "crops"
        context_dir = out_dir / "context"
        out_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        context_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        overlay = _to_gray50(img)

        for i, d in enumerate(doors):
            bb = d.get("bbox") if isinstance(d, dict) else None
            if not (isinstance(bb, list) and len(bb) == 4):
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
            if use_translation:
                x1 -= tx
                x2 -= tx
                y1 -= ty
                y2 -= ty
            # Expand crop by 10% on each side to add context.
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            pad_x = int(round(w * 0.10))
            pad_y = int(round(h * 0.10))
            x1 -= pad_x
            x2 += pad_x
            y1 -= pad_y
            y2 += pad_y
            x1 = max(0, min(img_w - 1, x1))
            x2 = max(0, min(img_w - 1, x2))
            y1 = max(0, min(img_h - 1, y1))
            y2 = max(0, min(img_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            crop_path = crops_dir / f"bbox_{i:03d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            context_img = img.copy()
            cv2.rectangle(context_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            context_path = context_dir / f"bbox_{i:03d}_context.jpg"
            cv2.imwrite(str(context_path), context_img)

            try:
                gm = _classify_with_gemini(gemini_model, crop_path, context_path)
                cls = gm["class"]
                conf = float(gm["confidence"])
                reason = gm["reason"]
                raw_resp = gm["raw_response"]
                cls, refine_note = _refine_class(cls, conf, reason, x1, y1, x2, y2)
                if refine_note:
                    reason = f"{reason} | {refine_note}".strip()
                if cls == "window" and garage_polys and _is_near_garage_wall(x1, y1, x2, y2, garage_polys):
                    cls = "garage_door"
                    reason = f"{reason} | window->garage_door (garage/carport context)".strip()
            except Exception as e:
                cls = "door"
                conf = 0.0
                reason = f"gemini_error: {e}"
                raw_resp = {"error": str(e)}

            if cls == "garage_door":
                color = (255, 0, 255)
            elif cls == "door":
                color = (0, 255, 0)
            elif cls == "window":
                color = (255, 0, 0)
            else:
                color = (0, 140, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"{i}:{cls}:{conf:.2f}",
                (x1, max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            results.append(
                {
                    "idx": i,
                    "bbox": [x1, y1, x2, y2],
                    "existing_type": d.get("type"),
                    "gemini_best_class": cls,
                    "gemini_best_confidence": conf,
                    "gemini_reason": reason,
                    "gemini_response": raw_resp,
                    "crop_path": str(crop_path),
                    "context_plan_path": str(context_path),
                }
            )
            print(f"[{plan_id}] bbox {i}: existing={d.get('type')} gemini={cls} ({conf:.2f})")

        (out_dir / "roboflow_on_review_bboxes.json").write_text(
            json.dumps({"run_id": args.run_id, "plan_id": plan_id, "results": results}, indent=2),
            encoding="utf-8",
        )
        cv2.imwrite(str(out_dir / "roboflow_on_review_bboxes_overlay.jpg"), overlay)
        print(f"[{plan_id}] saved {len(results)} bbox results -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

