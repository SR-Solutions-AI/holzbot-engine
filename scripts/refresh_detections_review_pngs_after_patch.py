#!/usr/bin/env python3
"""
Regenerates detections_review_{base,rooms,doors}.png after the API saved detections_edited.json.

Invoked from holzbot-api ComputeService.patchDetectionsReviewData so admin / local disk
see the same room & door overlays as the customer editor (not stale AI-only PNGs).

Usage: python scripts/refresh_detections_review_pngs_after_patch.py <raster_dir> [<raster_dir> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path


def _refresh_one(raster_dir: Path) -> None:
    from cubicasa_detector.raster_api import apply_detections_edited, save_detections_review_image

    edited = raster_dir / "detections_edited.json"
    if edited.exists():
        apply_detections_edited(raster_dir)
        save_detections_review_image(raster_dir, prefer_edited_room_polygons=True)
    else:
        save_detections_review_image(raster_dir, prefer_edited_room_polygons=False)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: refresh_detections_review_pngs_after_patch.py <raster_dir> [...]", file=sys.stderr)
        return 2
    for arg in sys.argv[1:]:
        rd = Path(arg)
        if not rd.is_dir():
            print(f"skip (not a dir): {rd}", file=sys.stderr)
            continue
        try:
            _refresh_one(rd)
            print(f"ok {rd}", flush=True)
        except Exception as e:
            print(f"error {rd}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
