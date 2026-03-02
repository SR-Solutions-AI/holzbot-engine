# cubicasa_detector/wall_gap_detector.py
"""
Detectează peretele lipsă dintr-o cameră (ex: garaj) folosind trei metode:

1. find_missing_wall_raycast()      — Ray cast + gap detection (4 pereți, gap-uri pe laturi)
2. find_missing_wall_raycast_3walls() — Ray cast + capete pereți adiacenți (3 pereți, o latură complet lipsă)
3. find_missing_wall()              — BFS flood-fill (fallback)

Ray cast (Algoritm 1): 4 raze din start → bbox → scanare laturi → gap-uri → segmente.
Ray cast 3 pereți (Algoritm 2): când o rază nu găsește perete → direcție lipsă → endpoint pe pereții
adiacenți → un segment care închide camera.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Utilitare comune
# ─────────────────────────────────────────────────────────────────────────────

def get_red_dot_mask(img: np.ndarray) -> np.ndarray:
    """Mască HSV pentru bulina roșie (public pentru plan_closed)."""
    return _mask_red_dot(img)


def _mask_red_dot(img: np.ndarray) -> np.ndarray:
    """Mască HSV pentru bulina roșie."""
    if len(img.shape) == 2:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        | cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    )
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# METODA NOUĂ: Ray Cast
# ─────────────────────────────────────────────────────────────────────────────

def find_missing_wall_raycast(
    img: np.ndarray,
    start_x: int | None = None,
    start_y: int | None = None,
    wall_threshold: int = 20,
    wall_mask: np.ndarray | None = None,
    min_gap_px: int = 3,
) -> list[dict] | None:
    """
    Detectează peretele (sau pereții) lipsă dintr-o cameră folosind ray cast.

    Algoritmul:
      1. Din start point, trage 4 raze (sus/jos/stânga/dreapta) →
         găsește primul pixel de perete în fiecare direcție → bbox-ul camerei.
      2. Scanează fiecare latură a bbox-ului pixel cu pixel →
         identifică gap-urile unde lipsesc pixeli de perete.
      3. Returnează lista de segmente corespunzătoare gap-urilor găsite.

    Args:
        img: Imagine BGR sau grayscale (pereți albi, fundal negru; opțional bulina roșie).
        start_x, start_y: Punct de start (interior cameră). Dacă None, se detectează bulina roșie.
        wall_threshold: Prag gri pentru perete (folosit dacă wall_mask nu e dat).
        wall_mask: Mască binară (h,w) – pixel > 0 = perete. Prioritar față de wall_threshold.
        min_gap_px: Gap-urile mai mici decât această valoare sunt ignorate (zgomot).

    Returns:
        Listă de dict-uri cu segmentele lipsă, fiecare de forma:
          {"type": "vertical",   "x": int, "y1": int, "y2": int}  sau
          {"type": "horizontal", "y": int, "x1": int, "x2": int}
        Returnează None dacă nu se poate determina start point-ul sau bbox-ul.
    """
    if len(img.shape) == 2:
        gray = img.copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bgr = img.copy()

    h, w = img.shape[:2]
    mask_red = _mask_red_dot(img_bgr)

    if wall_mask is not None and wall_mask.shape[:2] == (h, w):
        wall = (np.asarray(wall_mask) > 0).astype(np.uint8)
    else:
        wall = (gray > wall_threshold).astype(np.uint8)
    wall[mask_red > 0] = 0  # Bulina roșie nu e perete

    # Detectare start point din bulina roșie dacă nu e dat explicit
    if start_x is None or start_y is None:
        coords = np.where(mask_red > 0)
        if coords[0].size == 0:
            return None
        start_x = int(np.mean(coords[1]))
        start_y = int(np.mean(coords[0]))

    # ── Pas 1: Ray cast în cele 4 direcții ──────────────────────────────────
    def ray(sx: int, sy: int, dx: int, dy: int) -> tuple[int, int] | None:
        """Găsește primul pixel de perete din (sx,sy) în direcția (dx,dy)."""
        x, y = sx + dx, sy + dy
        while 0 <= x < w and 0 <= y < h:
            if wall[y, x]:
                return x, y
            x += dx
            y += dy
        return None

    hit_right = ray(start_x, start_y,  1,  0)
    hit_left  = ray(start_x, start_y, -1,  0)
    hit_down  = ray(start_x, start_y,  0,  1)
    hit_up    = ray(start_x, start_y,  0, -1)

    # Dacă vreo rază nu găsește perete → nu putem determina bbox
    if not all([hit_right, hit_left, hit_down, hit_up]):
        return None

    bbox_left   = hit_left[0]
    bbox_right  = hit_right[0]
    bbox_top    = hit_up[1]
    bbox_bottom = hit_down[1]

    # Sanity check: bbox valid
    if bbox_right <= bbox_left or bbox_bottom <= bbox_top:
        return None

    # ── Pas 2: Scanează fiecare latură și găsește gap-urile ─────────────────
    def find_gaps(axis: str, coord: int, mn: int, mx: int) -> list[tuple[int, int]]:
        """
        Parcurge latura de la mn la mx și returnează lista de gap-uri (start, end).
        axis='h' → linie orizontală: wall[coord, i]
        axis='v' → coloană verticală: wall[i, coord]
        """
        gaps = []
        in_gap = False
        gap_start = 0
        for i in range(mn, mx + 1):
            pixel = wall[coord, i] if axis == "h" else wall[i, coord]
            if not pixel and not in_gap:
                in_gap = True
                gap_start = i
            elif pixel and in_gap:
                in_gap = False
                if i - gap_start >= min_gap_px:
                    gaps.append((gap_start, i - 1))
        if in_gap and mx - gap_start + 1 >= min_gap_px:
            gaps.append((gap_start, mx))
        return gaps

    gaps_top    = find_gaps("h", bbox_top,    bbox_left, bbox_right)
    gaps_bottom = find_gaps("h", bbox_bottom, bbox_left, bbox_right)
    gaps_left   = find_gaps("v", bbox_left,   bbox_top,  bbox_bottom)
    gaps_right  = find_gaps("v", bbox_right,  bbox_top,  bbox_bottom)

    # ── Pas 3: Construiește segmentele ───────────────────────────────────────
    segments: list[dict] = []

    for (x1, x2) in gaps_top:
        segments.append({"type": "horizontal", "y": bbox_top,    "x1": x1, "x2": x2})
    for (x1, x2) in gaps_bottom:
        segments.append({"type": "horizontal", "y": bbox_bottom, "x1": x1, "x2": x2})
    for (y1, y2) in gaps_left:
        segments.append({"type": "vertical",   "x": bbox_left,   "y1": y1, "y2": y2})
    for (y1, y2) in gaps_right:
        segments.append({"type": "vertical",   "x": bbox_right,  "y1": y1, "y2": y2})

    return segments if segments else None


def draw_plan_closed_raycast(
    img: np.ndarray,
    segments: list[dict],
    mask_red: np.ndarray | None = None,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    Imagine finală: plan fără bulina roșie, segmentele lipsă desenate.

    Args:
        img: Imaginea originală BGR.
        segments: Lista de segmente returnată de find_missing_wall_raycast().
        mask_red: Mască bulina roșie (dacă None, se calculează automat).
        color: Culoarea segmentelor (implicit alb).
        thickness: Grosimea liniei.

    Returns:
        Copie a imaginii cu segmentele desenate.
    """
    out = img.copy()
    if mask_red is None:
        mask_red = _mask_red_dot(img)
    out[mask_red > 0] = (0, 0, 0)
    h, w = out.shape[:2]

    for seg in segments:
        if seg["type"] == "vertical":
            y1 = max(0, seg["y1"])
            y2 = min(h - 1, seg["y2"])
            cv2.line(out, (seg["x"], y1), (seg["x"], y2), color, thickness)
        else:
            x1 = max(0, seg["x1"])
            x2 = min(w - 1, seg["x2"])
            cv2.line(out, (x1, seg["y"]), (x2, seg["y"]), color, thickness)

    return out


def save_raycast_debug(
    img: np.ndarray,
    start_x: int,
    start_y: int,
    segments: list[dict],
    path: Path | str,
    wall_threshold: int = 20,
    wall_mask: np.ndarray | None = None,
) -> None:
    """
    Salvează o imagine de debug cu:
      - bbox-ul camerei (galben)
      - segmentele lipsă (roșu)
      - start point-ul (verde)
    """
    if len(img.shape) == 2:
        gray = img.copy()
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = img.copy()

    h, w = img.shape[:2]
    mask_red = _mask_red_dot(out)

    if wall_mask is not None and wall_mask.shape[:2] == (h, w):
        wall = (np.asarray(wall_mask) > 0).astype(np.uint8)
    else:
        wall = (gray > wall_threshold).astype(np.uint8)
    wall[mask_red > 0] = 0

    def ray(sx, sy, dx, dy):
        x, y = sx + dx, sy + dy
        while 0 <= x < w and 0 <= y < h:
            if wall[y, x]:
                return x, y
            x += dx
            y += dy
        return None

    hits = [
        ray(start_x, start_y,  1,  0),
        ray(start_x, start_y, -1,  0),
        ray(start_x, start_y,  0,  1),
        ray(start_x, start_y,  0, -1),
    ]

    if all(hits):
        bbox_right, bbox_left = hits[0][0], hits[1][0]
        bbox_bottom, bbox_top = hits[2][1], hits[3][1]
        cv2.rectangle(out, (bbox_left, bbox_top), (bbox_right, bbox_bottom), (0, 255, 255), 2)

    for seg in segments:
        if seg["type"] == "vertical":
            cv2.line(out, (seg["x"], seg["y1"]), (seg["x"], seg["y2"]), (0, 0, 255), 3)
        else:
            cv2.line(out, (seg["x1"], seg["y"]), (seg["x2"], seg["y"]), (0, 0, 255), 3)

    cv2.circle(out, (start_x, start_y), 8, (0, 255, 0), -1)
    cv2.imwrite(str(path), out)


# ─────────────────────────────────────────────────────────────────────────────
# RAY CAST — Algoritm 2: cameră cu 3 pereți (o latură complet lipsă)
# ─────────────────────────────────────────────────────────────────────────────

def find_missing_wall_raycast_3walls(
    img: np.ndarray,
    start_x: int | None = None,
    start_y: int | None = None,
    wall_threshold: int = 20,
    wall_mask: np.ndarray | None = None,
    endpoint_gap_tolerance: int = 5,
) -> list[dict] | None:
    """
    Fallback când ray cast-ul cu 4 raze nu găsește perete într-o direcție (cameră cu 3 pereți).
    Identifică direcția lipsă și construiește un segment care conectează capetele pereților
    adiacenți în acea direcție.

    Se apelează doar dacă find_missing_wall_raycast() returnează None (cel puțin o rază fără hit).

    Returns:
        Listă cu un singur segment (vertical sau orizontal) sau None dacă nu se poate determina.
    """
    if len(img.shape) == 2:
        gray = img.copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bgr = img.copy()

    h, w = img.shape[:2]
    mask_red = _mask_red_dot(img_bgr)
    if wall_mask is not None and wall_mask.shape[:2] == (h, w):
        wall = (np.asarray(wall_mask) > 0).astype(np.uint8)
    else:
        wall = (gray > wall_threshold).astype(np.uint8)
    wall[mask_red > 0] = 0

    if start_x is None or start_y is None:
        coords = np.where(mask_red > 0)
        if coords[0].size == 0:
            return None
        start_x = int(np.mean(coords[1]))
        start_y = int(np.mean(coords[0]))

    def ray(sx: int, sy: int, dx: int, dy: int) -> tuple[int, int] | None:
        x, y = sx + dx, sy + dy
        while 0 <= x < w and 0 <= y < h:
            if wall[y, x]:
                return (x, y)
            x += dx
            y += dy
        return None

    hits = {
        "right": ray(start_x, start_y,  1,  0),
        "left":  ray(start_x, start_y, -1,  0),
        "down":  ray(start_x, start_y,  0,  1),
        "up":    ray(start_x, start_y,  0, -1),
    }

    # Algoritmul 2 se aplică doar când exact o direcție nu a găsit perete
    none_count = sum(1 for v in hits.values() if v is None)
    if none_count == 0:
        return None  # toate 4 au găsit → Algoritmul 1 (gap-uri) se ocupă
    if none_count > 1:
        return None  # prea multe laturi lipsă, ambiguu

    missing_dir = next(k for k, v in hits.items() if v is None)

    max_dist = 400

    def _find_wall_end_from_center(col_vals: np.ndarray, start: int, direction: str) -> int | None:
        """
        Pornește de la start și merge în direcția dată.
        Returnează ultimul pixel de perete înainte de prima pauză (gaură).
        direction: 'down', 'up' -> col_vals = coloană (h,); start = y
                   'right', 'left' -> col_vals = linie (w,); start = x
        """
        n = len(col_vals)
        last_wall: int | None = None
        if direction == "down":
            for i in range(start, min(start + max_dist, n)):
                if col_vals[i] > 0:
                    last_wall = i
                elif last_wall is not None:
                    return last_wall
            return last_wall
        if direction == "up":
            for i in range(start, max(start - max_dist, -1), -1):
                if col_vals[i] > 0:
                    last_wall = i
                elif last_wall is not None:
                    return last_wall
            return last_wall
        if direction == "right":
            for i in range(start, min(start + max_dist, n)):
                if col_vals[i] > 0:
                    last_wall = i
                elif last_wall is not None:
                    return last_wall
            return last_wall
        if direction == "left":
            for i in range(start, max(start - max_dist, -1), -1):
                if col_vals[i] > 0:
                    last_wall = i
                elif last_wall is not None:
                    return last_wall
            return last_wall
        return None

    segments_out: list[dict] = []

    if missing_dir == "down":
        x_left, x_right = hits["left"][0], hits["right"][0]
        l_end = _find_wall_end_from_center(wall[:, x_left], start_y, "down")
        r_end = _find_wall_end_from_center(wall[:, x_right], start_y, "down")
        if l_end is None or r_end is None:
            return None
        draw_y = min(l_end, r_end)
        segments_out.append({"type": "horizontal", "y": draw_y, "x1": x_left, "x2": x_right})
        if draw_y < h and wall[draw_y, x_left] == 0:
            segments_out.append({"type": "vertical", "x": x_left, "y1": draw_y, "y2": l_end})
        if draw_y < h and wall[draw_y, x_right] == 0:
            segments_out.append({"type": "vertical", "x": x_right, "y1": draw_y, "y2": r_end})

    elif missing_dir == "up":
        x_left, x_right = hits["left"][0], hits["right"][0]
        l_end = _find_wall_end_from_center(wall[:, x_left], start_y, "up")
        r_end = _find_wall_end_from_center(wall[:, x_right], start_y, "up")
        if l_end is None or r_end is None:
            return None
        draw_y = max(l_end, r_end)
        segments_out.append({"type": "horizontal", "y": draw_y, "x1": x_left, "x2": x_right})
        if draw_y >= 0 and wall[draw_y, x_left] == 0:
            segments_out.append({"type": "vertical", "x": x_left, "y1": l_end, "y2": draw_y})
        if draw_y >= 0 and wall[draw_y, x_right] == 0:
            segments_out.append({"type": "vertical", "x": x_right, "y1": r_end, "y2": draw_y})

    elif missing_dir == "right":
        y_top, y_bot = hits["up"][1], hits["down"][1]
        t_end = _find_wall_end_from_center(wall[y_top, :], start_x, "right")
        b_end = _find_wall_end_from_center(wall[y_bot, :], start_x, "right")
        if t_end is None or b_end is None:
            return None
        draw_x = min(t_end, b_end)
        segments_out.append({"type": "vertical", "x": draw_x, "y1": y_top, "y2": y_bot})
        if draw_x < w and wall[y_top, draw_x] == 0:
            segments_out.append({"type": "horizontal", "y": y_top, "x1": draw_x, "x2": t_end})
        if draw_x < w and wall[y_bot, draw_x] == 0:
            segments_out.append({"type": "horizontal", "y": y_bot, "x1": draw_x, "x2": b_end})

    elif missing_dir == "left":
        y_top, y_bot = hits["up"][1], hits["down"][1]
        t_end = _find_wall_end_from_center(wall[y_top, :], start_x, "left")
        b_end = _find_wall_end_from_center(wall[y_bot, :], start_x, "left")
        if t_end is None or b_end is None:
            return None
        draw_x = max(t_end, b_end)
        segments_out.append({"type": "vertical", "x": draw_x, "y1": y_top, "y2": y_bot})
        if draw_x >= 0 and wall[y_top, draw_x] == 0:
            segments_out.append({"type": "horizontal", "y": y_top, "x1": t_end, "x2": draw_x})
        if draw_x >= 0 and wall[y_bot, draw_x] == 0:
            segments_out.append({"type": "horizontal", "y": y_bot, "x1": b_end, "x2": draw_x})

    return segments_out if segments_out else None


# ─────────────────────────────────────────────────────────────────────────────
# METODA ORIGINALĂ: BFS Flood Fill
# ─────────────────────────────────────────────────────────────────────────────

def find_missing_wall(
    img: np.ndarray,
    start_x: int | None = None,
    start_y: int | None = None,
    stable_min_px: int = 1000,
    steps_dir: Path | str | None = None,
    step_interval: int = 2000,
    wall_threshold: int = 20,
    max_iterations: int = 500000,
    wall_mask: np.ndarray | None = None,
    min_area_before_stable: int | None = None,
):
    """
    Detectează peretele lipsă dintr-o cameră cu BFS din punctul marker.
    Bariera: orice pixel de perete blochează flood-ul; prin găuri reale flood-ul poate ieși.

    Args:
        img: Imagine BGR (plan cu pereți albi, fundal negru; opțional bulina roșie).
        start_x, start_y: Punct de start. Dacă None, se detectează bulina roșie.
        stable_min_px: Iterații consecutivate cu aceeași dimensiune pentru Faza 2.
        steps_dir: Dacă setat, salvează frame-uri în acest folder.
        step_interval: La câte iterații salvează un frame (dacă steps_dir).
        wall_threshold: Valoare minimă gri considerată perete (folosit doar dacă wall_mask nu e dat).
        max_iterations: Limită BFS.
        wall_mask: Mască binară (h,w) – pixel > 0 = perete. Dacă dată, se folosește ca barieră exactă
            (flood-ul nu trece niciodată prin pereți; poate ieși doar prin zone cu 0). Ignoră wall_threshold.
        min_area_before_stable: Dacă setat, „stabil” se acceptă doar după ce zona umplută are
            cel puțin atâția pixeli (flood fill continuă până devine considerabil mai mare).

    Returns:
        dict cu type, x, y1, y2 (vertical) sau type, y, x1, x2 (orizontal), sau None.
    """
    if len(img.shape) == 2:
        gray = img.copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bgr = img.copy()

    h, w = img.shape[:2]
    mask_red = _mask_red_dot(img_bgr)
    if wall_mask is not None and wall_mask.shape[:2] == (h, w):
        wall = (np.asarray(wall_mask) > 0).astype(np.uint8)
    else:
        wall = (gray > wall_threshold).astype(np.uint8)
    wall[mask_red > 0] = 0

    if start_x is None or start_y is None:
        coords = np.where(mask_red > 0)
        if coords[0].size == 0:
            return None
        start_x = int(np.mean(coords[1]))
        start_y = int(np.mean(coords[0]))

    if steps_dir is not None:
        steps_path = Path(steps_dir)
        steps_path.mkdir(parents=True, exist_ok=True)
        cur0 = {"right": start_x, "left": start_x, "top": start_y, "bottom": start_y}
        vis0 = np.zeros((h, w), dtype=np.uint8)
        vis0[start_y, start_x] = 1
        _save_step(steps_path, 0, img_bgr, vis0, cur0, mask_red, None)

    for stable_axis in ("height", "width"):
        result = _bfs_detect(
            wall=wall,
            h=h,
            w=w,
            start_x=start_x,
            start_y=start_y,
            stable_axis=stable_axis,
            stable_min_px=stable_min_px,
            max_iterations=max_iterations,
            steps_dir=Path(steps_dir) if steps_dir else None,
            step_interval=step_interval,
            img_bgr=img_bgr,
            mask_red=mask_red,
            min_area_before_stable=min_area_before_stable,
        )
        if result is not None:
            return result
    return None


def _count_touches_on_segment(
    visited: np.ndarray,
    segment: dict,
    h: int,
    w: int,
) -> int:
    """
    Numără câte pixeli de pe linia segmentului sunt atinși de flood (visited sau vecin).
    Segment: vertical (x, y1..y2) sau orizontal (x1..x2, y).
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(visited, kernel)
    touch_count = 0
    if segment["type"] == "vertical":
        x = segment["x"]
        y1, y2 = segment["y1"], segment["y2"]
        if y2 < y1:
            y1, y2 = y2, y1
        for yy in range(y1, y2 + 1):
            if 0 <= yy < h and 0 <= x < w and dilated[yy, x]:
                touch_count += 1
    else:
        y = segment["y"]
        x1, x2 = segment["x1"], segment["x2"]
        if x2 < x1:
            x1, x2 = x2, x1
        for xx in range(x1, x2 + 1):
            if 0 <= y < h and 0 <= xx < w and dilated[y, xx]:
                touch_count += 1
    return touch_count


def _segment_length_px(result: dict) -> int:
    """Lungimea segmentului în pixeli (pentru a cere touch pe o suprafață suficientă)."""
    if result["type"] == "vertical":
        y1, y2 = result["y1"], result["y2"]
        return abs(y2 - y1) + 1
    x1, x2 = result["x1"], result["x2"]
    return abs(x2 - x1) + 1


def _bfs_detect(
    wall: np.ndarray,
    h: int,
    w: int,
    start_x: int,
    start_y: int,
    stable_axis: str,
    stable_min_px: int,
    max_iterations: int,
    steps_dir: Path | None,
    step_interval: int,
    img_bgr: np.ndarray,
    mask_red: np.ndarray,
    min_area_before_stable: int | None = None,
) -> dict | None:
    visited = np.zeros((h, w), dtype=np.uint8)
    visited[start_y, start_x] = 1
    queue = deque([(start_x, start_y)])
    cur = {"right": start_x, "left": start_x, "top": start_y, "bottom": start_y}

    stable_confirmed = False
    stable_val = None
    stable_count = 0
    iteration = 0

    touches_image_edge = False

    while queue and iteration < max_iterations:
        x, y = queue.popleft()
        iteration += 1
        if x == 0 or x == w - 1 or y == 0 or y == h - 1:
            touches_image_edge = True
            # Opresc instant: flood-ul a atins marginea înainte de al 4-lea perete → anulăm
            return None

        old = dict(cur)
        if x > cur["right"]:
            cur["right"] = x
        if x < cur["left"]:
            cur["left"] = x
        if y > cur["bottom"]:
            cur["bottom"] = y
        if y < cur["top"]:
            cur["top"] = y

        if stable_axis == "height":
            cur_dim = cur["bottom"] - cur["top"]
        else:
            cur_dim = cur["right"] - cur["left"]

        if not stable_confirmed:
            if stable_val is None or cur_dim != stable_val:
                stable_val = cur_dim
                stable_count = 1
            else:
                stable_count += 1
                area_ok = min_area_before_stable is None or np.sum(visited) >= min_area_before_stable
                if stable_count >= stable_min_px and area_ok:
                    stable_confirmed = True
        else:
            if stable_axis == "height":
                escaped = cur["bottom"] > old["bottom"] or cur["top"] < old["top"]
                if escaped:
                    if touches_image_edge:
                        return None  # Zona atinge marginea planului → nu poate fi închisă
                    result = {
                        "type": "vertical",
                        "x": x,
                        "y1": old["top"],
                        "y2": old["bottom"],
                    }
                    result = _clip_segment_at_perpendicular_wall(result, wall, w, h)
                    if result is not None:
                        touch_count = _count_touches_on_segment(visited, result, h, w)
                        seg_len = _segment_length_px(result)
                        # Touch trebuie pe o suprafață suficientă (min 15% din lungimea segmentului), nu doar vârful
                        min_touch = max(3, int(0.15 * seg_len))
                        if touch_count >= min_touch:
                            if steps_dir is not None:
                                _save_step(
                                    steps_dir, iteration, img_bgr, visited, cur, mask_red, result
                                )
                                _save_step(
                                    steps_dir, -1, img_bgr, visited, cur, mask_red, result
                                )  # step_final.png
                            return result
                    # touch_count prea mic sau result None: continuăm BFS
            else:
                escaped = cur["right"] > old["right"] or cur["left"] < old["left"]
                if escaped:
                    if touches_image_edge:
                        return None  # Zona atinge marginea planului → nu poate fi închisă
                    result = {
                        "type": "horizontal",
                        "y": y,
                        "x1": old["left"],
                        "x2": old["right"],
                    }
                    result = _clip_segment_at_perpendicular_wall(result, wall, w, h)
                    if result is not None:
                        touch_count = _count_touches_on_segment(visited, result, h, w)
                        seg_len = _segment_length_px(result)
                        min_touch = max(3, int(0.15 * seg_len))
                        if touch_count >= min_touch:
                            if steps_dir is not None:
                                _save_step(
                                    steps_dir, iteration, img_bgr, visited, cur, mask_red, result
                                )
                                _save_step(
                                    steps_dir, -1, img_bgr, visited, cur, mask_red, result
                                )  # step_final.png
                            return result
                    # touch_count prea mic sau result None: continuăm BFS

        if steps_dir is not None and iteration % step_interval == 0:
            _save_step(steps_dir, iteration, img_bgr, visited, cur, mask_red, None)

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < w
                and 0 <= ny < h
                and not visited[ny, nx]
                and not wall[ny, nx]
            ):
                visited[ny, nx] = 1
                queue.append((nx, ny))

    return None


def _clip_segment_at_perpendicular_wall(
    result: dict,
    wall: np.ndarray,
    w: int,
    h: int,
) -> dict | None:
    """
    Taie segmentul astfel încât să nu treacă prin pereți perpendiculari.
    - Segment vertical (x, y1..y2): taie la rândurile unde există perete orizontal (wall la (x±1, y)).
    - Segment orizontal (x1..x2, y): taie la coloanele unde există perete vertical (wall la (x, y±1)).
    """
    if result["type"] == "vertical":
        x = result["x"]
        y1, y2 = result["y1"], result["y2"]
        if y2 < y1:
            y1, y2 = y2, y1
        # Perete perpendicular = orizontal care traversează coloana x
        y1_new, y2_new = y1, y2
        for yy in range(y1, y2 + 1):
            if 0 <= yy < h and ((x - 1 >= 0 and wall[yy, x - 1]) or (x + 1 < w and wall[yy, x + 1])):
                y2_new = yy - 1  # Taie de la primul perete în jos
                break
        for yy in range(y2, y1 - 1, -1):
            if 0 <= yy < h and ((x - 1 >= 0 and wall[yy, x - 1]) or (x + 1 < w and wall[yy, x + 1])):
                y1_new = yy + 1  # Taie de la primul perete în sus
                break
        if y1_new > y2_new:
            return None
        return {"type": "vertical", "x": x, "y1": y1_new, "y2": y2_new}
    else:
        y = result["y"]
        x1, x2 = result["x1"], result["x2"]
        if x2 < x1:
            x1, x2 = x2, x1
        # Perete perpendicular = vertical care traversează rândul y
        x1_new, x2_new = x1, x2
        for xx in range(x1, x2 + 1):
            if 0 <= xx < w and ((y - 1 >= 0 and wall[y - 1, xx]) or (y + 1 < h and wall[y + 1, xx])):
                x2_new = xx - 1
                break
        for xx in range(x2, x1 - 1, -1):
            if 0 <= xx < w and ((y - 1 >= 0 and wall[y - 1, xx]) or (y + 1 < h and wall[y + 1, xx])):
                x1_new = xx + 1
                break
        if x1_new > x2_new:
            return None
        return {"type": "horizontal", "y": y, "x1": x1_new, "x2": x2_new}


def _save_step(
    steps_dir: Path,
    iteration: int,
    img_bgr: np.ndarray,
    visited: np.ndarray,
    bbox: dict,
    mask_red: np.ndarray,
    result: dict | None,
) -> None:
    """Salvează un frame: plan fără bulina roșie + overlay flood fill + bbox + segment albastru (dacă result)."""
    out = img_bgr.copy()
    out[mask_red > 0] = (0, 0, 0)

    overlay = out.copy()
    visited_bgr = np.stack([visited * 0, visited * 180, visited * 100], axis=-1)
    overlay[visited > 0] = overlay[visited > 0] * 0.5 + visited_bgr[visited > 0] * 0.5
    out = overlay.astype(np.uint8)

    cv2.rectangle(
        out,
        (bbox["left"], bbox["top"]),
        (bbox["right"], bbox["bottom"]),
        (0, 255, 255),
        1,
    )

    if result is not None:
        if result["type"] == "vertical":
            cv2.line(
                out,
                (result["x"], result["y1"]),
                (result["x"], result["y2"]),
                (255, 0, 0),
                2,
            )
        else:
            cv2.line(
                out,
                (result["x1"], result["y"]),
                (result["x2"], result["y"]),
                (255, 0, 0),
                2,
            )

    name = "step_final.png" if iteration < 0 else f"step_{iteration:06d}.png"
    path = steps_dir / name
    cv2.imwrite(str(path), out)


def draw_plan_closed(
    img: np.ndarray,
    result: dict,
    mask_red: np.ndarray | None = None,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
    extension_px: int = 0,
) -> np.ndarray:
    """
    Imagine finală: plan fără bulina roșie, segment perete lipsă desenat alb 1px.
    Segmentul este desenat exact ca în step_final (segmentul albastru): fără prelungire,
    astfel încât să nu treacă de pereții perpendiculari (același result clipat).
    extension_px=0 implicit; dacă se pasează >0, se prelungește (poate traversa pereți).
    """
    out = img.copy()
    if mask_red is not None:
        out[mask_red > 0] = (0, 0, 0)
    h, w = out.shape[:2]
    ext = max(0, extension_px)
    if result["type"] == "vertical":
        y1 = max(0, result["y1"] - ext)
        y2 = min(h - 1, result["y2"] + ext)
        cv2.line(out, (result["x"], y1), (result["x"], y2), color, thickness)
    else:
        x1 = max(0, result["x1"] - ext)
        x2 = min(w - 1, result["x2"] + ext)
        cv2.line(out, (x1, result["y"]), (x2, result["y"]), color, thickness)
    return out


def check_garage_flood_touches_edge(
    img: np.ndarray,
    start_x: int,
    start_y: int,
    wall_threshold: int = 20,
    wall_mask: np.ndarray | None = None,
) -> tuple[bool, np.ndarray]:
    """
    Flood fill complet din (start_x, start_y). Returnează (touches_edge, visited).
    Dacă flood-ul atinge cel puțin o margine (x=0, x=w-1, y=0, y=h-1) → touches_edge=True.
    Bariera: orice pixel de perete blochează; dacă e dat wall_mask, se folosește ca barieră exactă.
    """
    if len(img.shape) == 2:
        gray = img.copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    mask_red = _mask_red_dot(img) if len(img.shape) == 3 else np.zeros((h, w), dtype=np.uint8)
    if wall_mask is not None and wall_mask.shape[:2] == (h, w):
        wall = (np.asarray(wall_mask) > 0).astype(np.uint8)
    else:
        wall = (gray > wall_threshold).astype(np.uint8)
    wall[mask_red > 0] = 0

    visited = np.zeros((h, w), dtype=np.uint8)
    visited[start_y, start_x] = 1
    queue = deque([(start_x, start_y)])
    touches_edge = False

    while queue:
        x, y = queue.popleft()
        if x == 0 or x == w - 1 or y == 0 or y == h - 1:
            touches_edge = True
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and not wall[ny, nx]:
                visited[ny, nx] = 1
                queue.append((nx, ny))

    return touches_edge, visited


def compute_zone_boundary_length_and_area(
    wall_mask: np.ndarray,
    zone_center_x: int,
    zone_center_y: int,
    h: int,
    w: int,
) -> tuple[int, int]:
    """
    Pentru o zonă închisă (balcon/wintergarden): flood din colțuri = exterior,
    flood din centrul zonei = interior. Pereții care au vecin exterior și vecin interior
    = pereți exteriori ai zonei. Returnează (boundary_length_px, area_px).
    """
    wall = (np.asarray(wall_mask) > 0).astype(np.uint8)
    flood_base = np.where(wall > 0, 0, 255).astype(np.uint8)

    # Flood din colțuri + margini = exterior (128); modifică imaginea direct
    img = flood_base.copy()
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    step = max(1, min(50, w // 10, h // 10))
    for px in range(0, w, step):
        seeds.append((px, 0))
        seeds.append((px, h - 1))
    for py in range(0, h, step):
        seeds.append((0, py))
        seeds.append((w - 1, py))
    for sx, sy in seeds:
        if sy >= h or sx >= w or img[sy, sx] != 255:
            continue
        cv2.floodFill(img, None, (sx, sy), 128, None, None, 4)

    # Flood din centrul zonei = interior (64)
    cx = max(0, min(w - 1, zone_center_x))
    cy = max(0, min(h - 1, zone_center_y))
    if img[cy, cx] != 255:
        return 0, 0
    cv2.floodFill(img, None, (cx, cy), 64, None, None, 4)
    exterior = (img == 128)
    interior = (img == 64)
    area_px = int(np.sum(interior))

    # Graniță = pixeli perete cu vecin exterior și vecin interior
    boundary_length_px = 0
    for y in range(h):
        for x in range(w):
            if wall[y, x] == 0:
                continue
            has_ext = False
            has_int = False
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if exterior[ny, nx]:
                        has_ext = True
                    if interior[ny, nx]:
                        has_int = True
            if has_ext and has_int:
                boundary_length_px += 1

    return boundary_length_px, area_px


def save_check_garage_image(
    img: np.ndarray,
    visited: np.ndarray,
    mask_red: np.ndarray,
    path: Path | str,
) -> None:
    """Salvează imaginea check_garage: plan fără bulină + flood fill complet (overlay cyan)."""
    if len(img.shape) == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    out[mask_red > 0] = (0, 0, 0)
    overlay = out.copy()
    fill_bgr = (255, 200, 100)  # BGR cyan
    overlay[visited > 0] = (overlay[visited > 0].astype(np.int32) * 0.4 + np.array(fill_bgr) * 0.6).astype(np.uint8)
    cv2.imwrite(str(path), overlay)


def draw_segment_on_mask(
    mask: np.ndarray,
    result: dict,
    value: int = 255,
    extension_px: int = 6,
) -> None:
    """
    Desenează segmentul peretelui lipsă pe mască (1px), in-place.
    extension_px: prelungește segmentul în ambele sensuri ca să se lipească de pereții existenți.
    """
    h, w = mask.shape[:2]
    ext = max(0, extension_px)
    if result["type"] == "vertical":
        y1 = max(0, result["y1"] - ext)
        y2 = min(h - 1, result["y2"] + ext)
        cv2.line(mask, (result["x"], y1), (result["x"], y2), value, 1)
    else:
        x1 = max(0, result["x1"] - ext)
        x2 = min(w - 1, result["x2"] + ext)
        cv2.line(mask, (x1, result["y"]), (x2, result["y"]), value, 1)


def remove_walls_adjacent_to_region(
    wall_mask: np.ndarray,
    visited: np.ndarray,
) -> int:
    """
    Setează la 0 toți pixelii de perete (wall_mask > 0) care au cel puțin un vecin 4-conectat
    în regiunea umplută (visited > 0). Returnează numărul de pixeli eliminați.
    """
    h, w = wall_mask.shape
    to_remove = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if wall_mask[y, x] == 0:
                continue
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and visited[ny, nx] > 0:
                    to_remove[y, x] = 255
                    break
    n_removed = int(np.sum(to_remove > 0))
    wall_mask[to_remove > 0] = 0
    return n_removed
