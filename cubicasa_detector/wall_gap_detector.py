# cubicasa_detector/wall_gap_detector.py
"""
Detectează peretele lipsă dintr-o cameră (ex: garaj) folosind BFS din punctul marker (bulina roșie).
Salvează pașii flood fill în detection_steps și produce plan_closed.png cu segmentul închis.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from collections import deque


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
):
    """
    Detectează peretele lipsă dintr-o cameră cu BFS din punctul marker.
    Bariera: orice pixel de perete blochează flood-ul; prin găuri reale flood-ul poate ieși.

    Args:
        img: Imagine BGR (plan cu pereți albi, fundal negru; opțional bulina roșie).
        start_x, start_y: Punct de start. Dacă None, se detectează bulina roșie.
        stable_min_px: Pixeli consecutivi cu aceeași dimensiune pentru Faza 2.
        steps_dir: Dacă setat, salvează frame-uri în acest folder.
        step_interval: La câte iterații salvează un frame (dacă steps_dir).
        wall_threshold: Valoare minimă gri considerată perete (folosit doar dacă wall_mask nu e dat).
        max_iterations: Limită BFS.
        wall_mask: Mască binară (h,w) – pixel > 0 = perete. Dacă dată, se folosește ca barieră exactă
            (flood-ul nu trece niciodată prin pereți; poate ieși doar prin zone cu 0). Ignoră wall_threshold.

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
        )
        if result is not None:
            return result
    return None


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
                if stable_count >= stable_min_px:
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
                        if steps_dir is not None:
                            _save_step(
                                steps_dir, iteration, img_bgr, visited, cur, mask_red, result
                            )
                            _save_step(
                                steps_dir, -1, img_bgr, visited, cur, mask_red, result
                            )  # step_final.png
                        return result
                    return None
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
                        if steps_dir is not None:
                            _save_step(
                                steps_dir, iteration, img_bgr, visited, cur, mask_red, result
                            )
                            _save_step(
                                steps_dir, -1, img_bgr, visited, cur, mask_red, result
                            )  # step_final.png
                        return result
                    return None

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
