# file: engine/segmenter/pdf_utils.py
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
import shutil as _shutil

from pdf2image import pdfinfo_from_path
from PIL import Image, ImageFilter

from .common import (
    REQUESTED_DPI,
    DOWNSAMPLE_TARGET_DPI,
    MAX_RENDER_DIM,
    debug_print,
)

def _which(x: str) -> bool:
    return _shutil.which(x) is not None


def _safe_dpi_for_page(w_pt: float, h_pt: float, req_dpi: int) -> int:
    """
    Calculează DPI optim: Calitate MAXIMĂ (300 DPI), dar fără a ucide serverul.
    Returnează un DPI care garantează detalii clare (print quality),
    dar limitează rezoluția totală la ~8000px pe latura lungă.
    """
    # 300 DPI este standardul de aur pentru print și OCR.
    # Orice este peste 300 DPI pe un plan arhitectural este risipă de RAM (invizibil pentru AI).
    MAX_SAFE_DPI = 300
    
    # Punem o limită de siguranță în pixeli (8000px pe latura lungă)
    # 8000px este imens (aproape 8K). Detaliile sunt păstrate perfect.
    HARD_LIMIT_PX = 8000.0
    
    w_pt = max(w_pt, 1.0)
    h_pt = max(h_pt, 1.0)

    # Calculăm DPI-ul care ar rezulta în HARD_LIMIT_PX
    max_dpi_w = (HARD_LIMIT_PX / w_pt) * 72.0
    max_dpi_h = (HARD_LIMIT_PX / h_pt) * 72.0
    px_limit_dpi = min(max_dpi_w, max_dpi_h)
    
    # Alegem minimul dintre:
    # 1. Ce a cerut sistemul (req_dpi - de obicei 900)
    # 2. Limita de 300 DPI (calitate print)
    # 3. Limita de pixeli (să nu depășim 8000px și să crăpăm RAM-ul)
    safe_dpi = min(req_dpi, MAX_SAFE_DPI, int(px_limit_dpi))
    
    # Minim 72 DPI ca să se vadă ceva
    return max(72, safe_dpi)


def _verify_png(path: Path) -> None:
    try:
        im = Image.open(path)
        im.load()
        debug_print(f"✅ PNG: {path} ({im.width}x{im.height})")
    except Exception as e:
        debug_print(f"❌ PNG invalid {path}: {e}")


def _downsample_and_sharpen(src_path: Path, target_path: Path, scale_factor: float | None) -> None:
    # Folosim un bloc try-except pentru a gestiona erorile de memorie la resize (PIL)
    try:
        im = Image.open(src_path).convert("RGB")
        if scale_factor is not None and scale_factor < 1.0:
            new_w = max(1, int(im.width * scale_factor))
            new_h = max(1, int(im.height * scale_factor))
            im = im.resize((new_w, new_h), Image.LANCZOS)
            im = im.filter(ImageFilter.UnsharpMask(radius=0.75, percent=120, threshold=2))
        im.save(target_path, "PNG")
        _verify_png(target_path)
    except OSError:
        # Fallback: Dacă nu avem RAM de resize, copiem fișierul original așa cum e
        debug_print(f"⚠️ Eroare memorie la _downsample_and_sharpen. Copiez fisierul original.")
        _shutil.copy(src_path, target_path)


def _render_with_mutool(pdf_path: Path, page_idx: int, dpi: int, out_png: Path) -> None:
    page_spec = f"{page_idx}-{page_idx}"
    tool = "mutool" if _which("mutool") else ("mudraw" if _which("mudraw") else None)
    if tool is None:
        raise RuntimeError("MuPDF (mutool/mudraw) indisponibil")
    cmd = [
        tool,
        "draw",
        "-o",
        str(out_png),
        "-r",
        str(dpi),
        "-F",
        "png",
        "-c",
        "rgb",
        "-A",
        "8",
        str(pdf_path),
        page_spec,
    ]
    subprocess.check_call(cmd)


def _render_with_pdftoppm(pdf_path: Path, page_idx: int, dpi: int, out_prefix: Path) -> Path:
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        "-f",
        str(page_idx),
        "-l",
        str(page_idx),
        "-aa",
        "yes",
        "-aaVector",
        "yes",
        str(pdf_path),
        str(out_prefix),
    ]
    subprocess.check_call(cmd)
    return out_prefix.parent / f"{out_prefix.name}-{page_idx}.png"


def _render_with_ghostscript(pdf_path: Path, page_idx: int, dpi: int, out_png: Path) -> None:
    cmd = [
        "gs",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pngalpha",
        f"-r{dpi}",
        f"-dFirstPage={page_idx}",
        f"-dLastPage={page_idx}",
        "-dTextAlphaBits=4",
        "-dGraphicsAlphaBits=4",
        "-o",
        str(out_png),
        str(pdf_path),
    ]
    subprocess.check_call(cmd)


def convert_pdf_to_png(pdf_path: str | Path, output_dir: str | Path) -> list[Path]:
    """
    Convertește PDF-ul în PNG-uri de pagină și le pune în output_dir.
    Returnează lista de path-uri PNG (în ordine).
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = pdfinfo_from_path(str(pdf_path), userpw=None)
    page_count = int(info.get("Pages", 1))

    if "Page size" in info:
        try:
            parts = info["Page size"].split("x")
            default_w_pt = float(parts[0].strip())
            default_h_pt = float(parts[1].split()[0].strip())
        except Exception:
            default_w_pt, default_h_pt = 595.0, 842.0
    else:
        default_w_pt, default_h_pt = 595.0, 842.0

    have_mutool = _which("mutool") or _which("mudraw")
    have_pdftoppm = _which("pdftoppm")
    have_gs = _which("gs")

    out_paths: list[Path] = []

    for page_idx in range(1, page_count + 1):
        key = f"Page {page_idx} size"
        if key in info:
            try:
                parts = info[key].split("x")
                w_pt = float(parts[0].strip())
                h_pt = float(parts[1].split()[0].strip())
            except Exception:
                w_pt, h_pt = default_w_pt, default_h_pt
        else:
            w_pt, h_pt = default_w_pt, default_h_pt

        page_done = False
        last_error: Exception | None = None

        for req in REQUESTED_DPI:
            # AICI APELĂM NOUA LOGICĂ DE SIGURANȚĂ
            dpi = _safe_dpi_for_page(w_pt, h_pt, req)
            
            with tempfile.TemporaryDirectory() as tmpd_str:
                tmpd = Path(tmpd_str)
                raw_png = tmpd / f"page_{page_idx:03d}.png"

                if not page_done and have_mutool:
                    try:
                        debug_print(f"🖨️  MuPDF p.{page_idx} @ req {req} → safe {dpi} DPI ...")
                        _render_with_mutool(pdf_path, page_idx, dpi, raw_png)
                        page_done = True
                    except Exception as e:
                        last_error = e
                        debug_print(f"⚠️  MuPDF p.{page_idx} @ {dpi} DPI a eșuat: {e}")

                if not page_done and have_pdftoppm:
                    try:
                        debug_print(f"🖨️  Poppler p.{page_idx} @ req {req} → safe {dpi} DPI ...")
                        out_prefix = tmpd / "out"
                        raw_png_ppm = _render_with_pdftoppm(pdf_path, page_idx, dpi, out_prefix)
                        raw_png_ppm.rename(raw_png)
                        page_done = True
                    except Exception as e:
                        last_error = e
                        debug_print(f"⚠️  Poppler p.{page_idx} @ {dpi} DPI a eșuat: {e}")

                if not page_done and have_gs:
                    try:
                        debug_print(f"🖨️  Ghostscript p.{page_idx} @ req {req} → safe {dpi} DPI ...")
                        _render_with_ghostscript(pdf_path, page_idx, dpi, raw_png)
                        page_done = True
                    except Exception as e:
                        last_error = e
                        debug_print(f"⚠️  Ghostscript p.{page_idx} @ {dpi} DPI a eșuat: {e}")

                if page_done:
                    final_path = output_dir / f"page_{page_idx:03d}.png"
                    if DOWNSAMPLE_TARGET_DPI and DOWNSAMPLE_TARGET_DPI < dpi:
                        scale = DOWNSAMPLE_TARGET_DPI / float(dpi)
                    else:
                        scale = None
                    _downsample_and_sharpen(raw_png, final_path, scale)
                    out_paths.append(final_path)
                    break

        if not page_done:
            raise RuntimeError(f"Eșec conversie pagina {page_idx}. Ultima eroare: {last_error}")

    debug_print(f"📄 Conversie finalizată → {len(out_paths)} PNG-uri de calitate.")
    return out_paths