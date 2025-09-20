# packages/ingest/math_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import json, re, io

from PIL import Image
import fitz  # PyMuPDF

# Optional LaTeX OCR (pix2tex)
_HAS_PIX2TEX = False
try:
    from pix2tex.cli import LatexOCR
    _HAS_PIX2TEX = True
except Exception:
    _HAS_PIX2TEX = False


@dataclass
class MathConfig:
    # text pattern detection
    min_digits: int = 2
    require_operator: bool = True
    # LaTeX OCR options
    latex_ocr_enable: bool = False
    latex_engine: str = "pix2tex"   # reserved for future engines
    crop_margin: int = 4            # px padding around detected line


# crude but effective patterns
_OP    = re.compile(r"[=+\-/*×÷^()]")
_RATIO = re.compile(r"\b\d{1,3}/\d{1,3}\b")
_NUM   = re.compile(r"\d+(?:\.\d+)?")


def _looks_like_formula(line: str, min_digits: int, require_op: bool) -> bool:
    s = line.strip()
    if len(s) < 3:
        return False
    nums = _NUM.findall(s)
    if len(nums) < min_digits:
        return False
    if require_op and not (_OP.search(s) or _RATIO.search(s)):
        return False
    return True


def _page_image(page, dpi=300) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _line_boxes(page) -> List[Dict[str, Any]]:
    """
    Aggregate span boxes into line boxes via page.get_text('dict').
    Returns [{'bbox':[x0,y0,x1,y1],'text': '...'}, ...]
    """
    data = page.get_text("dict")
    out: List[Dict[str, Any]] = []
    for blk in data.get("blocks", []):
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            x0 = min(sp["bbox"][0] for sp in spans)
            y0 = min(sp["bbox"][1] for sp in spans)
            x1 = max(sp["bbox"][2] for sp in spans)
            y1 = max(sp["bbox"][3] for sp in spans)
            text = "".join(sp.get("text", "") for sp in spans)
            out.append({"bbox": [x0, y0, x1, y1], "text": text})
    return out


def extract_formulas_from_pages_jsonl(
    doc_id: str,
    pages_jsonl: Path,
    out_dir: Path,
    cfg: MathConfig,
    pdf_path: Optional[Path] = None,
    dpi_for_crops: int = 300,
) -> int:
    """
    Baseline: scan OCR'd text lines to flag formula-like strings.
    If pdf_path is provided AND latex_ocr_enable is True AND pix2tex is available:
      - crop line images and run LaTeX OCR; save crops + latex string.
    Outputs: {out_dir}/math/formulas.jsonl (+ crops/*.png if enabled)
    """
    out_dir = out_dir / "math"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    if cfg.latex_ocr_enable:
        crops_dir.mkdir(exist_ok=True)

    # Optional LaTeX OCR model
    model = None
    if cfg.latex_ocr_enable and cfg.latex_engine == "pix2tex" and _HAS_PIX2TEX:
        model = LatexOCR()

    # If we’ll crop/latex, open the PDF once and cache per-page assets
    doc = None
    if cfg.latex_ocr_enable and model and pdf_path:
        doc = fitz.open(str(pdf_path))
        page_cache: Dict[int, Dict[str, Any]] = {}  # page_no -> {'page': fitz.Page, 'img': PIL.Image, 'lines': [...]}

    found = 0
    out_fp = out_dir / "formulas.jsonl"

    with pages_jsonl.open("r", encoding="utf-8") as f_in, out_fp.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            page_no = int(rec.get("page") or 0)
            text = rec.get("text") or ""
            if not text:
                continue

            # Fast, text-only detection first
            text_candidates: List[str] = []
            for raw in text.splitlines():
                s = raw.strip()
                if _looks_like_formula(s, cfg.min_digits, cfg.require_operator):
                    text_candidates.append(s)

            # If no LaTeX OCR path, just dump the matches
            if not (doc and model):
                for s in text_candidates:
                    f_out.write(json.dumps({
                        "doc_id": doc_id,
                        "page": page_no,
                        "text": s,
                        "kind": "formula-ish"
                    }, ensure_ascii=False) + "\n")
                    found += 1
                continue

            # With LaTeX OCR: get (or build) the per-page assets
            if page_no not in page_cache:
                p = doc.load_page(page_no - 1)
                page_cache[page_no] = {
                    "page": p,
                    "img": _page_image(p, dpi=dpi_for_crops),
                    "lines": _line_boxes(p)
                }

            page_img = page_cache[page_no]["img"]
            lines    = page_cache[page_no]["lines"]

            # Walk the geometric lines; keep only those that also pass the text filter
            for lb in lines:
                s = lb["text"].strip()
                if not _looks_like_formula(s, cfg.min_digits, cfg.require_operator):
                    continue

                x0, y0, x1, y1 = lb["bbox"]
                m = int(cfg.crop_margin)
                crop = page_img.crop((max(0, x0 - m), max(0, y0 - m), x1 + m, y1 + m))

                crop_path = crops_dir / f"p{page_no}_{found+1}.png"
                try:
                    crop.save(crop_path)
                except Exception:
                    crop_path = None

                latex = None
                if model:
                    try:
                        latex = model(crop)
                    except Exception:
                        latex = None

                f_out.write(json.dumps({
                    "doc_id": doc_id,
                    "page": page_no,
                    "text": s,
                    "kind": "formula-ish",
                    "bbox": [x0, y0, x1, y1],
                    "crop": str(crop_path) if crop_path else None,
                    "latex": latex
                }, ensure_ascii=False) + "\n")
                found += 1

    if doc is not None:
        doc.close()

    return found
