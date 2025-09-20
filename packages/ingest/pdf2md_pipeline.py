# packages/ingest/pdf2md_pipeline.py
from __future__ import annotations
import io, json, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import fitz  # PyMuPDF
from PIL import Image

# Providers
from packages.providers.pdf2md.marker import MarkerLikeProvider
from packages.providers.ocr.tesseract import TesseractOCR


@dataclass
class PDF2MDConfig:
    ocr_enable: bool = True
    ocr_dpi: int = 300
    ocr_lang: str = "eng"
    min_chars_no_ocr: int = 60
    preserve_footnotes: bool = True
    keep_figure_captions: bool = True
    # sampling (optional)
    sample_n_pages: Optional[int] = None
    sample_random_seed: int = 42
    # override to skip docling/marker and build from OCR text
    force_ocr_only: bool = False
    # lower DPI for sampling runs
    sample_ocr_dpi: Optional[int] = None


@dataclass
class PDF2MDOutputs:
    markdown_path: Path
    toc_json_path: Path
    pages_jsonl_path: Path
    processed_pages: List[int]         # <--- NEW
    images_dir: Optional[Path] = None


# ---------- helpers ----------

_IMG_COMMENT = "<!-- image -->"

def _is_md_image_only(md_text: str) -> bool:
    lines = [l for l in (md_text or "").splitlines() if l.strip()]
    if not lines:
        return True
    img_lines = sum(1 for l in lines if _IMG_COMMENT in l or l.strip().startswith("![]("))
    letters = sum(1 for ch in md_text if ch.isalpha())
    return (img_lines / max(1, len(lines)) > 0.5) or (letters < 500)


_HEADING_PATTERNS = [
    re.compile(r"^(Chapter|CHAPTER)\s+[\wIVXLC]+(\.|:)?\s+.+$"),
    re.compile(r"^(Section|SECTION|Article|ARTICLE)\s+[\w\.\-]+(\.|:)?\s+.+$"),
    re.compile(r"^\d+(\.\d+)+\s+[A-Z][A-Za-z0-9 ,;\-\(\)]{5,}$"),
    re.compile(r"^[A-Z][A-Z0-9 \-]{8,}$"),
]

def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 6:
        return False
    for pat in _HEADING_PATTERNS:
        if pat.match(s):
            return True
    return False

def _synthetic_toc_from_pages(pages: List[Dict[str, Any]], max_items_per_page: int = 3) -> List[Dict[str, Any]]:
    toc: List[Dict[str, Any]] = []
    for rec in pages:
        page_no = int(rec.get("page") or 0)
        txt = (rec.get("text") or "").splitlines()
        hits = 0
        for ln in txt[:40]:
            if _looks_like_heading(ln):
                level = 1
                m = re.match(r"^(\d+(?:\.\d+)+)\s+", ln.strip())
                if m:
                    level = min(1 + m.group(1).count("."), 6)
                toc.append({"level": level, "title": ln.strip(), "page": page_no})
                hits += 1
                if hits >= max_items_per_page:
                    break
    return toc


# ---------- core ----------

def _extract_toc_and_pages(
    pdf_path: Path,
    pages_jsonl: Path,
    toc_json: Path,
    ocr_enable: bool,
    ocr_dpi: int,
    ocr_lang: str,
    min_chars_no_ocr: int,
    page_subset: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if page_subset:
        page_subset = sorted({p for p in page_subset if 1 <= p <= total_pages})
    else:
        page_subset = list(range(1, total_pages + 1))

    native_toc = doc.get_toc(simple=True) or []
    native = [{"level": int(l), "title": (t or "").strip(), "page": int(p)}
              for (l, t, p) in native_toc if int(p) in page_subset]

    pages_out: List[Dict[str, Any]] = []
    pages_jsonl.parent.mkdir(parents=True, exist_ok=True)
    ocr = TesseractOCR() if ocr_enable else None

    with pages_jsonl.open("w", encoding="utf-8") as f:
        for pno in page_subset:
            page = doc.load_page(pno - 1)
            text = page.get_text("text") or ""
            rec: Dict[str, Any] = {"page": pno, "text": text}

            if ocr and len((text or "").strip()) < min_chars_no_ocr:
                pix = page.get_pixmap(dpi=ocr_dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                try:
                    ocr_text = ocr.image_to_string(img, lang=ocr_lang) or ""
                    if ocr_text.strip():
                        rec["ocr_used"] = True
                        rec["text"] = (text + "\n" + ocr_text).strip() if text else ocr_text.strip()
                except Exception as e:
                    rec["ocr_error"] = str(e)

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pages_out.append(rec)

    toc_items = native if native else _synthetic_toc_from_pages(pages_out)
    toc_json.write_text(json.dumps(toc_items, ensure_ascii=False, indent=2), encoding="utf-8")
    return pages_out


def _build_markdown_from_ocr_pages(pages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for rec in pages:
        ptxt = (rec.get("text") or "").splitlines()
        buf: List[str] = []
        for ln in ptxt:
            s = ln.strip()
            if not s:
                if buf and buf[-1] != "":
                    buf.append("")
                continue
            if _looks_like_heading(s):
                if buf and buf[-1] != "":
                    buf.append("")
                buf.append(f"## {s}")
                buf.append("")
            else:
                buf.append(s)
        chunk = "\n".join(buf).strip()
        if chunk:
            parts.append(chunk)
        parts.append("\n---\n")
    md = "\n".join(parts).strip()
    if not md:
        md = "# Document\n" + "\n".join((rec.get("text") or "") for rec in pages)
    return md


def run_pdf_to_markdown(
    doc_id: str,
    pdf_path: Path,
    out_dir: Path,
    cfg: PDF2MDConfig,
    page_subset: Optional[List[int]] = None,
) -> PDF2MDOutputs:
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.sample_n_pages:
        doc = fitz.open(str(pdf_path))
        total = len(doc)
        rng = random.Random(cfg.sample_random_seed)
        if page_subset:
            pool = page_subset
        else:
            pool = list(range(1, total + 1))
        if cfg.sample_n_pages < len(pool):
            page_subset = sorted(rng.sample(pool, cfg.sample_n_pages))
        else:
            page_subset = sorted(pool)

    effective_dpi = cfg.sample_ocr_dpi or cfg.ocr_dpi
    pages_jsonl = out_dir / f"{doc_id}.pages.jsonl"
    toc_json = out_dir / f"{doc_id}.toc.json"
    pages = _extract_toc_and_pages(
        pdf_path=pdf_path,
        pages_jsonl=pages_jsonl,
        toc_json=toc_json,
        ocr_enable=cfg.ocr_enable,
        ocr_dpi=effective_dpi,
        ocr_lang=cfg.ocr_lang,
        min_chars_no_ocr=cfg.min_chars_no_ocr,
        page_subset=page_subset,
    )

    md_text: Optional[str] = None
    if not cfg.force_ocr_only and not cfg.sample_n_pages:
        provider = MarkerLikeProvider(
            preserve_footnotes=cfg.preserve_footnotes,
            keep_figure_captions=cfg.keep_figure_captions,
        )
        try:
            md_text = provider.to_markdown(str(pdf_path))
        except Exception:
            md_text = None

    if not md_text or _is_md_image_only(md_text):
        md_text = _build_markdown_from_ocr_pages(pages)

    md_path = out_dir / f"{doc_id}.md"
    md_path.write_text(md_text, encoding="utf-8")

    processed = [int(p.get("page")) for p in pages if p.get("page") is not None]

    return PDF2MDOutputs(
        markdown_path=md_path,
        toc_json_path=toc_json,
        pages_jsonl_path=pages_jsonl,
        processed_pages=processed,    # <---
        images_dir=None,
    )
