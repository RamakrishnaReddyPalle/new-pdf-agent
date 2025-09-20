# packages/ingest/tables_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import io, csv, json

import fitz  # PyMuPDF
from PIL import Image

# Optional: PaddleOCR PP-Structure
_HAS_PADDLE = False
try:
    from paddleocr import PPStructure
    _HAS_PADDLE = True
except Exception:
    _HAS_PADDLE = False

# Optional: Table Transformer (TATR)
_HAS_TATR = False
try:
    from packages.providers.layout.table_transformer import TableTransformer, TATRConfig
    _HAS_TATR = True
except Exception:
    _HAS_TATR = False


@dataclass
class TablesConfig:
    ocr_dpi: int = 300
    ocr_lang: str = "en"
    use_paddle: bool = True
    export_markdown: bool = True
    max_cols: int = 6
    merge_wrap_lines: bool = True
    min_row_height: float = 6.0
    # TATR cfg (if installed)
    tatr_detect_model: str = "microsoft/table-transformer-detection"              # HF id or local dir
    tatr_structure_model: str = "microsoft/table-transformer-structure-recognition"
    tatr_conf_thresh: float = 0.60
    tatr_device: str = "cpu"                                                     # "cpu" | "cuda" | "mps"


def _image_from_page(page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _rows_to_markdown(header: List[str], body: List[List[str]]) -> str:
    if not header:
        return ""
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        out.append("| " + " | ".join(r + [""] * (len(header) - len(r))) + " |")
    return "\n".join(out)


# ----------------------- TATR path (detection + structure) -----------------------

def _bbox_intersection(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]

def _normalize_header(header: List[str]) -> List[str]:
    out = []
    for i, h in enumerate(header):
        hh = (h or "").strip()
        out.append(hh if hh else f"col{i+1}")
    return out

def _text_in_bbox(page, bbox) -> str:
    words = page.get_text("words")  # x0, y0, x1, y1, word, block, line, word_no
    x0, y0, x1, y1 = bbox
    toks = []
    for w in words:
        cx = 0.5*(w[0]+w[2]); cy = 0.5*(w[1]+w[3])
        if (x0 <= cx <= x1) and (y0 <= cy <= y1):
            toks.append(w[4])
    return " ".join(toks).strip()

def _tatr_tables(page, img, tatr, conf_thresh):
    det = tatr.detect_tables(img)
    out = []

    # compute pixel→PDF scaling once
    sx = page.rect.width  / img.width
    sy = page.rect.height / img.height

    for tb in det:
        if tb.get("score", 0) < conf_thresh:
            continue

        struct = tatr.structure_blocks(img, tb["bbox"])
        # SCALE all structure bboxes from image pixels → PDF points
        for b in struct:
            x0, y0, x1, y1 = b["bbox"]
            b["bbox"] = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]

        rows = [b for b in struct if "row"    in b["label"].lower()]
        cols = [b for b in struct if "column" in b["label"].lower()]
        hdrs = [b for b in struct if "header" in b["label"].lower()]
        if not rows or not cols:
            continue

        rows = sorted(rows, key=lambda r: r["bbox"][1])
        cols = sorted(cols, key=lambda c: c["bbox"][0])

        cell_texts = []
        for rbox in rows:
            rline = []
            for cbox in cols:
                cell = _bbox_intersection(rbox["bbox"], cbox["bbox"])
                if cell is None:
                    rline.append("")
                else:
                    txt = _text_in_bbox(page, cell)  # now same coord system
                    rline.append(txt)
            cell_texts.append(rline)

        if not cell_texts:
            continue

        first_row_join = " ".join(cell_texts[0]).lower()
        if hdrs or any(k in first_row_join for k in ("code","desc","relative","value","split")):
            header = _normalize_header(cell_texts[0])
            body   = cell_texts[1:]
        else:
            ncols  = max(len(r) for r in cell_texts)
            header = [f"col{i+1}" for i in range(ncols)]
            body   = cell_texts

        out.append((header, body))
    return out


# ----------------------- Paddle path -----------------------

def _paddle_tables_from_image(img: Image.Image) -> List[Tuple[List[str], List[List[str]]]]:
    if not _HAS_PADDLE:
        return []
    engine = PPStructure(show_log=False, lang="en")
    res = engine(img)

    tables: List[Tuple[List[str], List[List[str]]]] = []
    for block in res:
        if block.get("type") != "table":
            continue
        html = block.get("res", {}).get("html", "")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            rows = []
            for tr in soup.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["th","td"])]
                rows.append(cells)
            if not rows:
                continue
            header = rows[0]
            body = rows[1:] if len(rows) > 1 else []
            tables.append((header, body))
        except Exception:
            continue
    return tables


# ----------------------- Coordinate (words) fallback -----------------------

def _get_words(page) -> List[Tuple[float,float,float,float,str,int,int,int]]:
    return page.get_text("words")

def _cluster_columns(words, max_cols=6) -> List[float]:
    if not words:
        return []
    centers = [(w[0] + w[2]) / 2.0 for w in words]
    min_x, max_x = min(centers), max(centers)
    bins = max(10, min(80, int((max_x - min_x) / 25)))
    hist = [0] * (bins + 1)
    for c in centers:
        bi = int((c - min_x) / max(1e-6, (max_x - min_x)) * bins)
        hist[min(bi, bins)] += 1
    peaks = sorted(range(len(hist)), key=lambda i: hist[i], reverse=True)[:max_cols]
    anchors = sorted(min_x + (p / bins) * (max_x - min_x) for p in peaks)
    dedup = []
    for a in anchors:
        if not dedup or abs(a - dedup[-1]) > 40:
            dedup.append(a)
    return dedup

def _assign_col(xc: float, anchors: List[float]) -> int:
    if not anchors:
        return 0
    return min(range(len(anchors)), key=lambda i: abs(xc - anchors[i]))

def _build_rows_from_words(words, anchors: List[float], min_row_h: float, merge_wrap: bool) -> Tuple[List[str], List[List[str]]]:
    if not words:
        return ([], [])
    words = sorted(words, key=lambda w: (w[1], w[0]))
    rows: List[List[List[str]]] = []
    y_thr = None; cur_y = None
    for (x0, y0, x1, y1, text, *_rest) in words:
        if not text.strip():
            continue
        xc = 0.5*(x0+x1)
        col = _assign_col(xc, anchors)
        if cur_y is None:
            rows.append([[] for _ in range(len(anchors) or 1)])
            rows[-1][col].append(text)
            cur_y = y0
            y_thr = max(min_row_h, 0.7*(y1-y0))
            continue
        if abs(y0 - cur_y) <= y_thr:
            rows[-1][col].append(text)
        else:
            rows.append([[] for _ in range(len(anchors) or 1)])
            rows[-1][col].append(text)
            cur_y = y0
    text_rows = [[" ".join(tok).strip() for tok in r] for r in rows]
    header, body = [], []
    if text_rows:
        hdr = text_rows[0]
        if any(k in " ".join(hdr).lower() for k in ("code", "desc", "relative", "value", "split")):
            header = _normalize_header(hdr); body = text_rows[1:]
        else:
            ncols = max(len(r) for r in text_rows)
            header = [f"col{i+1}" for i in range(ncols)]
            body = text_rows
    if merge_wrap and len(header) >= 2:
        desc_idx = next((i for i,h in enumerate(header) if "desc" in h.lower()), 1 if len(header)>=2 else None)
        merged: List[List[str]] = []
        for r in body:
            if merged and (not r[0] or r[0].strip()=="") and desc_idx is not None:
                merged[-1][desc_idx] = (merged[-1][desc_idx] + " " + (r[desc_idx] or "")).strip()
            else:
                merged.append(r + [""]*(len(header)-len(r)))
        body = merged
    return (header, body)


def _save_table_files(out_dir: Path, page_no: int, idx: int, header: List[str], body: List[List[str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"table_{page_no}_{idx}"
    obj = {"header": header, "rows": [dict(zip(header, r + [""] * (len(header) - len(r)))) for r in body]}
    base.with_suffix(".json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in body:
            w.writerow(r + [""] * (len(header) - len(r)))
    md = _rows_to_markdown(header, body)
    if md:
        base.with_suffix(".md").write_text(md, encoding="utf-8")


def extract_tables_from_pdf(
    doc_id: str,
    pdf_path: Path,
    out_dir: Path,
    cfg: TablesConfig,
    page_subset: Optional[List[int]] = None,
) -> int:
    out_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    N = len(doc)
    pages = sorted(set(p for p in (page_subset or range(1, N + 1)) if 1 <= p <= N))

    # Initialize TATR once
    tatr = None
    if _HAS_TATR:
        try:
            tatr = TableTransformer(TATRConfig(
                detect_model=cfg.tatr_detect_model,
                structure_model=cfg.tatr_structure_model,
                conf_thresh=cfg.tatr_conf_thresh,
                device=cfg.tatr_device,
            ))
        except Exception:
            tatr = None

    saved = 0
    for pno in pages:
        page = doc.load_page(pno - 1)
        img = _image_from_page(page, dpi=cfg.ocr_dpi)

        header_bodies: List[Tuple[List[str], List[List[str]]]] = []

        # 1) TATR (if available and initialized)
        if tatr is not None:
            try:
                header_bodies = _tatr_tables(page, img, tatr, cfg.tatr_conf_thresh)
            except Exception:
                header_bodies = []

        # 2) Paddle (if enabled and nothing yet)
        if not header_bodies and cfg.use_paddle and _HAS_PADDLE:
            try:
                header_bodies = _paddle_tables_from_image(img)
            except Exception:
                header_bodies = []

        # 3) word-grid fallback
        if not header_bodies:
            words = _get_words(page)
            anchors = _cluster_columns(words, max_cols=cfg.max_cols)
            h, b = _build_rows_from_words(words, anchors, cfg.min_row_height, cfg.merge_wrap_lines)
            if h and b:
                header_bodies = [(h, b)]

        for idx, (h, b) in enumerate(header_bodies, 1):
            if h and b:
                _save_table_files(out_dir, pno, idx, h, b)
                saved += 1

    return saved
