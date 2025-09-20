# text/scanned/mixed detection
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Dict, Any
import fitz  # PyMuPDF

Mode = Literal["text", "scanned", "mixed"]

@dataclass
class PageStat:
    page: int
    text_len_pymupdf: int

@dataclass
class ProfileResult:
    mode: Mode
    pages: List[PageStat]
    pct_texty_pages: float

def profile_document(
    pdf_path: Path,
    min_text_chars: int = 80,      # below this looks like scanned/low-text
    sample_first_n: int = 8,       # cheap pilot pass
    sample_random_n: int = 12,     # and a random handful
) -> ProfileResult:
    import random
    doc = fitz.open(str(pdf_path))
    N = len(doc)
    # pick first_n + random_n unique pages
    pool = list(range(1, N + 1))
    head = pool[:min(sample_first_n, N)]
    tail_pool = pool[len(head):]
    rand = random.sample(tail_pool, min(sample_random_n, len(tail_pool)))
    subset = sorted(set(head + rand))

    stats: List[PageStat] = []
    texty = 0
    for p in subset:
        page = doc.load_page(p - 1)
        t = page.get_text("text") or ""
        stats.append(PageStat(page=p, text_len_pymupdf=len(t.strip())))
        if len((t or "").strip()) >= min_text_chars:
            texty += 1

    pct = texty / max(1, len(subset))
    # heuristics:
    # >70% texty -> "text"
    # <30% texty -> "scanned"
    # else -> "mixed"
    if pct >= 0.7:
        mode: Mode = "text"
    elif pct <= 0.3:
        mode = "scanned"
    else:
        mode = "mixed"
    return ProfileResult(mode=mode, pages=stats, pct_texty_pages=pct)
