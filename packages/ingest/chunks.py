# hierarchical chunking + TOC graph
# packages/ingest/chunks.py
from __future__ import annotations
import json, re, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from rapidfuzz import fuzz
    _HAS_RAPID = True
except Exception:
    _HAS_RAPID = False

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b")
_DOT_LEADERS = re.compile(r"(?:\.\s){3,}\.?\s*")

@dataclass
class ChunkingConfig:
    max_chars: int = 1200
    overlap: int = 200
    drop_gibberish: bool = True
    drop_toc: bool = True
    min_align_score: int = 70

# ---------- I/O ----------

def _read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _load_pages_jsonl(pages_path: Optional[Path]) -> List[Dict[str, Any]]:
    if not pages_path or not pages_path.exists():
        return []
    pages: List[Dict[str, Any]] = []
    with pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                pages.append(json.loads(line))
            except Exception:
                continue
    return pages

# ---------- normalize & heuristics ----------

def _fix_spaced_letters(s: str) -> str:
    return _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), s)

def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u0131", "i").replace("\u0130", "I")
    s = _fix_spaced_letters(s)
    s = _DOT_LEADERS.sub(" … ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

def _is_gibberish(s: str) -> bool:
    t = s.strip()
    if len(t) < 30:  # keep small headings/labels
        return False
    alpha = sum(c.isalpha() for c in t)
    if alpha == 0:
        return True
    alpha_ratio = alpha / max(1, len(t))
    words = re.findall(r"[A-Za-z]+", t)
    if not words:
        return True
    vowels = set("aeiouAEIOU")
    vowelful = sum(1 for w in words if any(ch in vowels for ch in w))
    vowel_ratio = vowelful / max(1, len(words))
    return (alpha_ratio < 0.6) and (vowel_ratio < 0.5)

# ---------- parsing ----------

def _chunk_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = _normalize_text(text)
    if len(text) <= max_chars:
        return [text]
    sents = _SENT_SPLIT.split(text)
    out: List[str] = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                out.append(cur)
            if overlap > 0 and out:
                tail = out[-1][-overlap:]
                cur = (tail + " " + s).strip()
            else:
                cur = s.strip()
    if cur:
        out.append(cur)
    return out

def _parse_md(md: str) -> List[Dict[str, Any]]:
    lines = md.splitlines()
    blocks: List[Dict[str, Any]] = []
    heading_stack: List[str] = []
    i, N = 0, len(lines)
    in_code = False
    current = {"type": None, "lines": []}

    def flush_current():
        nonlocal current
        if current["type"]:
            text = "\n".join(current["lines"]).strip()
            text = _normalize_text(text)
            if text:
                blocks.append({"type": current["type"], "text": text, "heading_path": heading_stack.copy()})
        current = {"type": None, "lines": []}

    while i < N:
        ln = _normalize_text(lines[i])

        # code fences
        if ln.startswith("```"):
            if not in_code:
                flush_current()
                in_code = True
                current = {"type": "code", "lines": []}
            else:
                in_code = False
                flush_current()
            i += 1
            continue

        if in_code:
            current["lines"].append(ln); i += 1; continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush_current()
            level = len(m.group(1))
            title = m.group(2).strip()
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            blocks.append({"type": "heading", "text": title, "heading_path": heading_stack.copy()})
            i += 1; continue

        # simple table block
        if "|" in ln and re.search(r"\|\s*-{3,}\s*\|", ln):
            flush_current()
            tbl_lines = [ln]; i += 1
            while i < N and "|" in _normalize_text(lines[i]):
                tbl_lines.append(_normalize_text(lines[i])); i += 1
            blocks.append({"type": "table", "text": "\n".join(tbl_lines), "heading_path": heading_stack.copy()})
            continue

        # list
        if re.match(r"^\s*([-*+]|\d+\.)\s+", ln):
            if current["type"] != "list":
                flush_current()
                current = {"type": "list", "lines": []}
            current["lines"].append(ln); i += 1; continue

        # image
        if re.search(r"!\[.*?\]\(.*?\)", ln):
            flush_current()
            blocks.append({"type": "image", "text": ln.strip(), "heading_path": heading_stack.copy()})
            i += 1; continue

        # paragraph boundary
        if not ln.strip():
            flush_current(); i += 1; continue

        # paragraph
        if current["type"] not in (None, "para"):
            flush_current()
        current["type"] = "para"
        current["lines"].append(ln); i += 1

    flush_current()
    return blocks

# ---------- TOC / alignment ----------

def _is_toc_block(block_type: str, heading_path: List[str], text: str) -> bool:
    hp = " / ".join(h.lower() for h in heading_path)
    if "contents" in hp:
        return True
    if block_type in ("table", "list", "para"):
        if text.count("|") >= 3:
            return True
        if _DOT_LEADERS.search(text):
            return True
    return False

def _align_to_pages(text: str, pages: List[Dict[str, Any]], min_score: int) -> Dict[str, Optional[int]]:
    if not pages or not _HAS_RAPID:
        return {"page_start": None, "page_end": None}
    probe = text[:400]
    best_score, best_page = -1, None
    for rec in pages:
        score = fuzz.partial_ratio(probe, (rec.get("text") or "")[:4000])
        if score > best_score:
            best_score, best_page = score, rec.get("page")
    if best_score < min_score:
        return {"page_start": None, "page_end": None}
    p = int(best_page) if best_page else None
    return {"page_start": p, "page_end": p}

# ---------- public API ----------

def md_to_chunks(
    doc_id: str,
    md_path: Path,
    out_path: Path,
    cfg: ChunkingConfig,
    pages_jsonl: Optional[Path] = None,
) -> int:
    md = _read_md(md_path)
    pages = _load_pages_jsonl(pages_jsonl) if pages_jsonl else []
    blocks = _parse_md(md)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with out_path.open("w", encoding="utf-8") as f:
        for b in blocks:
            btype = b["type"]; btext = b["text"]; hpath = b["heading_path"]

            if cfg.drop_toc and _is_toc_block(btype, hpath, btext):
                continue
            if cfg.drop_gibberish and btype in ("para", "list") and _is_gibberish(btext):
                continue

            if btype == "heading":
                meta = {"doc_id": doc_id, "block_type": "heading", "heading_path": hpath}
                meta.update(_align_to_pages(btext, pages, cfg.min_align_score))
                n_chunks += 1
                f.write(json.dumps({"id": f"{doc_id}-h-{n_chunks}", "text": btext, "metadata": meta}, ensure_ascii=False) + "\n")
                continue

            for sub in _chunk_long_text(btext, max_chars=cfg.max_chars, overlap=cfg.overlap):
                meta = {"doc_id": doc_id, "block_type": btype, "heading_path": hpath}
                meta.update(_align_to_pages(sub, pages, cfg.min_align_score))
                n_chunks += 1
                f.write(json.dumps({"id": f"{doc_id}-{n_chunks}", "text": sub, "metadata": meta}, ensure_ascii=False) + "\n")
    return n_chunks
