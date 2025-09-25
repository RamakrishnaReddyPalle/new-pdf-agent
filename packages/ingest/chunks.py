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
    max_chars: int = 1800
    overlap: int = 100
    drop_gibberish: bool = True
    drop_toc: bool = True
    min_align_score: int = 70
    # NEW (backward compatible defaults)
    attach_heading_to_body: bool = True
    keep_heading_only: bool = False  # set True only if you really want separate heading chunks

def _read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _load_pages_jsonl(p: Optional[Path]) -> List[Dict[str, Any]]:
    if not p or not p.exists(): return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

def _fix_spaced_letters(s: str) -> str:
    return _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), s)

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u0131", "i").replace("\u0130", "I")
    s = _fix_spaced_letters(s)
    s = _DOT_LEADERS.sub(" … ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

def _is_gibberish(s: str) -> bool:
    t = s.strip()
    if len(t) < 30: return False
    alpha = sum(c.isalpha() for c in t)
    if alpha == 0: return True
    words = re.findall(r"[A-Za-z]+", t)
    if not words: return True
    vowels = set("aeiouAEIOU")
    vowelful = sum(1 for w in words if any(ch in vowels for ch in w))
    return (alpha / max(1, len(t)) < 0.6) and (vowelful / max(1, len(words)) < 0.5)

def _chunk_long(text: str, max_chars: int, overlap: int) -> List[str]:
    text = _normalize(text)
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
                cur = (out[-1][-overlap:] + " " + s).strip()
            else:
                cur = s.strip()
    if cur:
        out.append(cur)
    return out

def _parse_md(md: str) -> List[Dict[str, Any]]:
    lines = md.splitlines()
    blocks: List[Dict[str, Any]] = []
    heading_stack: List[str] = []
    in_code = False
    current = {"type": None, "lines": []}

    def flush_current():
        nonlocal current
        if current["type"]:
            text = "\n".join(current["lines"]).strip()
            text = _normalize(text)
            if text:
                blocks.append({"type": current["type"], "text": text, "heading_path": heading_stack.copy()})
        current = {"type": None, "lines": []}

    i, N = 0, len(lines)
    while i < N:
        ln = _normalize(lines[i])

        if ln.startswith("```"):
            if not in_code:
                flush_current(); in_code = True; current = {"type": "code", "lines": []}
            else:
                in_code = False; flush_current()
            i += 1; continue

        if in_code:
            current["lines"].append(ln); i += 1; continue

        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush_current()
            level = len(m.group(1)); title = m.group(2).strip()
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            # NOTE: don't immediately add heading-only chunk; we’ll attach to next body
            blocks.append({"type": "heading_marker", "text": title, "heading_path": heading_stack.copy()})
            i += 1; continue

        if "|" in ln and re.search(r"\|\s*-{3,}\s*\|", ln):
            flush_current()
            tbl_lines = [ln]; i += 1
            while i < N and "|" in _normalize(lines[i]):
                tbl_lines.append(_normalize(lines[i])); i += 1
            blocks.append({"type": "table", "text": "\n".join(tbl_lines), "heading_path": heading_stack.copy()})
            continue

        if re.match(r"^\s*([-*+]|\d+\.)\s+", ln):
            if current["type"] != "list":
                flush_current(); current = {"type": "list", "lines": []}
            current["lines"].append(ln); i += 1; continue

        if re.search(r"!\[.*?\]\(.*?\)", ln):
            flush_current()
            blocks.append({"type": "image", "text": ln.strip(), "heading_path": heading_stack.copy()})
            i += 1; continue

        if not ln.strip():
            flush_current(); i += 1; continue

        if current["type"] not in (None, "para"):
            flush_current()
        current["type"] = "para"
        current["lines"].append(ln); i += 1

    flush_current()
    return blocks

def _is_toc_block(btype: str, heading_path: List[str], text: str) -> bool:
    hp = " / ".join(h.lower() for h in heading_path)
    if "contents" in hp: return True
    if btype in ("table","list","para"):
        if text.count("|") >= 3: return True
        if _DOT_LEADERS.search(text): return True
    return False

def _align_to_pages(text: str, pages: List[Dict[str, Any]], min_score: int) -> Dict[str, Optional[int]]:
    if not pages or not _HAS_RAPID: return {"page_start": None, "page_end": None}
    probe = text[:400]
    best_score, best_page = -1, None
    for rec in pages:
        score = fuzz.partial_ratio(probe, (rec.get("text") or "")[:4000])
        if score > best_score:
            best_score, best_page = score, rec.get("page")
    if best_score < min_score:
        return {"page_start": None, "page_end": None}
    p = int(best_page) if best_page is not None else None
    return {"page_start": p, "page_end": p}

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
    pending_headings: List[str] = []

    def heading_prefix(hpath: List[str]) -> str:
        return " ▸ ".join(hpath[-3:])  # last 3 levels are enough

    with out_path.open("w", encoding="utf-8") as f:
        for b in blocks:
            btype, btext, hpath = b["type"], b["text"], b["heading_path"]

            # keep TOC & gibberish filters
            if cfg.drop_toc and _is_toc_block(btype, hpath, btext):
                continue
            if cfg.drop_gibberish and btype in ("para","list") and _is_gibberish(btext):
                continue

            if btype == "heading_marker":
                # buffer headings; we’ll attach to the next body
                pending_headings = hpath.copy()
                if cfg.keep_heading_only:
                    meta = {"doc_id": doc_id, "block_type": "heading", "heading_path": hpath}
                    meta.update(_align_to_pages(btext, pages, cfg.min_align_score))
                    n_chunks += 1
                    rec = {"id": f"{doc_id}-h-{n_chunks}", "text": btext, "metadata": meta}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            # attach heading text prefix to body/table/list/code chunks
            body_text = btext
            if cfg.attach_heading_to_body and pending_headings:
                pref = heading_prefix(pending_headings)
                body_text = f"{pref} — {btext}"

            # split & write
            for sub in _chunk_long(body_text, cfg.max_chars, cfg.overlap):
                meta = {"doc_id": doc_id, "block_type": btype, "heading_path": hpath}
                meta.update(_align_to_pages(sub, pages, cfg.min_align_score))
                n_chunks += 1
                rec = {"id": f"{doc_id}-{n_chunks}", "text": sub, "metadata": meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # once a body arrived, clear pending headings
            if btype in ("para","list","table","code","image"):
                pending_headings = []

    return n_chunks
