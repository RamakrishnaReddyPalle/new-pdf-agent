# packages/calculators/miner.py
from __future__ import annotations
import re, json, hashlib, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- Optional LLM (Ollama) ----
_HAS_OLLAMA = False
try:
    from packages.providers.llm.ollama import OllamaLLM, OllamaConfig  # uses your provider
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False

# ---- JSONL IO ----
def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows

def _write_jsonl(p: Path, rows: List[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- Configs ----
@dataclass
class MinerLLMConfig:
    enable: bool = True
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    temperature: float = 0.1
    max_new_tokens: int = 384
    connect_timeout: int = 30
    read_timeout: int = 600
    retries: int = 1

@dataclass
class MinerHeuristics:
    notation_headings: Tuple[str, ...] = ("definitions","notation","abbreviations","symbols","glossary","ground rules")
    # how we decide "mathy" lines
    min_math_symbols: int = 2        # count of operators in a line to treat as candidate formula
    min_line_len: int = 12
    max_line_len: int = 280
    # md pipe tables
    parse_pipe_tables: bool = True

@dataclass
class MinerConfig:
    use_llm: bool = True
    llm: MinerLLMConfig = field(default_factory=MinerLLMConfig)
    heur: MinerHeuristics = field(default_factory=MinerHeuristics)
    max_md_chars_per_page: int = 5000       # keep prompts small
    output_root: str = "data/mined"         # where we save mined jsonl
    verbose: bool = False                   # optional debug prints

# ---- IDs ----
def _hid(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

# ---- Heuristics: formulas, notations, tables from MD ----
_LATEX_INLINE = re.compile(r"\$(?P<x>[^$]{3,400}?)\$")
_LATEX_BLOCK  = re.compile(r"\\\[(?P<x>.{3,1200}?)\\\]", re.DOTALL)
_OPS = set(list("=+-/*^%≥≤≠≈∝()[]{}|"))

def _extract_latex(md: str) -> List[str]:
    out: List[str] = []
    for m in _LATEX_INLINE.finditer(md):
        out.append(m.group("x").strip())
    for m in _LATEX_BLOCK.finditer(md):
        out.append(m.group("x").strip())
    # de-dup preserve order
    uniq: List[str] = []
    seen = set()
    for x in out:
        if len(x) >= 3 and x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def _candidate_formula_lines(md: str, heur: MinerHeuristics) -> List[str]:
    out = []
    for raw in md.splitlines():
        s = raw.strip()
        if not (heur.min_line_len <= len(s) <= heur.max_line_len):
            continue
        op_count = sum(1 for ch in s if ch in _OPS)
        if op_count >= heur.min_math_symbols and any(ch.isdigit() for ch in s):
            out.append(s)
    return out

def _mine_notations(md: str, heur: MinerHeuristics) -> List[Dict[str, Any]]:
    """
    Grab lines like:
      RVU — Relative Value Unit (unit)
      RVU: Relative Value Unit
    We also give a mild boost when near headings containing "definitions/notation".
    """
    res = []
    lines = md.splitlines()
    bonus_zone = False

    for raw in lines:
        line = raw.strip()
        lower = line.lower()

        # heading-ish cue toggles the bonus zone
        if any(k in lower for k in heur.notation_headings) and (line.startswith("#") or line.endswith(":")):
            bonus_zone = True
            continue

        # Split on em dash, en dash, hyphen, or colon (hyphen escaped!)
        # e.g., "ABC — Some expansion", "ABC: Some expansion", "ABC - Some expansion"
        if 3 < len(line) < 300 and (":" in line or "—" in line or "–" in line or "-" in line):
            parts = re.split(r"\s*[—–\-:]\s*", line, maxsplit=1)
            if len(parts) == 2:
                left, right = parts
                key = left.strip()
                val = right.strip()
                if 1 <= len(key) <= 24 and len(val) >= 3 and (key.isupper() or key[:1].isalpha()):
                    unit = None
                    m = re.search(r"\(([^()]{1,15})\)$", val)
                    if m:
                        unit = m.group(1)
                    res.append({"symbol": key, "meaning": val, "unit": unit, "bonus": int(bonus_zone)})

        # reset bonus when blank line
        if line == "":
            bonus_zone = False

    # prefer unique symbols, keep highest bonus instance
    by_sym: Dict[str, Dict[str, Any]] = {}
    for r in res:
        s = r["symbol"]
        if s not in by_sym or r["bonus"] > by_sym[s].get("bonus", 0):
            by_sym[s] = r
    out = list(by_sym.values())
    for r in out:
        r.pop("bonus", None)
    return out

def _parse_pipe_tables(md: str) -> List[Dict[str, Any]]:
    """
    Parse simple Markdown pipe tables:
      | A | B |
      |---|---|
      | 1 | 2 |
    """
    blocks: List[List[str]] = []
    buf: List[str] = []
    for ln in md.splitlines():
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            buf.append(ln.rstrip())
        else:
            if buf:
                blocks.append(buf); buf = []
    if buf:
        blocks.append(buf)

    tables = []
    for blk in blocks:
        if len(blk) < 2:  # need header + sep
            continue
        header = [c.strip() for c in blk[0].strip("|").split("|")]
        sep    = [c.strip() for c in blk[1].strip("|").split("|")]
        if not any(set(c) >= {"-"} for c in sep):
            continue
        rows = []
        for row in blk[2:]:
            cells = [c.strip() for c in row.strip("|").split("|")]
            if len(cells) != len(header):
                continue
            rows.append(cells)
        if rows:
            tables.append({"title": None, "headers": header, "rows": rows})
    return tables

def _kv_table_like(md: str) -> List[Dict[str, Any]]:
    """
    Parse bullet ‘key: value’ or ‘key — value’ lists as 2-col tables.
    """
    rows = []
    for ln in md.splitlines():
        s = ln.strip()
        if not s.startswith(("-", "*")):
            continue
        s = s.lstrip("-* ").strip()
        if ":" in s or "—" in s or "–" in s or "-" in s:
            parts = re.split(r"\s*[—–\-:]\s*", s, maxsplit=1)
            if len(parts) == 2:
                key, val = parts
                if 1 <= len(key.strip()) <= 64 and len(val.strip()) >= 1:
                    rows.append([key.strip(), val.strip()])
    return [{"title": None, "headers": ["Key", "Value"], "rows": rows}] if rows else []

# ---- Abbreviations (extra)
# 'ABC (Some Expansion)' OR 'Some Expansion (ABC)'
_PAREN_ABBR = re.compile(r"\b([A-Z][A-Z0-9]{1,8})\s*\(([^)]+)\)")
_PAREN_ABBR_REV = re.compile(r"([A-Za-z][^()]{2,80})\s*\(([A-Z][A-Z0-9]{1,8})\)")

def _mine_abbreviations(md: str) -> List[Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {}
    for m in _PAREN_ABBR.finditer(md):
        abbr, exp = m.group(1).strip(), m.group(2).strip()
        if abbr not in found:
            found[abbr] = {"abbr": abbr, "expansion": exp}
    for m in _PAREN_ABBR_REV.finditer(md):
        exp, abbr = m.group(1).strip(), m.group(2).strip()
        if abbr not in found:
            found[abbr] = {"abbr": abbr, "expansion": exp}
    return list(found.values())

# ---- LLM normalization ----
def _mk_llm(cfg: MinerLLMConfig):
    if not _HAS_OLLAMA:
        return None
    oc = OllamaConfig(
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        connect_timeout=cfg.connect_timeout,
        read_timeout=cfg.read_timeout,
        warmup=True,
        warmup_prompt="OK",
        retries=max(0, int(cfg.retries)),
    )
    return OllamaLLM(oc)

_JSON_SNIP = re.compile(r"\{.*\}", re.DOTALL)

def _llm_structurize(llm, page_text: str, cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask LLM to return normalized JSON for {formulas, notations, abbreviations, tables}.
    We pass the page text AND the heuristic candidates to reduce hallucinations.
    """
    sys = (
        "You are a precise information extractor. "
        "From the given Markdown page, extract ONLY content that exists verbatim."
    )
    user = f"""
Return a single JSON object with keys: "formulas", "notations", "abbreviations", "tables".
- "formulas": list of objects: {{"latex": "...", "text": "...", "variables":[{{"name":"", "desc":"", "unit":null}}]}}
  * If LaTeX ($...$ or \\[...\\]) exists, prefer it and set "latex". If not, leave "latex" null and use "text".
- "notations": list of {{"symbol":"", "meaning":"", "unit": null}}
- "abbreviations": list of {{"abbr":"", "expansion":""}}
- "tables": list of {{"title": "", "headers": [...], "rows":[[...], ...]}}
Guidelines:
- Do not invent content. Only include items that clearly appear in the text.
- Keep tables small if they are extremely large (first 10 rows is fine).
- If nothing is found for a key, return an empty list for that key.

PAGE_MARKDOWN (truncated):
{page_text[:3500]}
 
HEURISTIC_CANDIDATES:
{json.dumps(cand, ensure_ascii=False)[:2000]}

Respond with ONLY JSON. No preface or code fences.
""".strip()

    prompt = f"{sys}\n\n{user}".strip()
    out = llm.generate(prompt)
    # Parse JSON object from output
    try:
        return json.loads(out)
    except Exception:
        m = _JSON_SNIP.search(out or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # fallback empty
    return {"formulas": [], "notations": [], "abbreviations": [], "tables": []}

# ---- Public API ----
def mine_page_from_md(
    doc_id: str,
    md_text: str,
    page: Optional[int] = None,
    heading_path: Optional[List[str]] = None,
    cfg: MinerConfig = MinerConfig(),
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract structures from a single page's MD text.
    Uses heuristics first; if cfg.use_llm, normalizes with LLM.
    """
    md = (md_text or "")[: cfg.max_md_chars_per_page]
    heur = cfg.heur

    # Heuristic candidates
    cand = {
        "latex": _extract_latex(md),
        "formula_lines": _candidate_formula_lines(md, heur),
        "notations": _mine_notations(md, heur),
        "abbreviations": _mine_abbreviations(md),
        "pipe_tables": _parse_pipe_tables(md) if heur.parse_pipe_tables else [],
        "kv_tables": _kv_table_like(md),
    }

    if cfg.use_llm and _HAS_OLLAMA:
        llm = _mk_llm(cfg.llm)
        if llm:
            try:
                llm.ensure_ready()
            except Exception:
                llm = None
        if llm:
            res = _llm_structurize(llm, md, cand)
        else:
            res = {
                "formulas": [],
                "notations": cand["notations"],
                "abbreviations": cand["abbreviations"],
                "tables": cand["pipe_tables"] + cand["kv_tables"],
            }
    else:
        # Pure-heuristic fallback
        # We surface latex + formula lines as formulas with minimal structure.
        formulas = []
        for t in cand["latex"]:
            formulas.append({"latex": t, "text": None, "variables": []})
        for t in cand["formula_lines"]:
            formulas.append({"latex": None, "text": t, "variables": []})
        res = {
            "formulas": formulas,
            "notations": cand["notations"],
            "abbreviations": cand["abbreviations"],
            "tables": cand["pipe_tables"] + cand["kv_tables"],
        }

    # decorate with basic provenance
    for k in ("formulas","notations","abbreviations","tables"):
        for x in res.get(k, []):
            (x.setdefault("provenance", {}))["doc_id"] = doc_id
            if page is not None: x["provenance"]["page"] = page
            if heading_path: x["provenance"]["heading_path"] = heading_path
    return res

# --- robust page detection ---
def _detect_page(meta: Dict[str, Any] | None, rid: Optional[str] = None) -> Optional[int]:
    """
    Try many places & patterns to recover a page number from metadata or id.
    Prefers page_start/page_end if provided; returns page_start.
    """
    def _first_int(val) -> Optional[int]:
        if val is None:
            return None
        m = re.search(r"\d+", str(val))
        return int(m.group(0)) if m else None

    m = meta or {}
    if isinstance(m, dict):
        # direct, common keys
        for k in (
            "page","page_num","page_no","page_index","page_idx","page_number",
            "pg","p","source_page","page_start","page_end","pageStart","pageEnd"
        ):
            v = m.get(k)
            vi = _first_int(v)
            if vi is not None:
                # If both start & end exist, we prefer start
                if k in ("page_end","pageEnd") and "page_start" in m:
                    vs = _first_int(m.get("page_start") or m.get("pageStart"))
                    return vs if vs is not None else vi
                return vi

        # nested dicts / strings
        for k in ("loc","span","provenance","source"):
            sub = m.get(k)
            if isinstance(sub, dict):
                vi = _detect_page(sub, rid)
                if vi is not None:
                    return vi
            elif isinstance(sub, str):
                vi = _first_int(sub)
                if vi is not None:
                    return vi

        # look for '#page=12'
        for v in m.values():
            if isinstance(v, str):
                mm = re.search(r"(?:#page=|page\s*[:=]?\s*)(\d{1,5})", v, flags=re.I)
                if mm:
                    return int(mm.group(1))

    # try the id
    if rid:
        mm = re.search(r"(?:^|[_\-])p(\d{1,5})(?:[_\-]|$)", rid, flags=re.I)
        if mm:
            return int(mm.group(1))
        vi = _first_int(rid)
        if vi is not None:
            return vi
    return None

def mine_from_chunks(
    doc_id: str,
    artifacts_root: Path = Path("data/artifacts"),
    chunks_path: Optional[Path] = None,
    pages: Optional[List[int]] = None,
    cfg: MinerConfig = MinerConfig(),
) -> Dict[str, Path]:
    """
    Group chunks by page -> concat page text -> mine page-wise.
    Saves JSONL per type under data/mined/{doc_id}/.
    Robust page detection across schemas.
    """
    # Accept both layouts:
    #   A) data/artifacts/chunks/<doc>.chunks.jsonl
    #   B) data/artifacts/<doc>/chunks/<doc>.chunks.jsonl
    if not chunks_path:
        candA = artifacts_root / "chunks" / f"{doc_id}.chunks.jsonl"
        candB = artifacts_root / doc_id / "chunks" / f"{doc_id}.chunks.jsonl"
        chunks_path = candA if candA.exists() else candB

    rows_iter = _read_jsonl(chunks_path)

    # group by page
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    total = 0
    for r in rows_iter:
        total += 1
        meta = r.get("metadata") or r.get("meta") or {}
        page = _detect_page(meta, r.get("id"))
        if page is None:
            page = -1
        if pages and page not in pages:
            continue
        by_page.setdefault(page, []).append(r)

    if cfg.verbose:
        present = sorted(by_page.keys())
        print(f"[miner] doc={doc_id} total_rows={total} pages_in_scope={present[:30]}... total_pages={len(present)}")

    out_dir = Path(cfg.output_root) / doc_id
    paths = {
        "formulas": out_dir / "formulas.jsonl",
        "notations": out_dir / "notations.jsonl",
        "abbreviations": out_dir / "abbreviations.jsonl",
        "tables": out_dir / "tables.jsonl",
    }
    agg = {k: [] for k in paths.keys()}

    # process page by page
    for p, items in sorted(by_page.items(), key=lambda t: t[0]):
        # pick text from multiple possible fields
        def _get_text(it: Dict[str, Any]) -> str:
            return (it.get("text") or it.get("document") or it.get("content") or "").strip()

        # stable-ish ordering
        items_sorted = sorted(items, key=lambda x: (
            (x.get("metadata") or {}).get("block_type",""),
            x.get("id","")
        ))
        page_text = "\n".join(t for t in (_get_text(it) for it in items_sorted) if t)

        # heading path (any)
        hp = None
        for it in items_sorted:
            hp = (it.get("metadata") or {}).get("heading_path")
            if hp: break

        mined = mine_page_from_md(doc_id, page_text, page=p if p >= 0 else None, heading_path=hp, cfg=cfg)

        for k in agg.keys():
            for x in mined.get(k, []):
                # assign stable id
                base = x.get("latex") or x.get("text") or x.get("title") or json.dumps(x, sort_keys=True)[:80]
                x["id"] = f"{k[:3]}-{(p if p is not None else 0)}-{_hid(base)}"
                agg[k].append(x)

    # save
    for k, path in paths.items():
        _write_jsonl(path, agg[k])

    return paths
