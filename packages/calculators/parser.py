# Build calculators from LaTeX/text (SymPy+Pint)
# packages/calculators/parser.py
from __future__ import annotations
import re, json, hashlib, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional SymPy Latex parsing (nice-to-have)
_HAS_PARSE_LATEX = False
try:
    from sympy.parsing.latex import parse_latex  # needs sympy + antlr extras
    _HAS_PARSE_LATEX = True
except Exception:
    _HAS_PARSE_LATEX = False

# ---------- IO helpers ----------
def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    rows = []
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

# ---------- Spec model ----------
@dataclass
class ToolVar:
    name: str
    desc: str = ""
    unit: Optional[str] = None

@dataclass
class ToolSpec:
    id: str
    name: str
    doc_id: str
    version: int = 1
    expression: str = ""      # sympy-friendly Python expression string (e.g., "F = m*a")
    target: Optional[str] = None  # variable to solve for; if None, evaluate expression verbatim
    inputs: List[ToolVar] = field(default_factory=list)
    notes: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["inputs"] = [asdict(v) for v in self.inputs]
        return d

# ---------- Helpers ----------
_idclean = re.compile(r"[^a-zA-Z0-9_]+")

def _slug(s: str, limit: int = 40) -> str:
    s = _idclean.sub("_", s.strip()).strip("_").lower()
    return s[:limit] or "calc"

def _hid(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

_EQ_SPLIT = re.compile(r"\s*=\s*")

def _simple_text_to_expr(text: str) -> Optional[str]:
    """
    Very small 'equation-like' parser for lines such as:
      'y = m*x + b'
      'power = voltage * current'
    Returns a normalized 'lhs = rhs' string or None.
    """
    if not text or "=" not in text:
        return None
    parts = _EQ_SPLIT.split(text, maxsplit=1)
    if len(parts) != 2:
        return None
    lhs, rhs = parts[0].strip(), parts[1].strip()
    if not lhs or not rhs:
        return None
    # normalize operators and spaces a bit
    rhs = rhs.replace("–", "-").replace("—", "-")
    rhs = re.sub(r"\s+", " ", rhs)
    lhs = _slug(lhs, 24)
    return f"{lhs} = {rhs}"

def _latex_to_expr(latex: str) -> Optional[str]:
    """
    If sympy latex parser is available, convert to a Pythonic 'lhs = rhs' string.
    Otherwise returns None and caller can fallback to heuristics.
    """
    if not latex:
        return None
    if not _HAS_PARSE_LATEX:
        return None
    try:
        # parse LaTeX; if it's an equation, sympy gives an Eq() or expression
        e = parse_latex(latex)
        from sympy import Eq
        if isinstance(e, Eq):
            lhs, rhs = str(e.lhs), str(e.rhs)
            return f"{lhs} = {rhs}"
        else:
            # treat as implicit "expr" (no lhs)
            return f"result = {str(e)}"
    except Exception:
        return None

def _collect_variables_from_text(expr: str) -> List[str]:
    """
    Simple variable name collector: extract tokens that look like identifiers.
    """
    toks = re.findall(r"[A-Za-z_]{1}[A-Za-z0-9_]*", expr or "")
    # remove common function names/constants
    blacklist = {"sin","cos","tan","log","ln","exp","sqrt","pi","E","I","result"}
    out = []
    seen = set()
    for t in toks:
        if t in blacklist:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ---------- Public: build specs from mined JSONLs ----------
def build_specs_from_mined(
    doc_id: str,
    mined_dir: Path = Path("data/mined"),
    prefer_latex: bool = True,
    max_specs: Optional[int] = None,
) -> List[ToolSpec]:
    """
    Pulls formulas + notations + abbreviations and returns a list of ToolSpec.
    Each formula becomes a candidate tool. Variables get optional descriptions from notations/abbreviations.
    """
    mdir = Path(mined_dir) / doc_id
    formulas = _read_jsonl(mdir / "formulas.jsonl")
    notations = _read_jsonl(mdir / "notations.jsonl")
    abbrev = _read_jsonl(mdir / "abbreviations.jsonl")

    # Index notations & abbreviations for variable mapping
    name2desc: Dict[str, Dict[str, str]] = {}
    for n in notations:
        sym = str(n.get("symbol","")).strip()
        if not sym:
            continue
        name2desc.setdefault(sym, {})
        name2desc[sym]["desc"] = n.get("meaning","")
        if n.get("unit"):
            name2desc[sym]["unit"] = n.get("unit")

    for a in abbrev:
        ab = str(a.get("abbr","")).strip()
        if not ab:
            continue
        name2desc.setdefault(ab, {})
        name2desc[ab]["desc"] = a.get("expansion","")

    specs: List[ToolSpec] = []

    for f in formulas:
        latex = f.get("latex")
        text  = f.get("text")
        prov  = f.get("provenance") or {}
        # choose expression text
        expr = None
        if prefer_latex and latex:
            expr = _latex_to_expr(latex)
        if not expr and text:
            expr = _simple_text_to_expr(text)
        if not expr:
            # nothing usable
            continue

        # figure out target (lhs) if any
        lhs = None
        if "=" in expr:
            lhs = expr.split("=", 1)[0].strip()

        # collect variables
        vars_ = _collect_variables_from_text(expr)
        # If we have a clear lhs, ensure it's included
        if lhs and lhs not in vars_:
            vars_.insert(0, lhs)

        # map variables to descriptions/units if available
        inputs: List[ToolVar] = []
        for v in vars_:
            d = name2desc.get(v, {})
            inputs.append(ToolVar(name=v, desc=d.get("desc",""), unit=d.get("unit")))

        # build id + name
        base = latex or text or expr
        short = (latex or text or expr)[:40]
        name = f"{lhs or 'expr'}_{_slug(short, 24)}"
        spec_id = f"{doc_id}:{_slug(name)}:{_hid(base)}"

        specs.append(
            ToolSpec(
                id=spec_id,
                name=name,
                doc_id=doc_id,
                version=1,
                expression=expr,
                target=lhs,
                inputs=inputs,
                notes="auto-generated from mined formulas",
                provenance=prov,
            )
        )
        if max_specs and len(specs) >= max_specs:
            break

    return specs
