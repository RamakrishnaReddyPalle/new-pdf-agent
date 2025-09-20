from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json, re

@dataclass
class NotationConfig:
    use_llm_adjudication: bool = False  # reserved for later

# “ABR — meaning” or “meaning (ABR)”
_ABBR_INLINE = re.compile(r"\b([A-Z]{2,})\b(?:\s*[-–—:]\s*|\s*\(([^)]+)\))")
# “term — definition”
_DEF_LINE    = re.compile(r"^\s*([A-Za-z0-9\-/()% ]{2,40})\s*[-–—:]\s+(.+)$")

def extract_notations(pages_jsonl: Path, out_dir: Path, cfg: NotationConfig) -> int:
    out_dir = out_dir / "notations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "abbrev_defs.jsonl"
    n = 0
    with pages_jsonl.open("r", encoding="utf-8") as f_in, out_fp.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            page = int(rec.get("page") or 0)
            text = rec.get("text") or ""
            for ln in text.splitlines():
                s = ln.strip()
                m1 = _ABBR_INLINE.search(s)
                m2 = _DEF_LINE.match(s)
                if m1:
                    abbr = m1.group(1); long = (m1.group(2) or "").strip()
                    f_out.write(json.dumps({"page": page, "abbr": abbr, "long": long, "line": s}) + "\n"); n+=1
                elif m2:
                    term = m2.group(1).strip(); desc = m2.group(2).strip()
                    f_out.write(json.dumps({"page": page, "term": term, "def": desc, "line": s}) + "\n"); n+=1
    return n
