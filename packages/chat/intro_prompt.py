# packages/chat/intro_prompt.py
from __future__ import annotations
from pathlib import Path
from typing import List
import json

def build_intro_prompt(doc_id: str,
                       artifacts_root: Path = Path("data/artifacts"),
                       mined_root: Path = Path("data/mined"),
                       max_headings: int = 30) -> str:
    chunks_path = artifacts_root / doc_id / "chunks" / f"{doc_id}.chunks.jsonl"
    heads: List[str] = []
    if chunks_path.exists():
        for i, ln in enumerate(chunks_path.open("r", encoding="utf-8")):
            if i > 2000: break
            try:
                r = json.loads(ln)
            except Exception: 
                continue
            md = r.get("metadata") or {}
            if (md.get("block_type") == "heading") and md.get("heading_path"):
                hp = md.get("heading_path") or []
                txt = " > ".join(hp[-3:])
                if txt and txt not in heads:
                    heads.append(txt)
            if len(heads) >= max_headings:
                break

    abbr_lines = []
    abbr_path = mined_root / doc_id / "abbreviations.jsonl"
    if abbr_path.exists():
        for i, ln in enumerate(abbr_path.open("r", encoding="utf-8")):
            if i >= 40: break
            try:
                a = json.loads(ln)
                abbr_lines.append(f"- {(a.get('abbr') or a.get('symbol',''))}: {a.get('expansion') or a.get('meaning','')}")
            except Exception:
                pass

    parts = [
        f"DOCUMENT ID: {doc_id}",
        "Key sections (partial):",
        *(f"- {h}" for h in heads),
        "",
        "Abbreviations (sample):",
        *abbr_lines,
        "",
        "Scope rules:",
        "- Only answer using this document’s content.",
        "- If the question is out-of-scope, say so and suggest the correct document.",
    ]
    return "\n".join(parts)
