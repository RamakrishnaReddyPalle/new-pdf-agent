# Plugin registry
# packages/calculators/registry.py
from __future__ import annotations
import json, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parser import ToolSpec

REGISTRY_ROOT = Path("data/calculators")

def _doc_dir(doc_id: str) -> Path:
    p = REGISTRY_ROOT / doc_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _reg_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "tools.jsonl"

def register(spec: ToolSpec) -> Dict[str, Any]:
    row = spec.to_dict()
    row["_ts"] = int(time.time())
    p = _reg_path(spec.doc_id)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row

def list_tools(doc_id: str) -> List[Dict[str, Any]]:
    p = _reg_path(doc_id)
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
    # return latest per id
    latest = {}
    for r in rows:
        rid = r.get("id")
        if not rid:
            continue
        if rid not in latest or r.get("_ts", 0) > latest[rid].get("_ts", -1):
            latest[rid] = r
    return list(latest.values())

def get(doc_id: str, tool_id: str) -> Optional[Dict[str, Any]]:
    for r in list_tools(doc_id):
        if r.get("id") == tool_id:
            return r
    return None
