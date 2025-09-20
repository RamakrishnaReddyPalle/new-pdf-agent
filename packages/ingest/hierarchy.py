# packages/ingest/hierarchy.py
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List

TOC_LINE = re.compile(r"^\s*\|.*\|\s*$")
SECTION_LINE = re.compile(r"^\s*§+\s*([0-9A-Za-z\-]+)[^\n]*", re.UNICODE)

@dataclass
class HierarchyConfig:
    min_node_chars: int = 400
    use_section_regex: bool = True  # keep your § detection
    fold_tiny_into_misc: bool = True

def _clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    lines = []
    for ln in t.splitlines():
        if TOC_LINE.match(ln):
            continue
        if len(ln.strip()) <= 1:
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

def _guess_section(obj: Dict[str, Any], cfg: HierarchyConfig) -> str | None:
    # 1) Prefer explicit field if present (future-proofing)
    sec = obj.get("metadata", {}).get("section") or obj.get("section")
    if isinstance(sec, str) and sec.strip():
        return sec.strip()

    # 2) Use heading_path tail (often more robust than regex)
    hp = obj.get("metadata", {}).get("heading_path")
    if isinstance(hp, list) and hp:
        return " > ".join(str(x) for x in hp[-2:])  # last 2 levels
    if isinstance(hp, str) and hp.strip():
        return hp.strip()

    # 3) Regex (§...) fallback if enabled
    if cfg.use_section_regex:
        txt = obj.get("text") or obj.get("content") or ""
        if isinstance(txt, str):
            for ln in txt.splitlines():
                m = SECTION_LINE.match(ln)
                if m:
                    return ln.strip()
    return None

def _iter_chunks(chunks_dir: Path):
    for fp in sorted(chunks_dir.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                yield obj

def build_hierarchy(chunks_dir: Path, out_dir: Path, cfg: HierarchyConfig) -> None:
    out_dir = out_dir / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all chunks
    chunks: List[Dict[str, Any]] = []
    for obj in _iter_chunks(chunks_dir):
        cid  = obj.get("id") or obj.get("chunk_id")
        text = _clean_text(obj.get("text") or obj.get("content") or "")
        if not text:
            continue
        chunks.append({"id": cid, "text": text, "obj": obj})

    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"chunk_ids": [], "texts": []})
    misc_key = "MISC (no heading detected)"

    for ch in chunks:
        name = _guess_section(ch["obj"], cfg) or misc_key
        groups[name]["chunk_ids"].append(ch["id"])
        groups[name]["texts"].append(ch["text"])

    # Fold tiny nodes
    if cfg.fold_tiny_into_misc:
        reassign: Dict[str, Dict[str, Any]] = {}
        for name, data in list(groups.items()):
            text = "\n\n".join(data["texts"]).strip()
            if len(text) < cfg.min_node_chars and name != misc_key:
                groups[misc_key]["chunk_ids"].extend(data["chunk_ids"])
                groups[misc_key]["texts"].append(text)
                reassign[name] = data
        for k in reassign.keys():
            del groups[k]

    # Build nodes
    nodes = []
    root_children = []
    name_to_text = {name: "\n\n".join(v["texts"]).strip() for name, v in groups.items()}
    for i, (name, data) in enumerate(groups.items(), 1):
        node_id = f"SEC-{i:05d}"
        nodes.append({
            "id": node_id,
            "name": name,
            "level": 1,
            "parent": "ROOT",
            "children": [],
            "chunk_ids": data["chunk_ids"],
        })
        root_children.append(node_id)

    if not nodes and chunks:
        node_id = "SEC-00001"
        nodes = [{
            "id": node_id,
            "name": "ALL",
            "level": 1,
            "parent": "ROOT",
            "children": [],
            "chunk_ids": [c["id"] for c in chunks],
        }]
        root_children = [node_id]
        name_to_text = {"ALL": "\n\n".join([c["text"] for c in chunks]).strip()}

    root = {
        "id": "ROOT",
        "name": "ROOT",
        "level": 0,
        "parent": None,
        "children": root_children,
        "chunk_ids": [],
    }
    hier = {
        "n_nodes": 1 + len(nodes),
        "n_chunks": len(chunks),
        "nodes": [root] + nodes,
    }

    (out_dir / "hierarchy.json").write_text(json.dumps(hier, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "node_texts.jsonl").open("w", encoding="utf-8") as f:
        for n in nodes:
            text = name_to_text.get(n["name"], "")
            f.write(json.dumps({
                "node_id": n["id"],
                "name": n["name"],
                "level": n["level"],
                "text": text
            }, ensure_ascii=False) + "\n")
