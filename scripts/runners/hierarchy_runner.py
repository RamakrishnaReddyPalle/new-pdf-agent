# scripts/runners/hierarchy_runner.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

from packages.core_config.config import load_yaml
from packages.ingest.hierarchy import HierarchyConfig, build_hierarchy

def _paths(doc_id: str):
    art_root   = Path(f"data/artifacts/{doc_id}")
    chunks_dir = art_root / "chunks"
    out_dir    = art_root
    return art_root, chunks_dir, out_dir

def run(doc_id: str) -> Dict[str, Any]:
    """
    Build hierarchy graph (graph/hierarchy.json) from existing chunks.
    """
    y = load_yaml("configs/pipelines/generic_legal.yaml")

    art_root, chunks_dir, out_dir = _paths(doc_id)
    cfg = HierarchyConfig(
        min_node_chars=int(y.get("hierarchy.min_node_chars", 400) or 400),
        use_section_regex=bool(y.get("hierarchy.use_section_regex", True)),
        fold_tiny_into_misc=bool(y.get("hierarchy.fold_tiny_into_misc", True)),
    )
    out = build_hierarchy(chunks_dir=chunks_dir, out_dir=out_dir, cfg=cfg)
    graph_path = out_dir / "graph" / "hierarchy.json"
    return {
        "doc_id": doc_id,
        "graph_path": str(graph_path),
        "exists": graph_path.exists(),
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m scripts.runners.hierarchy_runner <DOC_ID>")
    print(run(sys.argv[1]))
