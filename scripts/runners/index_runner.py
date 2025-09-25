# scripts/runners/index_runner.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional

from packages.core_config.config import load_yaml
from packages.retriever.indexer import Indexer, load_chunks

def _dump_jsonl(recs, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def _default_paths(doc_id: str) -> Dict[str, Path]:
    art_root = Path(f"data/artifacts/{doc_id}")
    chunks_path = art_root / "chunks" / f"{doc_id}.chunks.jsonl"
    tables_dir = art_root / "tables"
    return {"art_root": art_root, "chunks_path": chunks_path, "tables_dir": tables_dir}

def _filter_records(recs, min_chars: int, exclude_types: set[str]) -> tuple[list, int]:
    keep = []
    dropped = 0
    for r in recs:
        meta = r.get("metadata") or {}
        btype = meta.get("block_type") or ""
        text  = r.get("text") or ""
        if btype in exclude_types:
            dropped += 1
            continue
        if len(text.strip()) < min_chars:
            dropped += 1
            continue
        keep.append(r)
    return keep, dropped

def run(
    doc_id: Optional[str] = None,
    chunks_path: Optional[Path] = None,
    *,
    append_tables: bool = True,
) -> Dict[str, Any]:
    """
    Build a Chroma index from chunks, with filtering to avoid tiny/heading-only items.
    Reads settings from:
      - configs/providers.yaml
      - configs/pipelines/generic_legal.yaml  (index.*)
    """
    y = load_yaml("configs/providers.yaml", "configs/pipelines/generic_legal.yaml")

    # Resolve doc_id & paths
    if not doc_id and not chunks_path:
        raise ValueError("Provide either doc_id or chunks_path.")
    if not doc_id and chunks_path:
        doc_id = Path(chunks_path).stem.split(".")[0]
    assert doc_id
    paths = _default_paths(doc_id)
    if chunks_path is None:
        chunks_path = paths["chunks_path"]

    # Filtering knobs (safe defaults if missing)
    min_chars = int(y.get("index.filter.min_chars", 220) or 220)
    exclude_block_types = set(y.get("index.filter.exclude_block_types", ["heading", "image"]) or ["heading", "image"])

    # Embedder & vector settings
    embed_model = str(y.get("embedding.model", "BAAI/bge-base-en-v1.5"))
    device      = str(y.get("embedding.device", "cpu"))
    bge_prompt  = bool(y.get("index.bge_use_prompt", True))
    persist     = Path(str(y.get("vector.persist_path", "data/artifacts")))
    vs_max      = int(y.get("index.vector.max_add_batch", 2000) or 2000)

    # Load & filter
    recs = load_chunks(Path(chunks_path))
    recs_filt, dropped = _filter_records(recs, min_chars=min_chars, exclude_types=exclude_block_types)

    # Save a filtered JSONL (for reproducibility/audits)
    filt_path = Path(paths["art_root"]) / "chunks" / f"{doc_id}.chunks.filtered.jsonl"
    n_filt = _dump_jsonl(recs_filt, filt_path)

    # Build index
    indexer = Indexer(
        vector_provider="chroma",
        persist_path=persist,
        embed_model_or_path=embed_model,
        device=device,
        bge_use_prompt=bge_prompt,
        vs_max_add_batch=vs_max,
    )
    collection = indexer.build(chunks_path=filt_path, collection_name=doc_id, reset=True)

    # Optionally append table rows (if any)
    rows_added = 0
    if append_tables and paths["tables_dir"].exists():
        rows_added = indexer.append_table_rows(doc_id=doc_id, tables_dir=paths["tables_dir"], collection_name=collection)

    return {
        "doc_id": doc_id,
        "artifacts_root": str(paths["art_root"]),
        "chunks_in": len(recs),
        "filtered_out": dropped,
        "indexed_chunks": n_filt,
        "collection": collection,
        "tables_rows_added": rows_added,
        "filtered_chunks_path": str(filt_path),
    }

if __name__ == "__main__":
    import sys
    from pathlib import Path
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if p and p.exists():
        print(run(chunks_path=p))
    else:
        raise SystemExit("Usage: python -m scripts.runners.index_runner data/artifacts/<doc>/<doc>.chunks.jsonl")
