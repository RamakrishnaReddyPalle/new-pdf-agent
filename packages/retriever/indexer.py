# packages/retriever/indexer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from packages.retriever.embedder import LocalEmbedder
from packages.providers.vector.chromadb import ChromaVectorStore  # dev default


def _flatten_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma metadata must be scalars and non-None.
    - Drop None values
    - Join lists (e.g., heading_path) with ' > '
    - JSON-stringify unexpected types
    """
    flat: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, list):
            flat[k] = " > ".join(str(x) for x in v)
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            try:
                flat[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                flat[k] = str(v)
    return flat


def load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL chunks into a list of {id, text, metadata} dicts."""
    recs: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    if not recs:
        raise RuntimeError(f"No chunks found at {chunks_path}")
    return recs


def _load_tables_as_records(tables_dir: Path, doc_id: str) -> List[Dict[str, Any]]:
    """
    Convert saved table JSONs (produced by tables_pipeline) into indexable records.
    Each row becomes a doc with a compact pipe-joined "k=v" string for retrieval.
    """
    recs: List[Dict[str, Any]] = []
    for fp in sorted(tables_dir.glob("table_*_*.json")):
        try:
            page = int(fp.stem.split("_")[1])
        except Exception:
            page = None
        obj = json.loads(fp.read_text(encoding="utf-8"))
        header = obj.get("header") or []
        rows = obj.get("rows") or []
        for i, row in enumerate(rows, 1):
            # Build row_text like "Code=90681 | Description=... | RelativeValue=7.61 | Split=45/55"
            kv = []
            for k in header:
                v = str(row.get(k, "")).strip()
                if v:
                    kv.append(f"{k}={v}")
            row_text = " | ".join(kv)
            meta = {
                "doc_id": doc_id,
                "block_type": "table_row",
                "page": page,
                "table_id": fp.stem,
                "columns": header,
            }
            recs.append(
                {
                    "id": f"{doc_id}-row-{page}-{i}",
                    "text": row_text,
                    "metadata": meta,
                }
            )
    return recs


class Indexer:
    """
    Config-driven index builder. Dev vector provider: Chroma (local).
    Swap provider later by reading providers.yaml without changing call-sites.

    Notes:
    - Embedding is done via LocalEmbedder (SentenceTransformers / HF), which
      already batches internally; keep embed_batch_size here for forward-compat.
    - Chroma `.add()` has a hard max batch; we pass `vs_max_add_batch` down to
      the provider so adds are automatically chunked.
    """

    def __init__(
        self,
        vector_provider: str = "chroma",
        persist_path: str | Path = "data/artifacts",
        embed_model_or_path: str = "BAAI/bge-base-en-v1.5",
        embed_batch_size: int = 64,
        bge_use_prompt: bool = True,
        device: str = "cpu",
        vs_max_add_batch: int = 2000,  # <-- chunked add to avoid Chroma cap (~5461)
    ):
        self.vector_provider = vector_provider
        self.persist_path = Path(persist_path)

        # Embeddings
        self.embedder = LocalEmbedder(
            embed_model_or_path,
            device=device,
            normalize=True,
            bge_use_prompt=bge_use_prompt,
        )

        # Vector store
        if vector_provider == "chroma":
            self.vs = ChromaVectorStore(
                persist_path=self.persist_path,
                max_add_batch=vs_max_add_batch,
            )
        else:
            raise NotImplementedError(
                f"Vector provider '{vector_provider}' not implemented in dev mode."
            )

    def build(
        self,
        chunks_path: Path,
        collection_name: Optional[str] = None,
        reset: bool = False,
    ) -> str:
        """
        Build an index from chunk JSONL:
        - Flattens metadata
        - Embeds texts
        - Adds to vector store (chunked add inside provider)
        """
        recs = load_chunks(chunks_path)
        doc_id = recs[0]["metadata"]["doc_id"]
        name = collection_name or doc_id

        if reset:
            try:
                self.vs.delete_collection(name)
            except Exception:
                pass

        ids: List[str] = [r["id"] for r in recs]
        docs: List[str] = [r["text"] for r in recs]
        metas: List[Dict[str, Any]] = [_flatten_meta(r["metadata"]) for r in recs]

        # Embeddings (LocalEmbedder can batch internally)
        embs: List[List[float]] = self.embedder.encode_chunks(recs)

        assert len(ids) == len(embs) == len(metas) == len(docs)
        self.vs.add(name, ids=ids, docs=docs, metas=metas, embs=embs)
        return name

    def append_table_rows(
        self,
        doc_id: str,
        tables_dir: Path,
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Append table rows (as records) to an existing collection.
        - Each row is embedded and added (provider handles chunking).
        """
        table_recs = _load_tables_as_records(tables_dir, doc_id)
        if not table_recs:
            return 0

        name = collection_name or doc_id
        ids = [r["id"] for r in table_recs]
        docs = [r["text"] for r in table_recs]
        metas = [_flatten_meta(r["metadata"]) for r in table_recs]
        embs = self.embedder.encode_chunks(table_recs)

        self.vs.add(name, ids=ids, docs=docs, metas=metas, embs=embs)
        return len(table_recs)
