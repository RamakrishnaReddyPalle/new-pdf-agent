# packages/providers/vector/chromadb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

from chromadb import PersistentClient


def _chunks(n: int, size: int):
    """Yield (start, end) indices for chunking a list of length n by size."""
    for i in range(0, n, size):
        yield i, min(i + size, n)


@dataclass
class ChromaSettings:
    """Simple settings holder for local Chroma usage."""
    persist_path: str | Path = "data/index"
    # Keep this below Chroma's internal cap (~5461) to avoid InternalError.
    max_add_batch: int = 2000


class ChromaVectorStore:
    """Minimal Chroma wrapper with safe, chunked .add(), plus .query for retrieval."""

    def __init__(self, persist_path: str | Path = "data/index", max_add_batch: int = 2000):
        self.client = PersistentClient(path=str(persist_path))
        self.max_add_batch = int(max_add_batch)

    # ---- collection management ----
    def get_or_create(self, name: str):
        return self.client.get_or_create_collection(name)

    def delete_collection(self, name: str):
        """Delete collection if it exists (no-op on failure)."""
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

    # ---- writes ----
    def add(
        self,
        collection_name: str,
        ids: List[str],
        docs: List[str],
        metas: List[Dict[str, Any]],
        embs: List[List[float]],
    ):
        """Add vectors/documents in chunks to avoid Chroma's internal batch cap."""
        coll = self.get_or_create(collection_name)
        n = len(ids)
        b = int(self.max_add_batch)
        for i, j in _chunks(n, b):
            coll.add(
                ids=ids[i:j],
                documents=docs[i:j],
                metadatas=metas[i:j],
                embeddings=embs[i:j],
            )

    # ---- reads ----
    def query(self, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return normalized hits: [{id, document, metadata, score}]
        Converts Chroma distance to a simple similarity score = 1/(1+distance).
        """
        coll = self.get_or_create(collection_name)
        res = coll.query(
            query_embeddings=[query_embedding],
            n_results=int(top_k),
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[Dict[str, Any]] = []
        for _id, _doc, _meta, _dist in zip(ids, docs, metas, dists):
            try:
                score = float(1.0 / (1.0 + float(_dist)))
            except Exception:
                score = 0.0
            out.append(
                {
                    "id": _id,
                    "document": _doc,
                    "metadata": _meta or {},
                    "score": score,
                }
            )
        return out
