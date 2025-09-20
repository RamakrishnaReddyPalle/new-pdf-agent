# retriever, citation builder, calculators, table_explain
# packages/chat/tools.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from packages.providers.vector.chromadb import ChromaVectorStore
from packages.retriever.embedder import LocalEmbedder

# Optional cross-encoder reranker
_HAS_ST = False
try:
    from sentence_transformers import CrossEncoder
    _HAS_ST = True
except Exception:
    _HAS_ST = False


@dataclass
class RetrieverConfig:
    persist_path: str = "data/artifacts"
    collection: Optional[str] = None         # default to doc_id
    embed_model_or_path: str = "BAAI/bge-base-en-v1.5"
    device: str = "cpu"
    bge_use_prompt: bool = True
    top_k: int = 12
    rerank_top_k: int = 8
    return_top_k: int = 6
    # path to trained CE folder OR base id if you want a stock CE model
    reranker_model_path: Optional[str] = None
    reranker_base_fallback: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    """
    Vector search (+ optional CE rerank) returning (text, metadata, score).
    """
    def __init__(self, doc_id: str, cfg: RetrieverConfig):
        self.doc_id = doc_id
        self.cfg = cfg
        self.vs = ChromaVectorStore(cfg.persist_path)
        self.collection = cfg.collection or doc_id
        self.embedder = LocalEmbedder(
            cfg.embed_model_or_path,
            device=cfg.device,
            normalize=True,
            bge_use_prompt=cfg.bge_use_prompt,
        )
        self._ce = None
        if cfg.reranker_model_path and _HAS_ST:
            try:
                self._ce = CrossEncoder(cfg.reranker_model_path, num_labels=1, max_length=512)
            except Exception:
                # fallback to base (downloaded once) if local folder isn’t loadable
                self._ce = CrossEncoder(cfg.reranker_base_fallback, num_labels=1, max_length=512)

    def _format_ce_inputs(self, query: str, cands: List[Dict[str, Any]]) -> List[List[str]]:
        return [[query, it["text"]] for it in cands]

    def search(self, query: str) -> List[Dict[str, Any]]:
        # 1) embed the query
        q_emb = self.embedder.encode_query(query)
        # 2) vector search
        hits = self.vs.query(self.collection, query_embedding=q_emb, top_k=self.cfg.top_k)
        # hits: [{id, document, metadata, score}]
        if not hits:
            return []
        # 3) optional rerank
        if self._ce:
            pairs_in = [{"text": h["document"], "metadata": h.get("metadata", {}), "id": h["id"]} for h in hits[: self.cfg.rerank_top_k]]
            pairs = self._format_ce_inputs(query, pairs_in)
            scores = self._ce.predict(pairs)
            packed = list(zip(hits[: len(scores)], scores))
            packed.sort(key=lambda t: float(t[1]), reverse=True)
            hits = [h for (h, _s) in packed]
        # 4) return top_k
        out = []
        for h in hits[: self.cfg.return_top_k]:
            out.append({
                "id": h["id"],
                "text": h["document"],
                "metadata": h.get("metadata", {}),
                "score": float(h.get("score", 0.0)),
            })
        return out


def build_citations(snips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize minimal citation payload for UI and eval:
    [{id, page, heading_path, table_id, score}]
    """
    out = []
    for s in snips:
        m = s.get("metadata") or {}
        out.append({
            "id": s.get("id"),
            "page": m.get("page"),
            "heading_path": m.get("heading_path"),
            "table_id": m.get("table_id"),
            "score": s.get("score", 0.0),
        })
    return out
