# embed chunks (company embed API)
# packages/retriever/embedder.py
from __future__ import annotations
from typing import List, Dict, Any, Iterable

from sentence_transformers import SentenceTransformer


def _batched(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _maybe_bge_passage_prefix(embed_model_name: str, texts: List[str], use_prompt: bool = True) -> List[str]:
    """
    Preserve original behavior: for BGE v1.5 style models, prefix passages with 'passage: ' when enabled.
    """
    name = (embed_model_name or "").lower()
    if use_prompt and ("bge-" in name and "v1.5" in name):
        return [f"passage: {t}" for t in texts]
    return texts


def _maybe_bge_query_prefix(embed_model_name: str, text: str, use_prompt: bool = True) -> str:
    """
    Complementary query prompt for BGE: 'query: '.
    Added without changing existing passage behavior.
    """
    name = (embed_model_name or "").lower()
    if use_prompt and ("bge-" in name and "v1.5" in name):
        return f"query: {text}"
    return text


def _embed_input_with_headings(doc_text: str, meta: Dict[str, Any], max_levels: int = 3) -> str:
    """
    Keep original heading-path prefixing inside the vector text.
    """
    hp = meta.get("heading_path")
    if isinstance(hp, list) and hp:
        prefix = " > ".join(str(x) for x in hp[-max_levels:])
        return f"{prefix}\n{doc_text}"
    if isinstance(hp, str) and hp.strip():
        parts = [p.strip() for p in hp.split(">") if p.strip()]
        if parts:
            prefix = " > ".join(parts[-max_levels:])
            return f"{prefix}\n{doc_text}"
    return doc_text


class LocalEmbedder:
    """
    Loads a SentenceTransformer model from a local path or HF id.
    Normalizes embeddings by default.
    """
    def __init__(self, model_or_path: str, device: str = "cpu", normalize: bool = True, bge_use_prompt: bool = True):
        self.model_path = model_or_path
        self.model = SentenceTransformer(model_or_path, device=device)
        self.normalize = normalize
        self.bge_use_prompt = bge_use_prompt

    # ---------- NEW (keeps old behavior intact) ----------
    def encode_query(self, query: str, batch_size: int = 64) -> List[float]:
        """
        New method needed by RAG eval & retriever: embed a single query.
        Uses BGE 'query:' prefix when enabled, otherwise unchanged text.
        """
        text = _maybe_bge_query_prefix(self.model_path, query, use_prompt=self.bge_use_prompt)
        emb = self.model.encode([text], show_progress_bar=False, normalize_embeddings=self.normalize).tolist()[0]
        return emb

    def encode_docs(self, docs: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Convenience method if you want to embed raw doc strings (without metadata prefixes).
        Applies BGE 'passage:' prefixing when configured.
        """
        docs_p = _maybe_bge_passage_prefix(self.model_path, docs, use_prompt=self.bge_use_prompt)
        embs = self.model.encode(docs_p, show_progress_bar=False, normalize_embeddings=self.normalize).tolist()
        return embs
    # -----------------------------------------------------

    def encode_chunks(self, records: List[Dict[str, Any]], batch_size: int = 64) -> List[List[float]]:
        """
        Original API used by indexer: embed [{'text':..., 'metadata':...}, ...]
        Preserves heading-path prefixing and BGE 'passage:' behavior.
        """
        out: List[List[float]] = []
        for batch in _batched(records, batch_size):
            texts = [_embed_input_with_headings(r["text"], r["metadata"]) for r in batch]
            texts = _maybe_bge_passage_prefix(self.model_path, texts, use_prompt=self.bge_use_prompt)
            embs = self.model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=self.normalize
            ).tolist()
            out.extend(embs)
        return out
