# retriever, citation builder, calculators, table_explain
# packages/chat/tools.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json
import requests
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from packages.providers.vector.chromadb import ChromaVectorStore
from packages.retriever.embedder import LocalEmbedder

# Optional cross-encoder reranker
_HAS_ST = False
try:
    from sentence_transformers import CrossEncoder
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Config helpers
try:
    from packages.core_config.config import load_yaml  # unified loader
except Exception:
    import yaml
    from pathlib import Path
    def load_yaml(*paths):
        merged = {}
        for p in paths:
            if not p:
                continue
            p = Path(p)
            if p.exists():
                y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                merged.update(y)
        class _Cfg:
            def __init__(self, d): self.data = d
            def get(self, path, default=None):
                cur = self.data
                for part in path.split("."):
                    if not isinstance(cur, dict) or part not in cur:
                        return default
                    cur = cur[part]
                return cur
        return _Cfg(merged)


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
            "heading_path": m.get("heading_path") or m.get("heading"),
            "table_id": m.get("table_id"),
            "score": s.get("score", 0.0),
        })
    return out


# ---------- LangChain tools (config-driven; future API-ready) ----------

def make_retriever_tool(retriever: Retriever, default_top_k: Optional[int] = None) -> StructuredTool:
    y = load_yaml("configs/providers.yaml")
    if default_top_k is None:
        default_top_k = int(y.get("chat.return_top_k", 6) or 6)

    class SearchInput(BaseModel):
        question: str = Field(..., description="User question to search in the document.")
        top_k: int = Field(default_top_k, ge=1, le=50, description="How many snippets to return.")

    def _run(question: str, top_k: int = default_top_k) -> Dict[str, Any]:
        hits = retriever.search(question)[: top_k]
        return {
            "question": question,
            "snippets": [
                {"id": h["id"], "text": h["text"], "metadata": h.get("metadata", {}), "score": h.get("score", 0.0)}
                for h in hits
            ],
            "citations": build_citations(hits),
        }

    async def _arun(question: str, top_k: int = default_top_k) -> Dict[str, Any]:
        return _run(question, top_k=top_k)

    return StructuredTool.from_function(
        name="retrieve_context",
        description="Search the mounted document and return JSON snippets with citations.",
        func=_run,
        coroutine=_arun,
        args_schema=SearchInput,
        return_direct=False,
    )


def make_calc_tool(doc_id: str) -> StructuredTool:
    """
    Calculator placeholder with clean switch:
      chat.tools.calculators.provider: "disabled" | "local" | "api"
      chat.tools.calculators.api.base_url, .api.api_key
    """
    y = load_yaml("configs/providers.yaml")
    provider = (y.get("chat.tools.calculators.provider", "disabled") or "disabled").lower()

    class _Args(BaseModel):
        name: str = Field(..., description="Calculator id (from registry or API).")
        inputs_json: str = Field("{}", description="JSON object with inputs.")

    if provider == "local":
        try:
            from packages.calculators import registry as calc_registry
            from packages.calculators import executor as calc_executor
            _HAS_CALC = True
        except Exception:
            _HAS_CALC = False

        if not _HAS_CALC:
            def _run(name: str, inputs_json: str = "{}") -> str:
                return json.dumps({"calculator_id": name, "error": "local_calculators_not_available"})
            return StructuredTool.from_function(
                name="run_calculator",
                description="Execute a document-specific calculator with JSON inputs (local registry).",
                func=_run,
                args_schema=_Args,
                return_direct=False,
            )

        def _run(name: str, inputs_json: str = "{}") -> str:
            try:
                inputs = json.loads(inputs_json or "{}")
            except Exception:
                return json.dumps({"calculator_id": name, "error": "invalid_inputs_json"})
            try:
                result = calc_executor.execute(doc_id, name, **inputs)
                return json.dumps({"calculator_id": name, "inputs": inputs, "result": result})
            except Exception as e:
                return json.dumps({"calculator_id": name, "inputs": inputs, "error": str(e)})

        return StructuredTool.from_function(
            name="run_calculator",
            description="Execute a document-specific calculator with JSON inputs (local registry).",
            func=_run,
            args_schema=_Args,
            return_direct=False,
        )

    elif provider == "api":
        api_base = y.get("chat.tools.calculators.api.base_url")
        api_key  = y.get("chat.tools.calculators.api.api_key")
        timeout  = int(y.get("chat.tools.calculators.api.timeout_sec", 20) or 20)

        def _run(name: str, inputs_json: str = "{}") -> str:
            if not api_base or not api_key:
                return json.dumps({"calculator_id": name, "error": "api_not_configured"})
            try:
                payload = {"doc_id": doc_id, "name": name, "inputs": json.loads(inputs_json or "{}")}
            except Exception:
                return json.dumps({"calculator_id": name, "error": "invalid_inputs_json"})
            try:
                r = requests.post(
                    f"{api_base.rstrip('/')}/calc/run",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                if r.status_code != 200:
                    return json.dumps({"calculator_id": name, "error": f"http_{r.status_code}", "detail": r.text[:300]})
                return r.text
            except Exception as e:
                return json.dumps({"calculator_id": name, "error": f"request_failed:{type(e).__name__}", "detail": str(e)})

        return StructuredTool.from_function(
            name="run_calculator",
            description="Execute a document-specific calculator with JSON inputs (remote API).",
            func=_run,
            args_schema=_Args,
            return_direct=False,
        )

    else:  # disabled
        def _run(name: str, inputs_json: str = "{}") -> str:
            return json.dumps({"calculator_id": name, "error": "calculators_disabled"})
        return StructuredTool.from_function(
            name="run_calculator",
            description="Disabled calculator tool (placeholder).",
            func=_run,
            args_schema=_Args,
            return_direct=False,
        )


def make_toolbox(doc_id: str, retriever: Retriever, default_top_k: Optional[int] = None) -> List[StructuredTool]:
    tools = [make_retriever_tool(retriever, default_top_k=default_top_k)]
    tools.append(make_calc_tool(doc_id))
    # ready for future: table/abbrev tools once assets exist
    return tools
