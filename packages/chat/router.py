# selects doc profiles + LLMs + toolsets
# packages/chat/router.py
from __future__ import annotations
import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from packages.chat.tools import RetrieverConfig, Retriever
from packages.providers.llm.ollama import OllamaLLM, OllamaConfig  # replaceable later
# If you prefer HF base+adapter later, wire providers/llm/hf_infer.py here.

@dataclass
class DocProfile:
    doc_id: str
    collection: str
    reranker_model_path: Optional[str] = None
    base_model_hint: Optional[str] = None  # e.g., TinyLlama dir (optional for chat if using Ollama)


class Router:
    """
    Resolves a doc 'profile' (index + reranker + LLM provider settings).
    - Reads model registry: data/models/models.jsonl
    - Assumes vector index collection == doc_id
    - LLM: Ollama (config-driven), easily swappable later
    """
    def __init__(self, providers_cfg: Dict[str, Any], pipeline_cfg: Dict[str, Any]):
        self.providers_cfg = providers_cfg or {}
        self.pipeline_cfg = pipeline_cfg or {}
        self.models_index = Path("data/models/models.jsonl")

        # LLM settings for chat (can point to company API later)
        llm = (providers_cfg.get("chat") or {}).get("ollama", {})
        self.llm_cfg = OllamaConfig(
            base_url=str(llm.get("base_url", "http://localhost:11434")),
            model=str(llm.get("model", "llama3.2:latest")),
            temperature=float(llm.get("temperature", 0.2)),
            max_new_tokens=int(llm.get("max_new_tokens", 512)),
            connect_timeout=int(llm.get("connect_timeout", 30)),
            read_timeout=int(llm.get("read_timeout", 600)),
            warmup_prompt=str(llm.get("warmup_prompt", "OK")),
            healthcheck=bool(llm.get("healthcheck", True)),
            warmup=bool(llm.get("warmup", False)),
            retries=int(llm.get("retries", 1)),
        )
        self.llm = OllamaLLM(self.llm_cfg)

    def _latest_adapter_for_doc(self, doc_id: str) -> Optional[str]:
        if not self.models_index.exists():
            return None
        latest = None
        for ln in self.models_index.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("doc_id") == doc_id:
                latest = r
        if not latest:
            return None
        # You *may* use adapter later (HF infer). For Ollama chat, adapter is unused.
        return latest.get("run_dir")  # or ['adapter_path'] if you switch to HF infer

    def resolve(self, doc_id: str) -> Dict[str, Any]:
        # Reranker (latest run_dir if exists)
        rk_dir = Path("data/reranker") / doc_id
        reranker_model_path = None
        if rk_dir.exists():
            # pick last run by mtime
            runs = sorted([p for p in rk_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            if runs:
                reranker_model_path = str(runs[0] / "model")

        # Build retriever
        idx_cfg = (self.pipeline_cfg.get("index") or {})
        emb_prompt = bool(idx_cfg.get("bge_use_prompt", True))

        retr_cfg = RetrieverConfig(
            persist_path=str(Path("data/artifacts")),   # where Chroma lives
            collection=doc_id,
            embed_model_or_path=str((self.providers_cfg.get("embedding") or {}).get("model", "BAAI/bge-base-en-v1.5")),
            device=str((self.providers_cfg.get("embedding") or {}).get("device", "cpu")),
            bge_use_prompt=emb_prompt,
            top_k=int((self.pipeline_cfg.get("chat") or {}).get("search_top_k", 12)),
            rerank_top_k=int((self.pipeline_cfg.get("chat") or {}).get("rerank_top_k", 8)),
            return_top_k=int((self.pipeline_cfg.get("chat") or {}).get("return_top_k", 6)),
            reranker_model_path=reranker_model_path,
        )
        retriever = Retriever(doc_id, retr_cfg)

        return {
            "doc_id": doc_id,
            "retriever": retriever,
            "llm": self.llm,
        }
