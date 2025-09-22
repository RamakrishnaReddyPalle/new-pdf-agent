# packages/chat/models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

from packages.core_config.config import load_yaml

# LangChain chat models
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import PrivateAttr


# ---- Local HF core with (optional) LoRA adapter, wrapped as a LangChain chat model ----
class LocalHFChat(BaseChatModel):
    """
    Minimal LC wrapper around our HF local infer shim.
    Keeps a private engine attribute to satisfy Pydantic v2.
    """
    _engine: Any = PrivateAttr(default=None)

    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str],
        temperature: float,
        max_new_tokens: int,
        device: str = "cpu",
    ):
        super().__init__()
        from packages.providers.llm.hf_infer import LocalHFLLM
        self._engine = LocalHFLLM(
            base_model=base_model,
            adapter_path=adapter_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "local_hf_chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join([m.content for m in messages])
        out = self._engine.generate(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # lightweight async passthrough
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ---- Company API adapter (provider: "company") ----
class CompanyChatAdapter(BaseChatModel):
    """
    Thin LC wrapper around packages.providers.llm.company_api.
    Expects a client with .generate(prompt: str) -> str OR .chat(messages=[...]) -> str.
    """
    _engine: Any = PrivateAttr(default=None)

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_new_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.params = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }
        self.extra = extra or {}
        try:
            from packages.providers.llm.company_api import CompanyChat  # your SDK/shim
            self._engine = CompanyChat(**self.params, **self.extra)
        except Exception:
            # graceful fallback: you can still run locally without company client installed
            self._engine = None

    @property
    def _llm_type(self) -> str:
        return "company_chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join([m.content for m in messages])
        if self._engine is None:
            out = "[company_chat_unavailable] " + prompt[:2000]
        else:
            # prefer chat-style if available
            if hasattr(self._engine, "chat"):
                out = self._engine.chat(messages=[{"role": "user", "content": prompt}])
            else:
                out = self._engine.generate(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(out)))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ---- Registry ----
@dataclass
class ModelSpec:
    provider: str
    params: Dict[str, Any]


class ModelRegistry:
    """
    Central switchboard for Phase-7 chat models + retriever plumbing.
    Reads from configs/providers.yaml (via PipelineConfig).
    Supports provider: "ollama" | "hf_local" | "company".
    Exposes:
      - sessions_dir
      - vector_path, embed_model, embed_device, reranker_model (for retriever/reranker)
      - core_run_id / core_prefer_subdirs for adapter selection
      - chat models: intro, splitter, core, output
    """
    def __init__(self, providers_yaml: str = "configs/providers.yaml"):
        y = load_yaml(providers_yaml)

        # ---- Chat role models
        chat_cfg = y.get("chat", {}) or {}
        self.cfg = chat_cfg  # router uses this for search_top_k knobs, etc.
        cm = chat_cfg.get("models", {}) or {}

        self.intro    = ModelSpec(cm.get("intro", {}).get("provider", "ollama"),    cm.get("intro", {}))
        self.splitter = ModelSpec(cm.get("splitter", {}).get("provider", "ollama"), cm.get("splitter", {}))
        self.core     = ModelSpec(cm.get("core", {}).get("provider", "hf_local"),   cm.get("core", {}))
        self.output   = ModelSpec(cm.get("output", {}).get("provider", "ollama"),   cm.get("output", {}))

        self.sessions_dir = Path(chat_cfg.get("sessions_dir", "data/sessions"))

        # ---- Embedding / Vector defaults (for retriever & eval)
        # Prefer the dedicated 'embedding.*' block; fall back to 'llm.embedding' if present.
        self.embed_model  = y.get("embedding.model", None) or y.get("llm.embedding", "BAAI/bge-base-en-v1.5")
        self.embed_device = y.get("embedding.device", "cpu")
        self.vector_path  = y.get("vector.persist_path", "data/artifacts")

        # Optional reranker pointer (HF id or local)
        self.reranker_model = y.get("llm.reranker", None)

        # Optional search knobs (router also defaults if absent)
        self.search_top_k   = int(chat_cfg.get("search_top_k", 12))
        self.rerank_top_k   = int(chat_cfg.get("rerank_top_k", 8))
        self.return_top_k   = int(chat_cfg.get("return_top_k", 6))

        # ---- Adapter selection hints for hf_local core
        self.core_run_id = (self.core.params or {}).get("run_id")
        self.core_prefer_subdirs = (self.core.params or {}).get("prefer_subdirs", ["adapter", "hf_out"])

    # ---- factories ----
    def _mk_ollama(self, spec: ModelSpec) -> ChatOllama:
        p = spec.params or {}
        return ChatOllama(
            base_url=p.get("base_url", "http://localhost:11434"),
            model=p.get("model", "llama3.2:latest"),
            temperature=float(p.get("temperature", 0.2)),
            num_predict=int(p.get("max_new_tokens", 512)),
        )

    def _mk_hf_local(self, spec: ModelSpec, adapter_path: Optional[str]) -> LocalHFChat:
        p = spec.params or {}
        base = p.get("base_model_local_dir") or p.get("base_model_id")
        if not base:
            raise ValueError("core.base_model_id/base_model_local_dir is required for hf_local provider")
        return LocalHFChat(
            base_model=base,
            adapter_path=adapter_path if p.get("adapter_from_profile", True) else p.get("adapter_path"),
            temperature=float(p.get("temperature", 0.1)),
            max_new_tokens=int(p.get("max_new_tokens", 512)),
            device=p.get("device", "cpu"),
        )

    def _mk_company(self, spec: ModelSpec) -> CompanyChatAdapter:
        p = spec.params or {}
        return CompanyChatAdapter(
            base_url=p.get("base_url", ""),
            api_key=p.get("api_key", ""),
            model=p.get("model", ""),
            temperature=float(p.get("temperature", 0.2)),
            max_new_tokens=int(p.get("max_new_tokens", 512)),
            extra={k: v for k, v in p.items() if k not in {"base_url", "api_key", "model", "temperature", "max_new_tokens"}},
        )

    # ---- public getters ----
    def get_intro(self):    return self._mk_by_spec(self.intro, adapter_path=None)
    def get_splitter(self): return self._mk_by_spec(self.splitter, adapter_path=None)
    def get_output(self):   return self._mk_by_spec(self.output, adapter_path=None)
    def get_core(self, adapter_path: Optional[str]):
        return self._mk_by_spec(self.core, adapter_path=adapter_path)

    # ---- internal dispatch ----
    def _mk_by_spec(self, spec: ModelSpec, adapter_path: Optional[str]):
        prov = (spec.provider or "").lower()
        if prov in ("ollama", "local_ollama"):
            return self._mk_ollama(spec)
        if prov in ("hf_local", "local_hf"):
            return self._mk_hf_local(spec, adapter_path=adapter_path)
        if prov in ("company", "company_chat"):
            return self._mk_company(spec)
        # fallback to ollama for unknowns
        return self._mk_ollama(spec)
