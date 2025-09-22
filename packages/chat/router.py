# packages/chat/router.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Callable, Tuple
from pathlib import Path
import json

from pydantic import BaseModel, Field, create_model
from langchain.tools import StructuredTool

from packages.chat.models import ModelRegistry
from packages.chat.tools import Retriever, RetrieverConfig, build_citations

# Optional calculator registry/executor (Phase 6). We keep this optional.
_HAS_CALC = False
try:
    from packages.calculators import registry as calc_registry
    from packages.calculators import executor as calc_executor
    _HAS_CALC = True
except Exception:
    _HAS_CALC = False


# ---------------------------
# Document profile (mounted per doc)
# ---------------------------
@dataclass
class DocProfile:
    doc_id: str
    collection: str
    reranker_model_path: Optional[str]
    adapter_path: Optional[str]
    calculators_root: Optional[str] = None
    adapter_run_id: Optional[str] = None   # which run got picked (if any)


def _discover_runs(models_root: Path, doc_id: str) -> List[Dict[str, Optional[str]]]:
    """
    Return a list of runs with potential adapter/hf_out paths and latest mtime.
    Assumes structure: data/models/<doc_id>/<run_id>/(adapter|hf_out)/...
    """
    base = models_root / doc_id
    if not base.exists():
        return []

    runs: Dict[str, Dict[str, Optional[str]]] = {}
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        cand_adapter = run_dir / "adapter"
        cand_hfout   = run_dir / "hf_out"
        adapter_path = str(cand_adapter) if cand_adapter.exists() else None
        hf_out_path  = str(cand_hfout) if cand_hfout.exists() else None
        if not adapter_path and not hf_out_path:
            continue

        # latest mtime among the two (or run dir if needed)
        mtimes: List[float] = []
        for p in [cand_adapter, cand_hfout, run_dir]:
            try:
                mtimes.append(p.stat().st_mtime)
            except Exception:
                pass
        latest_mtime = max(mtimes) if mtimes else 0.0

        runs[run_id] = {
            "run_id": run_id,
            "adapter": adapter_path,
            "hf_out": hf_out_path,
            "mtime": latest_mtime,
        }

    # sort newest first
    out = sorted(runs.values(), key=lambda r: float(r.get("mtime", 0.0)), reverse=True)
    return out


def _select_adapter_from_runs(
    runs: List[Dict[str, Optional[str]]],
    run_id: Optional[str],
    prefer_subdirs: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Choose adapter path & run_id using preference rules.
    1) If run_id is provided, use that run if present.
    2) Else pick newest run; within a run, prefer subdirs in order (e.g., adapter -> hf_out).
    Returns (adapter_path, chosen_run_id).
    """
    subs = [s.lower() for s in (prefer_subdirs or ["adapter", "hf_out"])]

    def pick_from_run(r: Dict[str, Optional[str]]) -> Optional[str]:
        for s in subs:
            p = r.get(s)
            if p:
                return p
        return None

    if run_id:
        for r in runs:
            if r.get("run_id") == run_id:
                ap = pick_from_run(r)
                return ap, run_id
        # if desired run not found, fall through to newest

    for r in runs:
        ap = pick_from_run(r)
        if ap:
            return ap, r.get("run_id")
    return None, None


def load_profile(
    doc_id: str,
    artifacts_root: Path = Path("data/artifacts"),
    models_root: Path = Path("data/models"),
    run_id: Optional[str] = None,
    prefer_subdirs: Optional[List[str]] = None,
) -> DocProfile:
    """
    Reads {artifacts_root}/{doc_id}/profile.json if present.
    Selection order for adapter:
      - If profile.json has adapter_path -> use it.
      - Else if profile.json has adapter_run_id -> select that run.
      - Else if run_id was passed in (from YAML) -> select that run.
      - Else auto-pick newest run by mtime.
    Within a selected run, prefer subdirs in order from prefer_subdirs (default: ["adapter","hf_out"]).
    """
    p = artifacts_root / doc_id / "profile.json"
    collection = doc_id
    reranker_model_path = None
    adapter_path = None
    calculators_root = None
    profile_run_id = None

    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        collection = d.get("collection", collection)
        reranker_model_path = d.get("reranker_model_path")
        adapter_path = d.get("adapter_path")
        calculators_root = d.get("calculators_root")
        profile_run_id = d.get("adapter_run_id")

    # If we already have a concrete path, we're done.
    if adapter_path:
        return DocProfile(
            doc_id=doc_id,
            collection=collection,
            reranker_model_path=reranker_model_path,
            adapter_path=adapter_path,
            calculators_root=calculators_root,
            adapter_run_id=profile_run_id,
        )

    # discover and select a run
    runs = _discover_runs(models_root, doc_id)
    if not runs:
        # no adapters yet — still return a profile without adapter
        return DocProfile(
            doc_id=doc_id,
            collection=collection,
            reranker_model_path=reranker_model_path,
            adapter_path=None,
            calculators_root=calculators_root,
            adapter_run_id=None,
        )

    chosen_path, chosen_run = _select_adapter_from_runs(
        runs=runs,
        run_id=profile_run_id or run_id,
        prefer_subdirs=prefer_subdirs or ["adapter", "hf_out"],
    )

    return DocProfile(
        doc_id=doc_id,
        collection=collection,
        reranker_model_path=reranker_model_path,
        adapter_path=chosen_path,
        calculators_root=calculators_root,
        adapter_run_id=chosen_run,
    )


# ---------------------------
# Mount result
# ---------------------------
@dataclass
class ChatMount:
    registry: ModelRegistry
    profile: DocProfile
    retriever: Retriever
    llm_intro: Any
    llm_splitter: Any
    llm_core: Any
    llm_output: Any
    sessions_dir: Path
    tools: List[StructuredTool]          # retriever + calculators
    retriever_tool: StructuredTool       # convenience handle


# ---------------------------
# Helper: build retriever tool (LangChain StructuredTool)
# ---------------------------
def _build_retriever_tool(retr: Retriever, name: str = "doc_retrieve", description: str = "Search this document and return top snippets with citations.") -> StructuredTool:
    class SearchInput(BaseModel):
        query: str = Field(..., description="User question to search in the document.")
        top_k: int = Field(retr.cfg.return_top_k, ge=1, le=50, description="How many snippets to return.")

    def _run(query: str, top_k: int = retr.cfg.return_top_k) -> Dict[str, Any]:
        hits = retr.search(query)[: top_k]
        return {
            "snippets": [{"id": h["id"], "text": h["text"], "metadata": h.get("metadata", {}), "score": h.get("score", 0.0)} for h in hits],
            "citations": build_citations([{"id": h["id"], "metadata": h.get("metadata", {}), "score": h.get("score", 0.0)} for h in hits]),
        }

    async def _arun(query: str, top_k: int = retr.cfg.return_top_k) -> Dict[str, Any]:
        # retriever is sync; offload if needed later
        return _run(query, top_k=top_k)

    return StructuredTool.from_function(
        name=name,
        description=description,
        func=_run,
        coroutine=_arun,
        args_schema=SearchInput,
        return_direct=False,
    )


# ---------------------------
# Helper: build calculator tools (optional, dynamic)
# ---------------------------
def _build_calculator_tools(doc_id: str) -> List[StructuredTool]:
    tools: List[StructuredTool] = []
    if not _HAS_CALC:
        return tools

    try:
        specs = calc_registry.list_tools(doc_id)
    except Exception:
        specs = []

    if not specs:
        # Generic fallback tool
        class GenericCalcInput(BaseModel):
            calculator_id: str = Field(..., description="Calculator id from the document profile/registry.")
            inputs: Dict[str, Any] = Field(..., description="Key-value inputs for the calculator.")
        def _run(calculator_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = calc_executor.execute(doc_id, calculator_id, **inputs)
                return {"calculator_id": calculator_id, "inputs": inputs, "result": result}
            except Exception as e:
                return {"calculator_id": calculator_id, "error": str(e)}
        async def _arun(calculator_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return _run(calculator_id, inputs)
        tools.append(
            StructuredTool.from_function(
                name="calc_run",
                description="Execute a document-specific calculator by id with JSON inputs.",
                func=_run,
                coroutine=_arun,
                args_schema=GenericCalcInput,
                return_direct=False,
            )
        )
        return tools

    # Per-tool (preferred)
    for sp in specs:
        tool_id = sp.get("id") or sp.get("name")
        tool_name = sp.get("name") or f"calc_{tool_id}"
        description = sp.get("description", f"Calculator tool {tool_name}")

        # Build pydantic schema dynamically
        fields = {}
        for prm in (sp.get("params") or []):
            p_name = prm.get("name")
            p_desc = prm.get("description", "")
            required = bool(prm.get("required", True))
            default = ... if required else prm.get("default", None)
            fields[p_name] = (Any, Field(default, description=p_desc))
        ArgsSchema = create_model(f"{tool_name}_Args", **fields)  # type: ignore

        def _make_run(_tool_id: str) -> Callable[..., Dict[str, Any]]:
            def _run(**kwargs) -> Dict[str, Any]:
                try:
                    result = calc_executor.execute(doc_id, _tool_id, **kwargs)
                    return {"calculator_id": _tool_id, "inputs": kwargs, "result": result}
                except Exception as e:
                    return {"calculator_id": _tool_id, "error": str(e)}
            return _run

        run_fn = _make_run(tool_id)

        tools.append(
            StructuredTool.from_function(
                name=tool_name,
                description=description,
                func=run_fn,
                args_schema=ArgsSchema,
                return_direct=False,
            )
        )
    return tools


# ---------------------------
# public: mount_chat
# ---------------------------
def mount_chat(
    doc_id: str,
    providers_yaml: str = "configs/providers.yaml",
    artifacts_root: Path = Path("data/artifacts")
) -> ChatMount:
    reg = ModelRegistry(providers_yaml)

    # Pass optional selection hints from YAML (for hf_local core)
    prof = load_profile(
        doc_id,
        artifacts_root=artifacts_root,
        models_root=Path("data/models"),
        run_id=reg.core_run_id,
        prefer_subdirs=reg.core_prefer_subdirs,
    )

    # retriever knobs come from providers.yaml -> chat.search_top_k etc.
    top_k = int(reg.search_top_k or reg.cfg.get("search_top_k", 12))
    rerank_top_k = int(reg.rerank_top_k or reg.cfg.get("rerank_top_k", 8))
    return_top_k = int(reg.return_top_k or reg.cfg.get("return_top_k", 6))

    retr = Retriever(
        doc_id=doc_id,
        cfg=RetrieverConfig(
            persist_path=reg.vector_path,             # picked from registry/yaml
            collection=prof.collection,
            embed_model_or_path=reg.embed_model,
            device=reg.embed_device,
            bge_use_prompt=True,
            top_k=top_k, rerank_top_k=rerank_top_k, return_top_k=return_top_k,
            reranker_model_path=prof.reranker_model_path,
        )
    )

    llm_intro    = reg.get_intro()
    llm_splitter = reg.get_splitter()
    llm_core     = reg.get_core(adapter_path=prof.adapter_path)
    llm_output   = reg.get_output()

    # Tools (retriever is always present; calculators if available)
    retr_tool = _build_retriever_tool(retr)
    tools = [retr_tool] + _build_calculator_tools(doc_id)

    reg.sessions_dir.mkdir(parents=True, exist_ok=True)
    return ChatMount(
        registry=reg,
        profile=prof,
        retriever=retr,
        llm_intro=llm_intro,
        llm_splitter=llm_splitter,
        llm_core=llm_core,
        llm_output=llm_output,
        sessions_dir=reg.sessions_dir,
        tools=tools,
        retriever_tool=retr_tool,
    )
