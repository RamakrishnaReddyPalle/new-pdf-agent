# core reasoning with RAG
# packages/chat/core_llm.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Iterable
import asyncio

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from packages.core_config.config import load_yaml
try:
    from packages.chat.prompt_loader import load_prompt
except Exception:
    def load_prompt(rel_path: str, fallback: Optional[str] = None) -> str:
        return fallback or ""

from packages.schemas.chat import CoreAnswer
from packages.chat.tools import build_citations, Retriever

# ---------- config ----------
def _cfg():
    y = load_yaml("configs/providers.yaml")
    chat = y.get("chat", {}) or {}
    prompts = chat.get("prompts", {}) or {}
    so = chat.get("structured_output", {}) or {}
    return {
        "core_prompt_path": prompts.get("core_rag_path", "configs/prompts/chat/core_rag.txt"),
        "prefer_native": bool(so.get("prefer_native", True)),
        "method": str(so.get("method", "json_schema")),
        "return_top_k": int(chat.get("return_top_k", 6) or 6),
        "parallel_per_question": int(chat.get("parallel_per_question", 1) or 1),
    }

def _maybe_structured(llm: BaseChatModel, schema, sys_text: str, user_text: str, method: str) -> Optional[CoreAnswer]:
    try:
        runnable = llm.with_structured_output(schema, include_raw=False, method=method)
        res = runnable.invoke([SystemMessage(content=sys_text), HumanMessage(content=user_text)])
        if isinstance(res, CoreAnswer):
            return res
    except Exception:
        pass
    return None

# ---------- single answer ----------
def answer_one(llm_core: BaseChatModel, retriever: Retriever, question: str, top_k: Optional[int] = None) -> CoreAnswer:
    cfg = _cfg()
    k = int(top_k or cfg["return_top_k"])
    hits = retriever.search(question)[: k]
    context_json = json.dumps(
        [{"id": h["id"], "text": h["text"], "metadata": h.get("metadata", {}), "score": h.get("score", 0.0)} for h in hits],
        ensure_ascii=False, indent=2
    )

    tmpl = load_prompt(cfg["core_prompt_path"], fallback=(
        "You will answer from CONTEXT only.\nQUESTION:\n{question}\n\nCONTEXT_JSON:\n{context_json}\n\n"
        "Return JSON: {\"answer\":\"...\",\"citations\":[{\"id\":\"...\"}]}"
    ))
    sys_text = "Answer strictly from the provided snippets and return JSON."
    user_text = tmpl.format(question=question, context_json=context_json)

    # prefer native structured output
    if cfg["prefer_native"]:
        res = _maybe_structured(llm_core, CoreAnswer, sys_text, user_text, method=cfg["method"])
        if isinstance(res, CoreAnswer):
            # ensure citations ids exist; if model gave none, use top-2
            if not res.citations:
                res.citations = [c for c in map(lambda x: {"id": x["id"]}, hits[:2]) if "id" in x]
            return res

    # fallback to explicit Pydantic parser
    parser = PydanticOutputParser(pydantic_object=CoreAnswer)
    prompt = PromptTemplate.from_template("{sys}\n\n{user}\n\n{format_instructions}")
    raw = llm_core.invoke(prompt.format(
        sys=sys_text,
        user=user_text,
        format_instructions=parser.get_format_instructions()
    ))
    text = raw.content if hasattr(raw, "content") else str(raw)
    try:
        parsed = parser.parse(text)
        if not parsed.citations:
            parsed.citations = [c for c in map(lambda x: {"id": x["id"]}, hits[:2]) if "id" in x]
        return parsed
    except Exception:
        return CoreAnswer(
            answer="I couldn’t parse a structured answer. From the excerpts, no clear answer was found.",
            citations=[{"id": h["id"]} for h in hits[:2]]
        )

# ---------- batch (simple concurrency) ----------
async def answer_batch(llm_core: BaseChatModel, retriever: Retriever, questions: List[str], top_k: Optional[int] = None) -> List[CoreAnswer]:
    cfg = _cfg()
    sem = asyncio.Semaphore(max(1, cfg["parallel_per_question"]))
    out: List[CoreAnswer] = [None] * len(questions)  # type: ignore

    async def _one(i: int, q: str):
        async with sem:
            # run sync answer_one in default loop executor
            res = await asyncio.to_thread(answer_one, llm_core, retriever, q, top_k)
            out[i] = res

    await asyncio.gather(*[_one(i, q) for i, q in enumerate(questions)])
    return out
