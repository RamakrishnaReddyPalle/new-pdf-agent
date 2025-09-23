# packages/chat/core_rag.py
from __future__ import annotations
from typing import List, Dict, Any, AsyncGenerator, Optional
import asyncio, json, re

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from packages.core_config.config import load_yaml
try:
    from packages.chat.prompt_loader import load_prompt
except Exception:
    def load_prompt(rel_path: str, fallback: Optional[str] = None) -> str:
        return fallback or ""

from packages.schemas.chat import CoreAnswer, Citation
from packages.chat.tools import Retriever, build_citations


# --------- Config helpers ----------
def _so_cfg():
    y = load_yaml("configs/providers.yaml")
    return {
        "prefer_native": bool(y.get("chat.structured_output.prefer_native", True)),
        "method": str(y.get("chat.structured_output.method", "json_schema")),
    }

# --------- Prompt (from file, with safe fallback) ----------
_DEFAULT_CORE_PROMPT = """You are a careful, grounded assistant. Use ONLY the provided CONTEXT to answer.
Cite snippet ids you used.

Return strictly in this JSON schema:
{format_instructions}

QUESTION:
{question}

CONTEXT (JSON list of top snippets):
{context_json}
"""

def _load_core_prompt() -> str:
    # configurable prompt path (you can add to YAML later if you like)
    return load_prompt("configs/prompts/chat/core_rag.txt", fallback=_DEFAULT_CORE_PROMPT)

# --------- Try native structured output first ----------
def _maybe_structured(llm_core, schema, question: str, ctx_json: str) -> Optional[CoreAnswer]:
    cfg = _so_cfg()
    try:
        runnable = llm_core.with_structured_output(schema, include_raw=False, method=cfg["method"])
        sys = SystemMessage(content="Answer from CONTEXT only. Return the required JSON.")
        usr = HumanMessage(content=f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_json}")
        return runnable.invoke([sys, usr])
    except Exception:
        return None

# --------- JSON parse helper ----------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _parse_core_answer(text: str, fallback_citations: List[Dict[str, Any]]) -> CoreAnswer:
    # 1) direct model-validated JSON (fast path)
    try:
        return CoreAnswer.model_validate_json(text)
    except Exception:
        pass

    # 2) extract a JSON object if wrapped in prose/code fences
    m = _JSON_RE.search(text or "")
    if m:
        try:
            return CoreAnswer(**json.loads(m.group(0)))
        except Exception:
            pass

    # 3) fall back to plain text + minimal citations
    return CoreAnswer(
        answer=(text or "").strip(),
        citations=[Citation(**c) for c in (fallback_citations[:2] or [])]
    )

# --------- Core RAG class ----------
class CoreRAG:
    def __init__(self, retriever: Retriever, llm_core):
        self.retriever = retriever
        self.llm_core = llm_core
        self._prompt_tmpl = PromptTemplate.from_template(_load_core_prompt())

    async def _get_ctx(self, q: str, limit: int = 8) -> tuple[list[dict], list[dict]]:
        snips = await asyncio.to_thread(self.retriever.search, q)
        snips = snips[:limit]
        citations = build_citations(snips)
        return snips, citations

    async def answer_one(self, question: str) -> CoreAnswer:
        """
        Non-streaming: returns CoreAnswer
        """
        snips, citations = await self._get_ctx(question)
        ctx_json = json.dumps(snips, ensure_ascii=False)

        # 1) prefer native structured output when available
        res = _maybe_structured(self.llm_core, CoreAnswer, question, ctx_json)
        if isinstance(res, CoreAnswer):
            return res

        # 2) prompt + pydantic parser
        parser = PydanticOutputParser(pydantic_object=CoreAnswer)
        prompt = self._prompt_tmpl.format(
            question=question,
            context_json=ctx_json,
            format_instructions=parser.get_format_instructions(),
        )

        try:
            # LangChain chat models return AIMessage; .content is the text
            txt = await self.llm_core.ainvoke(prompt)
            content = getattr(txt, "content", str(txt))
            return _parse_core_answer(content, citations)
        except Exception as e:
            # final fallback (should be rare)
            return CoreAnswer(
                answer=f"[error: {type(e).__name__}]",
                citations=[Citation(**c) for c in citations[:1]]
            )

    async def astream_one(self, question: str) -> AsyncGenerator[dict, None]:
        """
        Streaming: yields {'type': 'core_token'|'core_final', ...}
        """
        snips, citations = await self._get_ctx(question)
        ctx_json = json.dumps(snips, ensure_ascii=False)

        # We still steer structure via the same prompt; LLM may stream raw text
        parser = PydanticOutputParser(pydantic_object=CoreAnswer)
        prompt = self._prompt_tmpl.format(
            question=question,
            context_json=ctx_json,
            format_instructions=parser.get_format_instructions(),
        )

        buf = ""
        # Prefer astream if available (ChatOllama supports it)
        if hasattr(self.llm_core, "astream"):
            async for chunk in self.llm_core.astream(prompt):
                token = getattr(chunk, "content", None) or (isinstance(chunk, str) and chunk) or ""
                if token:
                    buf += token
                    yield {"type": "core_token", "data": token}

            ans = _parse_core_answer(buf, citations)
            yield {"type": "core_final", "data": ans}
            return

        # Fallback to non-stream (single yield)
        msg = await self.llm_core.ainvoke(prompt)
        content = getattr(msg, "content", str(msg))
        ans = _parse_core_answer(content, citations)
        yield {"type": "core_final", "data": ans}

    async def answer_batch(self, questions: List[str], parallel: Optional[int] = None) -> List[CoreAnswer]:
        """
        Non-streaming batch answers with bounded concurrency.
        """
        y = load_yaml("configs/providers.yaml")
        par = int(parallel or y.get("chat.parallel_per_question", 1) or 1)
        sem = asyncio.Semaphore(max(1, par))

        async def _one(q: str) -> CoreAnswer:
            async with sem:
                return await self.answer_one(q)

        return await asyncio.gather(*[_one(q) for q in questions])
