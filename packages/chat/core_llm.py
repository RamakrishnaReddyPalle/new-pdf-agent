# core reasoning with RAG
# packages/chat/core_llm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio, json, textwrap

from packages.chat.tools import Retriever, RetrieverConfig, build_citations

_CORE_PROMPT = """You answer strictly from the given CONTEXT. Cite snippet ids you used.
Return JSON: {"answer":"...", "citations":[{"id":"...","page":...,"heading_path":[...]}]}
QUESTION:
{question}

CONTEXT (top snippets):
{context}
"""

@dataclass
class CoreAnswer:
    answer: str
    citations: List[Dict[str, Any]]

async def _retrieve_ctx(retr: Retriever, question: str) -> List[Dict[str, Any]]:
    # if retriever is sync, run in thread
    return await asyncio.to_thread(retr.search, question)

async def answer_one(
    question: str,
    retriever: Retriever,
    llm_core,                      # core model (LoRA)
    stream: bool = False
) -> AsyncGenerator[Dict[str, Any], None] | CoreAnswer:
    snips = await _retrieve_ctx(retriever, question)
    citations = build_citations(snips)
    ctx = "\n\n".join(f"[{s['id']}] {s['text']}" for s in snips)
    prompt = _CORE_PROMPT.format(question=question, context=ctx[:4000])

    if stream and hasattr(llm_core, "generate_stream"):
        # token streaming (if your provider supports)
        chunk_buf = ""
        async for tok in llm_core.generate_stream(prompt):
            chunk_buf += tok
            yield {"type":"token","data":tok}
        # parse at end
        try:
            data = json.loads(chunk_buf)
        except Exception:
            data = {"answer": chunk_buf, "citations": citations[:2]}
        yield {"type":"final","data": CoreAnswer(answer=data.get("answer",""), citations=citations)}
        return

    out = llm_core.generate(prompt)
    try:
        data = json.loads(out)
    except Exception:
        data = {"answer": out, "citations": citations[:2]}
    return CoreAnswer(answer=data.get("answer",""), citations=citations)
