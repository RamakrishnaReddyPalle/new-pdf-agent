# packages/chat/core_rag.py
from __future__ import annotations
from typing import List, Dict, Any, AsyncGenerator, Optional
import asyncio, json

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool

from packages.chat.schemas import CoreAnswer, Citation
from packages.chat.tools import Retriever, build_citations

@tool("retrieve_context", return_direct=False)
def retrieve_context(question: str) -> str:
    """Retrieve top-k context snippets for a question. Returns a JSON list of {id,text,metadata,score}."""
    # This body will be overwritten by injector at runtime to use the mounted retriever.
    raise RuntimeError("Injector must set retriever before use.")

_CORE_TMPL = PromptTemplate.from_template(
"""You are a careful, grounded assistant. Use ONLY the provided CONTEXT to answer.
Cite snippet ids you used.
{format_instructions}

QUESTION:
{question}

CONTEXT (JSON list of snippets):
{context_json}
""")

class CoreRAG:
    def __init__(self, retriever: Retriever, llm_core):
        self.retriever = retriever
        self.llm_core = llm_core

    async def _get_ctx(self, q: str) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.retriever.search, q)

    async def answer_one(self, question: str, stream: bool = False) -> CoreAnswer | AsyncGenerator[dict, None]:
        snips = await self._get_ctx(question)
        citations = build_citations(snips)
        ctx_json = json.dumps(snips[:8], ensure_ascii=False)

        parser = PydanticOutputParser(pydantic_object=CoreAnswer)
        prompt = _CORE_TMPL.format(
            question=question,
            context_json=ctx_json,
            format_instructions=parser.get_format_instructions()
        )

        if stream and hasattr(self.llm_core, "stream"):
            # ChatOllama.stream yields tokens
            buf = ""
            async for ev in self.llm_core.astream(prompt):
                token = getattr(ev, "content", None) or (isinstance(ev, str) and ev) or ""
                if token:
                    buf += token
                    yield {"type":"token", "data":token}
            try:
                data = json.loads(buf)
                ans = CoreAnswer(**data)
            except Exception:
                ans = CoreAnswer(answer=buf, citations=[Citation(**c) for c in citations[:2]])
            yield {"type":"final", "data":ans}
            return

        txt = await self.llm_core.ainvoke(prompt)
        content = getattr(txt, "content", str(txt))
        try:
            return CoreAnswer.model_validate_json(content)
        except Exception:
            return CoreAnswer(answer=content, citations=[Citation(**c) for c in citations[:2]])
