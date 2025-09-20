# core reasoning with RAG
# packages/chat/core_llm.py
from __future__ import annotations
from typing import Dict, Any, List

from packages.chat.tools import build_citations

SYSTEM_INSTRUCTIONS = """You answer strictly from the provided context.
- If the answer is in a table, quote the relevant rows.
- If math is involved, show the steps using symbols as written.
- Always include short, numbered citations like [1], [2] that map to the provided sources.
If the context is insufficient, say so briefly."""

def _format_context(snippets: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, s in enumerate(snippets, 1):
        meta = s.get("metadata") or {}
        head = " > ".join(meta.get("heading_path") or []) or "(no heading)"
        page = meta.get("page")
        tag = f"[{i}] Page {page} — {head}".strip()
        body = s["text"]
        blocks.append(f"{tag}\n{body}")
    return "\n\n".join(blocks)

def _append_citation_marks(text: str, n_sources: int) -> str:
    # naive: append [1], [2] for first two sources if not already present
    marks = " ".join(f"[{i}]" for i in range(1, min(n_sources, 3)+1))
    return f"{text}\n\n{marks}"

def answer_query(router_ctx: Dict[str, Any], query: str) -> Dict[str, Any]:
    retriever = router_ctx["retriever"]
    llm = router_ctx["llm"]

    snippets = retriever.search(query)
    if not snippets:
        return {"answer": "I couldn’t find anything relevant in the document.", "citations": []}

    context = _format_context(snippets)
    prompt = f"{SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nUser question: {query}\n\nAnswer:"

    llm.ensure_ready()
    raw = llm.generate(prompt).strip()
    # ensure we show some citation marks even if the model forgot
    final = _append_citation_marks(raw, len(snippets))
    return {"answer": final, "citations": build_citations(snippets)}
