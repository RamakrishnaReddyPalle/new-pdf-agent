# packages/chat/guardrails.py
from __future__ import annotations
import re
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

# --- prompt loader (no hard-coded strings) ---
try:
    # prefer the shared helper if present
    from packages.core_config.config import load_prompt, load_yaml  # type: ignore
except Exception:
    import yaml
    from pathlib import Path
    def load_prompt(path: str, fallback: str = "") -> str:
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else fallback
    def load_yaml(*paths):
        import yaml as _y
        from pathlib import Path as _P
        merged = {}
        for p in paths:
            if not p: continue
            p = _P(p)
            if p.exists():
                y = _y.safe_load(p.read_text(encoding="utf-8")) or {}
                merged.update(y)
        class _Cfg:
            def __init__(self, d): self.data = d
            def get(self, path, default=None):
                cur = self.data
                for part in path.split("."):
                    if not isinstance(cur, dict) or part not in cur: return default
                    cur = cur[part]
                return cur
        return _Cfg(merged)

# --- schema (import the real one if you already have it) ---
try:
    from packages.schemas.chat import ScopeDecision  # type: ignore
except Exception:
    class ScopeDecision(BaseModel):
        in_scope: bool
        intent: str
        reason: str
        rewritten: Optional[str] = None

# ---- config helpers ----
def _guard_cfg():
    y = load_yaml("configs/providers.yaml")
    return {
        "max_input_chars": int(y.get("chat.guardrails.max_input_chars", 4000) or 4000),
        "blocked_regex":   y.get("chat.guardrails.blocked_regex", []) or [],
        "pii_block":       bool(y.get("chat.guardrails.pii_block", False)),
        "pii_regex":       y.get("chat.guardrails.pii_regex", [
            # simple email / phone masks; tune as needed
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        ]),
        "prompt_path":     "configs/prompts/chat/intro_guard.txt",
    }

# ---- pre-LLM checks (cheap) ----
def prefilter_user_query(user_query: str) -> Tuple[bool, str, Optional[str]]:
    """
    Returns (ok, reason, normalized_query_or_none_if_blocked)
    """
    cfg = _guard_cfg()
    q = (user_query or "").strip()

    if not q:
        return False, "empty_query", None

    if len(q) > cfg["max_input_chars"]:
        return False, "too_long", None

    for pat in cfg["blocked_regex"]:
        if re.search(pat, q, flags=re.IGNORECASE):
            return False, "blocked_pattern", None

    if cfg["pii_block"]:
        for pat in cfg["pii_regex"]:
            if re.search(pat, q):
                return False, "pii_detected", None

    return True, "ok", q

# ---- LLM guard (structured output) ----
def build_intro_guard_text(doc_id: str,
                           sections: str,
                           abbreviations: str) -> str:
    path = _guard_cfg()["prompt_path"]
    tmpl = load_prompt(path, fallback="""
You are a scope router for document {doc_id}. Use ONLY this document.

Sections:
{sections}

Abbreviations:
{abbreviations}

Decide if the USER_QUERY is in scope of this document.
Return JSON:
{{"in_scope": true|false, "intent":"qa|table|math|glossary|meta|other", "reason":"...", "rewritten": "cleaned in-scope question or null"}}

Rules:
- If the user asks unrelated questions, set in_scope=false and suggest the relevant scope in "reason".
- If in-scope, pick an intent and provide a concise "rewritten" query if helpful.
""")
    return tmpl.format(doc_id=doc_id, sections=sections, abbreviations=abbreviations)

def route_scope(
    llm_intro: BaseChatModel,
    *,
    doc_id: str,
    sections_text: str,
    abbreviations_text: str,
    user_query: str
) -> Dict[str, Any]:
    """
    Applies prefilter, then LLM structured guard using Pydantic parser.
    Returns a dict:
      {"prefilter": {"ok": bool, "reason": str},
       "decision": ScopeDecision.dict() or None}
    """
    ok, reason, q_norm = prefilter_user_query(user_query)
    out: Dict[str, Any] = {"prefilter": {"ok": ok, "reason": reason}}
    if not ok:
        out["decision"] = None
        return out

    intro_text = build_intro_guard_text(
        doc_id=doc_id,
        sections=sections_text,
        abbreviations=abbreviations_text
    )

    parser = PydanticOutputParser(pydantic_object=ScopeDecision)
    prompt = PromptTemplate.from_template(
        "INTRO:\n{intro}\n\nUSER_QUERY:\n{query}\n\n{format_instructions}"
    )
    chain = prompt | llm_intro | parser

    try:
        decision: ScopeDecision = chain.invoke({
            "intro": intro_text,
            "query": q_norm,
            "format_instructions": parser.get_format_instructions()
        })
        out["decision"] = decision.dict()
        return out
    except Exception as e:
        # Fallback: if parser fails, default to in_scope=True with generic intent.
        out["decision"] = ScopeDecision(
            in_scope=True,
            intent="qa",
            reason=f"parser_error: {type(e).__name__}",
            rewritten=q_norm
        ).dict()
        return out
