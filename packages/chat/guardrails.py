# packages/chat/guardrails.py
from __future__ import annotations
import re
from typing import Any, Dict, Optional, Tuple

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

# config + prompt loading (no hard-coded strings)
try:
    from packages.core_config.config import load_yaml
except Exception:
    import yaml
    from pathlib import Path
    class _Cfg:
        def __init__(self, d): self.data = d
        def get(self, path, default=None):
            cur = self.data
            for part in path.split("."):
                if not isinstance(cur, dict) or part not in cur: return default
                cur = cur[part]
            return cur
    def load_yaml(*paths):
        merged = {}
        from pathlib import Path as _P
        for p in paths:
            if not p: continue
            p = _P(p)
            if p.exists():
                merged.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        return _Cfg(merged)

try:
    from packages.chat.prompt_loader import load_prompt
except Exception:
    from pathlib import Path
    def load_prompt(path: str, fallback: str = "") -> str:
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else fallback

from packages.schemas.chat import ScopeDecision

def _guard_cfg():
    y = load_yaml("configs/providers.yaml")
    return {
        "max_input_chars": int(y.get("chat.guardrails.max_input_chars", 4000) or 4000),
        "blocked_regex":   y.get("chat.guardrails.blocked_regex", []) or [],
        "pii_block":       bool(y.get("chat.guardrails.pii_block", False)),
        "pii_regex":       y.get("chat.guardrails.pii_regex", [
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        ]),
        "prompt_path":     str(y.get("chat.guardrails.prompt_path", "configs/prompts/chat/intro_guard.txt")),
        "so_prefer":       bool(y.get("chat.structured_output.prefer_native", True)),
        "so_method":       str(y.get("chat.structured_output.method", "json_schema")),
    }

# ---- pre-LLM checks ----
def prefilter_user_query(user_query: str) -> Tuple[bool, str, Optional[str]]:
    cfg = _guard_cfg()
    q = (user_query or "").strip()
    if not q: return False, "empty_query", None
    if len(q) > cfg["max_input_chars"]: return False, "too_long", None
    for pat in cfg["blocked_regex"]:
        if re.search(pat, q, flags=re.IGNORECASE): return False, "blocked_pattern", None
    if cfg["pii_block"]:
        for pat in cfg["pii_regex"]:
            if re.search(pat, q): return False, "pii_detected", None
    return True, "ok", q

# ---- prompt text from disk ----
def build_intro_guard_text(doc_id: str, sections: str, abbreviations: str) -> str:
    cfg = _guard_cfg()
    tmpl = load_prompt(cfg["prompt_path"], fallback=(
        "You are a scope router for document {doc_id}. Use ONLY this document.\n\n"
        "Sections (high-level TOC):\n{sections}\n\n"
        "Abbreviations (domain glossary):\n{abbreviations}\n\n"
        "TASK:\nGiven USER_QUERY, output a strict JSON object exactly like:\n"
        "{{\"in_scope\": true|false, \"intent\":\"qa|table|math|glossary|meta|other\", "
        "\"reason\":\"...\", \"rewritten\": \"...\"}}\n\n"
        "Guidelines:\n"
        "- \"in_scope\": true only if the question can be answered using THIS document.\n"
        "- \"intent\": qa|table|math|glossary|meta|other (pick one).\n"
        "- If unrelated, set in_scope=false and say why in \"reason\".\n"
        "- If in-scope, set \"rewritten\" to a concise form of the user question.\n"
        "- JSON only, no extra keys.\n"
    ))
    return tmpl.format(doc_id=doc_id, sections=sections, abbreviations=abbreviations)

def _try_structured(llm: BaseChatModel, schema, intro_text: str, user_query: str, method: str) -> Optional[ScopeDecision]:
    try:
        runnable = llm.with_structured_output(schema, include_raw=False, method=method)
        result = runnable.invoke([SystemMessage(content=intro_text),
                                  HumanMessage(content=f"USER_QUERY:\n{user_query}")])
        return result if isinstance(result, ScopeDecision) else None
    except Exception:
        return None

# ---- public entrypoint ----
def route_scope(
    llm_intro: BaseChatModel,
    *,
    doc_id: str,
    sections_text: str,
    abbreviations_text: str,
    user_query: str
) -> Dict[str, Any]:
    ok, reason, q_norm = prefilter_user_query(user_query)
    out: Dict[str, Any] = {"prefilter": {"ok": ok, "reason": reason}}
    if not ok:
        out["decision"] = None
        return out

    intro_text = build_intro_guard_text(doc_id=doc_id, sections=sections_text, abbreviations=abbreviations_text)
    cfg = _guard_cfg()

    # 1) Prefer native structured output
    if cfg["so_prefer"]:
        res = _try_structured(llm_intro, ScopeDecision, intro_text, q_norm, method=cfg["so_method"])
        if isinstance(res, ScopeDecision):
            out["decision"] = res.dict()
            return out

    # 2) Fallback: prompt + Pydantic parser
    parser = PydanticOutputParser(pydantic_object=ScopeDecision)
    prompt = PromptTemplate.from_template("INTRO:\n{intro}\n\nUSER_QUERY:\n{query}\n\n{format_instructions}")
    try:
        decision: ScopeDecision = (prompt | llm_intro | parser).invoke({
            "intro": intro_text,
            "query": q_norm,
            "format_instructions": parser.get_format_instructions(),
        })
        out["decision"] = decision.dict()
        return out
    except Exception as e:
        out["decision"] = ScopeDecision(in_scope=True, intent="qa", reason=f"fallback:{type(e).__name__}", rewritten=q_norm).dict()
        return out
