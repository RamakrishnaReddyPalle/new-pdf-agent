# packages/chat/splitter.py
from __future__ import annotations
import json, re
from typing import Optional, List

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

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
    def load_prompt(path: str, fallback: Optional[str] = None) -> str:
        return fallback or ""

from packages.schemas.chat import SplitQuestion, SplitPlan

_DEFAULT_SPLITTER_PROMPT = (
    "You are a careful query splitter.\n"
    "Split USER_QUERY into minimal atomic questions (max {max_questions} items).\n"
    "Only include questions that are actually asked. If there is just one intent, return one question.\n\n"
    "Return strictly in this JSON schema:\n{format_instructions}\n\n"
    "Rules:\n"
    "- Keep each question self-contained and concise.\n"
    "- Do not invent facts.\n"
    "- If you are unsure, prefer fewer questions.\n"
    "- If parsing is ambiguous, put full query as a single question.\n"
    "- \"notes\" may include short lint comments (only if allowed: {allow_notes}).\n"
)

def _cfg():
    y = load_yaml("configs/providers.yaml")
    return {
        "max_questions": int(y.get("chat.splitter.max_questions", 6) or 6),
        "allow_notes":   bool(y.get("chat.splitter.allow_notes", True)),
        "prompt_path":   str(y.get("chat.splitter.prompt_path", "configs/prompts/chat/splitter.txt")),
        "so_prefer":     bool(y.get("chat.structured_output.prefer_native", True)),
        "so_method":     str(y.get("chat.structured_output.method", "json_schema")),
    }

def _heuristic_split(q: str, max_questions: int) -> List[str]:
    parts = re.split(r"\s*[;•\n]\s*|(?<![A-Za-z])(?:[A-Ca-c]\))\s*", q.strip())
    if len(parts) == 1:
        parts = re.split(r"\s+\band\b\s+|\s+\balso\b\s+|,\s*(?=[A-Z])", q.strip())
    out = [p.strip().strip(".") for p in parts if p and p.strip()]
    return (out or [q.strip()])[:max_questions]

def _extract_json(text: str) -> Optional[str]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m: return m.group(1)
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    return m.group(1) if m else None

def _maybe_structured(llm: BaseChatModel, schema, sys_prompt: str, user_query: str, method: str) -> Optional[SplitPlan]:
    try:
        runnable = llm.with_structured_output(schema, include_raw=False, method=method)
        res = runnable.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=f"USER_QUERY:\n{user_query}")])
        return res if isinstance(res, SplitPlan) else None
    except Exception:
        return None

def split_and_clean(llm: BaseChatModel, user_query: str,
                    max_questions: Optional[int] = None,
                    allow_notes: Optional[bool] = None) -> SplitPlan:
    cfg = _cfg()
    max_q = int(max_questions if max_questions is not None else cfg["max_questions"])
    allow_n = bool(allow_notes if allow_notes is not None else cfg["allow_notes"])

    # 1) native structured output
    if cfg["so_prefer"]:
        sys_prompt = "Split the following user query into minimal atomic questions."
        res = _maybe_structured(llm, SplitPlan, sys_prompt, user_query, method=cfg["so_method"])
        if isinstance(res, SplitPlan):
            if not res.questions:
                res.questions = [SplitQuestion(id="q1", text=user_query if user_query.endswith("?") else user_query.rstrip(".") + "?")]
            for i, q in enumerate(res.questions, 1):
                if not q.id: q.id = f"q{i}"
            return res

    # 2) prompted JSON + parser
    parser = PydanticOutputParser(pydantic_object=SplitPlan)
    tmpl = load_prompt(cfg["prompt_path"], _DEFAULT_SPLITTER_PROMPT)
    prompt = PromptTemplate.from_template(tmpl)

    try:
        raw = llm.invoke([
            SystemMessage(content=prompt.format(
                max_questions=max_q,
                allow_notes=str(allow_n).lower(),
                format_instructions=parser.get_format_instructions(),
            )),
            HumanMessage(content=f"USER_QUERY:\n{user_query}")
        ])
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            return parser.parse(text)
        except Exception:
            j = _extract_json(text)
            if j:
                obj = json.loads(j)
                return SplitPlan(**obj)
            raise
    except Exception:
        pass

    # 3) heuristic fallback
    qs = _heuristic_split(user_query, max_q)
    return SplitPlan(
        questions=[SplitQuestion(id=f"q{i+1}", text=s if s.endswith("?") else s.rstrip(".") + "?")
                   for i, s in enumerate(qs)],
        notes="fallback: parse error"
    )

async def asplit_and_clean(llm: BaseChatModel, user_query: str,
                           max_questions: Optional[int] = None,
                           allow_notes: Optional[bool] = None) -> SplitPlan:
    return split_and_clean(llm, user_query, max_questions=max_questions, allow_notes=allow_notes)
