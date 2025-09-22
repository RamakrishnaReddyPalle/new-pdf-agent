# packages/chat/memory.py
from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel

# Optional prompt loader (Phase 7.0)
try:
    from packages.chat.prompt_loader import load_prompt
except Exception:
    def load_prompt(rel_path: str, fallback: Optional[str] = None) -> str:
        return fallback or ""

_DEFAULT_SUMMARY_PROMPT = """You are a summarizer.
Given the PRIOR_SUMMARY (may be empty) and the NEW_TURNS,
write a concise running summary that preserves facts and user intent.
Return only plain text, no bullets.

PRIOR_SUMMARY:
{prior_summary}

NEW_TURNS:
{new_turns}
"""

def _now_ts() -> float:
    return time.time()

def _uuid() -> str:
    return uuid.uuid4().hex[:12]

@dataclass
class Turn:
    role: str          # "user" | "assistant" | "system"
    content: str
    ts: float
    meta: Dict[str, Any]

@dataclass
class SessionState:
    doc_id: str
    session_id: str
    created_at: float
    updated_at: float
    summary: str
    turns: List[Turn]

    @staticmethod
    def new(doc_id: str, session_id: Optional[str] = None) -> "SessionState":
        sid = session_id or _uuid()
        ts = _now_ts()
        return SessionState(
            doc_id=doc_id,
            session_id=sid,
            created_at=ts,
            updated_at=ts,
            summary="",
            turns=[],
        )

class SessionStore:
    """
    JSON-backed chat session store.
    File path: sessions_dir / f"{doc_id}_{session_id}.json"
    """
    def __init__(self, sessions_dir: Path, doc_id: str, session_id: Optional[str] = None):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.doc_id = doc_id
        self.session_id = session_id
        self.state: Optional[SessionState] = None

    @property
    def path(self) -> Path:
        sid = self.session_id or "unknown"
        return self.sessions_dir / f"{self.doc_id}_{sid}.json"

    # IMPORTANT: return the STORE (so you can chain .append)
    def load_or_create(self) -> "SessionStore":
        if self.session_id and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                turns = [Turn(**t) for t in data.get("turns", [])]
                self.state = SessionState(
                    doc_id=data["doc_id"],
                    session_id=data["session_id"],
                    created_at=float(data["created_at"]),
                    updated_at=float(data["updated_at"]),
                    summary=data.get("summary", ""),
                    turns=turns,
                )
            except Exception:
                self.state = SessionState.new(self.doc_id, self.session_id)
                self.session_id = self.state.session_id
                self._flush()
        else:
            self.state = SessionState.new(self.doc_id, self.session_id)
            self.session_id = self.state.session_id
            self._flush()
        return self  # <— return the store, not the state

    def _flush(self) -> None:
        assert self.state is not None, "call load_or_create() first"
        self.state.updated_at = _now_ts()
        obj = {
            "doc_id": self.state.doc_id,
            "session_id": self.state.session_id,
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at,
            "summary": self.state.summary,
            "turns": [asdict(t) for t in self.state.turns],
        }
        self.path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self) -> None:
        self._flush()

    def append(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Append a turn and persist to disk."""
        assert self.state is not None, "call load_or_create() first"
        rec = Turn(role=role, content=content, ts=_now_ts(), meta=meta or {})
        self.state.turns.append(rec)
        self._flush()
        return asdict(rec)

    # Back-compat alias
    def add(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None):
        return self.append(role, content, meta=meta)

    def last_n(self, n: int = 10) -> List[Turn]:
        assert self.state is not None, "call load_or_create() first"
        return self.state.turns[-n:]

    def as_langchain_messages(self, n: int = 10, include_summary: bool = True) -> List[BaseMessage]:
        """Return LC messages: optional system summary + last n turns."""
        assert self.state is not None, "call load_or_create() first"
        msgs: List[BaseMessage] = []
        if include_summary and self.state.summary:
            msgs.append(SystemMessage(content=f"Chat so far (summary): {self.state.summary}"))
        for t in self.last_n(n):
            if t.role == "user":
                msgs.append(HumanMessage(content=t.content))
            elif t.role == "assistant":
                msgs.append(AIMessage(content=t.content))
            else:
                msgs.append(SystemMessage(content=t.content))
        return msgs

class SummaryBuffer:
    """
    Maintains a rolling summary using an LLM.
    Heuristics: summarize when (len(turns) >= summarize_every) or char_budget exceeded.
    """
    def __init__(
        self,
        store: SessionStore,
        llm: BaseChatModel,
        summarize_every: int = 12,
        char_budget: int = 6000,
        prompt_relpath: str = "chat/memory_summarize.txt",
    ):
        self.store = store
        self.llm = llm
        self.summarize_every = int(summarize_every)
        self.char_budget = int(char_budget)
        self.prompt = load_prompt(prompt_relpath, fallback=_DEFAULT_SUMMARY_PROMPT)

    def _need_summarize(self) -> bool:
        st = self.store.state or self.store.load_or_create().state
        if len(st.turns) >= self.summarize_every and (len(st.turns) % self.summarize_every == 0):
            return True
        total_chars = sum(len(t.content) for t in st.turns)
        return total_chars > self.char_budget

    def _format_new_turns(self, k: int = 8) -> str:
        st = self.store.state or self.store.load_or_create().state
        tail = st.turns[-k:]
        lines = [f"[{t.role}] {t.content}" for t in tail]
        return "\n".join(lines)

    def summarize_now(self) -> str:
        st = self.store.state or self.store.load_or_create().state
        new_turns = self._format_new_turns()
        prompt_text = self.prompt.format(prior_summary=st.summary or "", new_turns=new_turns)
        msgs = [SystemMessage(content="Summarize the conversation succinctly."), HumanMessage(content=prompt_text)]
        try:
            out = self.llm.invoke(msgs)
            new_summary = out.content if hasattr(out, "content") else str(out)
        except Exception:
            new_summary = st.summary  # keep old if LLM unavailable
        st.summary = (new_summary or "").strip()
        self.store.save()
        return st.summary

    async def asummarize_now(self) -> str:
        st = self.store.state or self.store.load_or_create().state
        new_turns = self._format_new_turns()
        prompt_text = self.prompt.format(prior_summary=st.summary or "", new_turns=new_turns)
        msgs = [SystemMessage(content="Summarize the conversation succinctly."), HumanMessage(content=prompt_text)]
        try:
            out = await self.llm.ainvoke(msgs)
            new_summary = out.content if hasattr(out, "content") else str(out)
        except Exception:
            new_summary = st.summary
        st.summary = (new_summary or "").strip()
        self.store.save()
        return st.summary

    def maybe_summarize(self) -> Optional[str]:
        if self._need_summarize():
            return self.summarize_now()
        return None

    async def amaybe_summarize(self) -> Optional[str]:
        if self._need_summarize():
            return await self.asummarize_now()
        return None
