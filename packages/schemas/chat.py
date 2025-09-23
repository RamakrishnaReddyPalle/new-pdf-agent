# packages/schemas/chat.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

# ---- Splitter ----
class SplitQuestion(BaseModel):
    id: str = Field(..., description="Stable id for this atomic question (e.g., q1, q2).")
    text: str = Field(..., description="Clean, unambiguous question text.")

class SplitPlan(BaseModel):
    questions: List[SplitQuestion] = Field(default_factory=list)
    notes: Optional[str] = ""

# ---- Guardrails (scope/intents) ----
class ScopeDecision(BaseModel):
    in_scope: bool = Field(..., description="True if question can be answered with THIS document")
    intent: str = Field(..., description="qa|table|math|glossary|meta|other")
    reason: str = Field("", description="Short rationale")
    rewritten: Optional[str] = Field(None, description="Cleaned in-scope user query")

# ---- Core RAG (structured answer) ----
class Citation(BaseModel):
    id: str = Field(..., description="Snippet id from retriever.")
    page: Optional[int] = Field(None, description="Page number if available.")
    heading_path: Optional[str] = Field(None, description="Full heading path if available.")
    table_id: Optional[str] = Field(None, description="Source table id if applicable.")
    score: Optional[float] = Field(None, description="Retriever score if available.")

class CoreAnswer(BaseModel):
    answer: str = Field(..., description="Grounded answer text derived strictly from context.")
    citations: List[Citation] = Field(default_factory=list, description="Minimal citation payload.")
    notes: Optional[str] = Field("", description="Optional caveats or confidence notes.")
