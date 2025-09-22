# packages/chat/schemas.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class ScopeDecision(BaseModel):
    in_scope: bool = Field(..., description="Whether the user question is in scope of this PDF.")
    intent: str = Field(..., description='One of ["qa","table","math","glossary","meta","other"].')
    reason: str = Field(..., description="Short reason for the decision.")
    rewritten: Optional[str] = Field(None, description="Cleaned/re-scoped version of user query if needed.")

class SplitQuestion(BaseModel):
    id: str = Field(..., description="Stable id for the sub-question.")
    text: str = Field(..., description="The atomic question text.")

class SplitPlan(BaseModel):
    questions: List[SplitQuestion] = Field(default_factory=list)
    notes: str = Field(default="", description="Any notes about the split/cleanup.")

class Citation(BaseModel):
    id: str
    page: Optional[int] = None
    heading_path: Optional[List[str]] = None
    score: Optional[float] = None

class CoreAnswer(BaseModel):
    answer: str = Field(..., description="Concise answer grounded in context.")
    citations: List[Citation] = Field(default_factory=list)
