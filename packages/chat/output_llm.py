# stitching & user-friendly answer
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import json

_STITCH_PROMPT = """Write a concise, user-friendly answer using the CORE_ANSWERS JSON and CHAT_MEMORY.
Keep citations as [id] markers when relevant. Do not invent content.

CORE_ANSWERS JSON:
{answers_json}

CHAT_MEMORY SUMMARY:
{memory}

Return ONLY the final user-facing text.
"""

@dataclass
class FinalMessage:
    text: str

def stitch_output(llm, core_answers: List[Dict[str, Any]], memory_summary: str) -> FinalMessage:
    prompt = _STITCH_PROMPT.format(answers_json=json.dumps(core_answers, ensure_ascii=False),
                                   memory=memory_summary or "(none)")
    text = llm.generate(prompt)
    return FinalMessage(text=text.strip())
