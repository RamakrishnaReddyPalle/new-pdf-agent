# packages/eval/closed_book.py
from __future__ import annotations
import json, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

from packages.sft.generate_pairs import _read_jsonl  # reuse helper
from packages.providers.llm.ollama import OllamaLLM, OllamaConfig

# ---------- tiny text utils ----------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\.,;:!\?\(\)\[\]\{\}\"'`]+")

def _norm(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    return _norm(s).split()

def _f1(gold: str, pred: str) -> float:
    g = _tokenize(gold); p = _tokenize(pred)
    if not g and not p: return 1.0
    if not g or not p:  return 0.0
    commons = {}
    for t in g:
        commons[t] = commons.get(t, 0) + 1
    match = 0
    for t in p:
        if commons.get(t, 0) > 0:
            match += 1
            commons[t] -= 1
    if match == 0: return 0.0
    prec = match / max(1, len(p))
    rec  = match / max(1, len(g))
    return 2 * prec * rec / max(prec + rec, 1e-9)

def _lcs(a: List[str], b: List[str]) -> int:
    # O(n*m) DP is okay for short eval strings
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def _rouge_l(gold: str, pred: str) -> float:
    g = _tokenize(gold); p = _tokenize(pred)
    if not g or not p: return 0.0
    l = _lcs(g, p)
    prec = l / len(p)
    rec  = l / len(g)
    if prec+rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

# ---------- config ----------
@dataclass
class ClosedBookEvalConfig:
    use_llm: bool = True
    # LLM provider (Ollama for local; swap later)
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest"
    llm_temperature: float = 0.2
    llm_max_new_tokens: int = 256
    connect_timeout: int = 30
    read_timeout: int = 600
    retries: int = 1

    # data
    max_questions: int = 50              # eval subset for quick runs
    datasets_root: str = "data/datasets" # where qa.jsonl lives (Phase 2 outputs)

def _mk_llm(cfg: ClosedBookEvalConfig) -> OllamaLLM:
    oc = OllamaConfig(
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_new_tokens=cfg.llm_max_new_tokens,
        connect_timeout=cfg.connect_timeout,
        read_timeout=cfg.read_timeout,
        retries=cfg.retries,
        warmup=False,
        healthcheck=True,
        warmup_prompt="OK"
    )
    return OllamaLLM(oc)

# ---------- main ----------
def run_closed_book(doc_id: str, cfg: ClosedBookEvalConfig) -> Dict[str, Any]:
    qa_path = Path(cfg.datasets_root) / doc_id / "sft" / "qa.jsonl"
    qa_rows = _read_jsonl(qa_path)

    if cfg.max_questions > 0:
        qa_rows = qa_rows[: cfg.max_questions]

    llm = _mk_llm(cfg) if cfg.use_llm else None
    if llm:
        llm.ensure_ready()

    n = 0
    f1_sum = 0.0
    rl_sum = 0.0
    for r in qa_rows:
        q = (r.get("question") or "").strip()
        gold = (r.get("answer") or "").strip()
        if not q or not gold:
            continue

        if llm:
            prompt = f"Question: {q}\nAnswer briefly and precisely."
            pred = llm.generate(prompt).strip()
        else:
            # no-LLM baseline: empty (forces low score; just a control)
            pred = ""

        f1_sum += _f1(gold, pred)
        rl_sum += _rouge_l(gold, pred)
        n += 1

    if n == 0:
        return {"n": 0, "f1": 0.0, "rougeL": 0.0}

    return {"n": n, "f1": f1_sum/n, "rougeL": rl_sum/n}
