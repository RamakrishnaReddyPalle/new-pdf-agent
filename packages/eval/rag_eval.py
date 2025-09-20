# packages/eval/rag_eval.py
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

from packages.sft.generate_pairs import _read_jsonl
from packages.chat.tools import Retriever, RetrieverConfig
from packages.providers.llm.ollama import OllamaLLM, OllamaConfig

# Reuse tiny metrics from closed_book
from packages.eval.closed_book import _f1, _rouge_l

@dataclass
class RAGEvalConfig:
    use_llm: bool = True
    # LLM provider
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest"
    llm_temperature: float = 0.2
    llm_max_new_tokens: int = 256
    connect_timeout: int = 30
    read_timeout: int = 600
    retries: int = 1

    # retriever settings
    persist_path: str = "data/artifacts"
    embed_model_or_path: str = "BAAI/bge-base-en-v1.5"
    device: str = "cpu"
    bge_use_prompt: bool = True
    top_k: int = 12
    rerank_top_k: int = 8
    return_top_k: int = 6
    reranker_model_path: Optional[str] = None

    # data
    max_questions: int = 50
    datasets_root: str = "data/datasets"

def _mk_llm(cfg: RAGEvalConfig) -> OllamaLLM:
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

def _mk_retriever(doc_id: str, cfg: RAGEvalConfig) -> Retriever:
    rc = RetrieverConfig(
        persist_path=cfg.persist_path,
        collection=doc_id,
        embed_model_or_path=cfg.embed_model_or_path,
        device=cfg.device,
        bge_use_prompt=cfg.bge_use_prompt,
        top_k=cfg.top_k,
        rerank_top_k=cfg.rerank_top_k,
        return_top_k=cfg.return_top_k,
        reranker_model_path=cfg.reranker_model_path,
    )
    return Retriever(doc_id, rc)

_SYSTEM = (
    "You must answer using ONLY the provided context. "
    "Cite sources as [1], [2], ... where appropriate. "
    "If the context is insufficient, say so briefly."
)

def run_rag_eval(doc_id: str, cfg: RAGEvalConfig) -> Dict[str, Any]:
    qa_path = Path(cfg.datasets_root) / doc_id / "sft" / "qa.jsonl"
    qa_rows = _read_jsonl(qa_path)
    if cfg.max_questions > 0:
        qa_rows = qa_rows[: cfg.max_questions]

    retr = _mk_retriever(doc_id, cfg)
    llm = _mk_llm(cfg) if cfg.use_llm else None
    if llm:
        llm.ensure_ready()

    n = 0
    f1_sum = 0.0
    rl_sum = 0.0
    recall_hits = 0

    for r in qa_rows:
        q = (r.get("question") or "").strip()
        gold = (r.get("answer") or "").strip()
        gold_chunk_id = r.get("source_chunk_id")
        if not q or not gold:
            continue

        # 1) retrieve
        snips = retr.search(q)

        # retrieval recall@K (does the gold chunk appear?)
        if gold_chunk_id and any(s.get("id") == gold_chunk_id for s in snips):
            recall_hits += 1

        # 2) build context
        ctx_blocks = []
        for i, s in enumerate(snips, 1):
            meta = s.get("metadata") or {}
            head = " > ".join(meta.get("heading_path") or []) or "(no heading)"
            page = meta.get("page")
            tag = f"[{i}] Page {page} — {head}".strip()
            ctx_blocks.append(f"{tag}\n{s['text']}")
        context = "\n\n".join(ctx_blocks)

        # 3) answer
        if llm:
            prompt = f"{_SYSTEM}\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
            pred = llm.generate(prompt).strip()
        else:
            # no-LLM baseline: choose first snippet’s first sentence as “answer”
            pred = (snips[0]["text"].split(".")[0] if snips else "")

        # 4) metrics
        f1_sum += _f1(gold, pred)
        rl_sum += _rouge_l(gold, pred)
        n += 1

    if n == 0:
        return {"n": 0, "f1": 0.0, "rougeL": 0.0, "retrieval_recall@K": 0.0}

    return {
        "n": n,
        "f1": f1_sum/n,
        "rougeL": rl_sum/n,
        f"retrieval_recall@{cfg.return_top_k}": recall_hits / n,
    }
