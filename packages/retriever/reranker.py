# packages/retriever/reranker.py
from __future__ import annotations

import json, random, time, uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from packages.core_config.config import load_yaml

# Optional: sentence-transformers cross-encoder
_HAS_ST = False
try:
    from sentence_transformers import CrossEncoder, InputExample, evaluation
    from torch.utils.data import DataLoader
    import torch
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# ------------------- IO helpers -------------------

def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    return rows

def _write_jsonl(p: Path, rows: List[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------- Configs -------------------

@dataclass
class RerankerMiningConfig:
    topk_candidates: int = 30
    negatives_per_pos: int = 4
    min_question_len: int = 8
    max_pairs: int = 2000
    seed: int = 123

@dataclass
class RerankerTrainConfig:
    epochs: int = 1
    batch_size: int = 16
    lr: float = 2.0e-5
    warmup_steps: int = 50
    eval_ratio: float = 0.1
    seed: int = 42

@dataclass
class RerankerConfig:
    enable: bool = True
    base_model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    base_model_local_dir: str = ""
    output_root: str = "data/reranker"
    mining: RerankerMiningConfig = field(default_factory=RerankerMiningConfig)
    train: RerankerTrainConfig = field(default_factory=RerankerTrainConfig)


# ------------------- Mining: QA -> pairs -------------------

def _normalize(s: str) -> str:
    return " ".join((s or "").split())

def _build_lookup(chunks_path: Path) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for r in _read_jsonl(chunks_path):
        lookup[r["id"]] = r
    return lookup

def mine_pairs_from_qa(
    doc_id: str,
    artifacts_root: Path,
    cfg: RerankerConfig,
    ann_top_k: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(cfg.mining.seed)
    art = Path(artifacts_root)

    qa_path = Path("data/datasets") / doc_id / "sft" / "qa.jsonl"
    chunks_path = art / "chunks" / f"{doc_id}.chunks.jsonl"
    ann_path = art / "index" / f"{doc_id}.ann.jsonl"  # optional

    qa = _read_jsonl(qa_path)
    lookup = _build_lookup(chunks_path)
    all_chunk_ids = list(lookup.keys())

    ann_map: Dict[str, List[str]] = {}
    if ann_path.exists():
        for r in _read_jsonl(ann_path):
            nbrs = r.get("neighbors") or []
            ann_map[r["id"]] = nbrs[: (ann_top_k or cfg.mining.topk_candidates)]

    pairs: List[Dict[str, Any]] = []
    for r in qa:
        q = _normalize(r.get("question") or "")
        if len(q) < cfg.mining.min_question_len:
            continue
        src_id = r.get("source_chunk_id")
        pos_text = _normalize((lookup.get(src_id) or {}).get("text") or "")
        if not pos_text:
            continue

        cand_ids = ann_map.get(src_id, [])
        if not cand_ids:
            cand_ids = rng.sample(all_chunk_ids, k=min(len(all_chunk_ids), cfg.mining.topk_candidates))
        cand_ids = [cid for cid in cand_ids if cid != src_id]

        neg_ids = rng.sample(cand_ids, k=min(len(cand_ids), cfg.mining.negatives_per_pos))
        neg_texts = [_normalize(lookup[n]["text"]) for n in neg_ids if lookup.get(n) and lookup[n].get("text")]

        pairs.append({
            "query": q,
            "positive": pos_text,
            "negatives": neg_texts,
            "source_chunk_id": src_id,
        })
        if len(pairs) >= cfg.mining.max_pairs:
            break

    n = len(pairs)
    m = max(1, int(n * cfg.train.eval_ratio)) if n > 0 else 0
    eval_pairs = pairs[:m] if m > 0 else []
    train_pairs = pairs[m:] if n > 0 else []
    return train_pairs, eval_pairs


# ------------------- Training -------------------

def _flatten_triplets_to_examples(
    pairs: List[Dict[str, Any]]
) -> List[InputExample]:
    exs: List[InputExample] = []
    for p in pairs:
        q = p["query"]
        pos = p["positive"]
        exs.append(InputExample(texts=[q, pos], label=1.0))
        for neg in p.get("negatives", []):
            exs.append(InputExample(texts=[q, neg], label=0.0))
    return exs

def train_reranker(
    doc_id: str,
    artifacts_root: Path,
    cfg: RerankerConfig,
) -> Dict[str, Any]:
    if not cfg.enable:
        return {"status": "DISABLED"}
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers not installed. `pip install sentence-transformers`")

    output_root = Path(cfg.output_root)
    run_dir = output_root / doc_id / uuid.uuid4().hex[:8]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cfg.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_pairs, eval_pairs = mine_pairs_from_qa(doc_id, artifacts_root, cfg)
    (run_dir / "train_pairs.jsonl").write_text("\n".join(json.dumps(p) for p in train_pairs), encoding="utf-8")
    (run_dir / "eval_pairs.jsonl").write_text("\n".join(json.dumps(p) for p in eval_pairs), encoding="utf-8")

    train_ex = _flatten_triplets_to_examples(train_pairs)
    eval_ex  = _flatten_triplets_to_examples(eval_pairs)

    if not train_ex:
        job = {
            "id": uuid.uuid4().hex[:8],
            "doc_id": doc_id,
            "created_at": int(time.time()),
            "status": "NO_DATA",
            "run_dir": str(run_dir),
            "model_path": "",
            "base_model": cfg.base_model_local_dir or cfg.base_model_id,
            "train_stats": {"train_pairs": 0, "eval_pairs": len(eval_ex)},
        }
        (run_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
        return job

    base = cfg.base_model_local_dir if cfg.base_model_local_dir else cfg.base_model_id
    model = CrossEncoder(base, num_labels=1, max_length=512)

    # ---- loss & evaluator (version-friendly) ----
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ce_eval = None
    if hasattr(evaluation, "CEBinaryClassificationEvaluator") and eval_ex:
        ce_eval = evaluation.CEBinaryClassificationEvaluator(eval_ex, name="dev")

    train_loader = DataLoader(train_ex, shuffle=True, batch_size=cfg.train.batch_size)

    model.fit(
        train_dataloader=train_loader,
        evaluator=ce_eval,
        epochs=cfg.train.epochs,
        warmup_steps=cfg.train.warmup_steps,
        optimizer_params={"lr": cfg.train.lr},
        output_path=str(run_dir / "model"),
        show_progress_bar=True,
        use_amp=False,
        loss_fct=loss_fn,
    )

    # Robust saving: try HF-style folder and always save raw state dict
    try:
        model.save(str(run_dir / "model"))  # sentence-transformers save
    except Exception:
        pass
    try:
        torch.save(model.model.state_dict(), run_dir / "weights.pt")
    except Exception:
        pass

    out_path = run_dir / "model"
    job = {
        "id": uuid.uuid4().hex[:8],
        "doc_id": doc_id,
        "created_at": int(time.time()),
        "status": "COMPLETED",
        "run_dir": str(run_dir),
        "model_path": str(out_path),
        "base_model": base,
        "train_stats": {
            "train_pairs": len(train_pairs),
            "eval_pairs": len(eval_pairs),
        },
    }
    (run_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job


# ------------------- Simple eval (MRR@10) -------------------

def eval_mrr_at_10(
    doc_id: str,
    artifacts_root: Path,
    model_path: Path,
) -> Dict[str, Any]:
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers not installed.")

    run_dir = Path(model_path).parent
    ep = run_dir / "eval_pairs.jsonl"
    if not ep.exists():
        return {"mrr@10": 0.0, "n": 0}

    eval_pairs = [json.loads(ln) for ln in ep.read_text(encoding="utf-8").splitlines()]
    if not eval_pairs:
        return {"mrr@10": 0.0, "n": 0}

    # Try to load directly; if it fails (older ST/HF), rebuild and load state_dict
    try:
        model = CrossEncoder(str(model_path), num_labels=1, max_length=512)
    except Exception:
        job_json = {}
        jj = run_dir / "job.json"
        if jj.exists():
            try:
                job_json = json.loads(jj.read_text(encoding="utf-8"))
            except Exception:
                pass
        base = job_json.get("base_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        model = CrossEncoder(base, num_labels=1, max_length=512)
        wts = run_dir / "weights.pt"
        if wts.exists():
            sd = torch.load(wts, map_location="cpu")
            try:
                model.model.load_state_dict(sd, strict=False)
            except Exception:
                # If strict load fails due to head shape, keep base weights (won't crash)
                pass

    ranks = []
    for p in eval_pairs:
        q = p["query"]
        cand = [p["positive"]] + p.get("negatives", [])
        pairs = [[q, c] for c in cand]
        scores = model.predict(pairs)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank = order.index(0) + 1  # positive is index 0
        ranks.append(1.0 / rank if rank <= 10 else 0.0)
    mrr = sum(ranks) / max(1, len(ranks))
    return {"mrr@10": mrr, "n": len(ranks)}
