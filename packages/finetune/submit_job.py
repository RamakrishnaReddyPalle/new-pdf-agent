# packages/finetune/submit_job.py
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
from inspect import signature

from datasets import load_dataset, Dataset

# --- Config dataclass ---------------------------------------------------------

@dataclass
class FinetuneConfig:
    # backend
    backend: str = "local_peft"  # "local_peft" | "company" (future)
    enable_training: bool = False

    # IO
    output_root: str = "data/models"          # where job runs & adapters live
    datasets_root: str = "data/datasets"      # where SFT artifacts live (from Phase 2)
    doc_id: str = "DOC"

    # base model resolution (prefer local dir if present)
    base_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    base_model_local_dir: Optional[str] = None

    # LoRA params
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")

    # Train params
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 20
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    bf16: bool = False
    fp16: bool = False
    seed: int = 42


# --- Helpers ------------------------------------------------------------------

def _now_ts() -> int:
    return int(time.time())

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def _resolve_base_model(cfg: FinetuneConfig) -> str | Path:
    """
    Prefer a local dir if it exists; else return the HF repo id.
    """
    if cfg.base_model_local_dir:
        local = Path(cfg.base_model_local_dir)
        if local.exists() and any(local.iterdir()):
            return str(local)
    return cfg.base_model_id

def _format_row_alpaca(row: Dict[str, Any]) -> str:
    """
    Turn a combined.jsonl row into a single training string.
    Supports two kinds: "qa" and "summary".
    """
    kind = row.get("kind", "qa")
    if kind == "qa":
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        if inp:
            prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
            prompt = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
        return prompt

    # summary
    instr = (row.get("instruction") or "Summarize the following text.").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return prompt

def _build_text_dataset(combined_path: Path, seed: int) -> Dataset:
    """
    Load combined.jsonl (Phase 2 output) and create a HF Dataset with a 'text' column.
    """
    ds = load_dataset("json", data_files=str(combined_path), split="train")
    def to_text(example: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": _format_row_alpaca(example)}
    ds_text = ds.map(to_text, remove_columns=ds.column_names, desc="Formatting to single text")
    ds_text = ds_text.shuffle(seed=seed)
    return ds_text

def _split_train_eval(ds: Dataset, eval_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    n = len(ds)
    eval_size = max(1, int(n * eval_ratio)) if n > 10 else max(1, n // 10 or 1)
    ds_eval = ds.select(range(0, eval_size))
    ds_train = ds.select(range(eval_size, n))
    if len(ds_train) == 0:
        ds_train = ds_eval
    return ds_train, ds_eval

def _safe_training_arguments(**kwargs):
    """
    Build transformers.TrainingArguments while filtering kwargs
    that aren't supported in the local transformers version.
    """
    from transformers import TrainingArguments
    sig = signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**safe_kwargs)

# --- Job state ----------------------------------------------------------------

def _init_job_state(run_dir: Path, cfg: FinetuneConfig, base_model_resolved: str | Path) -> Dict[str, Any]:
    job = {
        "id": uuid.uuid4().hex[:8],
        "doc_id": cfg.doc_id,
        "created_at": _now_ts(),
        "status": "SUBMITTED",
        "backend": cfg.backend,
        "enable_training": cfg.enable_training,
        "base_model": str(base_model_resolved),
        "run_dir": str(run_dir),
        "artifacts": {},
        "config": asdict(cfg),
    }
    (run_dir / "job_state.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job

def _update_job_state(run_dir: Path, **patch):
    p = Path(run_dir) / "job_state.json"
    job = json.loads(p.read_text(encoding="utf-8"))
    job.update(patch)
    p.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job

# --- Public API ----------------------------------------------------------------

def submit_job(cfg: FinetuneConfig) -> Dict[str, Any]:
    """
    Submit a fine-tuning job (stub or real LoRA).
    - In stub mode (enable_training=False): create adapter/ folder without training.
    - In training mode: run a tiny LoRA loop and save the adapter.
    """
    output_root = Path(cfg.output_root)
    run_dir = _ensure_dir(output_root / cfg.doc_id / uuid.uuid4().hex[:8])

    # Resolve dataset paths
    combined = Path(cfg.datasets_root) / cfg.doc_id / "sft" / "combined.jsonl"
    if not combined.exists():
        raise FileNotFoundError(f"Missing combined.jsonl at {combined}. Did you run Phase 2?")

    # Snapshot configs
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (run_dir / "datasets.json").write_text(json.dumps({"combined": str(combined)}, indent=2), encoding="utf-8")

    # Resolve base model
    base_model = _resolve_base_model(cfg)

    # Init job
    job = _init_job_state(run_dir, cfg, base_model)

    # Always create adapter/ path (stub or real)
    adapter_dir = _ensure_dir(run_dir / "adapter")

    # Stub path (fast, no deps)
    if not cfg.enable_training:
        job = _update_job_state(run_dir, status="COMPLETED", artifacts={"adapter_path": str(adapter_dir)})
        return job

    # --- Real LoRA training path (requires transformers + peft + torch + datasets) ---
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model

        job = _update_job_state(run_dir, status="RUNNING")

        # 1) Build text dataset and split
        ds_text = _build_text_dataset(combined, seed=cfg.seed)
        ds_train, ds_eval = _split_train_eval(ds_text, eval_ratio=0.1, seed=cfg.seed)

        # 2) Tokenizer + model
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or tok.unk_token or "[PAD]"

        def tokenize_batch(examples: Dict[str, list[str]]) -> Dict[str, Any]:
            enc = tok(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_attention_mask=True,
            )
            enc["labels"] = enc["input_ids"].copy()
            return enc

        ds_train_tok = ds_train.map(
            tokenize_batch,
            batched=True,
            remove_columns=ds_train.column_names,
            desc="Tokenizing train",
        )
        ds_eval_tok = ds_eval.map(
            tokenize_batch,
            batched=True,
            remove_columns=ds_eval.column_names,
            desc="Tokenizing eval",
        )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # 3) Apply LoRA
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)

        # 4) Training args (version-safe) + collator
        args = _safe_training_arguments(
            output_dir=str(run_dir / "hf_out"),
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_steps=cfg.max_steps,
            logging_steps=max(1, cfg.max_steps // 10),
            save_steps=cfg.max_steps,  # save at end
            seed=cfg.seed,
            bf16=cfg.bf16,
            fp16=cfg.fp16,
            report_to=[],
            # The following may not exist in older transformers; _safe_training_arguments will drop them:
            evaluation_strategy="no",
            do_eval=False,
        )

        collator = DataCollatorForLanguageModeling(tok, mlm=False)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train_tok,
            eval_dataset=ds_eval_tok,  # will be ignored if args/do_eval aren't supported
            tokenizer=tok,
            data_collator=collator,
        )

        trainer.train()

        # 5) Save adapter
        model.save_pretrained(str(adapter_dir))
        job = _update_job_state(
            run_dir,
            status="COMPLETED",
            artifacts={"adapter_path": str(adapter_dir)},
        )
        return job

    except Exception as e:
        job = _update_job_state(run_dir, status="FAILED", error=str(e))
        raise
