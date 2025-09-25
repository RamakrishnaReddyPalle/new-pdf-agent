# scripts/runners/sft_gen_runner.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Iterable
import json
import shutil
import yaml

from packages.core_config.config import load_yaml
from packages.sft.generate_pairs import SFTGenConfig, generate_pairs


BAD_Q_PREFIXES = ("Summarize the key point", "Summarize the main point")
MIN_Q_LEN = 20
MIN_A_LEN = 30


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_bad_qa(rec: dict) -> bool:
    q = (rec.get("question") or rec.get("q") or "").strip()
    a = (rec.get("answer") or rec.get("a") or "").strip()
    if not q or not a:
        return True
    if any(q.startswith(p) for p in BAD_Q_PREFIXES):
        return True
    if len(q) < MIN_Q_LEN or len(a) < MIN_A_LEN:
        return True
    return False


def run(
    doc_id: str,
    artifacts_root: Path,
    providers_yaml: str | Path = "configs/providers.yaml",
    pipeline_yaml: str | Path = "configs/pipelines/generic_legal.yaml",
    *,
    prompts_yaml_override: Optional[Path] = None,
    max_qa_override: Optional[int] = None,
    max_summary_override: Optional[int] = None,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    YAML-driven SFT generation with optional overrides and debug traces.
    After generation, applies a simple QA quality filter and emits filtered files.
    """
    cfg = load_yaml(str(providers_yaml), str(pipeline_yaml))

    # choose prompt pack: explicit override > doc-specific > default
    prompts_dir = Path("configs/pipelines/prompts")
    doc_prompt = prompts_dir / f"{doc_id}.yaml"
    prompts_yaml_path = (
        prompts_yaml_override
        if prompts_yaml_override and Path(prompts_yaml_override).exists()
        else (doc_prompt if doc_prompt.exists() else Path(cfg.get("sft.generation.prompts_yaml", "configs/pipelines/prompts/default.yaml")))
    )

    gen_cfg = SFTGenConfig(
        max_qa=int(max_qa_override if max_qa_override is not None else cfg.get("sft.generation.max_qa", 200)),
        max_summary=int(max_summary_override if max_summary_override is not None else cfg.get("sft.generation.max_summary", 36)),
        seed=int(cfg.get("sft.generation.seed", 13)),
        min_chunk_chars=int(cfg.get("sft.generation.min_chunk_chars", 300)),
        max_chunk_chars=int(cfg.get("sft.generation.max_chunk_chars", 2200)),
        dedup_regex=r"\s+",

        use_llm=bool(cfg.get("sft.generation.use_llm", True)),
        llm_provider=str(cfg.get("sft.generation.llm_provider", "ollama")),
        llm_model=str(cfg.get("sft.generation.llm_model", "llama3.1:8b-instruct-q5_1")),
        llm_url=str(cfg.get("sft.generation.llm_url", "http://127.0.0.1:11434")),
        llm_temperature=float(cfg.get("sft.generation.llm_temperature", 0.2)),
        llm_max_new_tokens=int(cfg.get("sft.generation.llm_max_new_tokens", 320)),

        summary_source=str(cfg.get("sft.generation.summary_source", "chunks")),
        summary_target_len=int(cfg.get("sft.generation.summary_target_len", 1600)),

        prompts_yaml=prompts_yaml_path,
        profile_rules_from_prompts_yaml=bool(cfg.get("sft.generation.profile_rules_from_prompts_yaml", True)),
        profile_rules=(cfg.get("sft.generation.profile_rules", {}) or {}),

        datasets_root=Path(cfg.get("sft.generation.datasets_root", "data/datasets")),
    )

    # Debug setup
    if debug_dir is None:
        debug_dir = Path(artifacts_root) / "sft_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    # Save the exact config and prompt file we used
    (debug_dir / "sft_config.used.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "providers_yaml": str(providers_yaml),
                "pipeline_yaml": str(pipeline_yaml),
                "prompts_yaml": str(prompts_yaml_path),
                "gen_cfg": gen_cfg.__dict__ | {"prompts_yaml": str(gen_cfg.prompts_yaml)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        shutil.copyfile(prompts_yaml_path, debug_dir / f"prompt_pack.{Path(prompts_yaml_path).name}")
    except Exception:
        pass

    # Run generation
    outs = generate_pairs(doc_id=doc_id, artifacts_root=artifacts_root, cfg=gen_cfg)

    # --- Post-filter low-effort QA ---
    qa_path = Path(outs["qa"])
    qa_rows = _read_jsonl(qa_path)
    kept = [r for r in qa_rows if not _is_bad_qa(r)]
    dropped = len(qa_rows) - len(kept)
    qa_filt = qa_path.with_name(qa_path.stem + ".filtered.jsonl")
    _write_jsonl(qa_filt, kept)

    # Also mirror filtering into combined.jsonl (only for QA items)
    comb_path = Path(outs["combined"])
    comb_rows = _read_jsonl(comb_path)
    comb_kept = []
    for r in comb_rows:
        kind = r.get("kind") or "qa"
        if kind.lower().startswith("qa"):
            q = (r.get("instruction") or r.get("q") or "").strip()
            a = (r.get("output") or r.get("a") or "").strip()
            if any(q.startswith(p) for p in BAD_Q_PREFIXES) or len(q) < MIN_Q_LEN or len(a) < MIN_A_LEN:
                continue
        comb_kept.append(r)
    comb_filt = comb_path.with_name(comb_path.stem + ".filtered.jsonl")
    _write_jsonl(comb_filt, comb_kept)

    # Stats
    stats = {
        "qa_total": len(qa_rows),
        "qa_kept": len(kept),
        "qa_dropped": dropped,
        "combined_total": len(comb_rows),
        "combined_kept": len(comb_kept),
    }
    (debug_dir / "postfilter.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return {
        "doc_id": doc_id,
        "qa": str(qa_path),
        "qa_filtered": str(qa_filt),
        "summaries": str(outs["summaries"]),
        "combined": str(comb_path),
        "combined_filtered": str(comb_filt),
        "debug_dir": str(debug_dir),
        "stats": stats,
    }
