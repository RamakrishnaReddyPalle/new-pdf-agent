# packages/sft/generate_pairs.py
from __future__ import annotations

import json, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from packages.sft.prompt_registry import PromptRenderer, PromptPack, detect_profile_from_rules

# Optional Ollama provider; you can replace with company API later.
try:
    from packages.providers.llm.ollama import OllamaLLM, OllamaConfig
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False


@dataclass
class SFTGenConfig:
    # Sampling
    max_qa: int = 100
    max_summary: int = 20
    seed: int = 13

    # Passage selection
    min_chunk_chars: int = 300
    max_chunk_chars: int = 1800
    dedup_regex: Optional[str] = r"\s+"

    # LLM provider toggle
    use_llm: bool = False
    llm_provider: str = "ollama"        # "ollama" | "company"
    llm_model: str = "llama3.2:latest"
    llm_url: str = "http://localhost:11434"
    llm_temperature: float = 0.4
    llm_max_new_tokens: int = 256
    # Optional timeouts/retries (safe defaults; no YAML required)
    llm_connect_timeout: int = 30
    llm_timeout_sec: int = 600
    llm_retries: int = 1
    llm_healthcheck: bool = True
    llm_warmup: bool = True

    # Summaries
    summary_source: str = "nodes"       # "nodes" | "chunks"
    summary_target_len: int = 1200

    # Prompts + profile detection
    prompts_yaml: Optional[Path] = None # e.g., Path("configs/pipelines/prompts/default.yaml")
    profile_rules_from_prompts_yaml: bool = True
    profile_rules: Dict[str, Any] = None  # optional overrides (merged)

    # Paths
    datasets_root: Path = Path("data/datasets")


# ------------------------- IO helpers -------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
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

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _normalize_whitespace(s: str, dedup_regex: Optional[str]) -> str:
    if not s:
        return ""
    if dedup_regex:
        s = re.sub(dedup_regex, " ", s).strip()
    return s


# ------------------------- Passage selection -------------------------

def _choose_passages(chunks_path: Path, cfg: SFTGenConfig, k: int) -> List[Dict[str, Any]]:
    rng = random.Random(cfg.seed)
    chunks = _read_jsonl(chunks_path)
    keep = []
    for r in chunks:
        txt = _normalize_whitespace(r.get("text") or "", cfg.dedup_regex)
        if not txt:
            continue
        if not (cfg.min_chunk_chars <= len(txt) <= cfg.max_chunk_chars):
            continue
        btype = (r.get("metadata") or {}).get("block_type") or ""
        if btype in ("heading", "image"):
            continue
        rr = dict(r)
        rr["text"] = txt
        keep.append(rr)
    rng.shuffle(keep)
    return keep[:k]


# ------------------------- Rule-based fallbacks -------------------------

def _fallback_qa(text: str, max_pairs: int = 3) -> List[Dict[str, str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pairs: List[Dict[str, str]] = []
    for ln in lines:
        if ":" in ln and len(ln) > 12:
            parts = ln.split(":", 1)
            q = f"What is {parts[0].strip()}?"
            a = parts[1].strip()
            if len(a) > 3:
                pairs.append({"question": q, "answer": a})
        elif " - " in ln and len(ln) > 12:
            parts = ln.split(" - ", 1)
            q = f"What about {parts[0].strip()}?"
            a = parts[1].strip()
            if len(a) > 3:
                pairs.append({"question": q, "answer": a})
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        first = text.split(".")[0]
        if len(first) > 20:
            pairs.append({"question": "Summarize the key point.", "answer": first.strip()})
    return pairs

def _fallback_summary(text: str, target_len: int) -> str:
    return _normalize_whitespace(text, r"\s+")[:target_len]


# ------------------------- LLM prompts (templated) -------------------------

def _qa_pairs_with_llm(text: str, pack: PromptPack, renderer: PromptRenderer, llm) -> List[Dict[str, str]]:
    sys = pack.qa_system
    user = renderer.render(pack.qa_user_template, passage=text)
    prompt = f"{sys}\n\n{user}"
    out = llm.generate(prompt)
    # try to read JSON array
    try:
        pairs = json.loads(out)
        if isinstance(pairs, list):
            return [{"question": (p.get("question","")).strip(), "answer": (p.get("answer","")).strip()} for p in pairs][:4]
    except Exception:
        pass
    # try to find JSON array embedded
    m = re.search(r"\[.*\]", out, flags=re.DOTALL)
    if m:
        try:
            pairs = json.loads(m.group(0))
            if isinstance(pairs, list):
                return [{"question": (p.get("question","")).strip(), "answer": (p.get("answer","")).strip()} for p in pairs][:4]
        except Exception:
            pass
    # fallback
    return _fallback_qa(text)

def _summary_with_llm(text: str, pack: PromptPack, renderer: PromptRenderer, llm) -> str:
    sys = pack.summary_system
    user = renderer.render(pack.summary_user_template, section=text)
    prompt = f"{sys}\n\n{user}"
    out = llm.generate(prompt)
    out = _normalize_whitespace(out, r"\s+")
    return out if out else _fallback_summary(text, 1200)


# ------------------------- Profile rules loading -------------------------

def _load_rules_from_prompts_yaml(prompts_yaml: Path) -> Dict[str, Any]:
    """
    Accept a few likely shapes. We normalize to:
      {"default": <label>, "profiles": {...}}
    """
    import yaml
    try:
        data = yaml.safe_load(prompts_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"default": "generic", "profiles": {}}

    if "profiles" in data or "default" in data:
        return {
            "default": data.get("default", "generic"),
            "profiles": data.get("profiles", {}),
        }
    # Some packs may nest rules under 'profile_rules'
    pr = data.get("profile_rules") or {}
    return {
        "default": pr.get("default", "generic"),
        "profiles": pr.get("profiles", {}),
    }


# ------------------------- Public API -------------------------

def generate_pairs(
    doc_id: str,
    artifacts_root: Path,
    cfg: SFTGenConfig,
) -> Dict[str, Path]:
    """
    Outputs:
      datasets/{doc_id}/sft/qa.jsonl
      datasets/{doc_id}/sft/summaries.jsonl
      datasets/{doc_id}/sft/combined.jsonl
    Prompts/profile are selected via YAML packs + detection rules.
    """
    art = Path(artifacts_root)
    ds_dir = cfg.datasets_root / doc_id / "sft"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load prompt pack & detect profile (safe defaults)
    renderer: Optional[PromptRenderer] = None
    pack: Optional[PromptPack] = None
    profile_label = "generic"

    # Use the first ~10 chunks concatenated to detect profile
    chunks_path = art / "chunks" / f"{doc_id}.chunks.jsonl"
    chunks = _read_jsonl(chunks_path)
    sample_text = " ".join((c.get("text") or "") for c in chunks[:10])[:8000]

    if cfg.prompts_yaml:
        py = Path(cfg.prompts_yaml)
        renderer = PromptRenderer(py)
        # Merge rules from prompts YAML (if requested) and explicit overrides
        rules = {"default": "generic", "profiles": {}}
        if cfg.profile_rules_from_prompts_yaml:
            try:
                from_rules = _load_rules_from_prompts_yaml(py)
                rules.update({**rules, **from_rules})
            except Exception:
                pass
        if cfg.profile_rules:
            # shallow merge; user overrides win
            base = rules.get("profiles", {})
            base.update(cfg.profile_rules or {})
            rules["profiles"] = base
        profile_label = detect_profile_from_rules(sample_text, rules, default_label=rules.get("default", "generic"))
        pack = renderer.pack_for(profile_label)

    # 2) LLM provider (optional)
    llm = None
    if cfg.use_llm:
        if cfg.llm_provider == "ollama":
            if not _HAS_OLLAMA:
                raise RuntimeError("Ollama provider not available; ensure module import works.")
            ocfg = OllamaConfig(
                base_url=cfg.llm_url,
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
                max_new_tokens=cfg.llm_max_new_tokens,
                connect_timeout=cfg.llm_connect_timeout,
                read_timeout=cfg.llm_timeout_sec,
                retries=cfg.llm_retries,
                healthcheck=cfg.llm_healthcheck,
                warmup=cfg.llm_warmup,
            )
            llm = OllamaLLM(ocfg)
            try:
                llm.ensure_ready()
            except Exception as e:
                # Fall back to heuristics if LLM not ready
                print(f"[SFT] Ollama not ready: {e}. Falling back to heuristic generation.")
                llm = None
        else:
            # Placeholder for company API integration later.
            raise NotImplementedError(f"LLM provider '{cfg.llm_provider}' not implemented.")

    # 3) QA from passages
    passages = _choose_passages(chunks_path, cfg, k=max(1, cfg.max_qa // 2))
    qa_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(passages, 1):
        passage = r["text"]
        if cfg.use_llm and renderer and pack and llm:
            pairs = _qa_pairs_with_llm(passage, pack, renderer, llm)
        else:
            pairs = _fallback_qa(passage)

        source_id = r.get("id")
        meta = r.get("metadata") or {}
        for j, qa in enumerate(pairs, 1):
            qa_rows.append({
                "id": f"{doc_id}-qa-{i:04d}-{j}",
                "doc_id": doc_id,
                "source_chunk_id": source_id,
                "page": meta.get("page"),
                "question": (qa.get("question") or "").strip(),
                "answer": (qa.get("answer") or "").strip(),
                "source_text": passage[:2000],
                "profile": profile_label,
            })

    _write_jsonl(ds_dir / "qa.jsonl", qa_rows)

    # 4) Summaries from nodes/chunks
    if cfg.summary_source == "nodes":
        src = art / "graph" / "node_texts.jsonl"
        nodes = _read_jsonl(src)
        items = [{"id": n.get("node_id"), "text": n.get("text") or ""} for n in nodes if (n.get("text") or "").strip()]
    else:
        items = [{"id": c.get("id"), "text": c.get("text") or ""} for c in chunks]

    items = [it for it in items if len(it["text"]) > 200]
    random.Random(cfg.seed).shuffle(items)
    items = items[: cfg.max_summary]

    summaries: List[Dict[str, Any]] = []
    for k, it in enumerate(items, 1):
        text = _normalize_whitespace(it["text"], cfg.dedup_regex)
        if cfg.use_llm and renderer and pack and llm:
            s = _summary_with_llm(text, pack, renderer, llm)
        else:
            s = _fallback_summary(text, cfg.summary_target_len)

        summaries.append({
            "id": f"{doc_id}-sum-{k:04d}",
            "doc_id": doc_id,
            "source_id": it["id"],
            "summary": s,
            "source_text": text[:3000],
            "profile": profile_label,
        })

    _write_jsonl(ds_dir / "summaries.jsonl", summaries)

    # 5) Combined format (for Alpaca packaging later)
    combined = []
    for r in qa_rows:
        combined.append({
            "kind": "qa",
            "instruction": r["question"],
            "input": "",
            "output": r["answer"],
            "meta": {k: r.get(k) for k in ("doc_id","source_chunk_id","page","profile")}
        })
    for r in summaries:
        combined.append({
            "kind": "summary",
            "instruction": "Summarize the following section.",
            "input": r["source_text"][:1500],
            "output": r["summary"],
            "meta": {k: r.get(k) for k in ("doc_id","source_id","profile")}
        })
    _write_jsonl(ds_dir / "combined.jsonl", combined)

    return {
        "qa": ds_dir / "qa.jsonl",
        "summaries": ds_dir / "summaries.jsonl",
        "combined": ds_dir / "combined.jsonl",
    }
