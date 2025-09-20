# packages/finetune/register_model.py

from __future__ import annotations
import json, time, uuid
from pathlib import Path
from typing import Dict, Any, Optional

REGISTRY_PATH = Path("data/registry/models.jsonl")

def register_model(
    doc_id: str,
    base_model: str,
    adapter_path: str | Path | None,
    run_dir: str | Path,
    profile: str = "generic",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Append a single registry record; returns the record.
    Your chat router can read this registry to mount doc profiles -> model ids.
    """
    rec = {
        "id": str(uuid.uuid4())[:8],
        "doc_id": doc_id,
        "profile": profile,
        "base_model": base_model,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "run_dir": str(run_dir),
        "created_at": int(time.time()),
        "extra": extra or {},
    }
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

