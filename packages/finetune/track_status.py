# packages/finetune/track_status.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def read_status(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    state_path = run_dir / "job_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"job_state.json not found in {run_dir}")
    return json.loads(state_path.read_text(encoding="utf-8"))
