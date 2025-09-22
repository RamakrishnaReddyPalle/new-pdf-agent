# packages/core-config/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable
import yaml

@dataclass
class PipelineConfig:
    data: Dict[str, Any]

    def get(self, path: str, default=None):
        """
        Dot-path accessor, e.g., cfg.get("ingest.ocr.enable", False)
        """
        cur: Any = self.data
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

def _load_one_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return y

def load_yaml(*paths: Iterable[str | Path]) -> PipelineConfig:
    """
    Shallow-merge multiple YAML files (later files override earlier ones).
    """
    merged: Dict[str, Any] = {}
    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            y = _load_one_yaml(p)
            merged.update(y)
    return PipelineConfig(merged)

def load_prompt(path: str, fallback: str = "") -> str:
    """
    Read a prompt text file. If it doesn't exist, return fallback.
    """
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else fallback
