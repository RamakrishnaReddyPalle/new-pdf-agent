# packages/sft/prompt_registry.py
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml  # pyyaml
except Exception as e:
    raise RuntimeError("pyyaml is required to load prompt packs") from e

# Optional templating via jinja2 (preferred). Fallback to str.format if missing.
try:
    import jinja2
    _HAS_JINJA = True
except Exception:
    _HAS_JINJA = False


@dataclass
class PromptPack:
    # fully rendered prompt strings; templates are in the YAML and resolved by PromptRenderer
    qa_system: str
    qa_user_template: str
    summary_system: str
    summary_user_template: str


class PromptRenderer:
    """
    Loads YAML prompt packs and renders templates with variables.
    Supports per-profile overrides: default <- profile override.
    """
    def __init__(self, yaml_path: Path):
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Prompt YAML not found: {self.yaml_path}")
        with self.yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self.default = cfg.get("default") or {}
        self.profiles = cfg.get("profiles") or {}

        # sanity fill
        for scope in (self.default,):
            scope.setdefault("qa", {})
            scope.setdefault("summary", {})
            scope["qa"].setdefault("system", "You are a helpful assistant.")
            scope["qa"].setdefault("user_template", "Passage:\n{passage}\nReturn JSON Q&A.")
            scope["summary"].setdefault("system", "You are a helpful assistant.")
            scope["summary"].setdefault("user_template", "Section:\n{section}\nWrite a concise summary.")

    def pack_for(self, profile: Optional[str]) -> PromptPack:
        base = self.default
        if profile and profile in self.profiles:
            # shallow merge default<-profile
            prof = self.profiles[profile]
            qa = dict(base.get("qa", {})); qa.update(prof.get("qa", {}))
            sm = dict(base.get("summary", {})); sm.update(prof.get("summary", {}))
        else:
            qa = dict(base.get("qa", {}))
            sm = dict(base.get("summary", {}))
        return PromptPack(
            qa_system=qa.get("system", ""),
            qa_user_template=qa.get("user_template", ""),
            summary_system=sm.get("system", ""),
            summary_user_template=sm.get("user_template", ""),
        )

    def render(self, template: str, **vars) -> str:
        if _HAS_JINJA:
            env = jinja2.Environment(autoescape=False, undefined=jinja2.StrictUndefined)
            return env.from_string(template).render(**vars)
        # fallback: {var} style
        return template.format(**vars)


def detect_profile_from_rules(
    chunks_sample_text: str,
    rules: Dict[str, Any],
    default_label: str = "generic",
) -> str:
    """
    Very light heuristic. Example rules:
      detection:
        default: generic
        profiles:
          fee_schedule:
            include_any: ["CPT", "Relative Value", "Modifier 51"]
            exclude_any: ["Income Tax Act"]
          tax_code:
            include_any: ["Section", "sub-section", "Schedule"]
    """
    if not rules:
        return default_label
    default_label = rules.get("default", default_label)
    profiles = rules.get("profiles") or {}
    text = (chunks_sample_text or "").lower()

    def hit_any(keywords):
        for k in (keywords or []):
            if k.lower() in text:
                return True
        return False

    for label, rule in profiles.items():
        inc = rule.get("include_any") or []
        exc = rule.get("exclude_any") or []
        if inc and not hit_any(inc):
            continue
        if exc and hit_any(exc):
            continue
        return label
    return default_label
