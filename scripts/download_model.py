# scripts/download_model.py
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

try:
    from packages.core_config.config import load_yaml
except Exception:
    import yaml
    class _Cfg:
        def __init__(self, d): self.data = d
        def get(self, path, default=None):
            cur = self.data
            for p in path.split("."):
                if not isinstance(cur, dict) or p not in cur: return default
                cur = cur[p]
            return cur
    def load_yaml(*paths):
        merged = {}
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                merged.update(yaml.safe_load(f) or {})
        return _Cfg(merged)

def sh(*cmd):
    print("$", " ".join(cmd))
    return subprocess.call(cmd)

def pull_ollama(tag: str):
    code = sh("ollama", "pull", tag)
    if code != 0:
        sys.exit(code)

def hf_snapshot(repo: str, dest: Path, allow_patterns=None):
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        sh(sys.executable, "-m", "pip", "install", "-U", "huggingface_hub")
        from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns or None,
        ignore_patterns=["*.msgpack"],
    )
    print(f"[OK] downloaded {repo} → {dest}")

def resolve_from_yaml(cfg, role: str):
    base = f"chat.models.{role}"
    provider = cfg.get(base + ".provider", "ollama")
    model = cfg.get(base + ".model", "")
    # Optional explicit repos (only needed for non-ollama)
    repo_tf  = cfg.get(base + ".hf_repo_transformers", "")
    repo_gg  = cfg.get(base + ".hf_repo_gguf", "")
    return provider, model, repo_tf, repo_gg

def main():
    ap = argparse.ArgumentParser(description="Download model for a chat role from YAML")
    ap.add_argument("--role", choices=["intro","splitter","core","output"], default="core")
    ap.add_argument("--providers", default="configs/providers.yaml")
    ap.add_argument("--mode", choices=["auto","ollama","transformers","gguf"], default="auto")
    ap.add_argument("--dest", default="models/llm")
    args = ap.parse_args()

    cfg = load_yaml(args.providers)
    provider, model, repo_tf, repo_gg = resolve_from_yaml(cfg, args.role)

    mode = args.mode
    if mode == "auto":
        # infer from provider
        if provider == "ollama":
            mode = "ollama"
        elif provider in ("hf_transformers", "transformers"):
            mode = "transformers"
        elif provider in ("hf_gguf", "gguf"):
            mode = "gguf"
        else:
            print(f"Unknown provider='{provider}', defaulting to ollama")
            mode = "ollama"

    if mode == "ollama":
        assert model, "Model tag missing in YAML"
        pull_ollama(model)
        return

    dest = Path(args.dest)
    if mode == "transformers":
        repo = repo_tf or {
            "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
        }.get(model, "")
        assert repo, f"Missing HF repo for transformers mode (role={args.role}, model={model})"
        hf_snapshot(repo, dest / repo.split("/")[-1],
                    allow_patterns=["*.json","*.safetensors","*.py","*.md","*.txt"])
        return

    if mode == "gguf":
        repo = repo_gg or {
            "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        }.get(model, "")
        assert repo, f"Missing HF repo for GGUF mode (role={args.role}, model={model})"
        hf_snapshot(repo, dest / repo.split("/")[-1])
        return

if __name__ == "__main__":
    main()

