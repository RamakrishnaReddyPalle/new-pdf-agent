from __future__ import annotations
import json, re
from pathlib import Path
from typing import Literal, Dict, Any

try:
    import yaml
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

import requests


def _read_first_pages(pages_jsonl: Path, max_pages: int = 8, stop_on_contents: bool = True) -> str:
    if not pages_jsonl.exists():
        raise FileNotFoundError(pages_jsonl)
    texts = []
    with pages_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > max_pages:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pg = rec.get("page")
            tx = (rec.get("text") or "").strip()
            if not tx:
                continue
            tx = re.sub(r"[ \t]+", " ", tx)
            tx = re.sub(r"\n{3,}", "\n\n", tx)
            texts.append(f"# Page {pg}\n{tx}")
            if stop_on_contents and re.search(r"\b(table\s+of\s+)?contents\b", tx, flags=re.I):
                break
    return "\n\n".join(texts[:max_pages])


def _ollama_health(url: str) -> bool:
    try:
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=(10, 10))
        r.raise_for_status()
        return True
    except Exception:
        return False


def _ollama_warmup(url: str, model: str, connect_timeout: int = 30, read_timeout: int = 600) -> None:
    """
    Fast single-token generation to force model load/compile.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "warmup"}],
        "stream": True,
        "options": {"temperature": 0.0, "num_predict": 1},
    }
    try:
        with requests.post(
            f"{url.rstrip('/')}/api/chat",
            json=payload,
            timeout=(connect_timeout, read_timeout),
            stream=True,
        ) as r:
            r.raise_for_status()
            # consume a couple of deltas
            for _ in r.iter_lines(decode_unicode=True):
                break
    except Exception:
        # non-fatal; seeding call will still try
        pass


def _ollama_chat(
    url: str,
    model: str,
    sys_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1400,
    stream: bool = True,
    connect_timeout: int = 30,
    read_timeout: int = 600,
) -> str:
    """
    Chat with Ollama. Defaults to stream=True to avoid read timeouts.
    Uses (connect_timeout, read_timeout) tuple so first-token stalls don’t kill the call.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": bool(stream),
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    try:
        with requests.post(
            f"{url.rstrip('/')}/api/chat",
            json=payload,
            timeout=(connect_timeout, read_timeout),
            stream=bool(stream),
        ) as r:
            r.raise_for_status()

            if not stream:
                # Try JSON; if server actually streamed NDJSON despite stream=False, stitch it.
                try:
                    data = r.json()
                    msg = (data.get("message") or {})
                    return msg.get("content", data.get("response", "")) or ""
                except Exception:
                    text = r.text
                    chunks = []
                    for line in text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if obj.get("done"):
                            break
                        msg = obj.get("message") or {}
                        delta = msg.get("content") or obj.get("response") or ""
                        if delta:
                            chunks.append(delta)
                    return "".join(chunks) if chunks else text

            # streaming: stitch deltas
            out = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("done"):
                    break
                msg = obj.get("message") or {}
                delta = msg.get("content") or obj.get("response") or ""
                if delta:
                    out.append(delta)
            return "".join(out)

    except requests.exceptions.ReadTimeout:
        # one retry with non-stream + longer read timeout
        with requests.post(
            f"{url.rstrip('/')}/api/chat",
            json={**payload, "stream": False},
            timeout=(connect_timeout, max(read_timeout, 900)),
            stream=False,
        ) as r2:
            r2.raise_for_status()
            try:
                data = r2.json()
                msg = (data.get("message") or {})
                return msg.get("content", data.get("response", "")) or ""
            except Exception:
                return r2.text


def _company_chat(
    base_url: str,
    api_key: str,
    model: str,
    sys_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=(30, 180),
    )
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


SEED_SYSTEM = """You draft prompt packs for SFT data generation from PDFs (fee schedules, legal/medical docs).
Return STRICT YAML only. No prose. Include:
- detection: profiles with include_any keywords (e.g., fee_schedule, math_heavy, tables)
- prompts.qa: instruction strings per profile; emphasize citations, math awareness, units, and table-reading
- prompts.summary: concise but complete section summaries (keep formulas intact, do not invent data)
Ensure YAML parses cleanly.
"""

SEED_USER_TMPL = """Document ID: {doc_id}

Below are early pages (before/around contents). Infer terminology and structure cues.
Pages Sample:
---
{pages_text}
---

Return YAML with keys: detection, prompts.
- detection: {{"default":"generic","profiles":{{"fee_schedule":{{"include_any":[...]}},"math_heavy":{{"include_any":[...]}},"tables":{{"include_any":[...]}}}}}}
- prompts.qa: provide 'generic', 'fee_schedule', 'math_heavy', 'tables' strings.
- prompts.summary: same keys as qa.
Aim for question styles covering abbreviations, conversion factors, code→description lookups, and step-by-step fee calculations (keep arithmetic grounded by text)."""


def seed_prompt_pack(
    doc_id: str,
    artifacts_root: Path,
    *,
    provider: Literal["ollama", "company"] = "ollama",
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.1:8b-instruct-q5_1",
    timeout_connect: int = 30,
    timeout_read: int = 600,
    company_base_url: str = "",
    company_api_key: str = "",
    company_model: str = "company-small",
    max_pages: int = 8,
    stop_on_contents: bool = True,
    out_dir: Path = Path("configs/pipelines/prompts"),
) -> Path:
    pages_jsonl = artifacts_root / "md" / f"{doc_id}.pages.jsonl"
    sample = _read_first_pages(pages_jsonl, max_pages=max_pages, stop_on_contents=stop_on_contents)
    user_prompt = SEED_USER_TMPL.format(doc_id=doc_id, pages_text=sample)

    # health + warmup for Ollama
    if provider == "ollama":
        if not _ollama_health(ollama_url):
            raise RuntimeError(f"Ollama not reachable at {ollama_url}")
        _ollama_warmup(ollama_url, ollama_model, timeout_connect, timeout_read)
        text = _ollama_chat(
            ollama_url,
            ollama_model,
            SEED_SYSTEM,
            user_prompt,
            temperature=0.3,
            max_tokens=1400,
            stream=True,
            connect_timeout=timeout_connect,
            read_timeout=timeout_read,
        )
    else:
        text = _company_chat(
            company_base_url,
            company_api_key,
            company_model,
            SEED_SYSTEM,
            user_prompt,
            temperature=0.3,
            max_tokens=1400,
        )

    # parse or fallback
    try:
        data = yaml.safe_load(text) or {}
        assert "detection" in data and "prompts" in data
    except Exception:
        data = {
            "detection": {
                "default": "generic",
                "profiles": {
                    "fee_schedule": {"include_any": ["conversion factor", "relative value", "CPT", "modifier", "fee schedule"]},
                    "math_heavy": {"include_any": ["multiply", "percentage", "per unit", "fee is determined by", "RVU"]},
                    "tables": {"include_any": ["table", "column", "code", "description", "value"]},
                },
            },
            "prompts": {
                "qa": {
                    "generic": "Ask grounded questions and answer strictly from the provided text; include units and definitions if present.",
                    "fee_schedule": "Generate Q&A about conversion factors, RVUs, section-specific rules, and modifiers. For answers, compute simple fees if explicitly possible from text.",
                    "math_heavy": "Generate Q&A that walks step-by-step through formulas in the text (keep equations intact, no hallucination).",
                    "tables": "Generate Q&A that read table rows: code→description, value lookups, column meanings; cite column headers.",
                },
                "summary": {
                    "generic": "Summarize clearly with key points; keep any formulas verbatim.",
                    "fee_schedule": "Summarize ground rules, conversion factor usage per section, and exceptions.",
                    "math_heavy": "Summarize methods for computing fees; preserve LaTeX/inline math if present.",
                    "tables": "Summarize what the table contains (columns, usage) without recreating the whole table.",
                },
            },
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.yaml"
    out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # debug artifacts
    (out_dir / "seed.raw.txt").write_text(text, encoding="utf-8")
    (out_dir / "seed.request.json").write_text(
        json.dumps(
            {
                "provider": provider,
                "ollama_url": ollama_url,
                "ollama_model": ollama_model,
                "timeout_connect": timeout_connect,
                "timeout_read": timeout_read,
                "company_base_url": company_base_url,
                "company_model": company_model,
                "system": SEED_SYSTEM,
                "user": user_prompt,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def run(doc_id: str, artifacts_root: Path, providers_yaml: Path | None = None) -> Dict[str, Any]:
    """
    Read provider+model+timeouts from providers.yaml (sft.llm.*), seed prompts, and return paths.
    """
    to = {"connect": 30, "read": 600}
    model = "llama3.1:8b-instruct-q5_1"
    url = "http://127.0.0.1:11434"
    prov = "ollama"
    company = {}

    if providers_yaml and providers_yaml.exists():
        y = yaml.safe_load(providers_yaml.read_text(encoding="utf-8")) or {}
        sft_llm = (y.get("sft") or {}).get("llm") or {}
        prov = str(sft_llm.get("provider", "ollama")).lower()
        if prov == "ollama":
            o = (sft_llm.get("ollama") or {})
            url = str(o.get("url", url))
            model = str(o.get("model", model))
            to["connect"] = int(o.get("connect_timeout", to["connect"]))
            to["read"] = int(o.get("read_timeout", to["read"]))
        else:
            company = sft_llm.get("company") or {}

    if prov == "ollama":
        out = seed_prompt_pack(
            doc_id,
            artifacts_root,
            provider="ollama",
            ollama_url=url,
            ollama_model=model,
            timeout_connect=to["connect"],
            timeout_read=to["read"],
        )
    else:
        out = seed_prompt_pack(
            doc_id,
            artifacts_root,
            provider="company",
            company_base_url=str(company.get("base_url", "")),
            company_api_key=str(company.get("api_key", "")),
            company_model=str(company.get("model", "company-small")),
        )

    return {
        "doc_id": doc_id,
        "prompt_pack": str(out),
        "debug_seed_raw": str(Path(out).with_name("seed.raw.txt")),
        "debug_seed_request": str(Path(out).with_name("seed.request.json")),
    }
