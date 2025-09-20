# packages/providers/llm/ollama.py
from __future__ import annotations
import time
import requests
from dataclasses import dataclass

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    temperature: float = 0.2
    max_new_tokens: int = 256
    connect_timeout: int = 30         # seconds to connect TCP
    read_timeout: int = 600           # seconds to wait for a full response
    warmup_prompt: str = "OK"
    healthcheck: bool = True
    warmup: bool = True
    retries: int = 2                  # simple retry on timeout

class OllamaLLM:
    def __init__(self, cfg: OllamaConfig):
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_new_tokens = cfg.max_new_tokens
        self.connect_timeout = cfg.connect_timeout
        self.read_timeout = cfg.read_timeout
        self.healthcheck_enabled = cfg.healthcheck
        self.warmup_enabled = cfg.warmup
        self.warmup_prompt = cfg.warmup_prompt
        self.retries = max(0, int(cfg.retries))

    # --- helpers ---
    def _tags(self) -> dict:
        r = requests.get(f"{self.base_url}/api/tags", timeout=(self.connect_timeout, self.read_timeout))
        r.raise_for_status()
        return r.json()

    def _has_model(self) -> bool:
        try:
            data = self._tags()
        except Exception:
            return False
        models = [m.get("name") for m in data.get("models", [])]
        return any(self.model.split(":")[0] in (m or "") for m in models)

    def ensure_ready(self):
        if not self.healthcheck_enabled and not self.warmup_enabled:
            return
        # basic server check
        try:
            _ = self._tags()
        except Exception as e:
            raise RuntimeError(
                f"Ollama not reachable at {self.base_url}. "
                f"Start it (e.g. `ollama serve`) and verify the port. Original: {e}"
            )
        # model availability hint
        if not self._has_model():
            # This call does NOT auto-pull to avoid hanging. Give a precise hint instead.
            raise RuntimeError(
                f"Ollama model '{self.model}' not found. "
                f"Run `ollama pull {self.model}` (or `ollama run {self.model}`) to download it first."
            )
        # warmup (first token can be slow on CPU)
        if self.warmup_enabled:
            try:
                _ = self.generate(self.warmup_prompt)  # short prompt; uses same timeouts
            except Exception:
                # don't hard fail on warmup—just proceed and let real call retry
                pass

    # --- main API expected by SFT ---
    def generate(self, prompt: str) -> str:
        last_exc = None
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        }
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
            except (requests.ReadTimeout, requests.ConnectTimeout) as e:
                last_exc = e
                # small backoff and try again
                time.sleep(1.0 + attempt * 1.0)
            except Exception:
                # no retry on other errors
                raise
        # if we exhausted retries:
        raise requests.ReadTimeout(
            f"Ollama timed out after {self.read_timeout}s (retries={self.retries}). "
            f"Try a smaller model, increase timeout, or lower max_new_tokens. Last error: {last_exc}"
        )
