# Optional local inference shim
# packages/providers/llm/hf_infer.py

from __future__ import annotations
from typing import Optional

def load_causal_pipeline(base_model_id_or_path: str, adapter_path: Optional[str] = None):
    """
    Return a callable generate(prompt: str) -> str.
    base_model_id_or_path can be a HF repo id or a local directory.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base_model_id_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_id_or_path)
    if adapter_path:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception:
            pass

    def _gen(prompt: str, max_new_tokens: int = 128, temperature: float = 0.2) -> str:
        import torch
        inputs = tok(prompt, return_tensors="pt")
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0], skip_special_tokens=True)
    return _gen

