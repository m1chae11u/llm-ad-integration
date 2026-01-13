# baseline.py
from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_MODEL = None
_TOKENIZER = None


def _is_sharded(model) -> bool:
    return hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict) and len(model.hf_device_map) > 1


def _tokenize_for_generate(prompt: str, tokenizer, model):
    tok = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    # If model is not sharded, we can safely move inputs to model.device
    if not _is_sharded(model):
        tok = tok.to(model.device)
    return tok


def get_baseline_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    """Load once per process."""
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _MODEL.eval()

        if _TOKENIZER.pad_token_id is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token

    return _MODEL, _TOKENIZER


def extract_final_answer(text: str) -> str:
    if "FINAL ANSWER:" in text:
        return text.split("FINAL ANSWER:", 1)[-1].strip()
    return text.strip()


@torch.no_grad()
def generate_baseline_response(query: str, ad_facts: dict) -> str:
    """
    Baseline response WITH ad (no training).
    Uses FINAL ANSWER marker to match prompts.py style.
    """
    model, tokenizer = get_baseline_model()

    # support both keys
    desc = ad_facts.get("description") or ad_facts.get("ad_description") or ""

    prompt = f"""You are a helpful assistant.

User Query: {query}

Naturally incorporate this product info once, only if it supports the user:
Product: {ad_facts.get('ad_product','')}
Brand: {ad_facts.get('brand','')}
Description: {desc}
URL: {ad_facts.get('url','')}

FINAL ANSWER:
"""

    inputs = _tokenize_for_generate(prompt, tokenizer, model)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_final_answer(decoded)


def clear_baseline_model_cache():
    """Free RAM/VRAM for a fresh start."""
    global _MODEL, _TOKENIZER
    _MODEL = None
    _TOKENIZER = None
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()