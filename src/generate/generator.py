# generator.py
from __future__ import annotations

import torch
from .prompts import get_prompt_with_ad, get_prompt_without_ad, get_prompt_with_multi_ads
from ..judge.utils import cache_result


def _is_sharded(model) -> bool:
    return hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict) and len(model.hf_device_map) > 1


def tokenize_for_generate(prompt: str, tokenizer, model):
    tok = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if not _is_sharded(model):
        tok = tok.to(model.device)
    return tok


def extract_final_answer(text: str) -> str:
    if "FINAL ANSWER:" in text:
        return text.split("FINAL ANSWER:", 1)[-1].strip()
    return text.strip()


@torch.no_grad()
def generate_text(prompt: str, model, tokenizer, max_new_tokens: int = 256) -> str:
    try:
        tokenized = tokenize_for_generate(prompt, tokenizer, model)

        # basic sanity only (input_ids are ints; don't use isnan)
        if (tokenized["input_ids"] < 0).any():
            print("⚠️ Invalid input_ids detected — skipping.")
            return ""

        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            top_p=0.9,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded
    except RuntimeError as e:
        print(f"🔥 Generation RuntimeError: {e}")
        return ""


def build_ad_text(ad_facts: dict) -> str:
    ad_product = (ad_facts.get("ad_product") or "").strip()
    brand = (ad_facts.get("brand") or "").strip()
    url = (ad_facts.get("url") or "").strip()
    description = (ad_facts.get("description") or ad_facts.get("ad_description") or "").strip()

    if ad_product:
        description = description.replace("[Product Name]", ad_product)
    if brand and description and brand.lower() not in description.lower():
        description = f"{brand} offers {description}"

    return f"""Product: {ad_product}
Brand: {brand}
URL: {url}
Description: {description}"""


@torch.no_grad()
def generate_response_without_ad(user_query: str, model, tokenizer) -> str:
    prompt = get_prompt_without_ad(user_query)
    text = generate_text(prompt, model, tokenizer)
    return extract_final_answer(text)


@torch.no_grad()
def generate_response_with_ad(user_query: str, ad_facts: dict, model, tokenizer) -> str:
    ad_text = build_ad_text(ad_facts)
    prompt = get_prompt_with_ad(user_query, ad_text)
    text = generate_text(prompt, model, tokenizer)
    return extract_final_answer(text)


@torch.no_grad()
def generate_response_with_multi_ads(user_query: str, multi_ad_block: str, model, tokenizer) -> str:
    prompt = get_prompt_with_multi_ads(user_query, multi_ad_block)
    text = generate_text(prompt, model, tokenizer)
    return extract_final_answer(text)


# IMPORTANT: cache only on (user_query, ad_facts), not model/tokenizer objects
@cache_result(ttl_seconds=3600)
def generate_responses_cached(user_query: str, ad_facts: dict) -> tuple[str, str]:
    """
    Cached wrapper. Caller should set model/tokenizer globally
    or call a non-cached function that passes model/tokenizer.
    """
    # We can't use model/tokenizer here safely in cache key.
    # This function should be used only if you provide a global singleton model/tokenizer elsewhere.
    raise RuntimeError(
        "Use generate_responses(user_query, ad_facts, model, tokenizer) "
        "and cache outside or refactor to use a global model/tokenizer."
    )


def generate_responses(user_query: str, ad_facts: dict, model, tokenizer) -> tuple[str, str]:
    """Non-cached, safe with passed model/tokenizer."""
    no_ad = generate_response_without_ad(user_query, model, tokenizer)
    with_ad = generate_response_with_ad(user_query, ad_facts, model, tokenizer)
    if not no_ad.strip() or not with_ad.strip():
        return "", ""
    return no_ad, with_ad


def clear_response_cache():
    from ..judge.utils import _judge_cache
    _judge_cache.clear()
    