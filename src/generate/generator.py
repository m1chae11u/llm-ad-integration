from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from .prompts import get_prompt_with_ad, get_prompt_without_ad, get_prompt_with_multi_ads
from tqdm import tqdm
from .baseline import generate_baseline_response
from ..judge.utils import cache_result

_model = None
_tokenizer = None


def extract_final_answer(response: str) -> str:
    """Extracts the text after 'FINAL ANSWER:' from the model's response."""
    if "FINAL ANSWER:" in response:
        return response.split("FINAL ANSWER:", 1)[-1].strip()
    # If the marker isn't found, return the original response, perhaps with a warning or log.
    # For now, returning the stripped original to avoid breaking if marker is missing.
    # print("Warning: 'FINAL ANSWER:' marker not found in response.") 
    return response.strip()


def get_token_count(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, truncation=False))

def truncate_ad_text(ad_text: str, tokenizer, max_tokens: int = 512) -> str:
    tokens = tokenizer.encode(ad_text, truncation=False)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return ad_text

def generate_text(prompt: str, model, tokenizer) -> str:
    """Generate response safely and skip any bad prompts that trigger CUDA asserts."""
    try:
        device = next(model.parameters()).device
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        prompt_len = tokenized.input_ids.shape[1]
        print(f"📏 Prompt length: {prompt_len} tokens")

        if torch.isnan(tokenized.input_ids).any() or (tokenized.input_ids < 0).any():
            print("⚠️ Invalid input_ids detected — skipping.")
            return ""

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **tokenized,
                    max_new_tokens=256,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            except RuntimeError as gen_err:
                if "probability tensor contains" in str(gen_err):
                    print("🔥 Skipping bad generation: probability tensor contained inf/nan/negative values.")
                    return ""
                raise gen_err

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ Generation complete.")
        return decoded

    except RuntimeError as e:
        print(f"🔥 Generation RuntimeError: {e}")
        return ""

def generate_response_without_ad(user_query: str, model, tokenizer) -> str:
    print("\n🟦 Generating response without ad...")
    prompt = get_prompt_without_ad(user_query)
    if get_token_count(prompt, tokenizer) > 2048:
        print("⚠️ Skipping: prompt too long (without ad)")
        return ""
    response = generate_text(prompt, model, tokenizer)
    return response

def generate_response_with_ad(user_query: str, ad_text: str, model, tokenizer) -> str:
    print("\n🟨 Generating response with ad...")
    base_prompt = get_prompt_without_ad(user_query)
    max_total_tokens = 2048
    max_ad_tokens = max_total_tokens - get_token_count(base_prompt, tokenizer)
    if max_ad_tokens <= 0:
        print("⚠️ Not enough space for ad content.")
        return ""
    safe_ad_text = truncate_ad_text(ad_text, tokenizer, max_ad_tokens)
    full_prompt = get_prompt_with_ad(user_query, safe_ad_text)
    if get_token_count(full_prompt, tokenizer) > max_total_tokens:
        print("⚠️ Skipping: full prompt too long (with ad)")
        return ""
    response = generate_text(full_prompt, model, tokenizer)
    return response

def generate_response_with_multi_ads(user_query: str, multi_ad_block: str, model, tokenizer) -> str:
    print("\n🟧 Generating response with MULTIPLE ads...")
    base_prompt = get_prompt_without_ad(user_query)
    max_total_tokens = 2048
    max_ad_tokens = max_total_tokens - get_token_count(base_prompt, tokenizer)
    if max_ad_tokens <= 0:
        print("⚠️ Not enough space for multi-ad content.")
        return ""
    # Truncate the multi_ad_block if needed
    tokens = tokenizer.encode(multi_ad_block, truncation=False)
    if len(tokens) > max_ad_tokens:
        multi_ad_block = tokenizer.decode(tokens[:max_ad_tokens], skip_special_tokens=True)
    full_prompt = get_prompt_with_multi_ads(user_query, multi_ad_block)
    if get_token_count(full_prompt, tokenizer) > max_total_tokens:
        print("⚠️ Skipping: full prompt too long (with multi ads)")
        return ""
    response = generate_text(full_prompt, model, tokenizer)
    return response

@cache_result(ttl_seconds=3600)  # Cache for 1 hour
def generate_responses(user_query: str, ad_facts: dict, model, tokenizer) -> tuple[str, str]:
    """Generate both responses - with and without ad, and clean them."""
    ad_product = ad_facts.get('ad_product', '').strip()
    brand = ad_facts.get('brand', '').strip()
    url = ad_facts.get('url', '').strip()
    description = ad_facts.get('description', '').replace("[Product Name]", ad_product).strip()

    if brand.lower() not in description.lower():
        description = f"{brand} offers {description}"

    ad_text = f"""Product: {ad_product}
Brand: {brand}
URL: {url}
Description: {description}"""

    raw_no_ad = generate_response_without_ad(user_query, model, tokenizer)
    raw_with_ad = generate_response_with_ad(user_query, ad_text, model, tokenizer)

    if not raw_no_ad.strip() or not raw_with_ad.strip():
        print("⚠️ Skipping: One or both responses were empty.")
        return "", ""

    cleaned_no_ad = extract_final_answer(raw_no_ad)
    cleaned_with_ad = extract_final_answer(raw_with_ad)

    return cleaned_no_ad, cleaned_with_ad

def clear_response_cache():
    """Clear the response cache used by generate_responses."""
    # The cache is stored in judge.utils._judge_cache
    from ..judge.utils import _judge_cache
    _judge_cache.clear()