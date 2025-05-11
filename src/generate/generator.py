from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from .prompts import get_prompt_with_ad, get_prompt_without_ad
from tqdm import tqdm
from .baseline import generate_baseline_response

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("\nLoading DeepSeek model and tokenizer...")
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded")
    return _model, _tokenizer


def clean_response(response: str) -> str:
    """Clean up the response by removing thinking processes and other unwanted content."""
    response = re.sub(r'^.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'(?:Let me|I\'ll|I will|First|Next|Then|Finally).*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)
    response = re.sub(r'(?:As an AI|I am an AI|I\'m an AI|As a language model).*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)
    if "FINAL ANSWER:" in response:
        response = response.split("FINAL ANSWER:")[-1]
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

    print(f"\n🧾 Raw response WITHOUT ad: {repr(raw_no_ad)}")
    print(f"🧾 Raw response WITH ad: {repr(raw_with_ad)}")

    if not raw_no_ad.strip() or not raw_with_ad.strip():
        print("⚠️ Skipping: One or both responses were empty.")
        return "", ""

    return clean_response(raw_no_ad), clean_response(raw_with_ad)