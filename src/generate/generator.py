from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import gc
from .prompts import get_prompt_with_ad, get_prompt_without_ad

def clean_response(response: str) -> str:
    """Clean up the response by removing thinking processes and other unwanted content."""
    response = re.sub(r'^.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'(?:Let me|I\'ll|I will|First|Next|Then|Finally).*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)
    response = re.sub(r'(?:As an AI|I am an AI|I\'m an AI|As a language model).*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)
    if "FINAL ANSWER:" in response:
        response = response.split("FINAL ANSWER:")[-1]
    return response.strip()

def generate_text(prompt: str, model, tokenizer) -> str:
    """Generate response safely and skip any bad prompts that trigger CUDA asserts."""
    try:
        device = next(model.parameters()).device
        print(f"Generating with prompt length: {len(prompt)}")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Additional safety checks
        if torch.isnan(inputs.input_ids).any() or (inputs.input_ids < 0).any():
            print("⚠️ Invalid input_ids detected — skipping.")
            return ""

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            except RuntimeError as gen_err:
                if "probability tensor contains" in str(gen_err):
                    print("🔥 Skipping bad generation: probability tensor contained inf/nan/negative values.")
                    return ""
                else:
                    raise gen_err

        print("Generation complete!")
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except RuntimeError as e:
        print(f"🔥 Generation RuntimeError: {e}")
        return ""
    

def generate_response_without_ad(user_query: str, model, tokenizer) -> str:
    print("\nGenerating response without ad...")
    prompt = get_prompt_without_ad(user_query)
    response = generate_text(prompt, model, tokenizer)
    print("Response without ad generated successfully!")
    return response

def generate_response_with_ad(user_query: str, ad_text: str, model, tokenizer) -> str:
    print("\nGenerating response with ad...")
    prompt = get_prompt_with_ad(user_query, ad_text)
    response = generate_text(prompt, model, tokenizer)
    print("Response with ad generated successfully!")
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

    raw_response_without_ad = generate_response_without_ad(user_query, model, tokenizer)
    raw_response_with_ad = generate_response_with_ad(user_query, ad_text, model, tokenizer)

    cleaned_without_ad = clean_response(raw_response_without_ad)
    cleaned_with_ad = clean_response(raw_response_with_ad)

    return cleaned_without_ad, cleaned_with_ad