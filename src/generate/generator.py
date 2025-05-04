from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
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
    """Generate text using passed-in model and tokenizer."""
    print(f"\nGenerating with prompt length: {len(prompt)}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Starting generation...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=1,
        repetition_penalty=1.1,
        length_penalty=1.0
    )
    print("Generation complete!")

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    return clean_response(response)

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
    """Generate both responses - with and without ad."""

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

    response_without_ad = generate_response_without_ad(user_query, model, tokenizer)
    response_with_ad = generate_response_with_ad(user_query, ad_text, model, tokenizer)

    return response_without_ad, response_with_ad