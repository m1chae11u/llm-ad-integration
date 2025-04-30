from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .prompts import get_prompt_with_ad, get_prompt_without_ad

print("\nLoading DeepSeek model and tokenizer...")
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def generate_text(prompt: str) -> str:
    """Generate text using local DeepSeek model."""
    print(f"\nGenerating with prompt length: {len(prompt)}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from the response
    response = response[len(prompt):].strip()
    return response

def generate_response_without_ad(user_query: str) -> str:
    prompt = get_prompt_without_ad(user_query)
    response = generate_text(prompt)
    if "FINAL ANSWER:" in response:
        return response.split("FINAL ANSWER:")[-1].strip()
    return response.strip()

def generate_response_with_ad(user_query: str, ad_text: str) -> str:
    prompt = get_prompt_with_ad(user_query, ad_text)
    response = generate_text(prompt)
    if "FINAL ANSWER:" in response:
        return response.split("FINAL ANSWER:")[-1].strip()
    return response.strip()

def generate_responses(user_query: str, ad_facts: dict) -> tuple[str, str]:
    """Generate both responses - with and without ad."""
    # Format ad text from facts
    ad_text = f"""Product: {ad_facts['ad_product']}
Brand: {ad_facts['brand']}
URL: {ad_facts['url']}
Description: {ad_facts['description']}"""

    # Generate both responses
    print("\nGenerating response without ad...")
    response_without_ad = generate_response_without_ad(user_query)
    print("\nGenerating response with ad...")
    response_with_ad = generate_response_with_ad(user_query, ad_text)
    
    return response_without_ad, response_with_ad