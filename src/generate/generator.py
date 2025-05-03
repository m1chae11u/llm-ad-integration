from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from .prompts import get_prompt_with_ad, get_prompt_without_ad
from tqdm import tqdm

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
    # If <think> is present, remove everything before and including it
    response = re.sub(r'^.*?<think>', '', response, flags=re.DOTALL)

    # Remove any content between <think> and </think> (if it wasn't already stripped above)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Remove any content that looks like thinking/planning
    response = re.sub(r'(?:Let me|I\'ll|I will|First|Next|Then|Finally).*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)

    # Remove any system messages or AI disclaimers
    response = re.sub(r'(?:As an AI|I am an AI|I\'m an AI|As a language model).*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)

    # If "FINAL ANSWER:" exists, keep only what's after it
    if "FINAL ANSWER:" in response:
        response = response.split("FINAL ANSWER:")[-1]

    # Clean up whitespace
    return response.strip()



def generate_text(prompt: str) -> str:
    """Generate text using local DeepSeek model."""
    print(f"\nGenerating with prompt length: {len(prompt)}")
    model, tokenizer = load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Starting generation...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,      # Reduced from 512 to speed up generation
        temperature=0.7,         # Balanced between creativity and coherence
        top_p=0.95,             # Allow for diverse but high-quality responses
        do_sample=True,         # Enable sampling for better quality
        pad_token_id=tokenizer.eos_token_id,
        num_beams=1,            # Use greedy search for faster generation
        repetition_penalty=1.1,  # Slightly reduce repetition
        length_penalty=1.0      # No length penalty
    )
    print("Generation complete!")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from the response
    response = response[len(prompt):].strip()
    
    # Clean up the response
    response = clean_response(response)
    
    return response

def generate_response_without_ad(user_query: str) -> str:
    print("\nGenerating response without ad...")
    prompt = get_prompt_without_ad(user_query)
    response = generate_text(prompt)
    print("Response without ad generated successfully!")
    return response

def generate_response_with_ad(user_query: str, ad_text: str) -> str:
    print("\nGenerating response with ad...")
    prompt = get_prompt_with_ad(user_query, ad_text)
    response = generate_text(prompt)
    print("Response with ad generated successfully!")
    return response

def generate_responses(user_query: str, ad_facts: dict) -> tuple[str, str]:
    """Generate both responses - with and without ad."""
    # Format ad text from facts
    ad_text = f"""Product: {ad_facts['ad_product']}
Brand: {ad_facts['brand']}
URL: {ad_facts['url']}
Description: {ad_facts['description']}"""

    # Generate both responses
    response_without_ad = generate_response_without_ad(user_query)
    response_with_ad = generate_response_with_ad(user_query, ad_text)
    
    return response_without_ad, response_with_ad