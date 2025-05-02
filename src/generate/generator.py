from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from .prompts import get_prompt_with_ad, get_prompt_without_ad
from tqdm import tqdm

# Load model and tokenizer only once
print("\nLoading DeepSeek model and tokenizer...")
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Optimize model for inference
print("Compiling model for faster inference...")
model = torch.compile(model)
print("Model loaded and compiled successfully!")

def clean_response(response: str) -> str:
    """Clean up the response by removing thinking processes and other unwanted content."""
    # Remove any content between <think> tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove any content that looks like thinking/planning
    response = re.sub(r'(?:Let me|I\'ll|I will|First|Next|Then|Finally).*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)
    
    # Remove any system messages or AI disclaimers
    response = re.sub(r'(?:As an AI|I am an AI|I\'m an AI|As a language model).*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)
    
    # Remove any content before "FINAL ANSWER:" if it exists
    if "FINAL ANSWER:" in response:
        response = response.split("FINAL ANSWER:")[-1]
    
    # Clean up any remaining whitespace and newlines
    response = response.strip()
    
    return response

def generate_text(prompt: str) -> str:
    """Generate text using local DeepSeek model."""
    print(f"\nGenerating with prompt length: {len(prompt)}")
    
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