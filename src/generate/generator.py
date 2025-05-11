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
    # If <think> is present, remove everything before and including it
    response = re.sub(r'^.*?</think>', '', response, flags=re.DOTALL)

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
        max_new_tokens=512,      
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

def generate_responses(query, ad_facts, use_optimized=True):
    """
    Generate responses with and without ads using either the baseline or optimized model.
    
    Args:
        query (str): The user's query
        ad_facts (dict): Dictionary containing ad information
        use_optimized (bool): Whether to use the optimized model (True) or baseline (False)
    
    Returns:
        tuple: (response_without_ad, response_with_ad)
    """
    if use_optimized:
        # Use the optimized model (PPO-trained)
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Generate response without ad
        prompt_without_ad = f"User Query: {query}\n\nPlease provide a helpful response:\n\nResponse:"
        inputs = tokenizer(prompt_without_ad, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        response_without_ad = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_without_ad = response_without_ad.split("Response:")[-1].strip()
        
        # Generate response with ad using optimized model
        response_with_ad = generate_baseline_response(query, ad_facts)  # For now, use baseline as placeholder
        
    else:
        # Use baseline model
        response_without_ad = generate_baseline_response(query, {})  # Empty ad_facts for no ad
        response_with_ad = generate_baseline_response(query, ad_facts)
    
    return response_without_ad, response_with_ad