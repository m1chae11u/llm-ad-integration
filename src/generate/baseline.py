from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_baseline_response(query, ad_facts):
    """
    Generate a response with an ad using a simple prompt template without any training.
    
    Args:
        query (str): The user's query
        ad_facts (dict): Dictionary containing ad information (product, brand, url, description)
    
    Returns:
        str: Generated response with ad
    """
    # Load model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create prompt template
    prompt = f"""User Query: {query}

Please provide a helpful response to the user's query. Naturally incorporate the following product information into your response:

Product: {ad_facts['ad_product']}
Brand: {ad_facts['brand']}
Description: {ad_facts['description']}
URL: {ad_facts['url']}

Your response should:
1. Directly answer the user's query
2. Naturally mention the product once
3. Include the product URL
4. Maintain a helpful and conversational tone

Response:"""
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part (after "Response:")
    response = response.split("Response:")[-1].strip()
    
    return response 