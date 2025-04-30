import requests
import json
import time
from .prompts import get_prompt_with_ad, get_prompt_without_ad
from runpod_config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, MAX_TOKENS, TEMPERATURE, TOP_P

print("\nInitializing RunPod connection...")
headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

def call_runpod(prompt: str) -> str:
    """Call RunPod API to generate response."""
    print(f"\nCalling RunPod API with prompt length: {len(prompt)}")
    
    payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P
        }
    }
    
    try:
        # Start the job
        response = requests.post(
            f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"Job started with ID: {job_id}")
        
        # Poll for completion
        while True:
            status_response = requests.get(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}",
                headers=headers
            )
            status_response.raise_for_status()
            status = status_response.json()
            
            if status["status"] == "COMPLETED":
                print("Job completed successfully")
                return status["output"]["text"]
            elif status["status"] == "FAILED":
                print("Job failed")
                raise Exception(f"RunPod job failed: {status.get('error', 'Unknown error')}")
            
            print("Job still running, waiting...")
            time.sleep(2)  # Wait 2 seconds before checking again
            
    except Exception as e:
        print(f"Error calling RunPod API: {e}")
        raise

def generate_response_without_ad(user_query: str) -> str:
    prompt = get_prompt_without_ad(user_query)
    response = call_runpod(prompt)
    if "FINAL ANSWER:" in response:
        return response.split("FINAL ANSWER:")[-1].strip()
    return response.strip()

def generate_response_with_ad(user_query: str, ad_text: str) -> str:
    prompt = get_prompt_with_ad(user_query, ad_text)
    response = call_runpod(prompt)
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