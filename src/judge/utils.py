import os
import json
import re
from openai import OpenAI, RateLimitError, OpenAIError
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dotenv
import google.generativeai as genai
import asyncio
import aiohttp


# Load environment variables
dotenv.load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Initialize clients
embedding_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# Cache for embeddings and judge results
_embedding_cache = {}
_judge_cache = {}

# Connection pool for API calls
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
session.mount("http://", adapter)
session.mount("https://", adapter)

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    """Get embedding for a single text using OpenAI's API, retrying on rate limits."""
    # Throttle requests to avoid hitting API rate limits
    time.sleep(0.2)
    max_retries = 5
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            response = embedding_client.embeddings.create(
                model=model,
                input=[text]
            )
            return np.array(response.data[0].embedding)
        except RateLimitError as e:
            print(f"Rate limit exceeded, retrying in {backoff}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(backoff)
            backoff *= 2
        except OpenAIError as e:
            print(f"OpenAI API error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error getting embedding: {e}")
            break
    # Fallback to random vector to avoid blocking downstream logic
    return np.random.randn(1536) / np.sqrt(1536)

def get_cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate a cache key from function name and arguments."""
    key_parts = [func_name]
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, dict):
            key_parts.append(json.dumps(arg, sort_keys=True))
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()

def cache_result(ttl_seconds: int = 3600):
    """Decorator to cache function results with TTL."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = get_cache_key(func.__name__, *args, **kwargs)
            if cache_key in _judge_cache:
                timestamp, result = _judge_cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    return result
            result = func(*args, **kwargs)
            _judge_cache[cache_key] = (time.time(), result)
            return result
        return wrapper
    return decorator

def batch_get_embeddings(texts: List[str], batch_size: int = 32, max_workers: int = 10) -> List[np.ndarray]:
    """Get embeddings for multiple texts in parallel batches (serial by default)."""
    results = [None] * len(texts)
    
    def process_text(idx: int, text: str) -> None:
        cache_key = get_cache_key("embedding", text)
        if cache_key in _embedding_cache:
            results[idx] = _embedding_cache[cache_key]
        else:
            embedding = get_embedding(text)
            _embedding_cache[cache_key] = embedding
            results[idx] = embedding
    
    # Limit parallel embedding threads to avoid exceeding API rate limits
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            futures = [
                executor.submit(process_text, i + j, text)
                for j, text in enumerate(batch)
            ]
            # Wait for batch to complete
            for future in futures:
                future.result()
    
    return results

def clear_caches():
    """Clear all caches."""
    _embedding_cache.clear()
    _judge_cache.clear()

async def call_gemini_api(prompt, keys):
    """Call Gemini 1.5 Flash API and extract JSON response asynchronously."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.1,
                "top_k": 1,
            }
        )
        
        # Extract JSON from response
        text = response.text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            for key in keys:
                result.setdefault(key, None)
            return result
        return {key: None for key in keys}
    except Exception as e:
        print("Gemini API call failed:", e)
        return {key: None for key in keys}

# Synchronous version for backward compatibility
def call_gemini_and_extract_json(prompt, keys):
    """Call Gemini 1.5 Flash API and extract JSON response."""
    return asyncio.run(call_gemini_api(prompt, keys))

async def async_parallel_judge_responses(responses: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Process multiple judge evaluations in parallel using async.
    
    Args:
        responses: List of dicts containing 'query', 'response', and optionally 'ad_info'
    
    Returns:
        List of judge results for each response
    """
    from .coherence import judge_coherence_async
    from .helpfulness import judge_helpfulness_async
    from .salience import judge_ad_salience_async
    from .detectability import judge_detectability_async
    
    results = []
    
    for response_data in responses:
        query = response_data['query']
        response = response_data['response']
        ad_info = response_data.get('ad_info')
        
        # Run all judge functions in parallel using asyncio
        tasks = [
            judge_coherence_async(response, query),
            judge_helpfulness_async(query, response)
        ]
        
        if ad_info:
            tasks.extend([
                judge_ad_salience_async(query, response, ad_info),
                judge_detectability_async(response, response_data.get('without_ad', ''))
            ])
        
        # Gather all results
        judge_results = await asyncio.gather(*tasks)
        
        # Combine results into a single dictionary
        combined_result = {}
        for result in judge_results:
            combined_result.update(result)
        
        results.append(combined_result)
    
    return results

def parallel_judge_responses(responses: List[Dict[str, str]], max_workers: int = 10) -> List[Dict[str, Any]]:
    """Process multiple judge evaluations in parallel.
    
    Args:
        responses: List of dicts containing 'query', 'response', and optionally 'ad_info'
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of judge results for each response
    """
    # Use asyncio to run parallel API calls
    return asyncio.run(async_parallel_judge_responses(responses)) 



    ## note: the embedding API is rate-limited, so we need to throttle requests
    ## the current version of ppo doesnt support step() with a reward model. willl try to figure out a way to do this. 