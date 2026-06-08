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
from google import genai
from google.genai import types as genai_types
import asyncio
import aiohttp


# Load environment variables
dotenv.load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# OpenAI API key is now optional (used as fallback only)
# Google API key is required for embeddings and Gemini
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set (required for embeddings and Gemini API)")

# Initialize clients
# Use Google embeddings instead of OpenAI to avoid rate limits
# Keep OpenAI client as optional fallback
embedding_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)  # For Gemini judge + embeddings

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

def get_embedding(text: str, model: str = None) -> np.ndarray:
    """Get embedding for a single text using Google's Gemini Embedding API."""
    # Use Google's gemini-embedding-001 model
    # Supports output_dimensionality: 768, 1536, or 3072 (we use 1536 to match existing code)
    
    max_retries = 5
    backoff = 1.0
    
    for attempt in range(max_retries):
        try:
            result = gemini_client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=genai_types.EmbedContentConfig(
                    output_dimensionality=1536,  # Match existing code dimension
                    task_type="SEMANTIC_SIMILARITY"  # Good for similarity comparisons
                )
            )
            embedding = np.array(result.embeddings[0].values)

            # Normalize embedding for better similarity calculations (recommended for non-3072 dims)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Ensure it's exactly 1536 dimensions
            if len(embedding) != 1536:
                if len(embedding) < 1536:
                    # Pad with zeros (shouldn't happen with output_dimensionality=1536)
                    padding = np.zeros(1536 - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                else:
                    embedding = embedding[:1536]
            
            return embedding
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Google embedding API error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"Google embedding API failed after {max_retries} attempts: {e}")
                # Fallback to OpenAI if available
                if embedding_client:
                    try:
                        response = embedding_client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=[text]
                        )
                        return np.array(response.data[0].embedding)
                    except Exception as openai_error:
                        print(f"OpenAI fallback also failed: {openai_error}")
    
    # Final fallback to random vector to avoid blocking downstream logic
    print("⚠️ All embedding APIs failed, using random vector fallback")
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

async def call_gemini_api(prompt, keys, max_retries=3, backoff_factor=1.0):
    """Call Gemini API and extract JSON response asynchronously with retry logic."""
    model_name = 'gemini-2.5-flash'  # Stable, best price-performance, fast, supports structured outputs

    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.1,
                    top_k=1,
                )
            )
            
            # Extract JSON from response with robust parsing
            if not hasattr(response, 'text') or not response.text:
                raise ValueError("Empty response from Gemini API")
            
            text = response.text.strip()
            
            # Try multiple JSON extraction strategies
            result = None
            
            # Strategy 1: Look for JSON object in the text
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Try parsing the entire text as JSON
            if result is None:
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Look for JSON in code blocks
            if result is None:
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if code_block_match:
                    try:
                        result = json.loads(code_block_match.group(1))
                    except json.JSONDecodeError:
                        pass
            
            # If we got a result, validate and return it
            if result is not None and isinstance(result, dict):
                # Ensure all required keys exist with defaults
                for key in keys:
                    result.setdefault(key, None)
                return result
            
            # If no valid JSON found, log and retry
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                print(f"⚠️ No valid JSON found in response (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                print(f"   Response preview: {text[:200]}...")
                await asyncio.sleep(wait_time)
            else:
                print(f"⚠️ Failed to extract JSON after {max_retries} attempts. Response: {text[:500]}")
                return {key: None for key in keys}
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                print(f"⚠️ Gemini API call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ Gemini API call failed after {max_retries} attempts: {e}")
                return {key: None for key in keys}
    
    # Fallback (should never reach here, but just in case)
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