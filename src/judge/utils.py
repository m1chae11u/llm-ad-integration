import os
import json
import re
from openai import OpenAI
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


# Load environment variables
dotenv.load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

# Initialize clients
embedding_client = OpenAI(api_key=OPENAI_API_KEY)
chat_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

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
    """Get embedding for a single text using OpenAI's API."""
    try:
        response = embedding_client.embeddings.create(
            model=model,
            input=[text]
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a random vector as fallback to avoid zero similarity
        return np.random.randn(1536) / np.sqrt(1536)  # Normalized random vector

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

def batch_get_embeddings(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """Get embeddings for multiple texts in batches."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = []
        for text in batch:
            cache_key = get_cache_key("embedding", text)
            if cache_key in _embedding_cache:
                batch_results.append(_embedding_cache[cache_key])
            else:
                # Make API call and cache result
                embedding = get_embedding(text)
                _embedding_cache[cache_key] = embedding
                batch_results.append(embedding)
        results.extend(batch_results)
    return results

def clear_caches():
    """Clear all caches."""
    _embedding_cache.clear()
    _judge_cache.clear()

def call_deepseek_and_extract_json(prompt, keys):
    """Call DeepSeek API and extract JSON response."""
    try:
        response = chat_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You return JSON only."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            timeout=30
        ).choices[0].message.content

        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            for key in keys:
                result.setdefault(key, None)
            return result
        return {key: None for key in keys}
    except Exception as e:
        print("DeepSeek API call failed:", e)
        return {key: None for key in keys} 