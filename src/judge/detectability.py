import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import asyncio
from .utils import batch_get_embeddings, cache_result

@cache_result(ttl_seconds=3600)  # Cache results for 1 hour
def judge_detectability(with_ad: str, without_ad: str) -> dict:
    """
    Computes the detectability and similarity scores between two text inputs
    based on their cosine distance in the embedding space.

    Args:
        with_ad (str): Text input with the advertisement.
        without_ad (str): Text input without the advertisement.

    Returns:
        dict: A dictionary containing the cosine-based detectability and similarity scores.
    """
    try:
        # Validate input
        if not with_ad.strip() or not without_ad.strip():
            raise ValueError("Empty input string provided.")

        # Get embeddings in a batch
        embeddings = batch_get_embeddings([with_ad, without_ad])
        emb_with, emb_without = embeddings
        
        # Avoid division by zero by setting similarity to 0 if the norm is 0
        if np.linalg.norm(emb_with) == 0 or np.linalg.norm(emb_without) == 0:
            warnings.warn("One or both embeddings are zero vectors. Setting similarity to 0.")
            similarity = 0.0
        else:
            # Compute cosine similarity directly
            similarity = cosine_similarity(emb_with.reshape(1, -1), emb_without.reshape(1, -1))[0][0]

        # Calculate detectability
        detectability_score = 1 - similarity

        return {
            "detectability_cosine": round(detectability_score, 4),
            "similarity_cosine": round(similarity, 4),
        }

    except Exception as e:
        warnings.warn(f"Detectability error: {e}")
        return {
            "detectability_cosine": None,
            "similarity_cosine": None,
        }

async def judge_detectability_async(with_ad: str, without_ad: str) -> dict:
    """
    Async version of the detectability judge.
    
    Args:
        with_ad (str): Text input with the advertisement.
        without_ad (str): Text input without the advertisement.

    Returns:
        dict: A dictionary containing the cosine-based detectability and similarity scores.
    """
    # Since the actual embedding operation is not async, we'll use asyncio.to_thread
    # to run the synchronous function in a separate thread
    return await asyncio.to_thread(judge_detectability, with_ad, without_ad)