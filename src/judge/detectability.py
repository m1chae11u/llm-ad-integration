import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from config import OPENAI_API_KEY
from judge.utils import batch_get_embeddings, cache_result

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).reshape(1, -1)

@cache_result(ttl_seconds=3600)  # Cache results for 1 hour
def judge_detectability(with_ad, without_ad):
    try:
        # Get embeddings in batch
        embeddings = batch_get_embeddings([with_ad, without_ad])
        emb_with, emb_without = embeddings[0], embeddings[1]
        
        # Reshape for sklearn
        emb_with = emb_with.reshape(1, -1)
        emb_without = emb_without.reshape(1, -1)
        
        similarity = cosine_similarity(emb_with, emb_without)[0][0]
        detectability_score = 1 - similarity

        return {
            "detectability_cosine": round(detectability_score, 4),
            "similarity_cosine": round(similarity, 4),
        }

    except Exception as e:
        print("Detectability error:", e)
        return {
            "detectability_cosine": None,
            "similarity_cosine": None,
        }