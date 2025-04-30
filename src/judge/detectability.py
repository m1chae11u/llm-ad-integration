import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).reshape(1, -1)

def judge_detectability(with_ad, without_ad):
    try:
        emb_with = get_embedding(with_ad)
        emb_without = get_embedding(without_ad)
        similarity = cosine_similarity(emb_with, emb_without)[0][0]
        return {
            "detectability": round(1 - similarity, 4),
            "similarity": round(similarity, 4)
        }
    except Exception as e:
        print("Detectability error:", e)
        return {"detectability": None, "similarity": None}