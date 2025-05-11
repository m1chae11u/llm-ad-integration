from bert_score import score as bert_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from src.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).reshape(1, -1)

def judge_detectability(with_ad, without_ad):
    try:
        # Cosine Similarity (OpenAI Embeddings)
        emb_with = get_embedding(with_ad)
        emb_without = get_embedding(without_ad)
        similarity = cosine_similarity(emb_with, emb_without)[0][0]
        detectability_score = 1 - similarity

        # BERTScore (token-level contextual match)
        P, R, F1 = bert_score([with_ad], [without_ad], lang="en", rescale_with_baseline=True)
        bert_f1 = round(F1[0].item(), 4)
        bert_detectability = round(1 - bert_f1, 4)

        return {
            "detectability_cosine": round(detectability_score, 4),
            "similarity_cosine": round(similarity, 4),
            "detectability_bert": bert_detectability,
            "bert_f1": bert_f1
        }

    except Exception as e:
        print("Detectability error:", e)
        return {
            "detectability_cosine": None,
            "similarity_cosine": None,
            "detectability_bert": None,
            "bert_f1": None
        }