from judge.coherence import judge_coherence
from judge.helpfulness import judge_helpfulness
from judge.salience import judge_ad_salience
from judge.detectability import judge_detectability

def compute_reward(query, response_with_ad, response_without_ad, ad_facts={}):
    """
    Evaluate LLM response using all judge modules and return scalar reward.
    This version matches your actual judge outputs and JSON scoring format.
    """

    try:
        # Run judges (using DeepSeek API + OpenAI Embedding)
        coherence = judge_coherence(response_with_ad, query)  # C1–C4
        helpfulness = judge_helpfulness(query, response_with_ad)  # H1
        salience = judge_ad_salience(query, response_with_ad, ad_facts)  # S1–S3
        detect = judge_detectability(response_with_ad, response_without_ad)  # cosine + BERT detectability

        # --- Extract scores ---
        coherence_score = sum([
            coherence.get("C1", 0),
            coherence.get("C2", 0),
            coherence.get("C3", 0),
            coherence.get("C4", 0)
        ])  # Max = 4

        helpfulness_score = helpfulness.get("H1", 0)  # Max = 1

        salience_score = sum([
            salience.get("S1", 0),
            salience.get("S2", 0),
            salience.get("S3", 0)
        ])  # Max = 3

        detectability_cosine = detect.get("detectability_cosine", 0.5)  # Range [0, 1]
        detectability_reward = 1 - detectability_cosine  # Lower similarity → higher reward

        # --- Combine into reward ---
        reward = (
            coherence_score +
            helpfulness_score +
            salience_score +
            detectability_reward  # keep weight 1 for now
        )

        # Construct the full details dictionary
        reward_details = {
            **coherence,
            **helpfulness,
            **salience,
            **detect,
            "Coherence Score": coherence_score,
            "Helpfulness Score": helpfulness_score,
            "Ad Salience Score": salience_score,
            "Detectability Reward": detectability_reward,
            "Total Score": reward
        }

        return reward_details

    except Exception as e:
        print(f"compute_reward failed: {e}")
        # Return a dictionary with default values on failure
        return {
            "C1": 0, "C2": 0, "C3": 0, "C4": 0,
            "H1": 0,
            "S1": 0, "S2": 0, "S3": 0,
            "detectability_cosine": 0.5, "similarity_cosine": 0.0, "detectability_bert": 0.0, "bert_f1": 0.0,
            "Coherence Explanation": "Failed", "Helpfulness Explanation": "Failed", "Ad Salience Explanation": "Failed",
            "Coherence Score": 0, "Helpfulness Score": 0, "Ad Salience Score": 0, "Detectability Reward": 0, "Total Score": 0.0
        }