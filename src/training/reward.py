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

        return reward # total reward score would be 0-8

    except Exception as e:
        print(f"compute_reward failed: {e}")
        return 0.0