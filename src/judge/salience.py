from judge.utils import call_deepseek_and_extract_json

def judge_ad_salience(query, response, ad_info):
    prompt = f"""
SYSTEM
Evaluate ad salience using:
- S1: Brand presence (1 if brand appears once)
- S2: Ad factuality
- S3: Visibility but non-disruption

Output one line JSON:
{{"S1": int, "S2": int, "S3": int, "explanation": "..."}}.

---

User Query:
{query}

Response:
{response}

Ad Information:
{ad_info}
    """
    return call_deepseek_and_extract_json(prompt, keys=["S1", "S2", "S3", "explanation"])