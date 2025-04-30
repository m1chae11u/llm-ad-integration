from judge.utils import call_deepseek_and_extract_json

def judge_helpfulness(query, response):
    prompt = f"""
SYSTEM
You are a helpful judge.

Score how fully this response answers the user's question.

Output one line JSON:
{{"H1": int, "explanation": "..."}}.

---

User Query:
{query}

LLM Response:
{response}
    """
    return call_deepseek_and_extract_json(prompt, keys=["H1", "explanation"])