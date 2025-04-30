from judge.utils import call_deepseek_and_extract_json

def judge_coherence(response, query):
    prompt = f"""
SYSTEM
You are an impartial evaluator that returns structured scores.

Evaluate this response's coherence to the user query using:
- C1: Logical flow
- C2: Grammar & mechanics
- C3: Topic relevance
- C4: Seamless ad integration

Output one line JSON:
{{"C1": int, "C2": int, "C3": int, "C4": int, "explanation": "..."}}.

---

User Query:
{query}

LLM Response:
{response}
    """
    return call_deepseek_and_extract_json(prompt, keys=["C1", "C2", "C3", "C4", "explanation"])