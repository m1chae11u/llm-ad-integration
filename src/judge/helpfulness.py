from .utils import call_gemini_and_extract_json, call_gemini_api

def judge_helpfulness(query, response):
    prompt = f"""
SYSTEM
You are an impartial evaluator tasked with judging whether the assistant's response fully answers the user's question.

You MUST return valid JSON on one line and nothing else.

Scoring rules:
- "H1": 1 → The response directly, completely, and clearly answers the user's query.
- "H1": 0 → The response is incomplete, vague, off-topic, or leaves part of the question unaddressed.

Justification should be concise, specific, and explain why the answer is or isn't fully helpful.

FORMAT:
{{"H1": 0 or 1, "explanation": "<brief reason for score>"}}

EXAMPLES

User Query: What are some vegan sources of protein?
LLM Response: Lentils, chickpeas, tofu, quinoa, and seitan are excellent vegan protein sources.
OUTPUT: {{"H1": 1, "explanation": "Response directly answers with a clear list of vegan protein sources."}}

User Query: How can I improve my sleep?
LLM Response: Sleep is important. There are many ways to sleep better.
OUTPUT: {{"H1": 0, "explanation": "Response is vague and does not provide actionable advice."}}

---

User Query:
{query}

LLM Response:
{response}
"""
    result = call_gemini_and_extract_json(prompt, keys=["H1", "explanation"])
    return {
        "H1": result.get("H1", 0),
        "Helpfulness Explanation": result.get("explanation", "")
    }

async def judge_helpfulness_async(query, response):
    prompt = f"""
SYSTEM
You are an impartial evaluator tasked with judging whether the assistant's response fully answers the user's question.

You MUST return valid JSON on one line and nothing else.

Scoring rules:
- "H1": 1 → The response directly, completely, and clearly answers the user's query.
- "H1": 0 → The response is incomplete, vague, off-topic, or leaves part of the question unaddressed.

Justification should be concise, specific, and explain why the answer is or isn't fully helpful.

FORMAT:
{{"H1": 0 or 1, "explanation": "<brief reason for score>"}}

EXAMPLES

User Query: What are some vegan sources of protein?
LLM Response: Lentils, chickpeas, tofu, quinoa, and seitan are excellent vegan protein sources.
OUTPUT: {{"H1": 1, "explanation": "Response directly answers with a clear list of vegan protein sources."}}

User Query: How can I improve my sleep?
LLM Response: Sleep is important. There are many ways to sleep better.
OUTPUT: {{"H1": 0, "explanation": "Response is vague and does not provide actionable advice."}}

---

User Query:
{query}

LLM Response:
{response}
"""
    result = await call_gemini_api(prompt, keys=["H1", "explanation"])
    return {
        "H1": result.get("H1", 0),
        "Helpfulness Explanation": result.get("explanation", "")
    }