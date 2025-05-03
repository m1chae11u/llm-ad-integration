from judge.utils import call_deepseek_and_extract_json

def judge_coherence(response, query):
    prompt = f"""
SYSTEM
You are an impartial evaluator.

Your task is to assess the *coherence* of the LLM's response to a user query.

You MUST reply with one line of valid JSON and nothing else:
{{"C1": 0 or 1, "C2": 0 or 1, "C3": 0 or 1, "C4": 0 or 1, "explanation": "<brief justification>"}}

Scoring Criteria (0 or 1 only):

- C1: Logical Flow  
  • 1 = The response has a clear beginning, middle, and end. Ideas progress smoothly.  
  • 0 = Disorganized, jumps around, lacks clear structure.

- C2: Grammar & Mechanics  
  • 1 = Response is grammatically correct and well-written.  
  • 0 = Contains grammar/spelling/punctuation errors that reduce clarity.

- C3: Topic Relevance  
  • 1 = Stays focused and directly addresses the user’s query.  
  • 0 = Wanders off-topic, includes unrelated info, or fails to answer.

- C4: Seamless Ad Integration  
  • 1 = If present, the product mention blends naturally into the response.  
  • 0 = Product mention is awkward, overly promotional, or disruptive.

EXAMPLES

✔ GOOD RESPONSE  
Query: What’s a quick, healthy lunch idea?  
Response: Try a fresh wrap with tofu and veggies. [VegDelight](https://example.com) makes protein-rich tofu wraps that are easy to prep and full of flavor.  
RETURN: {{"C1":1,"C2":1,"C3":1,"C4":1,"explanation":"Well-structured, clear grammar, relevant, smooth ad inclusion."}}

✘ BAD RESPONSE  
Query: What’s a quick, healthy lunch idea?  
Response: Many people like food. VegDelight is amazing. Just eat anything I guess.  
RETURN: {{"C1":0,"C2":0,"C3":0,"C4":0,"explanation":"Unstructured, vague, off-topic, and ad feels abrupt."}}

---

User Query:
{query}

LLM Response:
{response}
"""

    result = call_deepseek_and_extract_json(prompt, keys=["C1", "C2", "C3", "C4", "explanation"])

    total = sum(result.get(k, 0) for k in ["C1", "C2", "C3", "C4"])

    return {
        "C1": result.get("C1", 0),
        "C2": result.get("C2", 0),
        "C3": result.get("C3", 0),
        "C4": result.get("C4", 0),
        "Coherence Explanation": result.get("explanation", ""),
        "Coherence Score": total
    }