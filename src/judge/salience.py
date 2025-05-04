from judge.utils import call_deepseek_and_extract_json

def judge_ad_salience(query, response, ad_info):
    prompt = f"""
SYSTEM  
You are an impartial evaluator.

Your task is to assess the *ad salience* of this response — how well the ad is included — using the following criteria.

Return one line of valid JSON and **nothing else**:
{{"S1": 0 or 1, "S2": 0 or 1, "S3": 0 or 1, "explanation": "<brief justification>"}}

Scoring Criteria (0 or 1 only):

- S1: Brand Presence  
  • 1 = Brand name appears once, naturally.  
  • 0 = Brand is missing, awkwardly inserted, or hallucinated.

- S2: Ad Factuality  
  • 1 = Product facts match the ad info exactly — no hallucination, exaggeration, or omission.  
  • 0 = Misrepresents or fabricates any product feature.

- S3: Visibility w/o Disruption  
  • 1 = Ad content is clearly included and smoothly integrated into the tone and flow.  
  • 0 = Ad feels forced, disruptive, overly promotional, or breaks the helpful tone.  
    (Even slight awkwardness counts as 0.)

Additional Instructions:
- Your explanation must be concise and specific (1–2 lines max).
- Do not include overall impressions — focus only on how the ad is presented.

EXAMPLES

Ad Info:  
Brand: VegDelight  
Product: Tofu Wraps  
Description: Healthy vegan wraps high in protein and fiber.

✔ GOOD CASE  
Response: For a quick lunch, try protein-packed meals like tofu wraps. [VegDelight](https://example.com) makes a great one — high in fiber, easy to prep.  
RETURN: {{"S1": 1, "S2": 1, "S3": 1, "explanation": "Brand present, factual, and fits naturally into helpful sentence."}}

✘ BAD CASE  
Response: You can buy salad. Also VegDelight is tasty, delicious, and life-changing. Vegan is good.
RETURN: {{"S1": 1, "S2": 0, "S3": 0, "explanation": "Brand present, but ad facts are wrong and tone is exaggerated."}}
---

User Query:
{query}

LLM Response:
{response}

Ad Information:
{ad_info}
"""
    result = call_deepseek_and_extract_json(prompt, keys=["S1", "S2", "S3", "explanation"])

    total = sum(result.get(k, 0) for k in ["S1", "S2", "S3"])

    return {
        "S1": result.get("S1", 0),
        "S2": result.get("S2", 0),
        "S3": result.get("S3", 0),
        "Ad Salience Explanation": result.get("explanation", ""),
        "Ad Salience Score": total
    }