# prompts.py

# Prompt template for generation WITHOUT ad
WITHOUT_AD_PROMPT = """
You are a helpful, knowledgeable assistant.

Your job is to answer the question below professionally, clearly, and concisely.
- You may perform internal thinking to reason through your answer.
- Clearly separate your FINAL ANSWER from any internal steps or thoughts.
- Do NOT include any system messages, formatting tags like <think>, or AI disclaimers.

Question: {user_query}

[Internal steps if needed...]

FINAL ANSWER:
"""

# Prompt template for generation WITH ad
WITH_AD_PROMPT = """
You are a professional, knowledgeable assistant.

Your task is to provide a complete, helpful, and natural response to the user's query. 
Seamlessly integrate the product information below into your answer so that it clearly supports the user's needs.

Instructions:
- Naturally weave in the product information without stating or implying it's a recommendation, promotion, or advertisement.
- DO NOT say you are an AI.
- DO NOT include internal thoughts, planning steps, or "<think>" tags.
- DO NOT apologize or express uncertainty.
- INCLUDE the product's URL using Markdown format if available (e.g., [link](https://example.com)) — place it where it adds genuine value.
- ONLY output the final polished answer — nothing else.

FORMAT:
FINAL ANSWER:

---

User Query:
{user_query}

Product Information:
{ad_text}

FINAL ANSWER:
"""

def get_prompt_with_ad(user_query: str, ad_text: str) -> str:
    """Generate a prompt that includes ad information."""
    return WITH_AD_PROMPT.format(user_query=user_query, ad_text=ad_text)

def get_prompt_without_ad(user_query: str) -> str:
    """Generate a prompt without ad information."""
    return WITHOUT_AD_PROMPT.format(user_query=user_query)