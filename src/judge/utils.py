import json
import re
from openai import OpenAI
from config import DEEPSEEK_API_KEY  # config.py is in the same directory as judge/

# Initialize DeepSeek client
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

def call_deepseek_and_extract_json(prompt, keys):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You return JSON only."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            timeout=30
        ).choices[0].message.content

        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            for key in keys:
                result.setdefault(key, None)
            return result
        return {key: None for key in keys}
    except Exception as e:
        print("DeepSeek API call failed:", e)
        return {key: None for key in keys} 