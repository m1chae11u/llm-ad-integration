import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Model and Path Configurations
BASE_MODEL = "meta-llama/Llama-3.1-8B"  # Default value
CHECKPOINT_DIR = "checkpoints/ppo_llama"    # Changed from ppo_manual
DATA_FILE = "data/merged_queries_ads.csv" # Default value

# Environment Variable Names (used in main.py to load actual values)
HF_TOKEN_ENV_VAR = "HF_TOKEN"
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY" # Already used above, but main.py imports it
PROJECT_ID_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
CUSTOM_JUDGE_MODEL_ID_ENV_VAR = "CUSTOM_JUDGE_MODEL_ID"

