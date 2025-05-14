import os
from dotenv import load_dotenv
import multiprocessing # For JUDGE_MAX_WORKERS default

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

# PPO Training Hyperparameters
PPO_LEARNING_RATE = 1e-5
PPO_OPTIMIZER_TYPE = "AdamW"  # "AdamW" or "SGD"
DATA_PROCESSOR_BATCH_SIZE = 32 # Batch size for DataProcessor internal logic if different from training batch size
TRAINING_BATCH_SIZE = 32       # Batch size for iterating through the main dataset
JUDGE_MAX_WORKERS = multiprocessing.cpu_count() # Max workers for judging, defaults to CPU count
PPO_MAX_GRAD_NORM = 1.0       # Max gradient norm for clipping
VALIDATION_INTERVAL_BATCHES = 10 # Run validation every N batches
CHECKPOINT_INTERVAL_BATCHES = 50 # Save full checkpoint every N batches
LOG_FLUSH_INTERVAL_QUERIES = 5   # In DataProcessor, flush logs & save query checkpoint every N queries
CHECKPOINTS_TO_KEEP = 2          # Number of recent checkpoints to keep
VALIDATION_SET_SIZE = 100        # Fixed number of samples for validation set
VALIDATION_SET_RATIO = 0.1       # Ratio of dataset to use for validation if size is smaller (e.g., 10%)
PPO_CLIP_RANGE = 0.2          # PPO clipping range (epsilon for policy loss) - Note: TRL uses cliprange

# New PPO/GAE parameters for TRL
PPO_GAMMA = 0.99              # Discount factor for future rewards
PPO_LAMBDA = 0.95             # Lambda for Generalized Advantage Estimation (GAE)
PPO_CLIP_EPSILON = 0.2        # Clipping parameter for PPO (epsilon, same as PPO_CLIP_RANGE, TRL might use a different name)
PPO_EPOCHS = 4                # Number of epochs to train on a batch of data in PPO
KL_COEFF = 0.2                # Coefficient for the KL divergence penalty term (beta in TRL)
VF_COEFF = 0.5                # Coefficient for the value function loss in the PPO objective
TARGET_KL = None              # Target KL for adaptive KL controller (e.g., 0.1 or 0.05, None for fixed KL_COEFF)
PPO_BATCH_SIZE = 16           # Number of queries to collect before a PPO update (experience collection batch)
PPO_MINI_BATCH_SIZE = 4       # Mini-batch size for gradient updates within PPO epochs (if PPO_BATCH_SIZE is divisible)

# Prompt Engineering related (optional, can be added if frequently changed)
GENERATION_MAX_NEW_TOKENS = 128 # Max new tokens for generation during PPO
# GENERATOR_PROMPT_MAX_LENGTH = 1024
# GENERATOR_RESPONSE_MAX_LENGTH = 256
# PPO_INPUT_MAX_LENGTH = 384
# PPO_RESPONSE_MAX_LENGTH = 128

