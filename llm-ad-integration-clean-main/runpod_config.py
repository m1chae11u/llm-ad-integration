import os

# RunPod API Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")

# Model Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.95

# Hardware Configuration
GPU_TYPE = "NVIDIA A100 80GB"  # or whatever GPU you're using on RunPod
BATCH_SIZE = 1  # Adjust based on your GPU memory 