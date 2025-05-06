import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from training.evaluation import evaluate_logged_responses
from training.manual_ppo_loop import run_manual_ppo

def load_model_and_tokenizer():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    import gc
    gc.collect()
    print("Loading DeepSeek model...")
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    print("✅ Models loaded\n")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    run_manual_ppo(model, tokenizer)
    # Optionally evaluate logged responses
    try:
        evaluate_logged_responses("logs/ppo_manual_log.csv")
    except FileNotFoundError:
        print("Log file not found, skipping evaluation.")  