import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from training.evaluation import evaluate_logged_responses
from training.ppo_loop import run_ppo_finetuning
from trl import AutoModelForCausalLMWithValueHead

def load_model_and_tokenizer():
    print("Loading DeepSeek model...")
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load base model first (ensure it's fully on GPU if available)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    base_model.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

    # Ensure base model is on the correct device before creating value head
    if torch.cuda.is_available():
        base_model = base_model.to(torch.device("cuda"))

    # Create value head model from the same checkpoint
    model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model, # Load from the base model object
        trust_remote_code=True
    )

    # Explicitly copy generation config to value head model
    model_with_value_head.generation_config = base_model.generation_config

    # Explicitly set the base_model_prefix attribute needed by PPOTrainer
    if hasattr(base_model.config, "base_model_prefix"):
        model_with_value_head.base_model_prefix = base_model.config.base_model_prefix
    else:
        # Try a common default if not found (specific to model architecture)
        # The underlying model seems to be stored in 'pretrained_model' for AutoModelForCausalLMWithValueHead
        print("Warning: base_model.config.base_model_prefix not found. Setting default 'pretrained_model'.")
        model_with_value_head.base_model_prefix = 'pretrained_model' # This should point to the base model

    print("✅ Models loaded\n")
    return base_model, model_with_value_head, tokenizer


if __name__ == "__main__":
    base_model, model_with_value_head, tokenizer = load_model_and_tokenizer()
    # Pass only the value head model and tokenizer
    run_ppo_finetuning(model_with_value_head, tokenizer)
    # Evaluate the correct log file
    evaluate_logged_responses("logs/ppo_training_log.csv")