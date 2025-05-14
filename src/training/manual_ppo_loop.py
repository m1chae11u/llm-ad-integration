import os
import torch
import pandas as pd
import gc
from tqdm import tqdm
from torch.nn import functional as F
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading
import time
import shutil
import tempfile
import json
from pathlib import Path
import logging
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, DPOTrainer # Added TRL imports
import trl
print(f"TRL version in use by script: {trl.__version__}")
print(f"PPOTrainer methods: {dir(trl.PPOTrainer)}")
import sys
print(f"Python executable: {sys.executable}")
import csv # Added for detailed logging

from ..judge import (
    judge_coherence,
    judge_helpfulness,
    judge_ad_salience,
    judge_detectability
)
# from ..generate.generator import generate_responses, clear_response_cache # generate_responses is no longer directly used here.
from ..generate.generator import clear_response_cache # Keep clear_response_cache if used by clear_caches or elsewhere.
from ..judge.utils import clear_caches
# from .checkpoint_manager import CheckpointManager # CheckpointManager will be removed
from ..config import (
    PPO_LEARNING_RATE, PPO_OPTIMIZER_TYPE, 
    DATA_PROCESSOR_BATCH_SIZE, TRAINING_BATCH_SIZE, # TRAINING_BATCH_SIZE might map to PPO_BATCH_SIZE or mini_batch_size
    JUDGE_MAX_WORKERS, PPO_MAX_GRAD_NORM, 
    VALIDATION_INTERVAL_BATCHES, CHECKPOINT_INTERVAL_BATCHES, 
    LOG_FLUSH_INTERVAL_QUERIES, CHECKPOINTS_TO_KEEP,
    VALIDATION_SET_SIZE, VALIDATION_SET_RATIO, PPO_CLIP_RANGE, # PPO_CLIP_RANGE maps to PPO_CLIP_EPSILON or cliprange in PPOConfig
    # New PPO/GAE parameters from config
    PPO_GAMMA, PPO_LAMBDA, PPO_CLIP_EPSILON, PPO_EPOCHS, 
    KL_COEFF, VF_COEFF, TARGET_KL, PPO_BATCH_SIZE, PPO_MINI_BATCH_SIZE,
    GENERATION_MAX_NEW_TOKENS # Added import for new config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Dummy Reward Model to satisfy PPOTrainer init when reward_model=None is not handled well ---
class DummyRewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Add a dummy parameter to ensure it can be moved to a device, etc.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, *args, **kwargs):
        # This model will not actually be called for reward computation in our setup
        # as rewards are computed externally and passed to ppo_trainer.step()
        # Return a dummy tensor if it were ever called, to avoid errors.
        # The shape and type might need adjustment if it was actually called, 
        # but for PPO with external rewards, it shouldn't be.
        if args:
            return torch.zeros(args[0].shape[0], device=self.dummy_param.device) 
        return torch.tensor(0.0, device=self.dummy_param.device)
# --------------------------------------------------------------------------------------------------

# Configure requests for reliability
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=10,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

# Create an adapter with the retry strategy
adapter = HTTPAdapter(max_retries=retry_strategy)

# Mount the adapter on the session
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# DataProcessor class is being removed as its functionality is integrated into
# the main run_manual_ppo loop or handled by TRL PPOTrainer.

# compute_ppo_loss and compute_advantages are no longer needed as TRL handles this.
# def compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_range=PPO_CLIP_RANGE):
#     ratio = torch.exp(new_log_probs - old_log_probs)
#     clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
#     return -torch.min(ratio * advantages, clipped * advantages).mean()

# def compute_advantages(reward, value):
#     return reward - value

def run_trl_validation_epoch(ppo_trainer, ref_model, validation_data: pd.DataFrame, generation_kwargs: dict, hf_token: str = None):
    logger.info("🔄 Starting TRL validation epoch...")
    actor_model = ppo_trainer.model # This is the model with the value head
    tokenizer = ppo_trainer.tokenizer
    device = ppo_trainer.accelerator.device # Get device from PPO trainer

    actor_model.eval()
    if ref_model: # ref_model might not always be passed if not needed by all judges
        ref_model.eval()

    all_validation_rewards = []
    all_coh_scores, all_help_scores, all_sal_scores, all_det_scores = [], [], [], []

    # Consider a small batch size for validation generation if validation_data is large
    # For now, iterating one by one, but generation can be batched.
    # Judging is parallelized by ThreadPoolExecutor.

    with torch.no_grad(): # Ensure no gradients are computed during validation
        for idx, row in tqdm(validation_data.iterrows(), total=len(validation_data), desc="Validation"):
            vague_query_text = str(row['vague_query']) # Original vague query
            ad_facts = {
                "ad_product": str(row['ad_product']), "brand": str(row['brand']),
                "url": str(row['url']), "description": str(row['ad_description']),
            }
            ad_text_for_judge = f"""Product: {ad_facts['ad_product']}
Brand: {ad_facts['brand']}
URL: {ad_facts['url']}
Description: {ad_facts['description']}"""

            # Construct the augmented prompt for the policy model
            prompt_for_policy_model = f"""Here is a query: {vague_query_text}

Please generate a helpful response. If appropriate, seamlessly integrate the following advertisement into your response:
Ad Details:
Product: {ad_facts['ad_product']}
Brand: {ad_facts['brand']}
Description: {ad_facts['description']}

Response:"""

            try:
                # 1. Generate response with ad from the current policy model using the augmented prompt
                # PPOTrainer.generate expects a list of Tensors.
                query_tensor_policy = tokenizer.encode(prompt_for_policy_model, return_tensors="pt").to(device)
                
                # Ensure tokenizer padding side is correct for generation if not handled by ppo_trainer.generate
                original_padding_side = tokenizer.padding_side
                tokenizer.padding_side = "left" # Important for decoder-only models
                
                response_tensors_list = ppo_trainer.generate(
                    [query_tensor_policy], # Pass as a list, using the augmented prompt tensor
                    return_prompt=False, 
                    **generation_kwargs
                )
                response_with_ad_ids = response_tensors_list[0].squeeze() # Get the first (and only) response tensor
                
                tokenizer.padding_side = original_padding_side # Restore padding side

                response_with_ad = tokenizer.decode(response_with_ad_ids.cpu(), skip_special_tokens=True)

                # 2. Generate response without ad from the reference model (for detectability)
                response_without_ad = ""
                if ref_model:
                    input_ids_ref = tokenizer.encode(vague_query_text, return_tensors="pt").to(ref_model.device) # ref_model might be on different device
                    
                    # Ensure ref_model has generation_config and it's set up correctly
                    if not hasattr(ref_model, 'generation_config') or ref_model.generation_config is None:
                        # Try to load from base_model_name if available, else use actor's and adapt
                        # This requires base_model_name to be available or passed to this function
                        # For simplicity, let's assume it's configured or try to use a default.
                        # This part might need refinement if base_model_name isn't directly available here.
                        # A quick fix: copy from actor_model's config if not set on ref_model
                        if hasattr(actor_model, 'generation_config') and actor_model.generation_config is not None:
                            ref_model.generation_config = actor_model.generation_config
                        else: # Fallback: Initialize a basic one. This might miss specific settings.
                            ref_model.generation_config = GenerationConfig(pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                            logger.warning("ref_model.generation_config was not set, initialized a basic one for validation.")
                    
                    # Ensure pad_token_id and eos_token_id are set in ref_model's generation_config
                    if ref_model.generation_config.pad_token_id is None: ref_model.generation_config.pad_token_id = tokenizer.pad_token_id
                    if ref_model.generation_config.eos_token_id is None: ref_model.generation_config.eos_token_id = tokenizer.eos_token_id


                    response_without_ad_ids = ref_model.generate(
                        input_ids_ref,
                        max_new_tokens=generation_kwargs.get("max_new_tokens", 128), # Use from generation_kwargs
                        pad_token_id=ref_model.generation_config.pad_token_id, # Use from ref_model's config
                        eos_token_id=ref_model.generation_config.eos_token_id, # Use from ref_model's config
                        # Pass other relevant kwargs from generation_kwargs, ensuring compatibility
                        top_k=generation_kwargs.get("top_k", 0.0),
                        top_p=generation_kwargs.get("top_p", 1.0),
                        do_sample=generation_kwargs.get("do_sample", True),
                        # temperature=generation_kwargs.get("temperature", 0.7) # if used
                    )
                    response_without_ad = tokenizer.decode(response_without_ad_ids.squeeze().cpu(), skip_special_tokens=True)
                else:
                    logger.warning("Reference model not provided for validation, detectability score will be 0 or based on empty string.")


                # 3. Judge the responses
                with ThreadPoolExecutor(max_workers=JUDGE_MAX_WORKERS) as executor:
                    future_coh = executor.submit(judge_coherence, prompt_for_policy_model, response_with_ad) # Judge coherence against the full prompt policy saw
                    future_help = executor.submit(judge_helpfulness, prompt_for_policy_model, response_with_ad) # Judge helpfulness against the full prompt policy saw
                    future_sal = executor.submit(judge_ad_salience, prompt_for_policy_model, response_with_ad, ad_text_for_judge) # Judge salience against the full prompt policy saw
                    future_det = executor.submit(judge_detectability, response_with_ad, response_without_ad)
                
                    score_coh_dict = future_coh.result()
                    score_help_dict = future_help.result()
                    score_sal_dict = future_sal.result()
                    score_det_dict = future_det.result()

                coh_score = float(score_coh_dict.get("Coherence Score", 0.0))
                help_score = float(score_help_dict.get("Helpfulness Score", 0.0))
                sal_score = float(score_sal_dict.get("Ad Salience Score", 0.0))
                det_score = float(score_det_dict.get("detectability_cosine", 0.0) or 0.0)

                all_coh_scores.append(coh_score)
                all_help_scores.append(help_score)
                all_sal_scores.append(sal_score)
                all_det_scores.append(det_score)
                
                current_reward_values = [coh_score, help_score, sal_score, det_score]
                final_reward = sum(current_reward_values) / len(current_reward_values) if current_reward_values else 0.0
                all_validation_rewards.append(final_reward)

            except Exception as e:
                logger.error(f"Error during validation for query {idx} ('{vague_query_text}'): {e}")
                all_validation_rewards.append(0.0) # Append a 0 reward for errored samples
                # Also append 0 for individual scores to maintain list length
                all_coh_scores.append(0.0)
                all_help_scores.append(0.0)
                all_sal_scores.append(0.0)
                all_det_scores.append(0.0)
                continue
    
    avg_reward = sum(all_validation_rewards) / len(all_validation_rewards) if all_validation_rewards else 0.0
    avg_coh = sum(all_coh_scores) / len(all_coh_scores) if all_coh_scores else 0.0
    avg_help = sum(all_help_scores) / len(all_help_scores) if all_help_scores else 0.0
    avg_sal = sum(all_sal_scores) / len(all_sal_scores) if all_sal_scores else 0.0
    avg_det = sum(all_det_scores) / len(all_det_scores) if all_det_scores else 0.0

    logger.info(f"✅ Validation epoch complete. Average Reward: {avg_reward:.4f}")
    logger.info(f"  Avg Coherence: {avg_coh:.4f}, Avg Helpfulness: {avg_help:.4f}, Avg Salience: {avg_sal:.4f}, Avg Detectability: {avg_det:.4f}")

    # Log to PPOTrainer's logger if available (e.g., for WandB)
    if hasattr(ppo_trainer, 'accelerator') and hasattr(ppo_trainer.accelerator, 'log'):
        ppo_trainer.accelerator.log({
            "validation/avg_reward": avg_reward,
            "validation/avg_coherence": avg_coh,
            "validation/avg_helpfulness": avg_help,
            "validation/avg_salience": avg_sal,
            "validation/avg_detectability": avg_det,
        }) # Step will be managed by PPOTrainer's main loop step

    actor_model.train() # Set model back to training mode
    # ref_model does not need to be set to train as it's frozen

    return {"avg_reward": avg_reward, "avg_coherence": avg_coh, "avg_helpfulness": avg_help, "avg_salience": avg_sal, "avg_detectability": avg_det}

def run_manual_ppo(model, tokenizer, base_model_name: str, checkpoint_dir_str: str, hf_token: str = None):
    device = model.device if model is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) # Fallback device if model not loaded yet
    
    # --- Setup paths and load data early ---
    df = pd.read_csv("data/merged_queries_ads.csv") # TODO: Make data_file a parameter from config in main.py and pass here
    base_dir = Path(checkpoint_dir_str)
    log_dir = Path("logs") # Central log directory
    query_checkpoint_path = base_dir / "last_query_position.json"
    # -------------------------------------

    # --- Check for existing progress and attempt to load TRL checkpoint ---
    resume_from_query = None
    start_idx = 0
    actor_model = model # Use passed-in model if available
    # Tokenizer is also passed in with model, or loaded below

    loaded_from_specific_checkpoint = False

    if actor_model is None: # Only try to load if a model wasn't explicitly passed in
        if query_checkpoint_path.exists():
            try:
                with open(query_checkpoint_path, "r") as f:
                    query_checkpoint = json.load(f)
                    resume_from_query = query_checkpoint.get("last_processed_query")
                    last_batch_idx_chkpt = query_checkpoint.get("batch_idx")
                    
                    if resume_from_query is not None and last_batch_idx_chkpt is not None:
                        checkpoint_load_dir = base_dir / f"checkpoint_query_{resume_from_query}_batch_{last_batch_idx_chkpt}"
                        if checkpoint_load_dir.exists() and checkpoint_load_dir.is_dir():
                            logger.info(f"✅ Resuming from TRL checkpoint: {checkpoint_load_dir}")
                            try:
                                actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                                    str(checkpoint_load_dir),
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    trust_remote_code=True,
                                    low_cpu_mem_usage=True,
                                    token=hf_token
                                )
                                tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_load_dir), trust_remote_code=True, token=hf_token)
                                logger.info(f"✅ Successfully loaded actor model and tokenizer from TRL checkpoint: {checkpoint_load_dir}")
                                loaded_from_specific_checkpoint = True
                                start_idx = resume_from_query + 1 # Set start_idx based on checkpoint
                            except Exception as e:
                                logger.error(f"❌ Failed to load model/tokenizer from TRL checkpoint {checkpoint_load_dir}: {e}. Will attempt to load base model.")
                        else:
                            logger.info(f"⚠️ TRL checkpoint directory {checkpoint_load_dir} not found, though last_query_position.json exists. Will load base model.")
                            if resume_from_query is not None: # Still resume data processing
                                start_idx = resume_from_query + 1
                    else: # query_checkpoint.json exists but is malformed or missing keys
                        logger.warning(f"⚠️ Query checkpoint file {query_checkpoint_path} is malformed. Will load base model and start fresh or from data start_idx 0.")
                        start_idx = 0 # Reset to 0 if checkpoint is unusable for positioning
                
            except Exception as e:
                logger.error(f"❌ Error reading query checkpoint {query_checkpoint_path}: {e}. Will load base model.")
                start_idx = 0 # Default to start if checkpoint reading fails
        else: # No query_checkpoint.json
            logger.info("ℹ️ No last_query_position.json found. Starting fresh training or from model's own checkpoint if any.")
            start_idx = 0
    else: # A model was passed into run_manual_ppo
        logger.info("ℹ️ Using model and tokenizer passed directly to run_manual_ppo.")
        # device needs to be derived from the passed model
        device = actor_model.device
        # start_idx will be 0 unless `last_query_position.json` logic for data skipping is added here too for this case
        # For now, if model is passed, assume fresh data run or data handling is external.
        # To be complete, we could also check for last_query_position.json here to set start_idx for data,
        # even if model is provided.
        if query_checkpoint_path.exists():
            try:
                with open(query_checkpoint_path, "r") as f:
                    query_checkpoint = json.load(f)
                    resume_from_query_val = query_checkpoint.get("last_processed_query")
                    if resume_from_query_val is not None:
                        start_idx = resume_from_query_val + 1
                        logger.info(f"Model passed directly, but will resume data from query index {start_idx} based on last_query_position.json.")
                    else:
                        start_idx = 0 # Default if key missing
            except Exception as e:
                logger.warning(f"Could not read last_query_position.json for data skipping when model is passed in: {e}")
                start_idx = 0
        else:
            start_idx = 0


    # --- Load Base Model and Tokenizer if not loaded from TRL checkpoint or passed in ---
    if actor_model is None: # If not loaded from TRL checkpoint and not passed in
        logger.info(f"Attempting to load base actor model {base_model_name} with ValueHead...")
        try:
            actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token=hf_token,
            )
            logger.info(f"✅ Successfully loaded base actor model {base_model_name} with ValueHead.")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to load base model {base_model_name} with ValueHead: {e}")
            raise
    
    if tokenizer is None: # If not loaded from TRL checkpoint and not passed in
        logger.info(f"Attempting to load base tokenizer {base_model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=hf_token)
            logger.info(f"✅ Successfully loaded base tokenizer for {base_model_name}.")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to load tokenizer for {base_model_name}: {e}")
            raise
            
    # Ensure tokenizer has pad_token set (important for TRL)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.info("Tokenizer does not have a pad_token/pad_token_id. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id # Ensure ID is also set
    logger.info(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")

    # Update device if actor_model was loaded
    if actor_model is not None:
        # If device_map="auto" was used, model might be on multiple devices.
        # ppo_trainer.accelerator.device will be the primary device.
        # For single GPU or CPU, actor_model.device is fine.
        # Let's rely on ppo_trainer.accelerator.device later for most operations.
        pass 

    # --- Load Reference Model (always from base) ---
    ref_model = None
    logger.info(f"Attempting to load reference model {base_model_name}...")
    try:
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto", # Can also be set to the same device as actor_model if known
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        logger.info(f"✅ Successfully loaded and froze reference model {base_model_name}.")
    except Exception as e:
        logger.error(f"❌ Failed to load reference model {base_model_name}: {e}")
        # Depending on use, this might be critical or not. For detectability judge, it is.
        # For now, let it proceed, validation/judging will warn if ref_model is None.
        # raise # Or raise an error if ref_model is strictly required

    if actor_model is None or tokenizer is None: # Final check
        logger.error("❌ Actor model or Tokenizer is None after all loading attempts. Cannot proceed with PPO training.")
        return
    
    # --- PPO Configuration for TRL (Older Style) ---
    # Attempting to use PPOConfig again, assuming an older TRL version
    # based on the series of TypeErrors for PPOTrainer constructor arguments.
    # NOW: Reverting to TRL 0.17.0 standard arguments based on trl.__version__ output
    # HOWEVER, these are still causing TypeErrors. Reverting to the minimal set known to pass PPOConfig init.
    ppo_config = PPOConfig(
        learning_rate=PPO_LEARNING_RATE,
        batch_size=PPO_BATCH_SIZE,          # Was PPO_BATCH_SIZE (experience collection batch)
        mini_batch_size=PPO_MINI_BATCH_SIZE,  # Was PPO_MINI_BATCH_SIZE (gradient update batch)
        gradient_accumulation_steps=1,      # Kept from previous minimal version
        # ppo_epochs=PPO_EPOCHS,                   # Removed again due to persistent TypeError
        gamma=PPO_GAMMA,                    # Kept
        lam=PPO_LAMBDA,                     # Kept
        # cliprange=PPO_CLIP_EPSILON,              # Removed again
        # cliprange_value=PPO_CLIP_EPSILON,      # Removed again
        vf_coef=VF_COEFF,                   # Kept
        # adap_kl_ctrl=(TARGET_KL is not None and TARGET_KL > 0), # Removed again
        # init_kl_coef=KL_COEFF,                   # Removed again
        # target_kl=TARGET_KL if (TARGET_KL is not None and TARGET_KL > 0) else None, # Removed again
        seed=42,                            # Kept
    )
    # --------------------------------------------------

    # --- Initialize PPOTrainer ---
    # Using keyword arguments based on TRL 0.17.0 signature
    # NOW: Using the new signature provided by the user.
    logger.info("Initializing PPOTrainer with new user-provided signature...")
    
    # Instantiate the dummy reward model
    dummy_reward_model_instance = DummyRewardModel().to(device) # Move to appropriate device early

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer, # Correctly passed
        model=actor_model,
        ref_model=ref_model,
        reward_model=dummy_reward_model_instance,
        train_dataset=df,  
        value_model=actor_model    
    )
    # Attempt to also set .tokenizer if the object allows it, to silence deprecation warnings
    if tokenizer is not None:
        try:
            ppo_trainer.tokenizer = tokenizer 
            logger.info("Also set ppo_trainer.tokenizer to potentially silence warnings.")
        except AttributeError:
            logger.info("Could not directly set ppo_trainer.tokenizer (it might be a property).")

    logger.info("✅ PPOTrainer initialized successfully.")
    print(f"ppo_trainer object methods: {dir(ppo_trainer)}")
    # --- End of PPOTrainer Initialization ---

    # --- Setup Detailed CSV Logger for Judgements --- 
    # Ensure the logs/judgements directory exists
    judgements_log_dir = Path("logs/judgements")
    judgements_log_dir.mkdir(parents=True, exist_ok=True)
    judging_log_path = judgements_log_dir / "judging_log.csv"
    
    # Determine if we need to write headers (i.e., if it's a fresh run for this log or file doesn't exist)
    write_headers = not judging_log_path.exists() or start_idx == 0
    
    # Open in append mode, create if doesn't exist
    # Using a context manager for the file is better if the whole loop is refactored, 
    # but for now, manual open/close is fine if handled in finally.
    try:
        detailed_judging_log_file = open(judging_log_path, "a" if not write_headers else "w", newline="", encoding="utf-8")
        detailed_judging_logger = csv.writer(detailed_judging_log_file)
        if write_headers:
            detailed_judging_logger.writerow([
                "global_query_idx", "ppo_batch_idx", "original_vague_query", 
                "full_prompt_to_policy", "response_with_ad", "response_without_ad",
                "coherence_score", "helpfulness_score", "salience_score", "detectability_score",
                "final_reward", "judge_coherence_raw_json", "judge_helpfulness_raw_json", 
                "judge_salience_raw_json", "judge_detectability_raw_json"
            ])
            detailed_judging_log_file.flush() # Ensure headers are written immediately
            logger.info(f"Initialized detailed judging log with headers at: {judging_log_path}")
        else:
            logger.info(f"Appending to existing detailed judging log at: {judging_log_path}")
    except IOError as e:
        logger.error(f"Could not open or write to judging_log.csv: {e}. Detailed judging logs will not be saved.")
        detailed_judging_logger = None # Ensure it's None so we don't try to use it
        detailed_judging_log_file = None
    # -----------------------------------------------------

    if loaded_from_specific_checkpoint:
        logger.info(f"✅ Resumed model and tokenizer from specific TRL checkpoint. Data processing starts from query index {start_idx}.")
    elif start_idx > 0 :
        logger.info(f"✅ Resuming data processing from query index {start_idx} with a base or pre-passed model.")
    else:
        logger.info("ℹ️ Starting fresh training run with base or pre-passed model from query index 0.")
    
    # --- Resume data processing ---
    total_rows = len(df)
    if start_idx >= total_rows:
        logger.info("✅ Training already completed based on last_query_position.json!")
        return

    df_to_process = df.iloc[start_idx:]
    
    # Validation data handling (can be refined later)
    val_set_abs_size = VALIDATION_SET_SIZE
    val_set_ratio_size = int(len(df_to_process) * VALIDATION_SET_RATIO)
    validation_size = min(val_set_abs_size, val_set_ratio_size) if val_set_ratio_size > 0 else val_set_abs_size
    validation_size = min(validation_size, len(df_to_process) // 2) 

    validation_data = pd.DataFrame()
    if validation_size > 0 and len(df_to_process) > validation_size:
        validation_data = df_to_process.sample(n=validation_size, random_state=42)
        df_to_process = df_to_process.drop(validation_data.index)
    elif len(df_to_process) <= validation_size:
        logger.warning("Not enough data for a separate validation set after resuming/filtering. Skipping validation.")
        validation_data = pd.DataFrame()
    else:
        logger.info("No validation data to sample based on configuration or available data.")

    # The old DataProcessor class is largely replaced by direct logic here and PPOTrainer.
    # Stats logging will be handled via PPOTrainer's logging (e.g., to wandb) or custom logging.

    # --- TRL Generation Kwargs ---
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        # "temperature": 0.7, # Optional: Adjust temperature for sampling
        # "num_beams": 1, # Keep as 1 for sampling based PPO generation typically
    }
    # -----------------------------
    
    # Calculate starting batch index for tqdm progress bar if resuming
    # PPO_BATCH_SIZE is the number of queries processed before a ppo_trainer.step()
    # The loop iterates with PPO_BATCH_SIZE steps over df_to_process.
    # If start_idx is, e.g., 160, and PPO_BATCH_SIZE is 16, initial_batch_idx_for_tqdm should be 10.
    initial_batch_idx_for_tqdm = (start_idx - (start_idx % PPO_BATCH_SIZE)) // PPO_BATCH_SIZE if PPO_BATCH_SIZE > 0 else 0
    if resume_from_query is not None and 'last_batch_idx_chkpt' in locals() and last_batch_idx_chkpt is not None:
        # If we have a checkpointed batch_idx, use that as a more reliable start for tqdm
        # Ensure it aligns with data already processed by start_idx
        # The loop batch_idx is relative to df_to_process, not absolute.
        # The tqdm range should be `len(df_to_process) // PPO_BATCH_SIZE`
        # For resuming tqdm, we need to consider how many full PPO_BATCH_SIZE chunks were completed in previous run up to start_idx.
        pass # tqdm will start from 0 for the current df_to_process

    num_ppo_batches = (len(df_to_process) + PPO_BATCH_SIZE - 1) // PPO_BATCH_SIZE # Ceiling division

    try:
        # Loop over the dataset in PPO_BATCH_SIZE chunks for experience collection
        for batch_idx in tqdm(range(num_ppo_batches), desc="TRL PPO Training", initial=0):
            batch_start_offset = batch_idx * PPO_BATCH_SIZE
            batch_end_offset = min(batch_start_offset + PPO_BATCH_SIZE, len(df_to_process))
            
            current_experience_batch_df = df_to_process.iloc[batch_start_offset:batch_end_offset]
            if current_experience_batch_df.empty:
                logger.info(f"No more data to process in df_to_process at batch_idx {batch_idx}. Ending.")
                break

            # query_texts = [str(row['vague_query']) for _, row in current_experience_batch_df.iterrows()]
            # Constructing new query_texts that include ad information for the policy model
            query_texts = []
            for _, row in current_experience_batch_df.iterrows():
                vague_query = str(row['vague_query'])
                ad_product = str(row['ad_product'])
                brand = str(row['brand'])
                ad_description = str(row['ad_description'])
                # ad_url = str(row['url']) # Optionally include URL if desired
                
                prompt_for_policy_model = f"""Here is a query: {vague_query}

Please generate a helpful response. If appropriate, seamlessly integrate the following advertisement into your response:
Ad Details:
Product: {ad_product}
Brand: {brand}
Description: {ad_description}

Response:"""
                query_texts.append(prompt_for_policy_model)
            
            original_padding_side = None
            if hasattr(ppo_trainer, 'processing_class') and hasattr(ppo_trainer.processing_class, 'padding_side'):
                original_padding_side = ppo_trainer.processing_class.padding_side
                ppo_trainer.processing_class.padding_side = "left"
                logger.info(f"Set ppo_trainer.processing_class.padding_side to left for generation.")
            else:
                logger.warning("Could not set padding_side on ppo_trainer.processing_class. Ensure tokenizer used for encoding has padding_side='left' for generation.")

            # Tokenize and get attention masks
            tokenized_queries = []
            for q_text in query_texts:
                encoded_dict = tokenizer(
                    q_text, 
                    return_tensors="pt", 
                    padding=True, # Pad to max length in batch, or a specified max_length
                    truncation=True, # Truncate if longer than model can handle
                    # max_length=512 # Optional: set a max_length for consistency if not padding to batch max
                )
                tokenized_queries.append(encoded_dict)
            # query_tensors now refers to the list of dicts from the tokenizer

            # ppo_trainer.tokenizer.padding_side = "right" # Set back to right for training if needed, though TRL might handle.
            if original_padding_side is not None and hasattr(ppo_trainer, 'processing_class') and hasattr(ppo_trainer.processing_class, 'padding_side'):
                ppo_trainer.processing_class.padding_side = original_padding_side
                logger.info(f"Restored ppo_trainer.processing_class.padding_side to {original_padding_side}.")

            query_tensors = [q['input_ids'] for q in tokenized_queries if q['input_ids'].nelement() > 0 and q['input_ids'].shape[1] > 0]
            if not query_tensors:
                logger.warning(f"Batch {batch_idx} resulted in no valid query tensors. Skipping.")
                continue
            
            logger.info(f"🔄 Batch {batch_idx} - Generating responses for {len(query_tensors)} queries...")
            # response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
            # Using the policy_model directly for generation as ppo_trainer.generate() is not available
            
            policy_device = ppo_trainer.policy_model.device
            response_tensors_list = []
            for tokenized_input in tokenized_queries: # Iterate over the list of tokenized dicts
                input_ids_on_policy_device = tokenized_input['input_ids'].to(policy_device)
                attention_mask_on_policy_device = tokenized_input['attention_mask'].to(policy_device)
                
                response_tensor = ppo_trainer.policy_model.generate(
                    input_ids=input_ids_on_policy_device, 
                    attention_mask=attention_mask_on_policy_device, # Pass attention_mask
                    # return_prompt=False, # Not a standard HF generate param, remove.
                    **generation_kwargs
                )
                response_tensors_list.append(response_tensor)
            response_tensors = response_tensors_list # Keep variable name for downstream consistency
            
            response_texts_with_ad = []
            for i in range(len(response_tensors)):
                response_tokens = response_tensors[i].cpu().squeeze().tolist()
                decoded_text = ppo_trainer.tokenizer.decode(response_tokens, skip_special_tokens=True)
                response_texts_with_ad.append(decoded_text)

            if len(query_texts) != len(response_texts_with_ad):
                logger.error(f"Batch {batch_idx} - Mismatch num queries ({len(query_texts)}) and responses ({len(response_texts_with_ad)}). Skipping.")
                continue
            logger.info(f"✅ Batch {batch_idx} - Generated {len(response_texts_with_ad)} responses.")

            rewards_list = []
            logger.info(f"🔄 Batch {batch_idx} - Judging {len(response_texts_with_ad)} responses...")
            
            for i, query_text_for_reward in enumerate(query_texts): # Use query_texts which matches response_texts_with_ad order
                try:
                    response_with_ad_for_reward = response_texts_with_ad[i]
                    original_row = current_experience_batch_df.iloc[i]
                    ad_facts = {
                        "ad_product": str(original_row['ad_product']), "brand": str(original_row['brand']),
                        "url": str(original_row['url']), "description": str(original_row['ad_description']),
                    }
                    ad_text_for_judge = f"""Product: {ad_facts['ad_product']}
Brand: {ad_facts['brand']}
URL: {ad_facts['url']}
Description: {ad_facts['description']}"""
                    
                    # Use the original vague query for the reference model to get a response without ad knowledge
                    vague_query_for_ref_model_input = str(original_row['vague_query'])
                    input_ids_ref = tokenizer.encode(vague_query_for_ref_model_input, return_tensors="pt").to(ref_model.device)
                    if not hasattr(ref_model, 'generation_config') or ref_model.generation_config is None:
                        ref_model.generation_config = GenerationConfig.from_pretrained(base_model_name, token=hf_token)
                    if ref_model.generation_config.pad_token_id is None: ref_model.generation_config.pad_token_id = tokenizer.pad_token_id
                    if ref_model.generation_config.eos_token_id is None: ref_model.generation_config.eos_token_id = tokenizer.eos_token_id

                    with torch.no_grad():
                        response_without_ad_ids = ref_model.generate(
                            input_ids_ref, max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                            top_k=generation_kwargs["top_k"], top_p=generation_kwargs["top_p"], do_sample=generation_kwargs["do_sample"],
                        )
                    response_without_ad = tokenizer.decode(response_without_ad_ids.squeeze().cpu(), skip_special_tokens=True)

                    with ThreadPoolExecutor(max_workers=JUDGE_MAX_WORKERS) as executor:
                        future_coh = executor.submit(judge_coherence, query_text_for_reward, response_with_ad_for_reward)
                        future_help = executor.submit(judge_helpfulness, query_text_for_reward, response_with_ad_for_reward)
                        future_sal = executor.submit(judge_ad_salience, query_text_for_reward, response_with_ad_for_reward, ad_text_for_judge)
                        future_det = executor.submit(judge_detectability, response_with_ad_for_reward, response_without_ad)
                        score_coh_dict = future_coh.result()
                        score_help_dict = future_help.result()
                        score_sal_dict = future_sal.result()
                        score_det_dict = future_det.result()
                    
                    current_reward_values = [
                        score_coh_dict.get("Coherence Score", 0), score_help_dict.get("Helpfulness Score", 0),
                        score_sal_dict.get("Ad Salience Score", 0), score_det_dict.get("detectability_cosine", 0) or 0.0
                    ]
                    current_reward_values = [float(v) for v in current_reward_values]

                    coh_score = current_reward_values[0]
                    help_score = current_reward_values[1]
                    sal_score = current_reward_values[2]
                    det_score = current_reward_values[3]

                    final_reward = sum(current_reward_values) / len(current_reward_values) if current_reward_values else 0.0
                    rewards_list.append(torch.tensor(final_reward, device=ppo_trainer.accelerator.device))

                    # This assumes original_row['vague_query'] is the simple query
                    original_vague_query_for_log = str(original_row['vague_query'])
                    current_global_query_idx_for_log = start_idx + current_experience_batch_df.index[i] # Get current global index

                    if detailed_judging_logger:
                        detailed_judging_logger.writerow([
                            current_global_query_idx_for_log,
                            batch_idx, 
                            original_vague_query_for_log,
                            query_text_for_reward, # This is the full prompt with ad details
                            response_with_ad_for_reward, 
                            response_without_ad,
                            coh_score, 
                            help_score, 
                            sal_score, 
                            det_score,
                            final_reward.item() if isinstance(final_reward, torch.Tensor) else final_reward,
                            json.dumps(score_coh_dict), 
                            json.dumps(score_help_dict),
                            json.dumps(score_sal_dict), 
                            json.dumps(score_det_dict)
                        ])
                        if detailed_judging_log_file: detailed_judging_log_file.flush()

                except Exception as e:
                    logger.error(f"Error in reward calculation (Batch {batch_idx}, Query {i}): {e}. Appending 0 reward.")
                    rewards_list.append(torch.tensor(0.0, device=ppo_trainer.accelerator.device))
            
            if len(query_tensors) != len(rewards_list):
                 logger.error(f"Batch {batch_idx} - Mismatch query_tensors ({len(query_tensors)}) vs rewards_list ({len(rewards_list)}) before PPO step. FATAL.")
                 # This should ideally be caught by padding rewards for errors above. If still mismatch, something is very wrong.
                 continue # Skip PPO step for this batch to be safe
            if len(response_tensors) != len(rewards_list):
                 # This can happen if some generations failed or were filtered out, but rewards were still (expected to be) computed for original queries.
                 # The ppo_trainer.step() needs query, response, reward to be aligned for the samples it processes.
                 # We need to ensure query_tensors, response_tensors, and rewards_list are aligned and have the same length.
                 # The current loop structure implies they *should* be if error handling in reward calc is robust.
                 # For now, let's assume they are aligned. If not, TRL will likely error out.
                 logger.warning(f"Batch {batch_idx} - Mismatch response_tensors ({len(response_tensors)}) vs rewards_list ({len(rewards_list)}). Attempting to proceed but check alignment.")

            if not query_tensors or not response_tensors or not rewards_list:
                logger.warning(f"Batch {batch_idx} - Empty inputs for PPO step. Skipping PPO update.")
                continue
            
            # Ensure all lists passed to step have same length
            min_len = min(len(query_tensors), len(response_tensors), len(rewards_list))
            if min_len < len(query_tensors) or min_len < len(response_tensors) or min_len < len(rewards_list):
                logger.warning(f"Batch {batch_idx} - Trimming inputs to PPO step to minimum common length: {min_len}")
                query_tensors = query_tensors[:min_len]
                response_tensors = response_tensors[:min_len]
                rewards_list = rewards_list[:min_len]
            
            if not query_tensors: # Check again after potential trimming
                logger.warning(f"Batch {batch_idx} - No data left for PPO step after trimming. Skipping.")
                continue

            logger.info(f"🔄 Batch {batch_idx} - Performing PPO update step with {len(query_tensors)} samples.")
            train_stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
            logger.info(f"✅ Batch {batch_idx} - PPO update. Stats: {train_stats}")

            if 'ppo/mean_rewards' in train_stats: logger.info(f"Batch {batch_idx} - Mean reward (TRL): {train_stats['ppo/mean_rewards']:.4f}")
            if 'ppo/loss/total' in train_stats: logger.info(f"Batch {batch_idx} - Total PPO loss (TRL): {train_stats['ppo/loss/total']:.4f}")

            current_global_query_idx = start_idx + batch_end_offset -1 

            if batch_idx > 0 and batch_idx % VALIDATION_INTERVAL_BATCHES == 0:
                if not validation_data.empty:
                    logger.info(f"📋 Running validation at PPO batch_idx {batch_idx} (global query ~{current_global_query_idx})")
                    validation_stats = run_trl_validation_epoch(
                        ppo_trainer=ppo_trainer,
                        ref_model=ref_model, # Pass the ref_model used in the main loop
                        validation_data=validation_data,
                        generation_kwargs=generation_kwargs,
                        hf_token=hf_token 
                    )
                    logger.info(f"Validation Stats at PPO batch_idx {batch_idx}: {validation_stats}")
                else:
                    logger.info(f"No validation data available, skipping validation at PPO batch_idx {batch_idx}.")
            
            if batch_idx > 0 and batch_idx % CHECKPOINT_INTERVAL_BATCHES == 0:
                logger.info(f"🔄 Batch {batch_idx} - Saving TRL checkpoint at global query index {current_global_query_idx}...")
                checkpoint_save_dir = base_dir / f"checkpoint_query_{current_global_query_idx}_batch_{batch_idx}"
                checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
                ppo_trainer.save_model(str(checkpoint_save_dir))
                # tokenizer.save_pretrained(str(checkpoint_save_dir)) # PPOTrainer saves tokenizer if it's part of its init.
                
                query_chkpt_data = {
                    "last_processed_query": int(current_global_query_idx),
                    "batch_idx": batch_idx, # This is the ppo batch index for df_to_process
                    "timestamp": time.time()
                }
                with open(query_checkpoint_path, "w") as f: json.dump(query_chkpt_data, f)
                logger.info(f"✅ TRL Checkpoint saved in: {checkpoint_save_dir}")

            if batch_idx > 0 and batch_idx % CHECKPOINT_INTERVAL_BATCHES == 0: 
                clear_caches()
                clear_response_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopping training...")
        
        # `batch_idx` is from the PPO loop over df_to_process
        # `start_idx` is the global starting index for the current df_to_process
        
        # Try to get the info from the last *scheduled* checkpoint if one exists
        # This ensures that the saved model corresponds to the saved query position.
        saved_query_chkpt_data_on_interrupt = None
        interrupt_checkpoint_save_dir = None

        if query_checkpoint_path.exists():
            try:
                with open(query_checkpoint_path, "r") as f:
                    # This file should contain info about the last *successfully completed and saved* scheduled checkpoint
                    saved_query_chkpt_data_on_interrupt = json.load(f)
                    logger.info(f"Found existing query checkpoint from last scheduled save: {saved_query_chkpt_data_on_interrupt}")
            except Exception as e:
                logger.error(f"Error reading existing query checkpoint during interrupt: {e}. Will treat as no prior checkpoint.")
                saved_query_chkpt_data_on_interrupt = None # Ensure it's None if read fails

        if saved_query_chkpt_data_on_interrupt and \
           isinstance(saved_query_chkpt_data_on_interrupt, dict) and \
           "last_processed_query" in saved_query_chkpt_data_on_interrupt and \
           "batch_idx" in saved_query_chkpt_data_on_interrupt:
            
            # Save the model with a name corresponding to the last successfully scheduled checkpoint
            # The model in memory *is* the one from that last successful PPO step.
            lpd_val = saved_query_chkpt_data_on_interrupt['last_processed_query']
            bi_val = saved_query_chkpt_data_on_interrupt['batch_idx']
            logger.info(f"Attempting to save model state corresponding to last scheduled checkpoint: query {lpd_val}, batch {bi_val}")
            interrupt_checkpoint_save_dir = base_dir / f"checkpoint_query_{lpd_val}_batch_{bi_val}_INTERRUPT"
            # No need to rewrite last_query_position.json, as it already reflects this correct past state.
        
        # Condition for being in the very first PPO batch of a completely fresh run:
        # - `actor_model` exists (so models were loaded).
        # - `start_idx` is 0 (meaning no prior data was skipped based on a query_checkpoint_path).
        # - EITHER `batch_idx` is not yet defined (interrupt before PPO loop starts) OR `batch_idx` is 0 (interrupt in first PPO batch).
        # - AND no valid `saved_query_chkpt_data_on_interrupt` was found (or it was malformed).
        elif (saved_query_chkpt_data_on_interrupt is None) and \
             ('actor_model' in locals() and actor_model is not None) and \
             ('start_idx' in locals() and start_idx == 0) and \
             (('batch_idx' not in locals()) or ('batch_idx' in locals() and batch_idx == 0)):

            logger.info("Interrupted during the first PPO batch of a fresh run (no prior scheduled checkpoint). Saving model as initial.")
            interrupt_checkpoint_save_dir = base_dir / "checkpoint_query_initial_INTERRUPT"
            # Save a query_checkpoint that indicates resuming from the very beginning
            # This is crucial because no PPO step has completed, so no progress beyond query -1.
            query_chkpt_data = {
                "last_processed_query": -1, # To make start_idx = 0 on resume
                "batch_idx": -1, # Indicates before first batch processing for TRL checkpoint loading logic
                "timestamp": time.time()
            }
            # Ensure directory exists for query_checkpoint_path before writing
            query_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(query_checkpoint_path, "w") as f:
                json.dump(query_chkpt_data, f)
            logger.info(f"Saved query position for resuming from absolute start: {query_checkpoint_path}")
        else:
            # This case means: 
            # - An interrupt occurred.
            # - There wasn't a valid last scheduled checkpoint in query_checkpoint_path (e.g., file missing, malformed, or this is first run beyond first batch but before first scheduled checkpoint).
            # - And it's not the very specific case of the first batch of a fresh run.
            # To be safe and avoid saving a model that doesn't match any recorded progress point, we don't save the model.
            logger.warning("Could not determine a reliable prior checkpoint state to associate with the interrupted model. Model will not be saved on interrupt. Resume will use last scheduled checkpoint if any.")
            interrupt_checkpoint_save_dir = None

        if ppo_trainer and interrupt_checkpoint_save_dir:
            try:
                logger.info(f"Saving model checkpoint due to interrupt at: {interrupt_checkpoint_save_dir}")
                interrupt_checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
                ppo_trainer.save_model(str(interrupt_checkpoint_save_dir))
                logger.info(f"✅ Interrupted model checkpoint saved in: {interrupt_checkpoint_save_dir}")
            except Exception as e:
                logger.error(f"❌ Unexpected error during interrupt model save attempt: {e}")
                logger.exception("Detailed traceback:")
        elif ppo_trainer and not interrupt_checkpoint_save_dir:
            logger.info("No model saved on interrupt due to undetermined checkpoint state.")
        
    finally:
        # Ensure all logs are flushed before exiting
        # if processor: # processor object is removed
        #     processor._flush_logs()
        # Cleanup temporary directory
        if ppo_trainer and hasattr(ppo_trainer, 'temp_dir') and ppo_trainer.temp_dir.exists():
            shutil.rmtree(ppo_trainer.temp_dir)

        if detailed_judging_log_file: # Close the detailed log file
            try:
                detailed_judging_log_file.close()
                logger.info(f"Closed detailed judging log: {judging_log_path}")
            except Exception as e:
                logger.error(f"Error closing detailed judging log: {e}")

    logger.info("✅ PPO training complete. Log saved to logs/ppo_manual_log.csv")
    logger.info(f"All checkpoints and metrics saved in: {base_dir}")
