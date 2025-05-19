import os
import torch
import pandas as pd
import gc
from tqdm import tqdm
# Removed F from torch.nn import functional as F, as it was mainly for PPO log_probs
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import json
from pathlib import Path
import logging
import requests # For robust session
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Attempting common relative import structure, assuming PYTHONPATH includes project root
from src.judge import (
    judge_coherence,
    judge_helpfulness,
    judge_ad_salience,
)
from src.generate.generator import generate_response_with_ad, clear_response_cache
from src.judge.utils import clear_caches 
# CheckpointManager is not used in this baseline script for model saving during the loop.
# Model loading will be handled directly in run_baseline_evaluation.

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure requests for reliability
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=10,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

class BaselineDataProcessor:
    def __init__(self, model, tokenizer, device, logs_base_dir: Path, batch_size=32, base_model_name=None, hf_token=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.base_model_name = base_model_name # For reference if needed
        self.hf_token = hf_token # For reference if needed
        self.dataset_start_idx = 0 # For resuming processing of the main dataframe
        self.logs_base_dir = logs_base_dir # Base directory for all logs of this run

        # Create directories for intermediate results under logs_base_dir
        self.generation_dir = self.logs_base_dir / "generations"
        self.judging_dir = self.logs_base_dir / "judgments"
        self.evaluation_dir = self.logs_base_dir / "evaluations"
        self.stats_dir = self.logs_base_dir / "stats"
        
        for dir_path in [self.generation_dir, self.judging_dir, self.evaluation_dir, self.stats_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.generation_log = self.generation_dir / "generation_log.csv"
        self.judging_log = self.judging_dir / "judging_log.csv"
        self.evaluation_log = self.evaluation_dir / "evaluation_log.csv"
        self.stats_log = self.stats_dir / "evaluation_stats.csv"
        
        self.generation_log_buffer = []
        self.judging_log_buffer = []
        self.evaluation_log_buffer = []
        
        # Create headers for log files
        if not self.generation_log.exists() or os.path.getsize(self.generation_log) == 0:
            pd.DataFrame(columns=[
                "run_batch_idx", "global_item_idx", "original_query_idx", "query", "ad_facts", 
                "response_with_ad", "generation_time", "token_count"
            ]).to_csv(self.generation_log, index=False)
        
        if not self.judging_log.exists() or os.path.getsize(self.judging_log) == 0:
            pd.DataFrame(columns=[
                "run_batch_idx", "global_item_idx", "original_query_idx", "query", "response_with_ad", 
                "coherence_score", "helpfulness_score", "salience_score", 
                "coherence_explanation", "helpfulness_explanation", "salience_explanation", "judging_time"
            ]).to_csv(self.judging_log, index=False)
        
        if not self.evaluation_log.exists() or os.path.getsize(self.evaluation_log) == 0:
            pd.DataFrame(columns=[
                "run_batch_idx", "global_item_idx", "original_query_idx", "query", "response_with_ad", "reward"
            ]).to_csv(self.evaluation_log, index=False)
        
        self.stats = {
            "total_items_processed": 0, "successful_items": 0, "failed_items": 0,
            "total_generation_time": 0, "total_judging_time": 0, "total_tokens_generated": 0, "avg_reward": 0.0
        }
        self._load_stats()
        
        if not self.stats_log.exists() or os.path.getsize(self.stats_log) == 0:
            pd.DataFrame(columns=[
                "run_batch_idx", "global_item_idx", "timestamp", "total_items_processed", "successful_items", "failed_items",
                "avg_generation_time", "avg_judging_time", "avg_reward", "total_tokens_generated", "memory_usage_mb"
            ]).to_csv(self.stats_log, index=False)

    def _load_stats(self):
        try:
            if self.stats_log.exists() and os.path.getsize(self.stats_log) > 0:
                logger.info(f"Loading existing evaluation statistics from {self.stats_log}")
                stats_df = pd.read_csv(self.stats_log)
                if not stats_df.empty:
                    latest_stats = stats_df.iloc[-1]
                    self.stats["total_items_processed"] = int(latest_stats.get("total_items_processed", 0))
                    self.stats["successful_items"] = int(latest_stats.get("successful_items", 0))
                    self.stats["failed_items"] = int(latest_stats.get("failed_items", 0))
                    # Recalculate total times from averages and counts
                    self.stats["total_generation_time"] = latest_stats.get("avg_generation_time", 0) * self.stats["successful_items"]
                    self.stats["total_judging_time"] = latest_stats.get("avg_judging_time", 0) * self.stats["successful_items"]
                    self.stats["total_tokens_generated"] = int(latest_stats.get("total_tokens_generated", 0))
                    self.stats["avg_reward"] = float(latest_stats.get("avg_reward", 0.0))
                    logger.info(f"Resumed evaluation stats: {self.stats['total_items_processed']} items processed, avg_reward: {self.stats['avg_reward']:.4f}")
                else: logger.info("No existing evaluation stats found, starting fresh.")
            else: logger.info("No stats file exists, starting fresh.")
        except Exception as e: logger.error(f"Error loading stats: {e}. Starting fresh.")

    def _flush_logs(self):
        if self.generation_log_buffer:
            pd.DataFrame(self.generation_log_buffer).to_csv(self.generation_log, mode="a", header=False, index=False); self.generation_log_buffer.clear()
        if self.judging_log_buffer:
            pd.DataFrame(self.judging_log_buffer).to_csv(self.judging_log, mode="a", header=False, index=False); self.judging_log_buffer.clear()
        if self.evaluation_log_buffer:
            pd.DataFrame(self.evaluation_log_buffer).to_csv(self.evaluation_log, mode="a", header=False, index=False); self.evaluation_log_buffer.clear()
        logger.debug("Flushed logs to disk.") # Changed to debug for less verbose logging of flush

    def update_stats(self, run_batch_idx, global_item_idx, gen_time, judge_time, token_count, reward_value, success=True):
        self.stats["total_items_processed"] += 1
        if success:
            self.stats["successful_items"] += 1
            # Only update averages with successful items to prevent skew by 0s from failures
            n_success = self.stats["successful_items"]
            self.stats["avg_reward"] = (self.stats["avg_reward"] * (n_success - 1) + reward_value) / n_success if n_success > 0 else 0.0
            self.stats["total_generation_time"] += gen_time
            self.stats["total_judging_time"] += judge_time
            self.stats["total_tokens_generated"] += token_count
        else:
            self.stats["failed_items"] += 1
        
        # Log stats periodically (e.g., every 10 processed items)
        if self.stats["total_items_processed"] % 10 == 0:
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            n_success_for_avg = self.stats["successful_items"]
            avg_gen_time = self.stats["total_generation_time"] / n_success_for_avg if n_success_for_avg > 0 else 0
            avg_judge_time = self.stats["total_judging_time"] / n_success_for_avg if n_success_for_avg > 0 else 0

            pd.DataFrame([{
                "run_batch_idx": run_batch_idx, 
                "global_item_idx": global_item_idx,
                "timestamp": time.time(), 
                "total_items_processed": self.stats["total_items_processed"], 
                "successful_items": self.stats["successful_items"], 
                "failed_items": self.stats["failed_items"], 
                "avg_generation_time": avg_gen_time, 
                "avg_judging_time": avg_judge_time, 
                "avg_reward": self.stats["avg_reward"], 
                "total_tokens_generated": self.stats["total_tokens_generated"], 
                "memory_usage_mb": memory_usage
            }]).to_csv(self.stats_log, mode="a", header=False, index=False)
            logger.info(f"Stats updated at item {self.stats['total_items_processed']}. Avg Reward: {self.stats['avg_reward']:.4f}")

    def _run_judges_parallel(self, query, response_with_ad, ad_text):
        with ThreadPoolExecutor(max_workers=3) as executor: # Reduced from 4 to 3 judges
            future_coh = executor.submit(judge_coherence, query, response_with_ad)
            future_help = executor.submit(judge_helpfulness, query, response_with_ad)
            future_sal = executor.submit(judge_ad_salience, query, response_with_ad, ad_text)
            
            score_coh = future_coh.result()
            score_help = future_help.result()
            score_sal = future_sal.result()
        return score_coh, score_help, score_sal

    def process_single_item(self, original_query_idx, query_data, run_batch_idx, global_item_idx):
        query = str(query_data['vague_query'])
        ad_facts = {
            "ad_product": str(query_data['ad_product']), "brand": str(query_data['brand']),
            "url": str(query_data['url']), "description": str(query_data['ad_description']),
        }
        logger.info(f"Processing: RunBatch {run_batch_idx}, GlobalItem {global_item_idx}, OriginalQueryIdx {original_query_idx}")

        response_with_ad, gen_time, token_count = None, 0, 0
        score_coh, score_help, score_sal = {}, {}, {}
        judge_time = 0
        reward_value = 0.0
        success = False

        try:
            # Stage 1: Generation (only with ad)
            gen_start_time = time.time()
            with torch.no_grad():
                response_with_ad = generate_response_with_ad(query, ad_facts, self.model, self.tokenizer)
            gen_time = time.time() - gen_start_time
            token_count = len(self.tokenizer.encode(response_with_ad))
            self.generation_log_buffer.append({
                "run_batch_idx": run_batch_idx, "global_item_idx": global_item_idx, "original_query_idx": original_query_idx,
                "query": query, "ad_facts": json.dumps(ad_facts),
                "response_with_ad": response_with_ad,
                "generation_time": gen_time, "token_count": token_count
            })
            logger.debug(f"Generation for OriginalQueryIdx {original_query_idx} complete in {gen_time:.2f}s")

            # Stage 2: Judging (without detectability)
            judge_start_time = time.time()
            ad_text = f"Product: {ad_facts['ad_product']}\nBrand: {ad_facts['brand']}\nURL: {ad_facts['url']}\nDescription: {ad_facts['description']}"
            score_coh, score_help, score_sal = self._run_judges_parallel(query, response_with_ad, ad_text)
            judge_time = time.time() - judge_start_time
            self.judging_log_buffer.append({
                "run_batch_idx": run_batch_idx, "global_item_idx": global_item_idx, "original_query_idx": original_query_idx,
                "query": query, "response_with_ad": response_with_ad,
                "coherence_score": score_coh.get("Coherence Score", 0), "helpfulness_score": score_help.get("Helpfulness Score", 0),
                "salience_score": score_sal.get("Ad Salience Score", 0),
                "coherence_explanation": score_coh.get("Coherence Explanation", ""), "helpfulness_explanation": score_help.get("Helpfulness Explanation", ""),
                "salience_explanation": score_sal.get("Ad Salience Explanation", ""), "judging_time": judge_time
            })
            logger.debug(f"Judging for OriginalQueryIdx {original_query_idx} complete in {judge_time:.2f}s")

            # Stage 3: Reward Calculation (without detectability)
            reward_values = [
                score_coh.get("Coherence Score", 0), score_help.get("Helpfulness Score", 0),
                score_sal.get("Ad Salience Score", 0)
            ]
            reward_value = sum(reward_values) / len(reward_values) if reward_values else 0.0
            self.evaluation_log_buffer.append({
                "run_batch_idx": run_batch_idx, "global_item_idx": global_item_idx, "original_query_idx": original_query_idx,
                "query": query, "response_with_ad": response_with_ad, "reward": reward_value
            })
            logger.debug(f"Reward for OriginalQueryIdx {original_query_idx}: {reward_value:.4f}")
            success = True

        except Exception as e:
            logger.error(f"Error processing OriginalQueryIdx {original_query_idx}: {e}", exc_info=True)
        
        self.update_stats(run_batch_idx, global_item_idx, gen_time, judge_time, token_count, reward_value, success=success)
        
        # Save current position for resumability after each item is processed (success or fail)
        # This ensures that if the script crashes, it can resume from the very next item.
        # The file is stored in logs_base_dir to be specific to this evaluation run.
        resume_checkpoint_path = self.logs_base_dir / "baseline_resume_checkpoint.json"
        try:
            with open(resume_checkpoint_path, "w") as f:
                json.dump({"last_processed_global_item_idx": global_item_idx}, f)
        except Exception as e:
            logger.error(f"Failed to save resume checkpoint: {e}")

        return {
            "original_query_idx": original_query_idx, "query": query, "ad_facts": ad_facts,
            "response_with_ad": response_with_ad,
            "reward": reward_value, 
            "scores": {"coherence": score_coh, "helpfulness": score_help, "salience": score_sal},
            "processed_successfully": success
        }

    def process_batch(self, batch_data_df, run_batch_idx):
        """Processes a batch of data items (Pandas DataFrame)."""
        batch_results = []
        # dataset_start_idx is the starting global index for this entire run_baseline_evaluation call.
        # run_batch_idx is the current batch number within this run.
        # i is the 0-indexed counter within the current batch_data_df.
        # original_query_idx is the true index from the original, full dataset.
        
        for i, (original_query_idx, row_data) in enumerate(batch_data_df.iterrows()):
            global_item_idx = self.dataset_start_idx + (run_batch_idx * self.batch_size) + i
            
            result = self.process_single_item(original_query_idx, row_data, run_batch_idx, global_item_idx)
            batch_results.append(result)

            # Flush logs periodically within a batch if it's large, or rely on end-of-batch flush
            if (i + 1) % 10 == 0: # Every 10 items within a batch
                self._flush_logs()
        
        self._flush_logs() # Ensure logs are flushed at the end of each batch processing
        logger.info(f"Completed processing batch {run_batch_idx}. Processed {len(batch_results)} items.")
        # The last global_item_idx processed in this batch
        last_global_idx_in_batch = self.dataset_start_idx + (run_batch_idx * self.batch_size) + (len(batch_data_df) -1) if not batch_data_df.empty else -1
        return batch_results, last_global_idx_in_batch 


def run_baseline_evaluation(model, tokenizer, base_model_name: str, data_file_path: str, results_dir_str: str, hf_token: str = None, batch_size: int = 32, resume: bool = True):
    device = model.device if model is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # --- Model and Tokenizer Loading --- #
    if tokenizer is None and base_model_name:
        logger.info(f"Tokenizer was None, attempting to load {base_model_name}...")
        try: tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=hf_token)
        except Exception as e: logger.error(f"Failed to load tokenizer for {base_model_name}: {e}"); raise
    
    if model is None and base_model_name and tokenizer is not None:
        logger.info(f"Model was None, attempting to load {base_model_name}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, use_cache=True, low_cpu_mem_usage=True, token=hf_token
            )
            model.to(device)
        except Exception as e: logger.error(f"Failed to load model {base_model_name}: {e}"); raise
    elif model is not None: model.to(device)

    if model is None or tokenizer is None: logger.error("Model or tokenizer is None. Cannot proceed."); return
    model.eval()

    # --- Data Loading --- #
    try:
        df = pd.read_csv(data_file_path)
        logger.info(f"Loaded data from {data_file_path}, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data from {data_file_path}: {e}"); return

    # --- Setup Directories and Resuming --- #
    results_base_dir = Path(results_dir_str) # This will be like "checkpoints/baseline_run_1"
    logs_base_dir = results_base_dir / "logs" # Logs will go into a subfolder
    logs_base_dir.mkdir(parents=True, exist_ok=True)
    
    resume_checkpoint_path = logs_base_dir / "baseline_resume_checkpoint.json"
    start_global_item_idx = 0
    if resume and resume_checkpoint_path.exists():
        try:
            with open(resume_checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                start_global_item_idx = int(checkpoint_data.get("last_processed_global_item_idx", -1)) + 1
                logger.info(f"Resuming baseline evaluation from global item index {start_global_item_idx}")
        except Exception as e:
            logger.error(f"Error reading resume checkpoint {resume_checkpoint_path}: {e}. Starting from the beginning.")
            start_global_item_idx = 0
    
    if start_global_item_idx >= len(df):
        logger.info("Baseline evaluation already completed according to resume checkpoint.")
        return
    df_to_process = df.iloc[start_global_item_idx:]
    if df_to_process.empty:
        logger.info("No data to process after considering resume point.")
        return

    # --- Initialize Processor --- #
    processor = BaselineDataProcessor(model, tokenizer, device, logs_base_dir=logs_base_dir, batch_size=batch_size)
    processor.dataset_start_idx = start_global_item_idx # Inform processor of the global starting point in the dataset

    # --- Main Loop --- #
    logger.info(f"Starting baseline evaluation for {len(df_to_process)} items.")
    processed_item_count_in_this_run = 0
    last_processed_global_idx_overall = start_global_item_idx -1 # Ensure it's one less than the first item to be processed

    try:
        for run_batch_idx, batch_start_offset in enumerate(tqdm(range(0, len(df_to_process), batch_size), desc="Baseline Evaluation")):
            batch_end_offset = min(batch_start_offset + batch_size, len(df_to_process))
            # Get the slice of data for the current batch from df_to_process
            # iterrows() on this slice will give original_query_idx from the full df
            current_batch_df = df_to_process.iloc[batch_start_offset:batch_end_offset]
            
            if current_batch_df.empty:
                logger.info("Current batch is empty, skipping.")
                continue

            # Process the batch
            # process_batch now expects the DataFrame slice
            _, last_processed_global_idx_in_batch = processor.process_batch(current_batch_df, run_batch_idx) 
            last_processed_global_idx_overall = max(last_processed_global_idx_overall, last_processed_global_idx_in_batch)
            processed_item_count_in_this_run += len(current_batch_df)
            
            if run_batch_idx > 0 and run_batch_idx % 20 == 0: # Less frequent cleanup for baseline
                clear_caches(); clear_response_cache(); gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logger.info(f"Periodic cleanup at run_batch_idx {run_batch_idx}")
                
    except KeyboardInterrupt:
        logger.warning("\n⚠️ User interrupted baseline evaluation...")
    except Exception as e:
        logger.error(f"An unexpected error occurred during baseline evaluation: {e}", exc_info=True)
    finally:
        if 'processor' in locals() and processor:
            processor._flush_logs()
        logger.info(f"Baseline evaluation loop finished or interrupted. Total items processed in this run: {processed_item_count_in_this_run}.")
        logger.info(f"Last globally processed item index: {last_processed_global_idx_overall}")
        logger.info(f"Logs and results saved in: {logs_base_dir}")

    logger.info(f"✅ Baseline evaluation complete. Total items processed in this run: {processed_item_count_in_this_run}")

if __name__ == "__main__":
    logger.info("Starting Baseline Evaluation Script (Example Main Call)")
    
    # --- Configuration for Example Run --- #
    # These would ideally come from a config file or command-line arguments
    DEFAULT_BASE_MODEL = "gpt2" # A small model for quick testing if no env var is set
    DEFAULT_DATA_FILE = "data/merged_queries_ads.csv" # Path to your data
    DEFAULT_RESULTS_DIR = "results/baseline_run_test" # Specific directory for this run's outputs
    DEFAULT_BATCH_SIZE = 4 # Keep it small for testing

    # Get from environment or use defaults
    base_model_name = os.getenv("BASE_MODEL", DEFAULT_BASE_MODEL)
    data_file = os.getenv("DATA_FILE", DEFAULT_DATA_FILE)
    results_dir = os.getenv("RESULTS_DIR", DEFAULT_RESULTS_DIR)
    hf_token = os.getenv("HF_TOKEN")
    try:
        batch_size = int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    except ValueError:
        batch_size = DEFAULT_BATCH_SIZE
        logger.warning(f"Invalid BATCH_SIZE env var, using default: {batch_size}")

    # Ensure results directory exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved in: {results_dir}")
    logger.info(f"Using model: {base_model_name}, Data: {data_file}, Batch Size: {batch_size}")

    # --- Load Model and Tokenizer (Example) --- #
    # In a real setup, this might be passed from your main training/evaluation script.
    example_tokenizer = None
    example_model = None
    try:
        logger.info(f"Loading tokenizer: {base_model_name}")
        example_tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token, trust_remote_code=True)
        logger.info(f"Loading model: {base_model_name}")
        example_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            token=hf_token, 
            torch_dtype=torch.float16, # Use float16 for efficiency
            device_map="auto",         # Automatically map to available device (GPU if present)
            trust_remote_code=True,
            use_cache=True,
            low_cpu_mem_usage=True
        )
        logger.info(f"Example model {base_model_name} and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading example model/tokenizer ({base_model_name}): {e}", exc_info=True)
        logger.error("Script will exit. Please ensure model name is correct and accessible (e.g., HF_TOKEN if gated).")
        exit(1) # Exit if essential components can't be loaded

    # --- Run Baseline Evaluation --- #
    try:
        run_baseline_evaluation(
            model=example_model,
            tokenizer=example_tokenizer,
            base_model_name=base_model_name, # Pass for reference, though model is already loaded
            data_file_path=data_file,
            results_dir_str=results_dir, # Base directory for all outputs of this run
            hf_token=hf_token,
            batch_size=batch_size,
            resume=True # Enable resuming by default
        )
    except Exception as e:
        logger.error(f"An error occurred during run_baseline_evaluation: {e}", exc_info=True)

    logger.info("Baseline evaluation script finished.") 