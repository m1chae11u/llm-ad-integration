import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pandas as pd
import gc
from tqdm import tqdm
import time
import json
from pathlib import Path
import logging
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.judge import (
    judge_coherence_async,
    judge_helpfulness_async,
    judge_ad_salience_async,
)
from src.generate.generator import generate_responses, clear_response_cache
from src.judge.utils import clear_caches 
from src.config import BASE_MODEL, HF_TOKEN

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class BaselineDataProcessor:
    def __init__(self, model, tokenizer, device, logs_base_dir: Path, batch_size=2):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Get batch size from environment variable or use default
        self.batch_size = batch_size or int(os.getenv("BASELINE_BATCH_SIZE", "2"))
        self.dataset_start_idx = 0
        self.logs_base_dir = logs_base_dir

        # Enable memory optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Set memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available GPU memory
            torch.cuda.memory.set_per_process_memory_fraction(0.9)
        
        # Initialize response cache
        self.response_cache = {}
        self.cache_size = 1000  # Maximum number of cached responses

        # Create directories for intermediate results
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
                "run_batch_idx", "global_item_idx", "original_query_idx", "ad_id", "ad_index", "query", "ad_facts", 
                "response_with_ad", "generation_time", "token_count"
            ]).to_csv(self.generation_log, index=False)
        
        if not self.judging_log.exists() or os.path.getsize(self.judging_log) == 0:
            pd.DataFrame(columns=[
                "run_batch_idx", "global_item_idx", "original_query_idx", "ad_id", "ad_index", "query", "response_with_ad", 
                # Coherence subscores
                "C1", "C2", "C3", "C4",
                "coherence_score", "coherence_explanation",
                # Helpfulness subscore
                "H1",
                "helpfulness_score", "helpfulness_explanation",
                # Salience subscores
                "S1", "S2", "S3",
                "salience_score", "salience_explanation",
                # Remove detectability
                # "detectability_cosine",
                "judging_time"
            ]).to_csv(self.judging_log, index=False)
        
        if not self.evaluation_log.exists() or os.path.getsize(self.evaluation_log) == 0:
            pd.DataFrame(columns=[
                "run_batch_idx", "global_item_idx", "original_query_idx", "ad_id", "ad_index", "query", "response_with_ad", "reward"
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
        logger.debug("Flushed logs to disk.") 

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

    async def _run_judges_parallel(self, query, response_with_ad, ad_text, item_idx):
        logger.info(f"Starting parallel judges for item {item_idx}...")
        try:
            # Run all judges concurrently
            coherence_task = judge_coherence_async(query, response_with_ad)
            helpfulness_task = judge_helpfulness_async(query, response_with_ad)
            salience_task = judge_ad_salience_async(query, response_with_ad, ad_text)
            
            # Wait for all judges to complete
            score_coh, score_help, score_sal = await asyncio.gather(
                coherence_task, helpfulness_task, salience_task
            )
            
            logger.info(f"Item {item_idx} - Judges completed. Scores - Coherence: {score_coh.get('Coherence Score', 0)}, Helpfulness: {score_help.get('Helpfulness Score', 0)}, Salience: {score_sal.get('Ad Salience Score', 0)}")
            return score_coh, score_help, score_sal
        except Exception as e:
            logger.error(f"Error in parallel judges for item {item_idx}: {e}", exc_info=True)
            raise

    async def process_single_item(self, original_query_idx, query_data, run_batch_idx, global_item_idx):
        query = str(query_data['vague_query'])
        ad_facts = {
            "ad_product": str(query_data['ad_product']), 
            "brand": str(query_data['brand']),
            "url": str(query_data['url']), 
            "description": str(query_data['ad_description']),
        }
        # Get both ad_id and ad_index from the dataset
        ad_id = str(query_data.get('ad_id', 'N/A'))
        ad_index = str(query_data.get('ad_index', 'N/A'))
        logger.info(f"Processing: RunBatch {run_batch_idx}, GlobalItem {global_item_idx}, OriginalQueryIdx {original_query_idx}, AdID: {ad_id}, AdIndex: {ad_index}")

        response_with_ad, gen_time, token_count = None, 0, 0
        score_coh, score_help, score_sal = {}, {}, {}
        judge_time = 0
        reward_value = 0.0
        success = False

        try:
            # Generate response
            logger.info(f"Generating response for item {global_item_idx}...")
            gen_start_time = time.time()
            with torch.no_grad():
                _, response_with_ad = generate_responses(query, ad_facts, self.model, self.tokenizer)
            gen_time = time.time() - gen_start_time
            token_count = len(self.tokenizer.encode(response_with_ad))
            logger.info(f"Item {global_item_idx} - Response generated in {gen_time:.2f}s")
            
            # Log generation
            self.generation_log_buffer.append({
                "run_batch_idx": run_batch_idx, 
                "global_item_idx": global_item_idx, 
                "original_query_idx": original_query_idx,
                "ad_id": ad_id,
                "ad_index": ad_index,
                "query": query, 
                "ad_facts": json.dumps(ad_facts),
                "response_with_ad": response_with_ad,
                "generation_time": gen_time, 
                "token_count": token_count
            })

            # Run judges
            logger.info(f"Starting judging pipeline for item {global_item_idx}...")
            judge_start_time = time.time()
            ad_text = f"Product: {ad_facts['ad_product']}\nBrand: {ad_facts['brand']}\nURL: {ad_facts['url']}\nDescription: {ad_facts['description']}"
            score_coh, score_help, score_sal = await self._run_judges_parallel(query, response_with_ad, ad_text, global_item_idx)
            judge_time = time.time() - judge_start_time
            logger.info(f"Item {global_item_idx} - Judging completed in {judge_time:.2f}s")
            
            # Log judging results
            self.judging_log_buffer.append({
                "run_batch_idx": run_batch_idx, 
                "global_item_idx": global_item_idx, 
                "original_query_idx": original_query_idx,
                "ad_id": ad_id,
                "ad_index": ad_index,
                "query": query, 
                "response_with_ad": response_with_ad,
                # Coherence subscores
                "C1": score_coh.get("C1", 0),
                "C2": score_coh.get("C2", 0),
                "C3": score_coh.get("C3", 0),
                "C4": score_coh.get("C4", 0),
                "coherence_score": score_coh.get("Coherence Score", 0),
                "coherence_explanation": score_coh.get("Coherence Explanation", ""),
                # Helpfulness subscore
                "H1": score_help.get("H1", 0),
                "helpfulness_score": score_help.get("H1", 0),
                "helpfulness_explanation": score_help.get("Helpfulness Explanation", ""),
                # Salience subscores
                "S1": score_sal.get("S1", 0),
                "S2": score_sal.get("S2", 0),
                "S3": score_sal.get("S3", 0),
                "salience_score": score_sal.get("Ad Salience Score", 0),
                "salience_explanation": score_sal.get("Ad Salience Explanation", ""),
                "judging_time": judge_time
            })

            # Calculate reward
            reward_values = [
                score_coh.get("Coherence Score", 0), 
                score_help.get("H1", 0),
                score_sal.get("Ad Salience Score", 0)
            ]
            reward_value = sum(reward_values) / len(reward_values)
            logger.info(f"Item {global_item_idx} - Final reward: {reward_value:.2f}")
            
            # Log evaluation
            self.evaluation_log_buffer.append({
                "run_batch_idx": run_batch_idx, 
                "global_item_idx": global_item_idx, 
                "original_query_idx": original_query_idx,
                "ad_id": ad_id,
                "ad_index": ad_index,
                "query": query, 
                "response_with_ad": response_with_ad, 
                "reward": reward_value
            })
            success = True

        except Exception as e:
            logger.error(f"Error processing OriginalQueryIdx {original_query_idx}: {e}", exc_info=True)
        
        self.update_stats(run_batch_idx, global_item_idx, gen_time, judge_time, token_count, reward_value, success=success)
        
        # Save checkpoint
        resume_checkpoint_path = self.logs_base_dir / "baseline_resume_checkpoint.json"
        try:
            with open(resume_checkpoint_path, "w") as f:
                json.dump({
                    "last_processed_global_item_idx": global_item_idx,
                    "last_ad_id": ad_id,
                    "last_ad_index": ad_index,
                    "last_original_query_idx": original_query_idx
                }, f)
        except Exception as e:
            logger.error(f"Failed to save resume checkpoint: {e}")

        return {
            "original_query_idx": original_query_idx, 
            "query": query, 
            "ad_facts": ad_facts,
            "ad_id": ad_id,
            "ad_index": ad_index,
            "response_with_ad": response_with_ad,
            "reward": reward_value, 
            "scores": {"coherence": score_coh, "helpfulness": score_help, "salience": score_sal},
            "processed_successfully": success
        }

    def _clear_memory(self):
        """Aggressive memory cleanup."""
        clear_caches()
        clear_response_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Clear response cache if it's too large
        if len(self.response_cache) > self.cache_size:
            self.response_cache.clear()

    async def process_batch(self, batch_data_df, run_batch_idx):
        batch_results = []
        batch_start_time = time.time()
        
        # Pre-process all queries in the batch
        queries = []
        ad_facts_list = []
        ad_ids = []
        ad_indices = []
        for _, row in batch_data_df.iterrows():
            queries.append(str(row['vague_query']))
            ad_facts_list.append({
                "ad_product": str(row['ad_product']),
                "brand": str(row['brand']),
                "url": str(row['url']),
                "description": str(row['ad_description']),
            })
            ad_ids.append(str(row.get('ad_id', 'N/A')))
            ad_indices.append(str(row.get('ad_index', 'N/A')))

        # Batch generate responses
        with torch.no_grad():
            responses = []
            for query, ad_facts in zip(queries, ad_facts_list):
                _, response_with_ad = generate_responses(query, ad_facts, self.model, self.tokenizer)
                responses.append(response_with_ad)

        # Process items in parallel
        tasks = []
        for i, (query, response_with_ad, ad_facts, ad_id, ad_index) in enumerate(zip(queries, responses, ad_facts_list, ad_ids, ad_indices)):
            global_item_idx = self.dataset_start_idx + (run_batch_idx * self.batch_size) + i
            task = self.process_single_item(i, {
                'vague_query': query,
                'ad_product': ad_facts['ad_product'],
                'brand': ad_facts['brand'],
                'url': ad_facts['url'],
                'ad_description': ad_facts['description'],
                'ad_id': ad_id,
                'ad_index': ad_index
            }, run_batch_idx, global_item_idx)
            tasks.append(task)
        
        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*tasks)
        
        batch_time = time.time() - batch_start_time
        logger.info(f"✅ Completed batch {run_batch_idx} in {batch_time:.2f}s")
        
        self._flush_logs()
        logger.info(f"Completed processing batch {run_batch_idx}. Processed {len(batch_results)} items.")
        
        # Clear memory after each batch
        self._clear_memory()
        
        last_global_idx_in_batch = self.dataset_start_idx + (run_batch_idx * self.batch_size) + (len(batch_data_df) -1) if not batch_data_df.empty else -1
        return batch_results, last_global_idx_in_batch

async def run_baseline_evaluation(model, tokenizer, base_model_name: str, data_file_path: str, results_dir_str: str, hf_token: str = None, batch_size: int = 32, resume: bool = True):
    device = model.device if model is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Enable memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.memory.set_per_process_memory_fraction(0.9)

    # Load model and tokenizer if needed
    if tokenizer is None and base_model_name:
        logger.info(f"Loading tokenizer: {base_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=hf_token)
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {base_model_name}: {e}")
            raise
    
    if model is None and base_model_name and tokenizer is not None:
        logger.info(f"Loading model: {base_model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                torch_dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True, 
                use_cache=True, 
                low_cpu_mem_usage=True, 
                token=hf_token,
                # Add performance optimizations
                offload_folder="offload",
                offload_state_dict=True,
                max_memory={0: "24GB"},
                load_in_8bit=True
            )
            model.gradient_checkpointing_enable()
            model.to(device)
        except Exception as e:
            logger.error(f"Failed to load model {base_model_name}: {e}")
            raise

    if model is None or tokenizer is None:
        logger.error("Model or tokenizer is None. Cannot proceed.")
        return

    model.eval()

    # Load data
    try:
        df = pd.read_csv(data_file_path)
        logger.info(f"Loaded data from {data_file_path}, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data from {data_file_path}: {e}")
        return

    # Setup directories and resume point
    results_base_dir = Path(results_dir_str)
    logs_base_dir = results_base_dir / "logs"
    logs_base_dir.mkdir(parents=True, exist_ok=True)
    
    resume_checkpoint_path = logs_base_dir / "baseline_resume_checkpoint.json"
    start_global_item_idx = 0
    
    if resume and resume_checkpoint_path.exists():
        try:
            with open(resume_checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                start_global_item_idx = int(checkpoint_data.get("last_processed_global_item_idx", -1)) + 1
                logger.info(f"Resuming from global item index {start_global_item_idx}")
        except Exception as e:
            logger.error(f"Error reading resume checkpoint: {e}. Starting from beginning.")
            start_global_item_idx = 0
    
    if start_global_item_idx >= len(df):
        logger.info("Evaluation already completed according to resume checkpoint.")
        return

    df_to_process = df.iloc[start_global_item_idx:]
    if df_to_process.empty:
        logger.info("No data to process after resume point.")
        return

    # Initialize processor and run evaluation
    processor = BaselineDataProcessor(model, tokenizer, device, logs_base_dir=logs_base_dir, batch_size=batch_size)
    processor.dataset_start_idx = start_global_item_idx

    logger.info(f"Starting evaluation for {len(df_to_process)} items.")
    processed_item_count = 0
    last_processed_global_idx = start_global_item_idx - 1

    try:
        for run_batch_idx, batch_start_offset in enumerate(tqdm(range(0, len(df_to_process), batch_size), desc="Baseline Evaluation")):
            batch_end_offset = min(batch_start_offset + batch_size, len(df_to_process))
            current_batch_df = df_to_process.iloc[batch_start_offset:batch_end_offset]
            
            if current_batch_df.empty:
                continue

            _, last_processed_global_idx_in_batch = await processor.process_batch(current_batch_df, run_batch_idx)
            last_processed_global_idx = max(last_processed_global_idx, last_processed_global_idx_in_batch)
            processed_item_count += len(current_batch_df)
            
            # Clear memory more frequently
            if run_batch_idx > 0 and run_batch_idx % 10 == 0:  # Changed from 20 to 10
                processor._clear_memory()
                
    except KeyboardInterrupt:
        logger.warning("\n⚠️ User interrupted evaluation...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if 'processor' in locals() and processor:
            processor._flush_logs()
        logger.info(f"Evaluation finished or interrupted. Processed {processed_item_count} items.")
        logger.info(f"Last processed item index: {last_processed_global_idx}")
        logger.info(f"Results saved in: {logs_base_dir}")

if __name__ == "__main__":
    logger.info("Starting Baseline Evaluation")
    
    data_file = os.getenv("DATA_FILE", "data/merged_queries_ads.csv")
    results_dir = os.getenv("RESULTS_DIR", "results/baseline_run_test")
    batch_size = int(os.getenv("BATCH_SIZE", "2"))
    base_model_name = os.getenv("BASE_MODEL", BASE_MODEL)
    hf_token = os.getenv("HF_TOKEN", HF_TOKEN)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved in: {results_dir}")
    logger.info(f"Using model: {base_model_name}, Data: {data_file}, Batch Size: {batch_size}")

    try:
        logger.info(f"Loading tokenizer: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token, trust_remote_code=True)
        
        logger.info(f"Loading model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=True,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_state_dict=True,
            max_memory={0: "24GB"},
            load_in_8bit=True
        )
        model.gradient_checkpointing_enable()
        
        asyncio.run(run_baseline_evaluation(
            model=model,
            tokenizer=tokenizer,
            base_model_name=base_model_name,
            data_file_path=data_file,
            results_dir_str=results_dir,
            hf_token=hf_token,
            batch_size=batch_size,
            resume=True
        ))
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)

    logger.info("Baseline evaluation finished.") 