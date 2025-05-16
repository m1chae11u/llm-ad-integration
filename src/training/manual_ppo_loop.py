import os
import torch
import pandas as pd
import gc
from tqdm import tqdm
from torch.nn import functional as F
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from queue import Queue, Empty
import threading
import time
import shutil
import tempfile
import json
from pathlib import Path
import logging
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig # Added for base model loading

from ..judge import (
    judge_coherence,
    judge_helpfulness,
    judge_ad_salience,
    judge_detectability
)
from ..generate.generator import generate_responses, clear_response_cache
from ..judge.utils import clear_caches
from .checkpoint_manager import CheckpointManager

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

# Create an adapter with the retry strategy
adapter = HTTPAdapter(max_retries=retry_strategy)

# Mount the adapter on the session
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

def compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_range=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    return -torch.min(ratio * advantages, clipped * advantages).mean()

def compute_advantages(reward, value):
    return reward - value

class DataProcessor:
    def __init__(self, model, tokenizer, device, checkpoint_base_dir: Path, batch_size=32, checkpoint_manager=None, optimizer=None, base_model_name=None, hf_token=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_manager = checkpoint_manager
        self.optimizer = optimizer
        self.base_model_name = base_model_name
        self.hf_token = hf_token
        self.current_step = 0
        self.dataset_start_idx = 0
        self.checkpoint_base_dir = checkpoint_base_dir
        
        if self.checkpoint_manager is not None:
            self.model = self.checkpoint_manager.model
            self.tokenizer = self.checkpoint_manager.tokenizer
            self.optimizer = self.checkpoint_manager.optimizer
        
        # Create directories for intermediate results
        self.generation_dir = Path("logs/generations")
        self.judging_dir = Path("logs/judgments")
        self.training_dir = Path("logs/training")
        self.stats_dir = Path("logs/stats")
        
        for dir_path in [self.generation_dir, self.judging_dir, self.training_dir, self.stats_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize log files with batch information
        self.generation_log = self.generation_dir / "generation_log.csv"
        self.judging_log = self.judging_dir / "judging_log.csv"
        self.training_log = self.training_dir / "training_log.csv"
        self.stats_log = self.stats_dir / "training_stats.csv"
        
        # Buffers for logs
        self.generation_log_buffer = []
        self.judging_log_buffer = []
        self.training_log_buffer = []
        
        # Create headers for log files with batch information if they don't exist
        if not self.generation_log.exists() or os.path.getsize(self.generation_log) == 0:
            pd.DataFrame(columns=[
                "batch_idx", "global_batch_idx", "query_idx", "ad_source_id", "query", "ad_facts", "response_without_ad", "response_with_ad",
                "generation_time", "token_count"
            ]).to_csv(self.generation_log, index=False)
        
        if not self.judging_log.exists() or os.path.getsize(self.judging_log) == 0:
            pd.DataFrame(columns=[
                "batch_idx", "global_batch_idx", "query_idx", "ad_source_id", "query", "response_with_ad", 
                "coherence_score", "helpfulness_score", "salience_score", "detectability_score", 
                "coherence_explanation", "helpfulness_explanation", "salience_explanation",
                "judging_time"
            ]).to_csv(self.judging_log, index=False)
        
        if not self.training_log.exists() or os.path.getsize(self.training_log) == 0:
            pd.DataFrame(columns=[
                "batch_idx", "global_batch_idx", "query_idx", "ad_source_id", "query", "response_with_ad", "reward", "loss", "training_time"
            ]).to_csv(self.training_log, index=False)
        
        # Initialize stats - default values
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_generation_time": 0,
            "total_judging_time": 0,
            "total_training_time": 0,
            "total_tokens": 0,
            "avg_reward": 0,
            "avg_loss": 0
        }
        
        # Try to load existing stats if available
        self._load_stats()
        
        # Create stats file if it doesn't exist
        if not self.stats_log.exists() or os.path.getsize(self.stats_log) == 0:
            pd.DataFrame(columns=[
                "batch_idx", "global_batch_idx", "timestamp", "total_queries", "successful_queries", "failed_queries",
                "avg_generation_time", "avg_judging_time", "avg_training_time", "avg_reward", "avg_loss",
                "total_tokens", "memory_usage"
            ]).to_csv(self.stats_log, index=False)

    def _load_stats(self):
        """Try to load existing training statistics to resume from."""
        try:
            if self.stats_log.exists() and os.path.getsize(self.stats_log) > 0:
                logger.info(f"Loading existing training statistics from {self.stats_log}")
                stats_df = pd.read_csv(self.stats_log)
                
                if not stats_df.empty:
                    # Get the latest entry
                    latest_stats = stats_df.iloc[-1]
                    
                    # Load the stats
                    self.stats["total_queries"] = int(latest_stats.get("total_queries", 0))
                    self.stats["successful_queries"] = int(latest_stats.get("successful_queries", 0))
                    self.stats["failed_queries"] = int(latest_stats.get("failed_queries", 0))
                    
                    # Calculate accumulated times from averages * counts 
                    self.stats["total_generation_time"] = (
                        latest_stats.get("avg_generation_time", 0) * self.stats["total_queries"]
                    )
                    self.stats["total_judging_time"] = (
                        latest_stats.get("avg_judging_time", 0) * self.stats["total_queries"]
                    )
                    self.stats["total_training_time"] = (
                        latest_stats.get("avg_training_time", 0) * self.stats["total_queries"]
                    )
                    
                    self.stats["total_tokens"] = int(latest_stats.get("total_tokens", 0))
                    self.stats["avg_reward"] = float(latest_stats.get("avg_reward", 0))
                    self.stats["avg_loss"] = float(latest_stats.get("avg_loss", 0))
                    
                    logger.info(f"Resumed training stats: {self.stats['total_queries']} queries, "
                               f"avg_reward: {self.stats['avg_reward']:.4f}")
                else:
                    logger.info("No existing training stats found, starting with fresh statistics")
            else:
                logger.info("No stats file exists yet, starting with fresh statistics")
        except Exception as e:
            logger.error(f"Error loading stats: {e}. Starting with fresh statistics.")
            # Keep using the default values

    def _flush_logs(self):
        """Write buffered logs to disk."""
        if self.generation_log_buffer:
            pd.DataFrame(self.generation_log_buffer).to_csv(self.generation_log, mode="a", header=False, index=False)
            self.generation_log_buffer.clear()
            logger.info(f"Flushed {len(self.generation_log_buffer)} generation log entries.")

        if self.judging_log_buffer:
            pd.DataFrame(self.judging_log_buffer).to_csv(self.judging_log, mode="a", header=False, index=False)
            self.judging_log_buffer.clear()
            logger.info(f"Flushed {len(self.judging_log_buffer)} judging log entries.")

        if self.training_log_buffer:
            pd.DataFrame(self.training_log_buffer).to_csv(self.training_log, mode="a", header=False, index=False)
            self.training_log_buffer.clear()
            logger.info(f"Flushed {len(self.training_log_buffer)} training log entries.")

    def update_stats(self, batch_idx, current_query_position, gen_time, judge_time, train_time, token_count, reward, loss, success=True):
        """Update training statistics."""
        self.stats["total_queries"] += 1
        if success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
            
        self.stats["total_generation_time"] += gen_time
        self.stats["total_judging_time"] += judge_time
        self.stats["total_training_time"] += train_time
        self.stats["total_tokens"] += token_count
        
        # Update averages
        n = self.stats["successful_queries"]
        self.stats["avg_reward"] = (self.stats["avg_reward"] * (n-1) + reward) / n if n > 0 else 0
        self.stats["avg_loss"] = (self.stats["avg_loss"] * (n-1) + loss) / n if n > 0 else 0
        
        # Log stats every 10 queries
        if self.stats["total_queries"] % 10 == 0:
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            # Calculate global batch index for stats
            # (self.stats["total_queries"] -1) gives the index of the last processed query (0-indexed)
            # So, ((self.stats["total_queries"] -1) // self.batch_size) is its global batch index.
            # If current_query_position is available and accurate (passed from process_batch), it's better.
            global_batch_idx_for_stats = (current_query_position // self.batch_size) if current_query_position != -1 else ((self.stats["total_queries"] -1) // self.batch_size if self.stats["total_queries"] > 0 else 0)

            pd.DataFrame([{
                "batch_idx": batch_idx, # Run-specific batch_idx
                "global_batch_idx": global_batch_idx_for_stats, # Global batch_idx
                "timestamp": time.time(),
                "total_queries": self.stats["total_queries"],
                "successful_queries": self.stats["successful_queries"],
                "failed_queries": self.stats["failed_queries"],
                "avg_generation_time": self.stats["total_generation_time"] / self.stats["total_queries"],
                "avg_judging_time": self.stats["total_judging_time"] / self.stats["total_queries"],
                "avg_training_time": self.stats["total_training_time"] / self.stats["total_queries"],
                "avg_reward": self.stats["avg_reward"],
                "avg_loss": self.stats["avg_loss"],
                "total_tokens": self.stats["total_tokens"],
                "memory_usage": memory_usage
            }]).to_csv(self.stats_log, mode="a", header=False, index=False)

    def validate_model(self, validation_data, batch_idx):
        """Run validation on a subset of data."""
        logger.info(f"🔄 Starting validation for batch {batch_idx}")
        self.model.eval()
        validation_results = []
        
        with torch.no_grad():
            for idx, row in validation_data.iterrows():
                try:
                    query = str(row['vague_query'])
                    ad_facts = {
                        "ad_product": str(row['ad_product']),
                        "brand": str(row['brand']),
                        "url": str(row['url']),
                        "description": str(row['ad_description']),
                    }
                    
                    # Generate response
                    response_without_ad, response_with_ad = generate_responses(query, ad_facts, self.model, self.tokenizer)
                    
                    # Run judges
                    ad_text = f"""Product: {ad_facts['ad_product']}
                                Brand: {ad_facts['brand']}
                                URL: {ad_facts['url']}
                                Description: {ad_facts['description']}"""
                    
                    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                        future_coh = executor.submit(judge_coherence, query, response_with_ad)
                        future_help = executor.submit(judge_helpfulness, query, response_with_ad)
                        future_sal = executor.submit(judge_ad_salience, query, response_with_ad, ad_text)
                        future_det = executor.submit(judge_detectability, response_with_ad, response_without_ad)
                        
                        score_coh = future_coh.result()
                        score_help = future_help.result()
                        score_sal = future_sal.result()
                        score_det = future_det.result()
                    
                    # Compute reward
                    reward_values = [
                        score_coh.get("Coherence Score", 0),
                        score_help.get("Helpfulness Score", 0),
                        score_sal.get("Ad Salience Score", 0),
                        score_det.get("detectability_cosine", 0) or 0
                    ]
                    reward = torch.tensor(sum(reward_values) / len(reward_values) if reward_values else 0.0, dtype=torch.float32).to(self.device)
                    
                    # For validation, we don't need to compute loss

                    validation_results.append({
                        "idx": idx,
                        "query": query,
                        "ad_facts": ad_facts,
                        "response_without_ad": response_without_ad,
                        "response_with_ad": response_with_ad,
                        "reward": reward.item(),
                        "scores": {
                            "coherence": score_coh,
                            "helpfulness": score_help,
                            "salience": score_sal,
                            "detectability": score_det
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Validation error for query {idx}: {e}")
                    continue
        
        # Compute validation metrics safely
        num_val_results = len(validation_results)
        avg_reward = sum(r["reward"] for r in validation_results) / num_val_results if num_val_results > 0 else 0.0
        logger.info(f"✅ Validation complete for batch {batch_idx}. Average reward: {avg_reward:.4f}")
        
        return validation_results

    def process_batch(self, batch_data, batch_idx):
        """Process a batch of data."""
        results = []
        logger.info(f"🔄 Starting Run Batch {batch_idx} (approx. Global Batch { (self.dataset_start_idx + batch_idx * self.batch_size) // self.batch_size}) with {len(batch_data)} queries. Absolute start query for this run batch: {self.dataset_start_idx + batch_idx * self.batch_size}")
        batch_start_time = time.time()
        
        # Get the actual start index in the original dataset
        # Account for the dataset_start_idx offset
        start_idx_in_batch = batch_idx * self.batch_size
        # The absolute position is the offset plus the position in the current slice
        start_idx_in_dataset = self.dataset_start_idx + start_idx_in_batch
        current_query_position = start_idx_in_dataset  # Initialize to start of batch
        
        # Keep track of the position within batch
        for i, (idx, row) in enumerate(batch_data.iterrows()):
            # Update to current absolute position in dataset
            current_query_position = start_idx_in_dataset + i
            
            query = str(row['vague_query'])
            ad_facts = {
                "ad_product": str(row['ad_product']),
                "brand": str(row['brand']),
                "url": str(row['url']),
                "description": str(row['ad_description']),
            }
            # Attempt to get ad_source_id from a column named 'ad_index', default to 'N/A'
            ad_source_id = str(row.get('ad_index', 'N/A'))

            # Log which query we're on in dataset
            logger.info(f"🔄 Global Batch {current_query_position // self.batch_size} (Run Batch {batch_idx}) - Starting generation for query {idx} (dataset position: {current_query_position}, ad_source_id: {ad_source_id})")

            try:
                # Stage 1: Generation
                gen_start_time = time.time()
                
                with torch.no_grad():
                    response_without_ad, response_with_ad = generate_responses(query, ad_facts, self.model, self.tokenizer)
                
                gen_time = time.time() - gen_start_time
                token_count = len(self.tokenizer.encode(response_with_ad))
                
                # Log generation results
                self.generation_log_buffer.append({
                    "batch_idx": batch_idx, # This is the run-specific batch_idx
                    "global_batch_idx": current_query_position // self.batch_size, # Adding global batch index
                    "query_idx": idx,
                    "ad_source_id": ad_source_id,
                    "query": query,
                    "ad_facts": json.dumps(ad_facts),
                    "response_without_ad": response_without_ad,
                    "response_with_ad": response_with_ad,
                    "generation_time": gen_time,
                    "token_count": token_count
                })
                
                logger.info(f"✅ Batch {batch_idx} - Generation complete for query {idx} (dataset position: {current_query_position}) in {gen_time:.2f}s")

                # Stage 2: Judging
                logger.info(f"🔄 Batch {batch_idx} - Starting judging for query {idx} (dataset position: {current_query_position})")
                judge_start_time = time.time()
                
                ad_text = f"""Product: {ad_facts['ad_product']}
                            Brand: {ad_facts['brand']}
                            URL: {ad_facts['url']}
                            Description: {ad_facts['description']}"""

                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    future_coh = executor.submit(judge_coherence, query, response_with_ad)
                    future_help = executor.submit(judge_helpfulness, query, response_with_ad)
                    future_sal = executor.submit(judge_ad_salience, query, response_with_ad, ad_text)
                    future_det = executor.submit(judge_detectability, response_with_ad, response_without_ad)

                    score_coh = future_coh.result()
                    score_help = future_help.result()
                    score_sal = future_sal.result()
                    score_det = future_det.result()

                judge_time = time.time() - judge_start_time
                
                # Log the scores
                logger.info(f"👨‍⚖️ Batch {batch_idx} - Query {idx} (Dataset: {current_query_position}) Judge Scores: \n" +
                            f"   Coherence: {score_coh}\n" +
                            f"   Helpfulness: {score_help}\n" +
                            f"   Salience: {score_sal}\n" +
                            f"   Detectability: {score_det}")

                # Log judging results
                self.judging_log_buffer.append({
                    "batch_idx": batch_idx, # Run-specific
                    "global_batch_idx": current_query_position // self.batch_size, # Adding global batch index
                    "query_idx": idx,
                    "ad_source_id": ad_source_id,
                    "query": query,
                    "response_with_ad": response_with_ad,
                    "coherence_score": score_coh.get("Coherence Score", 0),
                    "helpfulness_score": score_help.get("Helpfulness Score", 0),
                    "salience_score": score_sal.get("Ad Salience Score", 0),
                    "detectability_score": score_det.get("detectability_cosine", 0),
                    "coherence_explanation": score_coh.get("Coherence Explanation", ""),
                    "helpfulness_explanation": score_help.get("Helpfulness Explanation", ""),
                    "salience_explanation": score_sal.get("Ad Salience Explanation", ""),
                    "judging_time": judge_time
                })
                
                logger.info(f"✅ Batch {batch_idx} - Judging complete for query {idx} (dataset position: {current_query_position}) in {judge_time:.2f}s")

                # Stage 3: Training
                logger.info(f"🔄 Batch {batch_idx} - Starting training for query {idx} (dataset position: {current_query_position})")
                train_start_time = time.time()

                input_ids = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=384).input_ids.to(self.device)
                response_ids = self.tokenizer(response_with_ad, return_tensors="pt", truncation=True, max_length=128).input_ids.to(self.device)[0]

                if input_ids.shape[1] + response_ids.shape[0] > 512:
                    logger.warning(f"⚠️ Batch {batch_idx} - Skipping: combined input too long for query {idx}")
                    continue

                input_plus_response = torch.cat([input_ids[0], response_ids])
                inputs = input_plus_response.unsqueeze(0)
                labels = input_plus_response[1:]

                self.model.train()
                logits = self.model(inputs).logits[0, :-1]
                new_log_probs = F.log_softmax(logits, dim=-1)[torch.arange(len(labels)), labels].sum()
                
                # Compute reward
                reward_values = [
                    score_coh.get("Coherence Score", 0),
                    score_help.get("Helpfulness Score", 0),
                    score_sal.get("Ad Salience Score", 0),
                    score_det.get("detectability_cosine", 0) or 0
                ]
                reward = torch.tensor(sum(reward_values) / len(reward_values) if reward_values else 0.0, dtype=torch.float32).to(self.device)
                
                loss = -new_log_probs * reward.detach()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.model.eval()

                train_time = time.time() - train_start_time
                
                # Log training results
                self.training_log_buffer.append({
                    "batch_idx": batch_idx, # Run-specific
                    "global_batch_idx": current_query_position // self.batch_size, # Adding global batch index
                    "query_idx": idx,
                    "ad_source_id": ad_source_id,
                    "query": query,
                    "response_with_ad": response_with_ad,
                    "reward": reward.item(),
                    "loss": loss.item(),
                    "training_time": train_time
                })
                
                logger.info(f"✅ Batch {batch_idx} - Training complete for query {idx} (dataset position: {current_query_position}) in {train_time:.2f}s")

                # Update statistics
                self.update_stats(batch_idx, current_query_position, gen_time, judge_time, train_time, token_count, reward.item(), loss.item(), success=True)

                results.append({
                    "idx": idx,
                    "query": query,
                    "ad_facts": ad_facts,
                    "response_without_ad": response_without_ad,
                    "response_with_ad": response_with_ad,
                    "reward": reward,
                    "scores": {
                        "coherence": score_coh,
                        "helpfulness": score_help,
                        "salience": score_sal,
                        "detectability": score_det
                    }
                })
                
                # Save current position after every query (not just every 5)
                # Create a small incremental checkpoint file to record last processed query
                if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager:
                    try:
                        # Save just the query position, not the full model (which is expensive)
                        query_checkpoint = {
                            "last_processed_query": int(current_query_position),
                            "original_idx": int(idx),
                            "batch_idx": batch_idx, # Run-specific batch_idx
                            "global_batch_idx": current_query_position // self.batch_size, # Adding global batch index
                            "timestamp": time.time()
                        }
                        
                        # Save to a special file that's quick to update
                        query_checkpoint_path = self.checkpoint_base_dir / "last_query_position.json"
                        query_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(query_checkpoint_path, "w") as f:
                            json.dump(query_checkpoint, f)
                    except Exception as e:
                        logger.error(f"Error saving query position checkpoint: {e}")
                
                # Also update the simple text file
                last_position_path = self.checkpoint_base_dir / "last_processed_position.txt"
                try:
                    with open(last_position_path, "w") as f:
                        f.write(str(current_query_position))
                except Exception as e:
                    logger.error(f"Failed to save position: {e}")
                
                # Flush logs and save formal checkpoint every 5 examples
                if len(results) % 5 == 0:
                    logger.info(f"Saving incremental checkpoint at dataset position {current_query_position} (batch {batch_idx}, query idx {idx})")
                    self._flush_logs()
                    logger.info(f"Saved query position checkpoint: dataset position {current_query_position} (original idx {idx})")

            except Exception as e:
                logger.error(f"❌ Batch {batch_idx} - Error processing query {idx}: {e}")
                self.update_stats(batch_idx, current_query_position if 'current_query_position' in locals() else -1, 0, 0, 0, 0, 0, 0, success=False) # Pass current_query_position or a placeholder
                continue

            # Clear caches periodically
            if len(results) % 10 == 0:
                clear_caches()
                clear_response_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        batch_time = time.time() - batch_start_time
        logger.info(f"✅ Completed batch {batch_idx} in {batch_time:.2f}s")
        
        # Always flush logs at the end of a batch
        self._flush_logs()
        logger.info(f"Logs flushed for batch {batch_idx}")
        
        # Save the absolute latest position to a file at end of batch
        last_position_path = self.checkpoint_base_dir / "last_processed_position.txt"
        try:
            with open(last_position_path, "w") as f:
                f.write(str(current_query_position))
            logger.info(f"Saved latest position at batch end: {current_query_position}")
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
        
        # Return final query position along with results
        return results, current_query_position

def run_manual_ppo(model, tokenizer, base_model_name: str, checkpoint_dir_str: str, hf_token: str = None):
    device = model.device if model is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # If model or tokenizer is None, it means main.py couldn't load them (e.g. gated model access issue initially)
    # We should try to load them here using the base_model_name and hf_token.
    if tokenizer is None and base_model_name:
        logger.info(f"Tokenizer was None, attempting to load {base_model_name} in run_manual_ppo...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=hf_token)
            logger.info(f"✅ Successfully loaded tokenizer for {base_model_name} in run_manual_ppo.")
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer for {base_model_name} in run_manual_ppo: {e}")
            raise # Re-raise if tokenizer is critical here

    if model is None and base_model_name and tokenizer is not None: # model needs tokenizer
        logger.info(f"Model was None, attempting to load {base_model_name} in run_manual_ppo...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=True,
                low_cpu_mem_usage=True,
                token=hf_token
            )
            try:
                model.generation_config = GenerationConfig.from_pretrained(base_model_name, token=hf_token)
            except Exception:
                logger.warning(f"Could not load generation_config for {base_model_name}. Using default.")
                # Attempt to create a default or load from a known good source if necessary
                # For now, let it proceed, PPO might not strictly need it if tokenizer has EOS etc.
            model.to(device) # Ensure model is on the correct device
            logger.info(f"✅ Successfully loaded model {base_model_name} in run_manual_ppo to device {device}.")
        except Exception as e:
            logger.error(f"❌ Failed to load model {base_model_name} in run_manual_ppo: {e}")
            raise # Re-raise if model is critical here
    elif model is not None:
        model.to(device) # Ensure existing model is on the correct device

    if model is None or tokenizer is None:
        logger.error("❌ Model or tokenizer is still None. Cannot proceed with PPO training.")
        return

    model.eval() # Ensure model is in eval mode initially

    df = pd.read_csv("data/merged_queries_ads.csv")
    optimizer = torch.optim.SGD(model.parameters(), lr=1.4e-7)

    base_dir = Path(checkpoint_dir_str)
    log_dir = Path("logs")
    
    directories = {
        "checkpoint_dir": base_dir,
        "log_dir": log_dir,
        "generation_dir": log_dir / "generations",
        "judging_dir": log_dir / "judgments",
        "training_dir": log_dir / "training",
        "stats_dir": log_dir / "stats"
    }
    
    for dir_name, dir_path in directories.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    checkpoint_manager = CheckpointManager(base_dir, model, tokenizer, optimizer, base_model_name=base_model_name, hf_token=hf_token)
    
    query_checkpoint_path = base_dir / "last_query_position.json"
    resume_from_query = None
    
    if query_checkpoint_path.exists():
        try:
            with open(query_checkpoint_path, "r") as f:
                query_checkpoint = json.load(f)
                # Prioritize last_processed_query for accurate resumption
                if query_checkpoint.get("last_processed_query") is not None:
                    resume_from_query = query_checkpoint.get("last_processed_query")
                    logger.info(f"Found query checkpoint! Last processed absolute query index: {resume_from_query}")
                elif query_checkpoint.get("original_idx") is not None: # Fallback to original_idx
                    # This case might indicate an older checkpoint or a point where last_processed_query wasn't reliably saved
                    resume_from_query = query_checkpoint.get("original_idx")
                    logger.warning(f"Found query checkpoint using original_idx (fallback): {resume_from_query}. This might occur with older checkpoints.")
                else:
                    logger.warning("Query checkpoint found but contains neither 'last_processed_query' nor 'original_idx'. Will rely on model checkpoint step.")
                
                # The 'batch_idx' and 'global_batch_idx' here are just for logging context from the last run
                last_run_batch_idx = query_checkpoint.get("batch_idx", "N/A")
                last_global_batch_idx = query_checkpoint.get("global_batch_idx", "N/A")
                logger.info(f"Context from last run checkpoint: run_batch_idx: {last_run_batch_idx}, global_batch_idx: {last_global_batch_idx}")

        except Exception as e:
            logger.error(f"Error reading query checkpoint: {e}")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Loading checkpoint via CheckpointManager...")
        # CheckpointManager will load the latest checkpoint or the base model if no checkpoint exists.
        # The model and tokenizer passed to CheckpointManager are now the ones potentially loaded in this function,
        # or the ones passed from main.py if they were not None.
        loaded_model, loaded_tokenizer, checkpoint_info = checkpoint_manager.load_latest_checkpoint()
        
        if loaded_model is not None:
            model = loaded_model
            if loaded_tokenizer is not None: # Tokenizer might also be updated by CheckpointManager
                tokenizer = loaded_tokenizer
            
            if str(model.device) == "cpu" and str(device) != "cpu":
                logger.info(f"Moving model from CPU to {device}...")
                model = model.to(device)
            
            if resume_from_query is not None:
                start_idx = resume_from_query + 1
                logger.info(f"✅ Resuming training from query {start_idx} (based on query checkpoint)")
            else:
                start_idx = checkpoint_info["step"] if checkpoint_info else 0
                logger.info(f"✅ Resuming training from step {start_idx} (based on model checkpoint)")
            
            model.eval()
            
            if not any(p.grad is not None for p in model.parameters()):
                logger.info("Optimizer state verified (no gradients found as expected).")
            else:
                logger.warning("Found gradients in model parameters. Clearing gradients...")
                optimizer.zero_grad()
        else:
            logger.info("ℹ️ No model loaded by CheckpointManager. This implies a fresh start or an issue.")
            # If model is still None here, it means neither main.py nor CheckpointManager could load it.
            # This path should ideally not be hit if base_model_name and hf_token are valid
            # and the initial load at the top of this function succeeded.
            if model is None: # Double check, should have been loaded or raised error
                logger.error("CRITICAL: Model is still None after CheckpointManager.load_latest_checkpoint. Training cannot proceed.")
                return 
            start_idx = 0
            logger.info("ℹ️ Starting fresh training run (or continuing with model loaded at function start).")

    except Exception as e:
        logger.error(f"❌ Error during checkpoint loading sequence in run_manual_ppo: {e}")
        if resume_from_query is not None:
            start_idx = resume_from_query + 1
            logger.info(f"✅ Resuming from query {start_idx} based on query checkpoint (despite model load error in sequence)")
        else:
            start_idx = 0
            logger.info("ℹ️ Starting fresh training run due to loading error in sequence")

    total_rows = len(df)
    if start_idx >= total_rows:
        logger.info("✅ Training already completed!")
        return

    df_to_process = df.iloc[start_idx:]
    
    validation_size = min(100, len(df_to_process) // 10) if len(df_to_process) > 0 else 0
    if validation_size > 0:
        validation_data = df_to_process.sample(n=validation_size, random_state=42)
        df_to_process = df_to_process.drop(validation_data.index)
    else:
        validation_data = pd.DataFrame() # Empty dataframe if no validation data
        logger.info("No validation data to sample.")
    
    processor = DataProcessor(model, tokenizer, device, checkpoint_base_dir=base_dir, checkpoint_manager=checkpoint_manager, optimizer=optimizer, base_model_name=base_model_name, hf_token=hf_token)
    processor.dataset_start_idx = start_idx
    
    try:
        batch_start = 0           # Initialize batch_start
        batch_results = []        # Initialize batch_results
        validation_results = None # Initialize validation_results
        current_query_idx = start_idx  # Initialize current query position (absolute index)
        
        # Process in batches
        for batch_idx, batch_start in enumerate(tqdm(range(0, len(df_to_process), processor.batch_size), desc="Manual PPO Training")):
            batch_end = min(batch_start + processor.batch_size, len(df_to_process))
            batch_data = df_to_process.iloc[batch_start:batch_end]
            
            # Process batch
            batch_results, current_query_idx = processor.process_batch(batch_data, batch_idx)
            
            # Run validation every 10 batches
            if batch_idx % 10 == 0:
                validation_results = processor.validate_model(validation_data, batch_idx)
            
            # Save checkpoint, flush logs, and cleanup periodically (e.g., every 50 batches)
            if checkpoint_manager and batch_idx > 0 and batch_idx % 50 == 0: # Adjusted frequency
                processor._flush_logs() # Flush logs before checkpointing
                checkpoint_manager.save_checkpoint(current_query_idx, batch_results, validation_results)
                checkpoint_manager.cleanup_old_checkpoints(keep_last_n=2)
                
                # Log checkpoint info
                logger.info(f"Checkpoint saved in: {checkpoint_manager.checkpoint_dir / f'checkpoint_{current_query_idx}'}")
                logger.info(f"Metrics saved in: {checkpoint_manager.checkpoint_dir / 'training_metrics.json'}")
            
            # Clear caches periodically (e.g., every 50 batches)
            if batch_idx > 0 and batch_idx % 50 == 0: # Adjusted frequency
                clear_caches()
                clear_response_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopping training...")
        
        # Try multiple sources to find the most reliable position
        current_position = None
        
        # 1. Try the last_processed_position.txt file (most reliable, written at end of batch)
        last_position_path = base_dir / "last_processed_position.txt"
        if last_position_path.exists():
            try:
                with open(last_position_path, "r") as f:
                    current_position = int(f.read().strip())
                    logger.info(f"Found batch-end position from file: {current_position}")
            except Exception as e:
                logger.error(f"Error reading batch-end position: {e}")
        
        # 2. Try the query checkpoint file (updated during batch)
        if current_position is None:
            query_checkpoint_path = base_dir / "last_query_position.json"
            if query_checkpoint_path.exists():
                try:
                    with open(query_checkpoint_path, "r") as f:
                        query_checkpoint = json.load(f)
                        current_position = query_checkpoint.get("last_processed_query")
                        logger.info(f"Found position from query checkpoint: {current_position}")
                except Exception as e:
                    logger.error(f"Error reading query checkpoint: {e}")
        
        # 3. Last resort: use the in-memory variable
        if current_position is None and 'current_query_idx' in locals():
            current_position = current_query_idx
            logger.info(f"Using in-memory position: {current_position}")
        
        # Use the most reliable position we found
        if current_position is not None:
            logger.info(f"Current dataset position at interruption: {current_position}")
            
            # Save query position checkpoint
            try:
                query_checkpoint_to_save = { # Renamed to avoid conflict with loaded query_checkpoint
                    "last_processed_query": int(current_position), # This is current_query_position
                    "original_idx": int(query_checkpoint.get("original_idx")) if query_checkpoint and query_checkpoint.get("original_idx") is not None else int(current_position), # Preserve original_idx if available from loaded checkpoint, else estimate
                    "batch_idx": batch_idx if 'batch_idx' in locals() else 0, # Run-specific batch_idx
                    "global_batch_idx": (int(current_position) // processor.batch_size) if processor and hasattr(processor, 'batch_size') and processor.batch_size > 0 else -1, # Global batch_idx based on current_position
                    "timestamp": time.time()
                }
                
                # Create checkpoint dir if needed
                query_checkpoint_path = base_dir / "last_query_position.json"
                query_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the checkpoint
                with open(query_checkpoint_path, "w") as f:
                    json.dump(query_checkpoint_to_save, f)
                    
                logger.info(f"✅ Saved query position checkpoint at dataset position {current_position}")
            except Exception as e:
                logger.error(f"❌ Error saving query position checkpoint: {e}")
        
        # Flush logs before saving checkpoint
        if processor:
            logger.info("Flushing logs before saving final checkpoint...")
            processor._flush_logs()
        
        # Save final model checkpoint before exiting
        if checkpoint_manager:
            try:
                # Use current_position instead of current_query_idx
                if current_position is not None:
                    logger.info(f"Saving final model checkpoint at dataset position {current_position}")
                    
                    # Ensure variables exist before calling save_checkpoint, using defaults if needed
                    final_batch_results = batch_results if 'batch_results' in locals() else []
                    final_validation_results = validation_results if 'validation_results' in locals() else None
                    
                    # Debug batch_results content
                    if final_batch_results:
                        logger.info(f"Saving checkpoint with {len(final_batch_results)} results")
                        # Check a sample result structure
                        sample_result = final_batch_results[0]
                        logger.info(f"Sample result structure: idx={sample_result.get('idx')}, has_reward={isinstance(sample_result.get('reward'), (float, int, torch.Tensor))}")
                        
                        # Calculate and verify metrics before saving
                        num_results = len(final_batch_results)
                        if num_results > 0:
                            avg_reward = sum(r["reward"].item() if isinstance(r["reward"], torch.Tensor) else r["reward"] for r in final_batch_results) / num_results
                            logger.info(f"Verified metrics before saving - Avg Reward: {avg_reward:.4f}")
                    else:
                        logger.warning("No batch results to save! Checkpoint will have empty metrics.")

                    checkpoint_manager.save_checkpoint(current_position, final_batch_results, final_validation_results)
                    logger.info(f"Final checkpoint saved in: {checkpoint_manager.checkpoint_dir / f'checkpoint_{current_position}'}")
                else:
                    logger.error("Cannot save model checkpoint: no position information available")
            except UnboundLocalError as e:
                # This should ideally not happen now, but catch just in case
                logger.error(f"❌ UnboundLocalError during final save attempt: {e}. Could not save final checkpoint.")
            except Exception as e:
                logger.error(f"❌ Unexpected error during final save attempt: {e}. Could not save final checkpoint.")
                logger.exception("Detailed traceback:")
        
    finally:
        # Ensure all logs are flushed before exiting
        if processor:
            processor._flush_logs()
        # Cleanup temporary directory
        if checkpoint_manager and hasattr(checkpoint_manager, 'temp_dir') and checkpoint_manager.temp_dir.exists():
            shutil.rmtree(checkpoint_manager.temp_dir)

    logger.info("✅ PPO training complete. Log saved to logs/ppo_manual_log.csv")
    logger.info(f"All checkpoints and metrics saved in: {checkpoint_manager.checkpoint_dir}")
