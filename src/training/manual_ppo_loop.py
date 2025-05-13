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

from judge import (
    judge_coherence,
    judge_helpfulness,
    judge_ad_salience,
    judge_detectability
)
from generate.generator import generate_responses, clear_response_cache
from judge.utils import clear_caches

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_range=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    return -torch.min(ratio * advantages, clipped * advantages).mean()

def compute_advantages(reward, value):
    return reward - value

class CheckpointManager:
    def __init__(self, checkpoint_dir, model, tokenizer, optimizer):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.checkpoint_info_path = self.checkpoint_dir / "checkpoint_info.json"
        self.metrics_path = self.checkpoint_dir / "training_metrics.json"
        self.lock = threading.Lock()
        
        # Initialize metrics file if it doesn't exist
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w") as f:
                json.dump({
                    "checkpoints": [],
                    "best_reward": 0,
                    "best_checkpoint": None,
                    "training_history": []
                }, f, indent=2)
    
    def save_checkpoint(self, current_step, batch_results, validation_results=None):
        """Save checkpoint atomically with verification."""
        with self.lock:
            try:
                # Create temporary checkpoint directory
                temp_checkpoint_dir = self.temp_dir / f"checkpoint_{current_step}"
                temp_checkpoint_dir.mkdir(exist_ok=True)
                
                # Save model and tokenizer
                self.model.save_pretrained(
                    temp_checkpoint_dir,
                    safe_serialization=False,  # Use pytorch_model.bin instead of safetensors
                    max_shard_size="2GB"  # Split into smaller files
                )
                self.tokenizer.save_pretrained(temp_checkpoint_dir)
                
                # Save optimizer state
                torch.save(
                    self.optimizer.state_dict(),
                    temp_checkpoint_dir / "optimizer.pt"
                )
                
                # Convert batch results to serializable format
                serializable_results = []
                for result in batch_results:
                    serializable_result = {
                        "idx": result["idx"],
                        "query": result["query"],
                        "ad_facts": result["ad_facts"],
                        "response_without_ad": result["response_without_ad"],
                        "response_with_ad": result["response_with_ad"],
                        "reward": result["reward"].item() if isinstance(result["reward"], torch.Tensor) else result["reward"],
                        "scores": {
                            "coherence": result["scores"]["coherence"],
                            "helpfulness": result["scores"]["helpfulness"],
                            "salience": result["scores"]["salience"],
                            "detectability": result["scores"]["detectability"]
                        }
                    }
                    serializable_results.append(serializable_result)
                
                # Calculate metrics safely
                num_results = len(serializable_results)
                avg_reward = sum(r["reward"] for r in serializable_results) / num_results if num_results > 0 else 0.0
                avg_coherence = sum(r["scores"]["coherence"].get("Coherence Score", 0) for r in serializable_results) / num_results if num_results > 0 else 0.0
                avg_helpfulness = sum(r["scores"]["helpfulness"].get("Helpfulness Score", 0) for r in serializable_results) / num_results if num_results > 0 else 0.0
                avg_salience = sum(r["scores"]["salience"].get("Ad Salience Score", 0) for r in serializable_results) / num_results if num_results > 0 else 0.0
                avg_detectability = sum(r["scores"]["detectability"].get("detectability_cosine", 0) for r in serializable_results) / num_results if num_results > 0 else 0.0
                
                # Save checkpoint info
                checkpoint_info = {
                    "step": current_step,
                    "timestamp": time.time(),
                    "batch_results": serializable_results,
                    "metrics": {
                        "avg_reward": avg_reward,
                        "avg_coherence": avg_coherence,
                        "avg_helpfulness": avg_helpfulness,
                        "avg_salience": avg_salience,
                        "avg_detectability": avg_detectability
                    }
                }
                
                if validation_results:
                    val_avg_reward = sum(r["reward"] for r in validation_results) / len(validation_results)
                    checkpoint_info["metrics"]["validation_avg_reward"] = val_avg_reward
                
                with open(temp_checkpoint_dir / "checkpoint_info.json", "w") as f:
                    json.dump(checkpoint_info, f, indent=2)
                
                # Update metrics file
                with open(self.metrics_path, "r") as f:
                    metrics = json.load(f)
                
                metrics["checkpoints"].append({
                    "step": current_step,
                    "path": str(temp_checkpoint_dir),
                    "metrics": checkpoint_info["metrics"]
                })
                
                # Update best checkpoint if validation reward is better
                if validation_results:
                    val_avg_reward = sum(r["reward"] for r in validation_results) / len(validation_results)
                    if val_avg_reward > metrics["best_reward"]:
                        metrics["best_reward"] = val_avg_reward
                        metrics["best_checkpoint"] = str(temp_checkpoint_dir)
                        logger.info(f"🎉 New best model found! Validation reward: {val_avg_reward:.4f}")
                
                # Add to training history
                metrics["training_history"].append({
                    "step": current_step,
                    "timestamp": time.time(),
                    "metrics": checkpoint_info["metrics"]
                })
                
                with open(self.metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                
                # Verify the saved files
                self._verify_checkpoint(temp_checkpoint_dir)
                
                # Atomic move to final location
                final_checkpoint_dir = self.checkpoint_dir / f"checkpoint_{current_step}"
                if final_checkpoint_dir.exists():
                    shutil.rmtree(final_checkpoint_dir)
                shutil.move(temp_checkpoint_dir, final_checkpoint_dir)
                
                # Update latest checkpoint info
                with open(self.checkpoint_info_path, "w") as f:
                    json.dump({"latest_checkpoint": str(final_checkpoint_dir)}, f)
                
                logger.info(f"✅ Checkpoint saved successfully at step {current_step}")
                logger.info(f"📊 Metrics - Reward: {avg_reward:.4f}, Coherence: {avg_coherence:.4f}, "
                          f"Helpfulness: {avg_helpfulness:.4f}, Salience: {avg_salience:.4f}, "
                          f"Detectability: {avg_detectability:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error saving checkpoint: {e}")
                if temp_checkpoint_dir.exists():
                    shutil.rmtree(temp_checkpoint_dir)
                raise
    
    def _verify_checkpoint(self, checkpoint_dir):
        """Verify checkpoint integrity."""
        required_files = [
            "config.json",
            "optimizer.pt",
            "checkpoint_info.json"
        ]
        
        # Check for model files (either safetensors or pytorch_model.bin)
        model_files = ["model.safetensors", "pytorch_model.bin"]
        has_model_file = any((checkpoint_dir / file).exists() for file in model_files)
        if not has_model_file:
            raise ValueError("Missing model file (neither model.safetensors nor pytorch_model.bin found)")
        
        # Check other required files
        for file in required_files:
            if not (checkpoint_dir / file).exists():
                raise ValueError(f"Missing required file: {file}")
    
    def load_latest_checkpoint(self):
        """Load the latest checkpoint if available."""
        if not self.checkpoint_info_path.exists():
            logger.info("No checkpoint info found. Starting fresh training.")
            return None
            
        try:
            with open(self.checkpoint_info_path, "r") as f:
                info = json.load(f)
            
            latest_checkpoint = Path(info["latest_checkpoint"])
            if not latest_checkpoint.exists():
                logger.warning(f"Checkpoint directory {latest_checkpoint} not found. Starting fresh training.")
                return None
            
            logger.info(f"Loading checkpoint from {latest_checkpoint}")
            
            # Load model state
            try:
                self.model = self.model.from_pretrained(
                    latest_checkpoint,
                    local_files_only=True,
                    torch_dtype=torch.float16
                )
                logger.info("✅ Model state loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading model state: {e}")
                return None
            
            # Load tokenizer
            try:
                self.tokenizer = self.tokenizer.from_pretrained(
                    latest_checkpoint,
                    local_files_only=True
                )
                logger.info("✅ Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading tokenizer: {e}")
                return None
            
            # Load optimizer state
            try:
                optimizer_path = latest_checkpoint / "optimizer.pt"
                if optimizer_path.exists():
                    optimizer_state = torch.load(optimizer_path)
                    self.optimizer.load_state_dict(optimizer_state)
                    logger.info("✅ Optimizer state loaded successfully")
                else:
                    logger.warning("No optimizer state found in checkpoint")
            except Exception as e:
                logger.error(f"❌ Error loading optimizer state: {e}")
                return None
            
            # Load checkpoint info
            try:
                with open(latest_checkpoint / "checkpoint_info.json", "r") as f:
                    checkpoint_info = json.load(f)
                logger.info(f"✅ Checkpoint info loaded successfully. Step: {checkpoint_info['step']}")
                return checkpoint_info
            except Exception as e:
                logger.error(f"❌ Error loading checkpoint info: {e}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Clean up old checkpoints, keeping only the last N."""
        try:
            checkpoints = sorted(
                [d for d in self.checkpoint_dir.glob("checkpoint_*") if d.is_dir()],
                key=lambda x: int(x.name.split("_")[1])
            )
            
            for checkpoint in checkpoints[:-keep_last_n]:
                shutil.rmtree(checkpoint)
                print(f"Cleaned up old checkpoint: {checkpoint}")
                
        except Exception as e:
            print(f"❌ Error cleaning up checkpoints: {e}")

class DataProcessor:
    def __init__(self, model, tokenizer, device, batch_size=32, checkpoint_manager=None, optimizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_manager = checkpoint_manager
        self.optimizer = optimizer
        self.current_step = 0
        
        # Statistics tracking
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
        
        # Create headers for log files with batch information
        if not self.generation_log.exists() or os.path.getsize(self.generation_log) == 0:
            pd.DataFrame(columns=[
                "batch_idx", "query_idx", "query", "ad_facts", "response_without_ad", "response_with_ad",
                "generation_time", "token_count"
            ]).to_csv(self.generation_log, index=False)
        
        if not self.judging_log.exists() or os.path.getsize(self.judging_log) == 0:
            pd.DataFrame(columns=[
                "batch_idx", "query_idx", "query", "response_with_ad", "coherence_score", "helpfulness_score",
                "salience_score", "detectability_score", "judging_time"
            ]).to_csv(self.judging_log, index=False)
        
        if not self.training_log.exists() or os.path.getsize(self.training_log) == 0:
            pd.DataFrame(columns=[
                "batch_idx", "query_idx", "query", "response_with_ad", "reward", "loss", "training_time"
            ]).to_csv(self.training_log, index=False)
        
        if not self.stats_log.exists() or os.path.getsize(self.stats_log) == 0:
            pd.DataFrame(columns=[
                "batch_idx", "timestamp", "total_queries", "successful_queries", "failed_queries",
                "avg_generation_time", "avg_judging_time", "avg_training_time", "avg_reward", "avg_loss",
                "total_tokens", "memory_usage"
            ]).to_csv(self.stats_log, index=False)

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

    def update_stats(self, batch_idx, gen_time, judge_time, train_time, token_count, reward, loss, success=True):
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
            pd.DataFrame([{
                "batch_idx": batch_idx,
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
                    
                    loss = -new_log_probs * reward.detach()

                    validation_results.append({
                        "query": query,
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
        logger.info(f"🔄 Starting batch {batch_idx} with {len(batch_data)} queries")
        batch_start_time = time.time()
        
        for idx, row in batch_data.iterrows():
            query = str(row['vague_query'])
            ad_facts = {
                "ad_product": str(row['ad_product']),
                "brand": str(row['brand']),
                "url": str(row['url']),
                "description": str(row['ad_description']),
            }

            try:
                # Stage 1: Generation
                logger.info(f"🔄 Batch {batch_idx} - Starting generation for query {idx}")
                gen_start_time = time.time()
                
                with torch.no_grad():
                    response_without_ad, response_with_ad = generate_responses(query, ad_facts, self.model, self.tokenizer)
                
                gen_time = time.time() - gen_start_time
                token_count = len(self.tokenizer.encode(response_with_ad))
                
                # Log generation results
                self.generation_log_buffer.append({
                    "batch_idx": batch_idx,
                    "query_idx": idx,
                    "query": query,
                    "ad_facts": json.dumps(ad_facts),
                    "response_without_ad": response_without_ad,
                    "response_with_ad": response_with_ad,
                    "generation_time": gen_time,
                    "token_count": token_count
                })
                
                logger.info(f"✅ Batch {batch_idx} - Generation complete for query {idx} in {gen_time:.2f}s")

                # Stage 2: Judging
                logger.info(f"🔄 Batch {batch_idx} - Starting judging for query {idx}")
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
                
                # Log judging results
                self.judging_log_buffer.append({
                    "batch_idx": batch_idx,
                    "query_idx": idx,
                    "query": query,
                    "response_with_ad": response_with_ad,
                    "coherence_score": score_coh.get("Coherence Score", 0),
                    "helpfulness_score": score_help.get("Helpfulness Score", 0),
                    "salience_score": score_sal.get("Ad Salience Score", 0),
                    "detectability_score": score_det.get("detectability_cosine", 0),
                    "judging_time": judge_time
                })
                
                logger.info(f"✅ Batch {batch_idx} - Judging complete for query {idx} in {judge_time:.2f}s")

                # Stage 3: Training
                logger.info(f"🔄 Batch {batch_idx} - Starting training for query {idx}")
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
                    "batch_idx": batch_idx,
                    "query_idx": idx,
                    "query": query,
                    "response_with_ad": response_with_ad,
                    "reward": reward.item(),
                    "loss": loss.item(),
                    "training_time": train_time
                })
                
                logger.info(f"✅ Batch {batch_idx} - Training complete for query {idx} in {train_time:.2f}s")

                # Update statistics
                self.update_stats(batch_idx, gen_time, judge_time, train_time, token_count, reward.item(), loss.item(), success=True)

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

            except Exception as e:
                logger.error(f"❌ Batch {batch_idx} - Error processing query {idx}: {e}")
                self.update_stats(batch_idx, 0, 0, 0, 0, 0, 0, success=False)
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
        return results

def run_manual_ppo(model, tokenizer):
    device = model.device
    model.eval()

    df = pd.read_csv("data/merged_queries_ads.csv")
    optimizer = torch.optim.SGD(model.parameters(), lr=1.4e-7)

    # Set up all required directories
    base_dir = Path("checkpoints/ppo_manual")
    log_dir = Path("logs")
    
    # Create directory structure
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

    # Set up file paths
    log_path = log_dir / "ppo_manual_log.csv"
    periodic_eval_log_path = log_dir / "periodic_eval_log.csv"
    checkpoint_dir = base_dir
    optimizer_path = checkpoint_dir / "optimizer.pt"

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir, model, tokenizer, optimizer)
    
    # Try to load latest checkpoint
    checkpoint_info = checkpoint_manager.load_latest_checkpoint()
    if checkpoint_info:
        start_idx = checkpoint_info["step"]
        logger.info(f"✅ Resuming training from step {start_idx}")
        
        # Verify model is in eval mode
        model.eval()
        
        # Verify optimizer state
        if not any(p.grad is not None for p in model.parameters()):
            logger.info("Optimizer state verified")
        else:
            logger.warning("Found gradients in model parameters. Clearing gradients...")
            optimizer.zero_grad()
    else:
        start_idx = 0
        logger.info("ℹ️ Starting fresh training run")

    # Prepare data
    total_rows = len(df)
    if start_idx >= total_rows:
        logger.info("✅ Training already completed!")
        return

    df_to_process = df.iloc[start_idx:]
    
    # Create validation set (10% of data)
    validation_size = min(100, len(df_to_process) // 10)
    validation_data = df_to_process.sample(n=validation_size, random_state=42)
    df_to_process = df_to_process.drop(validation_data.index)
    
    # Initialize processor with optimizer
    processor = DataProcessor(model, tokenizer, device, checkpoint_manager=checkpoint_manager, optimizer=optimizer)
    
    try:
        batch_start = 0           # Initialize batch_start
        batch_results = []        # Initialize batch_results
        validation_results = None # Initialize validation_results
        # Process in batches
        for batch_idx, batch_start in enumerate(tqdm(range(0, len(df_to_process), processor.batch_size), desc="Manual PPO Training")):
            batch_end = min(batch_start + processor.batch_size, len(df_to_process))
            batch_data = df_to_process.iloc[batch_start:batch_end]
            
            # Process batch
            batch_results = processor.process_batch(batch_data, batch_idx)
            
            # Run validation every 10 batches
            if batch_idx % 10 == 0:
                validation_results = processor.validate_model(validation_data, batch_idx)
            
            # Save checkpoint, flush logs, and cleanup periodically (e.g., every 50 batches)
            if checkpoint_manager and batch_idx > 0 and batch_idx % 50 == 0: # Adjusted frequency
                processor._flush_logs() # Flush logs before checkpointing
                checkpoint_manager.save_checkpoint(batch_start, batch_results, validation_results)
                checkpoint_manager.cleanup_old_checkpoints(keep_last_n=2)
                
                # Log checkpoint info
                logger.info(f"Checkpoint saved in: {checkpoint_dir / f'checkpoint_{batch_start}'}")
                logger.info(f"Metrics saved in: {checkpoint_dir / 'training_metrics.json'}")
            
            # Clear caches periodically (e.g., every 50 batches)
            if batch_idx > 0 and batch_idx % 50 == 0: # Adjusted frequency
                clear_caches()
                clear_response_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopping training...")
        
        # Save final checkpoint before exiting
        if checkpoint_manager:
            try:
                # Defensive check and logging before saving
                current_batch_start = batch_start if 'batch_start' in locals() else 'undefined'
                current_batch_results_len = len(batch_results) if 'batch_results' in locals() else 'undefined'
                current_validation_results_defined = 'validation_results' in locals()
                
                logger.info(f"Attempting final save. State: batch_start={current_batch_start}, batch_results defined={current_batch_results_len != 'undefined'}, validation_results defined={current_validation_results_defined}")
                
                # Ensure variables exist before calling save_checkpoint, using defaults if needed
                final_batch_start = batch_start if 'batch_start' in locals() else 0
                final_batch_results = batch_results if 'batch_results' in locals() else []
                final_validation_results = validation_results if 'validation_results' in locals() else None

                checkpoint_manager.save_checkpoint(final_batch_start, final_batch_results, final_validation_results)
                logger.info(f"Final checkpoint saved in: {checkpoint_dir / f'checkpoint_{final_batch_start}'}")
            except UnboundLocalError as e:
                # This should ideally not happen now, but catch just in case
                logger.error(f"❌ UnboundLocalError during final save attempt: {e}. Could not save final checkpoint.")
            except Exception as e:
                logger.error(f"❌ Unexpected error during final save attempt: {e}. Could not save final checkpoint.")
        
    finally:
        # Ensure all logs are flushed before exiting
        if processor:
            processor._flush_logs()
        # Cleanup temporary directory
        if checkpoint_manager:
            shutil.rmtree(checkpoint_manager.temp_dir)

    logger.info("✅ PPO training complete. Log saved to logs/ppo_manual_log.csv")
    logger.info(f"All checkpoints and metrics saved in: {checkpoint_dir}")
