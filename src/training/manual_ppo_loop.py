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

from judge import (
    judge_coherence,
    judge_helpfulness,
    judge_ad_salience,
    judge_detectability
)
from generate.generator import generate_responses
from judge.utils import clear_caches


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
        self.lock = threading.Lock()
        
    def save_checkpoint(self, current_step, batch_results):
        """Save checkpoint atomically with verification."""
        with self.lock:
            try:
                # Create temporary checkpoint directory
                temp_checkpoint_dir = self.temp_dir / f"checkpoint_{current_step}"
                temp_checkpoint_dir.mkdir(exist_ok=True)
                
                # Save model and tokenizer
                self.model.save_pretrained(
                    temp_checkpoint_dir,
                    safe_serialization=True,  # Use safetensors
                    max_shard_size="2GB"  # Split into smaller files
                )
                self.tokenizer.save_pretrained(temp_checkpoint_dir)
                
                # Save optimizer state
                torch.save(
                    self.optimizer.state_dict(),
                    temp_checkpoint_dir / "optimizer.pt"
                )
                
                # Save checkpoint info
                checkpoint_info = {
                    "step": current_step,
                    "timestamp": time.time(),
                    "batch_results": batch_results
                }
                with open(temp_checkpoint_dir / "checkpoint_info.json", "w") as f:
                    json.dump(checkpoint_info, f)
                
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
                
                print(f"✅ Checkpoint saved successfully at step {current_step}")
                
            except Exception as e:
                print(f"❌ Error saving checkpoint: {e}")
                if temp_checkpoint_dir.exists():
                    shutil.rmtree(temp_checkpoint_dir)
                raise
    
    def _verify_checkpoint(self, checkpoint_dir):
        """Verify checkpoint integrity."""
        required_files = [
            "config.json",
            "model.safetensors",
            "optimizer.pt",
            "checkpoint_info.json"
        ]
        
        for file in required_files:
            if not (checkpoint_dir / file).exists():
                raise ValueError(f"Missing required file: {file}")
    
    def load_latest_checkpoint(self):
        """Load the latest checkpoint if available."""
        if not self.checkpoint_info_path.exists():
            return None
            
        try:
            with open(self.checkpoint_info_path, "r") as f:
                info = json.load(f)
            
            latest_checkpoint = Path(info["latest_checkpoint"])
            if not latest_checkpoint.exists():
                return None
            
            # Load model and tokenizer
            self.model = self.model.from_pretrained(
                latest_checkpoint,
                local_files_only=True,
                torch_dtype=torch.float16
            )
            self.tokenizer = self.tokenizer.from_pretrained(
                latest_checkpoint,
                local_files_only=True
            )
            
            # Load optimizer state
            optimizer_state = torch.load(latest_checkpoint / "optimizer.pt")
            self.optimizer.load_state_dict(optimizer_state)
            
            # Load checkpoint info
            with open(latest_checkpoint / "checkpoint_info.json", "r") as f:
                checkpoint_info = json.load(f)
            
            print(f"✅ Loaded checkpoint from step {checkpoint_info['step']}")
            return checkpoint_info
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
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
    def __init__(self, model, tokenizer, device, batch_size=4, checkpoint_manager=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.result_queue = Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.checkpoint_manager = checkpoint_manager
        self.current_step = 0
        
    def process_batch(self, batch_data):
        """Process a batch of data in parallel."""
        results = []
        for _, row in batch_data.iterrows():
            query = str(row['vague_query'])
            ad_facts = {
                "ad_product": str(row['ad_product']),
                "brand": str(row['brand']),
                "url": str(row['url']),
                "description": str(row['ad_description']),
            }

            try:
                with torch.no_grad():
                    response_without_ad, response_with_ad = generate_responses(query, ad_facts, self.model, self.tokenizer)

                ad_text = f"""Product: {ad_facts['ad_product']}
                            Brand: {ad_facts['brand']}
                            URL: {ad_facts['url']}
                            Description: {ad_facts['description']}"""

                # Run judges in parallel
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    future_coh = executor.submit(judge_coherence, query, response_with_ad)
                    future_help = executor.submit(judge_helpfulness, query, response_with_ad)
                    future_sal = executor.submit(judge_ad_salience, query, response_with_ad, ad_text)
                    future_det = executor.submit(judge_detectability, response_with_ad, response_without_ad)

                    # Wait for all futures to complete
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
                reward = torch.tensor(sum(reward_values) / len(reward_values), dtype=torch.float32).to(self.device)

                results.append({
                    "query": query,
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
                print(f"❌ Error processing query: {e}")
                continue

        return results

    def producer(self, df_to_process):
        """Producer thread that processes data and puts results in queue."""
        for batch_start in range(0, len(df_to_process), self.batch_size):
            if self.stop_event.is_set():
                break
                
            batch_end = min(batch_start + self.batch_size, len(df_to_process))
            batch_data = df_to_process.iloc[batch_start:batch_end]
            
            # Process batch
            batch_results = self.process_batch(batch_data)
            
            # Put results in queue
            for result in batch_results:
                self.result_queue.put((batch_start, result))
            
            # Save checkpoint periodically
            if self.checkpoint_manager and batch_start % (self.batch_size * 10) == 0:
                self.checkpoint_manager.save_checkpoint(batch_start, batch_results)
                self.checkpoint_manager.cleanup_old_checkpoints()
            
            # Clear caches periodically
            if batch_start % (self.batch_size * 10) == 0:
                clear_caches()

    def consumer(self, optimizer, log_path):
        """Consumer thread that updates model with results from queue."""
        while not self.stop_event.is_set() or not self.result_queue.empty():
            try:
                batch_start, result = self.result_queue.get(timeout=1)
                
                try:
                    input_ids = self.tokenizer(result["query"], return_tensors="pt", truncation=True, max_length=384).input_ids.to(self.device)
                    response_ids = self.tokenizer(result["response_with_ad"], return_tensors="pt", truncation=True, max_length=128).input_ids.to(self.device)[0]

                    if input_ids.shape[1] + response_ids.shape[0] > 512:
                        print(f"⚠️ Skipping: combined input too long")
                        continue

                    input_plus_response = torch.cat([input_ids[0], response_ids])
                    inputs = input_plus_response.unsqueeze(0)
                    labels = input_plus_response[1:]

                    self.model.train()
                    logits = self.model(inputs).logits[0, :-1]
                    new_log_probs = F.log_softmax(logits, dim=-1)[torch.arange(len(labels)), labels].sum()
                    
                    loss = -new_log_probs * result["reward"].detach()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    self.model.eval()

                    # Log results
                    pd.DataFrame([{
                        "idx": batch_start,
                        "query": result["query"],
                        "response_trained": result["response_with_ad"],
                        "reward": result["reward"].item(),
                        "loss": loss.item(),
                        **{f"{k}_{i}": v for k, v in result["scores"]["coherence"].items() for i in range(1, 5)},
                        **{f"{k}_{i}": v for k, v in result["scores"]["helpfulness"].items() for i in range(1, 2)},
                        **{f"{k}_{i}": v for k, v in result["scores"]["salience"].items() for i in range(1, 4)},
                        "Detect_Cosine": result["scores"]["detectability"].get("detectability_cosine")
                    }]).to_csv(log_path, mode="a", header=False, index=False)

                except Exception as e:
                    print(f"❌ Error updating model: {e}")
                    continue

                self.result_queue.task_done()
                
            except Empty:
                continue

def run_manual_ppo(model, tokenizer):
    device = model.device
    model.eval()

    df = pd.read_csv("data/merged_queries_ads.csv")
    optimizer = torch.optim.SGD(model.parameters(), lr=1.4e-7)

    log_path = "logs/ppo_manual_log.csv"
    periodic_eval_log_path = "logs/periodic_eval_log.csv"
    checkpoint_dir = "checkpoints/ppo_manual"
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(periodic_eval_log_path), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir, model, tokenizer, optimizer)
    
    # Try to load latest checkpoint
    checkpoint_info = checkpoint_manager.load_latest_checkpoint()
    start_idx = checkpoint_info["step"] if checkpoint_info else 0
    
    if start_idx > 0:
        print(f"✅ Resuming training from step {start_idx}")
    else:
        print("ℹ️ Starting fresh training run")

    # Prepare data
    total_rows = len(df)
    if start_idx >= total_rows:
        print("✅ Training already completed!")
        return

    df_to_process = df.iloc[start_idx:]
    
    # Initialize processor with checkpoint manager
    processor = DataProcessor(model, tokenizer, device, checkpoint_manager=checkpoint_manager)
    
    # Start producer and consumer threads
    producer_thread = threading.Thread(target=processor.producer, args=(df_to_process,))
    consumer_thread = threading.Thread(target=processor.consumer, args=(optimizer, log_path))
    
    try:
        producer_thread.start()
        consumer_thread.start()
        
        # Monitor progress
        pbar = tqdm(total=len(df_to_process), desc="Manual PPO Training")
        last_processed = start_idx
        
        while producer_thread.is_alive() or consumer_thread.is_alive():
            current_size = processor.result_queue.qsize()
            pbar.update(current_size - last_processed)
            last_processed = current_size
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Stopping training...")
        processor.stop_event.set()
        
        # Save final checkpoint before exiting
        if checkpoint_manager:
            checkpoint_manager.save_checkpoint(last_processed, [])
        
    finally:
        producer_thread.join()
        consumer_thread.join()
        pbar.close()
        
        # Cleanup temporary directory
        if checkpoint_manager:
            shutil.rmtree(checkpoint_manager.temp_dir)

    print("✅ PPO training complete. Log saved to logs/ppo_manual_log.csv")
