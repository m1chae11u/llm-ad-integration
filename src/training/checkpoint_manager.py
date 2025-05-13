import os
import torch
import json
import shutil
import threading
import time
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING, Tuple

# For type hinting without circular imports
if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir, model: 'AutoModelForCausalLM', tokenizer: 'AutoTokenizer', optimizer: 'Optimizer'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.temp_dir = self.checkpoint_dir / "_temp_checkpoints"
        self.temp_dir.mkdir(exist_ok=True)
        # Store references - careful about modifications outside this class
        self._model = model 
        self._tokenizer = tokenizer
        self._optimizer = optimizer
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
    
    # Properties to access the potentially updated model/tokenizer/optimizer
    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def optimizer(self):
        return self._optimizer

    def save_checkpoint(self, current_step, batch_results, validation_results=None):
        """Save checkpoint atomically with verification."""
        with self.lock:
            try:
                # Create temporary checkpoint directory
                temp_checkpoint_dir = self.temp_dir / f"checkpoint_{current_step}"
                temp_checkpoint_dir.mkdir(exist_ok=True)
                
                # Define final checkpoint path early to use in references
                final_checkpoint_dir = self.checkpoint_dir / f"checkpoint_{current_step}"
                
                # Save model and tokenizer using internal references
                logger.info(f"Saving model to {temp_checkpoint_dir}...")
                self.model.save_pretrained(
                    temp_checkpoint_dir,
                    safe_serialization=False,  # Use pytorch_model.bin instead of safetensors
                    max_shard_size="2GB"  # Split into smaller files
                )
                logger.info("Model save call completed.")
                # Add immediate check after saving model
                model_bin_path = temp_checkpoint_dir / "pytorch_model.bin"
                sharded_files = list(temp_checkpoint_dir.glob('pytorch_model-*.bin'))
                # Check if either the single file or sharded files exist
                if model_bin_path.exists() or sharded_files:
                    logger.info(f"Verified model file(s) ({'single' if model_bin_path.exists() else 'sharded'}) exist immediately after save.")
                else:
                    logger.error(f"CRITICAL: Neither {model_bin_path} nor sharded files (pytorch_model-*.bin) exist immediately after save!")
                    # Optionally raise an error here immediately if preferred
                    # raise FileNotFoundError(f"Model file {model_bin_path} not found after save_pretrained")

                logger.info(f"Saving tokenizer to {temp_checkpoint_dir}...")
                self.tokenizer.save_pretrained(temp_checkpoint_dir)
                logger.info("Tokenizer save call completed.")
                
                # Save optimizer state using internal reference
                logger.info(f"Saving optimizer state to {temp_checkpoint_dir}...")
                torch.save(
                    self.optimizer.state_dict(),
                    temp_checkpoint_dir / "optimizer.pt"
                )
                
                # Convert batch results to serializable format
                serializable_results = []
                # Ensure batch_results is iterable and not None
                if batch_results is None:
                    logger.warning("batch_results is None, using empty list")
                    batch_results = []
                
                # Debug the batch_results
                logger.info(f"Processing {len(batch_results)} batch results for checkpoint")
                
                # Convert to serializable format with careful error handling
                for i, result in enumerate(batch_results):
                    try:
                        # Check that all required keys exist
                        required_keys = ["idx", "query", "ad_facts", "response_without_ad", "response_with_ad", "reward", "scores"]
                        missing_keys = [key for key in required_keys if key not in result]
                        
                        if missing_keys:
                            logger.warning(f"Skipping result {i} due to missing keys: {missing_keys}")
                            continue
                        
                        # Process the reward value carefully
                        if isinstance(result["reward"], torch.Tensor):
                            reward_value = result["reward"].item()
                        elif isinstance(result["reward"], (int, float)):
                            reward_value = result["reward"]
                        else:
                            logger.warning(f"Skipping result {i} due to invalid reward type: {type(result['reward'])}")
                            continue
                        
                        # Check scores structure
                        required_score_keys = ["coherence", "helpfulness", "salience", "detectability"]
                        missing_score_keys = [key for key in required_score_keys if key not in result["scores"]]
                        
                        if missing_score_keys:
                            logger.warning(f"Result {i} is missing score keys: {missing_score_keys}, using empty dicts")
                            # Add empty dicts for missing score types
                            for key in missing_score_keys:
                                result["scores"][key] = {}
                        
                        serializable_result = {
                            "idx": result["idx"],
                            "query": result["query"],
                            "ad_facts": result["ad_facts"],
                            "response_without_ad": result["response_without_ad"],
                            "response_with_ad": result["response_with_ad"],
                            "reward": reward_value,
                            "scores": {
                                "coherence": result["scores"]["coherence"],
                                "helpfulness": result["scores"]["helpfulness"],
                                "salience": result["scores"]["salience"],
                                "detectability": result["scores"]["detectability"]
                            }
                        }
                        serializable_results.append(serializable_result)
                    except Exception as e:
                        logger.warning(f"Error processing result {i}: {str(e)}")
                        continue
                
                # Calculate metrics safely
                num_results = len(serializable_results)
                logger.info(f"Successfully processed {num_results} serializable results")
                
                # Initialize metrics with defaults
                metrics = {
                    "avg_reward": 0.0,
                    "avg_coherence": 0.0, 
                    "avg_helpfulness": 0.0,
                    "avg_salience": 0.0,
                    "avg_detectability": 0.0
                }
                
                # Only calculate if we have results
                if num_results > 0:
                    try:
                        metrics["avg_reward"] = sum(r["reward"] for r in serializable_results) / num_results
                        metrics["avg_coherence"] = sum(r["scores"]["coherence"].get("Coherence Score", 0) for r in serializable_results) / num_results
                        metrics["avg_helpfulness"] = sum(r["scores"]["helpfulness"].get("Helpfulness Score", 0) for r in serializable_results) / num_results
                        metrics["avg_salience"] = sum(r["scores"]["salience"].get("Ad Salience Score", 0) for r in serializable_results) / num_results
                        metrics["avg_detectability"] = sum(r["scores"]["detectability"].get("detectability_cosine", 0) for r in serializable_results) / num_results
                        logger.info(f"Calculated metrics: reward={metrics['avg_reward']:.4f}")
                    except Exception as e:
                        logger.error(f"Error calculating metrics: {str(e)}")
                else:
                    logger.warning("No results to calculate metrics from")
                
                # Save checkpoint info
                checkpoint_info = {
                    "step": current_step,
                    "timestamp": time.time(),
                    "batch_results": serializable_results,
                    "metrics": metrics
                }
                
                # Add validation metrics if provided
                if validation_results:
                    try:
                        num_val_results = len(validation_results)
                        if num_val_results > 0:
                            val_rewards = []
                            for r in validation_results:
                                if "reward" in r:
                                    reward = r["reward"]
                                    if isinstance(reward, torch.Tensor):
                                        val_rewards.append(reward.item())
                                    elif isinstance(reward, (int, float)):
                                        val_rewards.append(reward)
                            
                            if val_rewards:
                                val_avg_reward = sum(val_rewards) / len(val_rewards)
                                checkpoint_info["metrics"]["validation_avg_reward"] = val_avg_reward
                                logger.info(f"Validation metrics calculated: avg_reward={val_avg_reward:.4f}")
                    except Exception as e:
                        logger.error(f"Error processing validation results: {str(e)}")
                
                with open(temp_checkpoint_dir / "checkpoint_info.json", "w") as f:
                    json.dump(checkpoint_info, f, indent=2)
                
                # Update metrics file
                try:
                    with open(self.metrics_path, "r") as f:
                        metrics_file = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                     metrics_file = { # Default structure if file is missing/corrupt
                        "checkpoints": [],
                        "best_reward": 0,
                        "best_checkpoint": None,
                        "training_history": []
                    }

                metrics_file["checkpoints"].append({
                    "step": current_step,
                    "path": str(final_checkpoint_dir), # Use final path here
                    "metrics": checkpoint_info["metrics"]
                })
                
                # Update best checkpoint if validation reward is better
                if validation_results and "validation_avg_reward" in checkpoint_info["metrics"]:
                    # We calculated val_avg_reward earlier
                    val_avg_reward = checkpoint_info["metrics"]["validation_avg_reward"]
                    if val_avg_reward > metrics_file.get("best_reward", 0): # Use .get for safety
                        metrics_file["best_reward"] = val_avg_reward
                        metrics_file["best_checkpoint"] = str(final_checkpoint_dir) # Use final path
                        logger.info(f"🎉 New best model found! Validation reward: {val_avg_reward:.4f}")
                
                # Add to training history
                metrics_file["training_history"].append({
                    "step": current_step,
                    "timestamp": time.time(),
                    "metrics": checkpoint_info["metrics"]
                })
                
                with open(self.metrics_path, "w") as f:
                    json.dump(metrics_file, f, indent=2)
                
                # Verify the saved files in the temporary directory
                self._verify_checkpoint(temp_checkpoint_dir)
                
                # Atomic move to final location
                if final_checkpoint_dir.exists():
                    shutil.rmtree(final_checkpoint_dir)
                shutil.move(str(temp_checkpoint_dir), str(final_checkpoint_dir)) # Use str() for move
                
                # Update latest checkpoint info to point to the final location
                with open(self.checkpoint_info_path, "w") as f:
                    json.dump({"latest_checkpoint": str(final_checkpoint_dir)}, f)
                
                logger.info(f"✅ Checkpoint saved successfully at step {current_step} to {final_checkpoint_dir}")
                logger.info(f"📊 Metrics - Reward: {metrics['avg_reward']:.4f}, Coherence: {metrics['avg_coherence']:.4f}, "
                          f"Helpfulness: {metrics['avg_helpfulness']:.4f}, Salience: {metrics['avg_salience']:.4f}, "
                          f"Detectability: {metrics['avg_detectability']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error saving checkpoint: {e}")
                # Attempt to clean up the specific temporary checkpoint dir if it exists
                if 'temp_checkpoint_dir' in locals() and temp_checkpoint_dir.exists():
                    shutil.rmtree(temp_checkpoint_dir)
                raise # Re-raise the exception after logging and cleanup attempt
    
    def _verify_checkpoint(self, checkpoint_dir):
        """Verify checkpoint integrity."""
        required_files = [
            "config.json",
            "optimizer.pt",
            "checkpoint_info.json"
        ]
        
        # Check for model files (single bin, safetensors, or sharded bin)
        model_safetensors_path = checkpoint_dir / "model.safetensors"
        model_bin_path = checkpoint_dir / "pytorch_model.bin"
        sharded_model_files = list(checkpoint_dir.glob('pytorch_model-*.bin'))
        
        has_model_file = model_safetensors_path.exists() or model_bin_path.exists() or bool(sharded_model_files)
        
        if not has_model_file:
            raise ValueError("Missing model file (neither model.safetensors, pytorch_model.bin, nor sharded pytorch_model-*.bin found)")
        
        # Check other required files
        for file in required_files:
            if not (checkpoint_dir / file).exists():
                raise ValueError(f"Missing required file: {file}")
    
    def load_latest_checkpoint(self) -> Tuple[Optional['AutoModelForCausalLM'], Optional['AutoTokenizer'], Optional[Dict]]:
        """Load the latest checkpoint if available.

        Returns:
            Tuple[Optional['AutoModelForCausalLM'], Optional['AutoTokenizer'], Optional[Dict]]: 
                The loaded model, tokenizer, and checkpoint info dict, or (None, None, None) if loading fails.
        """
        if not self.checkpoint_info_path.exists():
            logger.info("No top-level checkpoint info found. Starting fresh training.")
            return None, None, None
            
        try:
            with open(self.checkpoint_info_path, "r") as f:
                info = json.load(f)
            
            latest_checkpoint = Path(info["latest_checkpoint"])
            if not latest_checkpoint.exists():
                logger.warning(f"Checkpoint directory {latest_checkpoint} not found. Starting fresh training.")
                return None, None, None
            
            logger.info(f"Loading checkpoint from {latest_checkpoint}")
            
            loaded_model = None
            loaded_tokenizer = None
            checkpoint_info = None

            # Clear memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Try to load optimizer state first since it's smaller
            try:
                optimizer_path = latest_checkpoint / "optimizer.pt"
                if optimizer_path.exists():
                    # Use map_location='cpu' to avoid OOM issues
                    optimizer_state = torch.load(optimizer_path, map_location='cpu')
                    self.optimizer.load_state_dict(optimizer_state) # Load into the existing optimizer instance
                    logger.info("✅ Optimizer state loaded successfully")
                else:
                    logger.warning("No optimizer state found in checkpoint")
            except Exception as e:
                logger.error(f"❌ Error loading optimizer state: {e}. Continuing without loading optimizer state.")

            # Load checkpoint info
            try:
                with open(latest_checkpoint / "checkpoint_info.json", "r") as f:
                    checkpoint_info = json.load(f)
                logger.info(f"✅ Checkpoint info loaded successfully. Step: {checkpoint_info['step']}")
            except Exception as e:
                logger.error(f"❌ Error loading checkpoint info: {e}")
            
            # Load tokenizer
            try:
                # Use the class of the internally held tokenizer to load
                tokenizer_class = self.tokenizer.__class__
                loaded_tokenizer = tokenizer_class.from_pretrained(
                    latest_checkpoint,
                    local_files_only=True
                )
                self._tokenizer = loaded_tokenizer # Update internal tokenizer reference
                logger.info("✅ Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading tokenizer: {e}")
                return None, None, None
            
            # Load model state with extra safeguards
            try:
                # Clear CUDA cache again before loading the model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                logger.info("Loading model with CPU first to avoid memory issues...")
                # Use the class of the internally held model to load
                model_class = self.model.__class__ 
                
                # First load to CPU with minimal memory usage
                loaded_model = model_class.from_pretrained(
                    latest_checkpoint,
                    local_files_only=True,
                    torch_dtype=torch.float16,
                    device_map="cpu",  # Force CPU loading
                    low_cpu_mem_usage=True,  # Minimize CPU memory usage
                )
                
                self._model = loaded_model # Update internal model reference
                logger.info("✅ Model state loaded successfully to CPU")
                
                # Return the loaded model/tokenizer as is, let the caller move to GPU if needed
            except Exception as e:
                logger.error(f"❌ Error loading model state: {e}")
                return None, None, None
            
            # Return the loaded objects and info
            return loaded_model, loaded_tokenizer, checkpoint_info
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            return None, None, None
    
    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Clean up old checkpoints, keeping only the last N."""
        try:
            checkpoints = sorted(
                [d for d in self.checkpoint_dir.glob("checkpoint_*") if d.is_dir()],
                key=lambda x: int(x.name.split("_")[1])
            )
            
            # Determine which checkpoints to remove
            to_remove = checkpoints[:-keep_last_n]

            # Avoid deleting the best checkpoint if it's marked for removal
            try:
                with open(self.metrics_path, "r") as f:
                    metrics = json.load(f)
                best_checkpoint_path = metrics.get("best_checkpoint")
                if best_checkpoint_path:
                    best_checkpoint_path = Path(best_checkpoint_path)
                    to_remove = [cp for cp in to_remove if cp != best_checkpoint_path]
            except (FileNotFoundError, json.JSONDecodeError):
                 logger.warning("Metrics file not found or corrupt, cannot preserve best checkpoint during cleanup.")


            for checkpoint in to_remove:
                shutil.rmtree(checkpoint)
                logger.info(f"Cleaned up old checkpoint: {checkpoint}")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up checkpoints: {e}")
