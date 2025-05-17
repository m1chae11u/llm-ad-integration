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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# For type hinting without circular imports
if TYPE_CHECKING:
    # from transformers import AutoModelForCausalLM, AutoTokenizer # Already imported above
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir, model: Optional['AutoModelForCausalLM'], tokenizer: Optional['AutoTokenizer'], optimizer: Optional['Optimizer'], base_model_name: Optional[str] = None, hf_token: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.temp_dir = self.checkpoint_dir / "_temp_checkpoints"
        self.temp_dir.mkdir(exist_ok=True)
        self._model = model 
        self._tokenizer = tokenizer
        self._optimizer = optimizer
        self.base_model_name = base_model_name
        self.hf_token = hf_token
        self.checkpoint_info_path = self.checkpoint_dir / "checkpoint_info.json"
        self.metrics_path = self.checkpoint_dir / "training_metrics.json"
        self.lock = threading.Lock()
        
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w") as f:
                json.dump({
                    "checkpoints": [],
                    "best_reward": float('-inf'), # Initialize with negative infinity
                    "best_checkpoint": None,
                    "training_history": []
                }, f, indent=2)
    
    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def optimizer(self):
        return self._optimizer

    def _load_base_model_and_tokenizer(self) -> Tuple[Optional['AutoModelForCausalLM'], Optional['AutoTokenizer']]:
        """Loads the base model and tokenizer using base_model_name and hf_token."""
        if not self.base_model_name:
            logger.error("Cannot load base model: base_model_name is not set.")
            return None, None
        
        loaded_tokenizer = None
        loaded_model = None

        try:
            logger.info(f"Loading base tokenizer: {self.base_model_name}")
            loaded_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True, token=self.hf_token)
            logger.info(f"✅ Successfully loaded base tokenizer {self.base_model_name}.")
        except Exception as e:
            logger.error(f"❌ Failed to load base tokenizer {self.base_model_name}: {e}")
            return None, None # Tokenizer is essential for model loading

        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto", # Let transformers handle device mapping if possible, else move later
                trust_remote_code=True,
                use_cache=True,
                low_cpu_mem_usage=True,
                token=self.hf_token
            )
            # Ensure model is on the determined device, especially if device_map="auto" doesn't place all parts on GPU
            if loaded_model.device != device and device.type == 'cuda': # Check if not already on target CUDA device
                try:
                    loaded_model.to(device)
                except Exception as e_to_device:
                    logger.warning(f"Could not move all parts of the model to {device}: {e_to_device}. Using model.device: {loaded_model.device}")
            
            try:
                loaded_model.generation_config = GenerationConfig.from_pretrained(self.base_model_name, token=self.hf_token)
            except Exception:
                 logger.warning(f"Could not load generation_config for base model {self.base_model_name}. Using default.")
                 # Fallback or default config if needed

            logger.info(f"✅ Successfully loaded base model {self.base_model_name} to device {loaded_model.device}.")
            return loaded_model, loaded_tokenizer
        except Exception as e:
            logger.error(f"❌ Failed to load base model {self.base_model_name}: {e}")
            return None, loaded_tokenizer # Return tokenizer if it loaded, model is None

    def save_checkpoint(self, current_step, batch_results, validation_results=None):
        """Save checkpoint atomically with verification."""
        with self.lock:
            try:
                temp_checkpoint_dir = self.temp_dir / f"checkpoint_{current_step}"
                temp_checkpoint_dir.mkdir(exist_ok=True)
                final_checkpoint_dir = self.checkpoint_dir / f"checkpoint_{current_step}"

                if self.model is None or self.tokenizer is None or self.optimizer is None:
                    logger.error("Model, tokenizer, or optimizer is None. Cannot save checkpoint.")
                    return
                
                logger.info(f"Saving model to {temp_checkpoint_dir}...")
                self.model.save_pretrained(temp_checkpoint_dir, safe_serialization=False, max_shard_size="2GB")
                model_bin_path = temp_checkpoint_dir / "pytorch_model.bin"
                sharded_files = list(temp_checkpoint_dir.glob('pytorch_model-*.bin'))
                if not model_bin_path.exists() and not sharded_files:
                    logger.error(f"CRITICAL: Model file(s) not found after save in {temp_checkpoint_dir}!")
                    # shutil.rmtree(temp_checkpoint_dir) # Clean up failed attempt
                    return # Or raise error
                logger.info("Model save call completed.")

                logger.info(f"Saving tokenizer to {temp_checkpoint_dir}...")
                self.tokenizer.save_pretrained(temp_checkpoint_dir)
                logger.info("Tokenizer save call completed.")
                
                logger.info(f"Saving optimizer state to {temp_checkpoint_dir}...")
                torch.save(self.optimizer.state_dict(), temp_checkpoint_dir / "optimizer.pt")
                
                serializable_results = []
                if batch_results is None: batch_results = []
                for i, result in enumerate(batch_results):
                    try:
                        reward_value = result["reward"].item() if isinstance(result["reward"], torch.Tensor) else result["reward"]
                        serializable_results.append({
                            "idx": result["idx"],
                            "query": result["query"],
                            "ad_facts": result["ad_facts"],
                            "response_without_ad": result["response_without_ad"],
                            "response_with_ad": result["response_with_ad"],
                            "reward": reward_value,
                            "scores": result["scores"],
                        })
                    except Exception as e:
                        logger.warning(f"Error processing result {i} for checkpoint: {str(e)}")
                        continue
                
                metrics = {"avg_reward": 0.0, "avg_coherence": 0.0, "avg_helpfulness": 0.0, "avg_salience": 0.0, "avg_detectability": 0.0}
                if serializable_results:
                    num_results = len(serializable_results)
                    metrics["avg_reward"] = sum(r["reward"] for r in serializable_results) / num_results
                    metrics["avg_coherence"] = sum(r["scores"].get("coherence", {}).get("Coherence Score", 0) for r in serializable_results) / num_results
                    metrics["avg_helpfulness"] = sum(r["scores"].get("helpfulness", {}).get("Helpfulness Score", 0) for r in serializable_results) / num_results
                    metrics["avg_salience"] = sum(r["scores"].get("salience", {}).get("Ad Salience Score", 0) for r in serializable_results) / num_results
                    metrics["avg_detectability"] = sum(r["scores"].get("detectability", {}).get("detectability_cosine", 0) for r in serializable_results) / num_results
                
                checkpoint_info = {"step": current_step, "timestamp": time.time(), "batch_results": serializable_results, "metrics": metrics}
                
                if validation_results:
                    val_rewards = [r["reward"].item() if isinstance(r["reward"], torch.Tensor) else r["reward"] for r in validation_results if "reward" in r]
                    if val_rewards: checkpoint_info["metrics"]["validation_avg_reward"] = sum(val_rewards) / len(val_rewards)
                
                with open(temp_checkpoint_dir / "checkpoint_info.json", "w") as f: json.dump(checkpoint_info, f, indent=2)
                
                try:
                    with open(self.metrics_path, "r") as f: metrics_file = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError): metrics_file = {"checkpoints": [], "best_reward": float('-inf'), "best_checkpoint": None, "training_history": []}

                metrics_file["checkpoints"].append({"step": current_step, "path": str(final_checkpoint_dir), "metrics": checkpoint_info["metrics"]})
                if "validation_avg_reward" in checkpoint_info["metrics"] and checkpoint_info["metrics"]["validation_avg_reward"] > metrics_file.get("best_reward", float('-inf')):
                    metrics_file["best_reward"] = checkpoint_info["metrics"]["validation_avg_reward"]
                    metrics_file["best_checkpoint"] = str(final_checkpoint_dir)
                    logger.info(f"🎉 New best model! Val reward: {metrics_file['best_reward']:.4f}")
                metrics_file["training_history"].append({"step": current_step, "timestamp": time.time(), "metrics": checkpoint_info["metrics"]})
                with open(self.metrics_path, "w") as f: json.dump(metrics_file, f, indent=2)
                
                if final_checkpoint_dir.exists(): shutil.rmtree(final_checkpoint_dir)
                shutil.move(str(temp_checkpoint_dir), str(final_checkpoint_dir))
                logger.info(f"✅ Checkpoint {current_step} saved to {final_checkpoint_dir}")
                
                with open(self.checkpoint_info_path, "w") as f:
                    json.dump({"latest_checkpoint": str(final_checkpoint_dir)}, f)
                
            except Exception as e:
                logger.error(f"❌ Error saving checkpoint {current_step}: {e}")
                logger.exception("Detailed traceback for checkpoint saving error:")
                if temp_checkpoint_dir.exists(): shutil.rmtree(temp_checkpoint_dir) # Clean up
            finally:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    
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
        with self.lock:
            if not self.checkpoint_info_path.exists():
                logger.info("No checkpoint_info.json found. Attempting to load base model.")
                self._model, self._tokenizer = self._load_base_model_and_tokenizer()
                if self._model is None or self._tokenizer is None:
                    logger.error("Failed to load base model and tokenizer. Cannot proceed.")
                    return None, None, None
                # Optimizer is typically re-initialized by the caller (e.g., run_manual_ppo) 
                # when loading the base model, so no explicit reset needed here for self._optimizer.
                return self._model, self._tokenizer, None # No checkpoint info for base model load
            
            with open(self.checkpoint_info_path, "r") as f:
                info = json.load(f)
            latest_checkpoint_path = Path(info.get("latest_checkpoint"))

            if not latest_checkpoint_path or not latest_checkpoint_path.exists() or not self._verify_checkpoint(latest_checkpoint_path):
                logger.warning(f"Latest checkpoint {latest_checkpoint_path} is invalid or not found. Attempting to load base model.")
                self._model, self._tokenizer = self._load_base_model_and_tokenizer()
                if self._model is None or self._tokenizer is None:
                    logger.error("Failed to load base model and tokenizer after invalid checkpoint. Cannot proceed.")
                    return None, None, None
                # Optimizer is typically re-initialized by the caller (e.g., run_manual_ppo) 
                # when loading the base model, so no explicit reset needed here for self._optimizer.
                return self._model, self._tokenizer, None
            
            try:
                logger.info(f"Loading model from checkpoint: {latest_checkpoint_path}")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # If self._model was None (e.g. from main.py initial load failure), instantiate it first from base.
                # This scenario is less likely now that run_manual_ppo also tries to load the model.
                # However, to be robust, if _model is None, we load the base model before applying checkpoint.
                if self._model is None:
                    logger.warning("self._model is None before loading checkpoint. Initializing from base model first.")
                    base_m, base_t = self._load_base_model_and_tokenizer()
                    if base_m is None or base_t is None:
                        logger.error("Could not load base model to serve as structure for checkpoint loading.")
                        return None, None, None
                    self._model = base_m
                    self._tokenizer = base_t # Tokenizer might be overwritten by checkpoint's tokenizer anyway

                # Load checkpoint state into the existing model object if it exists, or the newly loaded base model.
                # Transformers from_pretrained can also load a checkpoint into an existing model if paths are set up, 
                # but direct state_dict loading is also an option if we ensure model architecture matches.
                # For simplicity with AutoModel, loading the checkpoint as a new instance is often safer.
                loaded_cpt_model = AutoModelForCausalLM.from_pretrained(
                    latest_checkpoint_path, 
                    torch_dtype=torch.float16,
                    device_map="auto", 
                    trust_remote_code=True, 
                    token=self.hf_token # Use token if checkpoint refers to external files that might be gated
                )
                self._model = loaded_cpt_model # Replace the model instance
                if self._model.device != device and device.type == 'cuda': self._model.to(device)

                logger.info(f"Loading tokenizer from checkpoint: {latest_checkpoint_path}")
                loaded_cpt_tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint_path, trust_remote_code=True, token=self.hf_token)
                self._tokenizer = loaded_cpt_tokenizer # Replace tokenizer instance

                if self._optimizer:
                    logger.info(f"Loading optimizer state from checkpoint: {latest_checkpoint_path}")
                    self._optimizer.load_state_dict(torch.load(latest_checkpoint_path / "optimizer.pt", map_location=device))
                
                with open(latest_checkpoint_path / "checkpoint_info.json", "r") as f:
                    checkpoint_specific_info = json.load(f)
                
                logger.info(f"✅ Checkpoint loaded successfully from {latest_checkpoint_path}")
                return self._model, self._tokenizer, checkpoint_specific_info

            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint from {latest_checkpoint_path}: {e}")
                logger.exception("Detailed traceback for checkpoint loading error:")
                logger.warning("Falling back to loading base model due to checkpoint load failure.")
                self._model, self._tokenizer = self._load_base_model_and_tokenizer()
                if self._model is None or self._tokenizer is None:
                    logger.error("Failed to load base model as fallback. Cannot proceed.")
                    return None, None, None
                # Optimizer is typically re-initialized by the caller (e.g., run_manual_ppo) 
                # when loading the base model, so no explicit reset needed here for self._optimizer.
                return self._model, self._tokenizer, None
    
    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Clean up old checkpoints, keeping only the last N."""
        with self.lock:
            try:
                with open(self.metrics_path, "r") as f:
                    metrics_file = json.load(f)
                
                all_checkpoints = sorted(
                    metrics_file.get("checkpoints", []),
                    key=lambda cp: cp["step"],
                    reverse=True
                )

                checkpoints_to_keep = set()
                if metrics_file.get("best_checkpoint"):
                    checkpoints_to_keep.add(Path(metrics_file["best_checkpoint"])) # Keep best checkpoint
                
                # Keep the last N checkpoints by step
                for cp_info in all_checkpoints[:keep_last_n]:
                    checkpoints_to_keep.add(Path(cp_info["path"])) 
                
                # Iterate through checkpoint directories and delete old ones
                for checkpoint_dir in self.checkpoint_dir.iterdir():
                    if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("checkpoint_") and checkpoint_dir not in checkpoints_to_keep:
                        logger.info(f"🗑️ Deleting old checkpoint: {checkpoint_dir}")
                        shutil.rmtree(checkpoint_dir)
                
                # Update checkpoint_info.json if latest_checkpoint was deleted
                if self.checkpoint_info_path.exists():
                    with open(self.checkpoint_info_path, "r") as f:
                        current_info = json.load(f)
                    latest_cpt_path = Path(current_info.get("latest_checkpoint", ""))
                    if latest_cpt_path not in checkpoints_to_keep and checkpoints_to_keep:
                        # Set to the most recent kept checkpoint
                        new_latest = max(checkpoints_to_keep, key=lambda p: int(p.name.split('_')[-1]))
                        with open(self.checkpoint_info_path, "w") as f:
                            json.dump({"latest_checkpoint": str(new_latest)}, f)
                        logger.info(f"Updated latest_checkpoint to {new_latest} after cleanup.")
                    elif not checkpoints_to_keep and latest_cpt_path.exists(): # All checkpoints deleted somehow
                        os.remove(self.checkpoint_info_path)
                        logger.info("Removed checkpoint_info.json as all checkpoints were deleted.")
                
            except Exception as e:
                logger.error(f"Error during checkpoint cleanup: {e}")
