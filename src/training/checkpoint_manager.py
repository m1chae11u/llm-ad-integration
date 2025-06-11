import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json

class CheckpointManager:
    def __init__(self, checkpoint_dir, model, tokenizer, optimizer, base_model_name, hf_token):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.base_model_name = base_model_name
        self.hf_token = hf_token

    def load_latest_checkpoint(self):
        # Find all checkpoint-* directories sorted by step number (descending)
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split('-')[-1]),
            reverse=True
        )

        for ckpt in checkpoints:
            model_path = ckpt / "pytorch_model.bin"
            tokenizer_path = ckpt / "tokenizer_config.json"
            if model_path.exists() and tokenizer_path.exists():
                model = AutoModelForCausalLM.from_pretrained(ckpt, use_auth_token=self.hf_token)
                tokenizer = AutoTokenizer.from_pretrained(ckpt, use_auth_token=self.hf_token)
                step = int(ckpt.name.split('-')[-1])
                return model, tokenizer, {
                    "step": step,
                    "latest_checkpoint": str(ckpt)
                }

        # No checkpoint found, load base model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name, use_auth_token=self.hf_token)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_auth_token=self.hf_token)
        return model, tokenizer, None

    def save_checkpoint(self, step):
        ckpt_path = self.checkpoint_dir / f"checkpoint-{step}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        
        # Save optimizer state if present
        if self.optimizer:
            torch.save(self.optimizer.state_dict(), ckpt_path / "optimizer.pt")

        # Optional: Save resume metadata like current query position
        with open(self.checkpoint_dir / "last_query_position.json", "w") as f:
            json.dump({"query_index": step}, f)