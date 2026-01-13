import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead


class CheckpointManager:
    """
    PPO-safe checkpoint manager.

    Key difference vs your version:
    - ALWAYS loads AutoModelForCausalLMWithValueHead so v_head is preserved.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        base_model_name: str,
        hf_token: str,
        model=None,
        tokenizer=None,
        optimizer=None,
        unwrap_model_fn: Optional[Callable[[Any], Any]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.base_model_name = base_model_name
        self.hf_token = hf_token

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer

        # If you're using accelerate, pass: trainer.accelerator.unwrap_model
        self.unwrap_model_fn = unwrap_model_fn

    # ---------------------------
    # Helpers
    # ---------------------------

    @staticmethod
    def _parse_step(ckpt_path: Path) -> int:
        # checkpoint-123 -> 123
        try:
            return int(ckpt_path.name.split("-")[-1])
        except Exception:
            return -1

    @staticmethod
    def _looks_like_model_dir(ckpt: Path) -> bool:
        # Covers: sharded safetensors, single safetensors, pytorch_model.bin
        return (
            (ckpt / "model.safetensors").exists()
            or (ckpt / "model.safetensors.index.json").exists()
            or (ckpt / "pytorch_model.bin").exists()
            or (ckpt / "pytorch_model.bin.index.json").exists()
            or any(ckpt.glob("model-*.safetensors"))
        )

    @staticmethod
    def _looks_like_tokenizer_dir(ckpt: Path) -> bool:
        return (ckpt / "tokenizer_config.json").exists() or (ckpt / "tokenizer.json").exists()

    def _load_valuehead_model(self, path: str | Path):
        # You can add torch_dtype/device_map here if you want,
        # but keep it minimal/stable for resume correctness.
        return AutoModelForCausalLMWithValueHead.from_pretrained(
            path,
            trust_remote_code=True,
            token=self.hf_token,
        )

    def _load_tokenizer(self, path: str | Path):
        tok = AutoTokenizer.from_pretrained(path, token=self.hf_token, use_fast=True, trust_remote_code=True)

        # Enforce consistent padding behavior (matches your PPODataCollator assumptions)
        tok.padding_side = "left"
        tok.truncation_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
        return tok

    # ---------------------------
    # Public API
    # ---------------------------

    def find_latest_checkpoint(self) -> Optional[Path]:
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=self._parse_step,
            reverse=True,
        )

        for ckpt in checkpoints:
            if self._looks_like_model_dir(ckpt) and self._looks_like_tokenizer_dir(ckpt):
                return ckpt

        return None

    def load_latest_checkpoint(self) -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
        ckpt = self.find_latest_checkpoint()

        if ckpt is not None:
            step = self._parse_step(ckpt)
            model = self._load_valuehead_model(ckpt)
            tokenizer = self._load_tokenizer(ckpt)
            return model, tokenizer, {"step": step, "latest_checkpoint": str(ckpt)}

        # No checkpoint found -> base model
        model = self._load_valuehead_model(self.base_model_name)
        tokenizer = self._load_tokenizer(self.base_model_name)
        return model, tokenizer, None

    def save_checkpoint(self, step: int) -> Path:
        """
        Saves model/tokenizer/optimizer into checkpoint-{step}.
        If accelerate is used, pass unwrap_model_fn to save the real underlying model.
        """
        ckpt_path = self.checkpoint_dir / f"checkpoint-{int(step)}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model
        if self.unwrap_model_fn is not None and model_to_save is not None:
            model_to_save = self.unwrap_model_fn(model_to_save)

        if model_to_save is None:
            raise ValueError("CheckpointManager.save_checkpoint: self.model is None")
        if self.tokenizer is None:
            raise ValueError("CheckpointManager.save_checkpoint: self.tokenizer is None")

        # Save policy WITH value head
        model_to_save.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)

        # Save optimizer
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), ckpt_path / "optimizer.pt")

        # Save resume metadata
        meta = {"step": int(step), "checkpoint": str(ckpt_path)}
        with open(self.checkpoint_dir / "last_position.json", "w") as f:
            json.dump(meta, f)

        return ckpt_path