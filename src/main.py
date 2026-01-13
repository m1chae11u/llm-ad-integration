import os
import sys
import json
from pathlib import Path

import pandas as pd
import torch

from config import BASE_MODEL, HF_TOKEN, DATA_FILE
from training.checkpoint_manager import CheckpointManager
from training.ppo_training import make_trainer, TRAINING_LOGS, MyPPOTrainer, logger


def _load_ad_facts(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["ad_id", "ad_product", "brand", "url", "ad_description"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    if len(df) == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    return df[required_columns].to_dict("records")


def _rebind_model_and_optimizer_for_accelerate(trainer, model, tokenizer, ckpt_dir: Path | None):
    """
    Make a checkpoint-loaded model usable inside an already-constructed Accelerate trainer,
    WITHOUT causing an extra model.to(cuda) pass (which is what OOMs).

    Steps:
    - attach model + tokenizer
    - ensure forward patch on the underlying model
    - (optionally) move model to accelerator.device only if needed
    - rebuild optimizer on new params
    - load optimizer state (optional, CPU load)
    - prepare optimizer with accelerate WITHOUT moving model again
    """

    if model is None or tokenizer is None:
        print("No checkpoint model/tokenizer provided; skipping rebind.")
        return
    # ----------------------------
    # 1) Attach checkpoint model + tokenizer
    # ----------------------------
    trainer.model = model
    trainer._custom_tokenizer = tokenizer

    # Keep this pointing to the vhead policy object that should be restored if LF overwrites
    trainer._policy_with_vhead = model

    # ----------------------------
    # 2) Reapply forward patch early (checkpoint load can drop monkeypatches)
    #    IMPORTANT: patch the *underlying* model object
    # ----------------------------
    trainer.model = MyPPOTrainer.ensure_llamafactory_tuple_output(trainer.model, logger)

    # ----------------------------
    # 3) Ensure model is on the right device (ONLY if needed)
    #    Do NOT rely on accelerate.prepare(model, ...) to move it (that can OOM).
    # ----------------------------
    target_device = getattr(trainer.accelerator, "device", torch.device("cpu"))
    try:
        base = getattr(trainer.model, "pretrained_model", trainer.model)
        cur_device = next(base.parameters()).device
    except StopIteration:
        cur_device = torch.device("cpu")

    if cur_device != target_device:
        if target_device.type == "cuda":
            # Reduce fragmentation before a big move
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        trainer.model.to(target_device)

    # ----------------------------
    # 4) Rebuild optimizer bound to *this* model's params
    # ----------------------------
    lr = getattr(trainer.args, "learning_rate", 5e-5) if hasattr(trainer, "args") else 5e-5
    wd = getattr(trainer.args, "weight_decay", 0.0) if hasattr(trainer, "args") else 0.0
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr, weight_decay=wd)

    # ----------------------------
    # 5) Load optimizer state if available (CPU load first)
    # ----------------------------
    if ckpt_dir is not None:
        opt_path = ckpt_dir / "optimizer.pt"
        if opt_path.exists():
            state = torch.load(opt_path, map_location="cpu")
            trainer.optimizer.load_state_dict(state)
            print(f"✅ Loaded optimizer state from {opt_path}")
        else:
            print(f"⚠️ No optimizer.pt found at {opt_path} (continuing with fresh optimizer)")

    # ----------------------------
    # 6) Prepare optimizer with Accelerate WITHOUT moving model again
    #    This is the critical part that avoids your OOM.
    # ----------------------------
    already_prepared = getattr(trainer, "_already_prepared", False)

    if already_prepared:
        # Trainer/model are already wrapped; just prepare optimizer alone.
        trainer.optimizer = trainer.accelerator.prepare(trainer.optimizer)
    else:
        # Prepare both, but DO NOT device-place the model (prevents model.to(cuda) inside accelerate)
        trainer.model, trainer.optimizer = trainer.accelerator.prepare(
            trainer.model,
            trainer.optimizer,
            device_placement=[False, True],
        )
        trainer._already_prepared = True

    # ----------------------------
    # 7) Final sanity prints (use unwrapped)
    # ----------------------------
    u = trainer.accelerator.unwrap_model(trainer.model)
    base_u = getattr(u, "pretrained_model", u)
    has_v = hasattr(u, "v_head") and (u.v_head is not None)

    print(f"🔎 Resume sanity: has_v_head={has_v}")
    print(f"🔎 Resume sanity: base_device={next(base_u.parameters()).device}")
    if has_v:
        print(f"🔎 Resume sanity: v_head_device={next(u.v_head.parameters()).device}")

def main():
    # 1) Load ad facts
    try:
        ad_facts_list = _load_ad_facts(DATA_FILE)
        print(f"✅ Loaded {len(ad_facts_list)} ad facts from {DATA_FILE}")
    except Exception as e:
        print(f"❌ Failed to load data file: {e}")
        sys.exit(1)

    # 2) Build trainer (fresh objects)
    trainer = make_trainer(
        model_name=BASE_MODEL,
        hf_token=HF_TOKEN,
        data_path=DATA_FILE,
        ad_facts_list=ad_facts_list,
    )

    output_dir = Path(trainer.training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) Checkpoint manager (use it to locate & load latest checkpoint)
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=output_dir,
        model=None,         # ckpt manager returns a model; we bind it properly below
        tokenizer=None,     # same
        optimizer=None,     # do not pass trainer.optimizer here; we rebuild after model load
        base_model_name=BASE_MODEL,
        hf_token=HF_TOKEN,
    )

    model, tokenizer, ckpt_info = ckpt_mgr.load_latest_checkpoint()

    if ckpt_info:
        ckpt_dir = Path(ckpt_info["latest_checkpoint"])
        resume_step = int(ckpt_info.get("step", 0))
        print(f"⏯ Resuming from checkpoint: {ckpt_dir} (step={resume_step})")

        _rebind_model_and_optimizer_for_accelerate(trainer, model, tokenizer, ckpt_dir)
    else:
        ckpt_dir = None
        resume_step = 0
        print("⏺ No valid checkpoint found, starting fresh PPO training")

    # 5) Run PPO training
    print("⏱ Starting PPO training...")
    try:
        trainer.ppo_train(
            resume_from_checkpoint=str(ckpt_dir) if ckpt_dir else None,
            start_step=resume_step,
            start_query_idx=0,  # query_idx is not deterministic with shuffle=True; keep 0
        )
    except KeyboardInterrupt:
        print("⚠️ Training interrupted. Saving checkpoint before exit…")
        try:
            current_step = getattr(trainer, "_current_step", 0) or getattr(trainer, "total_steps", 0)
            ckpt_out = output_dir / f"checkpoint-{current_step}"
            ckpt_out.mkdir(parents=True, exist_ok=True)

            # Save model + tokenizer
            trainer.model.save_pretrained(ckpt_out)
            if getattr(trainer, "_custom_tokenizer", None) is not None:
                trainer._custom_tokenizer.save_pretrained(ckpt_out)

            # Save optimizer
            if getattr(trainer, "optimizer", None) is not None:
                torch.save(trainer.optimizer.state_dict(), ckpt_out / "optimizer.pt")

            # Save step marker
            with open(output_dir / "last_query_position.json", "w") as f:
                json.dump({"query_index": int(current_step)}, f)

            print(f"✅ Checkpoint saved to {ckpt_out}")
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(0)
    except Exception as e:
        print(f"❌ PPO training crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    else:
        # Save at end
        try:
            trainer.save_model()
        except Exception:
            pass
        try:
            trainer.save_state()
        except Exception:
            pass
        print("🏁 PPO training complete")

    # 6) Dump judging logs
    log_path = output_dir / "ppo_judging_log.csv"
    pd.DataFrame(TRAINING_LOGS).to_csv(log_path, index=False)
    print(f"✅ Saved PPO judging logs to {log_path}")


if __name__ == "__main__":
    main()