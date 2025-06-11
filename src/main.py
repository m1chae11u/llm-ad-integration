# src/main.py
import os
import sys
import pandas as pd
from pathlib import Path
import torch
import json

from config import BASE_MODEL, HF_TOKEN, DATA_FILE
from training.checkpoint_manager import CheckpointManager
from training.ppo_training import make_trainer, TRAINING_LOGS

if __name__ == "__main__":
    # 1) Read your CSV of queries + ad facts
    df = pd.read_csv(DATA_FILE)
    ad_facts_list = df[["ad_id", "ad_product", "brand", "url", "ad_description"]].to_dict("records")

    # 2) Build the trainer
    trainer = make_trainer(
        model_name=BASE_MODEL,
        hf_token=HF_TOKEN,
        data_path=DATA_FILE,
        ad_facts_list=ad_facts_list,
    )

    # 2.a) Manually load base-or-latest
    output_dir = Path(trainer.training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=output_dir,
        model=None,
        tokenizer=None,
        optimizer=trainer.optimizer,
        base_model_name=BASE_MODEL,
        hf_token=HF_TOKEN,
    )
    model, tokenizer, ckpt_info = ckpt_mgr.load_latest_checkpoint()
    trainer.model = model
    trainer.tokenizer = tokenizer

    if ckpt_info:
        print(f"⏯ Resuming from checkpoint at step {ckpt_info['step']}")
        opt_path = Path(ckpt_info["latest_checkpoint"]) / "optimizer.pt"
        if opt_path.exists():
            trainer.optimizer.load_state_dict(torch.load(opt_path))
        resume_step = ckpt_info["step"]
        last_pos_path = output_dir / "last_query_position.json"
        if last_pos_path.exists():
            with open(last_pos_path) as f:
                resume_query_idx = json.load(f)["query_index"]
        else:
            resume_query_idx = 0
    else:
        print("⏺ No valid checkpoint found, starting fresh PPO training")
        resume_step = 0
        resume_query_idx = 0

    # 3) Run PPO training loop
    print("⏱ Starting PPO training...")
    try:
        trainer.ppo_train()
    except ValueError as e:
        if "will be supported in the future version" in str(e):
            print(f"⚠️ {e}. Running PPO without resume.")
            trainer.ppo_train()
        else:
            raise
    except KeyboardInterrupt:
        print("⚠️ Training interrupted. Saving checkpoint before exit…")
        trainer.save_model()
        sys.exit(0)
    else:
        trainer.save_model()
        try:
            trainer.save_state()
        except Exception:
            pass
        print("🏁 PPO training complete")

    # 4) Dump your judging logs in output_dir
    log_path = Path(trainer.training_args.output_dir) / "ppo_judging_log.csv"
    pd.DataFrame(TRAINING_LOGS).to_csv(log_path, index=False)
    print(f"✅ Saved PPO judging logs to {log_path}")