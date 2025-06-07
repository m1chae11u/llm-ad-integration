# src/main.py
import os
import sys
import pandas as pd
from pathlib import Path

from config import BASE_MODEL, HF_TOKEN, DATA_FILE
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

    # 2.a) Prepare for checkpoint and logs resumption
    output_dir = Path(trainer.training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "ppo_judging_log.csv"
    if log_path.exists():
        print(f"⏯ Resuming logs from {log_path}")
        existing_logs = pd.read_csv(log_path).to_dict("records")
        TRAINING_LOGS.extend(existing_logs)
    # Detect last checkpoint directory
    ckpt_dirs = list(output_dir.glob("checkpoint-*"))
    if ckpt_dirs:
        last_ckpt = sorted(ckpt_dirs, key=lambda x: int(x.name.split('-')[-1]))[-1]
        print(f"⏯ Resuming PPO from checkpoint {last_ckpt}")
        trainer.training_args.resume_from_checkpoint = str(last_ckpt)
    else:
        trainer.training_args.resume_from_checkpoint = None
        print("⏺ No checkpoint found, starting fresh PPO training")

    # 3) Run PPO training loop
    print("⏱ Starting PPO training...")
    try:
        trainer.ppo_train(resume_from_checkpoint=trainer.training_args.resume_from_checkpoint)
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
    output_dir = Path(trainer.training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "ppo_judging_log.csv"
    pd.DataFrame(TRAINING_LOGS).to_csv(log_path, index=False)
    print(f"✅ Saved PPO judging logs to {log_path}")