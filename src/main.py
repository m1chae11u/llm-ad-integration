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
    ad_facts_list = df[["ad_product", "brand", "url", "ad_description"]].to_dict("records")

    # 2) Build the trainer
    trainer = make_trainer(
        model_name=BASE_MODEL,
        hf_token=HF_TOKEN,
        data_path=DATA_FILE,
        ad_facts_list=ad_facts_list,
    )

    # 3) Run PPO, catch Ctrl+C to checkpoint
    try:
        trainer.ppo_train()
        trainer.save_model()
    except KeyboardInterrupt:
        print("⚠️ Training interrupted. Saving checkpoint before exit…")
        trainer.save_model()
        sys.exit(0)

    # 4) Dump your judging logs
    result_dir = Path(__file__).resolve().parents[1] / "training_result"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "ppo_judging_log.csv"
    pd.DataFrame(TRAINING_LOGS).to_csv(log_path, index=False)
    print(f"✅ Saved PPO judging logs to {log_path}")