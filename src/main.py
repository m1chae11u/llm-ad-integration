# src/main.py
import os
import sys
import pandas as pd
from pathlib import Path

from config import BASE_MODEL, HF_TOKEN, DATA_FILE
from training.ppo_training import make_trainer, TRAINING_LOGS, run_ppo
from transformers.trainer_utils import get_last_checkpoint

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

    # 3) Run PPO, auto-resume from last checkpoint if present
    output_dir = Path(trainer.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        last_ckpt = get_last_checkpoint(output_dir)
    except (FileNotFoundError, ValueError):
        last_ckpt = None
    resume_arg = last_ckpt
    # 3) Run PPO via manual loop with resume support and catch unsupported resume errors
    if resume_arg:
        print(f"⏯ Resuming PPO from checkpoint {resume_arg}")
    try:
        trainer.ppo_train(resume_from_checkpoint=resume_arg)
    except ValueError as e:
        if "will be supported in the future version" in str(e):
            print(f"⚠️ {e}. Running PPO without resume.")
            trainer.ppo_train()
        else:
            raise
    except KeyboardInterrupt:
        print("⚠️ Training interrupted. Saving checkpoint before exit…")
        trainer.save_model()
        try:
            model_dir = output_dir / "interrupted_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            unwrapped_model.save_pretrained(model_dir)
            trainer.tokenizer.save_pretrained(model_dir)
            print(f"⚠️ Saved interrupted model and tokenizer to {model_dir}")
        except Exception:
            pass
    else:
        trainer.save_model()
        try:
            trainer.save_state()
        except Exception:
            pass
        model_dir = output_dir / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped_model.save_pretrained(model_dir)
        trainer.tokenizer.save_pretrained(model_dir)
        print(f"✅ Saved final model and tokenizer to {model_dir}")

    # 4) Dump your judging logs in output_dir
    output_dir = Path(trainer.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "ppo_judging_log.csv"
    pd.DataFrame(TRAINING_LOGS).to_csv(log_path, index=False)
    print(f"✅ Saved PPO judging logs to {log_path}")