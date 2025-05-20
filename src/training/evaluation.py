import os
import pandas as pd
from judge.coherence import judge_coherence
from judge.helpfulness import judge_helpfulness
from judge.salience import judge_ad_salience

def evaluate_logged_responses(logged_csv_path, output_path="checkpoints/eval_logged_results.csv"):
    print("\n📊 Evaluating saved responses from log...")
    df = pd.read_csv(logged_csv_path)
    comparisons = []

    for idx, row in df.iterrows():
        user_query = row["query"]
        base_resp = row["Base Response"]
        ppo_resp = row["PPO Response"]

        # Optional: ad info if needed for salience
        ad_facts = {
            "ad_product": row.get("ad_product", ""),
            "brand": row.get("brand", ""),
            "url": row.get("url", ""),
            "description": row.get("ad_description", "")
        }

        # Judge base
        base_coh = judge_coherence(base_resp, user_query)
        base_help = judge_helpfulness(user_query, base_resp)
        base_sal = judge_ad_salience(user_query, base_resp, ad_facts)
        base_total = base_coh["Coherence Score"] + base_help["Helpfulness Score"] + base_sal["Ad Salience Score"]

        # Judge PPO
        ppo_coh = judge_coherence(ppo_resp, user_query)
        ppo_help = judge_helpfulness(user_query, ppo_resp)
        ppo_sal = judge_ad_salience(user_query, ppo_resp, ad_facts)
        ppo_total = ppo_coh["Coherence Score"] + ppo_help["Helpfulness Score"] + ppo_sal["Ad Salience Score"]

        comparisons.append({
            "Query": user_query,
            "Base Response": base_resp,
            "PPO Response": ppo_resp,

            # Base model subscores
            "Base C1": base_coh["C1"], "Base C2": base_coh["C2"], "Base C3": base_coh["C3"], "Base C4": base_coh["C4"],
            "Base H1": base_help["H1"],
            "Base S1": base_sal["S1"], "Base S2": base_sal["S2"], "Base S3": base_sal["S3"],
            "Base Detectability": base_det.get("detectability_cosine", 0),
            "Base Coherence Explanation": base_coh.get("Coherence Explanation", ""),
            "Base Helpfulness Explanation": base_help.get("Helpfulness Explanation", ""),
            "Base Salience Explanation": base_sal.get("Ad Salience Explanation", ""),
            "Base Total": base_total,

            # PPO model subscores
            "PPO C1": ppo_coh["C1"], "PPO C2": ppo_coh["C2"], "PPO C3": ppo_coh["C3"], "PPO C4": ppo_coh["C4"],
            "PPO H1": ppo_help["H1"],
            "PPO S1": ppo_sal["S1"], "PPO S2": ppo_sal["S2"], "PPO S3": ppo_sal["S3"],
            "PPO Detectability": ppo_det.get("detectability_cosine", 0),
            "PPO Coherence Explanation": ppo_coh.get("Coherence Explanation", ""),
            "PPO Helpfulness Explanation": ppo_help.get("Helpfulness Explanation", ""),
            "PPO Salience Explanation": ppo_sal.get("Ad Salience Explanation", ""),
            "PPO Total": ppo_total,

            "Winner": "PPO" if ppo_total > base_total else "Base" if base_total > ppo_total else "Tie"
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(comparisons).to_csv(output_path, index=False)
    print(f"✅ Evaluation complete. Saved to {output_path}")