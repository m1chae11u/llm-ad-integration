# main.py

from generate.generator import generate_responses
from judge.coherence import judge_coherence
from judge.helpfulness import judge_helpfulness
from judge.salience import judge_ad_salience
from judge.detectability import judge_detectability
import pandas as pd
import os
from tqdm import tqdm

# --- Load your data ---
print("Loading data...")
df = pd.read_csv("data/merged_queries_ads.csv") 
print(f"Loaded {len(df)} rows of data")

checkpoint_file = "checkpoints/judge_results.csv"
checkpoint_interval = 10

if os.path.exists(checkpoint_file):
    results = pd.read_csv(checkpoint_file).to_dict(orient="records")
    processed_ids = {int(r["Query ID"]) for r in results}
    print(f"Found {len(processed_ids)} previously processed queries")
else:
    results = []
    processed_ids = set()
    print("No previous checkpoint found, starting from scratch")

# --- Main Loop ---
print("\nStarting processing...")
for idx, row in enumerate(tqdm(df.to_dict(orient="records"))):
    query_id = int(row["ad_index"])
    if query_id in processed_ids:
        continue

    query = row["vague_query"]
    ad_facts = {
        "ad_product": row.get("ad_product"),
        "brand": row.get("brand"),
        "url": row.get("url"),
        "description": row.get("ad_description")
    }

    try:
        print(f"\nProcessing Query ID {query_id}:")
        print(f"Query: {query[:100]}...")
        print(f"Ad Product: {ad_facts['ad_product']}")

        print("Generating responses...")
        res_wo_ad, res_with_ad = generate_responses(query, ad_facts)
        print(f"Response without ad: {res_wo_ad[:100]}...")
        print(f"Response with ad: {res_with_ad[:100]}...")

        print("Running judges...")
        coh = judge_coherence(res_with_ad, query)
        help_score = judge_helpfulness(query, res_with_ad)
        sal = judge_ad_salience(query, res_with_ad, ad_facts)
        detect = judge_detectability(res_with_ad, res_wo_ad)

        print("\nResults:")
        print(f"Coherence: {coh}")
        print(f"Helpfulness: {help_score}")
        print(f"Ad Salience: {sal}")
        print(f"Detectability: {detect}")

        results.append({
            "Query ID": query_id,
            "User Query": query,
            "Ad Product": ad_facts["ad_product"],
            "Brand": ad_facts["brand"],
            "URL": ad_facts["url"],
            "Description": ad_facts["description"],
            "Response Without Ad": res_wo_ad,
            "Response With Ad": res_with_ad,
            "Coherence": coh,
            "Helpfulness": help_score,
            "Ad Salience": sal,
            "Detectability": detect.get("detectability"),
            "Similarity": detect.get("similarity")
        })

        if len(results) % checkpoint_interval == 0:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            print(f"\nSaved {len(results)} results to checkpoint.")

    except Exception as e:
        print(f"\nError on Query ID {query_id}: {e}")
        continue

pd.DataFrame(results).to_csv(checkpoint_file, index=False)
print("\nDone. Final results saved to:", checkpoint_file)