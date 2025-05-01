# main.py
from generate.generator import generate_responses
from judge.coherence import judge_coherence
from judge.helpfulness import judge_helpfulness
from judge.salience import judge_ad_salience
from judge.detectability import judge_detectability
import pandas as pd
import os
from tqdm import tqdm
import torch
import time

# Set batch size for processing
BATCH_SIZE = 4  # Process 4 queries at a time

# --- Load data ---
print("Loading data...")
df = pd.read_csv("data/merged_queries_ads.csv") 
print(f"Loaded {len(df)} rows of data")
print("\nAvailable columns:", df.columns.tolist())  # Print available columns

checkpoint_file = "checkpoints/judge_results.csv"
checkpoint_interval = 5

if os.path.exists(checkpoint_file):
    results = pd.read_csv(checkpoint_file).to_dict(orient="records")
    processed_ids = {int(r["ad_index"]) for r in results}  # Changed from Query ID to ad_index
    print(f"Found {len(processed_ids)} previously processed queries")
else:
    results = []
    processed_ids = set()
    print("No previous checkpoint found, starting from scratch")

# --- Main Loop ---
print("\nStarting processing...")
total_queries = len(df)
processed_count = 0

# Process in batches
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing batches"):
    batch = df.iloc[i:i+BATCH_SIZE]
    batch_results = []
    
    for _, row in batch.iterrows():
        query_id = int(row["ad_index"])  # Changed from Query ID to ad_index
        if query_id in processed_ids:
            continue
            
        try:
            print(f"\nProcessing query {query_id} ({processed_count + 1}/{total_queries})")
            
            # Format ad facts
            ad_facts = {
                'ad_product': row['ad_product'],
                'brand': row['brand'],
                'url': row['url'],
                'description': row['ad_description']
            }
            
            # Generate responses
            print("Starting response generation...")
            start_time = time.time()
            response_without_ad, response_with_ad = generate_responses(row['vague_query'], ad_facts)
            generation_time = time.time() - start_time
            print(f"Generation completed in {generation_time:.2f} seconds")
            
            # Run judgments
            print("Running judgments...")
            coherence = judge_coherence(response_with_ad, row['vague_query'])
            helpfulness = judge_helpfulness(row['vague_query'], response_with_ad)
            ad_salience = judge_ad_salience(row['vague_query'], response_with_ad, ad_facts)
            detectability = judge_detectability(response_with_ad, response_without_ad)
            print("Judgments completed")
            
            # Collect results
            result = {
                "ad_index": query_id,  
                "User Query": row['vague_query'],  
                "Response Without Ad": response_without_ad,
                "Response With Ad": response_with_ad,
                "Coherence": coherence,
                "Helpfulness": helpfulness,
                "Ad Salience": ad_salience,
                "Detectability": detectability
            }
            
            batch_results.append(result)
            processed_ids.add(query_id)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing query {query_id}: {str(e)}")
            continue
    
    # Save batch results
    if batch_results:
        results.extend(batch_results)
        if len(results) % checkpoint_interval == 0:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            print(f"Saved checkpoint with {len(results)} results")

# Final save
pd.DataFrame(results).to_csv(checkpoint_file, index=False)
print(f"\nProcessing complete. Total results: {len(results)}")