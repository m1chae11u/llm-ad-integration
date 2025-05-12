import pandas as pd

# Load your two source CSVs
queries_df = pd.read_json("data/user_queries.json")  
ads_df = pd.read_csv("data/sampled_ads.csv")       

# Check unique counts in both dataframes
print("\nUnique ad_ids in queries:", len(queries_df['ad_id'].unique()))
print("Total rows in queries:", len(queries_df))

print("\nUnique ad_ids in ads:", len(ads_df['ad_id'].unique()))
print("Total rows in ads:", len(ads_df))

# First, let's see how many unique ad_id + ad_product combinations we have in queries
print("\nUnique combinations in queries:")
print(queries_df.groupby(['ad_id', 'ad_product']).size().reset_index(name='count'))

# Merge on both ad_id and ad_product/ad_title
merged_df = queries_df.merge(
    ads_df,
    on="ad_id",
    how="inner"
)

# Inspect a few rows
print("\nFirst few rows of merged data:")
print(merged_df.head())

# Save the merged file
merged_df.to_csv("data/merged_queries_ads.csv", index=False)
print(f"\nMerged {len(merged_df)} rows to data/merged_queries_ads.csv")