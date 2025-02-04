# src/merge_business_info.py

import pandas as pd

def merge_business_metadata(
    business_json_path="data/raw/yelp_academic_dataset_business.json",
    aspect_csv="data/processed/yelp_aspect_agg_business.csv",
    output_csv="data/processed/yelp_aspect_biz_merged.csv"
):
    """
    1. Reads the Yelp business JSON
    2. Reads aggregated aspect CSV (one row per (business_id, aspect))
    3. Merges them on 'business_id' to attach 'name', 'city', 'state', 'stars', 'review_count', etc.
    4. Writes final merged dataset
    """
    print(f"Loading business data from {business_json_path}...")
    biz_df = pd.read_json(business_json_path, lines=True)
    
    # Include columns you want to display:
    # e.g. name, city, state, stars, review_count, etc.
    keep_cols = [
        "business_id", 
        "name", 
        "city", 
        "state", 
        "stars",        # official Yelp average
        "review_count"
    ]
    
    # Some columns may not exist in older versions of the dataset, adapt if needed
    for col in keep_cols:
        if col not in biz_df.columns:
            print(f"Warning: Column '{col}' not found in business JSON.")
    biz_df = biz_df[[c for c in keep_cols if c in biz_df.columns]]
    
    print(f"Loading aspect data from {aspect_csv}...")
    aspect_df = pd.read_csv(aspect_csv)
    
    # Merge on 'business_id'
    merged_df = aspect_df.merge(biz_df, on="business_id", how="left")
    
    # Reorder columns for clarity
    col_order = [
        "business_id", "name", "city", "state", 
        "stars", "review_count", 
        "aspect", "biz_avg_aspect_stars", "biz_avg_aspect_confidence"
    ]
    final_cols = [c for c in col_order if c in merged_df.columns]
    merged_df = merged_df[final_cols]
    
    print(f"Merged shape: {merged_df.shape}")
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved merged business + aspect data to: {output_csv}")

if __name__ == "__main__":
    merge_business_metadata()
