"""
src/hf_sentiment.py

Usage:
  python src/hf_sentiment.py

Make sure you have installed:
  pip install transformers torch
"""

import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# 1) Load model & tokenizer
# If you have a GPU, device=0 uses the first CUDA device. Otherwise use CPU (-1).
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def parse_star_label(label_str):
    """
    The pipeline returns strings like '1 star' or '5 stars'.
    Extract the integer (1..5) from the label.
    """
    match = re.search(r"(\d+)\sstar", label_str.lower())
    if match:
        return int(match.group(1))
    else:
        return None  # in case the format is unexpected

def run_inference_chunked(
    input_csv="data/processed/yelp_reviews_chunked_clean.csv",
    output_csv="data/processed/yelp_hf_sentiment.csv",
    chunk_size=10000
):
    """
    Reads the cleaned Yelp CSV in chunks, applies Hugging Face sentiment pipeline,
    parses the label to an integer star rating, and appends results to output_csv.

    :param input_csv: Path to your cleaned Yelp reviews (~7M rows).
    :param output_csv: Output CSV with new columns: hf_label, hf_stars, hf_score
    :param chunk_size: Number of rows per chunk in memory.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    first_write = True
    total_processed = 0

    # 2) Read in chunks
    for chunk_idx, df_chunk in enumerate(pd.read_csv(input_csv, chunksize=chunk_size)):
        print(f"\n[Chunk {chunk_idx+1}] Loaded {len(df_chunk)} rows.")
        if df_chunk.empty:
            continue

        # Convert text column to string (in case of dtype issues)
        df_chunk['text'] = df_chunk['text'].astype(str)

        # 3) Inference with HF pipeline
        # 'truncation=True' ensures we don't exceed model max length (512 tokens).
        # This can still be slow for very large datasets.
        results = sentiment_pipeline(
            list(df_chunk['text']),
            truncation=True
        )

        # 4) Parse pipeline outputs
        # results is a list of dicts like: [{'label': '4 stars', 'score': 0.93}, ...]
        labels = [r["label"] for r in results]
        scores = [r["score"] for r in results]
        star_values = [parse_star_label(lbl) for lbl in labels]

        # 5) Add columns to the chunk
        df_chunk['hf_label'] = labels     # e.g., "4 stars"
        df_chunk['hf_stars'] = star_values
        df_chunk['hf_score'] = scores     # confidence score for that label

        # 6) Write to CSV
        mode = 'w' if first_write else 'a'
        header = True if first_write else False
        df_chunk.to_csv(output_csv, index=False, mode=mode, header=header)
        
        total_processed += len(df_chunk)
        print(f"   Processed & appended {len(df_chunk)} rows. Total processed: {total_processed}")
        first_write = False

    print(f"\nDone. Total rows with HF sentiment: {total_processed}")


if __name__ == "__main__":
    run_inference_chunked()
