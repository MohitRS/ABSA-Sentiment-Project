# src/chunked_preprocessing.py

import json
import pandas as pd
import re
import os

# If you want to filter out non-English reviews, install langdetect
try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("langdetect not installed. Skipping language filtering.")

def read_yelp_json_in_chunks(json_path, chunk_size=100_000):
    """
    Generator function to read a large Yelp JSON lines file in chunks.
    Each line is a JSON object representing a single review.

    :param json_path: Path to the Yelp dataset (JSON lines).
    :param chunk_size: Number of JSON lines to accumulate before yielding a chunk.
    :yield: Pandas DataFrame of up to 'chunk_size' reviews.
    """
    chunk = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Parse line-level JSON into a dict
            data = json.loads(line)
            chunk.append(data)
            # If we hit the chunk size, yield a DataFrame and reset
            if (i + 1) % chunk_size == 0:
                yield pd.DataFrame(chunk)
                chunk = []

        # If there's anything left in the chunk buffer, yield it
        if chunk:
            yield pd.DataFrame(chunk)



def clean_text(text):
    """
    Basic text cleaning: lowercase, remove URLs/HTML, trim whitespace.
    """
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'<.*?>', '', text)                   # remove HTML
    text = text.strip()
    return text

def filter_non_english(df, text_col='text'):
    if not HAS_LANGDETECT:
        return df
    
    lang_list = []
    for txt in df[text_col]:
        try:
            lang = detect(txt)
        except LangDetectException:
            lang = 'unknown'
        lang_list.append(lang)
    
    df['detected_language'] = lang_list
    df = df[df['detected_language'] == 'en'].drop(columns=['detected_language'])
    return df

def process_yelp_in_chunks(
    json_path="data/raw/yelp_academic_dataset_review.json",
    out_csv="data/processed/yelp_reviews_chunked_clean.csv",
    chunk_size=100_000,
    drop_dupes_by_text=True,
    filter_language=False,
    min_length=10
):
    """
    Processes a large Yelp JSON lines file in chunks to avoid OOM errors.
    Each chunk is read, cleaned, and appended to 'out_csv'.

    :param json_path: Path to Yelp dataset (JSON lines)
    :param out_csv: Output CSV path
    :param chunk_size: Number of lines per chunk
    :param drop_dupes_by_text: Whether to drop duplicates based on 'text' within each chunk
    :param filter_language: If True, removes non-English reviews
    :param min_length: Drop reviews with text length < min_length
    """
    # Ensure output directory
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    first_write = True
    total_rows_written = 0

    for chunk_i, df_chunk in enumerate(read_yelp_json_in_chunks(json_path, chunk_size=chunk_size)):
        print(f"\nProcessing chunk {chunk_i+1} ... size={len(df_chunk)}")

        # --- Basic structure: 'review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date' ---
        # Drop missing text
        df_chunk = df_chunk.dropna(subset=['text'])
        df_chunk = df_chunk[df_chunk['text'].str.strip() != '']

        # Remove short reviews
        df_chunk = df_chunk[df_chunk['text'].str.len() >= min_length]

        # Clean text
        df_chunk['text'] = df_chunk['text'].apply(clean_text)

        # Optional: language filter
        if filter_language:
            df_chunk = filter_non_english(df_chunk, text_col='text')

        # Optional: drop duplicates by text (within the chunk)
        if drop_dupes_by_text:
            before = len(df_chunk)
            df_chunk = df_chunk.drop_duplicates(subset=['text'], keep='first')
            after = len(df_chunk)
            print(f"   Dropped {before - after} duplicates in this chunk.")

        # If there's nothing left after cleaning, skip
        if df_chunk.empty:
            print("   No data left after cleaning.")
            continue

        # Write to CSV (append mode after the first chunk)
        mode = 'w' if first_write else 'a'
        header = True if first_write else False

        df_chunk.to_csv(out_csv, index=False, mode=mode, header=header, encoding='utf-8')
        row_count = len(df_chunk)
        total_rows_written += row_count
        print(f"   Wrote {row_count} cleaned rows to {out_csv} (cumulative: {total_rows_written}).")

        first_write = False

    print(f"\n==> Finished processing. Total rows in final CSV: {total_rows_written}")

if __name__ == "__main__":
    process_yelp_in_chunks(
        json_path="data/raw/yelp_academic_dataset_review.json",
        out_csv="data/processed/yelp_reviews_chunked_clean.csv",
        chunk_size=100_000,       # Adjust based on your RAM
        drop_dupes_by_text=True,
        filter_language=False,    # Set True for language filtering
        min_length=10
    )

