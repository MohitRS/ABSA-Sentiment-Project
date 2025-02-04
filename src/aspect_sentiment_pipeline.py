#!/usr/bin/env python3

import os
import re
import sys
import argparse
import logging
import pandas as pd
import torch

# NLTK for sentence splitting
import nltk
from nltk.tokenize import sent_tokenize

# Transformers for the Hugging Face model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Make sure you have nltk 'punkt' data installed:
#   python -m nltk.downloader punkt

###############################################################################
# CONFIGURABLE ASPECT KEYWORDS DICTIONARY
###############################################################################
ASPECT_KEYWORDS = {
    "food": [
        "food", "taste", "flavor", "meal", "dish", "pizza", "burger", "fries",
        "menu", "sushi", "coffee", "drink", "delicious", "tasty", "beverage"
    ],
    "service": [
        "service", "waiter", "waitress", "staff", "server", "manager",
        "employee", "host", "customer service", "barista"
    ],
    "ambiance": [
        "ambiance", "atmosphere", "decor", "vibe", "music", "environment",
        "interior", "lighting"
    ],
    "price": [
        "price", "cost", "value", "expensive", "cheap", "overpriced", "bill", "money"
    ],
    # Add or adjust as needed...
}

###############################################################################
# SENTIMENT MODEL CONFIG
###############################################################################
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def parse_star_label(label_str: str) -> int:
    """
    Convert HF label like "4 stars" to an integer 4.
    Returns None if format is unexpected.
    """
    match = re.search(r"(\d+)\sstar", label_str.lower())
    if match:
        return int(match.group(1))
    return None

def detect_aspects_in_sentence(sentence: str) -> list:
    """
    Given a sentence, returns a list of aspects found (e.g. ["food", "service"]).
    Uses keyword matching from ASPECT_KEYWORDS dictionary.
    """
    aspects_found = []
    sent_lower = sentence.lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            # If the keyword is found in the sentence, mark the aspect
            if kw in sent_lower:
                aspects_found.append(aspect)
                break  # One match is enough to confirm this aspect
    return aspects_found

def analyze_aspects_in_review(
    review_text: str,
    hf_pipeline,
    batch_size: int = 32
) -> list:
    """
    1) Split the review into sentences
    2) For each sentence, find aspects
    3) Run HF sentiment (star rating) on that sentence
    4) Return a list of dicts: [{"aspect", "hf_stars", "score", "sentence"}, ...]

    We do *batched* inference for efficiency:
    - Collect all sentences that have at least 1 aspect
    - Do a pipeline call in mini-batches
    """
    results = []
    # Split review into sentences
    sentences = sent_tokenize(review_text)

    # Step A: gather all aspect sentences
    aspect_sentences = []
    aspect_map = []  # keep track of which aspects belong to which sentence
    for sent in sentences:
        aspects_in_sent = detect_aspects_in_sentence(sent)
        if aspects_in_sent:
            aspect_sentences.append(sent)
            aspect_map.append(aspects_in_sent)

    if not aspect_sentences:
        return results

    # Step B: run HF sentiment in mini-batches
    all_preds = []
    start_idx = 0
    while start_idx < len(aspect_sentences):
        end_idx = min(start_idx + batch_size, len(aspect_sentences))
        batch_sents = aspect_sentences[start_idx:end_idx]
        batch_preds = hf_pipeline(batch_sents, truncation=True)
        all_preds.extend(batch_preds)
        start_idx = end_idx

    # Step C: assemble results
    for idx, pred in enumerate(all_preds):
        label_str = pred["label"]   # e.g. "4 stars"
        score = pred["score"]       # confidence
        star_val = parse_star_label(label_str)

        # aspect_map[idx] is the list of aspects for this sentence
        sentence_txt = aspect_sentences[idx]
        for asp in aspect_map[idx]:
            results.append({
                "aspect": asp,
                "hf_stars": star_val,
                "hf_score": score,
                "sentence": sentence_txt
            })
    return results

###############################################################################
# MAIN CHUNK PROCESSING FUNCTION
###############################################################################
def process_aspect_sentiment(
    input_csv: str,
    output_csv: str,
    chunk_size: int = 5000,
    batch_size: int = 32,
    use_gpu: bool = False
):
    """
    Reads 'input_csv' in chunks, performs aspect-based sentiment analysis,
    and appends results to 'output_csv'.

    :param input_csv: Path to a large CSV with at least a 'text' column for each review.
    :param output_csv: Path to the resulting aspect-level CSV.
    :param chunk_size: Number of reviews to load into memory at once.
    :param batch_size: Batch size for HF pipeline inferences (speed optimization).
    :param use_gpu: If True, attempts to use the first CUDA device for faster inference.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading model/tokenizer: %s", MODEL_NAME)

    # Load HF model and tokenizer
    device_id = 0 if (use_gpu and torch.cuda.is_available()) else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    hf_pipeline = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device_id
    )

    logger.info("Reading input CSV in chunks from: %s", input_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    reader = pd.read_csv(input_csv, chunksize=chunk_size)
    first_write = True
    total_aspect_rows = 0
    total_reviews_processed = 0

    for chunk_idx, df_chunk in enumerate(reader, start=1):
        logger.info("Processing chunk %d with size=%d", chunk_idx, len(df_chunk))
        if df_chunk.empty:
            continue

        aspect_rows = []
        for row_idx, row in df_chunk.iterrows():
            review_text = str(row.get("text", "")).strip()
            if not review_text:
                continue

            # Optionally keep an ID if present
            review_id = row.get("review_id", None)
            business_id = row.get("business_id", None)

            # Perform aspect-level analysis
            aspects_info = analyze_aspects_in_review(
                review_text=review_text,
                hf_pipeline=hf_pipeline,
                batch_size=batch_size
            )
            if not aspects_info:
                continue

            for asp_dict in aspects_info:
                # Add extra columns from the original row if needed
                aspect_rows.append({
                    "review_id": review_id,
                    "business_id": business_id,
                    "aspect": asp_dict["aspect"],
                    "sentence": asp_dict["sentence"],
                    "hf_aspect_stars": asp_dict["hf_stars"],
                    "hf_aspect_score": asp_dict["hf_score"]
                    # You can also include row['stars'] if you want original star rating
                })

        # Convert aspect_rows to a DataFrame
        if aspect_rows:
            out_df = pd.DataFrame(aspect_rows)
            mode = 'w' if first_write else 'a'
            header = True if first_write else False
            out_df.to_csv(output_csv, index=False, mode=mode, header=header)

            total_aspect_rows += len(out_df)
            logger.info("Wrote %d aspect-level rows (cumulative %d).", len(out_df), total_aspect_rows)

        total_reviews_processed += len(df_chunk)
        logger.info("Total reviews processed so far: %d", total_reviews_processed)
        first_write = False

    logger.info("Finished aspect-based sentiment. Total aspect rows: %d", total_aspect_rows)
    logger.info("Output CSV located at: %s", output_csv)

###############################################################################
# CLI (COMMAND-LINE INTERFACE)
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Aspect-Based Sentiment: Dictionary + HF star model on large Yelp CSV"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to cleaned Yelp reviews CSV (must have a 'text' column)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV for aspect-level sentiment rows."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Number of rows to read per chunk (memory-based). Default=5000."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for HF pipeline inferences on sentences. Default=32."
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="If set, use GPU (device=0) if available."
    )
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    setup_logging()
    args = parse_arguments()

    logging.info("Starting aspect-based sentiment analysis with args: %s", args)
    process_aspect_sentiment(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu
    )

if __name__ == "__main__":
    main()
