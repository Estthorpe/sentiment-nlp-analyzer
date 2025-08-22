import argparse
import os
import re
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ---------------- Utilities ---------------- #

RATING_REGEX = re.compile(r"Rated\s+(\d(?:\.\d)?)\s+out\s+of\s+5\s+stars", re.IGNORECASE)

SENTIMENT_MAP_DOC = """\
Label strategy:
  stars 1–2 -> label 0 (negative)
  stars 3   -> label 1 (neutral)
  stars 4–5 -> label 2 (positive)
"""

# Minimal mojibake fixer for apostrophes/quotes commonly seen in scraped CSVs
MOJIBAKE_REPLACEMENTS = {
    "â€™": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€": "”",
    "â€“": "–",
    "â€”": "—",
    "â€¢": "•",
    "â€": "”",
    "Ã©": "é",
}

def fix_mojibake(text:Optional[str]) -> str:
    if not isinstance(text, str):
        return " "
    
    # Try latin1->utf8 roundtrip where needed
    try:
        repaired = text.encode("latin1", errors = "ignore").decode("utf-8", errors="ignore")

    except Exception:
        repaired = text
    #Replace common leftovers

    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    # Normalize whitespace
    repaired = " ".join(repaired.strip().split())
    return repaired


def extract_stars(rating_str: Optional[str]) -> Optional[float]:
    """Extract numeric stars from strings like "Rated 1 out of 5 stars"""
    if not isinstance (rating_str, str):
        return None
    m = RATING_REGEX.search(rating_str)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None
    

def map_stars_to_label(stars: Optional[float]) -> Optional[int]:
    if stars is None:
        return None
    if 1.0 <= stars < 3.0:
        return 0
    if stars == 3.0:
        return 1
    if 4.0 <=stars <= 5.0:
        return 2
    return None

def save_bar(series: pd.Series, title: str, out_png: str, xlab: str = "Class") -> None:
    counts = series.value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[saved] {out_png}")


def profile_print(df: pd.DataFrame, text_col: str, star_col: str, label_col: str) -> None:
    print("\n=== BASIC PROFILE ===")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Text column: '{text_col}' | Stars column: '{star_col}' | Label column: '{label_col}'")
    print("\nNull counts (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\nSamples:")
    print(df[[text_col, star_col, label_col]].head(5).to_string(index=False))
    print("\n" + SENTIMENT_MAP_DOC)


#---------------Main Prep---------------#
def prepare_dataset(
        raw_csv:str,
        out_dir:str = "data/processed",
        val_size: float = 0.10,
        test_size: float = 0.10,
        min_text_len: int = 5,
        random_state: int = 42, 
) -> Dict[str, str]:
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"CSV not found: {raw_csv}")
    
    # Try utf-8 first; fall back to Latin-1 if needed
    try:
        df = pd.read_csv(raw_csv, encoding="utf-8")
    except UnicoeDecodeError:
        df = pd.read_csv(raw_csv, encoding="latin-1")

    print(f"Loaded {raw_csv} with shape {df.shape}")

    required = {"Rating", "Review Title", "Review Text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required  columns missing:{missing}. Found: {list(df.columns)}")
    
    #Build unified text = "<title>. <body."
    title = df["Review Title"].map(fix_mojibake).fillna("")
    body = df["Review Text"].map(fix_mojibake).fillna("")
    text = (title + ". " + body).str.strip().str.replace(r"^\.\s*", "", regex=True)

    # Extract numeric stars
    stars = df["Rating"].map(extract_stars)

    work = pd.DataFrame(
        {
            "text": text,
            "stars": stars,
        }
    )

    # Remove null stars or empty/very-short texts
    before = len(work)
    work.dropna(subset=["stars"], inplace=True)
    work["text"] = work["text"].astype(str).str.strip()
    work = work[work["text"].str.len() >= min_text_len]
    work.drop_duplicates(subset=["text", "stars"], inplace=True)
    after = len(work)
    print(f"Filtered {before - after:,} rows (nulls/short/dupes). Remaining: {after:,}")

    # Map stars -> label
    work["label"] = work["stars"].map(map_stars_to_label)
    work = work.dropna(subset=["label"]).copy()
    work["label"] = work["label"].astype(int)

    # Optional keep some metadata if present
    for col in ["Reviewer Name", "Country", "Review Date", "Date of Experience"]:
        if col in df.columns:
            work[col.replace(" ", "_").lower()] = df[col]

    # Profile summary
    profile_print(work, "text", "stars", "label")

    # Plots
    os.makedirs("docs", exist_ok=True)
    save_bar(work["stars"], "Star Ratings Distribution", "docs/stars_distribution.png", xlab="Stars")
    save_bar(work["label"], "Sentiment Label Distribution (0=neg,1=neu,2=pos)", "docs/label_distribution.png", xlab="Label")

    # Save cleaned dataset
    os.makedirs(out_dir, exist_ok=True)
    cleaned_path = os.path.join(out_dir, "reviews.csv")
    work.to_csv(cleaned_path, index=False)
    print(f"[saved] cleaned -> {cleaned_path}")

    # Stratified train/val/test
    test_fraction = test_size
    val_fraction = val_size / (1 - test_fraction)

    train_df, test_df = train_test_split(
        work, test_size=test_fraction, stratify=work["label"], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_fraction, stratify=train_df["label"], random_state=random_state
    )

    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[saved] train -> {train_path}  ({len(train_df):,})")
    print(f"[saved] val   -> {val_path}    ({len(val_df):,})")
    print(f"[saved] test  -> {test_path}   ({len(test_df):,})")

    return {"cleaned": cleaned_path, "train": train_path, "val": val_path, "test": test_path}


# ---------------- CLI ---------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Prepare the Amazon reviews dataset for sentiment modeling.")
    p.add_argument("--raw_csv", required=True, help="Path to raw CSV (e.g., data/raw/amazon_reviews.csv)")
    p.add_argument("--out_dir", default="data/processed", help="Directory to write cleaned data and splits")
    p.add_argument("--val_size", type=float, default=0.10, help="Validation fraction")
    p.add_argument("--test_size", type=float, default=0.10, help="Test fraction")
    p.add_argument("--min_text_len", type=int, default=5, help="Minimum length of combined text to keep")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    _ = prepare_dataset(
        raw_csv=args.raw_csv,
        out_dir=args.out_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        min_text_len=args.min_text_len,
        random_state=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())



