"""
Minimal text cleaner for sentiment modeling.

Features:
- lowercase
- fix HTML entities (e.g., &amp; -> &)
- remove URLs/emails/@user/#hashtag
- strip punctuation (optional digits)
- collapse whitespace
- optional English stopword removal

Usages:
1) Function:
    from utils.text_clean import clean_text
    s = clean_text("I LOVE this! Visit https://x.co")

2) Sklearn transformer:
    from utils.text_clean import TextCleaner
    pipe = Pipeline([
        ("clean", TextCleaner(remove_stopwords=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])

3) CLI (quick check on a CSV with 'text' column):
    python utils\\text_clean.py --in_csv data\\processed\\reviews.csv --out_csv data\\processed\\reviews_clean.csv
"""

from __future__ import annotations
import argparse
import html
import re
import string
import unicodedata
from typing import Iterable, List, Optional

# Optional NLTK stopwords (handled gracefully if not available)
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    EN_STOP = set(stopwords.words("english"))
except Exception:
    EN_STOP = set()

# --- Regexes ---
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w\.-]+?@[\w\.-]+\.\w+\b")
RE_USER = re.compile(r"@\w+")
RE_HASH = re.compile(r"#\w+")
RE_HTML_TAG = re.compile(r"<[^>]+>")

def strip_accents(text: str) -> str:
    """Remove accents/diacritics (NFKD)."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def clean_text(
    text: Optional[str],
    *,
    lower: bool = True,
    deaccent: bool = False,
    fix_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_usernames: bool = True,
    remove_hashtags: bool = True,
    remove_html_tags: bool = True,
    remove_punct: bool = True,
    remove_digits: bool = False,
    remove_stopwords: bool = False,
) -> str:
    """Minimal, safe cleaner. Returns empty string for non-str inputs."""
    if not isinstance(text, str):
        return ""

    s = text.strip()

    # 1) HTML entities & tags
    if fix_html:
        s = html.unescape(s)
    if remove_html_tags:
        s = RE_HTML_TAG.sub(" ", s)

    # 2) URLs, emails, usernames, hashtags
    if remove_urls:
        s = RE_URL.sub(" ", s)
    if remove_emails:
        s = RE_EMAIL.sub(" ", s)
    if remove_usernames:
        s = RE_USER.sub(" ", s)
    if remove_hashtags:
        s = RE_HASH.sub(" ", s)

    # 3) Case / accents
    if lower:
        s = s.lower()
    if deaccent:
        s = strip_accents(s)

    # 4) Punctuation / digits
    if remove_punct:
        s = s.translate(str.maketrans("", "", string.punctuation))
    if remove_digits:
        s = re.sub(r"\d+", " ", s)

    # 5) Collapse whitespace
    s = " ".join(s.split())

    # 6) Optional stopwords (after cleanup)
    if remove_stopwords and EN_STOP:
        tokens = [t for t in s.split() if t not in EN_STOP]
        s = " ".join(tokens)

    return s


# --------------- Sklearn transformer ---------------
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible cleaner. Applies clean_text() element-wise.
    """
    def __init__(
        self,
        lower: bool = True,
        deaccent: bool = False,
        fix_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_usernames: bool = True,
        remove_hashtags: bool = True,
        remove_html_tags: bool = True,
        remove_punct: bool = True,
        remove_digits: bool = False,
        remove_stopwords: bool = False,
    ):
        self.lower = lower
        self.deaccent = deaccent
        self.fix_html = fix_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_usernames = remove_usernames
        self.remove_hashtags = remove_hashtags
        self.remove_html_tags = remove_html_tags
        self.remove_punct = remove_punct
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> List[str]:
        if isinstance(X, (pd.Series, np.ndarray)):
            iterable = X.tolist()
        else:
            iterable = list(X)
        return [
            clean_text(
                x,
                lower=self.lower,
                deaccent=self.deaccent,
                fix_html=self.fix_html,
                remove_urls=self.remove_urls,
                remove_emails=self.remove_emails,
                remove_usernames=self.remove_usernames,
                remove_hashtags=self.remove_hashtags,
                remove_html_tags=self.remove_html_tags,
                remove_punct=self.remove_punct,
                remove_digits=self.remove_digits,
                remove_stopwords=self.remove_stopwords,
            )
            for x in iterable
        ]


# --------------- Optional CLI ---------------
def _cli():
    ap = argparse.ArgumentParser(description="Clean a CSV with a 'text' column (quick check).")
    ap.add_argument("--in_csv", required=True, help="input CSV path (must contain column 'text')")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    ap.add_argument("--remove_stopwords", action="store_true", help="remove English stopwords")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    cleaner = TextCleaner(remove_stopwords=args.remove_stopwords)
    df["text"] = cleaner.transform(df["text"])
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")

if __name__ == "__main__":
    _cli()
