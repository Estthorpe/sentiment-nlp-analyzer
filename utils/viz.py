from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def plot_label_distribution(df: pd.DataFrame, out_png: str):
    ax = df["label"].value_counts().sort_index().plot(kind="bar", rot=0)
    ax.set_title("Sentiment Label Distribution (0=neg,1=neu,2=pos)")
    ax.set_xlabel("Label"); ax.set_ylabel("Count")
    plt.tight_layout(); ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=150); plt.close()


def wordcloud_from_texts(texts, out_png: str, extra_stopwords=None, max_words=200):
    sw = set(STOPWORDS)
    if extra_stopwords:
        sw |= set(extra_stopwords)
    wc = WordCloud(width=1200, height=600, background_color="white", stopwords=sw, max_words=max_words)
    wc.generate(" ".join([t for t in texts if isinstance(t, str)]))
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(); ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=150); plt.close()

def class_wordclouds(df: pd.DataFrame, out_dir="docs", prefix="wc", max_words=200):
    ensure_dir(out_dir)
    for label, group in df.groupby("label"):
        out_png = os.path.join(out_dir, f"{prefix}_label{label}.png")
        wordcloud_from_texts(group["text"].astype(str).tolist(), out_png=out_png, max_words=max_words)