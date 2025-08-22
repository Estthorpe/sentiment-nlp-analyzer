"""
Batch explainability:
- Sample N reviews from data/processed/test.csv (balanced across labels if possible)
- For each sample:
    * LIME explanation for baseline (sklearn pipeline)
    * LIME explanation for transformer (HF model)
- Save figures under:
    docs/explanations/baseline/*.png
    docs/explanations/transformer/*.png
- Save a global importance plot for baseline (coef-based), and optional SHAP for XGB if present.
"""

import argparse
import os
import gc
import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # ensure we can save figures headlessly
import matplotlib.pyplot as plt

from joblib import load

from utils.explain import (
    lime_explain_text,
    predict_proba_sklearn,
    predict_proba_transformer,
    get_logreg_top_tokens,
)

CLASS_NAMES = ["neg", "neu", "pos"]  # maps 0,1,2


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_models():
    """Load baseline sklearn pipeline and confirm transformer dir exists."""
    baseline_path = "models/baseline/model.joblib"
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline model not found at {baseline_path}")
    baseline = load(baseline_path)

    transformer_dir = "models/transformer"
    if not os.path.exists(transformer_dir):
        raise FileNotFoundError(f"Transformer model dir not found at {transformer_dir}")

    return baseline, transformer_dir


def sample_test_rows(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Balanced sample from test.csv if labels available, with safe top-up.
    """
    path = "data/processed/test.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test split not found at {path}. Run data prep first.")
    df = pd.read_csv(path)

    # Balanced by label if possible
    if "label" in df.columns and df["label"].nunique() > 1:
        per = max(1, n // df["label"].nunique())
        groups = []
        for _, g in df.groupby("label"):
            k = min(len(g), per)
            if k > 0:
                groups.append(g.sample(n=k, random_state=seed))
        if groups:
            df_s = pd.concat(groups, ignore_index=True)
        else:
            df_s = pd.DataFrame(columns=df.columns)

        # Top-up from remaining rows if needed
        if len(df_s) < n:
            remaining = df.drop(df_s.index, errors="ignore")
            if not remaining.empty:
                k_topup = min(n - len(df_s), len(remaining))
                df_s = pd.concat(
                    [df_s, remaining.sample(n=k_topup, random_state=seed)],
                    ignore_index=True,
                )
    else:
        df_s = df.sample(n=min(n, len(df)), random_state=seed)

    # Final safety: never request more than available
    if len(df_s) > n:
        df_s = df_s.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        df_s = df_s.reset_index(drop=True)
    return df_s


def explain_one_baseline(text: str, pipeline, out_png: str) -> str:
    fig = lime_explain_text(
        text=text,
        predict_proba=lambda X: predict_proba_sklearn(pipeline, X),
        class_names=CLASS_NAMES,
        num_features=12,
        num_samples=800,   # ↓ lighter than default (5k)
    )
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def explain_one_transformer(text: str, model_dir: str, out_png: str, max_len: int = 96) -> str:
    fig = lime_explain_text(
        text=text,
        predict_proba=lambda X: predict_proba_transformer(model_dir, X, max_len=max_len, batch_size=8),
        class_names=CLASS_NAMES,
        num_features=12,
        num_samples=600,
    )
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_global_importance(baseline_pipeline, out_dir: str = "docs/explanations") -> None:
    """
    Global importance for baseline:
      - If LogisticRegression: plot top +weighted tokens per class from TF-IDF coefficients
      - If XGB present and shap installed: save SHAP summary (optional)
    """
    ensure_dir(out_dir)

    # Coefficient-based importance for LogReg baselines
    try:
        info = get_logreg_top_tokens(baseline_pipeline, top_k=20)  # dict keyed by class id str
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        classes = ["0", "1", "2"]
        titles = ["neg (0)", "neu (1)", "pos (2)"]
        for ax, cls, title in zip(axes, classes, titles):
            pos = info[cls]["positive_tokens"]  # (token, weight), descending by weight
            tokens = [t for t, _ in pos[::-1]]   # smallest to largest for barh
            weights = [w for _, w in pos[::-1]]
            ax.barh(tokens, weights)
            ax.set_title(f"Top TF-IDF tokens for class {title}")
            ax.set_xlabel("weight")
        plt.tight_layout()
        p = os.path.join(out_dir, "global_importance_baseline_logreg.png")
        plt.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[saved] {p}")
    except Exception as e:
        print(f"[warn] Could not produce coef-based global importance: {e}")

    # Optional: XGB + SHAP (only if available)
    xgb_path = "models/xgb/model.joblib"
    if os.path.exists(xgb_path):
        try:
            import shap  # noqa: F401
            xgb_pipe = load(xgb_path)
            df = pd.read_csv("data/processed/test.csv").sample(
                n=min(500, len(pd.read_csv("data/processed/test.csv"))),
                random_state=42
            )
            vec = xgb_pipe.named_steps["tfidf"]
            X = vec.transform(df["text"].astype(str).tolist())
            xgb = xgb_pipe.named_steps["xgb"]
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X)
            shap.summary_plot(
                shap_values, X, feature_names=vec.get_feature_names_out(), show=False, max_display=30
            )
            p = os.path.join(out_dir, "shap_summary_xgb.png")
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            plt.close()
            print(f"[saved] {p}")
        except Exception as e:
            print(f"[warn] Skipping SHAP for XGB: {e}")


def run(n: int = 20, seed: int = 42, max_len: int = 128) -> None:
    baseline, transformer_dir = load_models()
    df_s = sample_test_rows(n=n, seed=seed)

    base_dir = "docs/explanations"
    out_b = os.path.join(base_dir, "baseline")
    out_t = os.path.join(base_dir, "transformer")
    ensure_dir(out_b)
    ensure_dir(out_t)

    # Generate explanations
    for i, row in df_s.iterrows():
        text = str(row["text"])
        true_label = int(row["label"])

        # Baseline prediction (to add predicted class into filename)
        probs_b = predict_proba_sklearn(baseline, [text])[0]
        pred_b = int(np.argmax(probs_b))
        f_b = os.path.join(out_b, f"{i:02d}_true{true_label}_pred{pred_b}.png")
        explain_one_baseline(text, baseline, f_b)

        # Transformer prediction
        probs_t = predict_proba_transformer(transformer_dir, [text], max_len=max_len)[0]
        pred_t = int(np.argmax(probs_t))
        f_t = os.path.join(out_t, f"{i:02d}_true{true_label}_pred{pred_t}.png")
        explain_one_transformer(text, transformer_dir, f_t, max_len=max_len)

        print(f"[{i+1}/{len(df_s)}] saved:\n  {f_b}\n  {f_t}")

        # ---- free memory between samples ----
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Global plots
    save_global_importance(baseline, out_dir=base_dir)

    print("\n✅ Done. See docs/explanations/ for images.")
    print(" - Baseline LIME:     docs/explanations/baseline/*.png")
    print(" - Transformer LIME:  docs/explanations/transformer/*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()
    run(args.n, args.seed, args.max_len)
