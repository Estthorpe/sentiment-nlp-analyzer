"""
Baseline trainer (robust): TF-IDF + Logistic Regression + MLflow

- Loads data/processed/{train,val,test}.csv
- Cleans text explicitly (TextCleaner) so we can probe tokens
- Tries 3 vectorizer strategies until one fits:
    A) word ngrams, min_df=3, max_df=0.9
    B) word ngrams, min_df=1, max_df=1.0
    C) char_wb ngrams (3,5)  <-- near-guaranteed fallback
- Builds a final Pipeline(cleaner -> tfidf -> logreg), fits, evaluates
- Logs metrics/artifacts to MLflow; saves model to models/baseline/model.joblib
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.exceptions import NotFittedError

from utils.text_clean import TextCleaner

# ----------------- helpers -----------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def show_sample(name, X, y, k=1):
    print(f"[{name}] samples={len(X):,} | label set={sorted(set(y)) if len(y)>0 else []}")
    for t in X[:10]:
        if isinstance(t, str) and t.strip():
            print("  EX:", (t[:140] + "…") if len(t) > 140 else t)
            break

def sanitize_texts(texts):
    s = pd.Series(texts, dtype="object").fillna("").astype(str).str.strip()
    if (s.str.len() == 0).all():
        raise ValueError("All texts are empty after initial sanitization.")
    return s.tolist()

def plot_cm(y_true, y_pred, labels, title, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=150); plt.close(fig)
    return out_png

def metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    auc = None
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        y_true_bin = np.eye(n_classes)[np.array(y_true, dtype=int)]
        try:
            auc = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
        except Exception:
            auc = None
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "roc_auc_ovr_macro": auc}

# ----------------- training -----------------

def run_training(data_dir: str, models_dir: str, run_name: str, seed: int = 42):
    # 1) Load splits
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train_raw, y_train = train["text"].tolist(), train["label"].astype(int).tolist()
    X_val_raw,   y_val   = val["text"].tolist(),   val["label"].astype(int).tolist()
    X_test_raw,  y_test  = test["text"].tolist(),  test["label"].astype(int).tolist()

    show_sample("train (raw)", X_train_raw, y_train)
    show_sample("val (raw)",   X_val_raw,   y_val)

    # 2) Clean explicitly (so we can probe if tokens survive)
    cleaner = TextCleaner(remove_stopwords=False)
    X_train = sanitize_texts(cleaner.transform(X_train_raw))
    X_val   = sanitize_texts(cleaner.transform(X_val_raw))
    X_test  = sanitize_texts(cleaner.transform(X_test_raw))

    show_sample("train (clean)", X_train, y_train)

    # 3) Try vectorizer strategies
    strategies = [
        dict(name="word_min3", params=dict(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9, token_pattern=r"(?u)\b\w\w+\b")),
        dict(name="word_min1", params=dict(analyzer="word", ngram_range=(1,2), min_df=1, max_df=1.0, token_pattern=r"(?u)\b\w\w+\b")),
        dict(name="char_fallback", params=dict(analyzer="char_wb", ngram_range=(3,5), min_df=1, max_df=1.0)),
    ]

    vec = None
    for strat in strategies:
        try:
            print(f"\n[vectorizer] trying: {strat['name']} -> {strat['params']}")
            candidate = TfidfVectorizer(**strat["params"])
            Xt = candidate.fit_transform(X_train)
            if Xt.shape[1] == 0:
                raise ValueError("Vectorizer produced 0 features.")
            vocab_size = len(candidate.vocabulary_) if hasattr(candidate, "vocabulary_") else Xt.shape[1]
            print(f"  OK: features={Xt.shape[1]:,}  (vocab≈{vocab_size:,})")
            vec = candidate
            break
        except Exception as e:
            print(f"  FAIL: {e}")

    if vec is None:
        raise RuntimeError("All vectorizer strategies failed. Inspect the data/cleaning settings.")

    # 4) Fit classifier on the cleaned + vectorized data
    Xt_train = vec.transform(X_train)
    clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
    clf.fit(Xt_train, y_train)

    # 5) Build final pipeline: cleaner -> fixed tfidf -> logreg
    #    We pass the chosen vectorizer into the pipeline (it will refit, which is fine)
    pipe = Pipeline([
        ("clean", TextCleaner(remove_stopwords=False)),
        ("tfidf", TfidfVectorizer(**vec.get_params())),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)),
    ])
    pipe.fit(X_train_raw, y_train)

    # 6) Evaluate (val + test)
    def eval_split(name, X_raw, y):
        y_pred = pipe.predict(X_raw)
        try:
            y_proba = pipe.predict_proba(X_raw)
        except Exception:
            y_proba = None
        m = metrics(y, y_pred, y_proba)
        print(f"\n== {name} metrics =="); print(m)
        return y_pred, m

    y_val_pred,  val_m  = eval_split("val",  X_val_raw,  y_val)
    y_test_pred, test_m = eval_split("test", X_test_raw, y_test)

    # 7) Artifacts
    labels = sorted(set(y_train) | set(y_val) | set(y_test))
    ensure_dir("docs")
    val_cm  = plot_cm(y_val,  y_val_pred,  labels, "Confusion Matrix (val)",  "docs/confusion_val.png")
    test_cm = plot_cm(y_test, y_test_pred, labels, "Confusion Matrix (test)", "docs/confusion_test.png")
    with open("docs/classification_report_val.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_val, y_val_pred, digits=3))
    with open("docs/classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_test_pred, digits=3))

    # 8) Save model
    ensure_dir(models_dir)
    model_path = os.path.join(models_dir, "model.joblib")
    joblib.dump(pipe, model_path)
    print(f"\n[saved] {model_path}")

    # 9) MLflow logging
    mlflow.set_experiment("baseline_tfidf_logreg")
    with mlflow.start_run(run_name=run_name):
        # Vectorizer params (for traceability)
        for k, v in vec.get_params().items():
            mlflow.log_param(f"tfidf_{k}", v)
        mlflow.log_param("clf", "logreg_balanced")
        mlflow.log_param("seed", seed)

        for k, v in val_m.items():
            if v is not None: mlflow.log_metric(f"val_{k}", float(v))
        for k, v in test_m.items():
            if v is not None: mlflow.log_metric(f"test_{k}", float(v))

        mlflow.log_artifact(val_cm); mlflow.log_artifact(test_cm)
        mlflow.log_artifact("docs/classification_report_val.txt")
        mlflow.log_artifact("docs/classification_report_test.txt")

        mlflow.sklearn.log_model(pipe, artifact_path="model")
        mlflow.log_artifact(model_path)

# ----------------- CLI -----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--models_dir", default="models/baseline")
    ap.add_argument("--run_name", default="baseline_tfidf_logreg")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training(args.data_dir, args.models_dir, args.run_name, args.seed)
