"""
Explainability helpers:
- LIME text explanations for any classifier with predict_proba(list[str]) -> np.ndarray
- Global importance:
    * Baseline (LogReg): top +/- tokens from TF-IDF coefficients
    * XGBoost: simple SHAP summary (optional, can be heavy); else feature gain
- Transformer predict_proba wrapper using saved HF model in models/transformer/

Usage examples (in a notebook or script):
    from utils.explain import lime_explain_text, get_logreg_top_tokens, predict_proba_sklearn, predict_proba_transformer

    # LIME for baseline pipeline
    from joblib import load
    pipe = load("models/baseline/model.joblib")
    fig = lime_explain_text(
        text="the delivery was late and packaging damaged",
        predict_proba=lambda X: predict_proba_sklearn(pipe, X),
        class_names=["neg","neu","pos"]
    )

    # LIME for transformer
    fig = lime_explain_text(
        text="the delivery was late and packaging damaged",
        predict_proba=lambda X: predict_proba_transformer("models/transformer", X),
        class_names=["neg","neu","pos"]
    )
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

# ----- LIME -----
from lime.lime_text import LimeTextExplainer

# Optional SHAP (safe import)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ----- Transformer inference -----
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ---------- Predict-proba wrappers ----------
def predict_proba_sklearn(pipeline, texts):
    """
    For scikit-learn pipelines ( baseline/XGB). Returns np.ndarray (n_samples, n_classes). 
    """
    return pipeline.predict_proba(texts)

class _HFWrapper:
    def __init__(self, model_dir: str, device: str | None = None, max_len: int = 128, batch_size: int = 16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.max_len = max_len
        self.batch_size = batch_size

    @torch.no_grad()
    def predict_proba(self, texts):
        """
        Batched inference to keep memory use low.
        """
        if isinstance(texts, str):
            texts = [texts]

        out = []
        # Small batches
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,           # pad just to the longest in this chunk
                max_length=self.max_len,
                return_tensors="pt"
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)

            # free memory early
            del enc, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(out)


def predict_proba_transformer(model_dir: str, texts, max_len: int = 128, batch_size: int = 16):
    """
    Convenience function to load once per session and compute probs.
    For batch scripts, consider instantiating _HFWrapper and reusing it.
    """
    wrapper = _HFWrapper(model_dir=model_dir, max_len=max_len, batch_size=batch_size)
    return wrapper.predict_proba(texts)
    

# ---------- LIME per-example explanation ----------

def lime_explain_text(text: str, predict_proba, class_names=("neg","neu","pos"), num_features=12, num_samples=800):
    """
    Build and return a LIME explanation for a single text.
    num_samples controls number of perturbations LIME generates (default 800 here).
    """
    explainer = LimeTextExplainer(class_names=list(class_names))
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=num_features,
        num_samples=num_samples,
    )
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    return fig

# ---------- Global importance (baseline: LogReg coefficients) ----------

def get_logreg_top_tokens(pipeline, top_k=20):
    """
    Assumes pipeline = (cleaner) -> TfidfVectorizer -> LogisticRegression
    Returns dict: {"neg": [(token, weight), ...], "neu": [...], "pos": [...]}
    """
    vec = pipeline.named_steps.get("tfidf", None)
    clf = pipeline.named_steps.get("clf", None) or pipeline.named_steps.get("xgb", None)
    if vec is None or clf is None:
        raise ValueError("Pipeline must include 'tfidf' and a classifier step.")

    if hasattr(clf, "coef_"):
        # Multinomial LogReg: coef_[class, feature]
        feature_names = np.array(vec.get_feature_names_out())
        coefs = clf.coef_
        classes = clf.classes_.tolist()  # e.g., [0,1,2]
        out = {}
        for i, cls in enumerate(classes):
            weights = coefs[i]
            idx_pos = np.argsort(weights)[-top_k:][::-1]
            idx_neg = np.argsort(weights)[:top_k]
            out[str(cls)] = {
                "positive_tokens": list(zip(feature_names[idx_pos].tolist(), weights[idx_pos].tolist())),
                "negative_tokens": list(zip(feature_names[idx_neg].tolist(), weights[idx_neg].tolist()))
            }
        return out
    else:
        # XGB fallback: feature importance by gain
        if hasattr(clf, "get_booster"):
            booster = clf.get_booster()
            score = booster.get_score(importance_type="gain")
            # score keys are like "f1234" -> feature index
            kv = []
            fn = vec.get_feature_names_out()
            for k, v in score.items():
                try:
                    idx = int(k[1:])
                    kv.append((fn[idx], v))
                except:
                    continue
            kv.sort(key=lambda x: x[1], reverse=True)
            return {"global_gain": kv[:top_k]}
        else:
            raise ValueError("Unsupported classifier type for global importance.")


# ---------- SHAP (optional) ----------

def shap_summary_for_xgb(pipeline, sample_texts=None, max_samples=1000, show=False):
    """
    SHAP summary for XGBoost on TF-IDF features. Can be memory-heavy.
    """
    if not HAS_SHAP:
        raise ImportError("shap not installed.")
    vec = pipeline.named_steps["tfidf"]
    xgb = pipeline.named_steps["xgb"]
    if sample_texts is None:
        raise ValueError("Provide sample_texts (list of strings).")
    X = vec.transform(sample_texts[:max_samples])
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X)
    # Non-interactive summary (text names may be long)
    shap.summary_plot(shap_values, X, feature_names=vec.get_feature_names_out(), show=show)