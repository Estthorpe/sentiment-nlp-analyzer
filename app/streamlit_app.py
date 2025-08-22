# app/streamlit_app.py
# Streamlit UI for sentiment analysis with side-by-side explanations and batch CSV predictions.

import os
import sys
import io
from glob import glob

# Ensure we can import from project root when running "streamlit run app/streamlit_app.py"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load

from utils.explain import (
    predict_proba_sklearn,
    predict_proba_transformer,
    lime_explain_text,
)

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="wide")
CLASS_NAMES = ["neg", "neu", "pos"]


# --------------------------- CACHED LOADERS ---------------------------

@st.cache_resource
def load_baseline():
    path = "models/baseline/model.joblib"
    return load(path) if os.path.exists(path) else None

@st.cache_resource
def load_transformer_dir():
    path = "models/transformer"
    return path if os.path.exists(path) else None


baseline = load_baseline()
transformer_dir = load_transformer_dir()


# --------------------------- SMALL UI HELPERS ---------------------------

def confidence_gauge(prob, label: str):
    """Show a simple confidence meter for the predicted class."""
    # Convert to plain Python float and clip to [0, 1]
    try:
        val = float(prob)  # handles numpy.float32/64
    except Exception:
        val = 0.0
    if np.isnan(val):
        val = 0.0
    val = max(0.0, min(1.0, val))

    st.write(f"**Confidence in `{label}`:** {val:.2%}")
    st.progress(val)

def predict_single(text: str, model_choice: str, max_len: int):
    """Return (probs np.array shape (3,), pred_idx int)."""
    if model_choice == "Transformer (DistilBERT)":
        if transformer_dir is None:
            st.error("Transformer model not found at models/transformer/")
            st.stop()
        probs = predict_proba_transformer(transformer_dir, [text], max_len=max_len)[0]
    elif model_choice == "Baseline (TF‑IDF + LogReg)":
        if baseline is None:
            st.error("Baseline model not found at models/baseline/model.joblib")
            st.stop()
        probs = predict_proba_sklearn(baseline, [text])[0]
    else:
        raise ValueError("Unknown model choice")
    return probs, int(np.argmax(probs))

def render_lime(text: str, model_choice: str, max_len: int, lime_features: int, lime_samples: int):
    """Return a Matplotlib Figure with a LIME explanation."""
    if model_choice == "Transformer (DistilBERT)":
        if transformer_dir is None:
            st.error("Transformer model not found.")
            st.stop()
        fig = lime_explain_text(
            text,
            predict_proba=lambda X: predict_proba_transformer(transformer_dir, X, max_len=max_len),
            class_names=CLASS_NAMES,
            num_features=lime_features,
            num_samples=lime_samples,
        )
    else:
        if baseline is None:
            st.error("Baseline model not found.")
            st.stop()
        fig = lime_explain_text(
            text,
            predict_proba=lambda X: predict_proba_sklearn(baseline, X),
            class_names=CLASS_NAMES,
            num_features=lime_features,
            num_samples=lime_samples,
        )
    return fig

def docs_gallery(base_dir="docs/explanations", max_imgs=8):
    st.subheader("Saved Explanations (from docs/explanations/)")
    if not os.path.exists(base_dir):
        st.info("No docs/explanations folder yet. Run batch_explain.py to generate images.")
        return
    cols = st.columns(2)
    for sub, col in zip(["baseline", "transformer"], cols):
        col.markdown(f"**{sub.capitalize()}**")
        folder = os.path.join(base_dir, sub)
        if not os.path.exists(folder):
            col.write("_No images found_")
            continue
        files = sorted(glob(os.path.join(folder, "*.png")))[:max_imgs]
        if not files:
            col.write("_No images found_")
        for f in files:
            col.image(f, caption=os.path.relpath(f))
            col.markdown(f"[Open file]({f})")


# --------------------------- SIDEBAR ---------------------------

with st.sidebar:
    st.header("Model")
    # Comparison mode toggle
    compare_mode = st.checkbox("Side-by-side (Baseline vs Transformer)", value=True, help="Show both models at once")
    if compare_mode:
        # Fixed to both models; the sliders below still apply to transformer
        st.caption("Comparing both models")
    else:
        model_choice = st.radio(
            "Choose a model",
            options=["Transformer (DistilBERT)", "Baseline (TF‑IDF + LogReg)"],
            index=0 if transformer_dir else 1
        )

    max_len = st.slider("Max tokens (Transformer)", min_value=64, max_value=256, value=128, step=32)
    lime_features = st.slider("LIME: number of features", 6, 20, 12)
    lime_samples = st.slider("LIME: perturbation samples", 200, 1500, 600, step=100)

st.title("💬 Product Review Sentiment Analyzer")
st.caption("TF‑IDF+LogReg baseline and DistilBERT transformer with LIME explanations")


# --------------------------- TABS ---------------------------

tab1, tab2, tab3 = st.tabs(["Single Review", "Batch CSV", "Docs Gallery"])

# ---- Single Review ----
with tab1:
    st.subheader("Single Review")
    text = st.text_area("Paste a review:", height=140,
                        value="The delivery was late and the packaging was damaged.")

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please paste a review.")
            st.stop()

        if compare_mode:
            # Predict with both models
            cols = st.columns(2, gap="large")

            # Baseline
            with cols[0]:
                st.markdown("### Baseline (TF‑IDF + LogReg)")
                if baseline is None:
                    st.warning("Baseline model missing.")
                else:
                    probs_b = predict_proba_sklearn(baseline, [text])[0]
                    pred_b = int(np.argmax(probs_b))
                    st.write(f"**Prediction:** {CLASS_NAMES[pred_b]}  |  **Probabilities:** {np.round(probs_b, 3)}")
                    confidence_gauge(probs_b[pred_b], CLASS_NAMES[pred_b])
                    fig_b = render_lime(text, "Baseline (TF‑IDF + LogReg)", max_len, lime_features, lime_samples)
                    st.pyplot(fig_b, clear_figure=True)

            # Transformer
            with cols[1]:
                st.markdown("### Transformer (DistilBERT)")
                if transformer_dir is None:
                    st.warning("Transformer model missing.")
                else:
                    probs_t = predict_proba_transformer(transformer_dir, [text], max_len=max_len)[0]
                    pred_t = int(np.argmax(probs_t))
                    st.write(f"**Prediction:** {CLASS_NAMES[pred_t]}  |  **Probabilities:** {np.round(probs_t, 3)}")
                    confidence_gauge(probs_t[pred_t], CLASS_NAMES[pred_t])
                    fig_t = render_lime(text, "Transformer (DistilBERT)", max_len, lime_features, lime_samples)
                    st.pyplot(fig_t, clear_figure=True)

        else:
            # Single-model mode
            probs, pred_idx = predict_single(text, model_choice, max_len)
            st.write(f"**Prediction:** {CLASS_NAMES[pred_idx]}  |  **Probabilities:** {np.round(probs, 3)}")
            confidence_gauge(probs[pred_idx], CLASS_NAMES[pred_idx])
            fig = render_lime(text, model_choice, max_len, lime_features, lime_samples)
            st.pyplot(fig, clear_figure=True)


# ---- Batch CSV ----
with tab2:
    st.subheader("Batch CSV")
    st.caption("Upload CSV with a column named **text**. We’ll add predicted label and probabilities.")
    up = st.file_uploader("CSV file", type=["csv"])

    if up is not None:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV must include a 'text' column.")
        else:
            # Choose which model(s) to use for batch
            batch_model = "both" if compare_mode else ("transformer" if model_choice.startswith("Transformer") else "baseline")
            do_lime = st.checkbox("Also generate LIME for first N rows", value=False)
            n_lime = st.slider("N for LIME (first rows)", 1, 10, 3)

            outputs = {}

            # Baseline predictions
            if batch_model in ("both", "baseline"):
                if baseline is None:
                    st.warning("Baseline model not found; skipping baseline.")
                else:
                    probs_b = predict_proba_sklearn(baseline, df["text"].astype(str).tolist())
                    preds_b = probs_b.argmax(axis=1)
                    out_b = df.copy()
                    out_b["pred_baseline"] = preds_b
                    out_b["prob_b_neg"] = probs_b[:, 0]
                    out_b["prob_b_neu"] = probs_b[:, 1]
                    out_b["prob_b_pos"] = probs_b[:, 2]
                    outputs["baseline"] = out_b

            # Transformer predictions
            if batch_model in ("both", "transformer"):
                if transformer_dir is None:
                    st.warning("Transformer model not found; skipping transformer.")
                else:
                    probs_t = predict_proba_transformer(transformer_dir, df["text"].astype(str).tolist(), max_len=max_len)
                    preds_t = probs_t.argmax(axis=1)
                    out_t = df.copy()
                    out_t["pred_transformer"] = preds_t
                    out_t["prob_t_neg"] = probs_t[:, 0]
                    out_t["prob_t_neu"] = probs_t[:, 1]
                    out_t["prob_t_pos"] = probs_t[:, 2]
                    outputs["transformer"] = out_t

            # Merge outputs if both
            if "baseline" in outputs and "transformer" in outputs:
                out = outputs["baseline"].join(outputs["transformer"].drop(columns=["text"]), rsuffix="_t")
            else:
                out = list(outputs.values())[0] if outputs else df

            st.markdown("**Preview:**")
            st.dataframe(out.head(20))

            # Optional LIME for first N rows
            if do_lime:
                st.markdown("**LIME explanations (first N rows):**")
                lim_cols = st.columns(2) if batch_model == "both" else [st]
                sample_rows = df.head(n_lime)
                for idx, row in sample_rows.iterrows():
                    st.write(f"**Row {idx}** — {row['text'][:120]}{'…' if len(row['text'])>120 else ''}")
                    if batch_model in ("both",):
                        with lim_cols[0]:
                            st.caption("Baseline")
                            if baseline is not None:
                                fig = lime_explain_text(
                                    row["text"],
                                    predict_proba=lambda X: predict_proba_sklearn(baseline, X),
                                    class_names=CLASS_NAMES,
                                    num_features=lime_features,
                                    num_samples=lime_samples,
                                )
                                st.pyplot(fig, clear_figure=True)
                        with lim_cols[1]:
                            st.caption("Transformer")
                            if transformer_dir is not None:
                                fig = lime_explain_text(
                                    row["text"],
                                    predict_proba=lambda X: predict_proba_transformer(transformer_dir, X, max_len=max_len),
                                    class_names=CLASS_NAMES,
                                    num_features=lime_features,
                                    num_samples=lime_samples,
                                )
                                st.pyplot(fig, clear_figure=True)
                    else:
                        fig = render_lime(row["text"],
                                          "Transformer (DistilBERT)" if batch_model=="transformer" else "Baseline (TF‑IDF + LogReg)",
                                          max_len, lime_features, lime_samples)
                        st.pyplot(fig, clear_figure=True)

            # Download
            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            st.download_button("Download predictions CSV", data=buf.getvalue(), file_name="predictions.csv", mime="text/csv")


# ---- Docs Gallery ----
with tab3:
    docs_gallery()
