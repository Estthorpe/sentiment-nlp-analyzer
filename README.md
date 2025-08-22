# 💬 Product Review Sentiment Analyzer

TF‑IDF+LogReg **baseline** and **DistilBERT** Transformer that classify Amazon/Yelp‑style reviews as **negative / neutral / positive**, with **LIME** and **SHAP** explanations and a **Streamlit** app.

<img alt="app hero" src="docs/screenshot_app.png" width="720"/>

## ✨ Features
- Clean text pipeline (lowercase, punctuation/emoji stripping, de‑noise)
- Label mapping from star ratings (1–2→neg, 3→neu, 4–5→pos)
- Stratified train/val/test splits
- Baseline: **TF‑IDF + Logistic Regression** (MLflow tracking)
- Optimized: **XGBoost on TF‑IDF** (optional; SHAP global plot)
- Transformer: **DistilBERT**, max_len 128, 3 epochs (MLflow)
- **Explainability**: LIME per‑example, SHAP (XGB), global token importances (LogReg)
- **Streamlit app**: single review + batch CSV, **side‑by‑side** baseline vs transformer, confidence gauge
- Batch script to save 20 LIME charts into `docs/explanations/`

# 🚀 Quickstart
```bash
# 1) create & activate venv (Windows PowerShell)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate

# 2) install deps
pip install -r requirements.txt

# 3) (once) prepare data
python utils/data_prep.py --in data/raw/amazon_reviews.csv --out data/processed

# 4) train baseline + xgb (optional)
python train_baseline.py
python train_xgb.py             # if present

# 5) train transformer
python train_transformer.py

# 6) generate LIME batches for docs
python batch_explain.py --n 20 --seed 42

# 7) run the app
streamlit run app/streamlit_app.py
🧪 Results (example)
Baseline (TF‑IDF + LogReg): val F1≈0.72, test F1≈0.71

XGB on TF‑IDF: val F1≈0.63, test F1≈0.64 (example run; varies)

DistilBERT: val F1≈0.753, test F1≈0.733

See MLflow for runs:

python -m mlflow ui --backend-store-uri mlruns --port 5500

🖼 Explainability
Per‑example LIME: docs/explanations/{baseline,transformer}/*.png

Global token importances (LogReg): docs/explanations/global_importance_baseline_logreg.png

SHAP summary (XGB, optional): docs/explanations/shap_summary_xgb.png

🗃 Data
Dataset: Amazon Product Reviews from Kaggle (example source: dongrelaxman/amazon-reviews-dataset).
Expected columns (after prep): text, stars, label, reviewer_name, country, review_date, date_of_experience.

Label mapping: 1–2 → 0 (neg), 3 → 1 (neu), 4–5 → 2 (pos).

⚠️ Notes
Models may be large. If pushing to GitHub, consider Git LFS for models/transformer/* or add those files to .gitignore and re‑generate locally.


📜 License & Attribution
Code: MIT

Model: DistilBERT (Hugging Face Transformers)

Data: Kaggle (respect dataset license)


