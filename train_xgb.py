"""
Optimized trainer: TF-IDF + XGBoost + MLflow
Run:
  python train_xgb.py --data_dir data/processed --models_dir models/xgb --run_name xgb_tfidf
"""

import argparse, os, joblib, numpy as np, pandas as pd, mlflow, mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from utils.text_clean import TextCleaner

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def plot_cm(y_true, y_pred, labels, title, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(); plt.imshow(cm); plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout(); ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=150); plt.close(fig); return out_png

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy":acc, "precision":pr, "recall":rc, "f1":f1}

def run(data_dir, models_dir, run_name, seed=42):
    train = pd.read_csv(os.path.join(data_dir,"train.csv"))
    val   = pd.read_csv(os.path.join(data_dir,"val.csv"))
    test  = pd.read_csv(os.path.join(data_dir,"test.csv"))

    X_train, y_train = train["text"].tolist(), train["label"].astype(int).tolist()
    X_val,   y_val   = val["text"].tolist(),   val["label"].astype(int).tolist()
    X_test,  y_test  = test["text"].tolist(),  test["label"].astype(int).tolist()

    pipe = Pipeline([
        ("clean", TextCleaner(remove_stopwords=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, token_pattern=r"(?u)\b\w\w+\b")),
        ("xgb", XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=4
        ))
    ])

    pipe.fit(X_train, y_train)

    def eval_split(name, X, y):
        y_pred = pipe.predict(X)
        m = metrics(y, y_pred)
        print(f"\n== {name} metrics ==\n{m}")
        return y_pred, m

    y_val_pred,  val_m  = eval_split("val",  X_val,  y_val)
    y_test_pred, test_m = eval_split("test", X_test, y_test)

    labels = sorted(set(y_train) | set(y_val) | set(y_test))
    ensure_dir("docs")
    val_cm  = plot_cm(y_val,  y_val_pred,  labels, "Confusion Matrix (val) - XGB",  "docs/confusion_val_xgb.png")
    test_cm = plot_cm(y_test, y_test_pred, labels, "Confusion Matrix (test) - XGB", "docs/confusion_test_xgb.png")
    with open("docs/classification_report_val_xgb.txt","w",encoding="utf-8") as f: f.write(classification_report(y_val,y_val_pred,digits=3))
    with open("docs/classification_report_test_xgb.txt","w",encoding="utf-8") as f: f.write(classification_report(y_test,y_test_pred,digits=3))

    ensure_dir(models_dir)
    model_path = os.path.join(models_dir,"model.joblib"); joblib.dump(pipe, model_path)
    print(f"\n[saved] {model_path}")

    mlflow.set_experiment("xgb_tfidf")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("tfidf_ngram","(1,2)"); mlflow.log_param("tfidf_min_df",3); mlflow.log_param("tfidf_max_df",0.9)
        for p,v in pipe.named_steps["xgb"].get_xgb_params().items():
            mlflow.log_param(f"xgb_{p}", v)
        for k,v in val_m.items(): mlflow.log_metric(f"val_{k}", float(v))
        for k,v in test_m.items(): mlflow.log_metric(f"test_{k}", float(v))
        mlflow.log_artifact(val_cm); mlflow.log_artifact(test_cm)
        mlflow.log_artifact("docs/classification_report_val_xgb.txt")
        mlflow.log_artifact("docs/classification_report_test_xgb.txt")
        mlflow.sklearn.log_model(pipe, artifact_path="model")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--models_dir", default="models/xgb")
    ap.add_argument("--run_name", default="xgb_tfidf")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.models_dir, args.run_name, args.seed)