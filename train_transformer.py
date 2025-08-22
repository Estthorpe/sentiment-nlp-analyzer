"""
Train DistilBERT for 3-class sentiment on CSV splits.

Inputs expected in data/processed/: train.csv, val.csv, test.csv
Each with columns: text, label (0=neg,1=neu,2=pos)

Run:
  python train_transformer.py --data_dir data/processed --out_dir models/transformer --run_name distilbert_base
"""

import argparse
import os
import numpy as np
import pandas as pd
import mlflow

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

NUM_LABELS = 3
MODEL_NAME = "distilbert-base-uncased"


def load_splits(data_dir: str) -> DatasetDict:
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    va = pd.read_csv(os.path.join(data_dir, "val.csv"))
    te = pd.read_csv(os.path.join(data_dir, "test.csv"))
    # Keep only the columns we need
    tr = tr[["text", "label"]].copy()
    va = va[["text", "label"]].copy()
    te = te[["text", "label"]].copy()
    return DatasetDict(
        train=Dataset.from_pandas(tr, preserve_index=False),
        validation=Dataset.from_pandas(va, preserve_index=False),
        test=Dataset.from_pandas(te, preserve_index=False),
    )


def tokenize_function(tokenizer, max_len):
    def _fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
    return _fn


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}


def main(data_dir, out_dir, run_name, max_len=128, epochs=3, bs=16, seed=42):
    print("Starting DistilBERT training…")

    mlflow.set_experiment("transformer_distilbert")

    # Load data
    ds = load_splits(data_dir)

    # Tokenizer + tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = ds.map(tokenize_function(tokenizer, max_len), batched=True)
    ds = ds.remove_columns(["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    # Training args — keep to a minimal, widely-supported set
    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "runs"),
        learning_rate=2e-5,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        seed=seed,
        report_to=[],  # disable external loggers; we use MLflow below
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],  # not required by args; we'll call evaluate() manually
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=run_name):
        # Params
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("max_len", max_len)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", bs)
        mlflow.log_param("seed", seed)

        # Train
        trainer.train()

        # Manual evaluation (validation + test)
        val_metrics = trainer.evaluate(ds["validation"])
        test_metrics = trainer.evaluate(ds["test"])

        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"val_{k}", float(v))
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", float(v))

        # Save model & tokenizer
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        mlflow.log_artifacts(out_dir)

        print("\n== Validation Metrics ==")
        print({k: v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        print("\n== Test Metrics ==")
        print({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        print(f"\nSaved transformer model -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="models/transformer")
    ap.add_argument("--run_name", default="distilbert_base")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.data_dir, args.out_dir, args.run_name, args.max_len, args.epochs, args.bs, args.seed)
