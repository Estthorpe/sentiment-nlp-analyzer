"""
LIME explanation for the fine-tuned Transformer (DistilBERT).

Usage (PowerShell):
  python lime_test_transformer.py --text "The delivery was late and the packaging was damaged."
  # or
  python lime_test_transformer.py --text "I love this product, super fast delivery!"

Outputs:
  - Shows a LIME bar chart window
  - Also saves the figure to: docs/lime_transformer.png
  - Prints predicted class and probabilities
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.explain import lime_explain_text, predict_proba_transformer

CLASS_NAMES = ["neg", "neu", "pos"]  # 0,1,2


def main(text: str, model_dir: str, max_len: int):
    # Get probabilities from the saved HF model
    probs = predict_proba_transformer(model_dir, [text], max_len=max_len)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    print("\nText:")
    print(text)
    print("\nPredicted:", pred_label)
    print("Probabilities (neg, neu, pos):", np.round(probs, 4).tolist())

    # Build LIME chart
    fig = lime_explain_text(
        text=text,
        predict_proba=lambda X: predict_proba_transformer(model_dir, X, max_len=max_len),
        class_names=CLASS_NAMES,
        num_features=12,
    )
    plt.tight_layout()

    os.makedirs("docs", exist_ok=True)
    out_png = os.path.join("docs", "lime_transformer.png")
    plt.savefig(out_png, dpi=150)
    print(f"\n[saved] {out_png}")

    # Also show interactively (close the window to return to the shell)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Review text to explain")
    ap.add_argument("--model_dir", default="models/transformer", help="Path to saved HF model directory")
    ap.add_argument("--max_len", type=int, default=128, help="Tokenizer max length")
    args = ap.parse_args()
    main(args.text, args.model_dir, args.max_len)
