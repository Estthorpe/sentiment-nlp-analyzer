from joblib import load
import matplotlib.pyplot as plt
from utils.explain import lime_explain_text, predict_proba_sklearn

pipe = load("models/baseline/model.joblib")
text = "The delivery was late and the packaging was damaged."

fig = lime_explain_text(
    text,
    predict_proba=lambda X: predict_proba_sklearn(pipe, X),
    class_names=["neg","neu","pos"]
)
plt.show()
