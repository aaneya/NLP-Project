"""Traditional ML baseline: TF-IDF (1-2 grams) + Logistic Regression."""
import argparse
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import MODELS_DIR, compute_metrics, load_parquet_splits, save_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--train_file", default="data/train-00000-of-00001.parquet")
ap.add_argument("--test_file", default="data/test-00000-of-00001.parquet")
ap.add_argument("--train_samples", type=int, default=0, help="Optional cap; 0 uses the full training split")
ap.add_argument("--test_samples", type=int, default=0, help="Optional cap; 0 uses the full test split")
args = ap.parse_args()

X_tr, y_tr, X_te, y_te = load_parquet_splits(
    args.train_file, args.test_file, args.train_samples, args.test_samples)
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True,
                    stop_words="english", min_df=2),
    LogisticRegression(max_iter=1000, C=4.0))

t0 = time.time()
model.fit(X_tr, y_tr)
train_time = time.time() - t0

metrics = compute_metrics(y_te, model.predict(X_te))
print({k: round(v, 4) for k, v in metrics.items() if k != "confusion_matrix"})
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODELS_DIR / "baseline.joblib")
save_metrics("TF-IDF + Logistic Regression", metrics,
             {"train_samples": len(X_tr), "test_samples": len(X_te),
              "train_seconds": round(train_time, 1), "source": "local parquet splits"})
