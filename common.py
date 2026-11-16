"""Shared helpers: cleaning, data loading, metrics."""
import json
import re
from pathlib import Path

MODELS_DIR = Path("models")
METRICS_FILE = MODELS_DIR / "metrics.json"
LABELS = ["Negative", "Positive"]


def clean_text(text: str) -> str:
    """Remove HTML line breaks/tags and extra whitespace (keep case & punctuation:
    BERT's tokenizer handles those, and they carry sentiment signal)."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_imdb(train_samples=None, test_samples=2000, seed=42):
    from datasets import load_dataset
    ds = load_dataset("imdb")
    train = ds["train"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=seed)
    if train_samples:
        train = train.select(range(min(train_samples, len(train))))
    if test_samples:
        test = test.select(range(min(test_samples, len(test))))
    return ([clean_text(t) for t in train["text"]], list(train["label"]),
            [clean_text(t) for t in test["text"]], list(test["label"]))


def load_parquet_splits(train_file, test_file, train_samples=0, test_samples=0, seed=42):
    """Read supervised local Parquet splits without touching the unlabeled split."""
    import pyarrow.parquet as pq

    train = pq.read_table(train_file).to_pydict()
    test = pq.read_table(test_file).to_pydict()
    required = {"text", "label"}
    for name, frame in (("train", train), ("test", test)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} parquet must contain text and label columns; missing {sorted(missing)}")
    def unpack(frame, limit):
        rows = [(clean_text(str(text)), int(label)) for text, label in zip(frame["text"], frame["label"])
                if text is not None and label is not None]
        random.Random(seed).shuffle(rows)
        if limit:
            rows = rows[:min(limit, len(rows))]
        return [row[0] for row in rows], [row[1] for row in rows]

    import random
    X_train, y_train = unpack(train, train_samples)
    X_test, y_test = unpack(test, test_samples)
    return X_train, y_train, X_test, y_test


def compute_metrics(y_true, y_pred) -> dict:
    tp = sum(int(a == 1 and b == 1) for a, b in zip(y_true, y_pred))
    tn = sum(int(a == 0 and b == 0) for a, b in zip(y_true, y_pred))
    fp = sum(int(a == 0 and b == 1) for a, b in zip(y_true, y_pred))
    fn = sum(int(a == 1 and b == 0) for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(y_true)
    return {"accuracy": (tp + tn) / total if total else 0.0, "precision": precision,
            "recall": recall, "f1": f1, "confusion_matrix": [[tn, fp], [fn, tp]]}


def save_metrics(name: str, metrics: dict, extra: dict | None = None):
    MODELS_DIR.mkdir(exist_ok=True)
    data = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    data[name] = {**metrics, **(extra or {})}
    METRICS_FILE.write_text(json.dumps(data, indent=2))
