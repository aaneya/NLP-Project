"""Fine-tune a pretrained BERT model for sentiment using local IMDb Parquet splits.

Use a GPU (e.g. free Google Colab T4). Example:
    python train_bert.py --train_file data/train-00000-of-00001.parquet \
        --test_file data/test-00000-of-00001.parquet --epochs 2
"""
import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from common import MODELS_DIR, compute_metrics, load_parquet_splits, save_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
ap.add_argument("--train_file", default="data/train-00000-of-00001.parquet")
ap.add_argument("--test_file", default="data/test-00000-of-00001.parquet")
ap.add_argument("--train_samples", type=int, default=0, help="Optional cap; 0 uses the full training split")
ap.add_argument("--test_samples", type=int, default=0, help="Optional cap; 0 uses the full test split")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lr", type=float, default=2e-5)
args = ap.parse_args()


class TextDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


X_all, y_all, X_te, y_te = load_parquet_splits(
    args.train_file, args.test_file, args.train_samples, args.test_samples)
val_rng = random.Random(args.seed)
val_indices = set()
for label in sorted(set(y_all)):
    class_indices = [i for i, value in enumerate(y_all) if value == label]
    val_rng.shuffle(class_indices)
    val_indices.update(class_indices[:max(1, round(len(class_indices) * 0.10))])
train_indices = [i for i in range(len(y_all)) if i not in val_indices]
val_indices = sorted(val_indices)
X_tr, y_tr = [X_all[i] for i in train_indices], [y_all[i] for i in train_indices]
X_val, y_val = [X_all[i] for i in val_indices], [y_all[i] for i in val_indices]
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1})

train_ds = TextDataset(X_tr, y_tr, tok, args.max_len)
val_ds = TextDataset(X_val, y_val, tok, args.max_len)
test_ds = TextDataset(X_te, y_te, tok, args.max_len)

trainer = Trainer(
    model=model, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok,
    args=TrainingArguments(
        output_dir="models/_runs", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16, gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=torch.cuda.is_available() and "distilbert" not in args.model.lower(), learning_rate=args.lr,
        weight_decay=0.01, warmup_ratio=0.06, logging_steps=100,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=args.seed, data_seed=args.seed,
        fp16=torch.cuda.is_available(), optim="adamw_torch",
        dataloader_num_workers=0, logging_dir="models/_logs"))

t0 = time.time()
trainer.train()
train_time = time.time() - t0

preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
metrics = compute_metrics(y_te, preds)
print({k: round(v, 4) for k, v in metrics.items() if k != "confusion_matrix"})

trainer.save_model(str(MODELS_DIR / "bert"))
tok.save_pretrained(str(MODELS_DIR / "bert"))
save_metrics(f"Fine-tuned {args.model}", metrics,
             {"train_samples": len(X_tr), "validation_samples": len(X_val), "test_samples": len(X_te),
              "epochs": args.epochs, "train_seconds": round(train_time, 1),
              "max_len": args.max_len, "seed": args.seed, "source": "local parquet splits"})
print("Saved model to models/bert")
