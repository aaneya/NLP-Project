# SentiBERT — Document Sentiment Analysis with BERT

NLP Capstone Project 10. Fine-tunes a pretrained DistilBERT sentiment model on IMDb reviews, compares it with a TF-IDF + Logistic Regression baseline, and serves predictions through a Streamlit app.

## Pipeline
1. Use the labeled IMDb train and test Parquet splits included in `data/`.
2. Strip HTML and normalize whitespace.
3. Train a TF-IDF (1–2 grams) + Logistic Regression baseline.
4. Fine-tune `distilbert-base-uncased-finetuned-sst-2-english` with `Trainer`.
5. Select checkpoints with a stratified validation slice from the training data; evaluate once on the separate 25,000-row test split.
6. Analyze pasted text or uploaded TXT/PDF documents in Streamlit.

## Improved app
- Interactive document probability trend with a 50% reference line and positive/negative regions.
- Confidence gauge, document metrics, and BERT/baseline comparisons.
- Benchmark charts and heatmap confusion matrices.
- Sidebar controls for neutral zone, tokens per window, overlap, maximum windows, and mean/median aggregation.
- Local trained-model status and clear guidance when the fallback model is active.

Long documents are processed in overlapping windows. The maximum-window setting limits runtime; text beyond that cap is truncated at the end. IMDb is binary, so Neutral/Mixed is a configurable confidence threshold, not a trained class. The model was trained on movie reviews and may perform differently on other domains. Scanned PDFs require OCR.

## Train on the included dataset
```bash
pip install -r requirements.txt
python train_baseline.py
python train_bert.py --epochs 2 --max_len 256 --batch_size 16 --gradient_accumulation_steps 1
streamlit run app.py
```

Training uses `data/train-00000-of-00001.parquet` and evaluates on `data/test-00000-of-00001.parquet`. The BERT trainer carves a stratified 10% validation set from training data for checkpoint selection, then reports final metrics once on the held-out test set. `--train_samples` and `--test_samples` can cap the splits for quick experiments; leave them at zero for the full data. The unlabeled split is not used for supervised training.

## Trained results

| Model | Accuracy | Precision | Recall | F1 | Test samples |
|---|---:|---:|---:|---:|---:|
| Fine-tuned DistilBERT | 91.16% | 88.69% | 94.35% | 91.43% | 25,000 |
| TF-IDF + Logistic Regression | 88.93% | 89.04% | 88.80% | 88.92% | 25,000 |

## Files
`app.py` — Streamlit interface and inference
`common.py` — text cleaning, local data loading, metrics
`train_baseline.py` — TF-IDF + Logistic Regression training
`train_bert.py` — DistilBERT fine-tuning and test evaluation
`models/bert/` — trained model and tokenizer
`models/baseline.joblib` — trained baseline pipeline
`models/metrics.json` — held-out metrics and confusion matrices
