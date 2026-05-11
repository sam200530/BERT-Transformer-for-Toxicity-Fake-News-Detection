# BERT-Based Fake News Detection

A from-scratch implementation of BERT (Bidirectional Encoder Representations from Transformers) for binary fake news classification using the **Fakeddit** dataset.

---

## Overview

This project has three parts:

| Task | What it does |
|------|-------------|
| **Task 1** | Builds BERT's core architecture from scratch using PyTorch |
| **Task 2** | Loads and preprocesses the Fakeddit dataset (posts + comments) |
| **Task 3** | Fine-tunes a pre-trained `bert-base-uncased` model for fake news classification |

---

## What is BERT? (Quick Primer)

BERT is a language model that reads text **bidirectionally** — it understands words in context of both what comes before *and* after them. It's built on the **Transformer** architecture, which uses a mechanism called **self-attention** to figure out how much each word in a sentence should "pay attention" to every other word.

---

## Architecture (Task 1 — Built from Scratch)

The custom BERT implementation stacks the following components:

```
Input Text
    ↓
Token Embeddings + Positional Embeddings
    ↓
Transformer Encoder Layers (×2)
    ├── Multi-Head Self-Attention
    ├── Layer Norm + Residual Connection
    ├── Feed-Forward Network (GELU activation)
    └── Layer Norm + Residual Connection
    ↓
[CLS] Token Pooling
    ↓
Classifier (Linear → 2 classes)
```

**Key hyperparameters:**
- Embedding dimension: 768
- Attention heads: 12
- Encoder layers: 2
- Max sequence length: 512
- Vocabulary size: from `bert-base-uncased` tokenizer (~30,522)

---

## Dataset — Fakeddit

[Fakeddit](https://github.com/entitize/Fakeddit) is a large-scale multimodal fake news dataset sourced from Reddit.

This project uses the **text-only** version with the `2_way_label` (binary: fake vs. non-fake).

### Data files expected:
```
nlp/
├── all_train.tsv
├── all_test_public.tsv
├── all_validate.tsv
└── all_comments.tsv
```

### Preprocessing pipeline (Task 2):

1. **Load** train / test / validate splits (TSV format)
2. **Select** relevant columns: `clean_title`, `2_way_label`, `id`
3. **Drop duplicates** based on post title
4. **Enrich with comments** — top 3 top-level comments appended using `[SEP]` tokens:
   ```
   [POST TITLE] [SEP] [COMMENT 1] [SEP] [COMMENT 2] [SEP] [COMMENT 3]
   ```
5. **Balance** the dataset — equal samples of fake and non-fake posts (min 2,500 each)
6. **Split** into 80% train / 20% validation

---

## Fine-Tuning (Task 3)

Uses HuggingFace's `BertForSequenceClassification` with `bert-base-uncased` weights.

### Training configuration:

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch size | 16 |
| Max token length | 128 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Scheduler | Linear warmup (100 steps) |
| Gradient clipping | 1.0 |

### Evaluation metrics (on test set):
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Requirements

```bash
pip install transformers torch pandas scikit-learn numpy tqdm
```

Python 3.8+ and a CUDA-capable GPU are recommended for training.

---

## How to Run

This notebook is designed for **Google Colab** with Google Drive mounted.

1. Upload the Fakeddit dataset files to `MyDrive/nlp/` and `MyDrive/all/`
2. Open `BERT_PRO__1_.ipynb` in Google Colab
3. Run cells sequentially — Task 1 → Task 2 → Task 3

---

## Project Structure

```
BERT_PRO__1_.ipynb   ← Main notebook (all 3 tasks)
README.md            ← This file
```

---

## Key Concepts Demonstrated

- **Multi-Head Self-Attention** — how Transformers relate each token to every other token
- **Positional Embeddings** — how BERT encodes word order (since attention has no built-in sense of position)
- **[CLS] token pooling** — using the first token's representation for classification
- **Transfer learning** — loading pre-trained weights and fine-tuning on a downstream task
- **Dataset balancing** — ensuring the model doesn't learn a class bias
- **Comment enrichment** — using social context (Reddit comments) to improve classification

---

## References

- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [Fakeddit Dataset (Nakamura et al., 2019)](https://arxiv.org/abs/1911.03854)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
