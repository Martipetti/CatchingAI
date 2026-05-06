# CatchingAI

A binary text classifier that distinguishes **human-written** from **AI-generated** text, fine-tuned on the [RAID benchmark](https://raid-bench.xyz) using LoRA (Low-Rank Adaptation).

---

## Overview

As large language models become increasingly capable of producing human-like text, the ability to detect AI-generated content is a growing challenge in academia, journalism, and content moderation.

CatchingAI addresses this by fine-tuning `bert-base-cased` on RAID — the largest and most rigorous benchmark for machine-generated text detection — using parameter-efficient training that runs on consumer hardware and Google Colab.

---

## Dataset — RAID

[RAID](https://raid-bench.xyz) (**R**obust **A**I-generated text **D**etection) was introduced at ACL 2024 and is the most comprehensive benchmark for this task.

| Property | Value |
|---|---|
| Total documents | 10M+ |
| Source LLMs | 11 (GPT-4, GPT-3, ChatGPT, GPT-2, LLaMA, Mistral, ...) |
| Text domains | 8 (news, abstracts, reviews, Wikipedia, ...) |
| Decoding strategies | 4 |
| Adversarial attacks | 11 |

The dataset is loaded via the `raid-bench` Python package, which handles downloading and caching automatically:

```python
from raid.utils import load_data

train_df = load_data("train", include_adversarial=True)
```

Labels are derived from the `model` column: `"human"` → 0, any LLM → 1.

---

## Approach

### Model

Base model: `bert-base-cased` (110M parameters, 12 transformer layers).

A classification head (`Linear(768 → 2)`) is added on top of the `[CLS]` token representation and fine-tuned from scratch.

### Fine-tuning with LoRA

Instead of updating all 110M parameters, [LoRA](https://arxiv.org/abs/2106.09685) injects small trainable rank-decomposition matrices into BERT's linear layers. Only ~1% of parameters are trained.

```
BERT-base-cased (frozen)
    └── 12 Transformer layers
    └── LoRA adapters (trained, r=16)         ← ~1% of params
    └── Pooler

Classifier Head (trained from scratch)
    └── Dropout → Linear(768 → 2)
```

LoRA configuration:

```python
LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    use_rslora=True,
    modules_to_save=["classifier"],
)
```

### Evaluation

Training uses standard accuracy and F1. Final evaluation uses the **official RAID metric**: accuracy at a controlled **5% false-positive rate** on human text, computed via `raid.run_detection` and `raid.run_evaluation`.

---

## Installation

Requires Python 3.11 (see note below).

```bash
# Clone the repo
git clone https://github.com/<your-username>/catchingai.git
cd catchingai

# Create a virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install raid-bench transformers peft evaluate accelerate datasets torch
```

> **Note on Python version.** `raid-bench` pins `scikit-learn~=1.3.2`, which does not support Python 3.12+ (Cython compilation fails). Use Python 3.11 to avoid build errors. If you are on a newer Python, install dependencies manually:
> ```bash
> pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2
> pip install raid-bench --no-deps
> ```

---

## Usage

### Training

```bash
python main.py
```

Key hyperparameters (edit at the top of `main.py`):

| Parameter | Default | Notes |
|---|---|---|
| `CHECKPOINT` | `bert-base-cased` | Base model |
| `EPOCHS` | `3` | Training epochs |
| `LR` | `5e-5` | Learning rate |
| `TRAIN_BS` | `8` | Per-device batch size (reduce to 4 on Colab T4) |
| `GRAD_ACC` | `4` | Gradient accumulation steps |
| `include_adversarial` | `True` | Include adversarially rephrased examples |

### Google Colab

The training script runs on Colab with a free T4 GPU. Set `TRAIN_BS=4` and `fp16=True`. Estimated training time: ~2–4 hours depending on dataset size.

### Evaluation

After training, the script automatically runs both evaluations:

```
=== HuggingFace eval (accuracy + F1) ===
{'eval_accuracy': 0.94, 'eval_f1': 0.93, ...}

=== RAID eval (accuracy @ 5% FPR) ===
{'scores': {...}, 'thresholds': {...}, 'fpr': {...}}
```

---

## Project Structure

```
catchingai/
├── main.py          # Training and evaluation script
├── README.md
└── models/
    └── raid-bert-classifier/
        └── final/   # Saved model weights and tokenizer
```

---

## Tech Stack

- [PyTorch](https://pytorch.org)
- [Transformers](https://huggingface.co/docs/transformers) — model and training
- [PEFT](https://huggingface.co/docs/peft) — LoRA fine-tuning
- [raid-bench](https://github.com/liamdugan/raid) — dataset and official evaluation
- [datasets](https://huggingface.co/docs/datasets) — data pipeline
- [evaluate](https://huggingface.co/docs/evaluate) — metrics

---

## References

- Dugan et al., [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](https://aclanthology.org/2024.acl-long.674), ACL 2024
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022

---

## License

MIT
