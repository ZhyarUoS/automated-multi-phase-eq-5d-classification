# Automated Classification of EQ-5D Literature in PubMed

> **Note:** This repository accompanies a manuscript currently under review.  
> Please cite the repository if you use this code or data in your research.

---

## Overview

This repository provides the complete implementation of a multi-phase classification framework for automated screening of EQ-5D literature in PubMed using only article metadata (titles, abstracts, and keywords). The framework evaluates five pre-trained language models (PLMs) — BERT, SciBERT, BioBERT, PubMedBERT, and BioLinkBERT — across four experimental phases:

1. **Supervised Learning** — Training from scratch and fine-tuning with MLM domain-adaptive pre-training
2. **Semi-Supervised Learning** — Teacher-student pseudo-labeling with 2,000 unlabeled records
3. **Co-Training (without LLM)** — Curriculum-based pairwise PLM co-training with MC-Dropout uncertainty filtering
4. **LLM-Assisted Co-Training** — Three-way majority voting with GPT-4o-mini and Claude Haiku 4.5

All results are reported over five random seeds with mean F1 ± standard deviation and 95% confidence intervals.

---

## Repository Structure

```
├── data/
│   ├── eq-5d-200-records.csv               # 200 manually labeled PubMed records
│   └── eq-5d-2000-unique-random.csv        # 2,000 unlabeled PubMed records
│
├── pipelines/
│   ├── 1_training_from_scratch.ipynb          # Supervised training from scratch
│   ├── 2_fine_tuning.ipynb                    # MLM pre-training + fine-tuning
│   ├── 3_semi_supervised.ipynb                # Semi-supervised teacher-student learning
│   ├── 4_co_training.ipynb                    # Co-training without LLM
│   ├── 5_gpt_co_training.ipynb                # Co-training with GPT-4o-mini
│   └── 6_claude_co_training.ipynb             # Co-training with Claude Haiku 4.5
│
├── results/                                # Output directory (auto-created)
├── cache/                                  # LLM response cache (auto-created)
├── requirements.txt
└── README.md
```

---

## Requirements

### Hardware
- GPU with at least 16GB VRAM (recommended: NVIDIA A100 or equivalent)
- At least 32GB RAM

### Software

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:

```
torch>=2.0.0
transformers>=4.35.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
openai>=1.0.0
anthropic>=0.20.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## Dataset

The dataset is located in the `data/` folder:

| File | Records | Description |
|------|---------|-------------|
| `eq-5d-200-records.csv` | 200 | Manually labeled PubMed records (Label: 0 or 1) |
| `eq-5d-2000-unique-random.csv` | 2,000 | Unlabeled PubMed records for pseudo-labeling |

Each record contains:

| Column | Description |
|--------|-------------|
| `Title` | Article title |
| `Abstract` | Article abstract |
| `Keywords` | Article keywords |
| `Label` | Binary label — 1 = EQ-5D reported, 0 = not reported *(labeled dataset only)* |

---

## API Keys Setup

### GPT-4o-mini (OpenAI)
Required for **Pipeline 5** only (`5_gpt_co_training.py`).

Open the file and replace the placeholder:
```python
client = OpenAI(api_key="your-openai-api-key-here")
```

Obtain your key at: https://platform.openai.com/api-keys

### Claude Haiku 4.5 (Anthropic)
Required for **Pipeline 6** only (`6_claude_co_training.py`).

Open the file and replace the placeholder:
```python
client = Anthropic(api_key="your-anthropic-api-key-here")
```

Obtain your key at: https://console.anthropic.com/

> **Security reminder:** Never commit API keys to a public repository.  
> Use environment variables instead:
> ```python
> import os
> client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
> client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
> ```

---

## Model Selection

All pipelines support the following models. Set `selected_model` (Pipelines 1–3) or `MODEL_1_NAME` / `MODEL_2_NAME` (Pipelines 4–6) at the top of each script:

| Key | Model | HuggingFace ID |
|-----|-------|----------------|
| `bert` | BERT | `bert-base-uncased` |
| `scibert` | SciBERT | `allenai/scibert_scivocab_uncased` |
| `biobert` | BioBERT | `dmis-lab/biobert-base-cased-v1.2` |
| `pubmedbert_base` | PubMedBERT | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` |
| `biolinkbert_base` | BioLinkBERT Base | `michiyasunaga/BioLinkBERT-base` |
| `biolinkbert_large` | BioLinkBERT Large | `michiyasunaga/BioLinkBERT-large` |

---

## User Preferences

Each pipeline contains the following toggleable flags at the top of the file:

```python
VISUALIZE_CLASS_DISTRIBUTION = True   # Show class distribution bar chart
USE_MIXED_PRECISION           = True   # Enable mixed precision (fp16) training
USE_LR_SCHEDULER              = True   # Enable linear scheduler with 10% warm-up
```

Set any flag to `False` to disable the corresponding feature.

---

## Running the Pipelines

### Pipeline 1 — Training from Scratch

Trains all models from randomly initialized weights without any pre-trained knowledge. Used as the baseline supervised phase.

**Before running — set in the script:**
```python
selected_model = "bert"          # Options: bert, scibert, biobert, pubmedbert_base,
                                 #          biolinkbert_base, biolinkbert_large
selected_input = "training_scratch_combined"
CLS_EPOCHS     = 100
PATIENCE       = 10
```

**Run:**
```bash
python pipelines/1_training_from_scratch.py
```

**Output:**
- Per-seed F1 scores and accuracy
- Mean F1 ± Std and 95% confidence intervals
- Classification report and confusion matrix
- Error analysis (false positives and false negatives)
- Results CSV saved to `results/`

---

### Pipeline 2 — Fine-Tuning

Loads pre-trained transformer checkpoints, applies unsupervised MLM domain-adaptive pre-training on 200 unlabeled records, then fine-tunes for EQ-5D binary classification. Evaluates all five input configurations (title, abstract, keywords, TAK, TA).

**Before running — set in the script:**
```python
selected_model = "biolinkbert_base"   # Options: same as Pipeline 1
USE_MLM        = True                 # Set False to skip MLM pre-training
MLM_EPOCHS     = 15
CLS_EPOCHS     = 100
PATIENCE       = 10
```

**Run:**
```bash
python pipelines/2_fine_tuning.py
```

**Output:**
- MLM pre-trained weights saved to `results/`
- Per-seed F1 scores and accuracy
- Mean F1 ± Std and 95% confidence intervals
- Classification report and confusion matrix
- Error analysis (false positives and false negatives)
- Results CSV saved to `results/`

---

### Pipeline 3 — Semi-Supervised Learning

Implements a teacher-student pseudo-labeling approach. The teacher model generates soft pseudo-labels for 2,000 unlabeled records. High-confidence samples (τ = 0.90, max 250 per class) are used to train a student model via a joint cross-entropy and KL-divergence loss.

**Before running — set in the script:**
```python
selected_model       = "biolinkbert_base"   # Options: same as Pipeline 1
MLM_EPOCHS           = 15
TEACHER_EPOCHS       = 25
STUDENT_EPOCHS       = 35
LR_LIST              = [1e-5, 2e-5, 3e-5, 5e-5]
CONF_THRESHOLD       = 0.90
MAX_PSEUDO_PER_CLASS = 250
UNSUP_WEIGHT         = 0.10
```

**Run:**
```bash
python pipelines/3_semi_supervised.py
```

**Output:**
- Unlabeled predictions with confidence scores saved to `results/`
- Per-seed F1 scores and accuracy
- Mean F1 ± Std and 95% confidence intervals
- Classification report
- Best student model saved to `results/`

---

### Pipeline 4 — Co-Training (Without LLM)

Implements curriculum-based pairwise PLM co-training across all 10 model combinations. At each of three iterations, both models in a pair generate pseudo-labels for unlabeled records using MC-Dropout uncertainty filtering. Only samples where both models agree, both exceed the confidence threshold, and both exhibit low predictive variance are retained.

**Before running — set in the script:**
```python
MODEL_1_NAME          = "bert"      # First model in the pair
MODEL_2_NAME          = "scibert"   # Second model in the pair
MLM_EPOCHS            = 15
ITERATIONS            = 3
EPOCH_LIST            = [25, 20, 15]
THRESHOLDS            = [0.95, 0.92, 0.90]
MAX_PSEUDO_LIST       = [100, 150, 200]
CONSISTENCY_WEIGHT    = 0.2
MC_DROPOUT_PASSES     = 8
UNCERTAINTY_THRESHOLD = 0.05
LR_LIST               = [1e-5, 2e-5, 3e-5, 5e-5]
```

**Run each pair separately by updating MODEL_1_NAME and MODEL_2_NAME.**

**All supported pairs (10 total):**

| Pair | | Pair |
|------|-|------|
| BERT ↔ SciBERT | | SciBERT ↔ BioLinkBERT |
| BERT ↔ BioBERT | | BioBERT ↔ PubMedBERT |
| BERT ↔ PubMedBERT | | BioBERT ↔ BioLinkBERT |
| BERT ↔ BioLinkBERT | | PubMedBERT ↔ BioLinkBERT |
| SciBERT ↔ BioBERT | | SciBERT ↔ PubMedBERT |

**Run:**
```bash
python pipelines/4_co_training.py
```

**Output:**
- Ensemble pseudo-labels with probabilities saved to `results/`
- Per-seed and per-LR F1 scores and accuracy
- Mean F1 ± Std and 95% confidence intervals
- Best ensemble model saved to `results/`

---

### Pipeline 5 — GPT-Assisted Co-Training

Extends Pipeline 4 by adding GPT-4o-mini as a third annotator. A pseudo-label is assigned only when at least 2 out of 3 annotators (Model 1, Model 2, GPT) agree, and the average confidence across all three exceeds the threshold τ.

> **Requires:** OpenAI API key — see API Keys Setup above.

**Before running — set in the script:**
```python
MODEL_1_NAME = "bert"       # Same pair options as Pipeline 4
MODEL_2_NAME = "scibert"
CACHE_PATH   = "cache/gpt_cache.json"
```

> **Cost note:** The estimated first-run annotation cost for 2,000 records is approximately **$0.1559** for GPT-4o-mini. All subsequent runs are served from the MD5-based cache at no additional cost.

**Run:**
```bash
python pipelines/5_gpt_co_training.py
```

**Output:**
- Same as Pipeline 4
- GPT response cache saved to `cache/gpt_cache.json`
- Ensemble pseudo-labels saved to `results/`

---

### Pipeline 6 — Claude-Assisted Co-Training

Extends Pipeline 4 by adding Claude Haiku 4.5 as a third annotator. Uses the same three-way majority voting and average confidence thresholding as Pipeline 5.

> **Requires:** Anthropic API key — see API Keys Setup above.

**Before running — set in the script:**
```python
MODEL_1_NAME = "bert"            # Same pair options as Pipeline 4
MODEL_2_NAME = "pubmedbert_base"
CACHE_PATH   = "cache/claude_cache.json"
```

> **Cost note:** The estimated first-run annotation cost for 2,000 records is approximately **$1.1780** for Claude Haiku 4.5. All subsequent runs are served from the MD5-based cache at no additional cost.

**Run:**
```bash
python pipelines/6_claude_co_training.py
```

**Output:**
- Same as Pipeline 4
- Claude response cache saved to `cache/claude_cache.json`
- Ensemble pseudo-labels saved to `results/`

---

## Reproducibility

All experiments use five fixed random seeds:

```python
SEEDS = [42, 123, 2023, 777, 999]
```

All reported results correspond to the mean ± standard deviation across these five seeds, with 95% confidence intervals computed via bootstrap resampling (n=1,000).

For LLM-assisted pipelines, reproducibility is further ensured by:
- Setting `temperature=0` for all API calls
- MD5-based response caching — identical inputs always produce identical outputs across all seeds and co-training iterations

---



## Notes

- All pipelines are designed to run on **Google Colab** with GPU support. Minor path adjustments may be needed for local execution.
- Update the data paths in each script if running locally:
  ```python
  labeled   = pd.read_csv("data/eq-5d-200-records.csv")
  unlabeled = pd.read_csv("data/eq-5d-2000-unique-random.csv")
  ```
- Mixed precision training (`USE_MIXED_PRECISION = True`) requires a CUDA-enabled GPU. Set to `False` for CPU-only execution.
- For co-training pipelines (4, 5, 6), run each of the 10 pairs separately by updating `MODEL_1_NAME` and `MODEL_2_NAME` at the top of the script.
- LLM response caches (`gpt_cache.json`, `claude_cache.json`) are stored in the `cache/` folder and persist across runs. Delete the cache file to force fresh API calls.

---


## Contact

For questions, issues, or suggestions, please open a GitHub issue or contact:

**Zhyar Rzgar K Rostam**  
Doctoral School of Applied Informatics and Applied Mathematics  
Obuda University, Budapest, Hungary  
Email: kwekha.rostam.zhyar@nik.uni-obuda.hu
