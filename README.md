---
title: Protein Secondary Structure Predictor
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: streamlit
python_version: 3.10
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# 🧬 Protein Secondary Structure Predictor

A full end-to-end deep learning pipeline for predicting protein secondary structures — **α-Helix (H)**, **β-Sheet (E)**, and **Coil (C)** — from raw amino acid sequences. The project uses **ProtT5-XL-UniRef50** as a protein language model backbone to produce per-residue embeddings, which are then classified by a custom architecture combining **1D-CNNs**, **BiLSTMs**, and **Multi-Head Attention**.

[![Live App](https://img.shields.io/badge/🤗%20Live%20App-HF%20Spaces-blue)](https://huggingface.co/spaces/Chimera418/protein-ssp)
[![Model Artifacts](https://img.shields.io/badge/🤗%20Artifacts-HF%20Model%20Hub-orange)](https://huggingface.co/Chimera418/protein-ssp-artifacts)
[![GitHub](https://img.shields.io/badge/Source-GitHub-black)](https://github.com/Chimera418/protein-ssp)

---

## 🌟 Key Features

- **ProtT5 Backbone**: Uses `Rostlab/prot_t5_xl_uniref50` — a large protein language model — to generate rich per-residue context-aware embeddings (1024-dim).
- **Five Prediction Modes**: Choose between five different feature engineering pipelines, each with its own trained model.
- **Custom Architecture**: 1D-CNN (local motif detection) → BiLSTM (sequence context) → Multi-Head Attention (global dependencies) → Linear head (3 classes).
- **Interactive Streamlit UI**: Real-time per-residue predictions, colour-coded sequence rendering, confidence charts, composition pie chart, and CSV export.
- **Decoupled Artifact Storage**: Model weights are hosted separately on HF Model Hub and downloaded on-demand at runtime.

---

## 🎯 Prediction Modes & Performance

All metrics are evaluated on a **held-out test set of proteins** not seen during training.

| Mode | Input Dims | Q3 Accuracy | Macro F1 | AUC | Notes |
|---|---|---|---|---|---|
| **Direct** | 1024 | 85.03% | 0.8494 | 0.9685 | Raw ProtT5, no reduction |
| **Pearson-filtered** | 1017 | 85.41% | 0.8530 | 0.9683 | Pearson correlation filter only |
| **PCA Pipeline** ★ | 739 | **85.67%** | **0.8563** | **0.9692** | Pearson → PCA — best performance |
| **Feature Selected V1** | 109 | 84.21% | 0.8413 | 0.9617 | Pearson → PCA → ExtraTrees (109 dims) |
| **Feature Selected V2** | 12 | 82.30% | 0.8215 | 0.9600 | Pearson → PCA → ExtraTrees top-12 |

> ★ **PCA Pipeline (739-dim)** is the default and recommended mode.

---

## 🚀 Live Demo

The Streamlit app is live on Hugging Face Spaces:

👉 **[Try the Live App Here](https://huggingface.co/spaces/Chimera418/protein-ssp)**

### How to use it
1. Open the app. The sidebar lets you select a prediction mode.
2. Paste any amino acid sequence in single-letter code (e.g. `MVLSPADKTNVK...`), or click **📋 Sample** to load a human haemoglobin example.
3. Click **🔮 Predict**.
4. The app will:
   - Encode your sequence using **ProtT5** (~2 min on first run while the model downloads)
   - Apply the feature pipeline for the selected mode
   - Run the selected deep learning model
   - Display a colour-coded annotated sequence, per-residue confidence chart, and composition breakdown
5. Download results as CSV using the **⬇️ Download Predictions** button.

> **Note**: On first launch, ProtT5 (~3 GB) downloads automatically from Hugging Face. Model weights (~300 MB) also download from the artifacts repo on first prediction. Subsequent runs use the cached files.

---

## 🏗️ Storage Architecture

Because the trained models, embeddings, and datasets are too large for a standard Git repository (~30 GB total), the project uses a **decoupled two-repository architecture**:

| Component | Location |
|---|---|
| Application code (`app.py`, `phase_*.py`) | [GitHub](https://github.com/Chimera418/protein-ssp) → synced to [HF Space](https://huggingface.co/spaces/Chimera418/protein-ssp) |
| Model weights & pkl artifacts | [`Chimera418/protein-ssp-artifacts/models/`](https://huggingface.co/Chimera418/protein-ssp-artifacts) |
| Pre-computed embeddings (~26.7 GB) | [`Chimera418/protein-ssp-artifacts/embeddings/`](https://huggingface.co/Chimera418/protein-ssp-artifacts) |
| Curated dataset CSVs (~1.1 GB) | [`Chimera418/protein-ssp-artifacts/data/`](https://huggingface.co/Chimera418/protein-ssp-artifacts) |
| Raw RCSB/PISCES source data (~2.2 GB) | [`Chimera418/protein-ssp-artifacts/raw_data/`](https://huggingface.co/Chimera418/protein-ssp-artifacts) |

The app's `ensure_model_exists()` function lazily pulls model files from HF Hub at runtime if they are not already present locally.

---

## 💻 Local Installation

### Prerequisites
- Python 3.9+
- ~5 GB free disk space (ProtT5 + model weights)
- GPU recommended for fast embedding generation; CPU works but is slow

### Steps

**1. Clone the repository:**
```bash
git clone https://github.com/Chimera418/protein-ssp.git
cd protein-ssp
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **First run**: ProtT5-XL-UniRef50 (~3 GB) downloads automatically from Hugging Face when you first run a prediction. The five trained model `.pt` files (~300 MB total) are also pulled on-demand from `Chimera418/protein-ssp-artifacts` on first prediction per mode, then cached locally under `models/`.

### Optional: Pre-download all model weights
To avoid waiting for downloads during the first prediction, you can pre-fetch all model files using:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Chimera418/protein-ssp-artifacts", allow_patterns="models/*", local_dir=".")
```

---

## 🔧 Run the Full Pipeline Yourself (Train From Scratch)

You are not required to use the pre-trained models or artifacts from the HF Hub. All the phase scripts are included in the repository — you can run the entire pipeline end-to-end on your own machine to generate your own data, embeddings, and trained models from scratch.

> **Hardware note**: Phase 4 (embedding generation for ~9,000 proteins) and Phase 8 (model training) benefit strongly from a GPU. On CPU, Phase 4 alone can take many hours.

### Full pipeline execution order

```bash
# Phase 1 — Download RCSB PDB sequences and parse raw CSV
python phase_1_raw_sequences.py

# Phase 2 — Deduplicate, filter, and remove redundant sequences via PISCES
python phase_2_data_curate.py

# Phase 3 — Match per-residue SST8/SST3 labels from RCSB to curated sequences
python phase_3_labelled_curated_sequence.py

# Phase 4 — Benchmark 4 protein LLMs, select best (ProtT5), generate full embeddings
# ⚠️ Requires significant RAM + GPU. Output: embeddings/Rostlab_prot_t5_xl_uniref50.pkl (~9 GB)
python phase_4_embedding_generation.py

# Phase 5 — Pearson correlation filter: 1024 → 1017 dims, saves keep_indices.pkl
python phase_5_feature_filtering.py

# Phase 6 — PCA: 1017 → 739 dims, saves pca_model.pkl
python phase_6_dimensionality_reduction.py

# Phase 7 — ExtraTrees feature selection: 739 → 109 dims, saves feature_selector_mask.pkl
python phase_7_feature_selection.py

# Phase 7.5 — Second-pass refinement: 109 → 12 dims, saves feature_selector_mask_v2.pkl
python phase_7_5_feature_refinement.py

# Phase 8 — Train all 5 CNN+BiLSTM+Attention models (one per feature space)
# Output: models/phase_8_best_model_*.pt
python phase_8_deep_learning_model.py

# Phase 9 (optional) — SHAP explainability analysis
python phase_9_explainable_ai.py
```

After running all phases, your `models/`, `embeddings/`, `data/`, and `raw_data/` directories will be fully populated with your own generated artifacts. You can then run the Streamlit app normally:

```bash
streamlit run app.py
```

The app detects locally present model files and skips the HF Hub download entirely.

### Using the CLI predictor (no Streamlit)

If you just want predictions from the command line without opening the web UI:

```bash
python phase_10_predict.py --sequence "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFD"
```

---



```
protein-ssp/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
│
├── phase_1_raw_sequences.py        # Data collection pipeline
├── phase_2_data_curate.py          # Dataset curation & redundancy removal
├── phase_3_labelled_curated_sequence.py  # Secondary structure labelling
├── phase_4_embedding_generation.py # ProtT5 embedding generation & model benchmarking
├── phase_5_feature_filtering.py    # Pearson correlation feature filtering
├── phase_6_dimensionality_reduction.py   # PCA dimensionality reduction
├── phase_7_feature_selection.py    # ExtraTrees feature importance selection
├── phase_7_5_feature_refinement.py # Second-pass top-12 feature refinement
├── phase_8_deep_learning_model.py  # CNN + BiLSTM + Attention model training
├── phase_9_explainable_ai.py       # SHAP/XAI analysis of model decisions
├── phase_10_predict.py             # CLI inference script (no Streamlit)
│
├── models/                         # Trained weights (git-ignored; downloaded from HF Hub)
├── embeddings/                     # Pre-computed embeddings (git-ignored; on HF Hub)
├── data/                           # Curated CSVs (git-ignored; on HF Hub)
└── raw_data/                       # Raw RCSB/PISCES downloads (git-ignored; on HF Hub)
```

---

## 🔬 Pipeline Description

The project is structured as a sequential 10-phase pipeline:

| Phase | Script | What it does |
|---|---|---|
| **1** | `phase_1_raw_sequences.py` | Downloads the RCSB PDB `ss.txt.gz` file and organism index (`source.idx`), parses all protein sequences, and exports `protein_sequences_raw.csv` |
| **2** | `phase_2_data_curate.py` | Cleans the raw dataset by removing duplicates, filtering by length (40–10,000 aa), stripping invalid amino acids, and performing redundancy removal using the PISCES culled list (≤70% sequence identity) |
| **3** | `phase_3_labelled_curated_sequence.py` | Parses per-residue 8-class secondary structure labels (SST8) from RCSB, maps them to the curated sequences, and converts them to 3-class labels (SST3: H/E/C) using standard DSSP mapping |
| **4** | `phase_4_embedding_generation.py` | Benchmarks four protein language models (ProtT5, ProtBERT, ProtALBERT, DistilProtBERT) on a 100-sequence subset using 5-fold stratified cross-validation, selects the best model (ProtT5), then generates full per-residue embeddings (1024-dim) for all ~9,000 sequences |
| **5** | `phase_5_feature_filtering.py` | Computes per-feature Pearson correlation with the secondary structure label, removes near-zero-variance and highly correlated dimensions, reducing 1024 → 1017 features, and saves `keep_indices.pkl` |
| **6** | `phase_6_dimensionality_reduction.py` | Fits a PCA transformation on the Pearson-filtered embeddings, retaining 739 principal components (99% variance explained), and saves `pca_model.pkl` |
| **7** | `phase_7_feature_selection.py` | Trains an ExtraTrees classifier in the PCA space and uses Boruta-style feature importance to select the top 109 discriminative PCA components, saving `feature_selector_mask.pkl` |
| **7.5** | `phase_7_5_feature_refinement.py` | Applies a second hard-capped ExtraTrees pass on the 109-feature space, composing both masks to select the top 12 most informative features, saving `feature_selector_mask_v2.pkl` |
| **8** | `phase_8_deep_learning_model.py` | Trains five separate CNN+BiLSTM+Attention models — one for each of the five feature spaces — with early stopping, saving the best checkpoint per mode as `.pt` files |
| **9** | `phase_9_explainable_ai.py` | Generates SHAP explainability reports, attention weight visualisations, and per-residue importance scores to interpret model decisions |
| **10** | `phase_10_predict.py` | Standalone CLI inference script that accepts a raw sequence and returns per-residue predictions without the Streamlit interface |

---

## 🛠️ Built With

- **[PyTorch](https://pytorch.org/)** — Deep learning model architecture and training
- **[Hugging Face Transformers](https://huggingface.co/docs/transformers)** — ProtT5-XL-UniRef50 protein language model
- **[Streamlit](https://streamlit.io/)** — Interactive web application
- **[Scikit-Learn](https://scikit-learn.org/)** — PCA, ExtraTrees, cross-validation
- **[Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)** — Artifact storage and on-demand model downloads
- **[Matplotlib](https://matplotlib.org/)** — Confidence and composition visualisations

---

## 📜 License

This project is licensed under the **MIT License**.
