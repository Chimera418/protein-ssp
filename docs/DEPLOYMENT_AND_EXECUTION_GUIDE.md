# 🧬 Protein Secondary Structure Prediction: Comprehensive Setup, Deployment & Execution Guide

This document is the definitive guide on how to clone, configure, connect artifacts, and run the **Protein Secondary Structure Predictor** both **in the cloud on Hugging Face Spaces** and **locally on a personal computer (Windows / macOS / Linux)**.

---

## Table of Contents
1. [Architecture & Storage Topology](#1-architecture--storage-topology)
2. [Method 1: Run Directly in the Cloud (Zero-Install)](#2-method-1-run-directly-in-the-cloud-zero-install)
3. [Method 2: Local Installation & Setup (PC / Workstation)](#3-method-2-local-installation--setup-pc--workstation)
   - [Prerequisites](#prerequisites)
   - [Step 1: Clone the Code Repository](#step-1-clone-the-code-repository)
   - [Step 2: Create a Python 3.10 Virtual Environment](#step-2-create-a-python-310-virtual-environment)
   - [Step 3: Install PyTorch & Dependencies](#step-3-install-pytorch--dependencies)
4. [Connecting and Managing Artifacts](#4-connecting-and-managing-artifacts)
   - [Strategy A: Automatic Lazy-Load (Recommended)](#strategy-a-automatic-lazy-load-recommended)
   - [Strategy B: Pre-downloading Model Weights for Offline Use](#strategy-b-pre-downloading-model-weights-for-offline-use)
   - [Strategy C: Downloading Pre-computed Embeddings & Curated Data](#strategy-c-downloading-pre-computed-embeddings--curated-data)
5. [Running the Predictor](#5-running-the-predictor)
   - [Interactive Web App (Streamlit Dashboard)](#interactive-web-app-streamlit-dashboard)
   - [Standalone Command-Line Predictor (CLI)](#standalone-command-line-predictor-cli)
6. [Reproducing the Full 10-Phase Pipeline from Scratch](#6-reproducing-the-full-10-phase-pipeline-from-scratch)
7. [Uploading and Synchronizing Artifacts back to Hugging Face](#7-uploading-and-synchronizing-artifacts-back-to-hugging-face)
8. [Troubleshooting & Frequently Asked Questions (FAQ)](#8-troubleshooting--frequently-asked-questions-faq)

---

## 1. Architecture & Storage Topology

Deep learning models for structural bioinformatics generate large artifacts (language model backbones, dense per-residue embeddings, covariance matrices, and neural network checkpoints). To prevent git bloat and ensure fast cloning, this project employs a **decoupled two-tier architecture**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: CODE REPOSITORY                         │
│  GitHub: https://github.com/Chimera418/protein-ssp                     │
│  HF Space: https://huggingface.co/spaces/Chimera418/protein-ssp        │
│  Contents: Python pipeline scripts, app.py, docs, requirements (<15 MB)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ pulls weights at runtime
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      TIER 2: ARTIFACT MODEL HUB                        │
│  HF Model Hub: https://huggingface.co/Chimera418/protein-ssp-artifacts │
│  Contents:                                                             │
│    • models/      : 5 trained PyTorch weights (.pt) + PCA / mask (.pkl)│
│    • data/        : Curated sequence CSVs & residue datasets           │
│    • embeddings/  : Pre-computed ProtT5 embeddings (~26.7 GB)          │
│    • output/      : Phase 1–9 evaluation metrics, ROC, & XAI plots     │
│    • raw_data/    : wwPDB ss.txt.gz & PISCES culled sequence lists     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Method 1: Run Directly in the Cloud (Zero-Install)

If you do not want to set up Python, install PyTorch, or configure GPU drivers locally, you can run the entire interactive dashboard directly in your web browser.

### Live URL
👉 **[https://huggingface.co/spaces/Chimera418/protein-ssp](https://huggingface.co/spaces/Chimera418/protein-ssp)**

### How It Works Under the Hood
1. Hugging Face Spaces boots a cloud container running Debian Linux with Python 3.10 and Streamlit.
2. The web application dynamically connects to `Chimera418/protein-ssp-artifacts`.
3. When you submit a protein sequence:
   - It lazily loads the pre-trained `ProtT5-XL-UniRef50` encoder.
   - It downloads only the specific model weights (`models/phase_8_best_model_*.pt`) and transforms required for your chosen prediction mode into the container's cache.
   - It renders secondary structure predictions (**Helix**, **Strand**, **Coil**), confidence bar charts, amino-acid composition distributions, and downloadable CSV exports.

---

## 3. Method 2: Local Installation & Setup (PC / Workstation)

Running locally gives you access to the CLI prediction tool, full training pipeline, custom feature engineering, and offline inference.

### Prerequisites
* **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS (Apple Silicon / Intel).
* **Python**: **Python 3.10 is strongly recommended** (Python 3.9–3.11 supported; avoid Python 3.13/3.14 due to binary C-extension and PyTorch compatibility).
* **Disk Space**:
  * Minimum for inference: **~6 GB** (ProtT5-XL language model cache + PyTorch model weights).
  * Full dataset & embeddings: **~35 GB** (if downloading pre-computed embeddings).
* **Hardware**:
  * **GPU (Recommended)**: NVIDIA GeForce GTX 1060+ / RTX Series with CUDA 11.8 or 12.4.
  * **CPU**: Multi-core x86_64 CPU (works for inference; sequence embedding takes ~1–2 minutes per sequence on CPU).

---

### Step 1: Clone the Code Repository

Open PowerShell (Windows) or Terminal (Linux/macOS):

```bash
# Clone the repository
git clone https://github.com/Chimera418/protein-ssp.git

# Enter the project directory
cd protein-ssp
```

*(Alternatively, you can clone directly from the Hugging Face Space: `git clone https://huggingface.co/spaces/Chimera418/protein-ssp`)*

---

### Step 2: Create a Python 3.10 Virtual Environment

Isolate project packages to avoid system-level package conflicts:

#### **On Windows (PowerShell):**
```powershell
# Using Python 3.10 explicitly
py -3.10 -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an `Execution_Policies` restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first)*.

#### **On Linux / macOS:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install PyTorch & Dependencies

#### 1. Install PyTorch with GPU Support (CUDA):
Visit [pytorch.org](https://pytorch.org/get-started/locally/) or run the standard CUDA 12.4 build:
```bash
# Windows / Linux with NVIDIA GPU (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
*If you are running on **CPU only** (no NVIDIA GPU):*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Install Project Requirements:
```bash
pip install -r requirements.txt
```

Verify your installation:
```bash
python -c "import torch, transformers, streamlit; print('✓ CUDA Available:', torch.cuda.is_available(), '| PyTorch:', torch.__version__)"
```

---

## 4. Connecting and Managing Artifacts

The repository code is clean and lightweight (~10 MB). The heavy model weights and transformation objects (`models/`), pre-computed embeddings (`embeddings/`), and datasets (`data/`) are hosted in [Chimera418/protein-ssp-artifacts](https://huggingface.co/Chimera418/protein-ssp-artifacts).

### Strategy A: Automatic Lazy-Load (Recommended)
You **do not need to download anything manually**!

The application includes an automatic artifact synchronizer (`ensure_model_exists()` in `app.py`). When you run the Streamlit app or inference script:
1. It inspects your local `models/` folder.
2. If the required `.pt` weight file or `.pkl` transformation mask is missing, it contacts Hugging Face Model Hub and pulls the file automatically into your local directory.
3. Once downloaded, the file is cached locally forever.

---

### Strategy B: Pre-downloading Model Weights for Offline Use (~1.46 GB)

If you are deploying on an air-gapped machine, high-performance computing cluster, or want instant startup without waiting for downloads during the first prediction:

Run this one-line Python script from your terminal:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Chimera418/protein-ssp-artifacts', allow_patterns=['models/*'], local_dir='.')"
```

This populates your local `models/` directory with:
- `models/phase_8_best_model_pca_embeddings.pt` (Best PCA Model, 739 dims)
- `models/phase_8_best_model_filtered_embeddings.pt` (Pearson Model, 1017 dims)
- `models/phase_8_best_model_Rostlab_prot_t5_xl_uniref50.pt` (Direct Model, 1024 dims)
- `models/phase_8_best_model_final_features.pt` (Feature Selected V1, 102 dims)
- `models/phase_8_best_model_final_features_v2.pt` (Ultralight V2, 12 dims)
- `models/pca_model.pkl` (Fitted IncrementalPCA transform)
- `models/keep_indices.pkl` (Pearson correlation index mask)
- `models/feature_selector_mask.pkl` & `feature_selector_mask_v2.pkl` (Feature importance masks)

---

### Strategy C: Downloading Pre-computed Embeddings & Curated Data

If you plan to re-train the models or run XAI explainability benchmarks locally without spending 24+ hours extracting ProtT5 embeddings from scratch:

#### Download Curated Datasets (~780 MB):
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Chimera418/protein-ssp-artifacts', allow_patterns=['data/*'], local_dir='.')"
```

#### Download Pre-computed Embeddings (~26.7 GB):
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Chimera418/protein-ssp-artifacts', allow_patterns=['embeddings/*'], local_dir='.')"
```

#### Download Evaluation Outputs & Reports (~101 MB):
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Chimera418/protein-ssp-artifacts', allow_patterns=['output/*'], local_dir='.')"
```

---

## 5. Running the Predictor

### Interactive Web App (Streamlit Dashboard)

Launch the web app with a single command:

```bash
streamlit run app.py
```

The application automatically opens in your default browser at `http://localhost:8501`.

```
======================================================================
  🧬 Protein Secondary Structure Predictor (Local Host)
  URL: http://localhost:8501
======================================================================
```

#### In the Web UI:
1. **Choose Prediction Pipeline**: Select from 5 modes in the sidebar (Default: **PCA Pipeline (739-dim) ★ Best**).
2. **Input Sequence**: Paste a raw amino-acid string (e.g. `MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR`) or click **📋 Sample**.
3. **Click Predict**: The app tokenizes the sequence, executes forward inference, and renders:
   - Per-residue colour annotations (**Red** = $\alpha$-Helix, **Blue** = $\beta$-Strand, **Teal** = Coil).
   - Interactive probability distribution bar charts.
   - Secondary structure composition percentage donut/pie chart.
   - **Download Predictions CSV** button.

---

### Standalone Command-Line Predictor (CLI)

For batch processing or embedding into automated bioinformatics workflows without launching a browser:

```bash
# Run using the recommended PCA pipeline:
python phase_10_predict.py --sequence "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR" --mode pipeline
```

**Output:**
```
[1/3] Loading ProtT5 Encoder on cuda...
[2/3] Extracting ProtT5 Embeddings for sequence (Length: 141)...
[3/3] Running Pipeline Mode (ProtT5 -> Pearson Filter -> PCA -> DL)...
------------------------------------------------------------
PREDICTION RESULTS:
Sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR
Pred SS : CCHHHHHHHHHHHHHHCCCCCHHHHHHHHHHHHHHHCCCCCHHHCCCHHHHHHHHHHHHHHHHHHHHHHHHHCHHHHHHHHHHHHHHHHHHCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCC
------------------------------------------------------------
```

---

## 6. Reproducing the Full 10-Phase Pipeline from Scratch

All source code is self-contained. If you want to train and build everything from raw wwPDB data:

```bash
# Phase 1: Download wwPDB flatfiles & parse raw sequences
python phase_1_raw_sequences.py

# Phase 2: Filter non-redundant sequences via PISCES culled list (resolution <= 2.0A, R <= 0.25)
python phase_2_data_curate.py

# Phase 3: Map DSSP 8-state and 3-state secondary structure labels
python phase_3_labelled_curated_sequence.py

# Phase 4: Benchmark 4 PLMs on GPU and generate 1024-dim ProtT5 embeddings (~9.4 GB)
python phase_4_embedding_generation.py

# Phase 5: Pearson correlation filtering (drop collinear dims > 0.85; 1024 -> 1017)
python phase_5_feature_filtering.py

# Phase 6: IncrementalPCA dimensionality reduction (95% variance; 1017 -> 739)
python phase_6_dimensionality_reduction.py

# Phase 7: ExtraTrees supervised feature importance ranking (739 -> 102 dims)
python phase_7_feature_selection.py

# Phase 7.5: Ultralight top-12 feature refinement (102 -> 12 dims)
python phase_7_5_feature_refinement.py

# Phase 8: Deep learning model training (CNN + BiLSTM + Attention + Focal Loss)
# Trains the primary PCA model:
python phase_8_deep_learning_model.py --input embeddings/pca_embeddings.pkl

# Phase 9: Explainable AI suite (Attention, Saliency, Integrated Gradients, SHAP, LIME)
python phase_9_explainable_ai.py --model models/phase_8_best_model_pca_embeddings.pt --input embeddings/pca_embeddings.pkl

# Phase 10: CLI verification
python phase_10_predict.py --sequence "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFD" --mode pipeline
```

---

## 7. Uploading and Synchronizing Artifacts back to Hugging Face

If you generate new models, updated CSVs, or logs and wish to sync them to the Model Hub, use [`upload_artifacts.py`](file:///c:/Users/anees/Documents/ASEB/SEM%204/SEM-4/IBS/endsem/Secondary_Structure_Prediction/upload_artifacts.py):

```bash
# Upload all folders sequentially:
python upload_artifacts.py --all

# Or upload specific directories:
python upload_artifacts.py --models        # Upload models/ directory (~1.46 GB)
python upload_artifacts.py --output        # Upload output/ evaluation plots (~101 MB)
python upload_artifacts.py --raw-output    # Upload raw execution logs (~76 KB)
python upload_artifacts.py --data          # Upload curated CSV datasets (~780 MB)
python upload_artifacts.py --raw-data      # Upload raw wwPDB files (~1.96 GB)
python upload_artifacts.py --embeddings    # Upload pre-computed embeddings (~26.7 GB)
```

---

## 8. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: `ModuleNotFoundError: No module named 'numpy._core'` or unpickling error
* **Cause**: Pickles generated with NumPy 2.x referencing internal `_core` modules when loaded under older NumPy versions.
* **Resolution**: Use Python 3.10 and run `pip install --upgrade numpy scikit-learn`. `app.py` already includes an automatic cross-version polyfill shim (`_patch_numpy_pickle_compat()`) to prevent this error.

### Q2: CUDA Out-Of-Memory (OOM) during Phase 4 or Phase 8
* **Cause**: Extremely long sequences allocate large attention tensors on GPU.
* **Resolution**: The pipeline scripts include `--max-len 512` to trim oversized proteins during training. For inference, sequences up to 1,500 residues comfortably fit within 6 GB of VRAM.

### Q3: Why does the first prediction take ~2 minutes?
* **Explanation**: On the very first run, PyTorch downloads the pre-trained `ProtT5-XL-UniRef50` encoder weights (~3.2 GB) from the official Rostlab repository. Subsequent inferences use the local cache and complete in ~1–2 seconds.

### Q4: Can I run without an internet connection?
* **Yes!** Pre-download the weights once via:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download(repo_id="Chimera418/protein-ssp-artifacts", allow_patterns="models/*", local_dir=".")
  ```
  Once the ProtT5 cache (`~/.cache/huggingface/hub`) and `models/` directory are populated, the entire application runs 100% offline.

