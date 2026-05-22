---
title: Protein Secondary Structure Predictor
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# 🧬 Protein Secondary Structure Predictor

A deep learning pipeline for predicting protein secondary structures (α-Helix, β-Sheet, Coil) from amino acid sequences. This project leverages the powerful **ProtT5-XL-UniRef50** protein language model to generate embeddings, followed by a custom architecture featuring **1D-CNNs, BiLSTMs, and Multi-Head Attention**.

## 🌟 Key Features
- **ProtT5 Embeddings**: Uses `Rostlab/prot_t5_xl_uniref50` to extract rich, context-aware sequence embeddings.
- **Advanced Architecture**: Combines local feature extraction (CNN), sequence modeling (BiLSTM), and global context (Multi-Head Attention).
- **Multiple Inference Modes**: Compare predictions across different feature engineering pipelines:
  - Direct (1024-dim)
  - Pearson-filtered (1017-dim)
  - PCA-reduced (739-dim) - **Best Performance**
  - Tree-selected (109-dim & 12-dim)
- **Interactive UI**: A Streamlit web app for real-time predictions, per-residue confidence visualization, and CSV exporting.

## 🚀 Deployment (Hugging Face Spaces)
The interactive web application is designed to be hosted on Hugging Face Spaces. Because the language model and trained weights are extremely large (~1.5GB+), they are tracked via Git LFS on the `hf-space` branch rather than the `main` GitHub repository.

## 💻 Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chimera418/prot-ssp.git
   cd prot-ssp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   *Note: Running the app requires the pre-trained model files in the `models/` directory, which are not included in the main GitHub repository due to file size limits.*

## 📂 Project Structure
- `app.py`: The main Streamlit dashboard.
- `phase_*.py`: Modular scripts covering the end-to-end ML lifecycle (data curation, embedding generation, feature selection, deep learning model training, and Explainable AI).
- `models/`: Contains the trained `.pt` deep learning models and `.pkl` artifacts (PCA transformations, feature masks).

## 🛠️ Built With
- **PyTorch**: Deep learning models.
- **Transformers (Hugging Face)**: ProtT5 language model.
- **Streamlit**: Web interface.
- **Scikit-Learn**: Dimensionality reduction and feature selection.
