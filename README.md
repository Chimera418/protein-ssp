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

## 🚀 Deployment

The interactive web application is deployed live on Hugging Face Spaces!
👉 **[Try the Live App Here](https://huggingface.co/spaces/Chimera418/protein-ssp)**

### Storage Architecture
Because the pre-trained ProtT5 models and generated embeddings are extremely large (~28GB total), we utilize a decoupled architecture to bypass Git size limitations:
- **Application Code:** Hosted on the [Hugging Face Space](https://huggingface.co/spaces/Chimera418/protein-ssp)
- **Model Weights & Embeddings:** Hosted securely in a dedicated Hugging Face Model Hub at [Chimera418/protein-ssp-artifacts](https://huggingface.co/Chimera418/protein-ssp-artifacts). The app downloads these heavy files on-the-fly at runtime.

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
