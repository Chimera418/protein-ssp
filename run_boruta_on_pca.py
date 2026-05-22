import os
import gc
import pickle
import random
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from boruta import BorutaPy

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def main():
    input_pkl = 'embeddings/pca_embeddings.pkl'
    labelled_csv = 'data/protein_labelled_curated.csv'

    print("\n[1/3] Loading PCA Embeddings")
    print("-" * 60)
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"Input file {input_pkl} not found. Please run Phase 6 first.")
    
    with open(input_pkl, 'rb') as f:
        embeddings = pickle.load(f)
    
    protein_keys = list(embeddings.keys())
    num_proteins = len(protein_keys)
    feature_dim = next(iter(embeddings.values())).shape[1]
    log.info(f"Loaded {num_proteins:,} proteins with {feature_dim} PCA features.")

    # Load metadata
    df_meta = pd.read_csv(labelled_csv)
    df_meta['key'] = df_meta['pdb_id'].astype(str) + '_' + df_meta['chain_code'].astype(str)
    meta_lookup = df_meta.set_index('key')[['seq', 'sst3']].to_dict('index')

    # -------------------------------------------------------------
    print("\n[2/3] Preparing Sample for Boruta")
    print("-" * 60)
    # Using 1000 proteins to ensure a large enough sample size relative to the 739 features
    sample_size = min(1000, num_proteins)
    random.seed(42)
    sample_keys = random.sample(protein_keys, sample_size)
    log.info(f"Sampling {sample_size} proteins to run Boruta directly on PCA features...")

    X_list, y_list = [], []
    for k in sample_keys:
        emb = embeddings[k]
        meta = meta_lookup.get(k)
        if meta is None or len(meta['sst3']) != emb.shape[0]:
            continue
        X_list.append(emb)
        y_list.extend(list(meta['sst3']))
    
    X_sample = np.vstack(X_list)
    le = LabelEncoder()
    y_sample = le.fit_transform(y_list)
    log.info(f"Sample matrix shape: {X_sample.shape} | Target classes: {le.classes_}")

    # -------------------------------------------------------------
    print("\n[3/3] Running Boruta on PCA")
    print("-" * 60)
    # n_jobs=-1 will utilize all CPU cores
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)

    # We do 30 iterations to see the intermediate selection speed and counts
    feat_selector = BorutaPy(
        clf,
        n_estimators='auto',
        verbose=2,
        random_state=42,
        max_iter=30,
        perc=95
    )

    log.info("Fitting Boruta on 739 features (this will take a few minutes)...")
    feat_selector.fit(X_sample, y_sample)

    # Analyze results
    confirmed_mask = feat_selector.support_
    tentative_mask = feat_selector.support_weak_
    
    confirmed_indices = np.where(confirmed_mask)[0]
    tentative_indices = np.where(tentative_mask)[0]

    num_confirmed = len(confirmed_indices)
    num_tentative = len(tentative_indices)
    num_rejected = feature_dim - num_confirmed - num_tentative

    print("\n" + "=" * 60)
    print("BORUTA ON PCA RESULTS")
    print("=" * 60)
    print(f"Total PCA features input  : {feature_dim}")
    print(f"Confirmed PCA features   : {num_confirmed}")
    print(f"Tentative PCA features   : {num_tentative}")
    print(f"Rejected PCA features    : {num_rejected}")
    print("-" * 60)

    confirmed_feature_names = [f"PC{idx + 1}" for idx in confirmed_indices]
    if num_confirmed > 0:
        log.info(f"Confirmed PCA feature names (first 50): {confirmed_feature_names[:50]}...")
    
    # Save Boruta results
    boruta_results = {
        'confirmed_pca_indices': confirmed_indices,
        'tentative_pca_indices': tentative_indices,
        'ranking': feat_selector.ranking_,
        'support': confirmed_mask,
        'support_weak': tentative_mask
    }
    
    out_mask_path = 'models/boruta_pca_selector_mask.pkl'
    with open(out_mask_path, 'wb') as f:
        pickle.dump(boruta_results, f)
    log.info(f"Saved Boruta PCA selector mask metadata -> {out_mask_path}")

    report_text = f"""============================================================
BORUTA ON PCA REPORT (Phase 6 -> Phase 7 Validation)
============================================================
Input Features : {feature_dim}
Sample Size    : {sample_size} proteins

RESULTS
------------------------------------------------------------
Confirmed features (Green)  : {num_confirmed}
Tentative features (Yellow) : {num_tentative}
Rejected features (Red)     : {num_rejected}

Confirmed PCA Feature Names (first 100):
{confirmed_feature_names[:100]}
============================================================
"""
    os.makedirs('output/boruta', exist_ok=True)
    report_path = 'output/boruta/boruta_pca_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    log.info(f"Saved Boruta report -> {report_path}")

    print("=" * 60 + "\\n")

if __name__ == "__main__":
    main()
