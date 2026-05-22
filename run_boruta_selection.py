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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def main():
    input_pkl = 'embeddings/final_features.pkl'
    phase7_mask_pkl = 'models/feature_selector_mask.pkl'
    labelled_csv = 'data/protein_labelled_curated.csv'

    print("\n[1/3] Loading Data")
    print("-" * 60)
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"Input file {input_pkl} not found. Please run Phase 7 first.")
    
    with open(input_pkl, 'rb') as f:
        embeddings = pickle.load(f)
    
    protein_keys = list(embeddings.keys())
    num_proteins = len(protein_keys)
    feature_dim = next(iter(embeddings.values())).shape[1]
    log.info(f"Loaded {num_proteins:,} proteins with {feature_dim} features from Phase 7.")

    # Load Phase 7 PCA space indices (needed to compose absolute indices)
    with open(phase7_mask_pkl, 'rb') as f:
        phase7_mask_data = pickle.load(f)
    phase7_pca_indices = phase7_mask_data['selected_indices'] # shape (106,)
    log.info(f"Loaded Phase 7 PCA indices mask with length {len(phase7_pca_indices)}")

    # Load metadata
    df_meta = pd.read_csv(labelled_csv)
    df_meta['key'] = df_meta['pdb_id'].astype(str) + '_' + df_meta['chain_code'].astype(str)
    meta_lookup = df_meta.set_index('key')[['seq', 'sst3']].to_dict('index')

    # -------------------------------------------------------------
    print("\n[2/3] Preparing Sample for Boruta")
    print("-" * 60)
    # Using 1000 proteins to ensure a large enough sample size relative to the 106 features
    sample_size = min(1000, num_proteins)
    random.seed(42)
    sample_keys = random.sample(protein_keys, sample_size)
    log.info(f"Sampling {sample_size} proteins to run Boruta feature selection...")

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
    print("\n[3/3] Running Boruta Feature Selection")
    print("-" * 60)
    # Use ExtraTreesClassifier as base estimator
    # max_depth=5 is recommended to avoid overfitting shadow features during Boruta's iterations
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)

    # Initialize Boruta
    # max_iter: number of iterations. 50-100 is typical. Let's do 50 to keep it relatively fast.
    # perc: threshold. 100 means strict. 90-100 is standard.
    feat_selector = BorutaPy(
        clf,
        n_estimators='auto',
        verbose=2,
        random_state=42,
        max_iter=50,
        perc=90
    )

    log.info("Fitting Boruta (this may take a minute or two)...")
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
    print("BORUTA SELECTION RESULTS")
    print("=" * 60)
    print(f"Total features input     : {feature_dim}")
    print(f"Confirmed features (green): {num_confirmed}")
    print(f"Tentative features (yellow): {num_tentative}")
    print(f"Rejected features (red)    : {num_rejected}")
    print("-" * 60)

    # Print rankings
    log.info("Ranking of all features:")
    rankings = feat_selector.ranking_
    for idx in range(feature_dim):
        status = "CONFIRMED" if confirmed_mask[idx] else "TENTATIVE" if tentative_mask[idx] else "REJECTED"
        if status != "REJECTED":
            log.info(f"  Feature {idx+1:3d} (PC{phase7_pca_indices[idx] + 1}): Rank = {rankings[idx]:2d} [{status}]")

    # Map the confirmed features back to absolute PCA indices
    confirmed_pca_indices = phase7_pca_indices[confirmed_indices]
    confirmed_feature_names = [f"PC{idx + 1}" for idx in confirmed_pca_indices]
    log.info(f"Confirmed absolute features ({num_confirmed}): {confirmed_feature_names}")

    # Save outputs if wanted
    boruta_results = {
        'confirmed_local_indices': confirmed_indices,
        'confirmed_pca_indices': confirmed_pca_indices,
        'tentative_local_indices': tentative_indices,
        'ranking': rankings,
        'support': confirmed_mask,
        'support_weak': tentative_mask
    }
    
    out_mask_path = 'models/boruta_selector_mask.pkl'
    with open(out_mask_path, 'wb') as f:
        pickle.dump(boruta_results, f)
    log.info(f"Saved Boruta selector mask metadata -> {out_mask_path}")

    report_text = f"""============================================================
BORUTA FEATURE SELECTION REPORT (Phase 7 -> Phase 7.5 Validation)
============================================================
Input Features : {feature_dim}
Sample Size    : {sample_size} proteins

RESULTS
------------------------------------------------------------
Confirmed features (Green)  : {num_confirmed}
Tentative features (Yellow) : {num_tentative}
Rejected features (Red)     : {num_rejected}

Confirmed Feature Names:
{confirmed_feature_names}
============================================================
"""
    os.makedirs('output/boruta', exist_ok=True)
    report_path = 'output/boruta/boruta_selection_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    log.info(f"Saved Boruta report -> {report_path}")

    print("=" * 60 + "\\n")

if __name__ == "__main__":
    main()
