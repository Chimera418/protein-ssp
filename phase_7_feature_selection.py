import os
import gc
import pickle
import random
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

"""
Phase 7: Tree-based Feature Selection
======================================

Pipeline Steps
--------------
[1/4] Load PCA embeddings from Phase 6 PKL
[2/4] Sample subset → train ExtraTreesClassifier → rank feature importances
[3/4] Apply feature mask to full dataset → save selected PKL
[4/4] Write residue-level CSV (batched), visualisations & report
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Tree-based Feature Selection")
    parser.add_argument('--input',        type=str, default='embeddings/pca_embeddings.pkl',
                        help="Path to PCA PKL from Phase 6")
    parser.add_argument('--output-csv',   type=str, default='data/final_selected_features.csv',
                        help="Path to save final residue-level CSV")
    parser.add_argument('--output-pkl',   type=str, default='embeddings/final_features.pkl',
                        help="Path to save final selected features PKL")
    parser.add_argument('--selector-pkl', type=str, default='models/feature_selector_mask.pkl',
                        help="Path to save fitted selector & indices (for inference)")
    parser.add_argument('--report-dir',   type=str, default='output/phase_7',
                        help="Directory for visualisations and report")
    parser.add_argument('--sample-size',  type=int, default=1000,
                        help="Number of proteins to sample for ExtraTreesClassifier")
    parser.add_argument('--max-features', type=int, default=0,
                        help="Max features to keep (0 = auto: mean importance threshold)")
    parser.add_argument('--batch-size',   type=int, default=200,
                        help="Proteins per batch when writing CSV")
    parser.add_argument('--csv-limit',    type=int, default=200,
                        help="Max proteins to write to CSV (0 = all). PKL always saves full dataset.")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    # ------------------------------------------------------------------
    print("\n[1/4] Load PCA embeddings")
    print("-" * 60)
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"PKL not found: {args.input}. Run Phase 6 first.")

    log.info(f"  Loading embeddings from {args.input} ...")
    with open(args.input, 'rb') as f:
        embeddings = pickle.load(f)

    # Phase 6 saves as  { pdb_id : numpy_array(seq_len, n_pca_components) }
    protein_keys = list(embeddings.keys())
    num_proteins = len(protein_keys)
    orig_dim = next(iter(embeddings.values())).shape[1]
    log.info(f"  Loaded {num_proteins:,} proteins  |  {orig_dim} PCA features per residue")

    # Metadata lookup for seq/label
    labelled_csv = 'data/protein_labelled_curated.csv'
    if not os.path.exists(labelled_csv):
        raise FileNotFoundError(f"Labelled CSV not found: {labelled_csv}. Run Phase 3 first.")
    df_meta = pd.read_csv(labelled_csv)
    df_meta['key'] = df_meta['pdb_id'].astype(str) + '_' + df_meta['chain_code'].astype(str)
    meta_lookup = df_meta.set_index('key')[['seq', 'sst3']].to_dict('index')
    log.info(f"  Metadata lookup built for {len(meta_lookup):,} proteins.")

    # ------------------------------------------------------------------
    print("\n[2/4] Train ExtraTreesClassifier & select features")
    print("-" * 60)
    sample_size = min(args.sample_size, num_proteins)
    sample_keys = random.sample(protein_keys, sample_size)
    log.info(f"  Sampling {sample_size} proteins for ExtraTreesClassifier ...")

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
    log.info(f"  Sample residue matrix: {X_sample.shape}  |  Classes: {le.classes_}")

    log.info("  Fitting ExtraTreesClassifier (n_estimators=100, n_jobs=-1) ...")
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_sample, y_sample)

    importances = clf.feature_importances_

    # Feature selection threshold
    max_feat = args.max_features if args.max_features > 0 else None
    selector = SelectFromModel(clf, prefit=True, max_features=max_feat)
    support  = selector.get_support()
    selected_indices = np.where(support)[0]
    final_dim = len(selected_indices)

    log.info(f"  Original PCA features  : {orig_dim}")
    log.info(f"  Features selected      : {final_dim}")
    log.info(f"  Reduction              : {orig_dim - final_dim} features dropped")

    # -- Feature importance bar chart (top 50) --
    log.info("  Saving feature importance plot ...")
    top_n = min(50, orig_dim)
    sorted_idx = np.argsort(importances)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Top-N importance bar
    axes[0].bar(range(top_n), importances[sorted_idx[:top_n]], color='tab:blue', alpha=0.8)
    axes[0].set_xlabel('Feature Rank')
    axes[0].set_ylabel('Importance Score')
    axes[0].set_title(f'Top {top_n} Feature Importances (ExtraTrees)', fontweight='bold')
    axes[0].axhline(importances.mean(), color='tab:red', linestyle='--',
                    lw=1.5, label=f'Mean = {importances.mean():.5f}')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # Selected vs dropped bar
    axes[1].bar(['Total PCA Dims', 'Selected', 'Dropped'],
                [orig_dim, final_dim, orig_dim - final_dim],
                color=['tab:blue', 'tab:green', 'tab:red'], width=0.5)
    for bar, val in zip(axes[1].patches, [orig_dim, final_dim, orig_dim - final_dim]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(val), ha='center', va='bottom', fontweight='bold')
    axes[1].set_ylabel('Number of Features')
    axes[1].set_title('Feature Counts After Tree Selection', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('Phase 7 — ExtraTreesClassifier Feature Selection', fontsize=13, fontweight='bold')
    fig.tight_layout()
    imp_plot_path = os.path.join(args.report_dir, 'feature_importance_plot.png')
    fig.savefig(imp_plot_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved importance plot -> {imp_plot_path}")

    # Save selector mask for inference pipeline
    selector_data = {'selected_indices': selected_indices, 'selector': selector, 'classes': le.classes_}
    with open(args.selector_pkl, 'wb') as f:
        pickle.dump(selector_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  Saved feature selector -> {args.selector_pkl}")

    del X_sample, y_sample, clf, selector
    gc.collect()

    # ------------------------------------------------------------------
    print("\n[3/4] Apply feature mask to full dataset & save PKL")
    print("-" * 60)
    log.info(f"  Applying mask (keeping {final_dim} features) to all {num_proteins:,} proteins ...")

    final_embeddings = {}
    for pdb_id in tqdm(protein_keys, desc="  Applying mask"):
        final_embeddings[pdb_id] = embeddings[pdb_id][:, selected_indices].astype(np.float32)

    del embeddings
    gc.collect()

    log.info(f"  Saving final PKL -> {args.output_pkl}")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(final_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    print("\n[4/4] Write residue-level CSV & generate report")
    print("-" * 60)
    log.info(f"  Writing CSV -> {args.output_csv}  (batches of {args.batch_size} proteins)")

    feat_cols   = [f'PC{selected_indices[i] + 1}' for i in range(final_dim)]
    header_cols = ['PDB_ID', 'Residue_Index', 'Amino_Acid', 'Label'] + feat_cols

    first_write = True
    proteins_written = 0
    residues_written = 0
    skipped = 0

    batch_embs, batch_pdb, batch_ridx, batch_aa, batch_labels = [], [], [], [], []
    items = list(final_embeddings.items())

    def _flush(first_write):
        if not batch_embs:
            return first_write
        emb_stack = np.vstack(batch_embs)
        meta_arr  = np.column_stack([batch_pdb, batch_ridx, batch_aa, batch_labels])
        full      = np.hstack([meta_arr, emb_stack.astype(str)])
        pd.DataFrame(full, columns=header_cols).to_csv(
            args.output_csv, mode='w' if first_write else 'a',
            header=first_write, index=False)
        return False

    csv_limit = args.csv_limit if args.csv_limit > 0 else len(items)
    items_csv = items[:csv_limit]
    log.info(f"  Writing {len(items_csv):,} proteins to CSV (full dataset in PKL)...")

    for pdb_id, emb in tqdm(items_csv, desc="  Writing CSV"):
        meta = meta_lookup.get(pdb_id)
        if meta is None:
            skipped += 1
            continue

        seq = meta['seq']
        lbl = meta['sst3']

        if len(seq) != emb.shape[0]:
            skipped += 1
            continue

        n = len(seq)
        batch_embs.append(emb)
        batch_pdb.extend([pdb_id] * n)
        batch_ridx.extend(range(1, n + 1))
        batch_aa.extend(list(seq))
        batch_labels.extend(list(lbl))
        proteins_written += 1
        residues_written += n

        if proteins_written % args.batch_size == 0:
            first_write = _flush(first_write)
            batch_embs.clear(); batch_pdb.clear()
            batch_ridx.clear(); batch_aa.clear(); batch_labels.clear()

    first_write = _flush(first_write)

    log.info(f"  Proteins written to CSV : {proteins_written:,} / {csv_limit:,} (limit)")
    log.info(f"  Residues written        : {residues_written:,}")
    log.info(f"  Proteins skipped        : {skipped}")
    log.info(f"  CSV saved -> {args.output_csv}")

    report_text = f"""============================================================
Phase 7 -- Tree-based Feature Selection Report
============================================================

INPUT
----------------------------------------
  PCA Embedding PKL     : {args.input}
  Total Proteins Loaded : {num_proteins:,}
  PCA Feature Dims      : {orig_dim}

FEATURE SELECTION (ExtraTreesClassifier)
----------------------------------------
  Sample Size (proteins)  : {sample_size}
  Classifier              : ExtraTreesClassifier (n_estimators=100)
  Mean Feature Importance : {importances.mean():.6f}
  Selection Threshold     : mean importance
  Original Features       : {orig_dim}
  Features Selected       : {final_dim}
  Features Dropped        : {orig_dim - final_dim}
  Retention Rate          : {100 * final_dim / orig_dim:.1f}%

OUTPUT
----------------------------------------
  Feature Selector PKL  : {args.selector_pkl}
  Final Features PKL    : {args.output_pkl}
  Final Features CSV    : {args.output_csv}
  Proteins written      : {proteins_written:,}
  Residues written      : {residues_written:,}
  Proteins skipped      : {skipped}

VISUALISATIONS
----------------------------------------
  Feature Importance Plot : {imp_plot_path}

============================================================
Phase 7 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, 'feature_selection_phase_7_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    log.info(f"  Report saved -> {report_path}")


if __name__ == "__main__":
    main()
