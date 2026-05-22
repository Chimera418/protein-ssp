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
from sklearn.preprocessing import LabelEncoder

"""
Phase 7.5: Second-Pass Feature Refinement
==========================================

Takes the 106-feature PKL from Phase 7 and applies a second hard-capped
ExtraTreesClassifier pass to retain only the top N features (default 12).

The new feature_selector_mask.pkl stores ABSOLUTE PCA-space indices
(Phase-7 mask ∘ Phase-7.5 local mask), so app.py works unchanged.

Pipeline Steps
--------------
[1/4] Load Phase-7 PKL (106 features) + Phase-7 selector mask
[2/4] Train ExtraTreesClassifier → pick top N by importance
[3/4] Apply combined mask → save final_features_v2.pkl
[4/4] Overwrite feature_selector_mask.pkl, write CSV + report
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 7.5: Second-Pass Feature Refinement")
    parser.add_argument('--input',          type=str, default='embeddings/final_features.pkl',
                        help="Phase-7 output PKL (106 features)")
    parser.add_argument('--phase7-mask',    type=str, default='models/feature_selector_mask.pkl',
                        help="Phase-7 selector mask (contains PCA-space selected_indices)")
    parser.add_argument('--output-pkl',     type=str, default='embeddings/final_features_v2.pkl',
                        help="Output PKL with refined features")
    parser.add_argument('--output-csv',     type=str, default='data/final_selected_features_v2.csv',
                        help="Residue-level CSV (capped at --csv-limit proteins)")
    parser.add_argument('--selector-pkl',   type=str, default='models/feature_selector_mask_v2.pkl',
                        help="New selector mask PKL with combined PCA-space indices")
    parser.add_argument('--report-dir',     type=str, default='output/phase_7_5',
                        help="Directory for plots and text report")
    parser.add_argument('--top-n',          type=int, default=12,
                        help="Hard cap: keep exactly this many features (must be < 15)")
    parser.add_argument('--sample-size',    type=int, default=2000,
                        help="Proteins sampled to train ExtraTreesClassifier")
    parser.add_argument('--n-estimators',   type=int, default=200,
                        help="Number of trees in ExtraTreesClassifier")
    parser.add_argument('--batch-size',     type=int, default=200,
                        help="Proteins per batch when writing CSV")
    parser.add_argument('--csv-limit',      type=int, default=200,
                        help="Max proteins to write to CSV (0 = all)")
    args = parser.parse_args()

    if args.top_n >= 15:
        raise ValueError(f"--top-n must be < 15, got {args.top_n}")

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs('data',   exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    print("\n[1/4] Load Phase-7 features + selector mask")
    print("-" * 60)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"PKL not found: {args.input}. Run Phase 7 first.")

    log.info(f"  Loading {args.input} ...")
    with open(args.input, 'rb') as f:
        embeddings = pickle.load(f)

    protein_keys = list(embeddings.keys())
    num_proteins = len(protein_keys)
    orig_dim     = next(iter(embeddings.values())).shape[1]
    log.info(f"  Loaded {num_proteins:,} proteins  |  {orig_dim} features per residue")

    # Load Phase-7 PCA-space indices (needed to compose the combined mask)
    if not os.path.exists(args.phase7_mask):
        raise FileNotFoundError(f"Phase-7 mask not found: {args.phase7_mask}. Run Phase 7 first.")
    with open(args.phase7_mask, 'rb') as f:
        phase7_data = pickle.load(f)
    phase7_pca_indices = phase7_data['selected_indices']   # shape (106,) — absolute PCA indices
    log.info(f"  Phase-7 PCA-space indices loaded: {len(phase7_pca_indices)} features")

    # Metadata
    labelled_csv = 'data/protein_labelled_curated.csv'
    if not os.path.exists(labelled_csv):
        raise FileNotFoundError(f"Labelled CSV not found: {labelled_csv}. Run Phase 3 first.")
    df_meta     = pd.read_csv(labelled_csv)
    df_meta['key'] = df_meta['pdb_id'].astype(str) + '_' + df_meta['chain_code'].astype(str)
    meta_lookup = df_meta.set_index('key')[['seq', 'sst3']].to_dict('index')
    log.info(f"  Metadata lookup built for {len(meta_lookup):,} proteins.")

    # ──────────────────────────────────────────────────────────────
    print(f"\n[2/4] Train ExtraTreesClassifier -> pick top {args.top_n} features")
    print("-" * 60)

    sample_size = min(args.sample_size, num_proteins)
    sample_keys = random.sample(protein_keys, sample_size)
    log.info(f"  Sampling {sample_size} proteins for ExtraTreesClassifier ...")

    X_list, y_list = [], []
    for k in sample_keys:
        emb  = embeddings[k]
        meta = meta_lookup.get(k)
        if meta is None or len(meta['sst3']) != emb.shape[0]:
            continue
        X_list.append(emb)
        y_list.extend(list(meta['sst3']))

    X_sample = np.vstack(X_list)
    le       = LabelEncoder()
    y_sample = le.fit_transform(y_list)
    log.info(f"  Sample residue matrix: {X_sample.shape}  |  Classes: {le.classes_}")

    log.info(f"  Fitting ExtraTreesClassifier "
             f"(n_estimators={args.n_estimators}, n_jobs=-1) ...")
    clf = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_sample, y_sample)

    importances = clf.feature_importances_   # shape (106,)
    # Hard cap: take the top-N by importance score
    sorted_local_idx  = np.argsort(importances)[::-1]         # descending
    local_top_indices = sorted_local_idx[:args.top_n]         # top-N local indices (into 106-space)
    local_top_indices = np.sort(local_top_indices)            # keep original order

    final_dim = len(local_top_indices)
    assert final_dim == args.top_n

    # Compose: absolute PCA indices for the retained features
    combined_pca_indices = phase7_pca_indices[local_top_indices]   # shape (top_n,)
    log.info(f"  Phase-7 features (in PCA space) : {len(phase7_pca_indices)}")
    log.info(f"  Phase-7.5 local top-{args.top_n} indices : {local_top_indices.tolist()}")
    log.info(f"  Combined PCA-space indices       : {combined_pca_indices.tolist()}")
    log.info(f"  ✓ Final feature count            : {final_dim}  (< 15 ✓)")

    # ── Visualisations ──────────────────────────────────────────
    log.info("  Saving visualisations ...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # (a) All 106 importances
    axes[0].bar(range(orig_dim), importances, color='tab:grey', alpha=0.5)
    axes[0].bar(local_top_indices, importances[local_top_indices],
                color='tab:orange', alpha=0.9, label=f'Top {args.top_n} retained')
    axes[0].axhline(importances.mean(), color='tab:red', linestyle='--', lw=1.5,
                    label=f'Mean = {importances.mean():.5f}')
    axes[0].set_xlabel('Feature Index (within 106-dim space)')
    axes[0].set_ylabel('Importance Score')
    axes[0].set_title('All Feature Importances\n(orange = retained)', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    # (b) Top-N importances bar
    top_scores = importances[local_top_indices]
    axes[1].bar(range(final_dim), top_scores, color='tab:orange', alpha=0.85)
    for i, (score, pca_idx) in enumerate(zip(top_scores, combined_pca_indices)):
        axes[1].text(i, score + 0.0001, f'PCA\n{pca_idx}',
                     ha='center', va='bottom', fontsize=7, color='#333')
    axes[1].set_xlabel('Retained Feature Rank')
    axes[1].set_ylabel('Importance Score')
    axes[1].set_title(f'Top {args.top_n} Retained Features\n(labeled by PCA component)',
                      fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # (c) Reduction funnel
    stages = ['ProtT5\n(1024)', 'After Phase5\nFilter', 'PCA\n(739)',
              'ExtraTrees\nPhase7 (106)', f'ExtraTrees\nPhase7.5 ({final_dim})']
    values = [1024, 1024, 739, 106, final_dim]   # Phase5 keeps all dims, just drops constant cols
    colors = ['#4c72b0', '#55a868', '#c44e52', '#dd8452', '#8172b3']
    bars = axes[2].bar(stages, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 10, str(val),
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
    axes[2].set_ylabel('Number of Features')
    axes[2].set_title('Full Reduction Funnel', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Phase 7.5 — Second-Pass Feature Refinement  |  {orig_dim} → {final_dim} features',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    plot_path = os.path.join(args.report_dir, 'feature_refinement_plot.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved plot -> {plot_path}")

    # ── Save updated selector mask (creates new file) ────
    new_selector_data = {
        'selected_indices': combined_pca_indices,   # absolute PCA-space — app.py compatible
        'local_indices_phase75': local_top_indices,  # local within 106-space
        'phase7_pca_indices': phase7_pca_indices,    # original Phase-7 mask (backup)
        'classes': le.classes_,
        'importances': importances,
    }
    with open(args.selector_pkl, 'wb') as f:
        pickle.dump(new_selector_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  OK Saved new selector mask -> {args.selector_pkl}")
    log.info(f"    (selected_indices now contains {final_dim} absolute PCA indices)")

    del X_sample, y_sample, clf
    gc.collect()

    # ──────────────────────────────────────────────────────────────
    print(f"\n[3/4] Apply mask to full dataset -> save PKL")
    print("-" * 60)
    log.info(f"  Applying local mask (keeping {final_dim} of {orig_dim} features) "
             f"to all {num_proteins:,} proteins ...")

    final_embeddings = {}
    for pdb_id in tqdm(protein_keys, desc="  Applying mask"):
        final_embeddings[pdb_id] = embeddings[pdb_id][:, local_top_indices].astype(np.float32)

    del embeddings
    gc.collect()

    log.info(f"  Saving final PKL → {args.output_pkl}")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(final_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  Verified output shape: {next(iter(final_embeddings.values())).shape[1]} features ✓")

    # ──────────────────────────────────────────────────────────────
    print("\n[4/4] Write residue-level CSV + report")
    print("-" * 60)

    feat_cols   = [f'PC{combined_pca_indices[i] + 1}' for i in range(final_dim)]
    header_cols = ['PDB_ID', 'Residue_Index', 'Amino_Acid', 'Label'] + feat_cols

    first_write      = True
    proteins_written = 0
    residues_written = 0
    skipped          = 0

    batch_embs, batch_pdb, batch_ridx, batch_aa, batch_labels = [], [], [], [], []
    items     = list(final_embeddings.items())
    csv_limit = args.csv_limit if args.csv_limit > 0 else len(items)
    items_csv = items[:csv_limit]
    log.info(f"  Writing {len(items_csv):,} proteins to CSV ...")

    def _flush(fw):
        if not batch_embs:
            return fw
        emb_stack = np.vstack(batch_embs)
        meta_arr  = np.column_stack([batch_pdb, batch_ridx, batch_aa, batch_labels])
        full      = np.hstack([meta_arr, emb_stack.astype(str)])
        pd.DataFrame(full, columns=header_cols).to_csv(
            args.output_csv, mode='w' if fw else 'a', header=fw, index=False)
        return False

    for pdb_id, emb in tqdm(items_csv, desc="  Writing CSV"):
        meta = meta_lookup.get(pdb_id)
        if meta is None:
            skipped += 1; continue
        seq = meta['seq']; lbl = meta['sst3']
        if len(seq) != emb.shape[0]:
            skipped += 1; continue
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

    log.info(f"  Proteins written : {proteins_written:,}")
    log.info(f"  Residues written : {residues_written:,}")
    log.info(f"  Proteins skipped : {skipped}")

    # ── Text report ──────────────────────────────────────────────
    report_text = f"""============================================================
Phase 7.5 -- Second-Pass Feature Refinement Report
============================================================

INPUT
----------------------------------------
  Phase-7 PKL           : {args.input}
  Phase-7 features      : {orig_dim}
  Phase-7 mask PKL      : {args.phase7_mask}
  Phase-7 PCA indices   : {len(phase7_pca_indices)} absolute PCA-space indices

FEATURE REFINEMENT (ExtraTreesClassifier — Hard Cap)
----------------------------------------
  Sample Size (proteins)    : {sample_size}
  Classifier                : ExtraTreesClassifier (n_estimators={args.n_estimators})
  Mean Feature Importance   : {importances.mean():.6f}
  Selection Strategy        : Top-{args.top_n} by importance (hard cap)
  Features In               : {orig_dim}
  Features Retained         : {final_dim}
  Features Dropped          : {orig_dim - final_dim}
  Reduction Ratio           : {100 * final_dim / orig_dim:.1f}% of Phase-7 features

COMBINED REDUCTION (full pipeline)
----------------------------------------
  ProtT5 embeddings         : 1024
  After Phase-5 filter      : 1024 (Pearson + variance filter)
  After Phase-6 PCA         : 739
  After Phase-7 ExtraTrees  : {len(phase7_pca_indices)}
  After Phase-7.5 (this)    : {final_dim}
  Overall retention         : {100 * final_dim / 1024:.2f}% of raw ProtT5

LOCAL INDICES (into 106-dim Phase-7 space)
----------------------------------------
  {local_top_indices.tolist()}

COMBINED PCA-SPACE INDICES (for inference)
----------------------------------------
  {combined_pca_indices.tolist()}

OUTPUT
----------------------------------------
  Updated selector PKL  : {args.selector_pkl}
  Final Features PKL    : {args.output_pkl}
  Final Features CSV    : {args.output_csv}
  Proteins written      : {proteins_written:,}
  Residues written      : {residues_written:,}
  Proteins skipped      : {skipped}

VISUALISATIONS
----------------------------------------
  Refinement Plot : {plot_path}

NEXT STEP
----------------------------------------
  Retrain Phase 8 on the new PKL:
    python phase_8_deep_learning_model.py --input {args.output_pkl}

  The app.py 'selected' mode will automatically use the updated
  feature_selector_mask.pkl (now storing {final_dim} PCA-space indices).
  Update MODEL_REGISTRY['selected'] dim to {final_dim} in app.py.

============================================================
Phase 7.5 -- Pipeline completed successfully
============================================================
"""

    report_path = os.path.join(args.report_dir, 'feature_refinement_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    log.info(f"  Report saved -> {report_path}")

    print("\n" + "=" * 60)
    print(f"  OK  Phase 7.5 complete!")
    print(f"     {orig_dim} features  ->  {final_dim} features  (< 15 OK)")
    print(f"     PKL  : {args.output_pkl}")
    print(f"     Mask : {args.selector_pkl}  (overwritten with {final_dim} PCA indices)")
    print(f"\n  Next: python phase_8_deep_learning_model.py \\")
    print(f"            --input {args.output_pkl}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
