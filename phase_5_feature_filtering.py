import os
import gc
import pickle
import random
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

"""
Phase 5: Pearson Correlation Feature Filtering
===============================================

Pipeline Steps
--------------
[1/4] Load embeddings from Phase 4 .pkl
[2/4] Compute Pearson Correlation Matrix on a sampled residue subset
[3/4] Drop highly correlated features & apply to full dataset
[4/4] Write filtered CSV, filtered PKL, and generate report + heatmap
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Pearson Correlation Filtering")
    parser.add_argument('--input',       type=str, default='embeddings/Rostlab_prot_t5_xl_uniref50.pkl',
                        help="Path to input embeddings .pkl file from Phase 4")
    parser.add_argument('--output-csv',  type=str, default='data/filtered_protein_embeddings.csv',
                        help="Path to save filtered residue-level CSV")
    parser.add_argument('--output-pkl',  type=str, default='embeddings/filtered_embeddings.pkl',
                        help="Path to save filtered embeddings PKL")
    parser.add_argument('--report-dir',  type=str, default='output/phase_5',
                        help="Directory for report and visualisations")
    parser.add_argument('--threshold',   type=float, default=0.85,
                        help="Pearson |r| threshold above which features are dropped")
    parser.add_argument('--sample-size', type=int,   default=500,
                        help="Number of proteins to sample for correlation matrix computation")
    parser.add_argument('--csv-limit',   type=int,   default=200,
                        help="Max proteins to write to CSV (0 = all). PKL always saves full dataset.")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    # ------------------------------------------------------------------
    print("\n[1/4] Load embeddings")
    print("-" * 60)
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"PKL not found: {args.input}. Run Phase 4 first.")

    log.info(f"  Loading embeddings from {args.input} ...")
    with open(args.input, 'rb') as f:
        embeddings = pickle.load(f)

    num_proteins = len(embeddings)
    log.info(f"  Loaded {num_proteins:,} protein embeddings.")

    # Phase 4 saves as  { pdb_id : numpy_array(seq_len, hidden_dim) }
    # We also need the matching labelled CSV to get sequence & label columns
    labelled_csv = 'data/protein_labelled_curated.csv'
    if not os.path.exists(labelled_csv):
        raise FileNotFoundError(f"Labelled CSV not found: {labelled_csv}. Run Phase 3 first.")
    df_meta = pd.read_csv(labelled_csv)
    # Build a quick lookup by pdb_id_chain
    df_meta['key'] = df_meta['pdb_id'].astype(str) + '_' + df_meta['chain_code'].astype(str)
    meta_lookup = df_meta.set_index('key')[['seq', 'sst3']].to_dict('index')
    log.info(f"  Metadata lookup built for {len(meta_lookup):,} proteins.")

    # ------------------------------------------------------------------
    print("\n[2/4] Compute Pearson Correlation Matrix on sampled residues")
    print("-" * 60)
    all_keys = list(embeddings.keys())
    sample_keys = random.sample(all_keys, min(args.sample_size, num_proteins))
    log.info(f"  Sampling {len(sample_keys)} proteins (~{len(sample_keys)*200:,} residues) for correlation.")

    sample_residues = []
    for k in sample_keys:
        emb = embeddings[k]         # shape: (seq_len, hidden_dim)
        sample_residues.append(emb)

    X_sample = np.vstack(sample_residues)   # (total_sample_residues, hidden_dim)
    hidden_dim = X_sample.shape[1]
    log.info(f"  Sampled residue matrix shape: {X_sample.shape}")

    log.info("  Computing Pearson correlation matrix via np.corrcoef (fast vectorized)...")
    # np.corrcoef expects shape (features, samples) so we transpose: (1024, n_residues)
    # This is orders of magnitude faster than pd.DataFrame.corr()
    corr_raw = np.corrcoef(X_sample.T)          # shape: (1024, 1024)
    corr_matrix = pd.DataFrame(np.abs(corr_raw))  # absolute values, as DataFrame for heatmap

    # Identify features to drop: upper triangle, |r| > threshold
    upper = np.triu(corr_raw, k=1)   # upper triangle only, raw values
    # For each feature (column), check if any other feature in its column has |r| > threshold
    to_drop = set(int(col) for col in range(corr_raw.shape[1])
                  if np.any(np.abs(upper[:col, col]) > args.threshold))
    keep_indices = [i for i in range(hidden_dim) if i not in to_drop]

    log.info(f"  Correlation threshold  : |r| > {args.threshold}")
    log.info(f"  Original features      : {hidden_dim}")
    log.info(f"  Features dropped       : {len(to_drop)}")
    log.info(f"  Features retained      : {len(keep_indices)}")

    # -- Heatmap of correlation matrix (sampled 50×50 block for readability) --
    log.info("  Saving correlation heatmap...")
    heatmap_size = min(50, hidden_dim)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Before filtering
    sub_before = corr_matrix.iloc[:heatmap_size, :heatmap_size].values
    im0 = axes[0].imshow(sub_before, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f'Correlation (Before) — first {heatmap_size} features', fontweight='bold')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Feature Index')
    plt.colorbar(im0, ax=axes[0])

    # After filtering (on the kept subset)
    kept_50 = [i for i in keep_indices if i < heatmap_size][:heatmap_size]
    sub_after = corr_matrix.iloc[kept_50, kept_50].values
    im1 = axes[1].imshow(sub_after, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f'Correlation (After filtering) — first {len(kept_50)} kept features', fontweight='bold')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Feature Index')
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(f'Pearson Correlation Heatmap  |  Threshold = {args.threshold}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    heatmap_path = os.path.join(args.report_dir, 'pearson_correlation_heatmap.png')
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved heatmap → {heatmap_path}")

    # -- Feature retention bar chart --
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    bars = ax2.bar(['Original', 'Dropped', 'Retained'],
                   [hidden_dim, len(to_drop), len(keep_indices)],
                   color=['tab:blue', 'tab:red', 'tab:green'], width=0.5)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('Number of Features')
    ax2.set_title(f'Feature Count — Pearson Filtering (threshold={args.threshold})', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    bar_path = os.path.join(args.report_dir, 'feature_retention_bar.png')
    fig2.savefig(bar_path, dpi=150)
    plt.close(fig2)
    log.info(f"  Saved feature retention bar chart → {bar_path}")

    # Free large correlation structures before full-dataset pass
    del X_sample, corr_matrix, upper, sample_residues
    gc.collect()

    # ------------------------------------------------------------------
    print("\n[3/4] Apply filtering to full dataset & write PKL")
    print("-" * 60)
    log.info(f"  Applying feature mask (keeping {len(keep_indices)} dims) to all {num_proteins:,} proteins...")

    filtered_embeddings = {}
    for pdb_id, emb in tqdm(embeddings.items(), desc="  Filtering features"):
        filtered_embeddings[pdb_id] = emb[:, keep_indices]   # (seq_len, n_kept)

    log.info(f"  Saving filtered PKL → {args.output_pkl}")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(filtered_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    keep_indices_path = 'models/keep_indices.pkl'
    os.makedirs('models', exist_ok=True)
    with open(keep_indices_path, 'wb') as f:
        pickle.dump(keep_indices, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  Saved inference mask → {keep_indices_path}")

    # Free raw embeddings; we only need filtered ones for CSV writing
    del embeddings
    gc.collect()

    # ------------------------------------------------------------------
    print("\n[4/4] Write residue-level CSV & generate report")
    print("-" * 60)
    log.info(f"  Writing residue-level CSV → {args.output_csv}")

    BATCH_SIZE = 200   # proteins per write — balances RAM vs I/O calls
    feat_cols = [f'ProtT5_dim_{i}' for i in keep_indices]
    header_cols = ['PDB_ID', 'Residue_Index', 'Amino_Acid', 'Label'] + feat_cols

    first_write = True
    proteins_written = 0
    residues_written = 0
    skipped = 0

    batch_embs   = []   # list of numpy arrays (seq_len, n_kept)
    batch_pdb    = []   # pdb_id repeated seq_len times
    batch_ridx   = []   # residue indices
    batch_aa     = []   # amino acid chars
    batch_labels = []   # SST3 label chars

    items     = list(filtered_embeddings.items())
    csv_limit = args.csv_limit if args.csv_limit > 0 else len(items)
    items_csv = items[:csv_limit]
    log.info(f"  Writing {len(items_csv):,} proteins to CSV (full dataset in PKL)...")

    def _flush_batch(first_write):
        if not batch_embs:
            return first_write
        emb_stack = np.vstack(batch_embs)
        meta_arr  = np.column_stack([batch_pdb, batch_ridx, batch_aa, batch_labels])
        full = np.hstack([meta_arr, emb_stack.astype(str)])
        df = pd.DataFrame(full, columns=header_cols)
        df.to_csv(args.output_csv,
                  mode='w' if first_write else 'a',
                  header=first_write, index=False)
        return False

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

        if proteins_written % BATCH_SIZE == 0:
            first_write = _flush_batch(first_write)
            batch_embs.clear(); batch_pdb.clear()
            batch_ridx.clear(); batch_aa.clear(); batch_labels.clear()

    # flush any remaining proteins
    first_write = _flush_batch(first_write)

    log.info(f"  Proteins written to CSV : {proteins_written:,} / {csv_limit:,} (limit)")
    log.info(f"  Residues written        : {residues_written:,}")
    log.info(f"  Proteins skipped        : {skipped}")
    log.info(f"  CSV saved -> {args.output_csv}")

    # Report
    report_text = f"""============================================================
Phase 5 -- Pearson Correlation Feature Filtering Report
============================================================

INPUT
----------------------------------------
  Embedding PKL  : {args.input}
  Labelled CSV   : {labelled_csv}
  Total Proteins : {num_proteins:,}

CORRELATION ANALYSIS
----------------------------------------
  Sample Size (proteins)   : {len(sample_keys)}
  Correlation Threshold    : |r| > {args.threshold}
  Original Feature Dims    : {hidden_dim}
  Features Dropped         : {len(to_drop)}
  Features Retained        : {len(keep_indices)}
  Retention Rate           : {100*len(keep_indices)/hidden_dim:.1f}%

OUTPUT
----------------------------------------
  Filtered PKL   : {args.output_pkl}
  Filtered CSV   : {args.output_csv}
  Proteins written: {proteins_written:,}
  Residues written: {residues_written:,}
  Proteins skipped: {skipped}

VISUALISATIONS
----------------------------------------
  Correlation Heatmap      : {heatmap_path}
  Feature Retention Chart  : {bar_path}

============================================================
Phase 5 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, 'feature_filtering_phase_5_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    log.info(f"  Report saved → {report_path}")


if __name__ == "__main__":
    main()
