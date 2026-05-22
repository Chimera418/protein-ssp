import os
import gc
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA, PCA

"""
Phase 6: PCA Dimensionality Reduction
======================================

Pipeline Steps
--------------
[1/4] Load filtered embeddings from Phase 5 PKL
[2/4] Estimate required n_components for target variance (sample-based PCA)
[3/4] Fit IncrementalPCA on full dataset in chunks, transform all residues
[4/4] Write PCA PKL, residue-level CSV (batched), visualisations & report
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 6: PCA Dimensionality Reduction")
    parser.add_argument('--input',       type=str, default='embeddings/filtered_embeddings.pkl',
                        help="Path to filtered PKL from Phase 5")
    parser.add_argument('--output-csv',  type=str, default='data/pca_protein_embeddings.csv',
                        help="Path to save residue-level PCA CSV")
    parser.add_argument('--output-pkl',  type=str, default='embeddings/pca_embeddings.pkl',
                        help="Path to save PCA embeddings PKL")
    parser.add_argument('--pca-model',   type=str, default='models/pca_model.pkl',
                        help="Path to save fitted IncrementalPCA model")
    parser.add_argument('--report-dir',  type=str, default='output/phase_6',
                        help="Directory for visualisations and report")
    parser.add_argument('--variance',    type=float, default=0.95,
                        help="Target cumulative explained variance (default 0.95 = 95%%)")
    parser.add_argument('--chunk-size',  type=int, default=200,
                        help="Proteins per chunk for IncrementalPCA partial_fit")
    parser.add_argument('--batch-size',  type=int, default=200,
                        help="Proteins per batch when writing CSV")
    parser.add_argument('--csv-limit',   type=int, default=200,
                        help="Max proteins to write to CSV (0 = all). PKL always saves full dataset.")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    # ------------------------------------------------------------------
    print("\n[1/4] Load filtered embeddings")
    print("-" * 60)
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"PKL not found: {args.input}. Run Phase 5 first.")

    log.info(f"  Loading embeddings from {args.input} ...")
    with open(args.input, 'rb') as f:
        embeddings = pickle.load(f)

    # Phase 5 saves as  { pdb_id : numpy_array(seq_len, n_features) }
    protein_keys = list(embeddings.keys())
    num_proteins = len(protein_keys)
    n_features = next(iter(embeddings.values())).shape[1]
    log.info(f"  Loaded {num_proteins:,} proteins  |  {n_features} features per residue")

    # Metadata lookup for seq/label during CSV write
    labelled_csv = 'data/protein_labelled_curated.csv'
    if not os.path.exists(labelled_csv):
        raise FileNotFoundError(f"Labelled CSV not found: {labelled_csv}. Run Phase 3 first.")
    df_meta = pd.read_csv(labelled_csv)
    df_meta['key'] = df_meta['pdb_id'].astype(str) + '_' + df_meta['chain_code'].astype(str)
    meta_lookup = df_meta.set_index('key')[['seq', 'sst3']].to_dict('index')
    log.info(f"  Metadata lookup built for {len(meta_lookup):,} proteins.")

    # ------------------------------------------------------------------
    print("\n[2/4] Estimate n_components for target variance")
    print("-" * 60)
    sample_size = min(1000, num_proteins)
    sample_keys = protein_keys[:sample_size]
    log.info(f"  Fitting standard PCA on {sample_size} proteins to estimate n_components ...")
    X_sample = np.vstack([embeddings[k] for k in sample_keys])
    log.info(f"  Sample residue matrix shape: {X_sample.shape}")

    pca_est = PCA(n_components=args.variance, svd_solver='full')
    pca_est.fit(X_sample)
    n_components = pca_est.n_components_
    cumulative_var = float(np.sum(pca_est.explained_variance_ratio_))
    log.info(f"  Target variance      : {args.variance * 100:.0f}%")
    log.info(f"  Components required  : {n_components}  (out of {n_features})")
    log.info(f"  Cumulative variance  : {cumulative_var * 100:.2f}%")

    # -- Scree / cumulative variance plot --
    log.info("  Saving explained variance plot ...")
    evr_full = pca_est.explained_variance_ratio_
    cumvar = np.cumsum(evr_full)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    ax1.bar(range(1, len(evr_full) + 1), evr_full * 100, color='tab:blue', alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('Scree Plot', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Cumulative variance plot
    ax2.plot(range(1, len(cumvar) + 1), cumvar * 100, color='tab:orange', lw=2)
    ax2.axhline(args.variance * 100, color='tab:red', linestyle='--', lw=1.5,
                label=f'{args.variance*100:.0f}% target')
    ax2.axvline(n_components, color='tab:green', linestyle='--', lw=1.5,
                label=f'{n_components} components')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title('Cumulative Explained Variance', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f'PCA Analysis  |  {n_components} components → {cumulative_var*100:.1f}% variance',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    var_plot_path = os.path.join(args.report_dir, 'explained_variance_plot.png')
    fig.savefig(var_plot_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved variance plot → {var_plot_path}")

    del X_sample, pca_est, evr_full, cumvar
    gc.collect()

    # ------------------------------------------------------------------
    print("\n[3/4] Fit IncrementalPCA + transform all proteins")
    print("-" * 60)
    log.info(f"  Fitting IncrementalPCA(n_components={n_components}) "
             f"in chunks of {args.chunk_size} proteins ...")

    ipca = IncrementalPCA(n_components=n_components)

    for i in tqdm(range(0, num_proteins, args.chunk_size), desc="  Fitting IPCA"):
        chunk_keys = protein_keys[i: i + args.chunk_size]
        X_chunk = np.vstack([embeddings[k] for k in chunk_keys])
        ipca.partial_fit(X_chunk)
        del X_chunk

    total_var = float(np.sum(ipca.explained_variance_ratio_))
    log.info(f"  IncrementalPCA fitted  |  Total explained variance: {total_var * 100:.2f}%")

    # Save PCA model for inference pipeline
    log.info(f"  Saving PCA model → {args.pca_model}")
    with open(args.pca_model, 'wb') as f:
        pickle.dump(ipca, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Transform and store in a flat dict
    log.info("  Transforming all proteins ...")
    pca_embeddings = {}
    for pdb_id in tqdm(protein_keys, desc="  Transforming"):
        pca_embeddings[pdb_id] = ipca.transform(embeddings[pdb_id]).astype(np.float32)

    del embeddings
    gc.collect()

    log.info(f"  Saving PCA embeddings PKL → {args.output_pkl}")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(pca_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    print("\n[4/4] Write residue-level CSV & generate report")
    print("-" * 60)
    log.info(f"  Writing CSV → {args.output_csv}  (batches of {args.batch_size} proteins)")

    pca_cols = [f'PC{i + 1}' for i in range(n_components)]
    header_cols = ['PDB_ID', 'Residue_Index', 'Amino_Acid', 'Label'] + pca_cols

    first_write = True
    proteins_written = 0
    residues_written = 0
    skipped = 0

    batch_embs   = []
    batch_pdb    = []
    batch_ridx   = []
    batch_aa     = []
    batch_labels = []

    items     = list(pca_embeddings.items())
    csv_limit = args.csv_limit if args.csv_limit > 0 else len(items)
    items_csv = items[:csv_limit]
    log.info(f"  Writing {len(items_csv):,} proteins to CSV (full dataset in PKL)...")

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

    first_write = _flush(first_write)   # flush remainder

    log.info(f"  Proteins written to CSV : {proteins_written:,} / {csv_limit:,} (limit)")
    log.info(f"  Residues written        : {residues_written:,}")
    log.info(f"  Proteins skipped        : {skipped}")
    log.info(f"  CSV saved -> {args.output_csv}")

    # Report
    report_text = f"""============================================================
Phase 6 -- PCA Dimensionality Reduction Report
============================================================

INPUT
----------------------------------------
  Filtered Embedding PKL : {args.input}
  Total Proteins Loaded  : {num_proteins:,}
  Original Features      : {n_features}

PCA ANALYSIS
----------------------------------------
  Target Variance          : {args.variance * 100:.0f}%
  Components Selected      : {n_components}
  IncrementalPCA Var (act) : {total_var * 100:.2f}%
  Reduction Ratio          : {n_features} → {n_components} ({100 * n_components / n_features:.1f}% of original)

OUTPUT
----------------------------------------
  PCA Model (pkl)   : {args.pca_model}
  PCA Embeddings    : {args.output_pkl}
  Residue CSV       : {args.output_csv}
  Proteins written  : {proteins_written:,}
  Residues written  : {residues_written:,}
  Proteins skipped  : {skipped}

VISUALISATIONS
----------------------------------------
  Scree + Cumulative Variance Plot : {var_plot_path}

============================================================
Phase 6 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, 'dimensionality_reduction_phase_6_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    log.info(f"  Report saved → {report_path}")


if __name__ == "__main__":
    main()
