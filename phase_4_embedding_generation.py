import os
import gc
import torch
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc as auc_fn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
Phase 4: Transformer Embedding Generation & Comparison
======================================================

Pipeline Steps
--------------
[1/5] Load labelled sequence dataset
[2/5] Prepare subset (100 sequences) for model comparison
[3/5] Evaluate multiple transformer models & Generate partial CSVs + ROC graphs
[4/5] Select Best Model and Generate Full Embeddings ONLY for the winner
[5/5] Save winning Embeddings to .pkl and Generate Report
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

MODELS_TO_TEST = [
    "Rostlab/prot_bert_bfd",
    "Rostlab/prot_t5_xl_uniref50",
    "Rostlab/prot_albert",
    "yarongef/DistilProtBert"
]

CLASS_COLORS = {'H': 'tab:red', 'E': 'tab:blue', 'C': 'tab:green'}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def space_sequence(seq):
    return " ".join(list(seq))

def extract_embeddings(model_name, sequences, device, batch_size=2, max_length=1024):
    log.info(f"  Loading model: {model_name}")
    try:
        if "prot_bert_bfd" in model_name.lower():
            from transformers import BertTokenizer, BertModel
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
            model = BertModel.from_pretrained(model_name)
        elif "prot_albert" in model_name.lower():
            from transformers import AlbertTokenizer, AlbertModel
            tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
            model = AlbertModel.from_pretrained(model_name)
        elif "t5" in model_name.lower():
            from transformers import T5EncoderModel
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False)
            model = T5EncoderModel.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False)
            model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        log.error(f"  Failed to load {model_name}: {e}")
        return None

    model = model.to(device)
    model.eval()

    all_embeddings = []
    spaced_seqs = [space_sequence(s) for s in sequences]

    with torch.no_grad():
        for i in tqdm(range(0, len(spaced_seqs), batch_size), desc=f"  Extracting {model_name}"):
            batch = spaced_seqs[i:i+batch_size]
            encoded = tokenizer(
                batch, add_special_tokens=True, padding=True,
                truncation=True, max_length=max_length, return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            for j in range(len(batch)):
                seq_len = len(sequences[i+j])
                if "t5" in model_name.lower():
                    emb = hidden_states[j, :seq_len, :].cpu().numpy()
                else:
                    emb = hidden_states[j, 1:seq_len+1, :].cpu().numpy()
                all_embeddings.append(emb)

            del input_ids, attention_mask, outputs, hidden_states
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_embeddings

def save_partial_csv(df_subset, subset_embeddings, model_name, output_dir, target_rows=1000):
    safe_name = model_name.replace("/", "_")
    csv_path = os.path.join(output_dir, f"{safe_name}_partial.csv")

    rows_written = 0
    first = True

    for row_idx, (row, emb) in enumerate(zip(df_subset.to_dict('records'), subset_embeddings)):
        if rows_written >= target_rows:
            break
        seq = row['seq']
        lbl = row['sst3']
        pdb_id = f"{row['pdb_id']}_{row['chain_code']}"

        df_prot = pd.DataFrame(emb, columns=[f'feat_{i}' for i in range(emb.shape[1])])
        df_prot.insert(0, 'PDB_ID', pdb_id)
        df_prot.insert(1, 'Residue_Index', range(1, len(seq) + 1))
        df_prot.insert(2, 'Amino_Acid', list(seq))
        df_prot.insert(3, 'Label', list(lbl))

        mode = 'w' if first else 'a'
        header = first
        df_prot.to_csv(csv_path, mode=mode, header=header, index=False)
        first = False
        rows_written += len(seq)

    log.info(f"    Saved partial CSV (~{rows_written} rows) to: {csv_path}")

def _plot_roc_on_ax(ax, encoder, y_test, probs, model_name):
    """Helper: draw per-class ROC curves on a given matplotlib Axes object."""
    for i, class_label in enumerate(encoder.classes_):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), probs[:, i])
        class_auc = auc_fn(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[class_label], lw=2,
                label=f'Class {class_label} (AUC = {class_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(model_name.split('/')[-1], fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

def save_individual_roc(encoder, y_test, probs, model_name, output_dir):
    """Save a standalone high-quality ROC graph for one model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_roc_on_ax(ax, encoder, y_test, probs, model_name)
    ax.set_title(f'ROC Curve — {model_name}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    safe_name = model_name.replace("/", "_")
    path = os.path.join(output_dir, f"roc_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"    Saved individual ROC graph → {path}")
    return path

def save_combined_grid_roc(roc_data, encoder, output_dir):
    """Save a 2×2 grid comparing all model ROC curves."""
    n = len(roc_data)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    axes = axes.flatten()

    for idx, (model_name, y_test, probs) in enumerate(roc_data):
        _plot_roc_on_ax(axes[idx], encoder, y_test, probs, model_name)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('ROC Curve Comparison — All Models (Per-Class)', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "roc_combined_grid.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"    Saved combined 2×2 ROC grid → {path}")

def save_macro_roc(roc_data, encoder, output_dir):
    """Save a single macro-average ROC comparison across all models."""
    fig, ax = plt.subplots(figsize=(10, 7))
    n_classes = len(encoder.classes_)
    palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',
               'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    for idx, (model_name, y_test, probs) in enumerate(roc_data):
        fpr_dict, tpr_dict = {}, {}
        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve((y_test == i).astype(int), probs[:, i])

        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= n_classes
        macro_auc = auc_fn(all_fpr, mean_tpr)

        short_name = model_name.split('/')[-1]
        ax.plot(all_fpr, mean_tpr, color=palette[idx % len(palette)],
                lw=2.5, label=f'{short_name}  (Macro AUC = {macro_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('Macro-Average ROC Comparison — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "roc_macro_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"    Saved macro-average ROC comparison → {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', default='data/protein_labelled_curated.csv')
    parser.add_argument('--embeddings-dir', default='embeddings')
    parser.add_argument('--report-dir', default='output/phase_4')
    parser.add_argument('--subset-size', type=int, default=100,
                        help="Number of sequences for quick model comparison")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="Batch size for embedding extraction")
    args = parser.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    # ------------------------------------------------------------------
    print("\n[1/5] Load dataset")
    print("-" * 60)
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}. Run Phase 3 first.")
    df = pd.read_csv(args.input_csv)
    log.info(f"  Loaded {len(df):,} sequences from {args.input_csv}")

    # ------------------------------------------------------------------
    print("\n[2/5] Prepare subset for model comparison")
    print("-" * 60)
    subset_df = df.sample(n=min(args.subset_size, len(df)), random_state=42)
    sequences_subset = subset_df['seq'].tolist()
    labels_subset = subset_df['sst3'].tolist()
    log.info(f"  Sampled {len(subset_df)} sequences for comparing transformer models.")

    device = get_device()
    log.info(f"  Using device: {device}")

    # ------------------------------------------------------------------
    print("\n[3/5] Evaluate Models (5-Fold Stratified CV)")
    print("-" * 60)
    results = []
    best_model_name = None
    best_f1 = 0.0

    encoder = LabelEncoder()
    encoder.fit(['H', 'E', 'C'])

    roc_data = []
    model_cv_scores = {}
    model_auc = {}
    
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import ttest_rel
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name in MODELS_TO_TEST:
        log.info(f"  Evaluating {model_name}...")
        subset_embeddings = extract_embeddings(model_name, sequences_subset, device,
                                               batch_size=args.batch_size)
        if subset_embeddings is None or len(subset_embeddings) == 0:
            continue

        # Save 1000-row preview CSV
        save_partial_csv(subset_df, subset_embeddings, model_name,
                         args.report_dir, target_rows=1000)

        X = np.vstack(subset_embeddings)
        y = np.array(list("".join(labels_subset)))

        if len(X) != len(y):
            log.warning(f"  Length mismatch for {model_name}. Skipping.")
            continue

        y_encoded = encoder.transform(y)
        
        fold_f1_scores = []
        all_y_test = []
        all_probs = []
        
        for train_idx, test_idx in skf.split(X, y_encoded):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced')
            clf.fit(X_train, y_train)

            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)

            f1 = f1_score(y_test, preds, average='macro')
            fold_f1_scores.append(f1)
            
            all_y_test.extend(y_test)
            all_probs.append(probs)

        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
        model_cv_scores[model_name] = fold_f1_scores
        
        all_y_test = np.array(all_y_test)
        all_probs = np.vstack(all_probs)
        
        try:
            auc = roc_auc_score(all_y_test, all_probs, multi_class='ovr')
        except ValueError:
            auc = 0.0
            
        model_auc[model_name] = auc

        log.info(f"  {model_name} → Mean F1: {mean_f1:.4f} (±{std_f1:.4f}), AUC: {auc:.4f}")

        # ── Individual ROC graph (one clean figure per model) ──────────
        save_individual_roc(encoder, all_y_test, all_probs, model_name, args.report_dir)

        # Stash for combined plots generated after all models are done
        roc_data.append((model_name, all_y_test, all_probs))

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model_name = model_name

    # Compute p-values relative to ProtT5
    baseline = "Rostlab/prot_t5_xl_uniref50"
    for model_name in MODELS_TO_TEST:
        if model_name not in model_cv_scores:
            continue
        
        mean_f1 = np.mean(model_cv_scores[model_name])
        std_f1 = np.std(model_cv_scores[model_name])
        
        if model_name == baseline or baseline not in model_cv_scores:
            p_val_str = "—"
            test_applied = "5-fold stratified CV (ref.)"
        else:
            _, p_val = ttest_rel(model_cv_scores[model_name], model_cv_scores[baseline])
            p_val_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            test_applied = "Paired t-test vs ProtT5"
            
        results.append({
            'Model': model_name,
            'Mean CV F1': mean_f1,
            'Std': std_f1,
            'Test': test_applied,
            'p-val': p_val_str,
            'ROC_AUC': model_auc[model_name]
        })

    # ── Combined 2×2 grid & macro-average (generated once, after all models) ──
    if roc_data:
        log.info("  Generating combined ROC graphs for all evaluated models...")
        save_combined_grid_roc(roc_data, encoder, args.report_dir)
        save_macro_roc(roc_data, encoder, args.report_dir)

    # ------------------------------------------------------------------
    print("\n[4/5] Select Best Model and Generate Full Embeddings ONLY for the winner")
    print("-" * 60)
    if best_model_name is None:
        raise RuntimeError("No models could be successfully evaluated.")

    log.info(f"  Best Model Selected: {best_model_name} (F1: {best_f1:.4f})")
    res_df = pd.DataFrame(results).sort_values(by='Mean CV F1', ascending=False)

    log.info(f"  Generating full embeddings using {best_model_name} for ALL {len(df):,} sequences...")
    all_sequences = df['seq'].tolist()
    full_embeddings = extract_embeddings(best_model_name, all_sequences, device,
                                         batch_size=args.batch_size)

    # ------------------------------------------------------------------
    print("\n[5/5] Save Embeddings to .pkl and Generate Report")
    print("-" * 60)

    safe_model_name = best_model_name.replace("/", "_")
    output_pkl = os.path.join(args.embeddings_dir, f"{safe_model_name}.pkl")

    emb_dict = {
        f"{row['pdb_id']}_{row['chain_code']}": emb
        for row, emb in zip(df.to_dict('records'), full_embeddings)
    }

    log.info("  Pickling best model embeddings (this may take a moment)...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"  Embeddings saved to: {output_pkl}")

    report_text = f"""============================================================
Phase 4 -- Transformer Embedding Generation Report
============================================================

MODEL COMPARISON (Subset Size: {args.subset_size})
----------------------------------------
{res_df.to_string(index=False)}

*Note: Partial preview CSV files (~1000 rows) and ROC graphs were
generated for ALL models tested above and placed in {args.report_dir}/*

ROC GRAPHS GENERATED
----------------------------------------
  Per-Model (individual)  : roc_<ModelName>.png
  Combined 2x2 Grid       : roc_combined_grid.png
  Macro-Avg Comparison    : roc_macro_comparison.png

BEST MODEL SELECTION
----------------------------------------
Selected Model                : {best_model_name}
Baseline Macro F1-Score       : {best_f1:.4f}

FULL EMBEDDINGS (WINNER ONLY)
----------------------------------------
Total Sequences Processed     : {len(full_embeddings):,}
Saved Location                : {output_pkl}

============================================================
Phase 4 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, "embedding_phase_4_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info(f"  Report saved to: {report_path}")

if __name__ == "__main__":
    main()
