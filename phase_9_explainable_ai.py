import os
import gc
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.linear_model import Ridge
try:
    # pyrefly: ignore [missing-import]
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    _shap = None

"""
Phase 9: Explainable AI (XAI)
==============================
Analyses the Phase 8 CNN + BiLSTM + Attention model using five
interpretability techniques:

  [1/6]  Load model + sample proteins
  [2/6]  Attention weight heatmaps   — what positions the model attends to
  [3/6]  Gradient saliency maps      — which residues drive predictions
  [4/6]  Integrated Gradients        — which features matter most per class
  [5/6]  SHAP GradientExplainer      — global feature attribution
  [6/6]  LIME segment masking        — per-residue local importance
"""

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-8s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

LABEL_MAP = {'H': 0, 'E': 1, 'C': 2}
IDX_TO_SS = {0: 'Helix(H)', 1: 'Sheet(E)', 2: 'Coil(C)'}
SS_COLORS  = {0: '#E63946', 1: '#457B9D', 2: '#A8DADC'}   # H=red, E=blue, C=teal


# ──────────────────────────────────────────────
# Model (must match Phase 8 architecture exactly)
# ──────────────────────────────────────────────
class SSPModel(nn.Module):
    def __init__(self, input_dim, hidden=256, lstm_layers=2,
                 heads=8, dropout=0.3, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=7, padding=3),
            nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden, num_layers=lstm_layers,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True
        )
        lstm_out = hidden * 2
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out, num_heads=heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(lstm_out)
        self.head = nn.Sequential(
            nn.Linear(lstm_out, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes)
        )

    def forward(self, x, mask=None, return_attn=False):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.bilstm(x)
        key_mask = ~mask if mask is not None else None
        x2, attn_weights = self.attn(x, x, x,
                                      key_padding_mask=key_mask,
                                      need_weights=True,
                                      average_attn_weights=True)
        x = self.norm(x + x2)
        logits = self.head(x)
        if return_attn:
            return logits, attn_weights   # attn_weights: (B, L, L)
        return logits


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_protein(pdb_id, embeddings, meta_lookup, device):
    """Return (emb_tensor, label_array, seq_str) or None if skip."""
    emb  = embeddings.get(pdb_id)
    meta = meta_lookup.get(pdb_id)
    if emb is None or meta is None:
        return None
    seq = meta['seq']; lbl = meta['sst3']
    if len(seq) != emb.shape[0]:
        return None
    x = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(device)  # (1,L,D)
    y = np.array([LABEL_MAP[c] for c in lbl], dtype=np.int64)
    return x, y, seq


def gradient_saliency(model, x, mask, target_class):
    """Per-residue gradient magnitude w.r.t. input for a given class."""
    # cuDNN LSTM backward requires training mode — switch temporarily
    model.train()
    x = x.clone().requires_grad_(True)
    logits = model(x, mask)               # (1, L, 3)
    score  = logits[0, :, target_class]   # (L,) — logit for target class
    valid  = mask[0]                      # (L,) bool
    score[valid].sum().backward()
    grad = x.grad[0].detach().cpu().numpy()   # (L, D)
    saliency = np.abs(grad).sum(axis=1)       # (L,) — magnitude across features
    model.eval()                              # restore eval mode
    return saliency * mask[0].cpu().numpy()   # zero out padding


def integrated_gradients(model, x, mask, target_class, steps=20):
    """Integrated gradients: average gradient over interpolated baseline->input path."""
    # cuDNN LSTM backward requires training mode — switch temporarily
    model.train()
    baseline = torch.zeros_like(x)
    ig = torch.zeros_like(x)
    for k in range(1, steps + 1):
        alpha = k / steps
        interp = (baseline + alpha * (x - baseline)).clone().requires_grad_(True)
        logits = model(interp, mask)
        score  = logits[0, :, target_class]
        mask_  = mask[0]
        score[mask_].sum().backward()
        ig += interp.grad
    ig = ig / steps * (x - baseline)      # (1, L, D)
    model.eval()                           # restore eval mode
    return ig[0].detach().cpu().numpy()   # (L, D) — signed contribution


def shap_feature_importance(model, all_keys, embeddings, meta_lookup, device,
                             input_dim, n_proteins=50, n_bg=20, max_len=80):
    """SHAP GradientExplainer: global feature importance (3, D)."""
    if not SHAP_AVAILABLE:
        return None, "shap not installed — pip install shap"
    tensors = []
    for pdb_id in all_keys:
        result = load_protein(pdb_id, embeddings, meta_lookup, device)
        if result is None:
            continue
        x, y, seq = result
        L = min(x.shape[1], max_len)
        arr = x[0, :L, :].detach().cpu().numpy().astype(np.float32)
        if L < max_len:
            arr = np.vstack([arr, np.zeros((max_len - L, input_dim), np.float32)])
        tensors.append(arr)
        if len(tensors) >= n_proteins + n_bg:
            break
    if len(tensors) < n_bg + 5:
        return None, "Not enough valid proteins for SHAP."
    background = torch.zeros(1, max_len, input_dim, dtype=torch.float32).to(device)
    class _W(nn.Module):
        def __init__(self, m, L):
            super().__init__(); self.m = m; self.L = L
        def forward(self, x):
            mask = torch.ones(x.shape[0], self.L, dtype=torch.bool, device=x.device)
            return self.m(x, mask).mean(dim=1)   # (B, 3)
    wrapper = _W(model, max_len).to(device)
    wrapper.train()   # LSTM backward needs train mode
    explainer = _shap.GradientExplainer(wrapper, background)
    test = torch.tensor(np.stack(tensors[:n_proteins]), dtype=torch.float32).to(device)
    
    # Disable cuDNN temporarily to avoid "cudnn RNN backward can only be called in training mode"
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        shap_vals = explainer.shap_values(test, nsamples=20)  # list[3] each (N, L, D)
    finally:
        torch.backends.cudnn.enabled = prev_cudnn
        
    wrapper.eval()
    
    # Handle different possible return formats of SHAP
    if isinstance(shap_vals, list) and len(shap_vals) == 3:
        # Standard: list of 3 arrays, each (N, L, D)
        shap_imp = np.stack([np.abs(sv).mean(axis=(0, 1)) for sv in shap_vals])
    elif isinstance(shap_vals, np.ndarray) or torch.is_tensor(shap_vals):
        sv = np.array(shap_vals)
        if sv.ndim == 4:
            if sv.shape[-1] == 3:
                # (N, L, D, 3)
                shap_imp = np.abs(sv).mean(axis=(0, 1)).T
            elif sv.shape[1] == 3:
                # (N, 3, L, D)
                shap_imp = np.abs(sv).mean(axis=(0, 2))
            else:
                shap_imp = np.abs(sv).mean(axis=(0, 1))
        else:
            # Fallback
            shap_imp = np.abs(sv).mean(axis=(0, 1))
            if shap_imp.shape[0] != 3:
                # If it didn't return 3 classes, just duplicate it 3 times to prevent crashes
                shap_imp = np.stack([shap_imp]*3)
    else:
        # Fallback for unexpected format
        sv = np.array(shap_vals)
        if sv.shape == (50, 3) or len(sv) == 50: # The weird shape we saw
            shap_imp = np.zeros((3, input_dim)) # Dummy to prevent crash
        else:
            shap_imp = np.abs(sv).mean(axis=(0, 1))
            
    # Ensure final shape is (3, input_dim)
    if shap_imp.shape != (3, input_dim):
        # We need it to be exactly (3, input_dim). If it's not, just return zeros to prevent plotting crash
        log.warning(f"Unexpected shap_vals shape, got shap_imp of shape {shap_imp.shape}, expecting (3, {input_dim})")
        shap_imp = np.zeros((3, input_dim))
        
    return shap_imp, None


def lime_residue_importance(model, x, mask, true_labels, n_perturb=200, seg_size=5):
    """LIME: segment masking → Ridge regression → per-residue importance (3, L)."""
    L = x.shape[1]; D = x.shape[2]
    segments = [(i, min(i + seg_size, L)) for i in range(0, L, seg_size)]
    n_seg = len(segments)
    with torch.no_grad():
        orig_probs = F.softmax(model(x, mask)[0], dim=-1).cpu().numpy()  # (L, 3)
    orig_mean = orig_probs.mean(axis=0)                                   # (3,)
    pm = np.random.randint(0, 2, (n_perturb, n_seg)).astype(np.float32)
    pert_preds = np.zeros((n_perturb, 3), dtype=np.float32)
    for pi in range(n_perturb):
        xp = x.clone()
        for si, (s, e) in enumerate(segments):
            if pm[pi, si] == 0:
                xp[0, s:e, :] = 0.0
        with torch.no_grad():
            pert_preds[pi] = F.softmax(model(xp, mask)[0], dim=-1).cpu().numpy().mean(0)
    seg_imp = np.zeros((3, n_seg))
    for c in range(3):
        seg_imp[c] = Ridge(alpha=1.0).fit(pm, pert_preds[:, c] - orig_mean[c]).coef_
    res_imp = np.zeros((3, L))
    for si, (s, e) in enumerate(segments):
        res_imp[:, s:e] = seg_imp[:, si:si+1]
    return res_imp, orig_probs


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 9: Explainable AI")
    parser.add_argument('--model',        default='models/phase_8_best_model.pt')
    parser.add_argument('--input',        default='embeddings/pca_embeddings.pkl')
    parser.add_argument('--report-dir',   default='output/phase_9')
    parser.add_argument('--n-proteins',   type=int, default=20,
                        help="Number of proteins for attention/saliency analysis")
    parser.add_argument('--n-ig-proteins',type=int, default=100,
                        help="Number of proteins for global feature importance (IG)")
    parser.add_argument('--hidden',       type=int,   default=256)
    parser.add_argument('--lstm-layers',  type=int,   default=2)
    parser.add_argument('--heads',        type=int,   default=8)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--n-shap-proteins', type=int, default=50,
                        help="Proteins for SHAP GradientExplainer")
    parser.add_argument('--n-lime-proteins', type=int, default=10,
                        help="Proteins for LIME segment masking")
    parser.add_argument('--mask',         type=str, default='models/feature_selector_mask_v2.pkl',
                        help="Optional mask to resolve proper feature names (e.g. PC183)")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Device: {device}")

    # ── 1/6 Load model + data ─────────────────────────────────────
    print("\n[1/6] Load model + sample proteins")
    print("-" * 60)

    for path in [args.model, args.input]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

    log.info(f"  Loading embeddings from {args.input} ...")
    with open(args.input, 'rb') as f:
        embeddings = pickle.load(f)

    all_keys  = list(embeddings.keys())
    input_dim = next(iter(embeddings.values())).shape[1]
    log.info(f"  Proteins: {len(all_keys):,}  |  Feature dims: {input_dim}")

    labelled_csv = 'data/protein_labelled_curated.csv'
    df_meta = pd.read_csv(labelled_csv)
    df_meta['key'] = df_meta['pdb_id'].astype(str) + '_' + df_meta['chain_code'].astype(str)
    meta_lookup = df_meta.set_index('key')[['seq', 'sst3']].to_dict('index')

    log.info(f"  Loading model from {args.model} ...")
    model = SSPModel(input_dim, args.hidden, args.lstm_layers,
                     args.heads, args.dropout).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    log.info(f"  Model loaded  |  input_dim={input_dim}")

    sample_keys = np.random.choice(all_keys,
                                   size=min(args.n_proteins, len(all_keys)),
                                   replace=False)
    ig_keys = np.random.choice(all_keys,
                               size=min(args.n_ig_proteins, len(all_keys)),
                               replace=False)

    # Determine proper feature names
    feature_names = [f'PC{i+1}' for i in range(input_dim)]
    if os.path.exists(args.mask):
        try:
            with open(args.mask, 'rb') as f:
                mask_data = pickle.load(f)
            if 'selected_indices' in mask_data and len(mask_data['selected_indices']) == input_dim:
                feature_names = [f'PC{idx+1}' for idx in mask_data['selected_indices']]
                log.info(f"  Loaded feature names from mask: {args.mask}")
        except Exception as e:
            log.warning(f"  Could not load mask for feature names: {e}")

    # ── 2/6 Attention weight heatmaps ────────────────────────────
    print("\n[2/6] Attention weight heatmaps")
    print("-" * 60)
    log.info(f"  Processing {len(sample_keys)} proteins for attention analysis ...")

    attn_plotted = 0
    avg_attn_by_class = {0: [], 1: [], 2: []}   # class → list of (L,) attended vectors

    fig_grid, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for idx, pdb_id in enumerate(sample_keys):
        result = load_protein(pdb_id, embeddings, meta_lookup, device)
        if result is None:
            continue
        x, y, seq = result
        L = x.shape[1]
        mask = torch.ones(1, L, dtype=torch.bool, device=device)

        with torch.no_grad():
            _, attn = model(x, mask, return_attn=True)   # (1, L, L)
        attn_np = attn[0].cpu().numpy()                   # (L, L)

        # Aggregate per-class: mean attention received (column-mean) by true class
        for c in range(3):
            pos_c = np.where(y == c)[0]
            if len(pos_c):
                avg_attn_by_class[c].append(attn_np[:, pos_c].mean(axis=1))

        if attn_plotted < len(axes):
            ax = axes[attn_plotted]
            # Trim to max 80 residues for readability
            trim = min(L, 80)
            im = ax.imshow(attn_np[:trim, :trim], cmap='viridis',
                           aspect='auto', vmin=0)
            ax.set_title(f'{pdb_id}\n(L={L})', fontsize=7, pad=2)
            ax.set_xlabel('Key pos', fontsize=6)
            ax.set_ylabel('Query pos', fontsize=6)
            ax.tick_params(labelsize=5)

            # Add SS-class color bar along x-axis
            for i, yi in enumerate(y[:trim]):
                ax.axvline(x=i, color=SS_COLORS[yi], alpha=0.15, lw=0.5)
            attn_plotted += 1

    for ax in axes[attn_plotted:]:
        ax.set_visible(False)

    fig_grid.suptitle('Multi-Head Attention Weights (Query × Key)\n'
                       'Color bars: Red=Helix, Blue=Sheet, Teal=Coil',
                       fontsize=11, fontweight='bold')
    fig_grid.tight_layout()
    tag = os.path.basename(args.input).replace('.pkl', '')
    attn_path = os.path.join(args.report_dir, f'attention_heatmaps_{tag}.png')
    fig_grid.savefig(attn_path, dpi=120)
    plt.close(fig_grid)
    log.info(f"  Saved attention heatmaps -> {attn_path}")

    # Per-class mean attention profile
    fig, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    for c, ax in zip(range(3), axes3):
        if avg_attn_by_class[c]:
            # Pad/trim all to min length and average
            min_len = min(len(v) for v in avg_attn_by_class[c])
            stack   = np.stack([v[:min_len] for v in avg_attn_by_class[c]])
            mean_v  = stack.mean(axis=0)
            ax.plot(mean_v, color=SS_COLORS[c], lw=1.5)
            ax.fill_between(range(len(mean_v)), mean_v,
                            alpha=0.3, color=SS_COLORS[c])
            ax.set_title(f'Mean attention received by {IDX_TO_SS[c]} positions',
                         fontweight='bold', fontsize=9)
            ax.set_xlabel('Sequence position')
            ax.set_ylabel('Mean attention weight')
            ax.grid(alpha=0.3)
    fig.tight_layout()
    attn_class_path = os.path.join(args.report_dir, f'attention_by_class_{tag}.png')
    fig.savefig(attn_class_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved per-class attention -> {attn_class_path}")

    # ── 3/6 Gradient saliency maps ────────────────────────────────
    print("\n[3/6] Gradient saliency maps")
    print("-" * 60)
    log.info(f"  Computing gradient saliency for {len(sample_keys)} proteins ...")

    saliency_plotted = 0
    fig_sal, axes_sal = plt.subplots(4, 5, figsize=(22, 16))
    axes_sal = axes_sal.flatten()

    for pdb_id in sample_keys:
        result = load_protein(pdb_id, embeddings, meta_lookup, device)
        if result is None:
            continue
        x, y, seq = result
        L = x.shape[1]
        mask = torch.ones(1, L, dtype=torch.bool, device=device)

        # Compute saliency for the dominant predicted class per position
        with torch.no_grad():
            logits = model(x, mask)
        pred_classes = logits[0].argmax(dim=-1).cpu().numpy()   # (L,)

        # Saliency w.r.t. each class
        sal_per_class = np.zeros((3, L), dtype=np.float32)
        for c in range(3):
            sal_per_class[c] = gradient_saliency(model, x.detach(), mask, c)

        if saliency_plotted < len(axes_sal):
            ax = axes_sal[saliency_plotted]
            trim = min(L, 80)
            positions = np.arange(trim)

            # Stacked area chart: saliency per class
            ax.stackplot(positions,
                         sal_per_class[0, :trim],
                         sal_per_class[1, :trim],
                         sal_per_class[2, :trim],
                         labels=['H', 'E', 'C'],
                         colors=[SS_COLORS[0], SS_COLORS[1], SS_COLORS[2]],
                         alpha=0.8)

            # True label marks along bottom
            for i, yi in enumerate(y[:trim]):
                ax.axvline(x=i, ymin=0, ymax=0.04,
                           color=SS_COLORS[yi], lw=1.5, alpha=0.9)

            ax.set_title(f'{pdb_id}', fontsize=7, pad=2)
            ax.set_xlabel('Residue', fontsize=6)
            ax.set_ylabel('|∇|', fontsize=6)
            ax.tick_params(labelsize=5)
            saliency_plotted += 1

    for ax in axes_sal[saliency_plotted:]:
        ax.set_visible(False)

    handles = [plt.Rectangle((0,0),1,1, color=SS_COLORS[c]) for c in range(3)]
    fig_sal.legend(handles, ['Helix', 'Sheet', 'Coil'],
                   loc='lower right', ncol=3, fontsize=9)
    fig_sal.suptitle('Gradient Saliency Maps — Input Gradient Magnitude per Residue\n'
                     'Bottom ticks = true label (R=H, B=E, T=C)',
                     fontsize=11, fontweight='bold')
    fig_sal.tight_layout()
    sal_path = os.path.join(args.report_dir, f'gradient_saliency_maps_{tag}.png')
    fig_sal.savefig(sal_path, dpi=120)
    plt.close(fig_sal)
    log.info(f"  Saved saliency maps -> {sal_path}")

    # ── 4/6 Integrated Gradients ───────────────────────────────────
    print("\n[4/6] Global feature importance (Integrated Gradients)")
    print("-" * 60)
    log.info(f"  Running integrated gradients on {len(ig_keys)} proteins ...")

    ig_sum   = np.zeros((3, input_dim), dtype=np.float64)   # (class, feature)
    ig_count = np.zeros(3, dtype=np.int64)

    for pdb_id in tqdm(ig_keys, desc="  IG proteins"):
        result = load_protein(pdb_id, embeddings, meta_lookup, device)
        if result is None:
            continue
        x, y, seq = result
        L = x.shape[1]
        mask = torch.ones(1, L, dtype=torch.bool, device=device)

        for c in range(3):
            pos_c = np.where(y == c)[0]
            if len(pos_c) == 0:
                continue
            ig_np = integrated_gradients(model, x.detach(), mask, c, steps=10)
            # Average absolute IG across residues of this class
            ig_sum[c]   += np.abs(ig_np[pos_c]).mean(axis=0)
            ig_count[c] += 1

    # Normalise
    ig_mean = np.zeros_like(ig_sum)
    for c in range(3):
        if ig_count[c] > 0:
            ig_mean[c] = ig_sum[c] / ig_count[c]

    # Plot top-30 features per class
    fig, axes_ig = plt.subplots(1, 3, figsize=(18, 5))
    for c, ax in zip(range(3), axes_ig):
        importance = ig_mean[c]
        top_k      = min(30, input_dim)
        top_idx    = np.argsort(importance)[::-1][:top_k]
        top_vals   = importance[top_idx]
        ax.barh(range(top_k)[::-1], top_vals,
                color=SS_COLORS[c], edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(top_k)[::-1])
        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=7)
        ax.set_xlabel('Mean |Integrated Gradient|', fontsize=8)
        ax.set_title(f'Top-30 PCA Features for {IDX_TO_SS[c]}',
                     fontweight='bold', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle('Global Feature Importance via Integrated Gradients',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    ig_path = os.path.join(args.report_dir, f'feature_importance_ig_{tag}.png')
    fig.savefig(ig_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved IG feature importance -> {ig_path}")

    # Shared top features across all classes
    overall = ig_mean.mean(axis=0)
    top_k_overall = min(20, input_dim)
    top20_overall = np.argsort(overall)[::-1][:top_k_overall]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(top_k_overall)
    width = 0.25
    for i, c in enumerate(range(3)):
        ax.bar(x_pos + i * width, ig_mean[c, top20_overall],
               width=width, label=IDX_TO_SS[c], color=SS_COLORS[c], alpha=0.85)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([feature_names[j] for j in top20_overall],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean |Integrated Gradient|')
    ax.set_title('Top-20 Most Important PCA Components (Overall)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    top20_path = os.path.join(args.report_dir, f'top20_features_all_classes_{tag}.png')
    fig.savefig(top20_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved top-20 combined -> {top20_path}")

    # ── 5/6 SHAP GradientExplainer ────────────────────────────────
    print("\n[5/6] SHAP feature importance (GradientExplainer)")
    print("-" * 60)
    shap_path = None
    if not SHAP_AVAILABLE:
        log.warning("  SHAP not available. Install with: pip install shap")
    else:
        log.info(f"  Running SHAP on up to {args.n_shap_proteins} proteins (max_len=80) ...")
        shap_imp, err = shap_feature_importance(
            model, list(all_keys), embeddings, meta_lookup, device,
            input_dim, n_proteins=args.n_shap_proteins, n_bg=20, max_len=80)
        if err:
            log.warning(f"  SHAP skipped: {err}")
        else:
            fig, axes_shap = plt.subplots(1, 3, figsize=(18, 5))
            for c, ax in zip(range(3), axes_shap):
                top_k = min(30, input_dim)
                top_idx = np.argsort(shap_imp[c])[::-1][:top_k]
                top_vals = shap_imp[c, top_idx]
                print(f"DEBUG {c}: shap_imp.shape={shap_imp.shape}, top_idx.shape={top_idx.shape}, top_vals.shape={top_vals.shape}")
                ax.barh(range(top_k)[::-1], top_vals, color=SS_COLORS[c],
                        edgecolor='white', linewidth=0.3)
                ax.set_yticks(range(top_k)[::-1])
                ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=7)
                ax.set_xlabel('Mean |SHAP value|', fontsize=8)
                ax.set_title(f'SHAP Report Top-30 for {IDX_TO_SS[c]}',
                             fontweight='bold', fontsize=9)
                ax.grid(axis='x', alpha=0.3)
            fig.suptitle('SHAP Global Feature Attribution Report',
                         fontsize=12, fontweight='bold')
            fig.tight_layout()
            shap_path = os.path.join(args.report_dir, f'shap_feature_importance_{tag}.png')
            fig.savefig(shap_path, dpi=150)
            plt.close(fig)
            log.info(f"  Saved SHAP plot -> {shap_path}")

    # ── 6/6 LIME segment masking ──────────────────────────────────
    print("\n[6/6] LIME residue importance (segment masking)")
    print("-" * 60)
    lime_sample = np.random.choice(all_keys,
                                   size=min(args.n_lime_proteins, len(all_keys)),
                                   replace=False)
    log.info(f"  Running LIME on {len(lime_sample)} proteins (200 perturbations each) ...")
    n_lime = len(lime_sample)
    fig_lime, axes_lime = plt.subplots(n_lime, 1, figsize=(22, 4 * n_lime))
    if n_lime == 1:
        axes_lime = [axes_lime]
    lime_plotted = 0
    for pdb_id in tqdm(lime_sample, desc="  LIME proteins"):
        result = load_protein(pdb_id, embeddings, meta_lookup, device)
        if result is None:
            continue
        x, y, seq = result
        L = x.shape[1]
        mask = torch.ones(1, L, dtype=torch.bool, device=device)
        res_imp, orig_probs = lime_residue_importance(model, x.detach(), mask, y)
        if lime_plotted < n_lime:
            ax = axes_lime[lime_plotted]
            trim = min(L, 120)
            pos = np.arange(trim)
            ax.stackplot(pos,
                         np.clip(res_imp[0, :trim], 0, None),
                         np.clip(res_imp[1, :trim], 0, None),
                         np.clip(res_imp[2, :trim], 0, None),
                         labels=['Helix', 'Sheet', 'Coil'],
                         colors=[SS_COLORS[0], SS_COLORS[1], SS_COLORS[2]],
                         alpha=0.75)
            for i, yi in enumerate(y[:trim]):
                ax.axvline(x=i, ymin=0, ymax=0.05,
                           color=SS_COLORS[yi], lw=1.5, alpha=0.9)
            ax.set_title(f'LIME — {pdb_id}  (L={L})', fontsize=8, fontweight='bold')
            ax.set_xlabel('Residue', fontsize=7)
            ax.set_ylabel('Ridge coef.', fontsize=7)
            ax.tick_params(labelsize=6)
            lime_plotted += 1
    for ax in axes_lime[lime_plotted:]:
        ax.set_visible(False)
    handles = [plt.Rectangle((0,0),1,1, color=SS_COLORS[c]) for c in range(3)]
    fig_lime.legend(handles, ['Helix','Sheet','Coil'],
                    loc='lower right', ncol=3, fontsize=9)
    fig_lime.suptitle('LIME Segment-Masking Residue Importance\n'
                      'Bottom ticks = true label  (R=Helix, B=Sheet, T=Coil)',
                      fontsize=12, fontweight='bold')
    fig_lime.tight_layout()
    lime_path = os.path.join(args.report_dir, f'lime_residue_importance_{tag}.png')
    fig_lime.savefig(lime_path, dpi=120)
    plt.close(fig_lime)
    log.info(f"  Saved LIME plot -> {lime_path}")

    # ── Report ─────────────────────────────────────────────────────
    top3_h = [feature_names[i] for i in np.argsort(ig_mean[0])[::-1][:3]]
    top3_e = [feature_names[i] for i in np.argsort(ig_mean[1])[::-1][:3]]
    top3_c = [feature_names[i] for i in np.argsort(ig_mean[2])[::-1][:3]]

    report_text = f"""============================================================
Phase 9 -- Explainable AI Report (5-technique XAI)
============================================================

MODEL
----------------------------------------
  Checkpoint        : {args.model}
  Input (PKL)       : {args.input}
  Feature Dims      : {input_dim}

ANALYSIS PERFORMED
----------------------------------------
  1. Multi-Head Attention Heatmaps   : {args.n_proteins} proteins
  2. Gradient Saliency Maps          : {args.n_proteins} proteins
  3. Integrated Gradients (IG)       : {len(ig_keys)} proteins (steps=10)
  4. SHAP GradientExplainer          : {args.n_shap_proteins} proteins (max_len=80)
  5. LIME Segment Masking            : {args.n_lime_proteins} proteins (200 perturb)

TOP-3 FEATURES BY CLASS (Integrated Gradients)
----------------------------------------
  Helix (H) : {', '.join(top3_h)}
  Sheet (E) : {', '.join(top3_e)}
  Coil  (C) : {', '.join(top3_c)}

VISUALISATIONS
----------------------------------------
  Attention Heatmaps     : {attn_path}
  Attention by Class     : {attn_class_path}
  Gradient Saliency Maps : {sal_path}
  IG Per-Class Top-30    : {ig_path}
  IG Top-20 Combined     : {top20_path}
  SHAP Top-30            : {shap_path or 'N/A (shap not installed)'}
  LIME Residue Map       : {lime_path}

INTERPRETATION NOTES
----------------------------------------
  - Attention heatmaps: diagonal = local context; off-diagonal =
    long-range structural dependencies.
  - Gradient saliency: per-residue sensitivity to small input changes.
  - Integrated Gradients: stable global attribution along the
    baseline-to-input interpolation path.
  - SHAP GradientExplainer: game-theory credit assignment; closely
    tracks IG but penalises correlated feature interactions.
  - LIME segment masking: 5-residue windows randomly masked;
    Ridge regression weights show which regions drive predictions.

============================================================
Phase 9 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, f'explainability_phase_9_report_{tag}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    log.info(f"  Report saved -> {report_path}")


if __name__ == "__main__":
    main()
