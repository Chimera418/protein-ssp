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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve, auc)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

"""
Phase 8: CNN + BiLSTM + Attention Deep Learning Model
======================================================

Pipeline Steps
--------------
[1/5] Load features PKL + metadata, split train/val/test
[2/5] Define model, loss, optimizer
[3/5] Train with early stopping
[4/5] Evaluate on test set (Q3, F1, confusion matrix, ROC)
[5/5] Save model, plots, and report
"""

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-8s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

LABEL_MAP = {'H': 0, 'E': 1, 'C': 2}
IDX_MAP   = {0: 'Helix(H)', 1: 'Sheet(E)', 2: 'Coil(C)'}


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, keys, embeddings, meta_lookup, max_len=512):
        self.samples = []
        skipped = 0
        for k in keys:
            emb  = embeddings.get(k)
            meta = meta_lookup.get(k)
            if emb is None or meta is None:
                skipped += 1; continue
            seq = meta['seq']; lbl = meta['sst3']
            if len(seq) != emb.shape[0]:
                skipped += 1; continue
            if max_len is not None and len(seq) > max_len:
                emb = emb[:max_len]
                lbl = lbl[:max_len]
            y = np.array([LABEL_MAP[c] for c in lbl], dtype=np.int64)
            self.samples.append((emb.astype(np.float32), y))
        if skipped:
            log.info(f"    Skipped {skipped} proteins (key/length mismatch)")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_fn(batch):
    embs, lbls = zip(*batch)
    max_len = max(e.shape[0] for e in embs)
    B, D = len(embs), embs[0].shape[1]

    X    = np.zeros((B, max_len, D), dtype=np.float32)
    Y    = np.full((B, max_len), -1, dtype=np.int64)
    mask = np.zeros((B, max_len), dtype=bool)

    for i, (e, l) in enumerate(zip(embs, lbls)):
        n = e.shape[0]
        X[i, :n] = e
        Y[i, :n] = l
        mask[i, :n] = True

    return (torch.from_numpy(X),
            torch.from_numpy(Y),
            torch.from_numpy(mask))


# ──────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha   # (num_classes,) tensor or None
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits : (B, L, C)   targets : (B, L)  [-1 = padding]
        B, L, C = logits.shape
        logits_f = logits.reshape(-1, C)
        targets_f = targets.reshape(-1)
        valid = targets_f >= 0
        logits_f = logits_f[valid]
        targets_f = targets_f[valid]
        ce = F.cross_entropy(logits_f, targets_f,
                             weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class SSPModel(nn.Module):
    def __init__(self, input_dim, hidden=256, lstm_layers=2,
                 heads=8, dropout=0.3, num_classes=3):
        super().__init__()

        # CNN block — local motif detection
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=7, padding=3),
            nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
        )

        # BiLSTM block — long-range dependencies
        self.bilstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=lstm_layers, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True
        )

        # Multi-head self-attention — global context
        lstm_out = hidden * 2
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out, num_heads=heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(lstm_out)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(lstm_out, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes)
        )

    def forward(self, x, mask=None):
        # x: (B, L, D)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # CNN
        x, _ = self.bilstm(x)                              # BiLSTM
        key_mask = ~mask if mask is not None else None     # True = ignore
        x2, _ = self.attn(x, x, x, key_padding_mask=key_mask)
        x = self.norm(x + x2)                              # residual
        return self.head(x)                                 # (B, L, C)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def get_flat_preds(model, loader, device):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for X, Y, M in loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            logits = model(X, M)                        # (B, L, C)
            probs  = F.softmax(logits, dim=-1)
            valid  = Y >= 0
            all_true.extend(Y[valid].cpu().numpy())
            all_pred.extend(logits.argmax(-1)[valid].cpu().numpy())
            all_prob.extend(probs[valid].cpu().numpy())
    return (np.array(all_true),
            np.array(all_pred),
            np.array(all_prob))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 8: SSP Deep Learning")
    parser.add_argument('--input',       default='embeddings/final_features.pkl',
                        help="Features PKL (Phase 7 = 107 dims | Phase 6 = 739 dims)")
    parser.add_argument('--report-dir',  default='output/phase_8')
    parser.add_argument('--model-dir',   default='models')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch-size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--hidden',      type=int,   default=256)
    parser.add_argument('--lstm-layers', type=int,   default=2)
    parser.add_argument('--heads',       type=int,   default=8)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--patience',    type=int,   default=7)
    parser.add_argument('--gamma',       type=float, default=2.0,
                        help="Focal loss gamma")
    parser.add_argument('--eval-only',   action='store_true',
                        help="Skip training, just evaluate the best saved model")
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--val-ratio',   type=float, default=0.1)
    parser.add_argument('--test-ratio',  type=float, default=0.1)
    parser.add_argument('--max-len',     type=int,   default=512,
                        help="Trim sequences longer than this to prevent CUDA OOM")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.model_dir,  exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Device: {device}")

    # ── 1/5 Load data ──────────────────────────────────────────────
    print("\n[1/5] Load data & split")
    print("-" * 60)
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"PKL not found: {args.input}")

    log.info(f"  Loading {args.input} ...")
    with open(args.input, 'rb') as f:
        embeddings = pickle.load(f)

    all_keys   = list(embeddings.keys())
    num_prot   = len(all_keys)
    input_dim  = next(iter(embeddings.values())).shape[1]
    log.info(f"  Proteins: {num_prot:,}  |  Feature dims: {input_dim}")

    labelled_csv = 'data/protein_labelled_curated.csv'
    df_meta = pd.read_csv(labelled_csv)
    df_meta['key'] = df_meta['pdb_id'].astype(str) + '_' + df_meta['chain_code'].astype(str)
    meta_lookup = df_meta.set_index('key')[['seq', 'sst3']].to_dict('index')

    # Stratified split by protein
    train_keys, tmp = train_test_split(all_keys, test_size=args.val_ratio + args.test_ratio,
                                       random_state=args.seed)
    val_ratio_adj = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_keys, test_keys = train_test_split(tmp, test_size=1 - val_ratio_adj,
                                           random_state=args.seed)
    log.info(f"  Train/Val/Test split: {len(train_keys)} / {len(val_keys)} / {len(test_keys)}")

    train_ds = ProteinDataset(train_keys, embeddings, meta_lookup, max_len=args.max_len)
    val_ds   = ProteinDataset(val_keys,   embeddings, meta_lookup, max_len=args.max_len)
    test_ds  = ProteinDataset(test_keys,  embeddings, meta_lookup, max_len=args.max_len)

    # Free the source PKL dict — datasets have their own numpy copies in self.samples
    del embeddings, df_meta
    gc.collect()
    log.info(f"  Dataset sizes: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")


    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Class weights for focal loss
    all_labels = []
    for _, y in train_ds.samples:
        all_labels.extend(y.tolist())
    counts    = np.bincount(all_labels, minlength=3).astype(float)
    weights   = torch.tensor(1.0 / (counts / counts.sum()),
                              dtype=torch.float32).to(device)
    weights  /= weights.sum()
    log.info(f"  Class counts (H/E/C): {counts.astype(int).tolist()}")
    log.info(f"  Class weights:        {weights.cpu().numpy().round(4).tolist()}")

    # ── 2/5 Model ──────────────────────────────────────────────────
    print("\n[2/5] Build model")
    print("-" * 60)
    model = SSPModel(input_dim, args.hidden, args.lstm_layers,
                     args.heads, args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Architecture : CNN → BiLSTM({args.lstm_layers}L) → MHA({args.heads}h) → FC")
    log.info(f"  Input dim    : {input_dim}  |  Hidden: {args.hidden}")
    log.info(f"  Trainable params: {total_params:,}")

    criterion = FocalLoss(alpha=weights, gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── 3/5 Train ──────────────────────────────────────────────────
    print("\n[3/5] Train")
    print("-" * 60)
    best_val_f1 = -1
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_q3': []}
    tag = os.path.basename(args.input).replace('.pkl', '')
    best_model_path = os.path.join(args.model_dir, f'phase_8_best_model_{tag}.pt')

    for epoch in range(1, args.epochs + 1):
        if getattr(args, 'eval_only', False):
            log.info("Skipping training (--eval-only flag is set). Loading best model directly.")
            break

        # — Train —
        model.train()
        train_loss = 0.0
        for X, Y, M in tqdm(train_loader, desc=f"  Epoch {epoch:02d} train", leave=False):
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            optimizer.zero_grad()
            logits = model(X, M)
            loss   = criterion(logits, Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # — Validate —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y, M in val_loader:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                logits = model(X, M)
                val_loss += criterion(logits, Y).item()
        val_loss /= len(val_loader)

        vt, vp, _ = get_flat_preds(model, val_loader, device)
        val_f1 = f1_score(vt, vp, average='macro', zero_division=0)
        val_q3 = (vt == vp).mean() * 100

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_q3'].append(val_q3)

        log.info(f"  Epoch {epoch:02d}/{args.epochs} | "
                 f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                 f"val_F1={val_f1:.4f}  val_Q3={val_q3:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            torch.save(model.state_dict(), best_model_path)
            log.info(f"    ✓ New best model saved (val_F1={val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log.info(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── 4/5 Evaluate ───────────────────────────────────────────────
    print("\n[4/5] Evaluate on test set")
    print("-" * 60)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    true, pred, prob = get_flat_preds(model, test_loader, device)

    q3  = (true == pred).mean() * 100
    mf1 = f1_score(true, pred, average='macro',  zero_division=0)
    wf1 = f1_score(true, pred, average='weighted', zero_division=0)
    try:
        auc_score = roc_auc_score(true, prob, multi_class='ovr', average='macro')
    except Exception:
        auc_score = float('nan')

    log.info(f"  Q3 Accuracy  : {q3:.2f}%")
    log.info(f"  Macro F1     : {mf1:.4f}")
    log.info(f"  Weighted F1  : {wf1:.4f}")
    log.info(f"  ROC-AUC (OvR): {auc_score:.4f}")
    log.info("\n" + classification_report(true, pred,
             target_names=['Helix(H)', 'Sheet(E)', 'Coil(C)'], zero_division=0))

    # — Confusion matrix —
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(['H','E','C']); ax.set_yticklabels(['H','E','C'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix  |  Q3={q3:.1f}%', fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()*0.6 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = os.path.join(args.report_dir, f'confusion_matrix_{tag}.png')
    fig.savefig(cm_path, dpi=150); plt.close(fig)
    log.info(f"  Saved confusion matrix -> {cm_path}")

    # — Training history —
    epochs_done = len(history['train_loss'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = range(1, epochs_done + 1)
    ax1.plot(x, history['train_loss'], label='Train Loss', color='tab:blue')
    ax1.plot(x, history['val_loss'],   label='Val Loss',   color='tab:orange')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(x, history['val_q3'], label='Val Q3 (%)', color='tab:green')
    ax2.plot(x, [v*100 for v in history['val_f1']], label='Val F1×100', color='tab:red', linestyle='--')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics', fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    hist_path = os.path.join(args.report_dir, f'training_history_{tag}.png')
    fig.savefig(hist_path, dpi=150); plt.close(fig)
    log.info(f"  Saved training history -> {hist_path}")

    # — ROC curves —
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (cls, col) in enumerate(zip(['Helix(H)', 'Sheet(E)', 'Coil(C)'], colors)):
        fpr, tpr, _ = roc_curve((true == i).astype(int), prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{cls} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves (OvR)  |  Macro AUC={auc_score:.3f}', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_path = os.path.join(args.report_dir, f'roc_curves_{tag}.png')
    fig.savefig(roc_path, dpi=150); plt.close(fig)
    log.info(f"  Saved ROC curves -> {roc_path}")

    # ── 5/5 Report ─────────────────────────────────────────────────
    print("\n[5/5] Save report")
    print("-" * 60)
    cr = classification_report(true, pred,
         target_names=['Helix(H)', 'Sheet(E)', 'Coil(C)'], zero_division=0)

    report_text = f"""============================================================
Phase 8 -- CNN + BiLSTM + Attention Model Report
============================================================

INPUT
----------------------------------------
  Feature PKL     : {args.input}
  Feature Dims    : {input_dim}
  Total Proteins  : {num_prot:,}
  Train / Val / Test : {len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}

ARCHITECTURE
----------------------------------------
  CNN (2 layers: k=7, k=5) -> BiLSTM ({args.lstm_layers}L, h={args.hidden}) -> MHA ({args.heads} heads) -> FC
  Trainable Parameters : {total_params:,}
  Loss                 : Focal Loss (gamma={args.gamma})
  Optimizer            : AdamW (lr={args.lr}, wd=1e-4)
  Scheduler            : CosineAnnealingLR
  Dropout              : {args.dropout}

TRAINING
----------------------------------------
  Epochs trained   : {epochs_done}
  Early stopping   : patience={args.patience}
  Best val Macro F1: {best_val_f1:.4f}

TEST SET RESULTS
----------------------------------------
  Q3 Accuracy      : {q3:.2f}%
  Macro F1         : {mf1:.4f}
  Weighted F1      : {wf1:.4f}
  ROC-AUC (OvR)   : {auc_score:.4f}

PER-CLASS REPORT
----------------------------------------
{cr}

OUTPUTS
----------------------------------------
  Best model  : {best_model_path}
  Conf matrix : {cm_path}
  History plot: {hist_path}
  ROC plot    : {roc_path}

============================================================
Phase 8 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, f'phase_8_report_{tag}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    log.info(f"  Report saved -> {report_path}")
    log.info(f"\n  Final Results: Q3={q3:.2f}%  Macro-F1={mf1:.4f}  AUC={auc_score:.4f}")


if __name__ == "__main__":
    main()
