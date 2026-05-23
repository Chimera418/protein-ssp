import os, re, pickle, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

# ── NumPy cross-version pickle compatibility shim ─────────────────
# Pickles created by different numpy/sklearn versions embed different
# internal module paths:
#   numpy <1.24  → numpy.core.numeric, numpy.core.multiarray, …
#   numpy 1.24-1.26 → numpy._core.numeric, numpy._core.multiarray, …
#   numpy 2.x    → same _core package, but submodule paths changed
#
# Strategy: register every expected submodule path in sys.modules so
# pickle.load() never raises ModuleNotFoundError regardless of which
# numpy version created the file.
def _patch_numpy_pickle_compat():
    try:
        import numpy._core as _nc
    except ImportError:
        try:
            import numpy.core as _nc
        except ImportError:
            return  # nothing we can do

    # submodule names that sklearn / numpy pickles commonly reference
    _submodules = [
        'numeric', 'multiarray', 'umath', 'fromnumeric',
        'function_base', 'shape_base', 'arrayprint',
        'defchararray', 'records', 'memmap',
        'getlimits', 'einsumfunc', 'overrides',
    ]

    for _name in _submodules:
        _obj = getattr(_nc, _name, _nc)   # fall back to the package itself
        # register under both numpy.core.X and numpy._core.X
        sys.modules.setdefault(f'numpy.core.{_name}',  _obj)
        sys.modules.setdefault(f'numpy._core.{_name}', _obj)

    # also make sure the top-level aliases exist
    sys.modules.setdefault('numpy.core',  _nc)
    sys.modules.setdefault('numpy._core', _nc)

_patch_numpy_pickle_compat()
# ──────────────────────────────────────────────────────────────────

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
try:
    from transformers import T5EncoderModel, T5Tokenizer
except ImportError:
    # transformers 5.x may not re-export T5EncoderModel from the package root
    try:
        from transformers.models.t5 import T5EncoderModel
    except ImportError:
        from transformers.models.t5.modeling_t5 import T5EncoderModel
    from transformers import AutoTokenizer as T5Tokenizer

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Protein Secondary Structure Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp { background:#0d1117; color:#e6edf3; font-family:'Inter',sans-serif; }

h1,h2,h3 { color:#e6edf3 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background:#161b22 !important; border-right:1px solid #30363d; }
[data-testid="stSidebar"] * { color:#e6edf3 !important; }

/* Cards */
.card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px; margin:8px 0; }
.card-accent { border-color:#7c3aed; background:#1a1130; }

/* Mode badge */
.badge-direct   { background:#0d4429; color:#3fb950; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-filtered { background:#0e3d4a; color:#39d4d4; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-pca      { background:#0c2d6b; color:#58a6ff; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-selected { background:#3d1f00; color:#f0883e; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }

/* Sequence display */
.seq-wrap { font-family:'JetBrains Mono',monospace; font-size:15px; line-height:2.4;
            background:#0d1117; border-radius:10px; padding:16px; border:1px solid #30363d; }
.res-H { background:#e63946; color:#fff; padding:0 3px; border-radius:3px; }
.res-E { background:#457b9d; color:#fff; padding:0 3px; border-radius:3px; }
.res-C { background:#2a9d8f; color:#fff; padding:0 3px; border-radius:3px; }
.pos   { color:#484f58; font-size:11px; margin-right:6px; }

/* Metrics */
.metric-box { background:#161b22; border:1px solid #30363d; border-radius:10px;
              padding:18px 14px; text-align:center; }
.metric-val { font-size:2rem; font-weight:800; }
.metric-lbl { color:#8b949e; font-size:0.8rem; margin-top:4px; }

/* Status icons */
.ok  { color:#3fb950; }
.err { color:#f85149; }
</style>
""", unsafe_allow_html=True)

# ── Model architecture (must match Phase 8 exactly) ───────────────
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
            batch_first=True)
        lstm_out = hidden * 2
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(lstm_out)
        self.head = nn.Sequential(
            nn.Linear(lstm_out, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x, mask=None):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.bilstm(x)
        key_mask = ~mask if mask is not None else None
        x2, _ = self.attn(x, x, x, key_padding_mask=key_mask, need_weights=False)
        x = self.norm(x + x2)
        return self.head(x)

# ── Model modes (test metrics from output/phase_8/phase_8_report_*.txt) ──
MODE_ORDER = ['direct', 'filtered', 'pca', 'selected_v1', 'selected_v2']

MODE_CONFIG = {
    'direct': {
        'label': 'Direct (1024-dim)',
        'radio': 'Direct (1024-dim)',
        'model_path': 'models/phase_8_best_model_Rostlab_prot_t5_xl_uniref50.pt',
        'embedding_pkl': 'Rostlab_prot_t5_xl_uniref50.pkl',
        'dims': 1024,
        'q3': '85.03%', 'f1': '0.8494', 'auc': '0.9685',
        'pipeline': 'ProtT5 (1024) → DL — no feature reduction.',
        'recommended': False,
    },
    'filtered': {
        'label': 'Pearson-filtered (1017-dim)',
        'radio': 'Pearson-filtered (1017-dim)',
        'model_path': 'models/phase_8_best_model_filtered_embeddings.pt',
        'embedding_pkl': 'filtered_embeddings.pkl',
        'dims': 1017,
        'q3': '85.41%', 'f1': '0.8530', 'auc': '0.9683',
        'pipeline': 'ProtT5 → Pearson filter (1017) → DL — no PCA.',
        'recommended': False,
    },
    'pca': {
        'label': 'PCA Pipeline (739-dim)',
        'radio': 'PCA Pipeline (739-dim) ★ Best',
        'model_path': 'models/phase_8_best_model_pca_embeddings.pt',
        'embedding_pkl': 'pca_embeddings.pkl',
        'dims': 739,
        'q3': '85.67%', 'f1': '0.8563', 'auc': '0.9692',
        'pipeline': 'ProtT5 → Pearson filter → PCA (739) → DL.',
        'recommended': True,
    },
    'selected_v1': {
        'label': 'Feature Selected V1 (109-dim)',
        'radio': 'Feature Selected V1 (109-dim)',
        'model_path': 'models/phase_8_best_model_final_features.pt',
        'embedding_pkl': 'final_features.pkl',
        'dims': 109,
        'q3': '84.21%', 'f1': '0.8413', 'auc': '0.9617',
        'pipeline': 'ProtT5 → Pearson → PCA → ExtraTrees (109) → DL.',
        'recommended': False,
    },
    'selected_v2': {
        'label': 'Feature Selected V2 (12-dim)',
        'radio': 'Feature Selected V2 (12-dim)',
        'model_path': 'models/phase_8_best_model_final_features_v2.pt',
        'embedding_pkl': 'final_features_v2.pkl',
        'dims': 12,
        'q3': '82.30%', 'f1': '0.8215', 'auc': '0.9600',
        'pipeline': 'ProtT5 → Pearson → PCA → top-12 refinement → DL.',
        'recommended': False,
    },
}

MODEL_REGISTRY = {
    k: (MODE_CONFIG[k]['model_path'], MODE_CONFIG[k]['dims'])
    for k in MODE_ORDER
}

INFRA = {
    'PCA model':           'models/pca_model.pkl',
    'Feature selector V1': 'models/feature_selector_mask.pkl',
    'Feature selector V2': 'models/feature_selector_mask_v2.pkl',
    'Keep indices':        'models/keep_indices.pkl',
}

def ensure_model_exists(local_path):
    """Download artifact from HF Hub if not present locally.
    Uses cache-then-copy strategy to guarantee a real file (never a symlink).
    """
    hf_path = local_path.replace("\\", "/")
    if not os.path.exists(local_path):
        try:
            import shutil
            from huggingface_hub import hf_hub_download
            # hf_hub_download always returns a real path in the HF cache
            cached = hf_hub_download(
                repo_id="Chimera418/protein-ssp-artifacts",
                filename=hf_path,
            )
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            shutil.copy2(cached, local_path)  # copy real bytes, never a symlink
        except Exception as e:
            st.error(
                f"**Download failed** for `{hf_path}`\n\n"
                f"`{type(e).__name__}: {e}`"
            )
    return local_path

@st.cache_resource(show_spinner="Loading ProtT5 encoder (first time only — ~2 min)…")
def load_prott5():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    enc = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
    enc.eval()
    return tok, enc, device

@st.cache_resource(show_spinner="Loading DL model…")
def load_dl_model(path, input_dim):
    ensure_model_exists(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = SSPModel(input_dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m, device

@st.cache_resource(show_spinner="Loading PCA transform…")
def load_pca():
    ensure_model_exists('models/pca_model.pkl')
    try:
        with open('models/pca_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"**Failed to load PCA model** — `{type(e).__name__}: {e}`")
        st.stop()

@st.cache_resource(show_spinner="Loading keep indices…")
def load_keep_indices():
    ensure_model_exists('models/keep_indices.pkl')
    path = 'models/keep_indices.pkl'
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading feature selector V1…")
def load_selector_v1():
    ensure_model_exists('models/feature_selector_mask.pkl')
    try:
        with open('models/feature_selector_mask.pkl', 'rb') as f:
            return pickle.load(f)['selected_indices']
    except Exception as e:
        st.error(f"**Failed to load selector V1** — `{type(e).__name__}: {e}`")
        st.stop()

@st.cache_resource(show_spinner="Loading feature selector V2…")
def load_selector_v2():
    ensure_model_exists('models/feature_selector_mask_v2.pkl')
    try:
        with open('models/feature_selector_mask_v2.pkl', 'rb') as f:
            return pickle.load(f)['selected_indices']
    except Exception as e:
        st.error(f"**Failed to load selector V2** — `{type(e).__name__}: {e}`")
        st.stop()

# ── Inference helpers ─────────────────────────────────────────────
IDX_SS   = {0: 'H', 1: 'E', 2: 'C'}
SS_COLOR = {'H': '#e63946', 'E': '#457b9d', 'C': '#2a9d8f'}
SS_NAME  = {'H': 'α-Helix', 'E': 'β-Sheet', 'C': 'Coil'}
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXUZOB")

def get_embeddings(seq, tok, enc, device):
    spaced = " ".join(list(re.sub(r"[UZOB]", "X", seq.upper())))
    ids = tok([spaced], return_tensors="pt").to(device)
    with torch.no_grad():
        out = enc(**ids)
    return out.last_hidden_state[0, :-1, :].cpu().numpy()   # (L, 1024)

def predict(emb_np, model_path, input_dim):
    m, device = load_dl_model(model_path, input_dim)
    x    = torch.tensor(emb_np, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones(1, emb_np.shape[0], dtype=torch.bool).to(device)
    with torch.no_grad():
        probs = F.softmax(m(x, mask)[0], dim=-1).cpu().numpy()  # (L, 3)
    return probs.argmax(axis=1), probs

def apply_feature_pipeline(emb, mode, progress_callback=None):
    """ProtT5 (L, 1024) → mode-specific features."""
    if mode == 'direct':
        return emb

    if mode in ('filtered', 'pca', 'selected_v1', 'selected_v2'):
        ensure_model_exists('models/keep_indices.pkl')
        if not os.path.exists('models/keep_indices.pkl'):
            raise FileNotFoundError("keep_indices.pkl missing. Run Phase 5.")
        ki = load_keep_indices()
        if ki is not None:
            emb = emb[:, ki]
        if progress_callback:
            progress_callback(45, f"Pearson filter applied ({emb.shape[1]}-dim).")

    if mode in ('pca', 'selected_v1', 'selected_v2'):
        ensure_model_exists('models/pca_model.pkl')
        if not os.path.exists('models/pca_model.pkl'):
            raise FileNotFoundError("pca_model.pkl missing. Run Phase 6.")
        emb = load_pca().transform(emb)
        if progress_callback:
            progress_callback(55, f"PCA transform applied ({emb.shape[1]}-dim).")

    if mode == 'selected_v1':
        ensure_model_exists('models/feature_selector_mask.pkl')
        if not os.path.exists('models/feature_selector_mask.pkl'):
            raise FileNotFoundError("feature_selector_mask.pkl missing. Run Phase 7.")
        emb = emb[:, load_selector_v1()]
        if progress_callback:
            progress_callback(70, f"ExtraTrees selection V1 ({emb.shape[1]}-dim).")

    elif mode == 'selected_v2':
        ensure_model_exists('models/feature_selector_mask_v2.pkl')
        if not os.path.exists('models/feature_selector_mask_v2.pkl'):
            raise FileNotFoundError("feature_selector_mask_v2.pkl missing. Run Phase 7.5.")
        emb = emb[:, load_selector_v2()]
        if progress_callback:
            progress_callback(70, f"ExtraTrees selection V2 ({emb.shape[1]}-dim).")

    return emb

# ── Visualisation helpers ─────────────────────────────────────────
def render_sequence(seq, preds, chunk=60):
    html = "<div class='seq-wrap'>"
    for s in range(0, len(seq), chunk):
        e  = min(s + chunk, len(seq))
        aa = seq[s:e]; pp = preds[s:e]
        html += f"<span class='pos'>{s+1:>5}</span>"
        for a, p in zip(aa, pp):
            ss = IDX_SS[p]
            html += f"<span class='res-{ss}' title='{SS_NAME[ss]}'>{a}</span>"
        html += "<br><span class='pos'>     </span>"
        for p in pp:
            ss = IDX_SS[p]
            html += f"<span style='color:{SS_COLOR[ss]};font-size:11px;'>{ss}</span>"
        html += "<br><br>"
    html += "</div>"
    return html

def confidence_figure(probs, seq):
    L = len(seq)
    fig, axes = plt.subplots(3, 1, figsize=(min(20, max(10, L/20)), 5), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    for i, (ax, ss) in enumerate(zip(axes, ['H','E','C'])):
        ax.fill_between(range(L), probs[:, i], color=SS_COLOR[ss], alpha=0.75)
        ax.set_facecolor('#161b22')
        ax.set_ylabel(f"{SS_NAME[ss]}", fontsize=8, color='#8b949e')
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=7)
    axes[-1].set_xlabel('Residue position', color='#8b949e', fontsize=9)
    fig.suptitle('Per-Residue Prediction Confidence', color='#e6edf3',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=130, facecolor='#0d1117')
    plt.close(fig)
    buf.seek(0)
    return buf

def composition_pie(preds):
    counts = {ss: int((preds == i).sum()) for i, ss in IDX_SS.items()}
    total  = len(preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    labels = [f"{SS_NAME[ss]}\n{counts[ss]} ({counts[ss]/total*100:.1f}%)"
              for ss in ['H','E','C']]
    colors = [SS_COLOR[ss] for ss in ['H','E','C']]
    vals   = [counts[ss] for ss in ['H','E','C']]
    wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                       wedgeprops={'edgecolor':'#0d1117', 'linewidth':2})
    ax.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.5, -0.22),
              fontsize=8, labelcolor='#e6edf3', frameon=False, ncol=1)
    ax.set_title('Composition', color='#e6edf3', fontsize=10, fontweight='bold', pad=8)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=130, facecolor='#161b22',
                bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 SSP Predictor")
    st.caption("CNN · BiLSTM · Multi-Head Attention")
    st.markdown("---")

    st.markdown("### 🎯 Prediction Mode")
    mode = st.radio(
        "Select mode:",
        MODE_ORDER,
        format_func=lambda x: MODE_CONFIG[x]['radio'],
        index=MODE_ORDER.index('pca'),
        label_visibility='collapsed',
    )

    cfg = MODE_CONFIG[mode]
    st.markdown("---")
    st.markdown(f"**Mode:** `{cfg['label']}`")
    if cfg['recommended']:
        st.caption("★ Recommended — highest test Q3 on held-out proteins")
    st.markdown(
        f"📊 **Test set:** Q3 **{cfg['q3']}** · Macro F1 **{cfg['f1']}** · AUC **{cfg['auc']}**"
    )
    st.caption(cfg['pipeline'])

    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    cmp = pd.DataFrame([
        {
            'Mode': MODE_CONFIG[m]['label'].replace(' ★ Best', ''),
            'Dims': MODE_CONFIG[m]['dims'],
            'Q3': MODE_CONFIG[m]['q3'],
            'F1': MODE_CONFIG[m]['f1'],
            'AUC': MODE_CONFIG[m]['auc'],
        }
        for m in MODE_ORDER
    ])
    st.dataframe(cmp, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📦 Artifact Status")
    for name, path in {**{k: v[0] for k, v in MODEL_REGISTRY.items()},
                       **INFRA}.items():
        ok = os.path.exists(path)
        st.markdown(
            f"<span class=\"{'ok' if ok else 'err'}\">{'✅' if ok else '❌'}</span> "
            f"`{os.path.basename(path)}`",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if not os.path.exists('models/keep_indices.pkl'):
        st.warning("Run `python setup_inference.py` once to generate keep_indices.pkl")

# ── Main ──────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>🧬 Protein Secondary Structure Predictor</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8b949e;margin-top:4px'>Five model variants · ProtT5 embeddings · "
    "CNN + BiLSTM + Multi-Head Attention</p>",
    unsafe_allow_html=True)
st.markdown("---")

SAMPLE = ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKV"
          "ADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLD"
          "KFLASVSTVLTSKYR")

col_seq, col_btn = st.columns([5, 1])
with col_seq:
    sequence = st.text_area(
        "Amino Acid Sequence",
        value=st.session_state.get('seq', ''),
        height=130,
        placeholder="Paste single-letter amino acid sequence here…",
    )
with col_btn:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("📋 Sample", use_container_width=True):
        st.session_state['seq'] = SAMPLE
        st.rerun()
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state['seq'] = ''
        st.rerun()

predict_btn = st.button(
    f"🔮 Predict  ({mode.upper()} mode)", type="primary", use_container_width=True)

# ── Run prediction ────────────────────────────────────────────────
if predict_btn:
    raw = sequence.strip().upper().replace(" ","").replace("\n","")
    if not raw:
        st.warning("Please enter a sequence."); st.stop()

    bad = [c for c in raw if c not in VALID_AA]
    if bad:
        st.error(f"Invalid characters: `{''.join(set(bad))}`. Use standard single-letter codes.")
        st.stop()

    cfg = MODE_CONFIG[mode]
    model_path, input_dim = MODEL_REGISTRY[mode]
    # Attempt to pull from HF Hub before checking existence
    ensure_model_exists(model_path)
    if not os.path.exists(model_path):
        st.error(
            f"Model file not found: `{model_path}`\n\n"
            f"The automatic download from Hugging Face Hub failed. "
            f"To generate it yourself, run:\n"
            f"`python phase_8_deep_learning_model.py --input embeddings/{cfg['embedding_pkl']}`"
        )
        st.stop()

    prog = st.progress(0, text="Extracting ProtT5 embeddings…")
    try:
        tok, enc, device = load_prott5()
        emb = get_embeddings(raw, tok, enc, device)
        prog.progress(30, text="ProtT5 embeddings ready (1024-dim).")

        def _prog(pct, msg):
            prog.progress(pct, text=msg)

        emb = apply_feature_pipeline(emb, mode, progress_callback=_prog)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    prog.progress(80, text="Running deep learning model…")
    preds, probs = predict(emb, model_path, input_dim)
    prog.progress(100, text="Done!")
    prog.empty()

    mean_conf = float(probs.max(axis=1).mean())
    badge_cls = {
        'direct': 'badge-direct', 'filtered': 'badge-filtered', 'pca': 'badge-pca',
        'selected_v1': 'badge-selected', 'selected_v2': 'badge-selected',
    }[mode]

    # ── Results ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"## Results · {cfg['label']} "
        f"<span class='{badge_cls}'>{mode.upper()}</span>",
        unsafe_allow_html=True,
    )

    counts = {ss: int((preds == i).sum()) for i, ss in IDX_SS.items()}
    total  = len(preds)
    m1, m2, m3 = st.columns(3)
    m1.metric("Input dim", f"{emb.shape[1]}")
    m2.metric("Mode", mode)
    m3.metric("Mean confidence", f"{mean_conf:.1%}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Length", f"{total} aa")
    c2.metric("Helix (H)", f"{counts['H']/total*100:.1f}%", f"{counts['H']} residues")
    c3.metric("Sheet (E)", f"{counts['E']/total*100:.1f}%", f"{counts['E']} residues")
    c4.metric("Coil (C)", f"{counts['C']/total*100:.1f}%", f"{counts['C']} residues")

    st.markdown("### Annotated Sequence")
    st.markdown(
        "<small style='color:#8b949e'>🔴 α-Helix &nbsp; 🔵 β-Sheet &nbsp; "
        "🟢 Coil — hover over residue for class name</small>",
        unsafe_allow_html=True)
    st.markdown(render_sequence(raw, preds), unsafe_allow_html=True)

    col_conf, col_pie = st.columns([3, 1])
    with col_conf:
        st.markdown("### Per-Residue Confidence")
        st.image(confidence_figure(probs, raw), use_column_width=True)
    with col_pie:
        st.markdown("### Composition")
        st.image(composition_pie(preds), use_column_width=True)

    with st.expander("📄 Raw Prediction Strings"):
        ss_str = "".join(IDX_SS[p] for p in preds)
        st.code(f"Sequence  : {raw}\nPrediction: {ss_str}", language="text")

    # Download
    df_out = pd.DataFrame({
        'Position':     range(1, total + 1),
        'Amino_Acid':   list(raw),
        'Predicted_SS': [IDX_SS[p] for p in preds],
        'SS_Name':      [SS_NAME[IDX_SS[p]] for p in preds],
        'Prob_Helix':   probs[:, 0].round(4),
        'Prob_Sheet':   probs[:, 1].round(4),
        'Prob_Coil':    probs[:, 2].round(4),
    })
    st.download_button(
        "⬇️ Download Predictions (CSV)",
        data=df_out.to_csv(index=False).encode(),
        file_name=f"ssp_predictions_{mode}.csv",
        mime="text/csv",
        use_container_width=True,
    )
