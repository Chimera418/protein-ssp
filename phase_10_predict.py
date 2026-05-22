import os
import re
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
try:
    from transformers import T5EncoderModel, T5Tokenizer
except ImportError:
    try:
        from transformers.models.t5 import T5EncoderModel
    except ImportError:
        from transformers.models.t5.modeling_t5 import T5EncoderModel
    from transformers import AutoTokenizer as T5Tokenizer

# ──────────────────────────────────────────────
# Architecture Definition (Matches Phase 8)
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
            return logits, attn_weights
        return logits

# ──────────────────────────────────────────────
# ProtT5 Feature Extraction
# ──────────────────────────────────────────────
def get_prott5_embeddings(sequence, tokenizer, model, device):
    """Generate 1024-dim ProtT5 embeddings for a given sequence."""
    sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    inputs = tokenizer([sequence], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract last hidden state, ignore batch dimension and EOS token
        embeddings = outputs.last_hidden_state[0, :-1, :]
    return embeddings.cpu().numpy()

# ──────────────────────────────────────────────
# Main Prediction Script
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 10: Inference Pipeline")
    parser.add_argument('--sequence', type=str, required=True,
                        help="Amino acid sequence to predict")
    parser.add_argument('--mode', type=str, choices=['pipeline', 'direct'], default='pipeline',
                        help="pipeline = ProtT5 -> PCA(739) -> DL | direct = ProtT5(1024) -> DL")
    parser.add_argument('--pca-model', type=str, default='models/pca_model.pkl')
    parser.add_argument('--dl-pipeline-model', type=str, default='models/phase_8_best_model_pca_embeddings.pt')
    parser.add_argument('--dl-direct-model', type=str, default='models/phase_8_best_model_Rostlab_prot_t5_xl_uniref50.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1/3] Loading ProtT5 Encoder on {device}...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
    encoder.eval()

    print(f"[2/3] Extracting ProtT5 Embeddings for sequence (Length: {len(args.sequence)})...")
    emb_1024 = get_prott5_embeddings(args.sequence, tokenizer, encoder, device)
    
    if args.mode == 'pipeline':
        print(f"[3/3] Running Pipeline Mode (ProtT5 -> PCA -> DL)...")
        if not os.path.exists(args.pca_model):
            raise FileNotFoundError(f"PCA model not found at {args.pca_model}")
        with open(args.pca_model, 'rb') as f:
            pca = pickle.load(f)
        
        # PCA expects 2D array (L, 1024), transforms to (L, 739)
        features = pca.transform(emb_1024)
        dl_model_path = args.dl_pipeline_model
        input_dim = 739
    else:
        print(f"[3/3] Running Direct Mode (ProtT5 -> DL)...")
        features = emb_1024
        dl_model_path = args.dl_direct_model
        input_dim = 1024

    if not os.path.exists(dl_model_path):
        raise FileNotFoundError(f"Deep learning model not found at {dl_model_path}")

    # Load Model
    dl_model = SSPModel(input_dim=input_dim).to(device)
    dl_model.load_state_dict(torch.load(dl_model_path, map_location=device))
    dl_model.eval()

    # Predict
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device) # (1, L, D)
    mask = torch.ones(1, len(args.sequence), dtype=torch.bool).to(device)
    
    with torch.no_grad():
        logits = dl_model(features_tensor, mask)
        preds = logits.argmax(dim=-1)[0].cpu().numpy()

    idx_to_ss = {0: 'H', 1: 'E', 2: 'C'}
    ss_preds = "".join([idx_to_ss[p] for p in preds])

    print("-" * 60)
    print("PREDICTION RESULTS:")
    print(f"Sequence: {args.sequence}")
    print(f"Pred SS : {ss_preds}")
    print("-" * 60)

if __name__ == "__main__":
    main()
