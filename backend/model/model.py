import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 1) Encoder: projects raw features and runs the Transformer stack
class PlayEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim, nhead=4, nlayers=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim*4,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=nlayers,
            norm=nn.LayerNorm(emb_dim)
        )

    def forward(self, x, mask):
        """
        x:    [B, T, D] raw sequence
        mask: [B, T]    True=real, False=pad
        → returns H [B, T, E]
        """
        # project to model dim
        x = self.input_proj(x)           # [B, T, E]
        x = x.transpose(0, 1)            # [T, B, E]
        pad_mask = ~mask                  # [B, T]
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        return out.transpose(0, 1)       # [B, T, E]


# 2) Decoder: from hidden states to next-frame features
class NextMomentDecoder(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super().__init__()
        self.output_proj = nn.Linear(emb_dim, out_dim)

    def forward(self, h):
        """
        h:    [B, T, E] hidden states
        → pred [B, T, D] next-frame predictions
        """
        return self.output_proj(h)


# 3) Assemble end-to-end
class NextMomentModel(nn.Module):
    def __init__(self, in_dim, emb_dim, nhead=4, nlayers=3):
        super().__init__()
        self.encoder = PlayEncoder(in_dim, emb_dim, nhead, nlayers)
        self.decoder = NextMomentDecoder(emb_dim, in_dim)

    def forward(self, x, mask):
        # Encode then decode
        h    = self.encoder(x, mask)     # [B, T, E]
        pred = self.decoder(h)           # [B, T, D]
        return pred


# 4) Training loop
def train(model, loader, epochs=5, lr=1e-4, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            x    = batch['sequence'].to(device)  # [B, T, D]
            mask = batch['mask'].to(device)      # [B, T]

            # prepare inputs (all but last) and targets (all but first)
            x_in  = x[:, :-1, :]    # [B, T-1, D]
            x_tgt = x[:,  1:, :]    # [B, T-1, D]
            m     = mask[:, :-1]    # [B, T-1]

            # forward
            pred = model(x_in, m)   # [B, T-1, D]

            # compute MSE only on real frames
            loss_mat = F.mse_loss(pred, x_tgt, reduction='none')  # [B, T-1, D]
            loss_seq = loss_mat.sum(-1) * m.float()                # [B, T-1]
            loss = loss_seq.sum() / m.sum()                        # scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — avg MSE: {avg_loss:.4f}")

# 5) Usage
# Assume `loader` is your DataLoader with collate_fn from before,
# and D = feature dim of your sequence (e.g. seq.shape[-1]).
model = NextMomentModel(in_dim=D, emb_dim=256, nhead=4, nlayers=3)
train(model, loader, epochs=10, lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu")
