# TODO implement Transformer with VAE bottleneck and (contrastive) loss?
import torch
import torch.nn as nn


class Baller2Play(nn.Module):
    def __init__(
        self, input_dim=50, latent_dim=64, hidden_dim=128, n_layers=2, n_heads=4, seq_len=30
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, norm_first=True, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bottleneck
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder input: sample -> broadcast to seq_len
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads), num_layers=n_layers
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        T, H = x.shape

        x_proj = self.input_proj(x)  # [B, T, H]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(1, 1, H)  # [B, 1, H]
        x_with_cls = torch.cat([cls_tokens, x_proj], dim=1)  # [B, T+1, H]

        encoded = self.encoder(x_with_cls)  # [B, T+1, H]

        # Use CLS output to compute mu and logvar
        cls_out = encoded[:, 0, :]  # [B, H]
        mu = self.mu(cls_out)
        logvar = self.logvar(cls_out)
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        # Decode from latent
        z_proj = self.latent_to_hidden(z)  # [B, H]
        z_proj = z_proj.unsqueeze(1).repeat(1, self.seq_len, 1)  # [T, B, H]
        memory = encoded[:, 1:, :]  # Exclude CLS for memory: [B, T, H]
        decoded = self.decoder(tgt=z_proj, memory=memory)  # [B, T, H]
        x_hat = self.output_proj(decoded)  # [B, T, F]

        return x_hat, mu, logvar
