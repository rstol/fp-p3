# TODO implement Transformer with VAE bottleneck and (contrastive) loss?
import torch
import torch.nn as nn


class Baller2Play(nn.Module):
    def __init__(
        self, input_dim, seq_len, latent_dim=64, n_players=10, d_model=128, nhead=4, num_layers=2
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_players = n_players

        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers
        )

        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers
        )
        self.output_proj = nn.Linear(d_model, input_dim)

    def encode(self, x):
        x = self.input_proj(x)  # [B, T, d_model]
        x = x.permute(1, 0, 2)  # [T, B, d_model]
        h = self.encoder(x)[0]  # Use only first token (or pool)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_seq_len):
        z = self.decoder_input(z).unsqueeze(0).repeat(target_seq_len, 1, 1)  # [T, B, d_model]
        memory = z  # dummy memory (z repeated)
        tgt = torch.zeros_like(memory)  # start token placeholders
        out = self.decoder(tgt, memory)
        out = out.permute(1, 0, 2)  # [B, T, d_model]
        return self.output_proj(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, x.size(1))
        return x_hat, mu, logvar
