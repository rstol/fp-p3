# TODO implement Transformer with VAE bottleneck and (contrastive) loss?
import einops as EO
import torch
from torch import nn


# TODO @rstol model does not work yet
class Baller2Play(nn.Module):
    def __init__(
        self,
        input_dim: int = 50,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        seq_len: int = 30,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.input_proj = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bottleneck
        self.mu = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

        # Decoder input: sample -> broadcast to seq_len
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, norm_first=True, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(in_features=hidden_dim, out_features=input_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape

        tokens = self.input_proj(x)  # [B, T, H]

        # Prepend CLS token
        cls_tokens = EO.repeat(self.cls_token, "H -> 1 1 H")
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, T+1, H]

        tokens = self.encoder(tokens)  # [B, T+1, H]

        # Use CLS output to compute mu and logvar
        cls_out = tokens[:, 0, :]  # [B, H]
        mu = self.mu(cls_out)
        logvar = self.logvar(cls_out)
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        # Decode from latent
        z = self.latent_to_hidden(z)  # [B, H]
        z = EO.repeat(z, "b h -> b s h", s=seq_len)

        # TODO: Need to add some kind of position to z (t or xy or something)
        # z = z + t_proj

        decoded = self.decoder(z)  # [B, T, H]
        decoded = self.output_proj(decoded)  # [B, T, F]

        return decoded, mu, logvar
