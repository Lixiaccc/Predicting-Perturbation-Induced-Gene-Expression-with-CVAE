"""CVAE for KO RNA prediction.

Architecture:
    [ATAC_LSI(50) ; gene_emb(32)]  ->  Linear(82->32) + GELU
                                       ->  mu head / logvar head  Linear(32->32)
    z (32) -> decoder Linear(32->50) -> delta_PCA (50)
    delta_HVG (2000) = delta_PCA @ V_delta.T
    HVG (2000) = NTC_mean_HVG + delta_HVG
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CVAE_v2(nn.Module):
    def __init__(self,
                 atac_dim: int = 50,
                 gene_emb_dim: int = 32,
                 hidden_dim: int = 32,
                 latent_dim: int = 32,
                 pca_dim: int = 50):
        super().__init__()
        in_dim = atac_dim + gene_emb_dim
        self.encoder_trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder   = nn.Linear(latent_dim, pca_dim)

    def encode(self, x: torch.Tensor):
        h = self.encoder_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + (0.5 * logvar).exp() * torch.randn_like(logvar)

    def forward(self,
                atac: torch.Tensor,
                gene_emb: torch.Tensor,
                stochastic: bool | None = None):
        x = torch.cat([atac, gene_emb], dim=-1)
        mu, logvar = self.encode(x)
        if stochastic is None:
            stochastic = self.training
        z = self.reparameterize(mu, logvar) if stochastic else mu
        return self.decoder(z), mu, logvar


def project_delta_to_hvg(delta_pca: torch.Tensor,
                         V_delta_T: torch.Tensor) -> torch.Tensor:
    """delta_pca: (B, 50)  V_delta_T: (50, 2000)  -> (B, 2000)"""
    return delta_pca @ V_delta_T
