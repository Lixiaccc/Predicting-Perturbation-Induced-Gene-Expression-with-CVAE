"""
cvae.py
CVAE for KO RNA prediction.

Design:
  - Predicts the DELTA from NTC mean (residual targeting), not absolute expression.
    The decoder outputs in delta-PCA space; the caller adds NTC mean back at inference.
  - Free-bits KL loss to prevent posterior collapse.
  - latent_dim and hidden_dim bumped from 32 to 64.

Architecture (post-changes):
    [ATAC_LSI(50) ; gene_emb(32)]  ->  Linear(82->64) + GELU
                                       ->  mu head Linear(64->64)
                                       ->  logvar head Linear(64->64)
    z = mu + sigma * eps
    z (64) -> decoder Linear(64->50) -> delta_PCA (50)
    delta_HVG (2000) = delta_PCA @ V_delta.T            # fixed projection at inference
    HVG (2000) = NTC_mean_HVG + delta_HVG               # add NTC mean to get absolute

Total learnable params: ~13.4k.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CVAE_v2(nn.Module):
    def __init__(self,
                 atac_dim: int = 50,
                 gene_emb_dim: int = 32,
                 hidden_dim: int = 64,
                 latent_dim: int = 64,
                 pca_dim: int = 50):
        super().__init__()
        in_dim = atac_dim + gene_emb_dim                  # 82
        self.encoder_trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder   = nn.Linear(latent_dim, pca_dim)   # outputs delta in PCA space

    def encode(self, x: torch.Tensor):
        h = self.encoder_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                atac: torch.Tensor,
                gene_emb: torch.Tensor,
                stochastic: bool | None = None):
        x = torch.cat([atac, gene_emb], dim=-1)
        mu, logvar = self.encode(x)
        if stochastic is None:
            stochastic = self.training
        z = self.reparameterize(mu, logvar) if stochastic else mu
        delta_pca_pred = self.decoder(z)
        return delta_pca_pred, mu, logvar


def project_delta_to_hvg(delta_pca: torch.Tensor,
                         V_delta_T: torch.Tensor) -> torch.Tensor:
    """delta_pca: (B, 50)  V_delta_T: (50, 2000)  -> (B, 2000) delta in HVG space."""
    return delta_pca @ V_delta_T


def free_bits_kl(mu: torch.Tensor, logvar: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """KL with a free-bits floor (per-dim min KL = tau nats)."""
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)   # (B, K)
    return torch.clamp(kl_per_dim - tau, min=0).sum(dim=-1).mean()
