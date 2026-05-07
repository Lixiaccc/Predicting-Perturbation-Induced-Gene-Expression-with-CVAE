#!/usr/bin/env python3
"""
02_preprocess.py

One-shot preprocessing for the perturbation-prediction CVAE.
Reads the source 040_epi_*.h5ad files, computes:
  * RNA: normalize_total -> log1p -> top 2000 HVG -> PCA(50) [save loadings V and gene means mu_g]
  * ATAC: TF-IDF -> TruncatedSVD(51) -> drop component 1 [save the SVD object]
  * GenePT: load 9 KO embeddings (1536-D), prepend NTC=zero, PCA-project to 32-D
Saves a single tidy npz at HIGH_DIM_final/processed/cells.npz with everything aligned by barcode.

Compute: not heavy on RAM (peak <32GB) but ATAC SVD on 524k peaks is non-trivial. Run on a CPU node.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

# ---------------------------------------------------------------------- paths
ROOT = Path("/insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final")
PROC = ROOT / "processed"
PROJ = PROC / "projectors"
PROJ.mkdir(parents=True, exist_ok=True)

RNA_H5 = Path("/insomnia001/depts/houlab/users/lc3716/epifoundatoin_v2_gene/040_epi_rna_filtered.h5ad")
ATAC_H5 = Path("/insomnia001/depts/houlab/users/lc3716/epifoundatoin_v2_gene/040_epi_atac_filtered.h5ad")
GENEPT_CSV = PROJ / "genept_embeddings.csv"

KO_LABELS = ["NTC", "ACTL6A", "DMAP1", "EP400", "EZH2",
             "SMARCA4", "SMARCB1", "SMARCE1", "SUZ12", "YY1"]


# ============================================================ RNA preprocessing
def preprocess_rna():
    print(f"\n[RNA] Loading {RNA_H5} ...")
    adata = sc.read_h5ad(RNA_H5)
    print(f"  shape: {adata.shape}")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection on log1p-normalized counts
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    hvg_mask = adata.var["highly_variable"].values
    hvg_names = adata.var_names[hvg_mask].tolist()
    print(f"  HVGs selected: {sum(hvg_mask)}")

    # Subset to HVGs and densify (2665 x 2000 is small)
    Xhvg = adata[:, hvg_mask].X
    if sparse.issparse(Xhvg):
        Xhvg = Xhvg.toarray()
    Xhvg = Xhvg.astype(np.float32)
    print(f"  HVG matrix: {Xhvg.shape}")

    # PCA on the HVG matrix
    pca = PCA(n_components=50, random_state=0)
    rna_pca = pca.fit_transform(Xhvg).astype(np.float32)
    V = pca.components_.T.astype(np.float32)              # (2000, 50)
    mu_g = pca.mean_.astype(np.float32)                   # (2000,)
    print(f"  PCA: rna_pca={rna_pca.shape}, V={V.shape}, mu_g={mu_g.shape}")

    np.save(PROJ / "pca_loadings_V.npy", V)
    np.save(PROJ / "pca_gene_means.npy", mu_g)
    with open(PROJ / "hvg_list.txt", "w") as f:
        for g in hvg_names:
            f.write(g + "\n")

    barcodes = adata.obs_names.values.astype(str)
    obs_df = adata.obs[["guide_target"]].copy()
    obs_df.index = barcodes
    return barcodes, obs_df, Xhvg, rna_pca, hvg_names


# ============================================================ ATAC preprocessing
def preprocess_atac(rna_barcodes):
    print(f"\n[ATAC] Loading {ATAC_H5} ...")
    adata = sc.read_h5ad(ATAC_H5)
    print(f"  shape: {adata.shape}")

    # Reorder ATAC to match RNA barcode order
    if not np.array_equal(adata.obs_names.values, rna_barcodes):
        idx = adata.obs_names.get_indexer(rna_barcodes)
        if (idx == -1).any():
            missing = (idx == -1).sum()
            raise RuntimeError(f"{missing} RNA barcodes not found in ATAC")
        adata = adata[idx].copy()
    print(f"  ATAC reordered to RNA barcode order: {adata.shape}")

    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    # TF-IDF (sublinear TF, L2 norm) — Signac/snapATAC2 style
    print("  TF-IDF ...")
    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(X).astype(np.float32)

    # TruncatedSVD; keep 51, drop component 1 (depth-correlated) -> 50-D LSI
    print("  TruncatedSVD(51) ...")
    svd = TruncatedSVD(n_components=51, random_state=0)
    lsi_full = svd.fit_transform(X_tfidf).astype(np.float32)   # (n_cells, 51)
    atac_lsi = lsi_full[:, 1:]                                  # drop comp 1 -> (n_cells, 50)
    print(f"  ATAC_LSI: {atac_lsi.shape}")

    # Persist projectors
    with open(PROJ / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(PROJ / "svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    return atac_lsi


# ============================================================ GenePT projection
def preprocess_genept():
    print(f"\n[GenePT] Loading {GENEPT_CSV} ...")
    df = pd.read_csv(GENEPT_CSV, index_col=0)
    print(f"  loaded: {df.shape}  (genes x dims)")

    # Build full 10x1536 matrix in KO_LABELS order; NTC = zero vector
    full = np.zeros((len(KO_LABELS), df.shape[1]), dtype=np.float32)
    for i, gene in enumerate(KO_LABELS):
        if gene == "NTC":
            continue                                             # leave row zero
        if gene not in df.index:
            raise KeyError(f"GenePT row missing for {gene}")
        full[i] = df.loc[gene].values.astype(np.float32)
    print(f"  full GenePT (NTC=zero): {full.shape}")

    # PCA-project to 32-D. With only 10 vectors PCA rank is at most 9,
    # but we keep 32 components so the encoder's 82->32 layer can choose.
    pca = PCA(n_components=min(32, full.shape[0]), random_state=0)
    proj = pca.fit_transform(full).astype(np.float32)            # (10, K) where K<=10
    if proj.shape[1] < 32:
        # zero-pad to 32 dims
        pad = np.zeros((proj.shape[0], 32 - proj.shape[1]), dtype=np.float32)
        proj = np.concatenate([proj, pad], axis=1)
    print(f"  projected GenePT: {proj.shape}")

    # The PCA on 9 nonzero rows + 1 zero row -> first row (NTC) becomes
    # the negated mean direction. Force NTC back to exact zero so the model
    # has a clean "no perturbation" anchor (per plan).
    ntc_idx = KO_LABELS.index("NTC")
    proj[ntc_idx] = 0.0
    print(f"  NTC row reset to zero. Final shape: {proj.shape}")

    np.save(PROJ / "gene_emb_genept.npy", proj)
    with open(PROJ / "gene_emb_labels.txt", "w") as f:
        for label in KO_LABELS:
            f.write(label + "\n")
    return proj


# ============================================================ main
def main():
    barcodes, obs_df, hvg_log1p, rna_pca, hvg_names = preprocess_rna()
    atac_lsi = preprocess_atac(barcodes)
    gene_emb_genept = preprocess_genept()

    # Map guide_target string to int label index in KO_LABELS
    label_to_int = {label: i for i, label in enumerate(KO_LABELS)}
    guide_target = obs_df["guide_target"].astype(str).values
    label_int = np.array([label_to_int[g] for g in guide_target], dtype=np.int64)
    print(f"\n[labels] guide_target counts:")
    print(pd.Series(guide_target).value_counts().to_string())

    out = PROC / "cells.npz"
    np.savez(
        out,
        barcode=barcodes,
        guide_target=guide_target,
        label_int=label_int,
        ATAC_LSI=atac_lsi.astype(np.float32),
        RNA_PCA=rna_pca.astype(np.float32),
        RNA_HVG_log1p=hvg_log1p.astype(np.float32),
        gene_emb_genept=gene_emb_genept.astype(np.float32),
        ko_labels=np.array(KO_LABELS),
    )
    print(f"\n[done] saved {out}")
    print(f"  ATAC_LSI:        {atac_lsi.shape}")
    print(f"  RNA_PCA:         {rna_pca.shape}")
    print(f"  RNA_HVG_log1p:   {hvg_log1p.shape}")
    print(f"  gene_emb_genept: {gene_emb_genept.shape}")
    print(f"  guide_target:    {guide_target.shape}, unique={np.unique(guide_target).tolist()}")


if __name__ == "__main__":
    main()
