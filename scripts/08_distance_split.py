#!/usr/bin/env python3
"""
08_distance_split.py

NTC distance split experiment:
  - Compute distance from each NTC control to the centroid of all perturbed cells
    (in PCA(50) of HVG-log1p space).
  - Split NTCs into closest 25%, middle 50%, farthest 25%.
  - Visualize the split in UMAP.
  - Train + evaluate two regimes:
        Regime A "far_train":   train on (farthest 25% NTC + all KO);
                                inference on closest 25% NTC.
        Regime B "close_train": train on (closest 25% NTC + all KO);
                                inference on farthest 25% NTC.
  - For each regime, compare 3 methods:
        CVAE (ATAC) and CVAE (RNA)
        mean_pert (training pseudobulk average)
        linear_mf (Y_hat = G W P^T + b)

Outputs:
  results/fig_distance_umap.png             — UMAP showing far/close 25% NTCs
  results/fig_distance_split_box.png        — box plot Δ-PCC_DE20 per (regime, model)
  results/distance_split_metrics.csv        — per-(regime, model, KO) numbers
"""

import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "figure.titlesize": 17,
})
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, TruncatedSVD

ROOT = Path("/insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final")
sys.path.insert(0, str(ROOT))
from model.cvae import CVAE_v2, project_delta_to_hvg

KO_LABELS = ["NTC", "ACTL6A", "DMAP1", "EP400", "EZH2",
             "SMARCA4", "SMARCB1", "SMARCE1", "SUZ12", "YY1"]
KO_GENES  = KO_LABELS[1:]
LINEAR_K  = 7   # min(16, n_cond - 1) for our 8-condition setup


# ============================================================ helpers
def pcc_safe(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10: return float("nan")
    return float(pearsonr(a, b)[0])


def topk_pcc(d_real, d_pred, k=20):
    if np.std(d_real) < 1e-10: return float("nan")
    idx = np.argsort(np.abs(d_real))[-min(k, len(d_real)):]
    return pcc_safe(d_real[idx], d_pred[idx])


# ============================================================ inline CVAE training
def train_cvae(train_idx, val_idx, input_feat, hvg, label, gtable, ntc_mean,
               epochs=200, patience=20, lr=1e-3, batch=64, seed=0):
    """Train CVAE_v2 on the given input features (ATAC LSI or RNA PCA).
    Returns (model, V_delta, mu_delta, sigma_delta, gene_weight, best_val_mse)."""
    delta_target = (hvg - ntc_mean).astype(np.float32)

    # z-score per gene using train-cells stats
    mu_delta    = delta_target[train_idx].mean(axis=0).astype(np.float32)
    sigma_delta = (delta_target[train_idx].std(axis=0) + 1e-3).astype(np.float32)
    target_z    = ((delta_target - mu_delta) / sigma_delta).astype(np.float32)

    # PCA on z-scored target
    pca = PCA(n_components=50, random_state=0)
    pca.fit(target_z[train_idx])
    V_delta = pca.components_.T.astype(np.float32)
    V_T     = torch.from_numpy(V_delta.T).float()                   # (50, 2000)

    # per-gene variance weight
    gw          = (delta_target[train_idx].var(axis=0) + 1e-3).astype(np.float32)
    gene_weight = gw / gw.mean()

    target_t = torch.from_numpy(target_z).float()
    atac_t   = torch.from_numpy(input_feat).float()
    label_t  = torch.from_numpy(label).long()
    gtable_t = torch.from_numpy(gtable).float()
    w_t      = torch.from_numpy(gene_weight).float()

    np.random.seed(seed); torch.manual_seed(seed)
    model = CVAE_v2(hidden_dim=32, latent_dim=32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf"); bad = 0; best_state = None
    rng = np.random.RandomState(seed)
    for epoch in range(epochs):
        beta = 0.5 * min(1.0, epoch / 20)
        model.train()
        order = train_idx.copy(); rng.shuffle(order)
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            pca_pred, mu, logvar = model(atac_t[b], gtable_t[label_t[b]])
            pred_target = project_delta_to_hvg(pca_pred, V_T)        # z-scored space
            mse = (w_t * (pred_target - target_t[b]) ** 2).sum(dim=-1).mean() / w_t.sum()
            kl  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = mse + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            pca_pred, _, _ = model(atac_t[val_idx], gtable_t[label_t[val_idx]],
                                   stochastic=False)
            pred_target = project_delta_to_hvg(pca_pred, V_T)
            val_mse = (w_t * (pred_target - target_t[val_idx]) ** 2).sum(dim=-1).mean().item() \
                      / w_t.sum().item()
        if val_mse < best_val - 1e-6:
            best_val = val_mse; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model, V_delta, mu_delta, sigma_delta, gene_weight, best_val


def cvae_predict_hvg(model, V_delta, mu_delta, sigma_delta,
                     ntc_input, ko_int, gtable, ntc_mean):
    """Run CVAE inference; returns (B, 2000) absolute HVG predictions."""
    V_T = torch.from_numpy(V_delta.T).float()
    model.eval()
    with torch.no_grad():
        x_t  = torch.from_numpy(ntc_input).float()
        gemb = torch.from_numpy(gtable[ko_int][None, :]).float().expand(len(x_t), -1)
        pca_pred, _, _ = model(x_t, gemb, stochastic=False)
        pred_z = project_delta_to_hvg(pca_pred, V_T).numpy()
    pred_delta = pred_z * sigma_delta + mu_delta                      # un-z-score
    return pred_delta + ntc_mean                                      # absolute HVG


# ============================================================ baselines
def fit_baselines(train_idx, label_int, full_X, hvg, ntc_train_mean,
                  full_gene_to_idx, ko_lbl, K=LINEAR_K):
    """Return (linear_mf_predict_fn, mean_pert_profile)."""
    # mean_pert: average of training KO pseudobulks (skip NTC)
    pert_pbs = []
    for k_int, kname in enumerate(ko_lbl):
        if kname == "NTC": continue
        idx_k = train_idx[label_int[train_idx] == k_int]
        if len(idx_k) > 0:
            pert_pbs.append(hvg[idx_k].mean(0))
    mean_pert_profile = np.mean(pert_pbs, axis=0)

    # linear_mf
    cond_full = list(ko_lbl)
    Y = np.stack([full_X[train_idx[label_int[train_idx] == ko_lbl.index(c)]].mean(0)
                  for c in cond_full], axis=1)                    # (n_genes_full, n_cond)
    b = Y.mean(axis=1)
    Y_c = Y - b[:, None]
    K_used = min(K, len(cond_full) - 1)
    pca_g = PCA(n_components=K_used, random_state=0)
    G = pca_g.fit_transform(Y_c).astype(np.float32)               # (n_genes_full, K)
    P = np.zeros((len(cond_full), K_used), dtype=np.float32)
    for ci, cond in enumerate(cond_full):
        if cond == "NTC": continue
        gi = full_gene_to_idx.get(cond)
        if gi is not None: P[ci] = G[gi]
    GtG = G.T @ G; PtP = P.T @ P; GtY_P = G.T @ Y_c @ P
    W = np.linalg.solve(GtG, np.linalg.solve(PtP.T, GtY_P.T).T)

    def linear_mf_predict(ko_gene):
        gi = full_gene_to_idx.get(ko_gene)
        if gi is None: return None
        return (G @ W @ G[gi] + b)[:hvg.shape[1]]                 # (2000,)

    return linear_mf_predict, mean_pert_profile


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results_fixCD",
                    help="Directory under HIGH_DIM_final/ for outputs (default: results)")
    args = ap.parse_args()
    OUT = ROOT / args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    print("[load] cells.npz")
    d = np.load(ROOT / "processed" / "cells.npz", allow_pickle=True)
    atac      = d["ATAC_LSI"]                # 50-D for CVAE-ATAC
    rna_pca   = d["RNA_PCA"]                 # 50-D for CVAE-RNA
    hvg       = d["RNA_HVG_log1p"]
    label_int = d["label_int"]
    barcodes  = d["barcode"]
    ko_lbl    = list(d["ko_labels"])
    gtable_np = d["gene_emb_genept"]
    label_to_int = {l: i for i, l in enumerate(ko_lbl)}

    # Augment for linear MF (need missing KO gene rows)
    print("[load] augmenting full RNA matrix for linear baseline ...")
    import scanpy as sc
    rna_full = sc.read_h5ad("/insomnia001/depts/houlab/users/lc3716/epifoundatoin_v2_gene/040_epi_rna_filtered.h5ad")
    sc.pp.normalize_total(rna_full, target_sum=1e4); sc.pp.log1p(rna_full)
    if not np.array_equal(rna_full.obs_names.values, barcodes):
        rna_full = rna_full[rna_full.obs_names.get_indexer(barcodes)].copy()
    with open(ROOT / "processed" / "projectors" / "hvg_list.txt") as f:
        hvgs = [l.strip() for l in f]
    missing = [g for g in KO_GENES if g not in hvgs]
    extra_idx = [rna_full.var_names.get_loc(g) for g in missing if g in rna_full.var_names]
    extra_X = rna_full.X[:, extra_idx]
    if hasattr(extra_X, "toarray"): extra_X = extra_X.toarray()
    full_X = np.concatenate([hvg, extra_X.astype(np.float32)], axis=1)
    full_gene_to_idx = {g: i for i, g in enumerate(list(hvgs) + missing)}

    # ---- compute NTC distances to perturbed centroid in PCA(50) of HVG ----
    print("[compute] NTC->perturbed distances")
    pca50 = PCA(n_components=50, random_state=0).fit_transform(hvg)
    ntc_mask = label_int == label_to_int["NTC"]
    pert_mask = ~ntc_mask
    pert_centroid = pca50[pert_mask].mean(axis=0)                    # (50,)
    ntc_idx_all = np.where(ntc_mask)[0]
    dists = np.linalg.norm(pca50[ntc_idx_all] - pert_centroid, axis=1)
    sorted_ntc = ntc_idx_all[np.argsort(dists)]
    n_q = len(ntc_idx_all) // 4
    closest_ntc = sorted_ntc[:n_q]
    farthest_ntc = sorted_ntc[-n_q:]
    print(f"  NTC: total={len(ntc_idx_all)}; closest 25%={len(closest_ntc)}; farthest 25%={len(farthest_ntc)}")

    # ---- UMAP visualization ----
    print("[viz] UMAP")
    from umap import UMAP
    um = UMAP(n_components=2, random_state=0, n_neighbors=30, min_dist=0.3).fit_transform(pca50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # left panel: NTC distance quartile + perturbed
    ax = axes[0]
    ax.scatter(um[:, 0], um[:, 1], s=2, alpha=0.15, color="#ddd")
    ax.scatter(um[pert_mask, 0], um[pert_mask, 1], s=4, alpha=0.5,
               color="#F18F01", marker="^", label=f"perturbed (n={pert_mask.sum()})")
    middle_ntc = sorted_ntc[n_q:3 * n_q]
    ax.scatter(um[middle_ntc, 0], um[middle_ntc, 1], s=4, alpha=0.5,
               color="#bbb", label=f"NTC middle 50% (n={len(middle_ntc)})")
    ax.scatter(um[closest_ntc, 0], um[closest_ntc, 1], s=6, alpha=0.85,
               color="#A23B72", label=f"NTC closest 25% (n={len(closest_ntc)})")
    ax.scatter(um[farthest_ntc, 0], um[farthest_ntc, 1], s=6, alpha=0.85,
               color="#2E86AB", label=f"NTC farthest 25% (n={len(farthest_ntc)})")
    ax.set_title("NTC controls split by distance to perturbed centroid",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best", markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

    # right panel: distance histogram with quartile lines
    ax = axes[1]
    ax.hist(dists, bins=60, color="#5DADE2", edgecolor="black", alpha=0.7)
    q25, q75 = np.quantile(dists, [0.25, 0.75])
    ax.axvline(q25, color="#A23B72", linestyle="--", label=f"q25 = {q25:.2f} (closest 25% boundary)")
    ax.axvline(q75, color="#2E86AB", linestyle="--", label=f"q75 = {q75:.2f} (farthest 25% boundary)")
    ax.set_xlabel("Euclidean distance (PCA-50) from NTC cell to perturbed centroid")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of NTC→perturbed distances")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_distance_umap.png", dpi=150)
    plt.close(fig)
    print(f"  saved fig_distance_umap.png")

    # ---- run two regimes ----
    pert_idx_all = np.where(pert_mask)[0]
    rng = np.random.RandomState(0)

    rows = []
    for regime, (train_ntc, test_ntc) in [
        ("far_train",   (farthest_ntc, closest_ntc)),
        ("close_train", (closest_ntc,  farthest_ntc)),
    ]:
        print(f"\n========== regime={regime} ==========")
        train_full = np.concatenate([train_ntc, pert_idx_all])
        # Use 90/10 split of train_full for train/val
        ord_ = train_full.copy(); rng.shuffle(ord_)
        n_val = max(1, int(0.1 * len(ord_)))
        val_idx   = np.array(sorted(ord_[:n_val]))
        train_idx = np.array(sorted(ord_[n_val:]))
        # NTC mean from train_ntc
        ntc_train_mean = hvg[train_ntc].mean(0).astype(np.float32)
        print(f"  train cells = {len(train_idx)}, val cells = {len(val_idx)}")
        print(f"  NTC for inference = {len(test_ntc)}")
        print(f"  ntc_train_mean computed from {len(train_ntc)} NTC cells")

        # ---- train CVAE (ATAC) ----
        t0 = time.time()
        print("  [train] CVAE (ATAC) ...")
        atac_model, atac_V, atac_mu, atac_sig, _, atac_val = train_cvae(
            train_idx, val_idx, atac, hvg, label_int, gtable_np, ntc_train_mean)
        print(f"    done in {time.time()-t0:.1f}s, best val_mse={atac_val:.5f}")

        # ---- train CVAE (RNA) ----
        t0 = time.time()
        print("  [train] CVAE (RNA) ...")
        rna_model, rna_V, rna_mu, rna_sig, _, rna_val = train_cvae(
            train_idx, val_idx, rna_pca, hvg, label_int, gtable_np, ntc_train_mean)
        print(f"    done in {time.time()-t0:.1f}s, best val_mse={rna_val:.5f}")

        # ---- fit baselines ----
        print("  [fit] mean_pert + linear_mf ...")
        linear_mf_pred, mean_pert_profile = fit_baselines(
            train_idx, label_int, full_X, hvg, ntc_train_mean,
            full_gene_to_idx, ko_lbl)

        # ---- evaluate on each KO using test_ntc as input ----
        test_atac    = atac[test_ntc]                # 50-D for CVAE-ATAC
        test_rna_pca = rna_pca[test_ntc]             # 50-D for CVAE-RNA
        test_ntc_rna = hvg[test_ntc]
        for ko in KO_GENES:
            ko_int = label_to_int[ko]
            real_ko_cells = hvg[label_int == ko_int]
            if len(real_ko_cells) == 0: continue
            real_ko_mean = real_ko_cells.mean(0)
            real_ntc_mean_test = test_ntc_rna.mean(0)
            delta_real = real_ko_mean - real_ntc_mean_test

            def emit(name, pred_mean, pred_cells):
                delta_pred = pred_mean - real_ntc_mean_test
                rows.append({
                    "regime": regime, "model": name, "ko": ko,
                    "n_train": len(train_idx), "n_test_ntc": len(test_ntc),
                    "n_real_ko": len(real_ko_cells),
                    "pcc_DE20":  topk_pcc(delta_real, delta_pred, 20),
                    "pcc_DE100": topk_pcc(delta_real, delta_pred, 100),
                    "pcc_all":   pcc_safe(delta_real, delta_pred),
                })

            # CVAE (ATAC)
            pred_atac = cvae_predict_hvg(atac_model, atac_V, atac_mu, atac_sig,
                                         test_atac, ko_int, gtable_np, ntc_train_mean)
            emit("CVAE (ATAC)", pred_atac.mean(0), pred_atac)
            # CVAE (RNA)
            pred_rna = cvae_predict_hvg(rna_model, rna_V, rna_mu, rna_sig,
                                        test_rna_pca, ko_int, gtable_np, ntc_train_mean)
            emit("CVAE (RNA)", pred_rna.mean(0), pred_rna)
            # linear_mf
            p_lin = linear_mf_pred(ko)
            if p_lin is not None:
                d_lin = p_lin - ntc_train_mean
                pred_lin_cells = test_ntc_rna + d_lin[None, :]
                emit("linear_mf", pred_lin_cells.mean(0), pred_lin_cells)
            # mean_pert
            d_mp = mean_pert_profile - ntc_train_mean
            pred_mp_cells = test_ntc_rna + d_mp[None, :]
            emit("mean_pert", pred_mp_cells.mean(0), pred_mp_cells)

    # ---- save + plot ----
    df = pd.DataFrame(rows)
    out_csv = OUT / "distance_split_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")

    # box plot of pcc_DE20 by (regime, model)
    PLOT_ORDER = ["CVAE (ATAC)", "CVAE (RNA)", "mean_pert", "linear_mf"]
    DISPLAY    = {"CVAE (ATAC)": "CVAE (ATAC)", "CVAE (RNA)": "CVAE (RNA)",
                  "mean_pert": "Mean Pert", "linear_mf": "Linear MF"}
    REGIMES      = ["far_train", "close_train"]
    REGIME_LABEL = {"far_train":   "Train on far 25%, test on close 25%",
                    "close_train": "Train on close 25%, test on far 25%"}
    COLORS = {"CVAE (ATAC)": "#2E86AB", "CVAE (RNA)": "#9D4EDD",
              "mean_pert": "#3D5A40", "linear_mf": "#C73E1D"}
    HATCH = {"far_train": "", "close_train": "//"}

    # Geometry: wider figure + tighter gap so the per-box annotation column fits
    # cleanly without overlapping the neighbour. With 4 models, model centers are
    # 1.0 axis-units apart; we use gap=0.34 between regime pairs and width=0.28
    # so each annotation column gets ~0.34 axis units (≈ 1.5 in at 18-in figure).
    fig, ax = plt.subplots(figsize=(18, 8.5))
    width = 0.28
    gap   = 0.34

    for j, model_name in enumerate(PLOT_ORDER):
        for k, regime in enumerate(REGIMES):
            sub = df[(df["model"] == model_name) & (df["regime"] == regime)]
            vals = sub["pcc_DE20"].dropna().values
            if not len(vals): continue
            x = j + (k - 0.5) * gap                                 # centers spaced ±0.17
            bp = ax.boxplot([vals], positions=[x], widths=width, patch_artist=True,
                            showmeans=True, showfliers=False,
                            meanprops=dict(marker="^", markerfacecolor="white",
                                           markeredgecolor="black", markersize=8))
            bp["boxes"][0].set_facecolor(COLORS[model_name])
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][0].set_hatch(HATCH[regime])
            # 3-line vertical annotation: each line is short (≈6 chars) so columns
            # don't collide even when boxes are close. Lines stack just above 1.0.
            ann = (f"μ={vals.mean():.2f}\n"
                   f"σ={vals.std():.2f}\n"
                   f"med={np.median(vals):.2f}")
            ax.text(x, 1.02, ann,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=11, color="#222",
                    linespacing=1.3)

    ax.set_xticks(range(len(PLOT_ORDER)))
    ax.set_xticklabels([DISPLAY[m] for m in PLOT_ORDER])
    ax.set_ylabel("Δ-PCC, top-20 DE genes")
    ax.set_title("Effect of NTC training subset on KO-effect prediction",
                 fontsize=16, fontweight="bold", pad=80)            # extra pad for the 3-line annotations
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlim(-0.6, len(PLOT_ORDER) - 0.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=COLORS[m], label=DISPLAY[m]) for m in PLOT_ORDER]
    legend_elems += [Patch(facecolor="white", edgecolor="black", hatch=h, label=REGIME_LABEL[r])
                     for r, h in HATCH.items()]
    ax.legend(handles=legend_elems, fontsize=12, loc="lower right", ncol=1, framealpha=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(OUT / "fig_distance_split_box.png", dpi=150)
    plt.close(fig)
    print(f"[saved] fig_distance_split_box.png")

    # short summary
    print("\n=== SUMMARY: mean Δ-PCC_DE20 by (regime, model) ===")
    print(df.pivot_table(index="regime", columns="model",
                         values="pcc_DE20", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
