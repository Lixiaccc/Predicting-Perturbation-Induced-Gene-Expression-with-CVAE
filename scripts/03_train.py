#!/usr/bin/env python3
"""
03_train.py — CVAE training script.

Trains the CVAE to predict post-knockout RNA expression (delta from NTC mean)
from chromatin accessibility (ATAC LSI) or RNA PCA features.

Usage:
    python 03_train.py --input atac                         # CVAE (ATAC)
    python 03_train.py --input rna                          # CVAE (RNA)
    python 03_train.py --input atac --holdout EZH2 SMARCA4  # leave-2-out
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path("/insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final")
sys.path.insert(0, str(ROOT))
from model.cvae import CVAE_v2, project_delta_to_hvg

HIDDEN_DIM   = 32
LATENT_DIM   = 32
LR           = 1e-3
EPOCHS       = 200
PATIENCE     = 20
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 64
BETA_MAX     = 0.5
BETA_WARMUP  = 20


def stratified_3way(label_int, valid_idx, val_frac, test_frac, seed):
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for lab in np.unique(label_int[valid_idx]):
        cells = valid_idx[label_int[valid_idx] == lab].copy()
        rng.shuffle(cells)
        n_test = max(1, int(round(test_frac * len(cells))))
        n_val  = max(1, int(round(val_frac  * len(cells))))
        test.extend(cells[:n_test].tolist())
        val.extend(cells[n_test:n_test + n_val].tolist())
        train.extend(cells[n_test + n_val:].tolist())
    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", nargs="*", default=[])
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--val_frac",  type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--input", choices=["atac", "rna"], default="atac",
                   help="encoder input: atac (ATAC_LSI 50-D) or rna (RNA_PCA 50-D)")
    args = p.parse_args()

    if args.holdout in (None, [], ["none"], ["None"]):
        args.holdout = []

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.holdout:
        out_name = f"cvae_{args.input}_loko_" + "_".join(sorted(args.holdout))
    else:
        out_name = f"cvae_{args.input}"

    # ---- Load data -----------------------------------------------------------
    proc = ROOT / "processed"
    print(f"[load] {proc / 'cells.npz'}")
    d = np.load(proc / "cells.npz", allow_pickle=True)
    feat_key = "ATAC_LSI" if args.input == "atac" else "RNA_PCA"
    print(f"[load] encoder input = {args.input}  (key '{feat_key}')")
    feat     = torch.from_numpy(d[feat_key]).float()
    hvg_np   = d["RNA_HVG_log1p"]
    label    = torch.from_numpy(d["label_int"]).long()
    gtable   = torch.from_numpy(d["gene_emb_genept"]).float()
    ko_lbl   = list(d["ko_labels"])
    label_np = label.numpy()
    n_cells  = feat.shape[0]
    n_genes  = hvg_np.shape[1]

    # ---- holdout filter ------------------------------------------------------
    if args.holdout:
        for h in args.holdout:
            if h not in ko_lbl:
                raise ValueError(f"holdout label {h!r} not in {ko_lbl}")
        held_int  = [ko_lbl.index(h) for h in args.holdout]
        held_mask = np.isin(label_np, held_int)
        in_idx    = np.where(~held_mask)[0]
        out_idx   = np.where(held_mask)[0]
        print(f"  Holding out {args.holdout}: {len(out_idx)} cells excluded")
    else:
        in_idx  = np.arange(n_cells)
        out_idx = np.array([], dtype=int)

    # ---- 3-way stratified split ----------------------------------------------
    train_idx, val_idx, test_idx = stratified_3way(
        label_np, in_idx, args.val_frac, args.test_frac, args.seed)
    print(f"  Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)} | Heldout {len(out_idx)}")

    # ---- NTC mean from train cells only -------------------------------------
    NTC_int       = ko_lbl.index("NTC")
    train_ntc_idx = train_idx[label_np[train_idx] == NTC_int]
    ntc_mean      = hvg_np[train_ntc_idx].mean(axis=0).astype(np.float32)
    print(f"  ntc_mean from {len(train_ntc_idx)} train-NTC cells")

    # ---- residual target: delta from NTC mean --------------------------------
    delta_hvg_np = (hvg_np - ntc_mean).astype(np.float32)

    # ---- per-gene z-score of delta (train stats only) -----------------------
    target_mu  = delta_hvg_np[train_idx].mean(axis=0).astype(np.float32)
    target_sig = (delta_hvg_np[train_idx].std(axis=0) + 1e-3).astype(np.float32)
    target_np  = ((delta_hvg_np - target_mu) / target_sig).astype(np.float32)

    # ---- PCA on z-scored delta (train cells only) ---------------------------
    pca = PCA(n_components=50, random_state=0)
    pca.fit(target_np[train_idx])
    V   = pca.components_.T.astype(np.float32)

    target = torch.from_numpy(target_np).float()
    V_T    = torch.from_numpy(V.T).float()              # (50, 2000)

    # ---- per-gene variance weight -------------------------------------------
    gw          = (delta_hvg_np[train_idx].var(axis=0) + 1e-3).astype(np.float32)
    gene_weight = gw / gw.mean()
    gw_t        = torch.from_numpy(gene_weight).float()

    # ---- Model + optimizer --------------------------------------------------
    model    = CVAE_v2(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- Train --------------------------------------------------------------
    rng      = np.random.RandomState(args.seed)
    best_val = float("inf"); bad = 0; history = []
    models_dir = ROOT / "models"; models_dir.mkdir(exist_ok=True)
    ckpt_path  = models_dir / f"{out_name}.pt"
    t0 = time.time()

    for epoch in range(EPOCHS):
        beta = BETA_MAX * min(1.0, epoch / max(1, BETA_WARMUP))

        model.train()
        order = train_idx.copy(); rng.shuffle(order)
        tot_mse = tot_kl = tot_n = 0
        for s in range(0, len(order), BATCH_SIZE):
            b      = order[s:s + BATCH_SIZE]
            feat_b = feat[b]; tgt_b = target[b]; gemb_b = gtable[label[b]]

            pca_pred, mu, logvar = model(feat_b, gemb_b)
            pred_hvg = project_delta_to_hvg(pca_pred, V_T)

            mse  = (((pred_hvg - tgt_b) ** 2) * gw_t).mean()
            kl   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = mse + beta * kl

            opt.zero_grad(); loss.backward(); opt.step()
            bs = len(b)
            tot_mse += mse.item() * bs; tot_kl += kl.item() * bs; tot_n += bs

        train_mse = tot_mse / tot_n; train_kl = tot_kl / tot_n

        model.eval()
        with torch.no_grad():
            pca_pred, mu, logvar = model(feat[val_idx], gtable[label[val_idx]],
                                         stochastic=False)
            pred_hvg = project_delta_to_hvg(pca_pred, V_T)
            val_mse  = (((pred_hvg - target[val_idx]) ** 2) * gw_t).mean().item()
            val_kl   = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()).item()

        history.append({"epoch": epoch, "train_mse": train_mse, "train_kl": train_kl,
                        "val_mse": val_mse, "val_kl": val_kl})
        if epoch < 5 or epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"  ep{epoch:3d} train_mse={train_mse:.5f} kl={train_kl:.4f}  "
                  f"val_mse={val_mse:.5f} kl={val_kl:.4f}")

        if val_mse < best_val - 1e-6:
            best_val = val_mse; bad = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_mse":     best_val,
                "epoch":            epoch,
                "train_idx":        train_idx,
                "val_idx":          val_idx,
                "test_idx":         test_idx,
                "holdout_idx":      out_idx,
                "holdout_labels":   args.holdout,
                "ko_labels":        ko_lbl,
                "ntc_mean":         ntc_mean,
                "V":                V,
                "gene_weight":      gene_weight,
                "hidden_dim":       HIDDEN_DIM,
                "latent_dim":       LATENT_DIM,
                "predicts_delta":   True,
                "input_modality":   args.input,
                "target_zscore":    True,
                "gene_var_weight":  True,
                "target_mu":        target_mu,
                "target_sig":       target_sig,
                "gene_emb_table":   d["gene_emb_genept"],
            }, ckpt_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    with open(models_dir / f"{out_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[done] {time.time()-t0:.1f}s | best val_mse={best_val:.6f}")
    print(f"  ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
