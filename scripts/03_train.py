#!/usr/bin/env python3
"""
03_train.py — CVAE training script (final model: only_fix1_CD).

The final recipe is:
    --fix1                 : residual targeting (predict δ from NTC mean)
    --target_zscore   (C)  : per-gene z-score the δ target
    --gene_var_weight (D)  : variance-weighted MSE on δ

Variant convenience: --variant only_fix1_CD sets all the right flags.
Other variants remain available for ablation reproducibility.

Splits: 3-way 80/10/10 (train/val/test) stratified by guide_target.
Always saves train_idx, val_idx, test_idx, holdout_idx in the checkpoint.

Usage examples:
    python 03_train.py --variant only_fix1_CD --holdout none
    python 03_train.py --variant only_fix1_CD --holdout SMARCA4 EZH2
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
from model.cvae import CVAE_v2, project_delta_to_hvg, free_bits_kl


# ---- variant table -----------------------------------------------------------
# index order:  fix1, fix2, fix3, fix4, fix5
VARIANT_FIXES = {
    "baseline":        [0, 0, 0, 0, 0],          # = v1 with 3-way split
    "only_fix1":       [1, 0, 0, 0, 0],
    "only_fix2":       [0, 1, 0, 0, 0],
    "only_fix3":       [0, 0, 1, 0, 0],
    "only_fix4":       [0, 0, 0, 1, 0],
    "only_fix5":       [0, 0, 0, 0, 1],
    "all_fixes":       [1, 1, 1, 1, 1],
    "only_fix1_mmd":   [1, 0, 0, 0, 0],          # only_fix1 + MMD distribution loss
    "only_fix1_mean":  [1, 0, 0, 0, 0],          # only_fix1 + per-KO mean-alignment loss
    "only_fix1_C":     [1, 0, 0, 0, 0],          # only_fix1 + target z-scoring (Fix C)
    "only_fix1_D":     [1, 0, 0, 0, 0],          # only_fix1 + variance-weighted MSE on delta (Fix D)
    "only_fix1_CD":    [1, 0, 0, 0, 0],          # only_fix1 + Fix C + Fix D
}

# Auto-set aux-loss weights / new target/weighting flags from variant name
VARIANT_AUX_LOSS = {
    "only_fix1_mmd":   {"mmd_weight": 1.0,  "mean_align_weight": 0.0},
    "only_fix1_mean":  {"mmd_weight": 0.0,  "mean_align_weight": 1.0},
    "only_fix1_C":     {"target_zscore": 1, "gene_var_weight": 0},
    "only_fix1_D":     {"target_zscore": 0, "gene_var_weight": 1},
    "only_fix1_CD":    {"target_zscore": 1, "gene_var_weight": 1},
}


def mmd_loss(X, Y, sigmas=(0.5, 1.0, 2.0, 5.0)):
    """Multi-scale RBF MMD² between two batches. X: (n, d), Y: (m, d). Scalar."""
    XX = torch.cdist(X, X) ** 2
    YY = torch.cdist(Y, Y) ** 2
    XY = torch.cdist(X, Y) ** 2
    out = X.new_zeros(())
    for sig in sigmas:
        s2 = 2.0 * (sig ** 2)
        Kxx = torch.exp(-XX / s2).mean()
        Kyy = torch.exp(-YY / s2).mean()
        Kxy = torch.exp(-XY / s2).mean()
        out = out + (Kxx + Kyy - 2 * Kxy)
    return out / len(sigmas)


def mean_align_loss(pred_delta, target_delta, labels):
    """For each KO label appearing >= 2 times in batch, compute MSE between
    mean predicted delta and mean target delta. Pushes per-KO population means
    to align (the metric we're evaluated on)."""
    out = pred_delta.new_zeros(())
    n = 0
    for k in torch.unique(labels):
        mask = labels == k
        if mask.sum() < 2:
            continue
        out = out + ((pred_delta[mask].mean(0) - target_delta[mask].mean(0)) ** 2).mean()
        n += 1
    return out / max(1, n)


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
    p.add_argument("--variant", type=str, default=None,
                   help="Convenience: sets fix1-5 from VARIANT_FIXES. "
                        f"Options: {list(VARIANT_FIXES)}")
    p.add_argument("--fix1", type=int, default=None, help="residual targeting")
    p.add_argument("--fix2", type=int, default=None, help="free-bits KL")
    p.add_argument("--fix3", type=int, default=None, help="gene-weighted MSE")
    p.add_argument("--fix4", type=int, default=None, help="latent=64 (else 32)")
    p.add_argument("--fix5", type=int, default=None, help="cosine LR + long train")
    p.add_argument("--holdout", nargs="*", default=[])
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--val_frac",  type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--free_bits_tau", type=float, default=0.1)
    p.add_argument("--kl_weight",     type=float, default=1.0)
    p.add_argument("--beta_warmup_v1",type=int,   default=20)
    p.add_argument("--beta_max_v1",   type=float, default=0.5)
    p.add_argument("--gene_emb", choices=["genept", "geneformer"], default="genept",
                   help="which gene embedding source to use as conditioning")
    p.add_argument("--mmd_weight",        type=float, default=0.0,
                   help="weight for MMD distribution-matching loss (0 = off)")
    p.add_argument("--mean_align_weight", type=float, default=0.0,
                   help="weight for per-KO mean-alignment delta loss (0 = off)")
    p.add_argument("--target_zscore",     type=int,   default=0,
                   help="Fix C: z-score the delta target per gene (0/1)")
    p.add_argument("--gene_var_weight",   type=int,   default=0,
                   help="Fix D: use var(delta_target across train cells) as gene weight (0/1)")
    p.add_argument("--input", choices=["atac", "rna"], default="atac",
                   help="encoder input modality: atac (ATAC_LSI 50-D) or rna (RNA_PCA 50-D)")
    args = p.parse_args()

    # Apply auto-set aux-loss weights from variant
    if args.variant in VARIANT_AUX_LOSS:
        for k, v in VARIANT_AUX_LOSS[args.variant].items():
            cur = getattr(args, k, None)
            # treat 0 / 0.0 / False as "default → override"
            if cur in (0, 0.0, False, None):
                setattr(args, k, v)

    # resolve variant -> fix flags
    if args.variant is not None:
        fixes = VARIANT_FIXES[args.variant]
        for k, v in zip(["fix1", "fix2", "fix3", "fix4", "fix5"], fixes):
            if getattr(args, k) is None:
                setattr(args, k, v)
    for k in ["fix1", "fix2", "fix3", "fix4", "fix5"]:
        if getattr(args, k) is None:
            setattr(args, k, 1)

    # holdout normalization
    if args.holdout in (None, [], ["none"], ["None"]):
        args.holdout = []

    # ---- Fix 4: model size ---------------------------------------------------
    hidden_dim = latent_dim = 64 if args.fix4 else 32

    # ---- Fix 5: schedule ----------------------------------------------------
    if args.fix5:
        lr, epochs, patience = 3e-4, 300, 50
    else:
        lr, epochs, patience = 1e-3, 200, 20
    weight_decay = 1e-4
    batch_size   = 64

    # checkpoint name: suffix with gene_emb (geneformer runs don't overwrite genept)
    # and input modality (rna runs don't overwrite atac)
    variant_tag = args.variant or f"f{''.join(str(getattr(args, f'fix{i}')) for i in range(1,6))}"
    gene_suffix  = "" if args.gene_emb == "genept" else f"_{args.gene_emb}"
    input_suffix = "" if args.input == "atac" else f"_{args.input}"
    suffix = gene_suffix + input_suffix
    out_name = (f"cvae_{variant_tag}{suffix}" if not args.holdout else
                f"cvae_{variant_tag}_loko_" + "_".join(sorted(args.holdout)) + suffix)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load data -----------------------------------------------------------
    proc = ROOT / "processed"
    proj = proc / "projectors"
    print(f"[load] {proc / 'cells.npz'}")
    d = np.load(proc / "cells.npz", allow_pickle=True)
    feat_key = "ATAC_LSI" if args.input == "atac" else "RNA_PCA"
    print(f"[load] encoder input modality = {args.input}  (using key '{feat_key}')")
    atac    = torch.from_numpy(d[feat_key]).float()        # 50-D feature, used as encoder input
    hvg_np  = d["RNA_HVG_log1p"]
    label   = torch.from_numpy(d["label_int"]).long()
    if args.gene_emb == "genept":
        gtable_np = d["gene_emb_genept"]
    elif args.gene_emb == "geneformer":
        gpath = proj / "gene_emb_geneformer.npy"
        if not gpath.exists():
            raise FileNotFoundError(
                f"{gpath} not found. Run scripts/01b_generate_geneformer.py first.")
        gtable_np = np.load(gpath)
    print(f"  gene_emb source: {args.gene_emb}, shape={gtable_np.shape}")
    gtable  = torch.from_numpy(gtable_np).float()
    ko_lbl  = list(d["ko_labels"])
    label_np = label.numpy()
    n_cells = atac.shape[0]
    n_genes = hvg_np.shape[1]

    print(f"\n[variant] {variant_tag}  fixes=[{args.fix1},{args.fix2},{args.fix3},{args.fix4},{args.fix5}]")
    print(f"  hidden={hidden_dim} latent={latent_dim} lr={lr} epochs={epochs} patience={patience}")

    # ---- holdout filter -----------------------------------------------------
    if args.holdout:
        for h in args.holdout:
            if h not in ko_lbl:
                raise ValueError(f"holdout label {h!r} not in {ko_lbl}")
        held_int = [ko_lbl.index(h) for h in args.holdout]
        held_mask = np.isin(label_np, held_int)
        in_idx  = np.where(~held_mask)[0]
        out_idx = np.where( held_mask)[0]
        print(f"  Holding out {args.holdout}: {len(out_idx)} cells excluded")
    else:
        in_idx  = np.arange(n_cells)
        out_idx = np.array([], dtype=int)

    # ---- 3-way stratified split ---------------------------------------------
    train_idx, val_idx, test_idx = stratified_3way(
        label_np, in_idx, args.val_frac, args.test_frac, args.seed)
    print(f"  Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)} | Heldout {len(out_idx)}")

    # ---- NTC mean from train cells only -------------------------------------
    NTC_int = ko_lbl.index("NTC")
    train_ntc_idx = train_idx[label_np[train_idx] == NTC_int]
    ntc_mean = hvg_np[train_ntc_idx].mean(axis=0).astype(np.float32)
    print(f"  ntc_mean computed from {len(train_ntc_idx)} train-NTC cells")

    # ---- residual targeting: predict delta from NTC mean (else absolute HVG) ----
    if args.fix1:
        delta_hvg_np = (hvg_np - ntc_mean).astype(np.float32)    # (n, 2000)
        target_np = delta_hvg_np
        print(f"  target = delta from NTC mean")
    else:
        delta_hvg_np = None
        target_np = hvg_np.astype(np.float32)
        print(f"  target = absolute HVG")

    # ---- per-gene z-score of the target using TRAIN cells only ----
    target_mu  = np.zeros(n_genes, dtype=np.float32)
    target_sig = np.ones(n_genes,  dtype=np.float32)
    if args.target_zscore and args.fix1:
        target_mu  = target_np[train_idx].mean(axis=0).astype(np.float32)
        target_sig = (target_np[train_idx].std(axis=0) + 1e-3).astype(np.float32)
        target_np  = ((target_np - target_mu) / target_sig).astype(np.float32)
        print(f"  target z-scored per gene; mu range [{target_mu.min():.3f},{target_mu.max():.3f}],"
              f" sig range [{target_sig.min():.3f},{target_sig.max():.3f}]")
    elif args.target_zscore and not args.fix1:
        print("  z-score requested but residual targeting is off; skipping.")

    # ---- PCA on (possibly z-scored) target, train cells only ---------------
    if args.fix1:
        pca = PCA(n_components=50, random_state=0)
        pca.fit(target_np[train_idx])
        V = pca.components_.T.astype(np.float32)
    else:
        # Use the absolute-HVG PCA we saved during preprocessing
        V = np.load(ROOT / "processed" / "projectors" / "pca_loadings_V.npy").astype(np.float32)

    target = torch.from_numpy(target_np).float()
    V_T    = torch.from_numpy(V.T).float()                   # (50, 2000)

    # ---- per-gene weight in the MSE loss ----
    #   variance-of-delta-target weighting (preferred), or pseudobulk-std fallback,
    #   or uniform if neither is on.
    if args.gene_var_weight and args.fix1:
        gw = (delta_hvg_np[train_idx].var(axis=0) + 1e-3).astype(np.float32)
        gw = gw / gw.mean()
        gene_weight = gw
        print(f"  gene_weight = var(delta_train) per gene; "
              f"max={gene_weight.max():.2f} min={gene_weight.min():.3f}")
    elif args.fix3:
        pb = np.zeros((n_genes, len(ko_lbl)), dtype=np.float32)
        for k_int in range(len(ko_lbl)):
            idx_k = train_idx[label_np[train_idx] == k_int]
            if len(idx_k) > 0:
                pb[:, k_int] = hvg_np[idx_k].mean(axis=0)
        gene_weight = pb.std(axis=1) + 1e-3
        gene_weight = gene_weight / gene_weight.mean()
        print(f"  gene_weight = std(pseudobulk across conditions); "
              f"max={gene_weight.max():.2f} min={gene_weight.min():.3f}")
    else:
        gene_weight = np.ones(n_genes, dtype=np.float32)
    gene_weight_t = torch.from_numpy(gene_weight).float()

    # ---- Model + optimizer --------------------------------------------------
    model = CVAE_v2(hidden_dim=hidden_dim, latent_dim=latent_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if args.fix5:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    else:
        sched = None

    # ---- Train --------------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    best_val = float("inf"); bad = 0; history = []
    models_dir = ROOT / "models"; models_dir.mkdir(exist_ok=True)
    ckpt_path = models_dir / f"{out_name}.pt"
    t0 = time.time()

    for epoch in range(epochs):
        # Fix 2 OFF: linear β anneal (v1 style) ; ON: free-bits KL (with kl_weight=1)
        if args.fix2:
            beta = 1.0
        else:
            beta = args.beta_max_v1 * min(1.0, epoch / max(1, args.beta_warmup_v1))

        model.train()
        order = train_idx.copy(); rng.shuffle(order)
        tot_mse = tot_kl = tot_n = 0
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            atac_b = atac[b]; tgt_b = target[b]; lab_b = label[b]; gemb_b = gtable[lab_b]

            pca_pred, mu, logvar = model(atac_b, gemb_b)
            pred_hvg = project_delta_to_hvg(pca_pred, V_T)        # (B, 2000)

            mse = (((pred_hvg - tgt_b) ** 2) * gene_weight_t).mean()
            if args.fix2:
                kl  = free_bits_kl(mu, logvar, tau=args.free_bits_tau)
            else:
                kl  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = mse + beta * kl

            # Optional auxiliary losses on the predicted HVG (in delta space if fix1, else absolute)
            if args.mmd_weight > 0:
                aux_mmd = pred_hvg.new_zeros(())
                n_ko_in_batch = 0
                for k in torch.unique(lab_b):
                    mask = lab_b == k
                    if mask.sum() < 4: continue
                    pred_KO = pred_hvg[mask]
                    real_KO = tgt_b[mask]
                    aux_mmd = aux_mmd + mmd_loss(pred_KO, real_KO)
                    n_ko_in_batch += 1
                if n_ko_in_batch > 0:
                    loss = loss + args.mmd_weight * (aux_mmd / n_ko_in_batch)

            if args.mean_align_weight > 0:
                loss = loss + args.mean_align_weight * mean_align_loss(pred_hvg, tgt_b, lab_b)

            opt.zero_grad(); loss.backward(); opt.step()
            bs = len(b)
            tot_mse += mse.item() * bs; tot_kl += kl.item() * bs; tot_n += bs

        train_mse = tot_mse / tot_n; train_kl = tot_kl / tot_n
        if sched: sched.step()

        # val
        model.eval()
        with torch.no_grad():
            atac_v = atac[val_idx]; tgt_v = target[val_idx]; gemb_v = gtable[label[val_idx]]
            pca_pred, mu, logvar = model(atac_v, gemb_v, stochastic=False)
            pred_hvg = project_delta_to_hvg(pca_pred, V_T)
            val_mse = (((pred_hvg - tgt_v) ** 2) * gene_weight_t).mean().item()
            if args.fix2:
                val_kl = free_bits_kl(mu, logvar, tau=args.free_bits_tau).item()
            else:
                val_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()).item()

        cur_lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "lr": cur_lr,
                        "train_mse": train_mse, "train_kl": train_kl,
                        "val_mse": val_mse, "val_kl": val_kl})
        if epoch < 5 or epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  ep{epoch:3d} lr={cur_lr:.2e} "
                  f"train_mse={train_mse:.5f} kl={train_kl:.4f}  "
                  f"val_mse={val_mse:.5f} kl={val_kl:.4f}")

        if val_mse < best_val - 1e-6:
            best_val = val_mse; bad = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args":             vars(args),
                "best_val_mse":     best_val,
                "epoch":            epoch,
                "train_idx":        train_idx,
                "val_idx":          val_idx,
                "test_idx":         test_idx,
                "holdout_idx":      out_idx,
                "holdout_labels":   args.holdout,
                "ko_labels":        ko_lbl,
                "ntc_mean":         ntc_mean,
                "V":                V,                          # absolute or delta basis
                "gene_weight":      gene_weight,
                "fixes":            [args.fix1, args.fix2, args.fix3, args.fix4, args.fix5],
                "hidden_dim":       hidden_dim,
                "latent_dim":       latent_dim,
                "model_version":    "v2",
                "predicts_delta":   bool(args.fix1),
                "gene_emb_source":  args.gene_emb,
                "gene_emb_table":   gtable_np,    # save the actual 10×32 table used
                "input_modality":   args.input,   # "atac" or "rna"
                "target_zscore":    bool(args.target_zscore),
                "gene_var_weight":  bool(args.gene_var_weight),
                "target_mu":        target_mu,    # for Fix C un-zscoring at inference
                "target_sig":       target_sig,
            }, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    with open(models_dir / f"{out_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[done] {time.time()-t0:.1f}s | best val_mse={best_val:.6f}")
    print(f"  ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
