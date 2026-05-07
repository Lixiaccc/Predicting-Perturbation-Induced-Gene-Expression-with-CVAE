#!/usr/bin/env python3
"""
05_evaluate.py

Evaluate every cvae_* checkpoint in models/ + the 3 baselines from
Nature Methods 2025, on multiple cell splits:

    split = "all"        : δ_real over all cells of that KO        (train+val+test, inflated)
    split = "test"       : δ_real over test cells of that KO       (clean in-distribution)
    split = "val_test"   : δ_real over val+test cells of that KO   (lightly contaminated)
    split = "heldout"    : δ_real over heldout cells (only for held-out KOs in loko models)

For each (model, ko, split): compute Δ-PCC_all and Δ-PCC_DE20.
Output: <out_dir>/metrics_v2.csv (long format).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

ROOT = Path("/insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final")
sys.path.insert(0, str(ROOT))
from model.cvae import CVAE_v2, project_delta_to_hvg


KO_LABELS = ["NTC", "ACTL6A", "DMAP1", "EP400", "EZH2",
             "SMARCA4", "SMARCB1", "SMARCE1", "SUZ12", "YY1"]
KO_GENES = KO_LABELS[1:]
LINEAR_K = 16


def pcc_safe(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(pearsonr(a, b)[0])


def topk_pcc(delta_real, delta_pred, k=20):
    if np.std(delta_real) < 1e-10:
        return float("nan")
    k = min(k, len(delta_real))
    idx = np.argsort(np.abs(delta_real))[-k:]
    return pcc_safe(delta_real[idx], delta_pred[idx])


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def per_cell_pcc(pred_cells, real_mean, k_or_idx, top_k_idx=None):
    """Vectorized average per-cell PCC: mean over rows of pearsonr(pred_cells[i], real_mean).
    pred_cells: (B, 2000), real_mean: (2000,), top_k_idx: optional array of column indices."""
    if top_k_idx is not None:
        X = pred_cells[:, top_k_idx]
        y = real_mean[top_k_idx]
    else:
        X = pred_cells
        y = real_mean
    if np.std(y) < 1e-10:
        return float("nan")
    Xc = X - X.mean(axis=1, keepdims=True)              # (B, K)
    yc = y - y.mean()                                    # (K,)
    Xn = np.linalg.norm(Xc, axis=1)                     # (B,)
    yn = np.linalg.norm(yc) + 1e-30
    valid = Xn > 1e-10
    if not valid.any():
        return float("nan")
    num = Xc[valid] @ yc                                 # (B_valid,)
    pccs = num / (Xn[valid] * yn)
    return float(pccs.mean())


DE_KS = (20, 50, 100, 500, 1000)   # K values for top-K DE PCC


def cvae_predict_hvg(ckpt, ntc_input, ko_int, gtable_default):
    """Run a CVAE checkpoint to predict HVG profiles.
    `ntc_input` is the 50-D encoder feature for each control cell — ATAC_LSI when
    ckpt['input_modality'] == 'atac' (default), RNA_PCA when 'rna'. The caller
    selects the right feature based on the checkpoint.
    Handles: predicts_delta + (Fix C: target z-scoring un-scaling).
    Returns: (B, 2000) absolute HVG_log1p predictions.
    """
    hidden = ckpt.get("hidden_dim", 64)
    latent = ckpt.get("latent_dim", 64)
    model = CVAE_v2(hidden_dim=hidden, latent_dim=latent)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    V_T = torch.from_numpy(ckpt["V"].T).float()                  # (50, 2000)
    if "gene_emb_table" in ckpt:
        gtable = torch.from_numpy(ckpt["gene_emb_table"]).float()
    else:
        gtable = gtable_default
    with torch.no_grad():
        x = torch.from_numpy(ntc_input).float()
        gemb = gtable[torch.full((x.shape[0],), ko_int, dtype=torch.long)]
        pca_pred, _, _ = model(x, gemb, stochastic=False)
        pred = project_delta_to_hvg(pca_pred, V_T).numpy()        # (B, 2000)
    # Fix C: un-z-score if the model trained on z-scored targets
    if ckpt.get("target_zscore", False):
        target_mu  = ckpt["target_mu"]
        target_sig = ckpt["target_sig"]
        pred = pred * target_sig + target_mu                      # back to original delta scale
    if ckpt.get("predicts_delta", False):
        pred = pred + ckpt["ntc_mean"]                            # add NTC mean back
    return pred


def linear_mf_fit(Y_train, conditions, full_gene_to_idx, K=LINEAR_K):
    K = min(K, len(conditions) - 1)
    if K < 2:
        K = max(2, len(conditions) - 1)
    b = Y_train.mean(axis=1)
    Y_c = Y_train - b[:, None]
    pca = PCA(n_components=K, random_state=0)
    G = pca.fit_transform(Y_c).astype(np.float32)
    P = np.zeros((len(conditions), K), dtype=np.float32)
    for ci, cond in enumerate(conditions):
        if cond == "NTC":
            continue
        gi = full_gene_to_idx.get(cond)
        if gi is not None:
            P[ci] = G[gi]
    GtG = G.T @ G; PtP = P.T @ P; GtY_P = G.T @ Y_c @ P
    W = np.linalg.solve(GtG, np.linalg.solve(PtP.T, GtY_P.T).T)
    return G, W, b


def linear_mf_pred(G, W, b, ko_gene, full_gene_to_idx, n_hvg=2000):
    gi = full_gene_to_idx.get(ko_gene)
    if gi is None:
        return None
    return (G @ W @ G[gi] + b)[:n_hvg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results",
                    help="Directory under HIGH_DIM_final/ for metrics_v2.csv (default: results)")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = ROOT / "processed"; proj = proc / "projectors"
    print("[load] cells.npz")
    d = np.load(proc / "cells.npz", allow_pickle=True)
    atac    = d["ATAC_LSI"]                  # 50-D for input_modality="atac"
    rna_pca = d["RNA_PCA"]                   # 50-D for input_modality="rna"
    hvg     = d["RNA_HVG_log1p"]
    label_int = d["label_int"]
    gtable  = torch.from_numpy(d["gene_emb_genept"]).float()
    barcodes = d["barcode"]
    ko_lbl  = list(d["ko_labels"])
    label_to_int = {l: i for i, l in enumerate(ko_lbl)}

    with open(proj / "hvg_list.txt") as f:
        hvgs = [l.strip() for l in f]

    # Augment with missing KO genes for linear baseline
    print("[load] full RNA h5ad to augment with missing KO gene rows")
    import scanpy as sc
    rna = sc.read_h5ad("/insomnia001/depts/houlab/users/lc3716/epifoundatoin_v2_gene/040_epi_rna_filtered.h5ad")
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    if not np.array_equal(rna.obs_names.values, barcodes):
        idx = rna.obs_names.get_indexer(barcodes); rna = rna[idx].copy()
    missing = [g for g in KO_GENES if g not in hvgs]
    extra_idx = [rna.var_names.get_loc(g) for g in missing if g in rna.var_names]
    extra_genes = [m for m, g in zip(missing, missing) if g in rna.var_names]
    extra_X = rna.X[:, extra_idx]
    if hasattr(extra_X, "toarray"): extra_X = extra_X.toarray()
    extra_X = extra_X.astype(np.float32)
    full_X = np.concatenate([hvg, extra_X], axis=1)
    full_gene_to_idx = {g: i for i, g in enumerate(list(hvgs) + extra_genes)}
    print(f"  augmented: {full_X.shape}; appended {len(extra_genes)} missing KO genes")

    # Discover CVAE checkpoints (matches both legacy `cvae_v2_*` and current `cvae_*`)
    all_ckpts = sorted((ROOT / "models").glob("cvae_*.pt"))
    ckpt_paths = list(all_ckpts)
    print(f"\n[ckpt] {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"   {p.name}")

    rows = []

    # ---- linear MF + mean_pert + ntc_identity baselines ----
    # We compute baselines ONCE per (loko vs full setting) since they don't
    # depend on a neural model. Use train cells from any v2 checkpoint to
    # define the conditions; if no checkpoint exists, fall back to all cells.
    print("\n[baselines] fitting linear MF for full and loko configurations")

    def build_baselines(train_idx, conditions):
        Y_train = np.stack(
            [full_X[train_idx[label_int[train_idx] == label_to_int[c]]].mean(0)
             for c in conditions], axis=1
        )
        G, W, b = linear_mf_fit(Y_train, conditions, full_gene_to_idx, K=LINEAR_K)
        pert_mean = np.stack(
            [full_X[train_idx[label_int[train_idx] == label_to_int[c]]].mean(0)
             for c in conditions if c != "NTC"], axis=1
        ).mean(axis=1)[:hvg.shape[1]]
        ntc_train_mean = full_X[train_idx[label_int[train_idx] == label_to_int["NTC"]]].mean(0)[:hvg.shape[1]]
        return G, W, b, pert_mean, ntc_train_mean

    # full_ckpts = checkpoints trained on ALL 9 KOs (no holdout in name)
    # loko_ckpts = leave-K-out checkpoints
    full_ckpts = [p for p in ckpt_paths if "_loko" not in p.name]
    loko_ckpts = [p for p in ckpt_paths if "_loko" in p.name]
    print(f"  full_ckpts: {len(full_ckpts)}  loko_ckpts: {len(loko_ckpts)}")

    # Fit FULL baseline (linear MF + mean_pert + NTC identity) using train_idx of any
    # full checkpoint, OR fall back to using all cells if no full checkpoint exists.
    if full_ckpts:
        ck = torch.load(full_ckpts[0], map_location="cpu", weights_only=False)
        train_full = ck["train_idx"]
        cond_full = list(ko_lbl)
        G_full, W_full, b_full, pert_full, ntc_train_full = build_baselines(train_full, cond_full)
        have_full_baseline = True
    else:
        print("  no full checkpoint -> using all cells as 'training set' for full baseline")
        train_full = np.arange(len(label_int))
        cond_full = list(ko_lbl)
        G_full, W_full, b_full, pert_full, ntc_train_full = build_baselines(train_full, cond_full)
        have_full_baseline = True

    # NOTE: per-loko-pair baselines are refit individually inside the loko loop below
    # (each pair has its own held-out conditions, so we can't use a single global fit).

    # ---- helper to compute metrics for one (model_name, ko, split) ----
    def emit(model_name, ko, split, n_real, delta_real, delta_pred,
             real_ko_profile=None, pred_ko_profile=None,
             pred_cells=None, real_ntc_mean=None):
        """delta_real, delta_pred: 2000-D delta vectors (population mean).
        real_ko_profile, pred_ko_profile: 2000-D RAW expression vectors.
        pred_cells: (B, 2000) per-cell CVAE predictions (HVG space).
        real_cells: (M, 2000) real KO cells in this split.
        real_ntc_mean: (2000,) NTC mean (for per-cell delta).
        """
        row = {
            "model": model_name, "ko": ko, "split": split,
            "n_real_ko_cells": int(n_real),
            "norm_delta_real": float(np.linalg.norm(delta_real)),
            "norm_delta_pred": float(np.linalg.norm(delta_pred)),
            "pcc_all":      pcc_safe(delta_real, delta_pred),
            "expr_pcc_all": (pcc_safe(real_ko_profile, pred_ko_profile)
                             if real_ko_profile is not None else float("nan")),
        }
        # multi-K mean-level Δ-PCC and expression-PCC + MSE
        for K in DE_KS:
            row[f"pcc_DE{K}"] = topk_pcc(delta_real, delta_pred, K)
            idx = np.argsort(np.abs(delta_real))[-K:]
            row[f"mse_DE{K}"] = float(((delta_real[idx] - delta_pred[idx]) ** 2).mean())
            if real_ko_profile is not None:
                row[f"expr_pcc_DE{K}"] = topk_pcc_with_idx_from(
                    real_ko_profile, pred_ko_profile, delta_real, K)
            else:
                row[f"expr_pcc_DE{K}"] = float("nan")
        # MSE on full delta (across all 2000 HVGs)
        row["mse_all"] = float(((delta_real - delta_pred) ** 2).mean())
        # cosine similarity (delta and expression)
        row["cos_sim_delta"]  = cos_sim(delta_real, delta_pred)
        row["cos_sim_expr"]   = (cos_sim(real_ko_profile, pred_ko_profile)
                                 if real_ko_profile is not None else float("nan"))

        # ---- per-cell metrics ----
        if pred_cells is not None and real_ntc_mean is not None:
            pred_deltas = pred_cells - real_ntc_mean[None, :]   # (B, 2000)
            for K in DE_KS:
                idx = np.argsort(np.abs(delta_real))[-K:]
                row[f"per_cell_pcc_DE{K}"] = per_cell_pcc(pred_deltas, delta_real, K, idx)
            row["per_cell_pcc_all"] = per_cell_pcc(pred_deltas, delta_real, None, None)
        else:
            for K in DE_KS:
                row[f"per_cell_pcc_DE{K}"] = float("nan")
            row["per_cell_pcc_all"] = float("nan")

        rows.append(row)

    def topk_pcc_with_idx_from(a, b, score, k=20):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return float("nan")
        idx = np.argsort(np.abs(score))[-k:]
        return pcc_safe(a[idx], b[idx])

    # ---- loop over checkpoints ----
    for cp in ckpt_paths:
        print(f"\n[eval] {cp.name}")
        ckpt = torch.load(cp, map_location="cpu", weights_only=False)
        train_idx = ckpt["train_idx"]; val_idx = ckpt["val_idx"]
        test_idx  = ckpt["test_idx"]; out_idx = ckpt["holdout_idx"]
        held_labels = ckpt.get("holdout_labels", [])
        is_loko = bool(held_labels)

        for ko in KO_GENES:
            ko_int = label_to_int[ko]
            cells_of_ko = np.where(label_int == ko_int)[0]
            cells_of_ntc = np.where(label_int == label_to_int["NTC"])[0]

            # define splits
            split_defs = {}
            if ko in held_labels:
                split_defs["heldout"] = (np.intersect1d(out_idx, cells_of_ko),
                                         np.intersect1d(np.concatenate([train_idx, val_idx, test_idx]), cells_of_ntc))
            else:
                # in-dist: define test, val_test, all
                test_ko_idx = np.intersect1d(test_idx, cells_of_ko)
                val_ko_idx  = np.intersect1d(val_idx,  cells_of_ko)
                split_defs["test"]     = (test_ko_idx,
                                          np.intersect1d(test_idx, cells_of_ntc))
                split_defs["val_test"] = (np.union1d(val_ko_idx, test_ko_idx),
                                          np.union1d(np.intersect1d(val_idx, cells_of_ntc),
                                                     np.intersect1d(test_idx, cells_of_ntc)))
                split_defs["all"]      = (cells_of_ko, cells_of_ntc)

            # CVAE prediction (uses split's NTC cells as input — feature picked by ckpt's input_modality)
            input_feat = rna_pca if ckpt.get("input_modality", "atac") == "rna" else atac
            for split_name, (ko_eval_idx, ntc_input_idx) in split_defs.items():
                if len(ko_eval_idx) == 0 or len(ntc_input_idx) == 0:
                    continue
                ntc_input = input_feat[ntc_input_idx]
                pred = cvae_predict_hvg(ckpt, ntc_input, ko_int, gtable)  # (n_ntc, 2000)
                pred_mean = pred.mean(axis=0)
                real_ntc_mean = hvg[ntc_input_idx].mean(axis=0)
                real_ko_mean = hvg[ko_eval_idx].mean(axis=0)
                delta_real = real_ko_mean - real_ntc_mean
                delta_pred = pred_mean - real_ntc_mean
                emit(cp.stem, ko, split_name, len(ko_eval_idx), delta_real, delta_pred,
                     real_ko_profile=real_ko_mean, pred_ko_profile=pred_mean,
                     pred_cells=pred, real_ntc_mean=real_ntc_mean)

    # ---- baselines on the same split definitions ----
    # We use the train_idx from one full and one loko checkpoint to define splits
    # (any v2 checkpoint with that holdout setting works).
    print("\n[baselines] computing per-split metrics")
    if full_ckpts:
        ck = torch.load(full_ckpts[0], map_location="cpu", weights_only=False)
        for ko in KO_GENES:
            ko_int = label_to_int[ko]
            cells_of_ko  = np.where(label_int == ko_int)[0]
            cells_of_ntc = np.where(label_int == label_to_int["NTC"])[0]
            test_ko = np.intersect1d(ck["test_idx"], cells_of_ko)
            val_ko  = np.intersect1d(ck["val_idx"],  cells_of_ko)
            for split_name, ko_eval, ntc_eval in [
                ("all",      cells_of_ko, cells_of_ntc),
                ("test",     test_ko,     np.intersect1d(ck["test_idx"], cells_of_ntc)),
                ("val_test", np.union1d(val_ko, test_ko),
                             np.union1d(np.intersect1d(ck["val_idx"], cells_of_ntc),
                                        np.intersect1d(ck["test_idx"], cells_of_ntc))),
            ]:
                if len(ko_eval) == 0 or len(ntc_eval) == 0: continue
                real_ko = hvg[ko_eval].mean(0); real_ntc = hvg[ntc_eval].mean(0)
                delta_real = real_ko - real_ntc
                ntc_cells_split = hvg[ntc_eval]
                # linear MF
                pred_lin = linear_mf_pred(G_full, W_full, b_full, ko, full_gene_to_idx)
                if pred_lin is not None:
                    delta_lin      = pred_lin - real_ntc
                    pred_cells_lin = ntc_cells_split + delta_lin[None, :]
                    emit("linear_mf_full", ko, split_name, len(ko_eval),
                         delta_real, delta_lin,
                         real_ko_profile=real_ko, pred_ko_profile=pred_lin,
                         pred_cells=pred_cells_lin, real_ntc_mean=real_ntc)
                # mean perturbations
                delta_mp      = pert_full - real_ntc
                pred_cells_mp = ntc_cells_split + delta_mp[None, :]
                emit("mean_pert_full", ko, split_name, len(ko_eval),
                     delta_real, delta_mp,
                     real_ko_profile=real_ko, pred_ko_profile=pert_full,
                     pred_cells=pred_cells_mp, real_ntc_mean=real_ntc)
                # NTC identity
                emit("ntc_identity", ko, split_name, len(ko_eval),
                     delta_real, np.zeros_like(delta_real),
                     real_ko_profile=real_ko, pred_ko_profile=real_ntc,
                     pred_cells=ntc_cells_split, real_ntc_mean=real_ntc)

    # ---- Per-loko-checkpoint baselines: same 7 training KOs the CVAE saw,
    #      evaluated on EVERY split (heldout/test/val_test/all) per pair ----
    # Dedupe: ATAC and RNA checkpoints for the same held-out pair share train_idx,
    # so we only refit baselines once per unique pair.
    seen_pairs = set()
    unique_ckpts = []
    for cp in loko_ckpts:
        ck_peek = torch.load(cp, map_location="cpu", weights_only=False)
        pair_key = tuple(sorted(ck_peek["holdout_labels"]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        unique_ckpts.append(cp)
    n_pairs = len(unique_ckpts)
    print(f"  refitting baselines on {n_pairs} unique held-out pairs ({len(loko_ckpts)} ckpts -> deduped)")

    for bi, cp in enumerate(unique_ckpts, 1):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        held_labels    = ck["holdout_labels"]
        train_idx_loko = ck["train_idx"]
        val_idx_loko   = ck["val_idx"]
        test_idx_loko  = ck["test_idx"]
        cond_loko      = [c for c in ko_lbl if c not in held_labels]
        held_tag = "_".join(sorted(held_labels))
        print(f"  [pair {bi}/{n_pairs}] heldout={held_tag}   ({n_pairs - bi} remaining)")
        G_l, W_l, b_l, pert_l, _ = build_baselines(train_idx_loko, cond_loko)

        for ko in KO_GENES:
            ko_int = label_to_int[ko]
            cells_of_ko  = np.where(label_int == ko_int)[0]
            cells_of_ntc = np.where(label_int == label_to_int["NTC"])[0]

            # Build split definitions matching how the CVAE was evaluated
            if ko in held_labels:
                # Held-out KO: only "heldout" split (its cells were never seen)
                split_defs = {"heldout": (cells_of_ko, cells_of_ntc)}
                tag_suffix = f"_HELDOUT__{held_tag}"
            else:
                # Seen KO: test / val_test / all on its train-time splits
                test_ko = np.intersect1d(test_idx_loko, cells_of_ko)
                val_ko  = np.intersect1d(val_idx_loko,  cells_of_ko)
                test_ntc = np.intersect1d(test_idx_loko, cells_of_ntc)
                val_ntc  = np.intersect1d(val_idx_loko,  cells_of_ntc)
                split_defs = {
                    "test":     (test_ko, test_ntc),
                    "val_test": (np.union1d(val_ko, test_ko),
                                 np.union1d(val_ntc, test_ntc)),
                    "all":      (cells_of_ko, cells_of_ntc),
                }
                tag_suffix = f"__{held_tag}"

            for split_name, (ko_eval, ntc_eval) in split_defs.items():
                if len(ko_eval) == 0 or len(ntc_eval) == 0:
                    continue
                real_ko  = hvg[ko_eval].mean(0)
                real_ntc = hvg[ntc_eval].mean(0)
                delta_real = real_ko - real_ntc
                ntc_cells_split = hvg[ntc_eval]

                # linear MF
                pred_lin = linear_mf_pred(G_l, W_l, b_l, ko, full_gene_to_idx)
                if pred_lin is not None:
                    delta_lin      = pred_lin - real_ntc
                    pred_cells_lin = ntc_cells_split + delta_lin[None, :]
                    emit(f"linear_mf_loko{tag_suffix}", ko, split_name, len(ko_eval),
                         delta_real, delta_lin,
                         real_ko_profile=real_ko, pred_ko_profile=pred_lin,
                         pred_cells=pred_cells_lin, real_ntc_mean=real_ntc)
                # mean_pert
                delta_mp      = pert_l - real_ntc
                pred_cells_mp = ntc_cells_split + delta_mp[None, :]
                emit(f"mean_pert_loko{tag_suffix}", ko, split_name, len(ko_eval),
                     delta_real, delta_mp,
                     real_ko_profile=real_ko, pred_ko_profile=pert_l,
                     pred_cells=pred_cells_mp, real_ntc_mean=real_ntc)
                # ntc_identity
                emit(f"ntc_identity_loko{tag_suffix}", ko, split_name, len(ko_eval),
                     delta_real, np.zeros_like(delta_real),
                     real_ko_profile=real_ko, pred_ko_profile=real_ntc,
                     pred_cells=ntc_cells_split, real_ntc_mean=real_ntc)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "metrics_v2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}  ({len(df)} rows)")

    print("\n=== ABLATION SUMMARY: Δ-PCC by (model, split) ===")
    for K in DE_KS:
        print(f"\n--- mean Δ-PCC_DE{K} ---")
        print(df.pivot_table(index="model", columns="split", values=f"pcc_DE{K}",
                             aggfunc="mean").round(3).to_string())
    print(f"\n--- mean Δ-PCC_all (over all 2000 HVGs) ---")
    print(df.pivot_table(index="model", columns="split", values="pcc_all",
                         aggfunc="mean").round(3).to_string())
    print(f"\n--- mean Expression-PCC_all (raw mean-to-mean) ---")
    print(df.pivot_table(index="model", columns="split", values="expr_pcc_all",
                         aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
