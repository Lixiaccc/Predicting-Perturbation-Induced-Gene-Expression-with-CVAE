#!/usr/bin/env python3
"""
06_plot_main_panels.py

Produces FOUR figures, one per split (heldout / test / val_test / all).
Each figure has a 2x3 panel of metrics:
    Δ-PCC top-20 DE  |  per-cell Δ-PCC top-20  |  Expression-PCC (raw)
    Cosine similarity (delta) | MSE on top-20 DE | Δ-PCC across all 2000 HVGs

X-axis of each panel: model variants (only_fix1, only_fix1_mean, mean_pert,
linear_mf, ntc_identity). One box per (model, split).

Output:
  results/fig_panels_heldout.png
  results/fig_panels_test.png
  results/fig_panels_val_test.png
  results/fig_panels_all.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "figure.titlesize": 17,
})

ROOT = Path("/insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final")

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", type=str, default="results",
                help="Directory under HIGH_DIM_final/ for input metrics_v2.csv "
                     "and output PNGs (default: results)")
args = ap.parse_args()
OUT = ROOT / args.out_dir
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(OUT / "metrics_v2.csv")

GENE_EMB_TAGS = ("geneformer", "genept")
INPUT_TAGS    = ("rna",)                                 # ATAC = default (no suffix)
KNOWN_VARIANTS = ["only_fix1_mmd", "only_fix1_mean",
                  "only_fix1_CD", "only_fix1_C", "only_fix1_D", "only_fix1",
                  "all_fixes", "baseline"]

def label(m):
    if m.startswith("cvae_"):
        rest = m[len("cvae_"):]
        # peel input-modality suffix first (rna), then gene_emb suffix (geneformer)
        input_suffix = ""
        for tag in INPUT_TAGS:
            if rest.endswith(f"_{tag}"):
                input_suffix = f"_{tag}"
                rest = rest[: -len(f"_{tag}")]
                break
        for tag in GENE_EMB_TAGS:
            if rest.endswith(f"_{tag}"):
                rest = rest[: -len(f"_{tag}")]
        for v in sorted(KNOWN_VARIANTS, key=len, reverse=True):
            if rest.startswith(v):
                return v + input_suffix
        return rest + input_suffix
    if "linear_mf" in m:    return "linear_mf"
    if "mean_pert" in m:    return "mean_pert"
    if "ntc_identity" in m: return "ntc_identity"
    return m

df["label"] = df["model"].apply(label)

# Models to compare (CVAE-ATAC + CVAE-RNA + 3 baselines)
PLOT_ORDER = ["only_fix1_CD", "only_fix1_CD_rna", "mean_pert", "linear_mf", "ntc_identity"]
PLOT_ORDER = [v for v in PLOT_ORDER if v in df["label"].unique()]
COLORS = {
    "only_fix1_CD":     "#2E86AB",                      # blue
    "only_fix1_CD_rna": "#9D4EDD",                      # purple
    "mean_pert":        "#3D5A40",
    "linear_mf":        "#C73E1D",
    "ntc_identity":     "#888888",
}
DISPLAY = {
    "only_fix1_CD":     "CVAE (ATAC)",
    "only_fix1_CD_rna": "CVAE (RNA)",
    "mean_pert":        "Mean Pert",
    "linear_mf":        "Linear MF",
    "ntc_identity":     "NTC identity",
}

METRICS = [
    # (column,            panel title,                                            ylim,         lower_better)
    ("pcc_DE20",          "Δ-PCC on top-20 DE genes (population mean)",           (-0.5, 1.0), False),
    ("per_cell_pcc_DE20", "Δ-PCC on top-20 DE genes (per cell)",                  (-0.5, 1.0), False),
    ("expr_pcc_all",      "Expression PCC across all 2,000 HVGs",                 ( 0.5, 1.0), False),
    ("cos_sim_delta",     "Cosine similarity of perturbation effect",             (-0.5, 1.0), False),
    ("mse_DE20",          "Mean squared error on top-20 DE genes",                None,        True),
    ("pcc_all",           "Δ-PCC across all 2,000 HVGs",                          (-0.5, 1.0), False),
]


def panel(ax, sub_df, metric, title, ylim, lower_is_better):
    """Box plot: x=variant, y=metric, one box per variant.
    Annotates each box with mean, std, median ABOVE the plot area."""
    data, kept_variants = [], []
    means, stds, medians = [], [], []
    for v in PLOT_ORDER:
        vals = sub_df[sub_df["label"] == v][metric].dropna().values
        if len(vals) >= 2:
            data.append(vals)
            kept_variants.append(v)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            medians.append(float(np.median(vals)))
    if not data:
        ax.set_title(f"{title} (no data)"); return

    labels = [DISPLAY.get(v, v) for v in kept_variants]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                    showmeans=True, showfliers=False,
                    meanprops=dict(marker="^", markerfacecolor="white",
                                   markeredgecolor="black", markersize=5))
    for patch, v in zip(bp["boxes"], kept_variants):
        patch.set_facecolor(COLORS.get(v, "#aaa"))
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linewidth=0.5)
    # Title needs to sit ABOVE the 3-line annotation -> generous pad
    ax.set_title(title, fontsize=14, pad=58)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=13)

    # ---- annotate each box with mean / std / median ABOVE the panel ----
    for i, (m, s, med) in enumerate(zip(means, stds, medians)):
        ax.text(i + 1, 1.015, f"μ={m:.2f}\nσ={s:.2f}\nmed={med:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=13,
                color="#222", linespacing=0.95)

    if lower_is_better:
        ax.text(0.99, 0.97, "lower=better", transform=ax.transAxes, ha="right",
                va="top", fontsize=13, style="italic", color="#666")


for split in ["heldout", "test", "val_test", "all"]:
    sub = df[df["split"] == split]
    if sub.empty:
        print(f"[skip] no rows for split={split}")
        continue
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    for ax, (m, t, yl, lb) in zip(axes.flat, METRICS):
        panel(ax, sub, m, t, yl, lb)
    title = {
        "heldout":  "Zero-shot prediction performance on held-out perturbations",
        "test":     "In-distribution prediction performance on held-out test cells",
        "val_test": "In-distribution prediction performance on validation + test cells",
        "all":      "Prediction performance on all cells",
    }[split]
    fig.suptitle(title, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=3.0, w_pad=2.0)
    out = OUT / f"fig_panels_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.name}")
