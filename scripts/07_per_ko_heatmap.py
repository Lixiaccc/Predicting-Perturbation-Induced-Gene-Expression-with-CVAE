#!/usr/bin/env python3
"""
07_per_ko_heatmap.py

"Confusion-matrix style" heatmaps: which (KO × model) combinations are
accurate, which fail. One heatmap per metric, one figure per split.

Rows: model variants  (only_fix1, mean_pert, linear_mf)
Cols: 9 KOs
Cell value: mean of metric across leave-2-out pairs that include this KO in this split.

Output (per split):
  results/fig_per_ko_heatmap_heldout.png
  results/fig_per_ko_heatmap_test.png
  results/fig_per_ko_heatmap_val_test.png
  results/fig_per_ko_heatmap_all.png
"""

import argparse
from pathlib import Path
import re
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

ROW_ORDER = ["only_fix1_CD", "only_fix1_CD_rna", "mean_pert", "linear_mf"]
DISPLAY = {
    "only_fix1_CD":     "CVAE (ATAC)",
    "only_fix1_CD_rna": "CVAE (RNA)",
    "mean_pert":      "Mean Pert",
    "linear_mf":      "Linear MF",
    "ntc_identity":   "NTC identity",
}
KOS = ["ACTL6A", "DMAP1", "EP400", "EZH2", "SMARCA4", "SMARCB1", "SMARCE1", "SUZ12", "YY1"]

METRICS = [
    ("pcc_DE20",          "Δ-PCC on top-20 DE genes (population mean)",   "RdBu_r", -0.5, 1.0),
    ("per_cell_pcc_DE20", "Δ-PCC on top-20 DE genes (per cell)",          "RdBu_r", -0.5, 1.0),
    ("pcc_all",           "Δ-PCC across all 2,000 HVGs",                  "RdBu_r", -0.5, 1.0),
    ("expr_pcc_all",      "Expression PCC across all 2,000 HVGs",         "viridis", 0.5, 1.0),
    ("cos_sim_delta",     "Cosine similarity of perturbation effect",     "RdBu_r", -0.5, 1.0),
    ("mse_DE20",          "Mean squared error on top-20 DE genes",        "viridis_r", 0,  None),
]

SPLIT_TITLES = {
    "heldout":  "Per-knockout accuracy on held-out perturbations (zero-shot)",
    "test":     "Per-knockout accuracy on held-out test cells (in-distribution)",
    "val_test": "Per-knockout accuracy on validation + test cells",
    "all":      "Per-knockout accuracy on all cells",
}

for split in ["heldout", "test", "val_test", "all"]:
    sub = df[df["split"] == split]
    rows_present = [v for v in ROW_ORDER if v in sub["label"].unique()]
    if not rows_present or sub.empty:
        print(f"[skip] no rows for split={split}")
        continue

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for ax, (metric, title, cmap, vmin, vmax) in zip(axes.flat, METRICS):
        pivot = sub.pivot_table(index="label", columns="ko",
                                values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=rows_present, columns=KOS)
        if vmax is None:
            vmax = float(np.nanquantile(pivot.values, 0.95)) if not np.isnan(pivot.values).all() else 1.0
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(KOS))); ax.set_xticklabels(KOS, rotation=30, ha="right", fontsize=13)
        ax.set_yticks(range(len(rows_present)))
        ax.set_yticklabels([DISPLAY.get(v, v) for v in rows_present], fontsize=13)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    text_color = "white" if (cmap == "RdBu_r" and abs(v) > 0.5) else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=13, color=text_color)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=13)
        ax.set_title(title, fontsize=14)

    fig.suptitle(SPLIT_TITLES[split], fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / f"fig_per_ko_heatmap_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.name}")
