#  CVAE for KO RNA prediction

A conditional VAE that predicts post-knockout RNA expression (HVG-log1p) from a
cell's ATAC profile and a gene-identity embedding for the KO target. Trained
and evaluated on a 9-KO + NTC dataset (10 conditions).

The final model is **`only_fix1_CD`**: residual targeting (predict δ from NTC
mean) + per-gene z-scored target (Fix C) + variance-weighted MSE on δ (Fix D).

## Layout

```
HIGH_DIM_final/
├── model/
│   └── cvae.py                # CVAE_v2 (encoder/decoder), free-bits KL,
│                              # delta->HVG projection helpers
├── scripts/
│   ├── 01_generate_genept.py  # GenePT embeddings for the 9 KO genes (OpenAI)
│   ├── 02_preprocess.py       # RNA -> HVG/PCA, ATAC -> TF-IDF/LSI,
│   │                          # GenePT -> 32-D, writes processed/cells.npz
│   ├── 03_train.py            # CVAE training (variants + ablations)
│   ├── 04_run_leave2out.sh    # leave-2-out CV across all C(9,2)=36 KO pairs
│   ├── 05_evaluate.py         # metrics for every checkpoint + 3 baselines
│   ├── 06_plot_main_panels.py # 2x3 metric panels per split
│   ├── 07_per_ko_heatmap.py   # per-KO heatmaps
│   └── 08_distance_split.py   # NTC distance-split robustness experiment
├── processed/
│   ├── cells.npz              # aligned per-cell arrays (ATAC_LSI, RNA_PCA,
│   │                          # RNA_HVG_log1p, label_int, gene embeddings)
│   └── projectors/            # PCA loadings, TF-IDF/SVD, HVG list,
│                              # GenePT/Geneformer gene-embedding tables
├── models/                    # trained CVAE checkpoints (.pt) + history
├── results/                   # metrics_v2.csv + figures (.png)
└── FINAL_MODEL_HANDOFF_minimal/  # OT/moscot baseline outputs (separate)
```

## Data

Cells: 9 epigenetic-regulator KOs (ACTL6A, DMAP1, EP400, EZH2, SMARCA4,
SMARCB1, SMARCE1, SUZ12, YY1) plus NTC controls. Multimodal RNA + ATAC measured
on the same cells; sources read by [02_preprocess.py](scripts/02_preprocess.py)
from `~/epifoundatoin_v2_gene/040_epi_{rna,atac}_filtered.h5ad`.

After preprocessing, [processed/cells.npz](processed/cells.npz) contains:
- `ATAC_LSI` (n, 50): TF-IDF + TruncatedSVD(51) with depth-correlated
  component dropped.
- `RNA_PCA` (n, 50): PCA(50) on HVG-log1p (alternate encoder input).
- `RNA_HVG_log1p` (n, 2000): the prediction target.
- `gene_emb_genept` (10, 32): GenePT embeddings, PCA-projected; NTC row=0.
- `label_int`, `guide_target`, `barcode`, `ko_labels`.

## Model

[model/cvae.py](model/cvae.py)::`CVAE_v2`:

```
[ATAC_LSI(50) ; gene_emb(32)]  ->  Linear(82->64) + GELU
                                ->  mu / logvar heads (64->64)
z (64) -> decoder Linear(64->50) -> delta_PCA(50)
delta_HVG(2000) = delta_PCA @ V_delta.T   (V fit on training-set delta)
HVG = NTC_mean + delta_HVG                (residual targeting)
```

~13.4k learnable params. Training: AdamW, batch 64, cosine LR (with `--fix5`).

## Running the pipeline

The Python env used in the leave-2-out runner is
`/insomnia001/depts/houlab/users/lc3716/envs/epifoundation`.

```bash
# 0. one-time: GenePT embeddings (needs OPENAI_API_KEY)
python scripts/01_generate_genept.py

# 1. one-time: build processed/cells.npz + projectors
python scripts/02_preprocess.py

# 2a. train final model on all KOs (3-way 80/10/10 split)
python scripts/03_train.py --variant only_fix1_CD

# 2b. or hold out a pair and run leave-2-out CV (36 pairs)
bash   scripts/04_run_leave2out.sh           # default: only_fix1_CD genept atac
bash   scripts/04_run_leave2out.sh only_fix1_CD genept rna   # RNA-input variant

# 3. metrics for every checkpoint in models/ + linear/mean/NTC baselines
python scripts/05_evaluate.py

# 4. figures
python scripts/06_plot_main_panels.py
python scripts/07_per_ko_heatmap.py
python scripts/08_distance_split.py          # separate distance-split experiment
```

## Variants and ablation flags

Variants live in `VARIANT_FIXES` / `VARIANT_AUX_LOSS` in
[scripts/03_train.py](scripts/03_train.py):

| variant            | fix1 (δ) | fix2 (free-bits KL) | fix3 (gene-w MSE) | fix4 (latent=64) | fix5 (cosine LR) | extras                    |
|--------------------|---------:|--------------------:|------------------:|-----------------:|-----------------:|---------------------------|
| `baseline`         |        0 |                   0 |                 0 |                0 |                0 |                           |
| `only_fix1`        |        1 |                   0 |                 0 |                0 |                0 |                           |
| `only_fix1_C`      |        1 |                   0 |                 0 |                0 |                0 | target z-score            |
| `only_fix1_D`      |        1 |                   0 |                 0 |                0 |                0 | var-weighted δ MSE        |
| `only_fix1_CD`     |        1 |                   0 |                 0 |                0 |                0 | C + D (final)             |
| `only_fix1_mmd`    |        1 |                   0 |                 0 |                0 |                0 | MMD aux loss              |
| `only_fix1_mean`   |        1 |                   0 |                 0 |                0 |                0 | per-KO mean-align loss    |
| `all_fixes`        |        1 |                   1 |                 1 |                1 |                1 |                           |

Useful CLI knobs: `--holdout EZH2 SMARCA4`, `--gene_emb {genept,geneformer}`,
`--input {atac,rna}`, `--seed`, `--val_frac`, `--test_frac`.

Each checkpoint stores the splits (`train_idx`/`val_idx`/`test_idx`/`holdout_idx`),
the `V` basis, `ntc_mean`, gene weights, target μ/σ (for un-z-scoring), and the
exact gene-embedding table used at training time, so evaluation is fully
reproducible from the `.pt` alone.

## Splits and evaluation

Stratified 3-way 80/10/10 split by `guide_target` is computed in
[stratified_3way](scripts/03_train.py#L91). [05_evaluate.py](scripts/05_evaluate.py)
reports δ-PCC and expression-PCC over multiple cell sets:

- `test`     — test cells of that KO (clean in-distribution)
- `val_test` — val + test cells (lightly contaminated)
- `all`      — all cells (inflated; useful as a sanity ceiling)
- `heldout`  — heldout-KO cells (only for `_loko_*` checkpoints)

Baselines computed in the same script: linear matrix factorization
(rank-`LINEAR_K`), mean-perturbation pseudobulk, and NTC-identity.
