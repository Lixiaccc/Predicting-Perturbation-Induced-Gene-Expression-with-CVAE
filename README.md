# Predicting Perturbation-Induced Gene Expression from Baseline Chromatin Accessibility Using OT and Multimodal Generative Modeling

## Optimal transport / moscot analysis

The optimal transport (OT) portion of the project is contained in two Google Colab notebooks under `notebooks/`. These notebooks were developed and run in Colab, and the submitted `.ipynb` files should still contain the executed outputs, figures, and summary tables used for the report. If rerunning them, update the Google Drive / repository paths near the top of each notebook to match your local setup.

For a quick test of the OT results we recommend using the moscot_RNA_only.ipynb since it is far smaller and quicker to run. The grid parameters are at the top of the notebook and can be reduced further to quicken it further.

```text
notebooks/
   ├── moscot_RNA_only.ipynb
   │   # RNA-only moscot grid search.
   │   # Fits OT plans from NTC cells to each KO population across epsilon/tau settings.
   │   # Uses common RNA PCA evaluation based on the saved transport plan barycenter.
   │   # Produces the RNA-only hyperparameter heatmap and the final OT handoff used by the CVAE.
   │   # The notebook output cells should contain the final heatmap and summary tables.
   │
   └── moscot_multimodal.ipynb
       # Multimodal OT comparison.
       # Compares RNA-only, ATAC LSI-only, RNA PCA + ATAC LSI concatenation,
       # GW with ATAC geometry, and FGW variants using ATAC/RNA geometry and gene activity.
       # Re-evaluates saved OT plans in common RNA PCA space using barycentric projection.
       # Also includes ATAC LSI evaluation and sanity checks comparing moscot translated
       # outputs against direct plan-barycenter evaluation.
       # The notebook output cells should contain the final multimodal heatmaps and sanity-check tables.
```

### OT input data

Large input data files are not tracked directly in this repository unless required for submission. The OT notebooks expect the Mixscape-filtered paired multiome dataset or equivalent processed files containing:

```text
- paired scRNA and scATAC profiles from the same cells
- cell condition / perturbation labels
- NTC control labels
- RNA PCA representation
- ATAC LSI representation
- ATAC gene activity representation for FGW experiments
```

The final analysis uses 1,144 NTC cells and 1,521 high-confidence knockout cells across nine perturbations after Mixscape filtering:

```text
ACTL6A, DMAP1, EP400, EZH2, SMARCA4, SMARCB1, SMARCE1, SUZ12, YY1
```

The raw public dataset is available from Zenodo:

```text
Metzner, Southard, and Norman, GEX and ATAC count matrices for pEM040 sgRNA singlets
https://zenodo.org/records/15116138
```

### OT output files

The RNA-only notebook generates the handoff used by the CVAE pipeline. This folder should remain stable because downstream scripts depend on it:

```text
FINAL_MODEL_HANDOFF_minimal/
   # Final RNA-only OT handoff used by the CVAE pipeline.
```

Report figures and OT summary files are stored under:

```text
results/ot/
   ├── OT_rna_rna.png
   │   # RNA-only moscot epsilon/tau grid search heatmap.
   │
   ├── OT_multi_rna.png
   │   # Multimodal method-family comparison evaluated in common RNA PCA space.
   │
   ├── OT_multi_atac.png
   │   # Multimodal method-family comparison evaluated in ATAC LSI space.
   │
   ├── moscot_RNA_only_all_KO_summary.csv
   │   # RNA-only OT summary table across perturbations and hyperparameter settings.
   │
   ├── multimodal_common_eval_summary.csv
   │   # Common-evaluation summary table for multimodal OT method families.
   │
   └── selected_method_family_configs.csv
       # One selected configuration per OT method family.
```

Optional QC outputs are stored under:

```text
results/ot/qc/
   ├── FGW_RNA_geometry_gene_activity_old_vs_common_eval_sanity.csv
   │   # Diagnostic table decomposing differences between moscot translated outputs
   │   # and explicit plan-barycenter evaluation.
   │
   └── FGW_RNA_geometry_gene_activity_old_vs_common_eval_sanity_heatmap.png
       # Heatmap visualization of the same sanity check.
```

### OT parameters

The main RNA-only OT grid varies:

```text
epsilon: entropic regularization strength
tau_a: source marginal relaxation parameter
tau_b: target marginal relaxation parameter
```

Balanced OT corresponds to:

```text
tau_a = tau_b = 1
```

The multimodal OT notebook also varies or defines:

```text
w_RNA: weight applied to RNA PCA features in RNA+ATAC concatenation
w_ATAC: weight applied to ATAC LSI features in RNA+ATAC concatenation
alpha: FGW tradeoff between direct feature matching and geometry matching
```

The main OT evaluation metric is the MMD² ratio:

```text
MMD²(mapped NTC, perturbed) / MMD²(original NTC, perturbed)
```

Values below 1 indicate that the OT plan moved the NTC distribution closer to the observed perturbed distribution. In the final evaluation, all OT plans are evaluated using the saved transport plan directly. The plan is row-normalized and applied to target-cell RNA PCA coordinates to compute a barycentric projection.

### Reproducing the OT results

The intended run order is:

```text
1. Open and run notebooks/moscot_RNA_only_end_to_end_grid_handoff_COMMON_EVAL.ipynb in Google Colab.

   This notebook:
   - fits RNA-only OT plans across epsilon/tau settings
   - evaluates plans in common RNA PCA space
   - generates the RNA-only heatmap
   - writes the FINAL_MODEL_HANDOFF_minimal/ folder used by the CVAE pipeline

2. Open and run notebooks/moscot_multimodal_FINAL_DEFENSIBLE_COMMON_EVAL_WITH_SANITY_CHECKS_v3_RNA_MATCHED.ipynb in Google Colab.

   This notebook:
   - compares RNA-only, ATAC-only, RNA+ATAC concat, GW, and FGW method families
   - evaluates all method families in common RNA PCA space
   - evaluates the same method families in ATAC LSI space
   - generates the multimodal OT report figures
   - saves sanity-check outputs for comparing moscot translated outputs against explicit plan-barycenter evaluation
```

The submitted notebooks should already contain the generated figures/results in their output cells, so it is not necessary to rerun the full OT pipeline just to inspect the final report figures.

### OT dependencies

The OT notebooks require a Python environment with:

```text
python >= 3.10
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
scanpy
anndata
torch
moscot
ott-jax
jax
```

The OT notebooks were run in Google Colab with Google Drive mounted. GPU is not required for the OT analysis, but some moscot / JAX operations may run faster with hardware acceleration.


## CVAE for KO RNA prediction Portion of the Project

A conditional VAE that predicts post-knockout RNA expression (HVG-log1p) from a
cell's ATAC profile and a gene-identity embedding for the KO target. Trained
and evaluated on a 9-KO + NTC dataset (10 conditions).

The final model: residual targeting (predict δ from NTC
mean)

## Layout

```text
model/
   └── cvae.py                # CVAE_v2 (encoder/decoder), free-bits KL,
                              # delta->HVG projection helpers

scripts/
   ├── 01_generate_genept.py  # GenePT embeddings for the 9 KO genes (OpenAI)
   ├── 02_preprocess.py       # RNA -> HVG/PCA, ATAC -> TF-IDF/LSI, GenePT -> 32-D, writes processed/cells.npz
   ├── 03_train.py            # CVAE training (variants + ablations)
   ├── 04_run_leave2out.sh    # leave-2-out CV across all C(9,2)=36 KO pairs
   ├── 05_evaluate.py         # metrics for every checkpoint + 3 baselines
   ├── 06_plot_main_panels.py # 2x3 metric panels per split
   ├── 07_per_ko_heatmap.py   # per-KO heatmaps
   └── 08_distance_split.py   # NTC distance-split robustness experiment

processed/
   ├── cells.npz              # aligned per-cell arrays (ATAC_LSI, RNA_PCA, RNA_HVG_log1p, label_int, gene embeddings)
   └── projectors/            # PCA loadings, TF-IDF/SVD, HVG list, GenePT/Geneformer gene-embedding tables

models/                       # trained CVAE checkpoints (.pt) + history
results/                      # metrics_v2.csv + figures (.png)
FINAL_MODEL_HANDOFF_minimal/  # OT/moscot baseline outputs (separate)
```

## Data

Cells: 9 epigenetic-regulator KOs (ACTL6A, DMAP1, EP400, EZH2, SMARCA4,
SMARCB1, SMARCE1, SUZ12, YY1) plus NTC controls. Multimodal RNA + ATAC measured
on the same cells.

## Running the pipeline

```bash
# 0. one-time: GenePT embeddings (needs OPENAI_API_KEY)
python scripts/01_generate_genept.py

# 1. one-time: build processed/cells.npz + projectors
python scripts/02_preprocess.py

# 2a. train final model on all KOs (3-way 80/10/10 split)
python scripts/03_train.py

# 2b. or hold out a pair and run leave-2-out CV (36 pairs)
bash scripts/04_run_leave2out.sh                 # default: only_genept atac
bash scripts/04_run_leave2out.sh only_genept rna # RNA-input variant

# 3. metrics for every checkpoint in models/ + linear/mean/NTC baselines
python scripts/05_evaluate.py

# 4. figures
python scripts/06_plot_main_panels.py
python scripts/07_per_ko_heatmap.py
python scripts/08_distance_split.py              # separate distance-split experiment
```

---
