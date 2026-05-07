# Predicting Perturbation-Induced Gene Expression from Baseline Chromatin Accessibility Using OT and Multimodal Generative Modeling

##  CVAE for KO RNA prediction Portion of the Project

A conditional VAE that predicts post-knockout RNA expression (HVG-log1p) from a
cell's ATAC profile and a gene-identity embedding for the KO target. Trained
and evaluated on a 9-KO + NTC dataset (10 conditions).

The final model: residual targeting (predict δ from NTC
mean) 

## Layout
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

                              
models/                    # trained CVAE checkpoints (.pt) + history

results/                   # metrics_v2.csv + figures (.png)

FINAL_MODEL_HANDOFF_minimal/  # OT/moscot baseline outputs (separate)


## Data

Cells: 9 epigenetic-regulator KOs (ACTL6A, DMAP1, EP400, EZH2, SMARCA4,
SMARCB1, SMARCE1, SUZ12, YY1) plus NTC controls. Multimodal RNA + ATAC measured
on the same cells;


## Running the pipeline

```bash
# 0. one-time: GenePT embeddings (needs OPENAI_API_KEY)
python scripts/01_generate_genept.py

# 1. one-time: build processed/cells.npz + projectors
python scripts/02_preprocess.py

# 2a. train final model on all KOs (3-way 80/10/10 split)
python scripts/03_train.py 

# 2b. or hold out a pair and run leave-2-out CV (36 pairs)
bash   scripts/04_run_leave2out.sh           # default: only_genept atac
bash   scripts/04_run_leave2out.sh only_genept rna   # RNA-input variant

# 3. metrics for every checkpoint in models/ + linear/mean/NTC baselines
python scripts/05_evaluate.py

# 4. figures
python scripts/06_plot_main_panels.py
python scripts/07_per_ko_heatmap.py
python scripts/08_distance_split.py          # separate distance-split experiment

---

## Optimal transport / moscot analysis

The optimal transport (OT) portion of the project is contained in two Colab notebooks under `notebooks/`.

```text
notebooks/

   ├── moscot_RNA_only_end_to_end_grid_handoff_COMMON_EVAL.ipynb
   │   # RNA-only moscot grid search.
   │   # Fits OT plans from NTC cells to each KO population across epsilon/tau settings.
   │   # Uses common RNA PCA evaluation based on the saved transport plan barycenter.
   │   # Produces the RNA-only hyperparameter heatmap and the final OT handoff used by the CVAE.

   └── moscot_multimodal_FINAL_DEFENSIBLE_COMMON_EVAL_WITH_SANITY_CHECKS_v3_RNA_MATCHED.ipynb
       # Multimodal OT comparison.
       # Compares RNA-only, ATAC LSI-only, RNA PCA + ATAC LSI concatenation,
       # GW with ATAC geometry, and FGW variants using ATAC/RNA geometry and gene activity.
       # Re-evaluates saved OT plans in common RNA PCA space using barycentric projection.
       # Also includes ATAC LSI evaluation and sanity checks comparing moscot translated
       # outputs against direct plan-barycenter evaluation.
