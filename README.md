# Predicting Perturbation-Induced Gene Expression from Baseline Chromatin Accessibility Using OT and Multimodal Generative Modeling

## Optimal transport / moscot analysis

The optimal transport (OT) portion of the project is contained in two Google Colab notebooks under `notebooks/`. These notebooks were developed and run in Colab, and the submitted `.ipynb` files should still contain the executed outputs, figures, and summary tables used for the report. If rerunning them, update the Google Drive / repository paths near the top of each notebook to match your local setup.

For a quick test run, we recommend starting with `moscot_RNA_only.ipynb` because it is much smaller and faster than the full multimodal notebook. The grid parameters are near the top of the notebook and can be reduced further for debugging.

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

The input files to the OT are too large for github so they are stored in Google Drive at the following urls. Note these input files are post-Mixscape filtering. Please be sure to save the post-mixscape files and update path names at the top of OT notebooks.
RNA data: https://drive.google.com/file/d/1-SdwjiF4emCchxUUxcejLPvBZ7ZV_Xfo/view?usp=sharing
ATAC data: https://drive.google.com/file/d/1-_d2k-2VgRnzwe63HcuptVht8NlKLbC4/view?usp=sharing

The final analysis uses 1,144 NTC cells and 1,521 high-confidence knockout cells across nine perturbations after Mixscape filtering:

```text
ACTL6A, DMAP1, EP400, EZH2, SMARCA4, SMARCB1, SMARCE1, SUZ12, YY1
```

The multimodal OT notebook also constructs an ATAC gene activity representation for the FGW experiments. This is generated inside `moscot_multimodal.ipynb` using `gencode.v49.annotation.gtf`. Peaks are overlapped with gene body plus promoter windows, using 2 kb upstream and 500 bp downstream of the TSS, and accessibility is aggregated to gene-level features before normalization and PCA.

The gencode file is too large to upload to github so please access it at the following google drive link and be sure to update path names to it in the colab notebook: https://drive.google.com/file/d/1lGtMte0fOYuy7SNlDpc94e34mYClS6hD/view?usp=sharing

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
   ├── common_space_reevaluation_all_methods.csv
   │   # Common-evaluation summary table for all multimodal OT method-family runs.
   │
   ├── Figure1_COMMON_RNA_PCA_eval_single_best_config_per_method_family_MMD2_ratio_concat_ATACw_ge_0.25_with_RNA_only_selected_configs.csv
   │   # Selected configuration table for the main RNA PCA multimodal heatmap.
   │
   ├── Figure1_COMMON_RNA_PCA_eval_single_best_config_per_method_family_MMD2_ratio_concat_ATACw_ge_0.25_with_RNA_only_selected_KO_level_rows.csv
   │   # KO-level rows used in the main RNA PCA multimodal heatmap.
   │
   └── Figure1_COMMON_RNA_PCA_eval_single_best_config_per_method_family_MMD2_ratio_concat_ATACw_ge_0.25_with_RNA_only_heatmap_matrix.csv
       # Matrix corresponding to the main RNA PCA multimodal heatmap.
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
1. Open and run `notebooks/moscot_RNA_only.ipynb` in Google Colab.

   This notebook:
   - fits RNA-only OT plans across epsilon/tau settings
   - evaluates plans in common RNA PCA space
   - generates the RNA-only heatmap
   - writes the `FINAL_MODEL_HANDOFF_minimal/` folder used by the CVAE pipeline

2. Open and run `notebooks/moscot_multimodal.ipynb` in Google Colab.

   This notebook:
   - compares RNA-only, ATAC-only, RNA+ATAC concat, GW, and FGW method families
   - constructs ATAC gene activity features using the GENCODE annotation file
   - evaluates all method families in common RNA PCA space
   - evaluates the same method families in ATAC LSI space
   - generates the multimodal OT report figures
   - saves sanity-check outputs comparing moscot translated outputs against explicit plan-barycenter evaluation
```

The submitted notebooks should already contain the generated figures/results in their output cells, so it is not necessary to rerun the full OT pipeline just to inspect the final report figures. If there is a desire to run the notebooks themselves, please note that the RNA only notebook is far smaller and quicker to run so we would recommend beginning there. 

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
pyranges
```

The OT notebooks were run in Google Colab with Google Drive mounted. GPU is not required for the OT analysis, but some moscot / JAX operations may run faster with hardware acceleration.

Please note that specific packages need specific versions which are defined at the top of each notebook or are included below:

```text
!pip install -q \
  "jax[cpu]==0.7.2" \
  "jaxlib==0.7.2" \
  "ott-jax==0.6.0" \
  "moscot==0.5.0"
```

## CVAE for KO RNA Prediction

A conditional variational autoencoder (CVAE) that predicts post-knockout RNA
expression (HVG log1p-normalized) from a cell's ATAC chromatin accessibility
profile and a language-model gene identity embedding for the knockout target.
Trained and evaluated on a 9-KO + NTC dataset (10 conditions total).

The final model uses **residual targeting**: the decoder predicts the delta (δ)
from the NTC mean rather than absolute expression, which substantially improves
the signal-to-noise ratio of the training target.

### File descriptions

```text
model/
   └── cvae.py
       # CVAE_v2 architecture definition plus two helper functions.
       # Architecture: [ATAC_LSI(50); gene_emb(32)] -> Linear(82->64) + GELU
       #               -> mu head Linear(64->64), logvar head Linear(64->64)
       #               -> z (64-D) -> decoder Linear(64->50) -> delta_PCA (50)
       #               -> project_delta_to_hvg: delta_PCA @ V_delta.T -> (2000,)
       #               -> add NTC mean to recover absolute HVG expression.
       # Total learnable parameters: ~13.4k.
       # free_bits_kl(): KL divergence with a per-dimension free-bits floor
       #                 (prevents posterior collapse).

scripts/
   ├── 01_generate_genept.py
   │   # Calls the OpenAI text-embedding-3-large API to generate 1536-D embeddings
   │   # for each of the 9 KO gene names using the prompt
   │   # "The gene {X} is a human protein-coding gene." and L2-normalizes each vector.
   │   # Reads OPENAI_API_KEY from the environment.
   │   # Output: processed/projectors/genept_embeddings.csv
   │
   ├── 02_preprocess.py
   │   # One-shot preprocessing. Reads the two post-Mixscape h5ad files and:
   │   #   RNA: normalize_total(1e4) -> log1p -> top 2000 HVG -> PCA(50)
   │   #        saves pca_loadings_V.npy, pca_gene_means.npy, hvg_list.txt
   │   #   ATAC: TF-IDF (sublinear_tf, L2 norm) -> TruncatedSVD(51)
   │   #         -> drop component 1 (depth-correlated) -> 50-D LSI
   │   #         saves tfidf.pkl, svd.pkl
   │   #   GenePT: loads genept_embeddings.csv, stacks 9 KO rows + NTC=zero,
   │   #           PCA-projects to 32-D, forces NTC row back to zero, saves gene_emb_genept.npy
   │   # Output: processed/cells.npz (all arrays aligned by cell barcode)
   │
   ├── 03_train.py
   │   # Main CVAE training script. Handles all ablation variants and leave-K-out holdouts.
   │   # Performs 3-way stratified (80/10/10) split by guide_target.
   │   # Saves a .pt checkpoint + _history.json per run.
   │
   ├── 04_run_leave2out.sh
   │   # Loops over all C(9,2)=36 KO pairs and calls 03_train.py for each,
   │   # passing --holdout A B to hold out the two KOs from training.
   │
   ├── 05_evaluate.py
   │   # Loads every cvae_*.pt checkpoint plus fits 3 baselines (linear matrix
   │   # factorization, mean perturbation, NTC identity) and computes a full
   │   # suite of metrics across all evaluation splits.
   │   # Output: results/metrics_v2.csv
   │
   ├── 06_plot_main_panels.py
   │   # Reads metrics_v2.csv and produces four 2×3 panel figures (one per split):
   │   # Δ-PCC top-20 DE | per-cell Δ-PCC top-20 | Expression-PCC
   │   # Cosine similarity (delta) | MSE top-20 DE | Δ-PCC all 2000 HVGs
   │   # Output: results/fig_panels_{heldout,test,val_test,all}.png
   │
   ├── 07_per_ko_heatmap.py
   │   # Produces per-KO heatmaps of Δ-PCC and related metrics across model variants.
   │   # Output: results/fig_per_ko_heatmap_{heldout,test,val_test,all}.png
   │
   └── 08_distance_split.py
       # NTC distance-split robustness experiment.
       # Splits NTC cells into closest 25%, middle 50%, farthest 25% to the
       # centroid of all perturbed cells (in 50-D HVG PCA space).
       # Trains and evaluates two regimes (far_train and close_train) across
       # three models: CVAE, mean_pert, linear_mf.
       # Output: results/fig_distance_umap.png
       #         results/fig_distance_split_box.png
       #         results/distance_split_metrics.csv

processed/
   ├── cells.npz
   │   # Aligned per-cell arrays (all indexed by the same cell-barcode order):
   │   #   barcode          : (N,)      cell barcode strings
   │   #   guide_target     : (N,)      KO label strings
   │   #   label_int        : (N,)      integer label index into ko_labels
   │   #   ATAC_LSI         : (N, 50)   50-D LSI features (encoder input, default)
   │   #   RNA_PCA          : (N, 50)   50-D RNA PCA features (alternative encoder input)
   │   #   RNA_HVG_log1p    : (N, 2000) log1p-normalized HVG expression (prediction target)
   │   #   gene_emb_genept  : (10, 32)  GenePT gene embedding table (one row per KO_LABELS entry)
   │   #   ko_labels        : (10,)     ordered condition names
   │   #                      ["NTC","ACTL6A","DMAP1","EP400","EZH2",
   │   #                       "SMARCA4","SMARCB1","SMARCE1","SUZ12","YY1"]
   │
   └── projectors/
       # genept_embeddings.csv     : raw 1536-D GenePT embeddings (output of script 01)
       # pca_loadings_V.npy        : (2000, 50) absolute-HVG PCA loadings
       # pca_gene_means.npy        : (2000,)    per-gene mean used by the HVG PCA
       # hvg_list.txt              : 2000 highly-variable gene names (one per line)
       # tfidf.pkl                 : fitted TfidfTransformer for ATAC
       # svd.pkl                   : fitted TruncatedSVD(51) for ATAC LSI
       # gene_emb_genept.npy       : (10, 32) projected GenePT embedding table
       # gene_emb_labels.txt       : condition names in KO_LABELS order

models/
   # cvae_{variant}.pt             : best-val checkpoint (full training set)
   # cvae_{variant}_loko_{A}_{B}.pt: leave-2-out checkpoint with KOs A and B held out
   # *_history.json                : per-epoch train/val MSE and KL history

results/
   ├── metrics_v2.csv
   ├── fig_panels_{heldout,test,val_test,all}.png
   ├── fig_per_ko_heatmap_{heldout,test,val_test,all}.png
   ├── fig_distance_umap.png
   ├── fig_distance_split_box.png
   ├── distance_split_metrics.csv
   └── ot/                           # OT figures (see OT section above)

FINAL_MODEL_HANDOFF_minimal/         # OT/moscot handoff used as CVAE input baseline
```

### CVAE input data

The same post-Mixscape h5ad files used by the OT notebooks serve as the CVAE
input. They are not included in this repository. Download them from Google Drive
and update the `RNA_H5` / `ATAC_H5` paths at the top of `02_preprocess.py`:

```text
RNA data  (040_epi_rna_filtered.h5ad):
  https://drive.google.com/file/d/1-SdwjiF4emCchxUUxcejLPvBZ7ZV_Xfo/view?usp=sharing

ATAC data (040_epi_atac_filtered.h5ad):
  https://drive.google.com/file/d/1-_d2k-2VgRnzwe63HcuptVht8NlKLbC4/view?usp=sharing
```

After preprocessing, the pipeline operates entirely from `processed/cells.npz`
and does not require the original h5ad files (except `05_evaluate.py`, which
re-reads the RNA h5ad once to augment the linear-MF baseline with any KO genes
that were not among the top 2000 HVGs).

### CVAE architecture parameters (`model/cvae.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atac_dim` | 50 | Dimensionality of the encoder input (ATAC LSI or RNA PCA) |
| `gene_emb_dim` | 32 | Dimensionality of the gene identity embedding |
| `hidden_dim` | 64 | Width of the encoder trunk linear layer |
| `latent_dim` | 64 | Dimensionality of the VAE latent space z |
| `pca_dim` | 50 | Dimensionality of the decoder output (delta in PCA space) |

`free_bits_kl(mu, logvar, tau=0.1)`: the free-bits floor `tau` is the minimum
KL contribution per latent dimension (in nats). Dimensions below the floor are
not penalized, which prevents the KL term from collapsing uninformative dimensions
to exactly zero and masking the reconstruction loss.

### Training parameters (`scripts/03_train.py`)

Two model variants are trained, differing only in the encoder input modality:

| Model name | `--input` flag | Encoder input |
|------------|---------------|---------------|
| CVAE (ATAC) | `atac` (default) | 50-D ATAC LSI features |
| CVAE (RNA)  | `rna`            | 50-D RNA PCA features  |

**Data and split flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--holdout KO1 [KO2 ...]` | `[]` | KO label(s) to exclude from training (for leave-K-out CV) |
| `--seed` | 0 | Random seed for split and training |
| `--val_frac` | 0.1 | Fraction of in-distribution cells used for validation |
| `--test_frac` | 0.1 | Fraction of in-distribution cells used for test |
| `--gene_emb` | `genept` | Gene embedding source: `genept` or `geneformer` |

**Loss and optimization flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--kl_weight` | 1.0 | Weight applied to the KL term |
| `--free_bits_tau` | 0.1 | Free-bits floor τ per latent dimension (nats) |
| `--mmd_weight` | 0.0 | Weight of the MMD² distribution-matching auxiliary loss |
| `--mean_align_weight` | 0.0 | Weight of the per-KO mean-alignment auxiliary loss |

Default optimizer settings: lr=1e-3, epochs=200, patience=20, weight_decay=1e-4, batch_size=64.

### Evaluation metrics (`scripts/05_evaluate.py`)

Metrics are computed for each (model, KO gene, split) combination:

| Metric | Description |
|--------|-------------|
| `pcc_all` | Pearson correlation of predicted vs. real mean δ across all 2000 HVGs |
| `pcc_DE{K}` | Pearson correlation restricted to top-K differentially expressed genes (K ∈ 20, 50, 100, 500, 1000) |
| `mse_all` | MSE on full 2000-D δ vector |
| `mse_DE{K}` | MSE restricted to top-K DE genes |
| `expr_pcc_all` | Pearson correlation of predicted vs. real mean absolute expression (all HVGs) |
| `expr_pcc_DE{K}` | Expression PCC restricted to top-K DE genes |
| `cos_sim_delta` | Cosine similarity between predicted and real δ vectors |
| `per_cell_pcc_all` | Average per-NTC-cell PCC of predicted δ vs. the real KO mean δ |
| `per_cell_pcc_DE{K}` | Per-cell PCC restricted to top-K DE genes |
| `mmd2_pred_vs_real_DE100` | MMD² between predicted and real KO cell clouds (top-100 DE genes) |
| `spread_pred` / `spread_real` | Mean per-gene std of predicted / real KO cell clouds |

Evaluation splits:

| Split | Cells used |
|-------|-----------|
| `test` | Test cells of that KO (clean held-out, 10%) |
| `val_test` | Validation + test cells (20%) |
| `all` | All cells of that KO (inflated; training cells visible) |
| `heldout` | Cells from a KO that was excluded from training entirely (leave-K-out only) |

### Reproducing the CVAE results

**Before running**: update the hardcoded `ROOT`, `RNA_H5`, and `ATAC_H5` path
variables at the top of each script to match your local directory structure.

```bash
# Step 0 — one-time: generate GenePT gene embeddings (requires OpenAI API key)
export OPENAI_API_KEY=sk-...
python scripts/01_generate_genept.py
# Output: processed/projectors/genept_embeddings.csv

# Step 1 — one-time: build the aligned processed/cells.npz and projector files
python scripts/02_preprocess.py
# Output: processed/cells.npz, processed/projectors/

# Step 2a — train CVAE (ATAC) and CVAE (RNA) on all 9 KOs (3-way 80/10/10 split)
python scripts/03_train.py --input atac   # CVAE (ATAC)
python scripts/03_train.py --input rna    # CVAE (RNA)

# Step 2b — run leave-2-out CV across all C(9,2)=36 KO pairs (~6 min on CPU)
bash scripts/04_run_leave2out.sh          # CVAE (ATAC)
bash scripts/04_run_leave2out.sh CVAE genept rna  # CVAE (RNA)

# Step 3 — compute metrics for every checkpoint + 3 baselines
python scripts/05_evaluate.py
# Output: results/metrics_v2.csv

# Step 4 — generate all figures
python scripts/06_plot_main_panels.py
# Output: results/fig_panels_{heldout,test,val_test,all}.png

python scripts/07_per_ko_heatmap.py
# Output: results/fig_per_ko_heatmap_{heldout,test,val_test,all}.png

python scripts/08_distance_split.py
# Output: results/fig_distance_umap.png, fig_distance_split_box.png,
#         results/distance_split_metrics.csv
```

For a quick test run of just the model training:

```bash
python scripts/03_train.py --input atac
# Trains in under 2 minutes on CPU; the submitted checkpoint already contains
# the full trained model so retraining is not required to inspect results.
```

The submitted `results/` directory already contains the generated figures and
`metrics_v2.csv`, and the submitted `models/` directory already contains trained
checkpoints, so it is not necessary to rerun the full pipeline to inspect the
final report outputs.

### CVAE system requirements and dependencies

**Hardware:** CPU is sufficient for all steps. Steps 0–2 and step 5 are more
memory-intensive (ATAC SVD on 524k peaks; step 5 reloads the full RNA h5ad).
A node with ≥32 GB RAM is recommended. GPU is optional; training on CPU takes
under 2 minutes per run for the default hyperparameters.

**Python version:** 3.9 or higher.

**Required packages:**

```text
numpy
pandas
scipy
scikit-learn
matplotlib
scanpy
anndata
torch
openai          # step 0 only (GenePT embeddings)
umap-learn      # step 4 only (08_distance_split.py UMAP figure)
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib scanpy anndata torch openai umap-learn
```

The CVAE scripts were developed and run on a Linux HPC cluster (Python 3.10,
PyTorch 2.x). No special CUDA installation is required.

---
