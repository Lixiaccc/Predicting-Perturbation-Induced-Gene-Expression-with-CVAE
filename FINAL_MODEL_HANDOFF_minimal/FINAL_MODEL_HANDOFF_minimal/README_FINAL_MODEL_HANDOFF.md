
# Final minimal model handoff

This folder contains the clean RNA-only moscot outputs for downstream modeling.

## Most important file per KO

- `X_mapped_barycentric_RNA_PCA.npy`
  - This is the barycentric OT projection: each control/source cell mapped toward the KO/perturbed distribution.
  - Row order matches `X_source_control_RNA_PCA.npy`.

## Files per KO

- `X_source_control_RNA_PCA.npy`
  - Control/source cells in the processed RNA PCA feature space used by moscot.

- `X_mapped_barycentric_RNA_PCA.npy`
  - Main downstream target. Same number/order of rows as the source/control file.

- `X_target_perturbed_RNA_PCA.npy`
  - Real perturbed cells in the same processed RNA PCA feature space. Useful for evaluation/comparison.

- `obs_source_control.csv`, `obs_mapped_barycentric.csv`, `obs_target_perturbed.csv`
  - Metadata. Row order matches the corresponding `.npy` file.

- `transport_plan_optional.npz`
  - Optional low-level OT transport matrix. Not needed for most ML workflows.

- `metrics.csv`
  - Alignment metrics for this KO under the selected global moscot config.

## Important caveat

These arrays are in the shared RNA PCA space used by moscot, not raw gene expression.
The mapping is symmetric finite-sample OT/barycentric projection, not a learned neural perturbation model.

## Selected global moscot config

- epsilon: 0.001
- tau_a: 1.0
- tau_b: 1.0
