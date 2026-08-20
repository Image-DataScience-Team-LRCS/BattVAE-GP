# Configuration Files

The project is controlled through YAML configuration files so experiments can be reproduced without editing source code.

## `vae.yaml`

Defines VAE training, inference, architecture, and normalization behavior.

Important sections:

- `GENERAL`: run mode, experiment name, random seed, training datasets, validation datasets.
- `HYPER_PARAMETERS`: sequence length, channel count, latent dimension, transformer dimensions, learning-rate schedule, KL annealing, SOH loss factor, and decoder baseline-freezing epoch.
- `NORMALIZATION`: voltage, derivative, inverse-derivative, and hysteresis normalization rules.

## `gp.yaml`

Defines GP training and interpolation behavior.

Important sections:

- `GENERAL`: GP experiment name, holdout datasets, random seed.
- `FEATURES`: input columns `Cycle` and `c_rate`, target columns `z1` and `z2`.
- `MODEL`: sparse GP type, kernel, lengthscale, variance, noise variance, inducing points, and jitter.
- `TRAINING`: epochs, batch size, optimizer settings, scheduler, and early stopping.
- `INTERPOLATION`: unseen C-rate datasets and latent interpolation settings.

## `datasets.yaml`

Registers raw PyBaMM simulation CSV files and latent-space CSV files.

- `VAE_DATASETS`: raw simulation datasets used by VAE training and inference.
- `GP_DATASETS`: per-C-rate latent CSVs used by GP training.
- `GP_INTERPOLATION_DATASETS`: unseen C-rate datasets and target latent output paths.

## `paths.yaml`

Centralizes output directories for visualizations, metrics, checkpoints, latent spaces, GP outputs, and decoded interpolation outputs.
