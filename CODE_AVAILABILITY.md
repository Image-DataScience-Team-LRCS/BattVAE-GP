# Code Availability

This repository contains the custom code used for the BattVAE-GP workflow described in:

Raghvender Raghvender, Mahdi Abid, Ferran Brosa Planella, Charles Delacourt, and Arnaud Demortiere. **BattVAE-GP: Generative Modeling of Long-Horizon Battery Degradation with Uncertainty Quantification.** arXiv preprint arXiv:2607.11943, 2026. https://arxiv.org/abs/2607.11943

## Repository Access

The code used for the reported workflow is available from the public repository and archived release:

- Public repository: `https://github.com/Image-DataScience-Team-LRCS/BattVAE-GP`
- Archived release: `https://doi.org/10.5281/zenodo.22031676`
- License: MIT License, see `LICENSE`

The archived release provides a persistent snapshot of the code associated with the manuscript.

## Reproducibility Scope

The code supports the full computational workflow:

1. Loading PyBaMM DFN/P2D NMC811 cycling simulations.
2. Capacity-grid preprocessing and multichannel feature construction.
3. Degradation-aware VAE training and inference.
4. Latent-space extraction for each charging protocol.
5. Sparse GP training with protocol-level holdout validation.
6. GP interpolation at unseen charging rates.
7. Frozen-decoder reconstruction and SOH prediction from GP-predicted latent states.

## Versioning

The public repository may continue to receive updates after publication. The archived release should be used when reproducing the manuscript results.
