# Data Availability

This project uses cycle-resolved lithium-ion battery degradation trajectories generated with PyBaMM using a DFN/P2D electrochemical model for NMC811 cells. Charging rates are varied while the discharge rate is fixed at 1C.

## Configured Simulation Data

The dataset registry is defined in `configs/datasets.yaml`.

| Dataset key | Charging rate | Discharge rate | Cycles | Default path |
| --- | ---: | ---: | ---: | --- |
| `data1` | 1.00C | 1.00C | 5000 | `data/raw/time_data_C_1.0.csv` |
| `data2` | 0.75C | 1.00C | 5000 | `data/raw/time_data_C_0.75.csv` |
| `data3` | 0.50C | 1.00C | 5000 | `data/raw/time_data_C_0.5.csv` |
| `data4` | 0.30C | 1.00C | 5000 | `data/raw/time_data_C_0.3.csv` |
| `data5` | 0.20C | 1.00C | 5000 | `data/raw/time_data_C_0.2.csv` |
| `data6` | 0.60C | 1.00C | 5000 | `data/raw/time_data_C_0.6.csv` |
| `data7` | 0.85C | 1.00C | 5000 | `data/raw/time_data_C_0.85.csv` |
| `data8` | 0.70C | 1.00C | 5000 | `data/raw/time_data_C_0.7.csv` |
| `data9` | 0.55C | 1.00C | 5000 | `data/raw/time_data_C_0.55.csv` |

The VAE is trained and evaluated on the known protocols. The 0.70C and 0.55C datasets are configured as GP interpolation/validation targets.

## Minimum Dataset for Reproduction

The raw PyBaMM CSV files needed to reproduce the reported workflow are available from the data archive:

- Data archive: `https://doi.org/10.5281/zenodo.22031676`
- Data license: `Creative Commons Attribution 4.0 International (CC BY 4.0)`

After downloading the archive, place the CSV files in the repository under `data/raw/` using the filenames listed in the table above. The default paths in `configs/datasets.yaml` assume this directory layout.

## Generated Artifacts

The following outputs can be regenerated from the raw data and code:

- `artifacts/saved_models/`: trained VAE checkpoints.
- `artifacts/latent_space_data*/`: VAE latent coordinates per charging rate.
- `artifacts/gp_outputs/`: GP models, summaries, predictions, histories, and diagnostics.
- `artifacts/gp_interpolation/`: GP-inferred latent trajectories at unseen C-rates.
- `artifacts/vae_interpolation/`: decoded voltage/SOH predictions from GP latents.
- `artifacts/visualizations*/`: generated analysis and diagnostic figures.
