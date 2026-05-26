# Bridging the microstructural gap in human connectomics using HiP-CT as a reference for diffusion MRI

Code repository for the paper:

> **Bridging the microstructural gap in human connectomics using hierarchical phase-contrast tomography as a reference for diffusion MRI in the human brain**
>
> Eric Wanjau, Matthieu Chourrout, Chiara Maffei, Yael Balbastre, Andrew Keenlyside, Joseph Brunet, Aikta Sharma, Susie Y. Huang, Paul Tafforeau, Bruce Fischl, Anastasia Yendiki, Peter D. Lee, Claire L. Walsh
>
> *Nature Communications* (submitted) | [Preprint on bioRxiv](https://www.biorxiv.org/content/10.64898/2026.04.02.715729v1)

## Overview

This repository contains the analysis code for validating diffusion MRI (dMRI) fiber orientation estimates using Hierarchical Phase-Contrast Tomography (HiP-CT) as a ground truth microscopic reference. The pipeline applies Structure Tensor Analysis (STA) to HiP-CT data to compute fiber Orientation Distribution Functions (fODFs) and perform tractography analogous to dMRI, enabling direct cross-modal comparison.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `scripts/01_zarr_conversion.py` | Convert raw HiP-CT tomographic data to chunked Zarr format |
| 2 | `scripts/02_structure_tensor.py` | 3D structure tensor analysis (Gaussian derivatives, eigendecomposition) |
| 3 | `scripts/03_fodf_estimation.py` | Aggregate eigenvectors into supervoxel fODFs via spherical harmonics |
| 4 | `scripts/04_registration.py` | Cross-modal dMRI-to-HiP-CT registration (ANTsPy) |
| 5 | `scripts/05_tractography.py` | Probabilistic tractography from fODFs |
| 6 | `scripts/06_vessel_masking.py` | Vascular signal suppression (pre-STA and gradient-level masking) |
| 7 | `scripts/07_quantitative_comparison.py` | Angular correlation coefficient, peak analysis, FA/GFA metrics |

## Setup

Requires Python >= 3.11. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Key dependencies

- [structure-tensor](https://github.com/Skielex/structure-tensor) — 3D structure tensor computation
- [DIPY](https://dipy.org/) — spherical harmonics, fODF tools, tractography
- [ANTsPy](https://github.com/ANTsX/ANTsPy) — cross-modal image registration
- [Zarr](https://zarr.dev/) — chunked array storage for large volumes
- [Dask](https://dask.org/) — parallel out-of-core processing

## License

See [LICENSE](LICENSE).
