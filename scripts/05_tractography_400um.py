#!/usr/bin/env python
"""Probabilistic and deterministic fiber tractography from SH coefficients (400um).

Loads spherical harmonic (SH) coefficient volumes produced by
03_fodf_estimation.py at 400um resolution, constructs stopping criteria
from GFA and white-matter masks, generates seed points, runs DIPY
tractography in seed-chunks, and saves raw streamlines as .npy.

This is the 400um variant of 05_tractography.py. The pipeline logic is
identical; only the configuration section differs.
"""

import gc
import os
import sys
from pathlib import Path
from time import perf_counter

import cv2
import nibabel as nib
import numpy as np
import zarr
from fiberorient.util import make_sphere
from natsort import natsorted
from nibabel.streamlines import ArraySequence
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import filters, morphology
from skimage.measure import block_reduce
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.tracker import deterministic_tracking, probabilistic_tracking
from dipy.tracking.streamline import Streamlines

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.analysis import get_voxel_ratio, get_analysis_shape, compute_gfa_sh
# from utils.visualization import show_tracts_img_2d, line_colors


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_tracking_lengths(resolution_mm, step_size_ratio,
                         min_dist_mm=3.2, max_dist_mm=300.0):
    """Compute min/max streamline lengths in points from physical constraints.

    Parameters
    ----------
    resolution_mm : float
        Isotropic voxel size in mm.
    step_size_ratio : float
        Step size as a fraction of the voxel size.
    min_dist_mm, max_dist_mm : float
        Physical distance bounds in mm.

    Returns
    -------
    min_len, max_len : int
        Number of tracking steps.
    """
    physical_step_mm = resolution_mm * step_size_ratio
    if physical_step_mm == 0:
        raise ValueError("Step size results in 0mm progression.")

    min_len = int(np.ceil(min_dist_mm / physical_step_mm))
    max_len = int(np.floor(max_dist_mm / physical_step_mm))

    print(f"Res: {resolution_mm}mm | Step: {physical_step_mm:.4f}mm | "
          f"Lengths: {min_len}–{max_len} points ({min_dist_mm}–{max_dist_mm}mm)")
    return min_len, max_len


def sample_seeds(seeds, percentage_to_keep=0.10, deterministic=True, seed=42):
    """Sample a subset of seed points.

    Parameters
    ----------
    seeds : ndarray, shape (N, 3)
    percentage_to_keep : float
        Fraction in (0, 1].
    deterministic : bool
        If True, use uniform striding; otherwise random sampling.
    seed : int
        Random seed (only used when deterministic=False).

    Returns
    -------
    sampled_seeds : ndarray, shape (M, 3)
    """
    assert 0 < percentage_to_keep <= 1.0
    size = int(seeds.shape[0] * percentage_to_keep)
    print(f"Sampling {size} seeds from {seeds.shape[0]} ({percentage_to_keep * 100:.2f}%)")

    if deterministic:
        step = max(1, seeds.shape[0] // size)
        indices = np.arange(0, seeds.shape[0], step=step)
        sampled = seeds[indices]
        sampled = np.vstack([sampled, seeds[-1, :]])
    else:
        np.random.seed(seed)
        indices = np.random.choice(seeds.shape[0], size=size, replace=False)
        sampled = seeds[indices]
    return sampled


def ensure_c_contig(arr, dtype=np.float32, name=None):
    """Ensure array is a numeric, C-contiguous numpy array."""
    a = np.asarray(arr)
    if a.dtype == object:
        raise ValueError(f"{name or 'Array'} has dtype=object")
    return np.ascontiguousarray(a.astype(dtype, copy=False))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration (400um — differs from 800um version) ----
    ODF_ANALYSIS_DIR = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis")
    HIPCT_LEVEL4_PATH = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/registration_new_dmri/hipct_hemi_downsampled_level_4_masked_norm_percentile_83_99.nii.gz")
    WM_MASK_DIR = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/registration_new_dmri/hipct_hemi_downsampled_level_4_jp2_masked_slices_masks")

    HIPCT_RESOLUTION = 15.13   # um
    DMRI_RESOLUTION = 400.0    # um  (800.0 in 800um version)

    DEGREE = 8
    N_BINS = 6500

    START_IDX = 0
    END_IDX = 9880

    GFA_THRESHOLD = 0.8
    Z_SAMPLER = 16
    DESIRED_STREAMLINES = 8e6  # (8e6 in 800um version)
    N_SEEDS_PER_CHUNK = 100000
    PERFORM_DETERMINISTIC = False

    # ---- Derived parameters ----
    voxel_ratio = get_voxel_ratio(HIPCT_RESOLUTION, DMRI_RESOLUTION)
    n_coefs = int(((DEGREE + 1) * (DEGREE + 2)) // 2)
    sphere = make_sphere(N_BINS)

    TRACKING_KWARGS = dict(
        random_seed=1,
        sphere=sphere,
        max_angle=45,
        step_size=0.2,
        min_len=125,   # (20 in 800um version)
        max_len=4000,  # (2000 in 800um version)
        seed_buffer_fraction=0.5,
        return_all=False,
    )

    assert ODF_ANALYSIS_DIR.exists(), f"Analysis dir not found: {ODF_ANALYSIS_DIR}"

    # ---- Load SH coefficients (400um-specific zarr name) ----
    sh_coef_zarr_path = ODF_ANALYSIS_DIR / f"sh_coefficients_degree_{DEGREE}_{DMRI_RESOLUTION}um_xyz.zarr"
    assert sh_coef_zarr_path.exists()
    sh_coeffs = zarr.open(sh_coef_zarr_path, mode='r')
    print(f"SH coefficients zarr shape: {sh_coeffs.shape}")

    # Compute step indices
    seed_mask_zarr_path = ODF_ANALYSIS_DIR / "seed_mask_xyz.zarr"
    assert seed_mask_zarr_path.exists()
    seed_mask = zarr.open(seed_mask_zarr_path, mode='r')
    print(f"Seed mask shape: {seed_mask.shape}")

    seed_mask_steps = np.arange(0, seed_mask.shape[-1], voxel_ratio)[:sh_coeffs.shape[2]]
    start_step_idx = seed_mask_steps.tolist().index(START_IDX)
    end_step_idx = seed_mask_steps.tolist().index(END_IDX)
    print(f"Step indices: start={start_step_idx}, end={end_step_idx}")

    seed_mask_steps_subset = seed_mask_steps[start_step_idx:end_step_idx + 1].tolist()
    assert seed_mask_steps_subset[0] == START_IDX
    assert seed_mask_steps_subset[-1] == END_IDX

    # ---- Load seed mask subset (with z subsampling) ----
    sampling_in_z = np.arange(seed_mask_steps_subset[0], seed_mask_steps_subset[-1], Z_SAMPLER)
    sampling_in_z = np.append(sampling_in_z, seed_mask_steps_subset[-1])
    print(f"Z sampling points: {sampling_in_z.shape[0]}")

    seed_mask_subset = seed_mask[:, :, sampling_in_z]
    print(f"Seed mask subset shape: {seed_mask_subset.shape}")

    # ---- Load SH coefficients subset ----
    sh_coeffs_subset = sh_coeffs[:, :, start_step_idx:end_step_idx, :]
    sh_coeffs_subset = np.nan_to_num(sh_coeffs_subset, nan=0.0)
    print(f"SH coefficients subset shape: {sh_coeffs_subset.shape}")

    # ---- Compute GFA ----
    gfa = compute_gfa_sh(sh_coeffs_subset, sh0_index=0)
    gfa = np.nan_to_num(gfa, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"GFA shape: {gfa.shape}")

    # ---- Load stopping criterion cleaning mask ----
    sc_mask_zarr_path = ODF_ANALYSIS_DIR / "15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_stopping_criteria_cleaning_mask.zarr"
    assert sc_mask_zarr_path.exists()
    sc_mask = zarr.open(sc_mask_zarr_path, mode='r')
    sc_mask_subset = sc_mask[seed_mask_steps_subset].transpose(2, 1, 0)
    print(f"SC cleaning mask subset shape: {sc_mask_subset.shape}")

    sc_mask_subset = (sc_mask_subset - sc_mask_subset.min()) / (sc_mask_subset.max() - sc_mask_subset.min())
    sc_mask_subset = np.clip(sc_mask_subset, 0.0, 1.0)

    sc_analysis_shape = get_analysis_shape(
        sc_mask_subset.shape[0], sc_mask_subset.shape[1], sc_mask_subset.shape[2], voxel_ratio,
    )
    sc_mask_subset = sc_mask_subset[:sc_analysis_shape[0], :sc_analysis_shape[1], :]
    sc_mask_subset = block_reduce(sc_mask_subset, block_size=(voxel_ratio, voxel_ratio, 1), func=np.mean)
    sc_mask_subset = (sc_mask_subset > 0.5).astype(np.uint8)
    sc_mask_subset = 1 - sc_mask_subset[..., :gfa.shape[2]]
    print(f"Processed SC mask shape: {sc_mask_subset.shape}")

    # ---- Load and prepare WM masks ----
    assert HIPCT_LEVEL4_PATH.exists()
    hipct_level4_nifti = nib.load(str(HIPCT_LEVEL4_PATH))
    hipct_level4_data = hipct_level4_nifti.get_fdata()
    print(f"HiP-CT level 4 shape: {hipct_level4_data.shape}")

    wm_mask_files = natsorted(list(WM_MASK_DIR.glob("*tiff")))
    wm_mask_vol = np.stack(
        [cv2.imread(str(p), -1).T for p in wm_mask_files], axis=-1,
    ).astype(bool)
    assert wm_mask_vol.shape == hipct_level4_data.shape
    print(f"WM mask shape: {wm_mask_vol.shape}")

    wm_mask_vol = ndimage.binary_fill_holes(wm_mask_vol)
    wm_mask_vol = ndimage.binary_dilation(wm_mask_vol, iterations=1)

    wm_gm_threshold = np.percentile(hipct_level4_data, 85)
    wm_gm_mask = hipct_level4_data >= wm_gm_threshold
    wm_mask = wm_mask_vol

    start_z4 = round(START_IDX / 2 ** 4)
    end_z4 = round(END_IDX / 2 ** 4)
    wm_gm_mask = wm_gm_mask[:, :, start_z4:end_z4]
    wm_mask = wm_mask[:, :, start_z4:end_z4]
    print(f"WM/GM mask cropped shape: {wm_gm_mask.shape}")

    scale_factors = (
        gfa.shape[0] / wm_mask.shape[0],
        gfa.shape[1] / wm_mask.shape[1],
        gfa.shape[2] / wm_mask.shape[2],
    )
    wm_gm_mask_binned = (zoom(wm_gm_mask.astype(np.uint8), scale_factors, order=0, prefilter=False) > 0.5).astype(bool)
    wm_mask_binned = (zoom(wm_mask.astype(np.uint8), scale_factors, order=0, prefilter=False) > 0.5).astype(bool)
    print(f"Downsampled WM mask shape: {wm_mask_binned.shape}")
    del wm_mask, wm_gm_mask, wm_mask_vol, hipct_level4_data

    # ---- Create composite stopping criterion ----
    gfa_masked = np.where(sc_mask_subset, gfa, 0)
    stopping_mask = ((gfa_masked >= GFA_THRESHOLD) & wm_gm_mask_binned) | wm_mask_binned

    footprint = morphology.ball(radius=1)
    stopping_mask = morphology.binary_opening(stopping_mask, footprint=footprint)
    stopping_mask = filters.gaussian(stopping_mask.astype(float), sigma=0.5)
    stopping_mask = (stopping_mask > 0.5).astype(np.uint8)
    print(f"Stopping criterion mask shape: {stopping_mask.shape}")

    stopping_criterion = BinaryStoppingCriterion(stopping_mask)

    # ---- Generate and sample seeds ----
    seeds = np.argwhere(seed_mask_subset > 0)
    print(f"Total seeds before sampling: {seeds.shape[0]}")

    percentage_to_keep = DESIRED_STREAMLINES / seeds.shape[0]
    sampled_seeds = sample_seeds(seeds, percentage_to_keep=percentage_to_keep, deterministic=True)
    print(f"Sampled seeds: {sampled_seeds.shape[0]}")

    # Scale z (from subsampled z indices) and convert to dMRI voxel space
    sampled_seeds[:, -1] = sampled_seeds[:, -1] * Z_SAMPLER
    sampled_seeds = sampled_seeds / voxel_ratio

    # ---- Run tractography in seed chunks ----
    affine = np.eye(4)
    streamlines_all = ArraySequence()
    script_start = perf_counter()

    tracking_fn = deterministic_tracking if PERFORM_DETERMINISTIC else probabilistic_tracking
    tracking_type = "deterministic" if PERFORM_DETERMINISTIC else "probabilistic"
    print(f"\nPerforming {tracking_type} tractography")

    for chunk_start in range(0, sampled_seeds.shape[0], N_SEEDS_PER_CHUNK):
        chunk_end = min(chunk_start + N_SEEDS_PER_CHUNK, sampled_seeds.shape[0])
        print(f"\nSeed chunk {chunk_start}–{chunk_end} / {sampled_seeds.shape[0]}")
        seeds_chunk = sampled_seeds[chunk_start:chunk_end]

        generator = tracking_fn(
            seed_positions=seeds_chunk,
            sc=stopping_criterion,
            affine=affine,
            sh=ensure_c_contig(sh_coeffs_subset),
            **TRACKING_KWARGS,
        )

        streamlines = Streamlines(generator)
        streamlines = Streamlines([s.astype(np.float32) for s in streamlines])
        streamlines_all.extend(streamlines)

        del generator, streamlines
        gc.collect()

    n_total = len(streamlines_all)
    print(f"\nTotal streamlines: {n_total}")

    # ---- Save raw streamlines (400um-specific filename prefix) ----
    fname = ODF_ANALYSIS_DIR / f"{DMRI_RESOLUTION}um_raw_{tracking_type}_streamlines_{START_IDX}_{END_IDX}_{GFA_THRESHOLD}_{n_total:.4e}.npy"
    print(f"Saving to {fname}")
    np.save(fname, np.array(streamlines_all, dtype=object))

    elapsed = (perf_counter() - script_start) / 3600
    print(f"Total tractography time: {elapsed:.2f} hours")
