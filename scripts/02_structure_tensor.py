#!/usr/bin/env python
"""Chunked structure tensor analysis (STA) on HiP-CT volumes stored as Zarr.

Converts raw HiP-CT image slices to a temporary Zarr volume, computes the 3D
structure tensor with Gaussian derivative filters, performs eigendecomposition,
and writes eigenvalue/eigenvector fields back to Zarr arrays. Processing is
parallelised over spatial chunks using Dask.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import math
import shutil
import sys
import time
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import zarr
from dask import compute, delayed
from natsort import natsorted
from numcodecs import Blosc
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.eigen_decomposition import eig_special_3d


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_voxel_ratio(hipct_resolution, dmri_resolution):
    """Return the number of HiP-CT voxels that fit in one dMRI voxel."""
    voxel_ratio = int(dmri_resolution // hipct_resolution)
    print(f"Number of HiP-CT voxels in 1 dMRI voxel: {voxel_ratio}")
    return voxel_ratio


def get_start_end_indices(start_idx, desired_depth, voxel_ratio, padding):
    """Compute padded start/end Z indices, ensuring depth is divisible by voxel_ratio."""
    if desired_depth % voxel_ratio != 0:
        adjusted = desired_depth + (voxel_ratio - (desired_depth % voxel_ratio))
        print(f"Adjusted desired_depth from {desired_depth} to {adjusted} (divisible by voxel_ratio)")
        desired_depth = adjusted

    start_idx_padded = start_idx - padding
    end_idx_padded = start_idx + desired_depth + padding
    print(f"Padded indices: start={start_idx_padded}, end={end_idx_padded}")
    return start_idx_padded, end_idx_padded


def get_processing_chunks(voxel_ratio, total_img_slices, voxel_ratio_scale=20):
    """Split total Z slices into chunks whose depth is a multiple of voxel_ratio."""
    chunk_depth = voxel_ratio * voxel_ratio_scale
    print(f"Processing chunk depth: {chunk_depth} slices")
    chunks = []
    for start in range(0, total_img_slices, chunk_depth):
        end = min(start + chunk_depth, total_img_slices)
        chunks.append((start, end))
    return chunks


def write_to_zarr(idx, num_slices_per_write, image_files, zarr_path, update_bar):
    """Read a batch of image slices and write them into a Zarr array."""
    zarr_file = zarr.open(zarr_path, mode='a')
    batch_files = natsorted(image_files[idx : idx + num_slices_per_write])
    for i, filepath in enumerate(batch_files):
        img = cv2.imread(filepath, -1)
        zarr_file[idx + i] = img
        if update_bar is not None:
            update_bar()


# ---------------------------------------------------------------------------
# Core STA computation for a single spatial chunk
# ---------------------------------------------------------------------------

def sta_analysis(zarr_volume_path, eval_zarr_path, evec_zarr_path,
                 z_idx, y_idx, x_idx, padding, sigma, rho, truncate,
                 dtype=np.float32, update_bar=None):
    """Compute structure tensor eigendecomposition for one spatial chunk.

    Reads a padded chunk from the Zarr volume, computes Gaussian-derivative
    gradients, assembles and integrates the structure tensor, strips padding,
    runs eigendecomposition, and writes results to output Zarr arrays.
    """
    print(f"\nProcessing chunk: Z({z_idx[0]}:{z_idx[1]}), Y({y_idx[0]}:{y_idx[1]}), X({x_idx[0]}:{x_idx[1]})")

    zarr_volume = zarr.open(zarr_volume_path, mode='r')

    # Padded region (clamped to volume bounds)
    start_z, end_z = max(0, z_idx[0] - padding), min(zarr_volume.shape[0], z_idx[1] + padding)
    start_y, end_y = max(0, y_idx[0] - padding), min(zarr_volume.shape[1], y_idx[1] + padding)
    start_x, end_x = max(0, x_idx[0] - padding), min(zarr_volume.shape[2], x_idx[1] + padding)

    # Offsets for stripping padding after filtering
    slice_start_z = z_idx[0] - start_z
    slice_start_y = y_idx[0] - start_y
    slice_start_x = x_idx[0] - start_x

    size_z = z_idx[1] - z_idx[0]
    size_y = y_idx[1] - y_idx[0]
    size_x = x_idx[1] - x_idx[0]
    final_chunk_shape = (size_z, size_y, size_x)

    slice_end_z = slice_start_z + size_z
    slice_end_y = slice_start_y + size_y
    slice_end_x = slice_start_x + size_x

    # Load padded chunk
    print(f"Extracting padded chunk: Z({start_z}:{end_z}), Y({start_y}:{end_y}), X({start_x}:{end_x})")
    chunk_volume = zarr_volume[start_z:end_z, start_y:end_y, start_x:end_x].copy().astype(dtype)

    if chunk_volume.shape != (end_z - start_z, end_y - start_y, end_x - start_x):
        raise ValueError(
            f"Extracted chunk shape {chunk_volume.shape} does not match "
            f"expected shape {(end_z - start_z, end_y - start_y, end_x - start_x)}"
        )

    # Handle empty chunks
    if np.allclose(chunk_volume, 0, atol=1e-10):
        print("Warning: chunk is all zeros, writing NaN eigen arrays.")
        chunk_volume = None
        nan_vals = np.full((3, *final_chunk_shape), np.nan, dtype=np.float32)

        eval_zarr = zarr.open(eval_zarr_path, mode='a')
        evec_zarr = zarr.open(evec_zarr_path, mode='a')
        eval_zarr[:, 0:size_z, y_idx[0]:y_idx[1], x_idx[0]:x_idx[1]] = nan_vals
        evec_zarr[:, 0:size_z, y_idx[0]:y_idx[1], x_idx[0]:x_idx[1]] = nan_vals

        print(f"Finished chunk Z[{z_idx[0]}:{z_idx[1]}], Y[{y_idx[0]}:{y_idx[1]}], X[{x_idx[0]}:{x_idx[1]}].")
        gc.collect()
        if update_bar is not None:
            update_bar()
        return

    # --- Compute gradients via Gaussian derivative convolution ---
    Vx = ndimage.gaussian_filter(chunk_volume, sigma, order=[0, 0, 1], mode='nearest', truncate=truncate)
    Vy = ndimage.gaussian_filter(chunk_volume, sigma, order=[0, 1, 0], mode='nearest', truncate=truncate)
    Vz = ndimage.gaussian_filter(chunk_volume, sigma, order=[1, 0, 0], mode='nearest', truncate=truncate)
    chunk_volume_shape = chunk_volume.shape
    chunk_volume = None

    # --- Assemble and integrate structure tensor ---
    S = np.empty((6,) + chunk_volume_shape, dtype=np.float64)

    np.multiply(Vx, Vx, out=S[0])
    ndimage.gaussian_filter(S[0], rho, mode='nearest', output=S[0], truncate=truncate)

    np.multiply(Vy, Vy, out=S[1])
    ndimage.gaussian_filter(S[1], rho, mode='nearest', output=S[1], truncate=truncate)

    np.multiply(Vz, Vz, out=S[2])
    ndimage.gaussian_filter(S[2], rho, mode='nearest', output=S[2], truncate=truncate)

    np.multiply(Vx, Vy, out=S[3])
    ndimage.gaussian_filter(S[3], rho, mode='nearest', output=S[3], truncate=truncate)

    np.multiply(Vx, Vz, out=S[4])
    ndimage.gaussian_filter(S[4], rho, mode='nearest', output=S[4], truncate=truncate)

    np.multiply(Vy, Vz, out=S[5])
    ndimage.gaussian_filter(S[5], rho, mode='nearest', output=S[5], truncate=truncate)

    Vx, Vy, Vz = None, None, None

    # Strip padding
    S = S[:, slice_start_z:slice_end_z, slice_start_y:slice_end_y, slice_start_x:slice_end_x]
    assert S.shape == (6, *final_chunk_shape), f"Unexpected S shape: {S.shape}"

    # --- Eigendecomposition ---
    print(f"Computing eigenvalues and eigenvectors for chunk")
    eigen_values, eigen_vectors = eig_special_3d(S, full=False)
    S = None

    # --- Write results ---
    eval_zarr = zarr.open(eval_zarr_path, mode='a')
    evec_zarr = zarr.open(evec_zarr_path, mode='a')
    eval_zarr[:, 0:size_z, y_idx[0]:y_idx[1], x_idx[0]:x_idx[1]] = eigen_values.astype(np.float32)
    evec_zarr[:, 0:size_z, y_idx[0]:y_idx[1], x_idx[0]:x_idx[1]] = eigen_vectors.astype(np.float32)
    eigen_values, eigen_vectors = None, None

    print(f"Finished chunk Z[{z_idx[0]}:{z_idx[1]}], Y[{y_idx[0]}:{y_idx[1]}], X[{x_idx[0]}:{x_idx[1]}].")
    gc.collect()
    if update_bar is not None:
        update_bar()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration ----
    IMAGE_DIR = Path("15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked")
    EXTENSION = ".jp2"
    SAVE_DIR = Path("/media/eric/sta_results")
    ZARR_VOL_DIR = Path("temporary_hipct_zarr_volume")

    HIPCT_RESOLUTION = 15.13   # um
    DMRI_RESOLUTION = 800.0    # um

    SIGMA = 2.0                # Gaussian noise scale (HiP-CT voxels)
    RHO = 4.0                  # Gaussian integration scale
    TRUNCATE = 4.0             # Filter truncation (in sigma units)

    VOXEL_RATIO_SCALE = 20     # Chunk depth = voxel_ratio * this
    DESIRED_CHUNK_IDX = (8,)   # Which Z-chunks to process
    NUM_WORKERS = 4            # Dask thread workers

    # ---- Derived parameters ----
    pad = int(math.ceil(TRUNCATE * max(SIGMA, RHO)) + 2)
    print(f"Padding needed: {pad} voxels")

    ZARR_VOL_DIR.mkdir(parents=True, exist_ok=True)

    image_files = natsorted([str(f) for f in IMAGE_DIR.rglob(f"*{EXTENSION}")])
    print(f"Number of image files: {len(image_files)}")

    voxel_ratio = get_voxel_ratio(HIPCT_RESOLUTION, DMRI_RESOLUTION)
    total_chunks = get_processing_chunks(voxel_ratio, len(image_files), voxel_ratio_scale=VOXEL_RATIO_SCALE)
    print(f"Total processing chunks: {len(total_chunks)}")

    chunks_to_process = [total_chunks[i] for i in DESIRED_CHUNK_IDX]
    print(f"Chunks to process: {chunks_to_process}")

    # ---- Process each Z-chunk ----
    for desired_idx in DESIRED_CHUNK_IDX:
        script_start = perf_counter()
        processing_chunk = total_chunks[desired_idx]
        print(f"\nSelected processing chunk {desired_idx}: {processing_chunk}")

        start_idx, end_idx = get_start_end_indices(
            processing_chunk[0], processing_chunk[1] - processing_chunk[0], voxel_ratio, pad
        )
        start_idx = max(0, start_idx)
        end_idx = min(len(image_files), end_idx)
        print(f"Padded Z range: [{start_idx}, {end_idx})")

        image_files_for_saving = natsorted(image_files[start_idx:end_idx])
        num_images = len(image_files_for_saving)

        sample_img = cv2.imread(image_files_for_saving[0], -1)
        Y, X = sample_img.shape
        print(f"Image shape YX: ({Y}, {X}), dtype: {sample_img.dtype}")

        # Spatial sub-chunk indices for parallel STA
        group_size = voxel_ratio * VOXEL_RATIO_SCALE
        z_start = max(0, processing_chunk[0] - start_idx)
        z_end = z_start + (processing_chunk[1] - processing_chunk[0])

        z_indices = [(z_start, z_end)]
        y_indices = [(i, min(i + group_size, Y)) for i in range(0, Y, group_size)]
        x_indices = [(i, min(i + group_size, X)) for i in range(0, X, group_size)]

        # ---- Create temporary Zarr volume from image slices ----
        zarr_vol_path = ZARR_VOL_DIR / f"volume_zyx_{processing_chunk[0]}_{processing_chunk[1]}.zarr"
        total_volume_shape = (num_images, Y, X)
        chunk_volume_shape = (1, Y, X)
        num_slices_per_write = voxel_ratio

        zarr_is_empty = True
        if zarr_vol_path.exists():
            test_slice = zarr.open(zarr_vol_path, mode='r')[group_size // 2, :, :]
            zarr_is_empty = (test_slice.min() == 0 and test_slice.max() == 0)

        if not zarr_vol_path.exists() or zarr_is_empty:
            print(f"Creating temporary Zarr volume at {zarr_vol_path}")
            zarr.open(
                zarr_vol_path, mode='a',
                shape=total_volume_shape, chunks=chunk_volume_shape,
                dtype=sample_img.dtype,
                compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE),
                order='C',
            )
            with tqdm(total=num_images, desc="Writing to zarr") as bar:
                delayed_tasks = [
                    delayed(write_to_zarr)(idx, num_slices_per_write, image_files_for_saving, zarr_vol_path, bar.update)
                    for idx in range(0, num_images, num_slices_per_write)
                ]
                compute(*delayed_tasks, scheduler='threads', num_workers=min(os.cpu_count(), 52))
        else:
            print(f"Zarr volume at {zarr_vol_path} already exists. Skipping creation.")

        # ---- Create output Zarr arrays for eigenvalues and eigenvectors ----
        save_dir = SAVE_DIR / f"{IMAGE_DIR.name}_structure_tensor_{SIGMA}_{RHO}"
        save_dir.mkdir(parents=True, exist_ok=True)

        output_shape = (3, processing_chunk[1] - processing_chunk[0], Y, X)
        output_chunks = (3, processing_chunk[1] - processing_chunk[0], group_size // 4, group_size // 4)
        print(f"Output Zarr shape: {output_shape}, chunks: {output_chunks}")

        eval_zarr_path = save_dir / f"eigenvalues_zyx_{processing_chunk[0]}_{processing_chunk[1]}.zarr"
        evec_zarr_path = save_dir / f"eigenvectors_zyx_{processing_chunk[0]}_{processing_chunk[1]}.zarr"

        for path in (eval_zarr_path, evec_zarr_path):
            zarr.open(
                path, mode='a',
                shape=output_shape, chunks=output_chunks,
                dtype=np.float32,
                compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE),
                order='C',
            )

        # ---- Run parallel STA ----
        total_tasks = len(z_indices) * len(y_indices) * len(x_indices)
        with tqdm(total=total_tasks, desc="STA for chunks") as bar:
            delayed_tasks = [
                delayed(sta_analysis)(
                    zarr_volume_path=zarr_vol_path,
                    eval_zarr_path=eval_zarr_path,
                    evec_zarr_path=evec_zarr_path,
                    z_idx=z_idx, y_idx=y_idx, x_idx=x_idx,
                    padding=pad, sigma=SIGMA, rho=RHO, truncate=TRUNCATE,
                    dtype=np.float32, update_bar=bar.update,
                )
                for z_idx in z_indices
                for y_idx in y_indices
                for x_idx in x_indices
            ]
            compute(*delayed_tasks, scheduler='threads', num_workers=NUM_WORKERS)

        elapsed = (perf_counter() - script_start) / 3600
        print(f"Total time for chunk {desired_idx}: {elapsed:.2f} hours")

        # Clean up temporary zarr volume
        shutil.rmtree(zarr_vol_path)
        gc.collect()
        time.sleep(10)
