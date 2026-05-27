#!/usr/bin/env python
"""Fiber ODF estimation from structure tensor eigenvectors (400um).

Reads eigenvalue/eigenvector Zarr arrays produced by 02_structure_tensor.py,
bins eigenvectors into supervoxels at dMRI resolution, fits spherical harmonic
(SH) coefficients per supervoxel using a spherical histogram projection, and
writes the resulting fODF SH coefficient volume to Zarr. Optionally computes
fractional anisotropy (FA) and creates seed masks for tractography.

This is the 400um variant of 03_fodf_estimation.py. The pipeline logic is
identical; only the configuration section differs.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import re
import sys
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
from fiberorient.util import make_sphere

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.odf import vector_field_SH
from utils.analysis import (
    get_analysis_shape,
    get_voxel_ratio,
    compute_FA,
    extract_rotation_matrix,
    rotate_vector_field,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def extract_z_indices_from_filename(filename):
    """Parse start and end Z indices from a zarr filename stem like 'eigenvalues_zyx_4160_5200'."""
    m = re.search(r'(\d+)_(\d+)$', filename)
    if m:
        return tuple(map(int, m.groups()))
    raise ValueError(f"Filename does not contain valid Z indices: {filename}")


def write_masks_to_zarr(idx, num_slices_per_write, image_files, zarr_path, update_bar):
    """Read mask PNGs, apply binary erosion, and write to a Zarr array."""
    zarr_file = zarr.open(zarr_path, mode='a')
    batch_files = natsorted(image_files[idx : idx + num_slices_per_write])
    for i, filepath in enumerate(batch_files):
        img = cv2.imread(filepath, -1)
        img = ndimage.binary_erosion(img, structure=np.ones((20, 20)), iterations=1).astype(np.uint8)
        zarr_file[idx + i] = img
        if update_bar is not None:
            update_bar()


def load_eigen_vectors_chunk_and_reorient(
    evec_zarr_path,
    chunk_location,
    transpose_xyz_order=(2, 1, 0),
    rgb_order=None,
    update_bar=None,
):
    """Load an eigenvector chunk from Zarr and reorient from ZYX to XYZ.

    Parameters
    ----------
    evec_zarr_path : Path or str
        Path to eigenvector Zarr array with shape (3, Z, Y, X).
    chunk_location : tuple of (start, end) pairs
        ((z_start, z_end), (y_start, y_end), (x_start, x_end)).
    transpose_xyz_order : tuple, optional
        Axis permutation to apply to spatial dimensions (after the component axis).
    rgb_order : tuple, optional
        If provided, reorder the component axis (axis 0) accordingly.

    Returns
    -------
    evec_chunk : ndarray, shape (3, x_chunk, y_chunk, z_chunk) after transpose
    """
    evec_zarr = zarr.open(evec_zarr_path, mode='r')
    z_start, z_end = chunk_location[0]
    y_start, y_end = chunk_location[1]
    x_start, x_end = chunk_location[2]

    evec_chunk = evec_zarr[:, z_start:z_end, y_start:y_end, x_start:x_end]

    if transpose_xyz_order is not None:
        transpose_xyz_order = (np.asarray(transpose_xyz_order) + 1).tolist()
        evec_chunk = evec_chunk.transpose(0, *transpose_xyz_order)

    if rgb_order is not None:
        evec_chunk = evec_chunk[rgb_order, ...]

    if update_bar is not None:
        update_bar()

    return evec_chunk


# ---------------------------------------------------------------------------
# Orientation image generation
# ---------------------------------------------------------------------------

def orientation_analysis(
    eigen_vectors,
    image_dir_to_save_images,
    rgb_order=(0, 1, 2),
    save_index=0,
    preserve_sign=False,
    normalize_vectors=False,
    clip_range=(0.0, 1.0),
):
    """Create RGB orientation images from eigenvector components.

    Parameters
    ----------
    eigen_vectors : ndarray, shape (3, X, Y, Z)
        Component axis 0 = [x_comp, y_comp, z_comp].
    image_dir_to_save_images : Path or str
        Output directory for PNG slices.
    rgb_order : tuple
        Mapping from component index to RGB channel.
    save_index : int
        Offset added to slice index for output filenames.
    preserve_sign : bool
        If True, map [-1, 1] -> [0, 1]. If False, take absolute value.
    normalize_vectors : bool
        If True, normalize each voxel vector to unit length.
    clip_range : tuple
        (min, max) clipping range before scaling to uint8.
    """
    output_dir = Path(image_dir_to_save_images)
    output_dir.mkdir(parents=True, exist_ok=True)

    eigen_vectors = np.asarray(eigen_vectors)
    if eigen_vectors.ndim != 4 or eigen_vectors.shape[0] != 3:
        raise ValueError(f"Expected eigen_vectors shape (3, X, Y, Z). Got: {eigen_vectors.shape}")

    ev = eigen_vectors.copy().astype(np.float32)

    if normalize_vectors:
        norms = np.sqrt(np.sum(ev ** 2, axis=0))
        norms[norms == 0] = 1.0
        ev /= norms

    if preserve_sign:
        ev = np.where(np.isnan(ev), -1.0, ev)
        ev = (ev + 1.0) / 2.0
    else:
        ev = np.where(np.isnan(ev), 0.0, ev)
        ev = np.abs(ev)

    ev = np.clip(ev, clip_range[0], clip_range[1])

    red_x = ev[rgb_order[0]]
    green_y = ev[rgb_order[1]]
    blue_z = ev[rgb_order[2]]

    num_slices = ev.shape[-1]

    def save_slice(slice_idx, update_bar):
        rgb = np.stack([red_x[:, :, slice_idx],
                        green_y[:, :, slice_idx],
                        blue_z[:, :, slice_idx]], axis=-1)
        rgb = (rgb * 255.0).round().astype(np.uint8)
        out_path = output_dir / f"orientation_{slice_idx + save_index:05d}.png"
        cv2.imwrite(str(out_path), rgb.transpose(1, 0, 2)[:, :, ::-1])
        update_bar()

    with tqdm(total=num_slices, desc="Saving orientation images") as bar:
        delayed_tasks = [delayed(save_slice)(i, bar.update) for i in range(num_slices)]
        compute(*delayed_tasks, scheduler='threads')


# ---------------------------------------------------------------------------
# Per-supervoxel fODF computation
# ---------------------------------------------------------------------------

def compute_odf_for_supervoxels(
    supervoxel_array, x, y, z, sh_coeffs,
    method="spherical_histogram",
    degree=None, n_coefs=None, n_bins=None, sphere=None,
    update_bar=None,
):
    """Fit SH coefficients for a single supervoxel.

    Parameters
    ----------
    supervoxel_array : ndarray, shape (3, n_sv_x, vr, n_sv_y, vr, n_sv_z, vr)
        Reshaped eigenvector field binned into supervoxels.
    x, y, z : int
        Supervoxel indices.
    sh_coeffs : ndarray, shape (n_sv_x, n_sv_y, n_sv_z, n_coefs)
        Output array (written in-place).
    method : str
        'spherical_histogram' or 'dirac_delta'.
    degree : int
        SH order max.
    n_coefs : int
        Number of SH coefficients.
    n_bins : int
        Number of bins for spherical histogram.
    sphere : dipy Sphere
        Sphere for ODF sampling.
    """
    sv_evec = supervoxel_array[:, x, :, y, :, z, :]
    sv_evec = sv_evec.transpose(1, 2, 3, 0)
    sv_shape = sv_evec.shape

    if np.any(np.isnan(sv_evec)) or np.any(np.isinf(sv_evec)):
        mask = np.ones(sv_shape[:-1], dtype=bool)
        for dim in range(3):
            mask[np.isnan(sv_evec[..., dim])] = False
            mask[np.isinf(sv_evec[..., dim])] = False

        sv_evec = sv_evec[mask]

        if sv_evec.size == 0:
            sh_coeffs[x, y, z, :] = np.full((n_coefs,), np.nan, dtype=np.float32)
            if update_bar is not None:
                update_bar()
            return

    odf = vector_field_SH(sh_order_max=degree).fit(sv_evec, method=method, n_bins=n_bins)
    sh_coeffs[x, y, z, :] = odf.shm_coeffs

    if update_bar is not None:
        update_bar()


# ---------------------------------------------------------------------------
# Main processing function for one chunk
# ---------------------------------------------------------------------------

def process_eigenvalues_eigenvectors(
    chunk_location,
    start_slice=None,
    end_slice=None,
    evec_zarr_path=None,
    eval_zarr_path=None,

    load_eigen_vectors=True,
    perform_eigenvector_masking=False,
    mask_dir=None,

    perform_vector_field_rotation=False,
    affine_matrix=None,

    save_eigen_vectors_as_rgb=False,
    rgb_order=(0, 1, 2),
    save_eigen_vectors_as_rgb_path=None,

    perform_fodf_analysis=True,
    shc_zarr_path=None,
    voxel_ratio=None,
    degree=None,
    n_coefs=None,
    n_bins=None,
    sphere=None,

    perform_FA_analysis=False,
    save_FA_maps=False,
    fa_analysis_dir=None,
    create_seed_mask=True,
    seed_mask_threshold=0.4,
    seed_mask_zarr_path=None,

    num_workers=None,
    pbar=None,
):
    """Process eigenvectors/eigenvalues for one spatial chunk.

    Orchestrates: loading -> optional masking -> optional rotation ->
    optional RGB orientation images -> fODF fitting -> optional FA / seed mask.
    """
    if num_workers is None:
        num_workers = os.cpu_count() // 2

    print(f"\n\nProcessing chunk at location {chunk_location}")

    chunk_location_z = chunk_location[0]
    chunk_location_y = chunk_location[1]
    chunk_location_x = chunk_location[2]
    print(f"Chunk location Z: {chunk_location_z}, Y: {chunk_location_y}, X: {chunk_location_x}")

    # --- Optional mask loading ---
    masks = None
    if perform_eigenvector_masking:
        print("Loading masks for eigenvector masking")
        mask_list = zarr.open(Path(mask_dir), mode='r')
        masks = mask_list[
            chunk_location_z[0]:chunk_location_z[1],
            chunk_location_y[0]:chunk_location_y[1],
            chunk_location_x[0]:chunk_location_x[1],
        ].transpose(2, 1, 0)

    # --- Load eigenvectors ---
    eigen_vector = None
    if load_eigen_vectors:
        print(f"Loading eigenvectors")
        eigen_vector = load_eigen_vectors_chunk_and_reorient(
            evec_zarr_path=evec_zarr_path,
            chunk_location=((0, end_slice - start_slice), chunk_location_y, chunk_location_x),
            transpose_xyz_order=(2, 1, 0),
        )
        print(f"Loaded eigenvectors shape: {eigen_vector.shape}")

        if perform_eigenvector_masking and masks is not None:
            print("Masking eigenvectors")
            eigen_vector = np.where(masks, eigen_vector, np.nan)

    # --- Optional vector field rotation ---
    if perform_vector_field_rotation and eigen_vector is not None:
        print("Rotating eigenvectors using affine matrix")
        rotation_matrix = extract_rotation_matrix(affine_matrix)
        eigen_vector = rotate_vector_field(rotation_matrix, eigen_vector)

    # --- Optional RGB orientation images ---
    if save_eigen_vectors_as_rgb and eigen_vector is not None:
        print("Saving RGB orientation images")
        image_dir = Path(save_eigen_vectors_as_rgb_path)
        image_dir.mkdir(parents=True, exist_ok=True)
        num_slices = eigen_vector.shape[-1]
        orientation_analysis(
            eigen_vectors=eigen_vector[..., num_slices // 2 : num_slices // 2 + 1],
            image_dir_to_save_images=image_dir,
            rgb_order=rgb_order,
            save_index=start_slice,
        )

    # --- fODF analysis ---
    if perform_fodf_analysis and eigen_vector is not None:
        print("Performing fODF analysis")

        remainder_x, remainder_y, remainder_z = get_analysis_shape(
            eigen_vector.shape[1], eigen_vector.shape[2], eigen_vector.shape[3], voxel_ratio,
        )

        eigen_vector_binned = eigen_vector[:, :remainder_x, :remainder_y, :remainder_z]
        print(f"Cropped eigenvectors shape: {eigen_vector_binned.shape}")
        eigen_vector = None

        eigen_vector_binned = eigen_vector_binned.reshape(
            eigen_vector_binned.shape[0],
            eigen_vector_binned.shape[1] // voxel_ratio, voxel_ratio,
            eigen_vector_binned.shape[2] // voxel_ratio, voxel_ratio,
            eigen_vector_binned.shape[3] // voxel_ratio, voxel_ratio,
        )
        print(f"Reshaped to supervoxels: {eigen_vector_binned.shape}")

        sh_coeffs = np.zeros(
            (eigen_vector_binned.shape[1], eigen_vector_binned.shape[3],
             eigen_vector_binned.shape[5], n_coefs),
            dtype=np.float32,
        )

        total = eigen_vector_binned.shape[1] * eigen_vector_binned.shape[3] * eigen_vector_binned.shape[5]
        eigen_vector_binned_shape = eigen_vector_binned.shape
        eigen_vector_binned = delayed(eigen_vector_binned)
        sphere_delayed = delayed(sphere)

        with tqdm(total=total, desc="Computing ODFs for supervoxels") as bar:
            delayed_tasks = [
                delayed(compute_odf_for_supervoxels)(
                    supervoxel_array=eigen_vector_binned,
                    x=x, y=y, z=z,
                    sh_coeffs=sh_coeffs,
                    method="spherical_histogram",
                    degree=degree, n_coefs=n_coefs, n_bins=n_bins,
                    sphere=sphere_delayed,
                    update_bar=bar.update,
                )
                for z in range(eigen_vector_binned_shape[5])
                for y in range(eigen_vector_binned_shape[3])
                for x in range(eigen_vector_binned_shape[1])
            ]
            compute(*delayed_tasks, scheduler='threads', num_workers=num_workers)

        eigen_vector_binned = None

        shc_zarr = zarr.open(shc_zarr_path, mode='a')
        xs = (np.asarray(chunk_location_x) // voxel_ratio).tolist()
        ys = (np.asarray(chunk_location_y) // voxel_ratio).tolist()
        zs = (np.asarray(chunk_location_z) // voxel_ratio).tolist()
        print(f"Saving SH coefficients to zarr at xs: {xs}, ys: {ys}, zs: {zs}")
        shc_zarr[xs[0]:xs[1], ys[0]:ys[1], zs[0]:zs[1], :] = sh_coeffs

    eigen_vector = None

    # --- FA analysis ---
    if perform_FA_analysis:
        print("Performing FA analysis")
        eigen_value = load_eigen_vectors_chunk_and_reorient(
            evec_zarr_path=eval_zarr_path,
            chunk_location=((0, end_slice - start_slice), chunk_location_y, chunk_location_x),
            transpose_xyz_order=(2, 1, 0),
        )
        print(f"Loaded eigenvalues shape: {eigen_value.shape}")

        if perform_eigenvector_masking and masks is not None:
            print("Masking eigenvalues")
            eigen_value = np.where(masks, eigen_value, np.nan)
            masks = None

        FA = compute_FA(eigen_value)
        print(f"Computed FA shape: {FA.shape}")
        eigen_value = None

        if save_FA_maps:
            print("Saving FA images")
            image_dir_to_save = Path(fa_analysis_dir)
            image_dir_to_save.mkdir(parents=True, exist_ok=True)

            def save_FA_images(slice_idx, save_index, out_dir, update_bar=None):
                image_slice = np.multiply(FA[..., slice_idx], 255).round().astype(np.uint8)
                image_slice = np.clip(image_slice, 0, 255)
                out_path = out_dir / f"FA_{slice_idx + save_index:05d}.png"
                cv2.imwrite(str(out_path), image_slice.T)
                if update_bar:
                    update_bar()

            num_slices = FA.shape[-1]
            with tqdm(desc="Saving FA images") as bar:
                delayed_tasks = [
                    delayed(save_FA_images)(idx, start_slice, image_dir_to_save, bar.update)
                    for idx in range(num_slices // 2, num_slices // 2 + 1)
                ]
                compute(*delayed_tasks, scheduler='threads')

        if create_seed_mask:
            print("Creating seed mask from FA map")
            seed_mask = np.where((FA > seed_mask_threshold) & (FA < 0.8), 1, 0).astype(np.uint8)
            FA = None

            seed_mask_zarr = zarr.open(seed_mask_zarr_path, mode='a')
            seed_mask_zarr[
                chunk_location_x[0]:chunk_location_x[1],
                chunk_location_y[0]:chunk_location_y[1],
                chunk_location_z[0]:chunk_location_z[1],
            ] = seed_mask

        FA = None

    print(f"Finished processing chunk at location {chunk_location}")
    print("_" * 50)

    if pbar is not None:
        pbar()

    gc.collect()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration (400um — differs from 800um version) ----
    IMAGE_DIR = Path("15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked")
    EVS_PATH = Path("/media/eric/sta_4160_5200")
    SAVE_DIR = Path("/media/eric/sta_analysis")

    MASK_PNG_DIR = Path("15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masks")
    MASK_ZARR_PATH = Path("/hdd/eric/brain/whole_brain_i58_paper/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masks.zarr")

    HIPCT_RESOLUTION = 15.13   # um
    DMRI_RESOLUTION = 400.0    # um  (800.0 in 800um version)

    SIGMA = 2.0
    RHO = 4.0
    GROUP_SIZE = 1040          # STA chunk size (800um derives as VOXEL_RATIO_SCALE * voxel_ratio = 20 * 52)
    DEGREE = 8
    N_BINS = 6500
    NUM_WORKERS = 4

    # ---- Derived parameters ----
    n_coefs = int(((DEGREE + 1) * (DEGREE + 2)) // 2)
    sphere = make_sphere(N_BINS)

    voxel_ratio = get_voxel_ratio(HIPCT_RESOLUTION, DMRI_RESOLUTION)
    group_size = GROUP_SIZE

    # ---- Read image metadata ----
    image_files_for_saving = list(natsorted(IMAGE_DIR.glob("*.jp2")))
    sample_img = cv2.imread(str(image_files_for_saving[0]), -1)
    Y, X = sample_img.shape
    print(f"Sample image shape YX: {sample_img.shape}, dtype: {sample_img.dtype}")

    # ---- Locate STA output zarrs ----
    evs_dir = EVS_PATH / f"{IMAGE_DIR.name}_structure_tensor_{SIGMA}_{RHO}"
    eval_zarr_paths = natsorted(list(evs_dir.glob("eigenvalues_zyx_*.zarr")))
    evec_zarr_paths = natsorted(list(evs_dir.glob("eigenvectors_zyx_*.zarr")))
    print(f"Found {len(eval_zarr_paths)} eigenvalue zarr paths.")
    print(f"Found {len(evec_zarr_paths)} eigenvector zarr paths.")

    # ---- Create output Zarr arrays ----
    odf_analysis_dir = SAVE_DIR / f"{IMAGE_DIR.name}_structure_tensor_{SIGMA}_{RHO}_analysis"
    odf_analysis_dir.mkdir(parents=True, exist_ok=True)

    shc_zarr_path = odf_analysis_dir / f"sh_coefficients_degree_{DEGREE}_{DMRI_RESOLUTION}um_xyz.zarr"

    analysis_shape_xyz = get_analysis_shape(X, Y, len(image_files_for_saving), voxel_ratio)
    print(f"Analysis shape (X, Y, Z): {analysis_shape_xyz}")

    shc_shape = (
        analysis_shape_xyz[0] // voxel_ratio,
        analysis_shape_xyz[1] // voxel_ratio,
        analysis_shape_xyz[2] // voxel_ratio,
        n_coefs,
    )
    shc_chunks = (
        group_size // voxel_ratio,
        group_size // voxel_ratio,
        group_size // voxel_ratio,
        n_coefs,
    )
    print(f"SH coefficients zarr shape: {shc_shape}, chunks: {shc_chunks}")
    zarr.open(
        shc_zarr_path, mode='a',
        shape=shc_shape, chunks=shc_chunks,
        dtype=np.float32,
        compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE),
    )

    seed_mask_zarr_path = odf_analysis_dir / f"seed_mask_xyz_{DMRI_RESOLUTION}um.zarr"
    seed_mask_shape = (X, Y, len(image_files_for_saving))
    seed_mask_chunks = (group_size, group_size, group_size)
    print(f"Seed mask zarr shape: {seed_mask_shape}, chunks: {seed_mask_chunks}")
    zarr.open(
        seed_mask_zarr_path, mode='a',
        shape=seed_mask_shape, chunks=seed_mask_chunks,
        dtype=np.uint8,
        compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE),
    )

    # ---- Create mask Zarr from PNGs (if needed) ----
    png_mask_files = natsorted(list(MASK_PNG_DIR.glob("*.png")))
    if not MASK_ZARR_PATH.exists() or zarr.open(MASK_ZARR_PATH, mode='r').shape[0] != len(png_mask_files):
        print(f"Creating mask zarr at {MASK_ZARR_PATH}")
        num_slices_per_write = 20
        zarr.open(
            MASK_ZARR_PATH, mode='a',
            shape=(len(png_mask_files), Y, X),
            chunks=(1, Y, X),
            dtype=np.uint8,
            compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE),
        )
        with tqdm(total=len(png_mask_files), desc="Writing masks to zarr") as bar:
            delayed_tasks = [
                delayed(write_masks_to_zarr)(idx, num_slices_per_write, png_mask_files, MASK_ZARR_PATH, bar.update)
                for idx in range(0, len(png_mask_files), num_slices_per_write)
            ]
            compute(*delayed_tasks, scheduler='threads', num_workers=os.cpu_count() // 2)
    else:
        print(f"Mask zarr at {MASK_ZARR_PATH} already exists. Skipping creation.")

    # ---- Process each eigenvector zarr ----
    for evec_zarr_path in tqdm(evec_zarr_paths, desc="Processing eigenvector zarrs"):
        script_start = perf_counter()

        eval_zarr_path = evec_zarr_path.as_posix().replace("eigenvectors_zyx", "eigenvalues_zyx")
        print(f"\n\nProcessing: {evec_zarr_path.stem}")

        z_indices = [extract_z_indices_from_filename(evec_zarr_path.stem)]
        y_indices = [(idx, min(idx + group_size, Y)) for idx in range(0, Y, group_size)]
        x_indices = [(idx, min(idx + group_size, X)) for idx in range(0, X, group_size)]

        chunk_locations = [
            (z_idx, y_idx, x_idx)
            for z_idx in z_indices
            for y_idx in y_indices
            for x_idx in x_indices
        ]

        with tqdm(total=len(chunk_locations)) as pbar:
            dtasks = [
                delayed(process_eigenvalues_eigenvectors)(
                    chunk_location=cl,
                    start_slice=cl[0][0],
                    end_slice=cl[0][1],
                    evec_zarr_path=evec_zarr_path,
                    eval_zarr_path=eval_zarr_path,
                    load_eigen_vectors=True,
                    perform_eigenvector_masking=True,
                    mask_dir=MASK_ZARR_PATH,
                    perform_vector_field_rotation=False,
                    affine_matrix=None,
                    save_eigen_vectors_as_rgb=False,
                    rgb_order=(0, 1, 2),
                    save_eigen_vectors_as_rgb_path=None,
                    perform_fodf_analysis=True,
                    shc_zarr_path=shc_zarr_path,
                    voxel_ratio=voxel_ratio,
                    degree=DEGREE,
                    n_coefs=n_coefs,
                    n_bins=N_BINS,
                    sphere=sphere,
                    perform_FA_analysis=True,
                    save_FA_maps=False,
                    fa_analysis_dir=None,
                    create_seed_mask=True,
                    seed_mask_threshold=0.40,
                    seed_mask_zarr_path=seed_mask_zarr_path,
                    num_workers=NUM_WORKERS,
                    pbar=pbar.update,
                )
                for cl in chunk_locations
            ]
            compute(*dtasks, scheduler='threads', num_workers=NUM_WORKERS)

        elapsed = (perf_counter() - script_start) / 3600
        print(f"Total time for {evec_zarr_path.stem}: {elapsed:.2f} hours")
