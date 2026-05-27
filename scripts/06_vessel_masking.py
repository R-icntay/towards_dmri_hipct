#!/usr/bin/env python
"""Vessel masking analysis on LADAF-2021-17 Pons VOI.

Performs structure tensor analysis (STA) with and without gradient-level
vessel masking on the LADAF-2021-17 Pons VOI (6.54µm resolution). Compares
the effect of vasculature on fiber orientation by:

1. Running STA on the full VOI (no vessel mask)
2. Running STA with an inverted vessel mask applied at the gradient level
3. Generating RGB orientation images and FA maps for both
4. Computing per-supervoxel fODFs at dMRI-equivalent resolution with vessel
   volume/fraction tracking (both masked and unmasked eigenvectors)
5. Fitting global fODFs across the entire VOI (both masked and unmasked)

Gradient-level masking: the inverted vessel mask is applied to the image
intensity gradients (between Gaussian derivative and tensor integration),
suppressing vascular contributions before eigendecomposition.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from fiberorient.odf import ODF
from fiberorient.util import make_sphere
from fiberorient.vis import show_odf
from fury import actor, window
from natsort import natsorted
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.analysis import compute_FA, drop_nans_infs, get_voxel_ratio, validate_eigenvalue_order
from utils.eigen_decomposition import eig_special_3d


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sta_analysis(
    image_dir, extension, start_index, end_index,
    sigma, rho, save_dir, mask_dir=None, save_index=0,
):
    """Load image slices, compute structure tensor, and run eigendecomposition.

    When mask_dir is provided, an inverted vessel mask is applied to the
    intensity gradients before tensor integration (gradient-level masking).
    """
    print(f"\nProcessing image directory {image_dir}")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(extension)]
    image_files = natsorted(image_files)[start_index:end_index]
    print(f"Processing image files from {image_files[0]} to {image_files[-1]}")

    mask_files = None
    if mask_dir is not None:
        print(f"Processing mask directory {mask_dir}")
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(extension)]
        mask_files = natsorted(mask_files)[start_index:end_index]
        print(f"Length of mask files: {len(mask_files)}")

    def read_image(filepath):
        if extension in (".tiff", ".tif"):
            return np.array(Image.open(filepath))
        elif extension == ".jp2":
            return np.array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED))

    dummy_image = read_image(os.path.join(image_dir, image_files[0]))
    image_3d = np.zeros(
        (dummy_image.shape[0], dummy_image.shape[1], len(image_files)),
        dtype=np.float64,
    )
    mask_3d = np.zeros_like(image_3d)

    print("Storing images in a numpy array")
    for i, image_file in enumerate(image_files):
        image_3d[:, :, i] = read_image(os.path.join(image_dir, image_file))

    if mask_files is not None:
        print("Storing masks in a numpy array")
        for i, mask_file in enumerate(mask_files):
            mask_3d[:, :, i] = read_image(os.path.join(mask_dir, mask_file))

    # Normalize volume to [0, 255]
    print("Normalizing the volume")
    volume = image_3d
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    volume = np.multiply(volume, 255).astype(np.float32)

    if mask_dir is not None:
        print("Binarizing mask")
        mask_3d = (mask_3d - np.min(mask_3d)) / (np.max(mask_3d) - np.min(mask_3d))
        mask_3d = mask_3d.astype(np.float32)
        print(f"Max of mask: {np.max(mask_3d)}, Min of mask: {np.min(mask_3d)}")

    print(f"Shape before re-orientation: {volume.shape}")

    # Transpose YX → XY to match dMRI orientation
    volume = volume.transpose(1, 0, 2)
    if mask_dir is not None:
        mask_3d = mask_3d.transpose(1, 0, 2)

    print(f"Shape after re-orientation: {volume.shape}")

    # Invert vessel mask (vessels=1 → vessels=0, tissue=0 → tissue=1)
    if mask_dir is not None:
        mask_3d = 1 - mask_3d
        mask_3d[mask_3d == np.min(mask_3d)] = 0

    print("\nStarting Structure Tensor computation")
    print(f"Volume dtype: {volume.dtype}")

    # Compute image intensity gradients via Gaussian derivatives
    truncate = 4.0
    print("Computing gradient by convolution with Gaussian derivative")
    Vx = gaussian_filter(volume, sigma, order=[0, 0, 1], mode='nearest', truncate=truncate)
    Vy = gaussian_filter(volume, sigma, order=[0, 1, 0], mode='nearest', truncate=truncate)
    Vz = gaussian_filter(volume, sigma, order=[1, 0, 0], mode='nearest', truncate=truncate)

    # Gradient-level vessel masking: suppress vascular gradients
    if mask_dir is not None:
        print("Masking the vessel gradients")
        Vx = np.multiply(Vx, mask_3d)
        Vy = np.multiply(Vy, mask_3d)
        Vz = np.multiply(Vz, mask_3d)

    # Integrate structure tensor elements
    print("Computing the structure tensor")
    S = np.empty((6,) + volume.shape, dtype=volume.dtype)
    tmp = np.empty(volume.shape, dtype=volume.dtype)
    np.multiply(Vx, Vx, out=tmp)
    gaussian_filter(tmp, rho, mode='nearest', output=S[0], truncate=truncate)
    np.multiply(Vy, Vy, out=tmp)
    gaussian_filter(tmp, rho, mode='nearest', output=S[1], truncate=truncate)
    np.multiply(Vz, Vz, out=tmp)
    gaussian_filter(tmp, rho, mode='nearest', output=S[2], truncate=truncate)
    np.multiply(Vx, Vy, out=tmp)
    gaussian_filter(tmp, rho, mode='nearest', output=S[3], truncate=truncate)
    np.multiply(Vx, Vz, out=tmp)
    gaussian_filter(tmp, rho, mode='nearest', output=S[4], truncate=truncate)
    np.multiply(Vy, Vz, out=tmp)
    gaussian_filter(tmp, rho, mode='nearest', output=S[5], truncate=truncate)

    # Eigendecomposition
    print("Computing eigenvalues and eigenvectors")
    eigen_values, eigen_vectors = eig_special_3d(S, full=False)

    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start_index + save_index}_{end_index + save_index}"
    print(f"Saving to {save_dir}")
    np.save(save_dir / f"{tag}_eigenvectors.npy", eigen_vectors)
    np.save(save_dir / f"{tag}_eigenvalues.npy", eigen_values)
    np.save(save_dir / f"{tag}_structure_tensor.npy", S)

    print("Done")


def orientation_analysis(
    eigen_vectors, image_dir_to_save_images,
    red_x_ind=0, green_y_ind=1, blue_z_ind=2,
    save_index=0, preserve_sign=False,
):
    """Create RGB orientation images from eigenvector components."""
    output_dir = Path(image_dir_to_save_images)
    output_dir.mkdir(parents=True, exist_ok=True)

    if preserve_sign:
        eigen_vectors[np.isnan(eigen_vectors)] = -1
        eigen_vectors = (eigen_vectors + 1) / 2
    else:
        eigen_vectors[np.isnan(eigen_vectors)] = 0
        eigen_vectors = abs(eigen_vectors)

    red_x = eigen_vectors[red_x_ind, :, :, :]
    green_y = eigen_vectors[green_y_ind, :, :, :]
    blue_z = eigen_vectors[blue_z_ind, :, :, :]
    vec_rgba_image = np.stack([red_x, green_y, blue_z], axis=-1)
    num_slices = eigen_vectors.shape[-1]

    print("Saving orientation slices as TIFF")
    for slice_index in range(num_slices):
        image_slice = np.multiply(vec_rgba_image[:, :, slice_index, :], 255).astype(np.uint8)
        image_slice = Image.fromarray(image_slice.transpose(1, 0, 2))
        output_path = output_dir / f"orientation_{slice_index + save_index:05d}.tif"
        image_slice.save(str(output_path))

    print("Done")


def show_single_odf(odf_values, sphere, min_pix=1000, interactive=False, out_path=None):
    """Render a single ODF glyph using FURY."""
    if odf_values.ndim != 1:
        raise ValueError("odf_values must have shape (n_points,)")

    slicer_args = {'norm': False, 'colormap': None, 'scale': 0.5}
    size = (int(min_pix), int(min_pix))

    odf_4d = np.expand_dims(odf_values, (0, 1, 2))
    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    odf_actor = actor.odf_slicer(odf_4d, sphere=sphere, **slicer_args)
    scene.add(odf_actor)
    scene.reset_camera_tight(margin_factor=1.5)

    if interactive:
        window.show(scene, size=size, reset_camera=False)
    elif out_path is not None:
        window.record(scene, size=size, reset_camera=False, out_path=str(out_path))
    return scene


def compute_binned_fodfs(
    eigen_vector_binned, vessel_mask_binned,
    odf_array, odf_coef, vessel_volume, vessel_frac,
    save_dir, degree, sphere,
):
    """Compute per-supervoxel fODFs with vessel volume/fraction tracking.

    Parameters
    ----------
    eigen_vector_binned : ndarray, shape (3, n_sv_x, vr, n_sv_y, vr, n_sv_z, vr)
    vessel_mask_binned : ndarray, shape (n_sv_x, vr, n_sv_y, vr, n_sv_z, vr)
    odf_array : ndarray, output for ODF on sphere
    odf_coef : ndarray, output for SH coefficients
    vessel_volume : ndarray, output for vessel voxel counts
    vessel_frac : ndarray, output for vessel volume fractions
    save_dir : Path
    degree : int
    sphere : fiberorient Sphere
    """
    save_dir = Path(save_dir)

    for z in range(eigen_vector_binned.shape[5]):
        for y in range(eigen_vector_binned.shape[3]):
            for x in range(eigen_vector_binned.shape[1]):
                print(f"Processing supervoxel ({x}, {y}, {z})")

                sv_evec = eigen_vector_binned[:, x, :, y, :, z, :]
                vec_roi = np.transpose(sv_evec, (1, 2, 3, 0))
                vec_roi_shape = vec_roi.shape

                if vessel_mask_binned is not None:
                    sv_vessel_mask = vessel_mask_binned[x, :, y, :, z, :]

                # Handle NaN/Inf in eigenvectors
                if np.any(np.isnan(vec_roi)) or np.any(np.isinf(vec_roi)):
                    mask = np.ones(vec_roi_shape[:-1], dtype=bool)
                    for dim in range(3):
                        mask[np.isnan(vec_roi[..., dim])] = False
                        mask[np.isinf(vec_roi[..., dim])] = False

                    nan_inf_fraction = 1 - np.sum(mask) / mask.size
                    print(f"  NaN/Inf fraction: {nan_inf_fraction * 100:.2f}%")
                    vec_roi = vec_roi[mask]

                    if vec_roi.size == 0:
                        vec_roi = np.random.randn(*vec_roi_shape).astype(np.float32)

                odf = ODF(degree, method='precompute').fit(vec_roi)
                odf2sphere = odf.to_sphere(sphere)
                odf_array[x, y, :, z] = odf2sphere
                odf_coef[x, y, :, z] = odf.coef

                if vessel_mask_binned is not None:
                    vessel_volume[x, y, z] = np.sum(sv_vessel_mask)
                    vessel_frac[x, y, z] = np.sum(sv_vessel_mask) / sv_vessel_mask.size

        # Save ODF images for this z-slice
        print(f"Saving ODF images for z-slice {z}")
        odf_dir = save_dir / f"odfs_{eigen_vector_binned.shape[1]}_{eigen_vector_binned.shape[3]}_{z}"
        odf_dir.mkdir(parents=True, exist_ok=True)

        for xod in range(odf_array.shape[0]):
            for yod in range(odf_array.shape[1]):
                show_single_odf(
                    odf_array[xod, yod, :, z], sphere,
                    out_path=odf_dir / f"fodf_{xod}_{yod}_{z}.png",
                )

    # Save ODF arrays and coefficients
    print("Saving ODF arrays and coefficients")
    np.save(save_dir / "odf_array.npy", odf_array)
    np.save(save_dir / "odf_coef.npy", odf_coef)

    if vessel_mask_binned is not None:
        print("Saving vessel volume and fraction")
        np.save(save_dir / "vessel_volume.npy", vessel_volume)
        np.save(save_dir / "vessel_frac.npy", vessel_frac)

    print("Done")


def _load_eigen_files(eigen_dir):
    """Load eigenvalue and eigenvector .npy files from a directory."""
    eigen_dir = Path(eigen_dir)
    assert eigen_dir.exists(), f"Directory {eigen_dir} does not exist"

    evec_files = natsorted([f for f in os.listdir(eigen_dir) if f.endswith('eigenvectors.npy')])
    eval_files = natsorted([f for f in os.listdir(eigen_dir) if f.endswith('eigenvalues.npy')])
    print(f"Found {len(evec_files)} eigenvector files, {len(eval_files)} eigenvalue files")

    eigen_vector = np.load(eigen_dir / evec_files[0])
    eigen_value = np.load(eigen_dir / eval_files[0])
    print(f"Eigenvector shape: {eigen_vector.shape}")
    print(f"Eigenvalue shape: {eigen_value.shape}")
    return eigen_vector, eigen_value


def _bin_eigenvectors(eigen_vector, voxel_ratio, rgb_order=(0, 2, 1)):
    """Crop, reorder RGB channels, and reshape eigenvectors into supervoxels."""
    remainder_x = eigen_vector.shape[1] % voxel_ratio
    remainder_y = eigen_vector.shape[2] % voxel_ratio
    remainder_z = eigen_vector.shape[3] % voxel_ratio
    print(f"Remainder voxels XYZ: {remainder_x}, {remainder_y}, {remainder_z}")

    slices = [
        slice(None, -remainder_x if remainder_x else None),
        slice(None, -remainder_y if remainder_y else None),
        slice(None, -remainder_z if remainder_z else None),
    ]
    evec_cropped = eigen_vector[:, slices[0], slices[1], slices[2]]
    print(f"Shape after cropping: {evec_cropped.shape}")

    # Reorder RGB channels
    evec_reordered = np.stack([
        evec_cropped[rgb_order[0]],
        evec_cropped[rgb_order[1]],
        evec_cropped[rgb_order[2]],
    ], axis=0)

    # Reshape into supervoxels: (3, n_sv_x, vr, n_sv_y, vr, n_sv_z, vr)
    evec_binned = evec_reordered.reshape(
        3,
        evec_reordered.shape[1] // voxel_ratio, voxel_ratio,
        evec_reordered.shape[2] // voxel_ratio, voxel_ratio,
        evec_reordered.shape[3] // voxel_ratio, voxel_ratio,
    )
    print(f"Supervoxel shape (3, n_sv_x, vr, n_sv_y, vr, n_sv_z, vr): {evec_binned.shape}")
    return evec_binned, (remainder_x, remainder_y, remainder_z)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration ----
    SIGMA = 1.5
    RHO = 3
    EXTENSION = ".tiff"
    HIPCT_RESOLUTION = 6.54  # µm
    DMRI_RESOLUTION = 800    # µm
    SH_DEGREE = 8
    N_SPHERE_POINTS = 6500
    RGB_ORDER = (0, 2, 1)    # eigenvector component → RGB channel mapping

    IMAGE_DIR = Path("../6.54um_LADAF-2021-17_brain_ROI-03_pag-0.03_0.07_jp2_1392_2575_8000")
    VESSEL_MASK_DIR = Path("../processed_masks")

    SAMPLE_TAG = f"6.54um_LADAF-2021-17_brain_ROI-03_pag-0.03_0.07_jp2_1392_2575_8000_{SIGMA}_dot{RHO}"
    UNMASKED_DIR = Path(SAMPLE_TAG)
    MASKED_DIR = Path(f"gradient_masked_{SAMPLE_TAG}")

    # ---- Stage flags ----
    RUN_STA_UNMASKED = True
    RUN_STA_GRADIENT_MASKED = True
    RUN_ORIENTATION_AND_FA = True
    RUN_SUPERVOXEL_FODF = True
    RUN_GLOBAL_FODF = True

    # ======================================================================
    # Stage 1: STA without vessel mask
    # ======================================================================
    if RUN_STA_UNMASKED:
        print("\n" + "=" * 60)
        print("Stage 1: STA without vessel mask")
        print("=" * 60)

        assert IMAGE_DIR.exists(), f"Image directory {IMAGE_DIR} does not exist"

        image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(EXTENSION)]
        image_files = natsorted(image_files)
        print(f"Number of image files: {len(image_files)}")

        sta_analysis(
            image_dir=str(IMAGE_DIR),
            mask_dir=None,
            extension=EXTENSION,
            start_index=0,
            end_index=len(image_files),
            sigma=SIGMA,
            rho=RHO,
            save_dir=UNMASKED_DIR,
        )

    # ======================================================================
    # Stage 2: STA with gradient-level vessel mask
    # ======================================================================
    if RUN_STA_GRADIENT_MASKED:
        print("\n" + "=" * 60)
        print("Stage 2: STA with gradient-level vessel mask")
        print("=" * 60)

        assert IMAGE_DIR.exists(), f"Image directory {IMAGE_DIR} does not exist"
        assert VESSEL_MASK_DIR.exists(), f"Mask directory {VESSEL_MASK_DIR} does not exist"

        image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(EXTENSION)]
        image_files = natsorted(image_files)
        print(f"Number of image files: {len(image_files)}")

        sta_analysis(
            image_dir=str(IMAGE_DIR),
            mask_dir=str(VESSEL_MASK_DIR),
            extension=EXTENSION,
            start_index=0,
            end_index=len(image_files),
            sigma=SIGMA,
            rho=RHO,
            save_dir=MASKED_DIR,
        )

    # ======================================================================
    # Stage 3: RGB orientation images + FA maps
    # ======================================================================
    if RUN_ORIENTATION_AND_FA:
        print("\n" + "=" * 60)
        print("Stage 3: Orientation and FA analysis")
        print("=" * 60)

        # --- Unmasked ---
        print("\n--- Unmasked eigenvectors ---")
        eigen_vector, eigen_value = _load_eigen_files(UNMASKED_DIR)
        assert validate_eigenvalue_order(eigen_value), \
            "Eigenvalues are not ordered λ₁ ≥ λ₂ ≥ λ₃"

        orientation_dir = Path(f"{SAMPLE_TAG}_orientation")
        orientation_analysis(
            eigen_vector, orientation_dir,
            red_x_ind=RGB_ORDER[0], green_y_ind=RGB_ORDER[1], blue_z_ind=RGB_ORDER[2],
        )

        FA = compute_FA(eigenvalues=eigen_value)
        print(f"FA shape: {FA.shape}")
        odf_dir = Path(f"{SAMPLE_TAG}_odf_analysis")
        odf_dir.mkdir(parents=True, exist_ok=True)
        np.save(odf_dir / "FA.npy", FA)
        fa_dir = Path(f"{SAMPLE_TAG}_FA")
        fa_dir.mkdir(parents=True, exist_ok=True)
        for slice_index in range(FA.shape[-1]):
            image_slice = np.multiply(FA[:, :, slice_index], 255).astype(np.uint8)
            image_slice = Image.fromarray(image_slice.transpose(1, 0))
            image_slice.save(str(fa_dir / f"FA_{slice_index:05d}.tif"))
        print("Saved unmasked FA images")

        # --- Masked ---
        print("\n--- Masked eigenvectors ---")
        eigen_vector_masked, eigen_value_masked = _load_eigen_files(MASKED_DIR)

        assert not np.allclose(eigen_vector, eigen_vector_masked), \
            "Eigen vector files are identical — masking had no effect"

        masked_orientation_dir = Path(f"gradient_masked_{SAMPLE_TAG}_orientation")
        orientation_analysis(
            eigen_vector_masked, masked_orientation_dir,
            red_x_ind=RGB_ORDER[0], green_y_ind=RGB_ORDER[1], blue_z_ind=RGB_ORDER[2],
        )

        FA_masked = compute_FA(eigenvalues=eigen_value_masked)
        print(f"Masked FA shape: {FA_masked.shape}")
        masked_odf_dir = Path(f"gradient_masked_{SAMPLE_TAG}_odf_analysis")
        masked_odf_dir.mkdir(parents=True, exist_ok=True)
        np.save(masked_odf_dir / "masked_FA.npy", FA_masked)
        masked_fa_dir = Path(f"gradient_masked_{SAMPLE_TAG}_FA")
        masked_fa_dir.mkdir(parents=True, exist_ok=True)
        for slice_index in range(FA_masked.shape[-1]):
            image_slice = np.multiply(FA_masked[:, :, slice_index], 255).astype(np.uint8)
            image_slice = Image.fromarray(image_slice.transpose(1, 0))
            image_slice.save(str(masked_fa_dir / f"FA_{slice_index:05d}.tif"))
        print("Saved masked FA images")

    # ======================================================================
    # Stage 4: Supervoxel fODF estimation
    # ======================================================================
    if RUN_SUPERVOXEL_FODF:
        print("\n" + "=" * 60)
        print("Stage 4: Supervoxel fODF estimation")
        print("=" * 60)

        voxel_ratio = get_voxel_ratio(HIPCT_RESOLUTION, DMRI_RESOLUTION)

        # Load eigenvectors
        eigen_vector, _ = _load_eigen_files(UNMASKED_DIR)
        eigen_vector_masked, _ = _load_eigen_files(MASKED_DIR)

        # Bin into supervoxels
        eigen_vector_binned, remainders = _bin_eigenvectors(
            eigen_vector, voxel_ratio, rgb_order=RGB_ORDER,
        )
        masked_eigen_vector_binned, _ = _bin_eigenvectors(
            eigen_vector_masked, voxel_ratio, rgb_order=RGB_ORDER,
        )
        assert not np.allclose(eigen_vector_binned, masked_eigen_vector_binned), \
            "Binned eigen vectors are identical"

        # Load and bin vessel mask
        print(f"\nLoading vessel mask from {VESSEL_MASK_DIR}")
        assert VESSEL_MASK_DIR.exists(), f"Vessel mask directory does not exist"
        mask_files = natsorted([f for f in os.listdir(VESSEL_MASK_DIR) if f.endswith(EXTENSION)])
        print(f"Number of mask files: {len(mask_files)}")

        vessel_mask = [
            np.array(Image.open(os.path.join(VESSEL_MASK_DIR, f))).transpose(1, 0)
            for f in mask_files
        ]
        vessel_mask = np.stack(vessel_mask, axis=-1)
        print(f"Vessel mask shape (x, y, z): {vessel_mask.shape}")

        remainder_x, remainder_y, remainder_z = remainders
        slices = [
            slice(None, -remainder_x if remainder_x else None),
            slice(None, -remainder_y if remainder_y else None),
            slice(None, -remainder_z if remainder_z else None),
        ]
        vessel_mask_binned = vessel_mask[slices[0], slices[1], slices[2]]
        vessel_mask_binned = vessel_mask_binned.reshape(
            vessel_mask_binned.shape[0] // voxel_ratio, voxel_ratio,
            vessel_mask_binned.shape[1] // voxel_ratio, voxel_ratio,
            vessel_mask_binned.shape[2] // voxel_ratio, voxel_ratio,
        )
        print(f"Binned vessel mask shape: {vessel_mask_binned.shape}")

        # SH setup
        sphere = make_sphere(N_SPHERE_POINTS)
        n_coefs = int((SH_DEGREE + 1) * (SH_DEGREE + 2) / 2)
        n_sv_x = eigen_vector_binned.shape[1]
        n_sv_y = eigen_vector_binned.shape[3]
        n_sv_z = eigen_vector_binned.shape[5]
        print(f"SH coefficients: {n_coefs}, Supervoxels: ({n_sv_x}, {n_sv_y}, {n_sv_z})")

        # --- Unmasked fODFs ---
        print("\n--- Computing unmasked supervoxel fODFs ---")
        odf_analysis_dir = Path(f"{SAMPLE_TAG}_odf_analysis")
        odf_array = np.zeros((n_sv_x, n_sv_y, sphere.phi.size, n_sv_z), dtype=np.float32)
        odf_coef = np.zeros((n_sv_x, n_sv_y, n_coefs, n_sv_z), dtype=np.float32)
        vessel_volume = np.zeros((n_sv_x, n_sv_y, n_sv_z), dtype=np.float32)
        vessel_frac = np.zeros((n_sv_x, n_sv_y, n_sv_z), dtype=np.float32)

        compute_binned_fodfs(
            eigen_vector_binned=eigen_vector_binned,
            vessel_mask_binned=vessel_mask_binned,
            odf_array=odf_array,
            odf_coef=odf_coef,
            vessel_volume=vessel_volume,
            vessel_frac=vessel_frac,
            save_dir=odf_analysis_dir,
            degree=SH_DEGREE,
            sphere=sphere,
        )

        # --- Masked fODFs ---
        print("\n--- Computing masked supervoxel fODFs ---")
        masked_odf_analysis_dir = Path(f"gradient_masked_{SAMPLE_TAG}_odf_analysis")
        masked_odf_array = np.zeros_like(odf_array)
        masked_odf_coef = np.zeros_like(odf_coef)
        masked_vessel_volume = np.zeros_like(vessel_volume)
        masked_vessel_frac = np.zeros_like(vessel_frac)

        compute_binned_fodfs(
            eigen_vector_binned=masked_eigen_vector_binned,
            vessel_mask_binned=vessel_mask_binned,
            odf_array=masked_odf_array,
            odf_coef=masked_odf_coef,
            vessel_volume=masked_vessel_volume,
            vessel_frac=masked_vessel_frac,
            save_dir=masked_odf_analysis_dir,
            degree=SH_DEGREE,
            sphere=sphere,
        )

    # ======================================================================
    # Stage 5: Global fODF (whole VOI)
    # ======================================================================
    if RUN_GLOBAL_FODF:
        print("\n" + "=" * 60)
        print("Stage 5: Global fODF estimation")
        print("=" * 60)

        sphere = make_sphere(N_SPHERE_POINTS)

        # --- Unmasked ---
        print("\n--- Unmasked global fODF ---")
        eigen_vector, _ = _load_eigen_files(UNMASKED_DIR)
        eigen_vector = np.transpose(eigen_vector, (1, 2, 3, 0))
        if np.any(np.isnan(eigen_vector)) or np.any(np.isinf(eigen_vector)):
            eigen_vector = drop_nans_infs(eigen_vector)

        odf_unmasked = ODF(degree=SH_DEGREE, method='precompute').fit(eigen_vector)
        show_odf(odf_unmasked, sphere, interactive=False, out_path="pons_voi_odf_before_masking.tif")
        print(f"Global fODF coefficients shape: {odf_unmasked.coef.shape}")
        unmasked_odf_dir = Path(f"{SAMPLE_TAG}_odf_analysis")
        unmasked_odf_dir.mkdir(parents=True, exist_ok=True)
        np.save(unmasked_odf_dir / "whole_voi_odf_coef.npy", odf_unmasked.coef)

        # --- Masked ---
        print("\n--- Masked global fODF ---")
        eigen_vector_masked, _ = _load_eigen_files(MASKED_DIR)
        eigen_vector_masked = np.transpose(eigen_vector_masked, (1, 2, 3, 0))
        if np.any(np.isnan(eigen_vector_masked)) or np.any(np.isinf(eigen_vector_masked)):
            eigen_vector_masked = drop_nans_infs(eigen_vector_masked)

        odf_masked = ODF(degree=SH_DEGREE, method='precompute').fit(eigen_vector_masked)
        show_odf(odf_masked, sphere, interactive=False, out_path="pons_voi_odf_after_masking.tif")
        print(f"Masked global fODF coefficients shape: {odf_masked.coef.shape}")
        masked_odf_dir = Path(f"gradient_masked_{SAMPLE_TAG}_odf_analysis")
        masked_odf_dir.mkdir(parents=True, exist_ok=True)
        np.save(masked_odf_dir / "whole_voi_odf_coef.npy", odf_masked.coef)
