#!/usr/bin/env python
"""Cross-modal dMRI-to-HiP-CT registration pipeline.

Prepares HiP-CT and dMRI volumes for registration, runs ANTsPy SyNRA
alignment, and warps dMRI volumes into HiP-CT space.  Six independent
stages are controlled by boolean flags:

1. HiP-CT OME-Zarr extraction -- extract level 4 from OME-Zarr, save
   as NIfTI and JP2 slices (for organ-masker input)
2. Level-4 mask application -- apply organ-masker masks to level-4 JP2
   slices
3. Masked volume normalization -- read masked slices, percentile-
   normalize, save NIfTI and NIfTI-Zarr
4. dMRI b-value extraction -- compute b0, b4000, b10000 means and
   b_high/b0 ratio map for T1-like contrast
5. ANTsPy registration -- SyNRA registration with MI-only or hybrid
   MI+CC metrics, starting from a manual ITK-SNAP alignment
6. Volume warping -- apply forward transforms to dMRI volumes and
   convert to NIfTI-Zarr for Neuroglancer visualization
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ants
import cv2
import nibabel as nib
import numpy as np
import zarr
from dask import compute, delayed
from natsort import natsorted
from niizarr import nii2zarr, zarr2nii
from ome_zarr_models.v04 import Image
from tqdm import tqdm

# import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def save_as_jp2(zarr_array, save_path, idx, cratio=10, update_bar=None):
    """Save one z-slice from a ZYX zarr array as a JPEG2000 file.

    Parameters
    ----------
    zarr_array : zarr.Array
        Source volume (ZYX layout).
    save_path : Path
        Output directory.
    idx : int
        Z-index to extract.
    cratio : int
        Compression ratio (1 = lossless, 10 = 10x compression).
    update_bar : callable or None
        Progress callback.
    """
    slice_data = zarr_array[idx, :, :]
    params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(1000 / cratio)]
    cv2.imwrite(str(save_path / f"slice_{idx:04d}.jp2"), slice_data, params)
    if update_bar is not None:
        update_bar()


def bhigh_b0_ratio(bhigh_vol, b0_vol, background_threshold_pct=5,
                   brightness_percentile=99.9):
    """Compute b_high / b0 ratio map with background thresholding.

    Voxels where b0 falls below a data-driven threshold (indicating
    air/background) are set to zero in the output.

    Parameters
    ----------
    bhigh_vol, b0_vol : ndarray
        Volumes at the high b-value shell and b=0.
    background_threshold_pct : float
        Percentage of the b0 brightness percentile used as mask threshold.
    brightness_percentile : float
        Percentile of b0 used to determine the intensity scale.

    Returns
    -------
    ratio_map : ndarray, float32
    """
    background_threshold_pct = background_threshold_pct / 100
    bhigh_vol = bhigh_vol.astype(np.float32)
    b0_vol = b0_vol.astype(np.float32)

    bhigh_vol[bhigh_vol < 0] = 0
    b0_vol[b0_vol < 0] = 0

    mask_threshold = np.percentile(b0_vol, brightness_percentile) * background_threshold_pct
    print(f"Background threshold (based on b0 volume): {mask_threshold}")
    valid_mask = b0_vol > mask_threshold

    ratio_map = np.zeros_like(bhigh_vol, dtype=np.float32)
    np.divide(bhigh_vol, b0_vol, where=valid_mask, out=ratio_map)

    ratio_map = np.clip(ratio_map, 0, None)
    ratio_map = np.nan_to_num(ratio_map, nan=0.0, posinf=0.0, neginf=0.0)
    return ratio_map


def apply_mask_slice(i, image_files, mask_files, image_dir, mask_dir,
                     output_dir, cratio=1):
    """Apply a binary mask to one image slice and save as JP2."""
    img = cv2.imread(str(Path(image_dir) / image_files[i]), -1)
    mask = cv2.imread(str(Path(mask_dir) / mask_files[i]), -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(1000 / cratio)]
    cv2.imwrite(str(Path(output_dir) / image_files[i]), masked, params)


def read_masked_image_slice(idx, masked_image_files, update_bar=None):
    """Read one masked JP2 slice and transpose from YX to XY."""
    masked_image = cv2.imread(str(masked_image_files[idx]), -1)
    masked_image = masked_image.T
    if update_bar is not None:
        update_bar()
    return masked_image


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration ----
    USE_MI_ONLY = False  # True = MI-only (~11min); False = hybrid MI+CC (~110min)

    BASE_PATH = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/registration_new_dmri")
    HIPCT_OME_ZARR_PATH = Path("/media/eric/eric_backup_03/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/sub-I58_sample-01_chunk-01_hipCT.ome.zarr")
    HIPCT_LEVEL = 4

    DMRI_NIFTI = "sub-I58_sample-hemi_desc-preproc_dwi.nii.gz"
    DMRI_BVALS = "sub-I58_sample-hemi_desc-preproc_dwi.bvals"
    DMRI_BVECS = "sub-I58_sample-hemi_desc-preproc_dwi.bvecs"

    INITIAL_TRANSFORM = "initial_bhighb0_to_hipct_transform.txt"

    NORM_PMIN = 83
    NORM_PMAX = 99

    # ---- Stage flags ----
    RUN_HIPCT_EXTRACTION = True
    RUN_LEVEL4_MASKING = True
    RUN_MASKED_VOLUME_NORMALIZATION = True
    RUN_DMRI_BVALUE_EXTRACTION = True
    RUN_REGISTRATION = True
    RUN_VOLUME_WARPING = True

    # ---- Derived names ----
    TRANSFORM_PREFIX = (
        "bhigh_b0_ratio_map_registered_to_hipct_level4_synra_mi_only"
        if USE_MI_ONLY else
        "bhigh_b0_ratio_map_registered_to_hipct_level4_synra_hybrid"
    )
    REG_PREFIX = "synra_mi_only" if USE_MI_ONLY else "synra_hybrid"
    HIPCT_NORM_FNAME = f"hipct_hemi_downsampled_level_4_masked_norm_percentile_{NORM_PMIN}_{NORM_PMAX}.nii.gz"

    # ======================================================================
    # Stage 1: Extract HiP-CT level 4 from OME-Zarr
    # ======================================================================
    if RUN_HIPCT_EXTRACTION:
        print("\n" + "=" * 60)
        print("Stage 1: HiP-CT OME-Zarr extraction")
        print("=" * 60)

        hipct_ome_group = zarr.open_group(store=str(HIPCT_OME_ZARR_PATH), mode="r")
        multiscale_hipct_image = Image.from_zarr(hipct_ome_group)
        for idx, dataset in enumerate(multiscale_hipct_image.datasets[0]):
            print(f"Level {idx}, path: {dataset.path}, shape: {hipct_ome_group[dataset.path].shape}")

        hipct_hemi_nifti = zarr2nii(inp=hipct_ome_group, level=HIPCT_LEVEL)
        print(f"HiP-CT level {HIPCT_LEVEL} NIfTI shape: {hipct_hemi_nifti.shape}")
        print(f"HiP-CT level {HIPCT_LEVEL} NIfTI affine:\n{hipct_hemi_nifti.affine}")

        zarr2nii(
            inp=hipct_ome_group,
            level=HIPCT_LEVEL,
            out=str(BASE_PATH / f"hipct_hemi_downsampled_level_{HIPCT_LEVEL}.nii.gz"),
        )
        print(f"Saved HiP-CT level {HIPCT_LEVEL} NIfTI")

        # Save level-4 slices as JP2 for organ-masker input
        hipct_image_data = hipct_ome_group[multiscale_hipct_image.datasets[0][HIPCT_LEVEL].path]
        hipct_jp2_dir = BASE_PATH / f"hipct_hemi_downsampled_level_{HIPCT_LEVEL}_jp2_slices"
        hipct_jp2_dir.mkdir(parents=True, exist_ok=True)

        num_slices = hipct_image_data.shape[0]
        print(f"Saving {num_slices} slices as JP2 to {hipct_jp2_dir}")

        with tqdm(total=num_slices, desc="Saving HiP-CT slices as JP2") as bar:
            tasks = [
                delayed(save_as_jp2)(
                    zarr_array=hipct_image_data,
                    save_path=hipct_jp2_dir,
                    idx=slice_idx,
                    cratio=1,
                    update_bar=bar.update,
                )
                for slice_idx in range(num_slices)
            ]
            compute(*tasks, scheduler='threads', num_workers=min(16, os.cpu_count()))

        print("Done saving JP2 slices.")

    # ======================================================================
    # Stage 2: Apply organ-masker masks to level-4 images
    # Run first: organ-masker <hipct_jp2_dir> --model large --fill-holes
    # ======================================================================
    if RUN_LEVEL4_MASKING:
        print("\n" + "=" * 60)
        print("Stage 2: Level-4 mask application")
        print("=" * 60)

        image_dir = str(BASE_PATH / f"hipct_hemi_downsampled_level_{HIPCT_LEVEL}_jp2_slices")
        mask_dir = str(BASE_PATH / f"hipct_hemi_downsampled_level_{HIPCT_LEVEL}_jp2_slices_masks")
        masked_dir = str(BASE_PATH / f"hipct_hemi_downsampled_level_{HIPCT_LEVEL}_jp2_masked_slices")
        Path(masked_dir).mkdir(parents=True, exist_ok=True)

        image_files = natsorted(os.listdir(image_dir))
        mask_files = natsorted(os.listdir(mask_dir))
        dummy_img = cv2.imread(str(Path(image_dir) / image_files[0]), -1)
        dummy_mask = cv2.imread(str(Path(mask_dir) / mask_files[0]), -1)
        assert (len(image_files), *dummy_img.shape) == (len(mask_files), *dummy_mask.shape), \
            "Image and mask stacks have different shapes"

        max_workers = min(16, os.cpu_count() or 4)
        print(f"Applying masks with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [
                exe.submit(apply_mask_slice, i, image_files, mask_files,
                           image_dir, mask_dir, masked_dir)
                for i in range(len(image_files))
            ]
            for f in tqdm(as_completed(futures), total=len(futures)):
                f.result()

        print(f"All masked images saved to: {masked_dir}")

    # ======================================================================
    # Stage 3: Read masked slices, normalize, save as NIfTI + NIfTI-Zarr
    # ======================================================================
    if RUN_MASKED_VOLUME_NORMALIZATION:
        print("\n" + "=" * 60)
        print("Stage 3: Masked volume normalization")
        print("=" * 60)

        masked_image_dir = BASE_PATH / f"hipct_hemi_downsampled_level_{HIPCT_LEVEL}_jp2_masked_slices"
        masked_image_files = natsorted(list(masked_image_dir.glob("*.jp2")))
        num_masked_slices = len(masked_image_files)
        print(f"Found {num_masked_slices} masked slices")

        # Construct affine from OME-Zarr metadata (level 4 voxel size)
        hipct_ome_group = zarr.open_group(store=str(HIPCT_OME_ZARR_PATH), mode="r")
        multiscale_hipct_image = Image.from_zarr(hipct_ome_group)
        voxel_scale = np.array(
            multiscale_hipct_image.datasets[0][HIPCT_LEVEL].coordinateTransformations[0].scale
        ) / 1000  # um -> mm
        masked_affine = np.diag(voxel_scale.tolist() + [1])
        print(f"Voxel size (mm): {voxel_scale}")

        # Read all masked slices in parallel
        print("Reading masked slices...")
        with tqdm(total=num_masked_slices, desc="Reading masked slices") as bar:
            tasks = [
                delayed(read_masked_image_slice)(
                    idx=slice_idx,
                    masked_image_files=masked_image_files,
                    update_bar=bar.update,
                )
                for slice_idx in range(num_masked_slices)
            ]
            masked_slices = compute(*tasks, scheduler='threads',
                                    num_workers=min(16, os.cpu_count()))

        masked_volume = np.stack(masked_slices, axis=-1)  # XYZ
        print(f"Masked volume shape: {masked_volume.shape}, dtype: {masked_volume.dtype}")

        # Percentile normalization
        masked_volume = masked_volume.astype(np.float32)
        vmin = np.percentile(masked_volume, NORM_PMIN).astype(np.float32)
        vmax = np.percentile(masked_volume, NORM_PMAX).astype(np.float32)
        print(f"Clipping intensities to [{vmin}, {vmax}] (percentile {NORM_PMIN}-{NORM_PMAX})")
        hipct_data_clipped = np.clip(
            (masked_volume - vmin) / (vmax - vmin), 0, 1
        ).astype(np.float32)
        print(f"Normalized volume: shape={hipct_data_clipped.shape}, "
              f"min={hipct_data_clipped.min():.3f}, max={hipct_data_clipped.max():.3f}")

        # Save as NIfTI
        nib.save(
            nib.Nifti1Image(hipct_data_clipped, affine=masked_affine),
            str(BASE_PATH / HIPCT_NORM_FNAME),
        )
        print(f"Saved normalized NIfTI: {HIPCT_NORM_FNAME}")

        # Verify shape matches unmasked NIfTI
        unmasked_fname = HIPCT_NORM_FNAME.split("_masked_")[0] + ".nii.gz"
        unmasked_nifti = nib.load(str(BASE_PATH / unmasked_fname))
        masked_nifti = nib.load(str(BASE_PATH / HIPCT_NORM_FNAME))
        assert unmasked_nifti.shape == masked_nifti.shape, \
            f"Masked shape {masked_nifti.shape} != unmasked shape {unmasked_nifti.shape}"
        print(f"Shape verification passed: {masked_nifti.shape}")

        # Save as NIfTI-Zarr for Neuroglancer
        nii2zarr(
            masked_nifti,
            str(BASE_PATH / HIPCT_NORM_FNAME).replace(".nii.gz", ".nii.zarr"),
        )
        print("Saved NIfTI-Zarr")

        # plt.imshow(masked_nifti.get_fdata()[:, :, masked_nifti.shape[2] // 2].T, cmap='bone')

    # ======================================================================
    # Stage 4: Extract dMRI b-value volumes and compute ratio map
    # ======================================================================
    if RUN_DMRI_BVALUE_EXTRACTION:
        print("\n" + "=" * 60)
        print("Stage 4: dMRI b-value extraction")
        print("=" * 60)

        bvals = np.loadtxt(BASE_PATH / DMRI_BVALS)
        bvecs = np.loadtxt(BASE_PATH / DMRI_BVECS).T
        print(f"bvals shape: {bvals.shape}")
        print(f"bvecs shape: {bvecs.shape}")

        dwi_img = nib.load(str(BASE_PATH / DMRI_NIFTI))
        print(f"dMRI NIfTI shape: {dwi_img.shape}")
        print(f"dMRI NIfTI affine:\n{dwi_img.affine}")

        dwi_data = dwi_img.get_fdata()
        affine = dwi_img.affine

        # Extract b-value shell means
        b0_vol = np.clip(dwi_data[..., bvals == 0].mean(axis=-1), 0, None)
        print(f"b0 volume: shape={b0_vol.shape}, range=[{b0_vol.min():.1f}, {b0_vol.max():.1f}]")

        b4000_vol = np.clip(dwi_data[..., bvals == 4000].mean(axis=-1), 0, None)
        print(f"b4000 volume: shape={b4000_vol.shape}, range=[{b4000_vol.min():.1f}, {b4000_vol.max():.1f}]")

        b10000_vol = np.clip(dwi_data[..., bvals == 10000].mean(axis=-1), 0, None)
        print(f"b10000 volume: shape={b10000_vol.shape}, range=[{b10000_vol.min():.1f}, {b10000_vol.max():.1f}]")

        # Compute b10000/b0 ratio map (provides T1-like contrast for registration)
        bhigh_b0_ratio_map = bhigh_b0_ratio(
            b10000_vol, b0_vol, background_threshold_pct=5, brightness_percentile=99.9,
        )
        print(f"b10000/b0 ratio map: shape={bhigh_b0_ratio_map.shape}, "
              f"range=[{bhigh_b0_ratio_map.min():.3f}, {bhigh_b0_ratio_map.max():.3f}]")

        # Save all volumes as NIfTI
        for name, vol in [("b0_mean_volume", b0_vol),
                          ("b4000_mean_volume", b4000_vol),
                          ("b10000_mean_volume", b10000_vol),
                          ("bhigh_b0_ratio_map", bhigh_b0_ratio_map)]:
            nib.save(
                nib.Nifti1Image(vol.astype(np.float32), affine=affine),
                str(BASE_PATH / f"{name}.nii.gz"),
            )
            print(f"Saved {name}.nii.gz")

        # fig, axis = plt.subplots(1, 4, figsize=(20, 5))
        # vol_slice = b0_vol.shape[1] // 2
        # axis[0].imshow(b0_vol[:, vol_slice, :].T, cmap='bone', origin='lower')
        # axis[1].imshow(b4000_vol[:, vol_slice, :].T, cmap='bone', origin='lower')
        # axis[2].imshow(b10000_vol[:, vol_slice, :].T, cmap='bone', origin='lower')
        # axis[3].imshow(bhigh_b0_ratio_map[:, vol_slice, :].T, cmap='bone', origin='lower')
        # plt.show()

    # ======================================================================
    # Stage 5: ANTsPy SyNRA registration
    # ======================================================================
    if RUN_REGISTRATION:
        print("\n" + "=" * 60)
        print("Stage 5: ANTsPy registration")
        print("=" * 60)

        moving = ants.image_read(str(BASE_PATH / "bhigh_b0_ratio_map.nii.gz"))
        print(f"Moving (b_high/b0 ratio) shape: {moving.shape}")

        fixed = ants.image_read(str(BASE_PATH / HIPCT_NORM_FNAME))
        print(f"Fixed (HiP-CT level {HIPCT_LEVEL}) shape: {fixed.shape}")

        initial_transform = str(BASE_PATH / INITIAL_TRANSFORM)

        # Sanity check: apply initial manual alignment
        manual_aligned = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[initial_transform],
            interpolator="linear",
            whichtoinvert=[False],
        )
        print(f"Manual-aligned shape: {manual_aligned.shape}")
        ants.image_write(
            image=manual_aligned,
            filename=str(BASE_PATH / "initial_bhigh_b0_ratio_map_itksnap_aligned.nii.gz"),
        )

        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # axs[0].imshow(manual_aligned.numpy()[:, :, manual_aligned.shape[2] // 2].T, cmap='bone')
        # axs[1].imshow(fixed.numpy()[:, :, fixed.shape[2] // 2].T, cmap='bone')
        # plt.show()

        # Run registration
        if USE_MI_ONLY:
            print("Running Mutual Information only registration...")
            ants_reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform="SyNRA",
                initial_transform=initial_transform,
                aff_metric='mattes',
                syn_metric='mattes',
                aff_sampling=32,
                syn_sampling=32,
                aff_iterations=[2100, 1200, 1200, 10],
                reg_iterations=[100, 70, 50, 10],
                grad_step=0.2,
                flow_sigma=3.0,
                total_sigma=0,
                use_legacy_histogram_matching=False,
                verbose=True,
            )
        else:
            print("Running hybrid registration (MI affine + CC SyN)...")
            ants_reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform="SyNRA",
                initial_transform=initial_transform,
                # Affine stage: Mutual Information (robust global scaling)
                aff_metric='mattes',
                # SyN stage: Cross-Correlation (precise local alignment)
                syn_metric='CC',
                # For MI: 32 histogram bins; for CC: 4-voxel radius
                aff_sampling=32,
                syn_sampling=4,
                aff_iterations=[2100, 1200, 1200, 10],
                reg_iterations=[100, 70, 50, 20],
                grad_step=0.2,
                flow_sigma=3,
                total_sigma=0,
                use_legacy_histogram_matching=False,
                verbose=True,
            )

        # Save registered image
        ants.image_write(
            image=ants_reg['warpedmovout'],
            filename=str(BASE_PATH / f"{TRANSFORM_PREFIX}.nii.gz"),
        )

        # Save forward transforms
        fwd_warp_tmp = ants_reg['fwdtransforms'][0]
        fwd_affine_tmp = ants_reg['fwdtransforms'][1]
        shutil.copy(fwd_warp_tmp, str(BASE_PATH / f"fwd_{TRANSFORM_PREFIX}_warp.nii.gz"))
        shutil.copy(fwd_affine_tmp, str(BASE_PATH / f"fwd_{TRANSFORM_PREFIX}_affine.mat"))

        # Save inverse transforms
        inv_affine_tmp = ants_reg['invtransforms'][0]
        inv_warp_tmp = ants_reg['invtransforms'][1]
        shutil.copy(inv_affine_tmp, str(BASE_PATH / f"inv_{TRANSFORM_PREFIX}_affine.mat"))
        shutil.copy(inv_warp_tmp, str(BASE_PATH / f"inv_{TRANSFORM_PREFIX}_warp.nii.gz"))

        print("Saved all transforms")

        # slice_idx = int(ants_reg['warpedmovout'].shape[2] // 2)
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # axs[0].imshow(ants_reg['warpedmovout'].numpy()[:, :, slice_idx].T, cmap='bone')
        # axs[1].imshow(fixed.numpy()[:, :, slice_idx].T, cmap='bone')
        # plt.show()

    # ======================================================================
    # Stage 6: Warp dMRI volumes to HiP-CT space + NIfTI-Zarr conversion
    # ======================================================================
    if RUN_VOLUME_WARPING:
        print("\n" + "=" * 60)
        print(f"Stage 6: Volume warping ({REG_PREFIX})")
        print("=" * 60)

        hipct_ants = ants.image_read(str(BASE_PATH / HIPCT_NORM_FNAME))
        print(f"HiP-CT level {HIPCT_LEVEL} shape: {hipct_ants.shape}")

        fwd_warp = str(BASE_PATH / f"fwd_{TRANSFORM_PREFIX}_warp.nii.gz")
        fwd_affine = str(BASE_PATH / f"fwd_{TRANSFORM_PREFIX}_affine.mat")
        fwd_transform_chain = [fwd_warp, fwd_affine]

        for bval_name in ["b0", "b4000", "b10000"]:
            vol_path = str(BASE_PATH / f"{bval_name}_mean_volume.nii.gz")
            out_name = f"{bval_name}_mean_volume_registered_to_hipct_level4_{REG_PREFIX}"

            print(f"\nWarping {bval_name} to HiP-CT space...")
            vol_ants = ants.image_read(vol_path)
            print(f"{bval_name} shape: {vol_ants.shape}")

            warped = ants.apply_transforms(
                fixed=hipct_ants,
                moving=vol_ants,
                transformlist=fwd_transform_chain,
                interpolator="bSpline",
            )
            print(f"{bval_name} warped shape: {warped.shape}")

            # Save as NIfTI
            ants.image_write(
                image=warped,
                filename=str(BASE_PATH / f"{out_name}.nii.gz"),
            )

            # Convert to NIfTI-Zarr for Neuroglancer
            registered_nifti = nib.load(str(BASE_PATH / f"{out_name}.nii.gz"))
            nii2zarr(registered_nifti, str(BASE_PATH / f"{out_name}.nii.zarr"))
            print(f"Saved {out_name}.nii.gz + .nii.zarr")

        print("\nAll volumes warped and converted.")
