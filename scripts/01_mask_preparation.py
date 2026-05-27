#!/usr/bin/env python
"""Upsample organ masks and apply them to HiP-CT image stacks.

Downsampled binary masks are produced by the organ-masker tool
(https://github.com/HiPCTProject/organ-masker), which runs SAM2 on a
4×-downsampled volume.  This script upsamples those masks back to the
native resolution using nearest-neighbour replication with
neighbour-averaged smoothing, then applies the masks to the original
(or level-4 downsampled) image slices.

Four independent scenarios are controlled by boolean flags:

1. Brain mask upsampling + application to original JP2 images
2. Mask application to level-4 downsampled images
3. White-matter mask upsampling (no image masking)
4. Stopping-criterion cleaning mask upsampling to Zarr (masks out
   streak-artefact regions to prevent spurious fiber tractography)
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import zarr
from dask import compute, delayed
from natsort import natsorted
from numcodecs import Blosc
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_original_shape(image_dir, image_files=None):
    """Return (z, y, x) shape of the full image stack."""
    if image_files is None:
        image_files = natsorted(os.listdir(image_dir))
    first = cv2.imread(str(Path(image_dir) / image_files[0]), -1)
    shape = [len(image_files), *first.shape]
    print(f"Original shape (z, y, x): {shape}")
    return shape


def load_downsampled_masks(mask_dir):
    """Load all mask slices from a directory into a (z, y, x) uint8 volume."""
    files = natsorted(os.listdir(mask_dir))
    stack = np.stack(
        [cv2.imread(str(Path(mask_dir) / f), -1) for f in files], axis=0,
    ).astype(np.uint8)
    print(f"Downsampled mask shape: {stack.shape}")
    return stack


def process_and_upsample_slice(i, downsampled_stack, factor, original_shape,
                                output_dir, kernel):
    """Upsample one z-index from the downsampled mask and write output slices.

    Averages the current slice with its z-neighbours (when available),
    upsamples in XY via nearest-neighbour replication, thresholds,
    dilates, and writes ``factor`` identical output slices to simulate
    z-axis upsampling.
    """
    neighbors = [downsampled_stack[i]]
    if i > 0:
        neighbors.append(downsampled_stack[i - 1])
    if i < downsampled_stack.shape[0] - 1:
        neighbors.append(downsampled_stack[i + 1])

    upsampled = []
    for sl in neighbors:
        up = np.repeat(sl, factor, axis=0)
        up = np.repeat(up, factor, axis=1)
        up = up[:original_shape[1], :original_shape[2]]
        upsampled.append(up)

    avg = np.mean(upsampled, axis=0)
    smoothed = (avg > 0.5).astype(np.uint8)
    dilated = cv2.dilate(smoothed, kernel, iterations=1) * 255
    dilated = cv2.resize(dilated, (original_shape[2], original_shape[1]),
                         interpolation=cv2.INTER_NEAREST)

    start_z = i * factor
    end_z = min((i + 1) * factor, original_shape[0])
    for z in range(start_z, end_z):
        cv2.imwrite(str(Path(output_dir) / f"mask_{z:04d}.png"), dilated)


def upsample_mask_volume(downsampled_stack, factor, original_shape, output_dir):
    """Upsample an entire mask volume in parallel and validate output."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kernel = np.ones((3, 3), np.uint8)

    max_workers = min(16, os.cpu_count() or 4)
    print(f"Upsampling masks with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [
            exe.submit(process_and_upsample_slice, i, downsampled_stack,
                       factor, original_shape, output_dir, kernel)
            for i in range(downsampled_stack.shape[0])
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()

    final_files = natsorted(os.listdir(output_dir))
    final_img = cv2.imread(str(Path(output_dir) / final_files[0]), -1)
    final_shape = [len(final_files), *final_img.shape]
    assert final_shape == original_shape, (
        f"Output shape {final_shape} != expected {original_shape}"
    )
    print(f"All masks saved to: {output_dir}")


def apply_mask_slice(i, image_files, mask_files, image_dir, mask_dir,
                     output_dir, jp2_compression):
    """Apply a binary mask to one image slice and save as JP2."""
    img = cv2.imread(str(Path(image_dir) / image_files[i]), -1)
    mask = cv2.imread(str(Path(mask_dir) / mask_files[i]), -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, jp2_compression]
    cv2.imwrite(str(Path(output_dir) / image_files[i]), masked, params)


def apply_masks_to_images(image_dir, mask_dir, output_dir, jp2_compression):
    """Apply masks to all image slices in parallel."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_files = natsorted(os.listdir(image_dir))
    mask_files = natsorted(os.listdir(mask_dir))

    dummy_img = cv2.imread(str(Path(image_dir) / image_files[0]), -1)
    dummy_mask = cv2.imread(str(Path(mask_dir) / mask_files[0]), -1)
    assert (len(image_files), *dummy_img.shape) == (len(mask_files), *dummy_mask.shape), (
        "Image and mask stacks have different shapes"
    )

    max_workers = min(16, os.cpu_count() or 4)
    print(f"Applying masks with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [
            exe.submit(apply_mask_slice, i, image_files, mask_files,
                       image_dir, mask_dir, output_dir, jp2_compression)
            for i in range(len(image_files))
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()

    print(f"All masked images saved to: {output_dir}")


def sc_mask_postprocess(mask):
    """Morphological close + open cleanup for stopping-criterion masks."""
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return mask


def process_and_upsample_sc_slice(idx, mask_files, factor, desired_shape,
                                   zarr_path, kernel):
    """Upsample one SC mask slice (from file) and write to zarr.

    Reads the mask TIFF from disk, applies morphological cleanup to the
    current and neighbouring slices, upsamples in XY, averages, dilates,
    and writes ``factor`` z-slices to the output zarr volume.
    """
    neighbors = [cv2.imread(str(mask_files[idx]), -1)]
    if idx > 0:
        neighbors.append(cv2.imread(str(mask_files[idx - 1]), -1))
    if idx < len(mask_files) - 1:
        neighbors.append(cv2.imread(str(mask_files[idx + 1]), -1))

    neighbors = [sc_mask_postprocess(m) for m in neighbors]

    upsampled = []
    for sl in neighbors:
        up = np.repeat(sl, factor, axis=0)
        up = np.repeat(up, factor, axis=1)
        up = up[:desired_shape[1], :desired_shape[2]]
        upsampled.append(up)

    avg = np.mean(np.stack(upsampled, axis=0), axis=0)
    smoothed = (avg >= 0.5).astype(np.uint8)
    dilated = cv2.dilate(smoothed, kernel, iterations=1) * 255

    zarr_file = zarr.open(zarr_path, mode='a')
    start_z = idx * factor
    end_z = min((idx + 1) * factor, desired_shape[0])
    for z in range(start_z, end_z):
        zarr_file[z] = dilated


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    UPSAMPLE_FACTOR = 4

    # ---- Scenario flags ----
    RUN_BRAIN_MASK_UPSAMPLING = True
    RUN_LEVEL4_MASKING = True
    RUN_WM_MASK_UPSAMPLING = True
    RUN_SC_MASK_UPSAMPLING = True

    # ======================================================================
    # Scenario 1: Upsample brain masks + apply to original images
    # ======================================================================
    if RUN_BRAIN_MASK_UPSAMPLING:
        print("\n" + "=" * 60)
        print("Scenario 1: Brain mask upsampling + image masking")
        print("=" * 60)

        ORIGINAL_IMAGE_DIR = "/home/eric/Downloads/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_"
        DOWNSAMPLED_MASK_DIR = "/hdd/eric/brain/whole_brain_i58_paper/organ-masker/segmentation_SAM2_60.52um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_/masks"
        BRAIN_MASK_DIR = "/hdd/eric/brain/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masks"
        MASKED_IMAGE_DIR = "/hdd/eric/brain/whole_brain_i58_paper/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked"

        original_shape = get_original_shape(ORIGINAL_IMAGE_DIR)
        mask_stack = load_downsampled_masks(DOWNSAMPLED_MASK_DIR)
        upsample_mask_volume(mask_stack, UPSAMPLE_FACTOR, original_shape, BRAIN_MASK_DIR)
        apply_masks_to_images(ORIGINAL_IMAGE_DIR, BRAIN_MASK_DIR, MASKED_IMAGE_DIR,
                              jp2_compression=1000)

    # ======================================================================
    # Scenario 2: Apply masks to level-4 downsampled images
    # ======================================================================
    if RUN_LEVEL4_MASKING:
        print("\n" + "=" * 60)
        print("Scenario 2: Level-4 image masking")
        print("=" * 60)

        LEVEL4_IMAGE_DIR = "/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/registration/hipct_jp2_level4_slices"
        LEVEL4_MASK_DIR = "/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/registration/hipct_jp2_level4_slices_masks"
        LEVEL4_MASKED_DIR = "/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/registration/hipct_jp2_level4_masked_slices"

        apply_masks_to_images(LEVEL4_IMAGE_DIR, LEVEL4_MASK_DIR, LEVEL4_MASKED_DIR,
                              jp2_compression=100)

    # ======================================================================
    # Scenario 3: Upsample white-matter masks
    # ======================================================================
    if RUN_WM_MASK_UPSAMPLING:
        print("\n" + "=" * 60)
        print("Scenario 3: WM mask upsampling")
        print("=" * 60)

        WM_ORIGINAL_IMAGE_DIR = "/home/eric/Downloads/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_"
        WM_DOWNSAMPLED_MASK_DIR = "/media/eric/ErHDD/wm_mask_closed"
        WM_MASK_DIR = "/hdd/eric/brain/whole_brain_i58_paper/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_wm_masks"
        WM_MAX_SLICES = 10000

        image_files = natsorted(os.listdir(WM_ORIGINAL_IMAGE_DIR))[:WM_MAX_SLICES]
        original_shape = get_original_shape(WM_ORIGINAL_IMAGE_DIR, image_files=image_files)
        mask_stack = load_downsampled_masks(WM_DOWNSAMPLED_MASK_DIR)
        upsample_mask_volume(mask_stack, UPSAMPLE_FACTOR, original_shape, WM_MASK_DIR)

    # ======================================================================
    # Scenario 4: Upsample stopping-criterion cleaning mask to Zarr
    # Masks out streak-artefact regions to prevent spurious fiber
    # tractography (used as part of the stopping criterion in step 05).
    # ======================================================================
    if RUN_SC_MASK_UPSAMPLING:
        print("\n" + "=" * 60)
        print("Scenario 4: SC mask upsampling to Zarr")
        print("=" * 60)

        SC_IMAGE_DIR = Path("15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked")
        SC_DOWNSAMPLED_MASK_DIR = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/60.52um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_stopping_criteria_cleaning_mask")
        SC_MASK_ZARR_PATH = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_stopping_criteria_cleaning_mask.zarr")

        assert SC_IMAGE_DIR.exists(), f"Image directory {SC_IMAGE_DIR} does not exist!"

        image_files_sc = list(natsorted(SC_IMAGE_DIR.glob("*.jp2")))
        sample_img = cv2.imread(str(image_files_sc[0]), -1)
        Y, X = sample_img.shape
        desired_shape_zyx = (len(image_files_sc), Y, X)
        print(f"Desired shape (Z, Y, X): {desired_shape_zyx}")

        sc_mask_files = natsorted(list(SC_DOWNSAMPLED_MASK_DIR.glob("*.tiff")))
        print(f"Found {len(sc_mask_files)} SC mask files")

        sc_needs_creation = (
            not SC_MASK_ZARR_PATH.exists()
            or np.max(zarr.open(SC_MASK_ZARR_PATH)[5200]) == 0
        )

        if sc_needs_creation:
            print("Creating zarr volume for upsampled SC mask...")
            script_start = perf_counter()

            zarr.open(
                SC_MASK_ZARR_PATH,
                mode='a',
                shape=desired_shape_zyx,
                chunks=(1, Y, X),
                dtype=np.uint8,
                compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE),
            )

            dilating_kernel = np.ones((3, 3), np.uint8)
            num_workers = max(os.cpu_count() // 2, 52)

            with tqdm(total=len(sc_mask_files), desc="Upsampling SC mask") as bar:
                delayed_tasks = [
                    delayed(process_and_upsample_sc_slice)(
                        idx=idx,
                        mask_files=sc_mask_files,
                        factor=UPSAMPLE_FACTOR,
                        desired_shape=desired_shape_zyx,
                        zarr_path=SC_MASK_ZARR_PATH,
                        kernel=dilating_kernel,
                    )
                    for idx in range(len(sc_mask_files))
                ]
                compute(*delayed_tasks, scheduler='threads', num_workers=num_workers)
                bar.update(len(sc_mask_files))

            elapsed = (perf_counter() - script_start) / 3600
            print(f"SC mask upsampling completed in {elapsed:.2f} hours")
        else:
            print(f"Zarr at {SC_MASK_ZARR_PATH} already exists and is non-empty. Skipping.")
