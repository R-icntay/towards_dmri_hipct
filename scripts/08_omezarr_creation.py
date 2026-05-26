#!/usr/bin/env python
"""Create multi-resolution OME-Zarr image pyramids from HiP-CT slices.

Reads HiP-CT image slices (JP2), applies CLAHE contrast enhancement,
normalizes to uint8, and writes a multi-resolution OME-Zarr pyramid
for visualization alongside tractography results. Processing is
parallelised over slices using Dask.
"""

import math
import os
import time
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import zarr
from dask import compute, delayed
from natsort import natsorted
from numcodecs import Zstd
from ome_zarr_models.v04 import Image
from ome_zarr_models.v04.axes import Axis
from pydantic_zarr.v2 import ArraySpec
from skimage.measure import block_reduce
from skimage.transform import resize
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def write_to_ome_zarr(idx, num_slices_per_write, image_dirs, ome_group, dataset, update_bar):
    """Read a batch of image slices, preprocess, and write to an OME-Zarr level."""
    len_image_files = len(image_dirs)
    downsample_scale = int(dataset.coordinateTransformations[0].scale[0])
    image_array = []
    clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(8, 8))
    for i in range(idx, min(idx + num_slices_per_write, len_image_files)):
        image_path = image_dirs[i]
        img = cv2.imread(str(image_path), -1)  # Y, X
        img = clahe.apply(img).T  # Transpose to X, Y

        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)

        image_array.append(img)

    image_array = np.stack(image_array, axis=-1)  # X, Y, Z

    if downsample_scale != 1:
        image_array = block_reduce(
            image_array,
            block_size=(downsample_scale, downsample_scale, downsample_scale),
            func=np.mean,
        )
        image_array = image_array.astype(np.uint8)

    assert image_array.shape[2] == 1, f"Expected 1 slice after downsampling, got {image_array.shape[2]}"
    image_array = image_array[:, :, 0]  # X, Y

    out_shape = ome_group[dataset.path].shape[:2]  # X, Y
    if image_array.shape != out_shape:
        print(f"Resizing image array from {image_array.shape} to {out_shape}")
        image_array = resize(image_array.astype(np.float32), out_shape, order=1, anti_aliasing=True)
        image_array = image_array.astype(np.uint8)

    ome_group[dataset.path][:, :, idx // downsample_scale] = image_array

    if update_bar:
        update_bar()

    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration ----
    IMAGE_DIR = Path("15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked")
    EXTENSION = "*.jp2"
    SAVE_DIR = Path("/media/eric/eric_backup_03/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis")

    START_IDX = 0
    END_IDX = 9880

    VOXEL_SIZE = 1
    DOWNSAMPLE_LEVELS = [0, 1, 2, 3, 4, 5]

    # ---- Load image list ----
    assert IMAGE_DIR.exists(), f"Image directory {IMAGE_DIR} does not exist!"
    image_files = natsorted(list(IMAGE_DIR.glob(EXTENSION)))[START_IDX:END_IDX]
    print(f"Number of image files: {len(image_files)}")

    sample_img = cv2.imread(str(image_files[0]), -1)  # Y, X
    Y, X = sample_img.shape
    print(f"Sample image shape YX: ({Y}, {X}), dtype: {sample_img.dtype}")

    # ---- Create OME-Zarr store ----
    ome_zarr_path = SAVE_DIR / f"hipct_images_{START_IDX}_{END_IDX}_xyz.ome.zarr"

    full_res_spec = ArraySpec(
        shape=(X, Y, len(image_files)),
        chunks=(X, Y, 1),
        dtype=np.uint8,
        compressor=Zstd(level=5),
    )
    print(f"Full res array specs: {full_res_spec}")

    downsampled_specs = [
        full_res_spec.model_copy(
            update={
                "shape": tuple(math.ceil(s / 2**d) for s in full_res_spec.shape),
                "chunks": tuple(math.ceil(c / 2**d) for c in full_res_spec.chunks),
            }
        )
        for d in DOWNSAMPLE_LEVELS
    ]

    multiscale_image = Image.new(
        array_specs=downsampled_specs,
        paths=[f"level_{d}" for d in DOWNSAMPLE_LEVELS],
        axes=[
            Axis(name="x", type="space", unit="millimeter"),
            Axis(name="y", type="space", unit="millimeter"),
            Axis(name="z", type="space", unit="millimeter"),
        ],
        global_scale=[VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE],
        scales=[[2**d, 2**d, 2**d] for d in DOWNSAMPLE_LEVELS],
        translations=[[0, 0, 0] for _ in DOWNSAMPLE_LEVELS],
        name=f"HiP-CT images {START_IDX} to {END_IDX}",
    )

    try:
        ome_store = zarr.storage.LocalStore(str(ome_zarr_path))
    except Exception:
        ome_store = zarr.DirectoryStore(str(ome_zarr_path))

    ome_group = multiscale_image.to_zarr(store=ome_store, path="/")
    for idx, dataset in enumerate(multiscale_image.datasets[0]):
        print(f"Level {idx}, path: {dataset.path}, shape: {ome_group[dataset.path].shape}")

    # ---- Write all pyramid levels (lowest res first) ----
    start_time = perf_counter()
    with tqdm(total=len(multiscale_image.datasets[0]), desc="Writing OME-Zarr levels") as level_bar:
        for dataset in multiscale_image.datasets[0][::-1]:
            print(f"\n\nWriting level {dataset.path}")
            downsample_scale = dataset.coordinateTransformations[0].scale[0]
            num_slices_per_write = int(downsample_scale)
            print(f"Scale: {downsample_scale}")

            write_indexes = list(range(0, len(image_files), num_slices_per_write))
            print(f"Number of write indexes in {dataset.path}: {len(write_indexes)}")

            time.sleep(2)
            write_time_start = perf_counter()

            with tqdm(total=len(write_indexes), desc=f"Writing {dataset.path}") as bar:
                delayed_tasks = [
                    delayed(write_to_ome_zarr)(
                        idx, num_slices_per_write, image_files,
                        ome_group, dataset, bar.update,
                    )
                    for idx in write_indexes
                ]
                compute(*delayed_tasks, scheduler='threads', num_workers=os.cpu_count() // 2)

            write_time_end = perf_counter()
            print(f"\nTime taken to write level {dataset.path}: {(write_time_end - write_time_start) / 60:.2f} minutes")
            level_bar.update()

    elapsed = (perf_counter() - start_time) / 3600
    print(f"\nTotal time to write OME-ZARR: {elapsed:.2f} hours")
