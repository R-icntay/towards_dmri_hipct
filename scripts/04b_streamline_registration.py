#!/usr/bin/env python
"""Register dMRI streamlines to HiP-CT space and post-process.

Loads dMRI CSD tractography (.trk), applies ANTsPy registration
transforms (computed by 04_registration.py) to all streamline points
using the inverse transform chain, and performs streamline cleanup.
Three independent stages are controlled by boolean flags:

1. Streamline registration -- RAS/LPS coordinate conversion, apply ANTs
   point transforms via the inverse chain, reconstruct tractogram
2. Streamline compression -- compress streamlines using tolerance error
3. Streamline filtering -- z-range filter, length filter, deterministic
   subsampling
"""

from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import pandas as pd
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import length
from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines import ArraySequence

# import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration ----
    USE_MI_ONLY = False  # must match 04_registration.py

    BASE_PATH = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/registration_new_dmri")
    HIPCT_NIFTI = "hipct_hemi_downsampled_level_4_masked_norm_percentile_83_99.nii.gz"
    DMRI_TRK = "sub-I58_sample-hemi_desc-CSD_tractography.trk"

    COMPRESS_TOL = 0.1         # mm (0.01 for deterministic, 0.1 for probabilistic)
    MIN_LENGTH_MM = 20.0       # filter streamlines shorter than this
    Z_RANGE = (200, 500)       # filter by average z in voxel space (None to skip)
    TARGET_COUNT = 390000      # deterministic subsampling target (None to skip)

    # ---- Stage flags ----
    RUN_STREAMLINE_REGISTRATION = True
    RUN_STREAMLINE_COMPRESSION = True
    RUN_STREAMLINE_FILTERING = True

    # ---- Derived names ----
    REG_PREFIX = "synra_mi_only" if USE_MI_ONLY else "synra_hybrid"
    TRANSFORM_PREFIX = f"bhigh_b0_ratio_map_registered_to_hipct_level4_{REG_PREFIX}"
    TRK_STEM = Path(DMRI_TRK).stem
    REGISTERED_TRK = f"{TRK_STEM}_registered_to_hipct_{REG_PREFIX}.trk"

    hipct_nifti = nib.load(str(BASE_PATH / HIPCT_NIFTI))
    print(f"Reference HiP-CT shape: {hipct_nifti.shape}")

    # ======================================================================
    # Stage 1: Register dMRI streamlines to HiP-CT space
    # ======================================================================
    if RUN_STREAMLINE_REGISTRATION:
        print("\n" + "=" * 60)
        print(f"Stage 1: Streamline registration ({REG_PREFIX})")
        print("=" * 60)

        # Load transform file paths
        fwd_affine = str(BASE_PATH / f"fwd_{TRANSFORM_PREFIX}_affine.mat")
        inv_warp = str(BASE_PATH / f"inv_{TRANSFORM_PREFIX}_warp.nii.gz")

        # Inverse chain: Affine (global) -> InverseWarp (local deformation)
        inv_transform_chain = [fwd_affine, inv_warp]

        # Load dMRI tractography in RASMM space
        dmri_trk = load_tractogram(
            filename=str(BASE_PATH / DMRI_TRK),
            reference='same',
            to_space=Space.RASMM,
            bbox_valid_check=False,
        )
        print(f"Loaded tractogram: {dmri_trk}")
        print(f"Number of streamlines: {len(dmri_trk.streamlines)}")

        total_points = dmri_trk.streamlines._data.shape[0]
        print(f"Total points: {total_points}")

        # RAS -> LPS conversion for ANTs coordinate system
        ras_to_lps = np.array([-1, -1, 1], dtype=dmri_trk.streamlines._data.dtype)
        points_ras = dmri_trk.streamlines._data
        df_points = pd.DataFrame(points_ras, columns=['x', 'y', 'z'])
        print("Converting points from RAS to LPS for ANTs...")
        df_points[['x', 'y', 'z']] = df_points[['x', 'y', 'z']] * ras_to_lps

        # Apply ANTs transforms to all streamline points
        # Affine describes HiP-CT->dMRI (fixed->moving), so we INVERT it
        # to go dMRI->HiP-CT. The inv_warp IS already the inverse, so no inversion.
        print(f"Applying ANTs transforms to {total_points} points...")
        warped_df = ants.apply_transforms_to_points(
            dim=3,
            points=df_points,
            transformlist=inv_transform_chain,
            whichtoinvert=[True, False],
        )

        # LPS -> RAS conversion back to DIPY convention
        warped_points = (warped_df[['x', 'y', 'z']] * ras_to_lps).to_numpy(
            dtype=dmri_trk.streamlines._data.dtype,
        )

        # Reconstruct ArraySequence with original streamline structure
        registered_streamlines = ArraySequence()
        registered_streamlines._data = warped_points
        registered_streamlines._offsets = dmri_trk.streamlines._offsets
        registered_streamlines._lengths = dmri_trk.streamlines._lengths

        # Save registered tractogram
        registered_sft = StatefulTractogram(
            streamlines=registered_streamlines,
            reference=hipct_nifti,
            space=Space.RASMM,
        )
        print(f"Registered tractogram: {registered_sft}")

        save_tractogram(
            registered_sft,
            filename=str(BASE_PATH / REGISTERED_TRK),
            bbox_valid_check=False,
        )
        print(f"Saved registered tractography: {REGISTERED_TRK}")

    # ======================================================================
    # Stage 2: Compress registered streamlines
    # ======================================================================
    if RUN_STREAMLINE_COMPRESSION:
        print("\n" + "=" * 60)
        print("Stage 2: Streamline compression")
        print("=" * 60)

        trk_path = str(BASE_PATH / REGISTERED_TRK)
        registered_sft = load_tractogram(
            filename=trk_path,
            reference='same',
            to_space=Space.RASMM,
            bbox_valid_check=False,
        )
        print(f"Loaded registered tractogram: {registered_sft}")

        compressed = compress_streamlines(registered_sft.streamlines, tol_error=COMPRESS_TOL)
        compressed_sft = StatefulTractogram(
            streamlines=compressed,
            reference=hipct_nifti,
            space=Space.RASMM,
        )
        print(f"Compressed tractogram: {compressed_sft}")

        compressed_path = trk_path.replace(".trk", "_compressed.trk")
        save_tractogram(compressed_sft, filename=compressed_path, bbox_valid_check=False)
        print(f"Saved compressed tractography: {compressed_path}")

    # ======================================================================
    # Stage 3: Filter by z-range, length, and deterministic subsampling
    # ======================================================================
    if RUN_STREAMLINE_FILTERING:
        print("\n" + "=" * 60)
        print("Stage 3: Streamline filtering")
        print("=" * 60)

        trk_path = str(BASE_PATH / REGISTERED_TRK)
        sft = load_tractogram(
            filename=trk_path,
            reference='same',
            to_space=Space.RASMM,
            bbox_valid_check=False,
        )
        print(f"Loaded registered tractogram: {sft}")

        # Z-range filter (in voxel space)
        if Z_RANGE is not None:
            sft.to_space(Space.VOX)
            z_min, z_max = Z_RANGE
            filtered = [sl for sl in sft.streamlines
                        if z_min <= np.mean(sl[:, 2]) <= z_max]
            print(f"Z-range filter [{z_min}, {z_max}]: "
                  f"{len(filtered)} / {len(sft.streamlines)} streamlines")

            sft_filtered = StatefulTractogram(
                streamlines=filtered,
                reference=hipct_nifti,
                space=Space.VOX,
            )
            sft_filtered.to_space(Space.RASMM)
        else:
            sft_filtered = sft

        # Length filter
        if MIN_LENGTH_MM is not None:
            sl_lengths = length(sft_filtered.streamlines)
            long_sl = ArraySequence([
                s for s, l in zip(sft_filtered.streamlines, sl_lengths)
                if l >= MIN_LENGTH_MM
            ])
            print(f"Length filter (>= {MIN_LENGTH_MM}mm): "
                  f"{len(long_sl)} / {len(sft_filtered.streamlines)} "
                  f"({len(long_sl) / len(sft_filtered.streamlines) * 100:.1f}%)")
        else:
            long_sl = sft_filtered.streamlines

        # Deterministic subsampling
        if TARGET_COUNT is not None and len(long_sl) > TARGET_COUNT:
            indices = np.linspace(0, len(long_sl) - 1, num=TARGET_COUNT, dtype=int)
            long_sl = long_sl[indices]
            print(f"Subsampled to {TARGET_COUNT} streamlines")

        # Compress and save filtered tractogram
        compressed_filtered = compress_streamlines(long_sl, tol_error=COMPRESS_TOL)
        sft_final = StatefulTractogram(
            streamlines=compressed_filtered,
            reference=hipct_nifti,
            space=Space.RASMM,
        )
        print(f"Final tractogram: {sft_final}")

        z_tag = f"_filtered_z{Z_RANGE[0]}-{Z_RANGE[1]}" if Z_RANGE else ""
        len_tag = f"_minlen{int(MIN_LENGTH_MM)}mm" if MIN_LENGTH_MM else ""
        count_tag = f"_deterministic_sampled_{TARGET_COUNT:.4e}" if TARGET_COUNT else ""
        filtered_fname = BASE_PATH / f"{TRK_STEM}_registered_to_hipct_{REG_PREFIX}{z_tag}{len_tag}{count_tag}_compressed.trk"
        save_tractogram(sft_final, str(filtered_fname), bbox_valid_check=False)
        print(f"Saved filtered tractogram: {filtered_fname}")
