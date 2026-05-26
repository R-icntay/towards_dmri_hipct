#!/usr/bin/env python
"""Post-process raw tractography streamlines.

Loads raw streamlines (.npy) from 05_tractography.py, rescales them from
dMRI voxel space to HiP-CT level-4 voxel space, converts to RASMM via
the reference NIfTI affine, compresses, and saves as .trk. Optionally
filters streamlines by z-range and minimum physical length.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import length
from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines import ArraySequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.analysis import get_voxel_ratio


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration ----
    ODF_ANALYSIS_DIR = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis")
    HIPCT_LEVEL4_PATH = Path("/media/eric/sta_analysis/15.13um_I58_brain-hemi_complete-sample_pag-0.17_0.31_jp2_masked_structure_tensor_2.0_4.0_analysis/registration_new_dmri/hipct_hemi_downsampled_level_4_masked_norm_percentile_83_99.nii.gz")

    HIPCT_RESOLUTION = 15.13   # um
    DMRI_RESOLUTION = 800.0    # um
    DEGREE = 8

    START_IDX = 0
    END_IDX = 9880
    GFA_THRESHOLD = 0.8

    COMPRESS_TOL = 0.1         # mm — 0.01 for deterministic, 0.1 for probabilistic
    MIN_LENGTH_MM = 20.0       # filter streamlines shorter than this
    Z_RANGE = (200, 500)       # filter by average z in voxel space (None to skip)
    TARGET_COUNT = 700000      # deterministic subsampling target (None to skip)

    # ---- Derived parameters ----
    voxel_ratio = get_voxel_ratio(HIPCT_RESOLUTION, DMRI_RESOLUTION)

    assert ODF_ANALYSIS_DIR.exists()
    assert HIPCT_LEVEL4_PATH.exists()

    # ---- Find raw streamlines file ----
    raw_pattern = f"raw_*_streamlines_{START_IDX}_{END_IDX}_{GFA_THRESHOLD}_*.npy"
    raw_files = sorted(ODF_ANALYSIS_DIR.glob(raw_pattern))
    assert len(raw_files) >= 1, f"No raw streamlines found matching {raw_pattern}"
    raw_fname = raw_files[-1]
    print(f"Loading raw streamlines from {raw_fname}")
    streamlines_all = np.load(raw_fname, allow_pickle=True)
    print(f"Loaded {len(streamlines_all)} streamlines")

    # ---- Load SH coefficients shape for scaling ----
    sh_coef_zarr_path = ODF_ANALYSIS_DIR / f"sh_coefficients_degree_{DEGREE}_xyz.zarr"
    sh_coeffs = zarr.open(sh_coef_zarr_path, mode='r')
    start_step_idx = 0
    end_step_idx = END_IDX // voxel_ratio
    sh_coeffs_subset_shape = sh_coeffs[:, :, start_step_idx:end_step_idx, :].shape[:3]
    print(f"SH coeff subset shape: {sh_coeffs_subset_shape}")

    # ---- Load reference NIfTI ----
    hipct_level4_nifti = nib.load(str(HIPCT_LEVEL4_PATH))
    print(f"Reference shape: {hipct_level4_nifti.shape}")

    # ---- Rescale streamlines to HiP-CT level 4 voxel space ----
    trk_shape = np.array(sh_coeffs_subset_shape, dtype=np.float32)
    hipct_shape_level4 = (trk_shape * voxel_ratio) / 2 ** 4
    scaling = hipct_shape_level4 / trk_shape
    print(f"Scaling factors (dMRI → level 4): {scaling}")

    scaled_streamlines = ArraySequence([s * scaling for s in streamlines_all])
    z_offset = np.array([0, 0, start_step_idx * scaling[-1]])
    scaled_streamlines = ArraySequence([s + z_offset for s in scaled_streamlines])

    # ---- Create tractogram and convert to RASMM ----
    sft = StatefulTractogram(
        streamlines=scaled_streamlines,
        reference=hipct_level4_nifti,
        space=Space.VOX,
    )
    print(f"Tractogram (VOX): {sft}")

    sft.to_space(Space.RASMM)

    # ---- Compress streamlines ----
    compressed = compress_streamlines(sft.streamlines, tol_error=COMPRESS_TOL)
    sft_compressed = StatefulTractogram(
        streamlines=compressed,
        reference=hipct_level4_nifti,
        space=Space.RASMM,
    )
    print(f"Compressed tractogram (RASMM): {sft_compressed}")

    out_stem = raw_fname.stem.replace("raw_", "800um_")
    compressed_fname = ODF_ANALYSIS_DIR / f"{out_stem}_compressed.trk"
    save_tractogram(sft_compressed, str(compressed_fname), bbox_valid_check=False)
    print(f"Saved compressed tractogram: {compressed_fname}")

    # ---- Optional: filter by z-range ----
    if Z_RANGE is not None:
        sft.to_space(Space.VOX)
        z_min, z_max = Z_RANGE
        filtered = [sl for sl in sft.streamlines if z_min <= np.mean(sl[:, 2]) <= z_max]
        print(f"Z-range filter [{z_min}, {z_max}]: {len(filtered)} / {len(sft.streamlines)} streamlines")

        sft_filtered = StatefulTractogram(
            streamlines=filtered,
            reference=hipct_level4_nifti,
            space=Space.VOX,
        )
        sft_filtered.to_space(Space.RASMM)
    else:
        sft_filtered = sft_compressed

    # ---- Optional: filter by minimum length ----
    if MIN_LENGTH_MM is not None:
        sl_lengths = length(sft_filtered.streamlines)
        long_sl = ArraySequence([
            s for s, l in zip(sft_filtered.streamlines, sl_lengths) if l >= MIN_LENGTH_MM
        ])
        print(f"Length filter (>= {MIN_LENGTH_MM}mm): {len(long_sl)} / {len(sft_filtered.streamlines)}")
    else:
        long_sl = sft_filtered.streamlines

    # ---- Optional: deterministic subsampling ----
    if TARGET_COUNT is not None and len(long_sl) > TARGET_COUNT:
        indices = np.linspace(0, len(long_sl) - 1, num=TARGET_COUNT, dtype=int)
        long_sl = long_sl[indices]
        print(f"Subsampled to {TARGET_COUNT} streamlines")

    # ---- Save filtered tractogram ----
    if Z_RANGE is not None or MIN_LENGTH_MM is not None or TARGET_COUNT is not None:
        compressed_filtered = compress_streamlines(long_sl, tol_error=COMPRESS_TOL)
        sft_final = StatefulTractogram(
            streamlines=compressed_filtered,
            reference=hipct_level4_nifti,
            space=Space.RASMM,
        )
        z_tag = f"_z_{Z_RANGE[0]}-{Z_RANGE[1]}" if Z_RANGE else ""
        len_tag = f"_minlen{int(MIN_LENGTH_MM)}" if MIN_LENGTH_MM else ""
        count_tag = f"_n{TARGET_COUNT:.0e}" if TARGET_COUNT else ""
        filtered_fname = ODF_ANALYSIS_DIR / f"{out_stem}{z_tag}{len_tag}{count_tag}_compressed.trk"
        save_tractogram(sft_final, str(filtered_fname), bbox_valid_check=False)
        print(f"Saved filtered tractogram: {filtered_fname}")
