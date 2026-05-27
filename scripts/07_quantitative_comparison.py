#!/usr/bin/env python
"""Quantitative comparison of vessel masking effects on fiber orientation.

Compares STA-derived fiber orientation estimates with and without gradient-level
vessel masking on the LADAF-2021-17 Pons VOI. Analysis stages:

1. Compute Westin anisotropy metrics and bin to dMRI supervoxels
2. Statistical tests on whole VOI (paired t-test, Wilcoxon, Cohen's d)
3. Sub-VOI analysis (vessel-dense region) — anisotropy stats + fODF + ACC
4. Per-supervoxel ACC computation
5. ACC vs vessel fraction scatter plot (Nature Comms styled)
6. Vessel fraction distribution plot
7. ACC between masked and unmasked global fODF SH coefficients
8. Violin plots — FA, linear anisotropy, planar anisotropy
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.analysis import (
    cohen_d,
    compute_ACC,
    compute_structure_tensor_metrics,
    get_voxel_ratio,
    validate_eigenvalue_order,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_nature_comms_style():
    """Apply Nature Communications figure styling."""
    mpl.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 2,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.major.size": 5.5,
        "ytick.major.size": 5.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.bottom": True,
        "ytick.left": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sns.set_theme(style="white", rc={"axes.grid": False})


def _nature_violin(data, labels, ylabel, title, out_path, ylim=(0, 1.0)):
    """Create a Nature Comms styled violin plot comparing two conditions."""
    medians = [np.median(d) for d in data]

    fig, ax = plt.subplots(figsize=(8, 6))

    set2_colors = sns.color_palette("Set2")
    custom_palette = [set2_colors[1], set2_colors[0]]

    sns.violinplot(
        data=data,
        palette=custom_palette,
        linewidth=2,
        inner="quartile",
        ax=ax,
    )

    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=20)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

    ax.set_ylim(*ylim)
    yticks = np.arange(ylim[0], ylim[1] + 0.01, 0.2)
    ax.set_yticks(np.round(yticks, 1))
    ax.tick_params(axis='x', which='major', bottom=True, width=2.0, length=5.5)
    ax.tick_params(axis='y', which='major', left=True, width=2.0, length=5.5, labelsize=18)

    for i, median in enumerate(medians):
        ax.text(
            i, median,
            f"{median:.4f}",
            ha='center', va='center',
            fontsize=14,
            fontweight='bold',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black',
                      boxstyle='round,pad=0.3', linewidth=1.5),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved violin plot to {out_path}")
    plt.close(fig)


def _bin_to_supervoxels(arr, voxel_ratio):
    """Reshape a 3D or 4D array for supervoxel binning.

    3D (X, Y, Z) → (X//vr, vr, Y//vr, vr, Z//vr, vr)
    4D (C, X, Y, Z) → (C, X//vr, vr, Y//vr, vr, Z//vr, vr)
    """
    vr = voxel_ratio
    if arr.ndim == 3:
        X, Y, Z = arr.shape
        return arr.reshape(X // vr, vr, Y // vr, vr, Z // vr, vr)
    elif arr.ndim == 4:
        C, X, Y, Z = arr.shape
        return arr.reshape(C, X // vr, vr, Y // vr, vr, Z // vr, vr)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {arr.ndim}D")


def _drop_nans_infs(vec_roi):
    """Drop NaN and Inf values from an eigenvector array of shape (x, y, z, 3)."""
    assert vec_roi.shape[-1] == 3, "Eigenvectors should be of shape (x, y, z, 3)"
    print(f"Shape before dropping NaN/Inf: {vec_roi.shape}")

    mask = np.ones(vec_roi.shape[:-1], dtype=bool)
    for dim in range(3):
        mask[np.isnan(vec_roi[..., dim])] = False
        mask[np.isinf(vec_roi[..., dim])] = False

    nan_inf_fraction = 1 - np.sum(mask) / mask.size
    print(f"Fraction of NaN/Inf: {nan_inf_fraction * 100:.2f}%")

    vec_roi = vec_roi[mask]
    print(f"Shape after dropping NaN/Inf: {vec_roi.shape}")
    return vec_roi


def _run_statistical_battery(unmasked, masked, metric_name):
    """Run paired t-test, Cohen's d, one-sample t-test, and Wilcoxon for a metric."""
    t_stat, p_val = ttest_rel(unmasked, masked)
    print(f"\nPaired t-test for {metric_name}: t-stat: {t_stat:.4f}, p-val: {p_val:.4g}")
    print(f"  Significant at alpha=0.05: {p_val < 0.05}")

    d, conf_int = cohen_d(unmasked, masked)
    print(f"  Cohen's d: {d:.4f}")
    print(f"  95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

    differences = unmasked - masked
    t_statistic, p_value_ttest = stats.ttest_1samp(differences, 0)
    n = len(differences)
    mean_diff = np.mean(differences)
    sem_diff = stats.sem(differences)
    ci = stats.t.interval(0.95, n - 1, loc=mean_diff, scale=sem_diff)
    print(f"  One-sample t-test on differences: t={t_statistic:.4f}, p={p_value_ttest:.4g}")
    print(f"  Mean difference: {mean_diff:.4f}, 95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")

    statistic_w, p_value_w = stats.wilcoxon(differences)
    print(f"  Wilcoxon signed-rank: statistic={statistic_w:.4f}, p={p_value_w:.4g}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIGMA = 1.5
RHO = 3
HIPCT_RESOLUTION = 6.54   # µm
DMRI_RESOLUTION = 800      # µm
SH_DEGREE = 8
N_SPHERE_POINTS = 6500
REMAINDER = 48
RGB_ORDER = (2, 1, 0)

SAMPLE_TAG = f"6.54um_LADAF-2021-17_brain_ROI-03_pag-0.03_0.07_jp2_1392_2575_8000_{SIGMA}_dot{RHO}"

UNMASKED_EIGEN_DIR = Path(SAMPLE_TAG)
MASKED_EIGEN_DIR = Path(f"gradient_masked_{SAMPLE_TAG}")
UNMASKED_ODF_DIR = Path(f"{SAMPLE_TAG}_odf_analysis")
MASKED_ODF_DIR = Path(f"gradient_masked_{SAMPLE_TAG}_odf_analysis")

SUB_VOI_X = slice(0, 488)
SUB_VOI_Y = slice(488, None)
SUB_VOI_Z = slice(None)

WONG_BLUE = "#0072B2"
LABELS = ["Vessels present", "Vessels masked"]

# ---------------------------------------------------------------------------
# Stage flags
# ---------------------------------------------------------------------------

RUN_COMPUTE_ANISOTROPY = True
RUN_WHOLE_VOI_STATS = True
RUN_SUB_VOI_ANALYSIS = True
RUN_PER_SUPERVOXEL_ACC = True
RUN_ACC_SCATTER = True
RUN_VESSEL_FRACTION_DIST = True
RUN_GLOBAL_ACC = True
RUN_VIOLIN_PLOTS = True


# ===========================================================================
# Stage 1: Compute anisotropy metrics and bin to supervoxels
# ===========================================================================
if RUN_COMPUTE_ANISOTROPY:
    print("\n" + "=" * 60)
    print("Stage 1: Compute anisotropy metrics and bin to supervoxels")
    print("=" * 60)

    assert UNMASKED_EIGEN_DIR.exists(), f"Unmasked eigen dir {UNMASKED_EIGEN_DIR} does not exist"
    assert MASKED_EIGEN_DIR.exists(), f"Masked eigen dir {MASKED_EIGEN_DIR} does not exist"

    # Load eigenvalues (ascending order from eigensolver → reverse to descending)
    eigen_values = np.load(UNMASKED_EIGEN_DIR / "0_1024_eigenvalues.npy")
    eigen_values = np.stack([eigen_values[2], eigen_values[1], eigen_values[0]], axis=0)
    print(f"Eigenvalues shape: {eigen_values.shape}")

    masked_eigen_values = np.load(MASKED_EIGEN_DIR / "0_1024_eigenvalues.npy")
    masked_eigen_values = np.stack([masked_eigen_values[2], masked_eigen_values[1], masked_eigen_values[0]], axis=0)
    print(f"Masked eigenvalues shape: {masked_eigen_values.shape}")

    assert validate_eigenvalue_order(eigen_values), "Eigenvalues not ordered λ₁ ≥ λ₂ ≥ λ₃"
    assert validate_eigenvalue_order(masked_eigen_values), "Masked eigenvalues not ordered λ₁ ≥ λ₂ ≥ λ₃"

    # Load eigenvectors
    eigen_vectors = np.load(UNMASKED_EIGEN_DIR / "0_1024_eigenvectors.npy")
    masked_eigen_vectors = np.load(MASKED_EIGEN_DIR / "0_1024_eigenvectors.npy")
    print(f"Eigenvectors shape: {eigen_vectors.shape}")
    print(f"Masked eigenvectors shape: {masked_eigen_vectors.shape}")

    # Drop remainder pixels
    eigen_values = eigen_values[:, :-REMAINDER, :-REMAINDER, :-REMAINDER]
    masked_eigen_values = masked_eigen_values[:, :-REMAINDER, :-REMAINDER, :-REMAINDER]
    eigen_vectors = eigen_vectors[:, :-REMAINDER, :-REMAINDER, :-REMAINDER]
    masked_eigen_vectors = masked_eigen_vectors[:, :-REMAINDER, :-REMAINDER, :-REMAINDER]
    print(f"Shapes after dropping remainder ({REMAINDER}px):")
    print(f"  Eigenvalues: {eigen_values.shape}, Masked: {masked_eigen_values.shape}")

    # Load FA
    assert UNMASKED_ODF_DIR.exists(), f"Unmasked ODF dir {UNMASKED_ODF_DIR} does not exist"
    assert MASKED_ODF_DIR.exists(), f"Masked ODF dir {MASKED_ODF_DIR} does not exist"
    FA = np.load(UNMASKED_ODF_DIR / "FA.npy")
    masked_FA = np.load(MASKED_ODF_DIR / "masked_FA.npy")
    FA = FA[:-REMAINDER, :-REMAINDER, :-REMAINDER]
    masked_FA = masked_FA[:-REMAINDER, :-REMAINDER, :-REMAINDER]
    print(f"FA shape: {FA.shape}, Masked FA shape: {masked_FA.shape}")

    assert not np.allclose(masked_eigen_values, eigen_values), "Eigenvalues identical — masking had no effect"
    assert not np.allclose(masked_FA, FA), "FA identical — masking had no effect"

    # Compute anisotropy metrics
    anisotropy_metrics = compute_structure_tensor_metrics(eigen_values)
    masked_anisotropy_metrics = compute_structure_tensor_metrics(masked_eigen_values)

    # Bin to supervoxel resolution
    voxel_ratio = get_voxel_ratio(HIPCT_RESOLUTION, DMRI_RESOLUTION)

    eigen_values_binned = _bin_to_supervoxels(eigen_values, voxel_ratio)
    masked_eigen_values_binned = _bin_to_supervoxels(masked_eigen_values, voxel_ratio)
    eigen_vectors_binned = _bin_to_supervoxels(eigen_vectors, voxel_ratio)
    masked_eigen_vectors_binned = _bin_to_supervoxels(masked_eigen_vectors, voxel_ratio)
    FA_binned = _bin_to_supervoxels(FA, voxel_ratio)
    masked_FA_binned = _bin_to_supervoxels(masked_FA, voxel_ratio)

    linear_binned = _bin_to_supervoxels(anisotropy_metrics['linear_anisotropy'], voxel_ratio)
    masked_linear_binned = _bin_to_supervoxels(masked_anisotropy_metrics['linear_anisotropy'], voxel_ratio)
    planar_binned = _bin_to_supervoxels(anisotropy_metrics['planar_anisotropy'], voxel_ratio)
    masked_planar_binned = _bin_to_supervoxels(masked_anisotropy_metrics['planar_anisotropy'], voxel_ratio)

    print("Binning complete")


# ===========================================================================
# Stage 2: Statistical tests — whole VOI
# ===========================================================================
if RUN_WHOLE_VOI_STATS:
    print("\n" + "=" * 60)
    print("Stage 2: Statistical tests — whole VOI")
    print("=" * 60)

    mean_linear = np.mean(linear_binned, axis=(1, 3, 5)).flatten()
    masked_mean_linear = np.mean(masked_linear_binned, axis=(1, 3, 5)).flatten()
    _run_statistical_battery(mean_linear, masked_mean_linear, "linear anisotropy")

    mean_planar = np.mean(planar_binned, axis=(1, 3, 5)).flatten()
    masked_mean_planar = np.mean(masked_planar_binned, axis=(1, 3, 5)).flatten()
    _run_statistical_battery(mean_planar, masked_mean_planar, "planar anisotropy")

    FA_mean = np.mean(FA_binned, axis=(1, 3, 5)).flatten()
    masked_FA_mean = np.mean(masked_FA_binned, axis=(1, 3, 5)).flatten()
    _run_statistical_battery(FA_mean, masked_FA_mean, "FA")


# ===========================================================================
# Stage 3: Sub-VOI analysis (vessel-dense region)
# ===========================================================================
if RUN_SUB_VOI_ANALYSIS:
    print("\n" + "=" * 60)
    print("Stage 3: Sub-VOI analysis (vessel-dense region)")
    print("=" * 60)

    # Crop sub-VOI
    voi_eigen_values = eigen_values[:, SUB_VOI_X, SUB_VOI_Y, SUB_VOI_Z]
    voi_eigen_vectors = eigen_vectors[:, SUB_VOI_X, SUB_VOI_Y, SUB_VOI_Z]
    masked_voi_eigen_values = masked_eigen_values[:, SUB_VOI_X, SUB_VOI_Y, SUB_VOI_Z]
    masked_voi_eigen_vectors = masked_eigen_vectors[:, SUB_VOI_X, SUB_VOI_Y, SUB_VOI_Z]
    print(f"Sub-VOI eigenvalues shape: {voi_eigen_values.shape}")

    voi_FA = FA[SUB_VOI_X, SUB_VOI_Y, SUB_VOI_Z]
    masked_voi_FA = masked_FA[SUB_VOI_X, SUB_VOI_Y, SUB_VOI_Z]
    np.save("voi_FA.npy", voi_FA)
    np.save("masked_voi_FA.npy", masked_voi_FA)
    print(f"Sub-VOI FA shape: {voi_FA.shape}")

    # Bin sub-VOI FA
    voi_FA_binned = _bin_to_supervoxels(voi_FA, voxel_ratio)
    masked_voi_FA_binned = _bin_to_supervoxels(masked_voi_FA, voxel_ratio)

    # Compute sub-VOI anisotropy metrics
    voi_anisotropy_metrics = compute_structure_tensor_metrics(voi_eigen_values)
    masked_voi_anisotropy_metrics = compute_structure_tensor_metrics(masked_voi_eigen_values)

    np.savez("voi_anisotropy_metrics.npz",
             linear_anisotropy=voi_anisotropy_metrics['linear_anisotropy'],
             planar_anisotropy=voi_anisotropy_metrics['planar_anisotropy'],
             spherical_anisotropy=voi_anisotropy_metrics['spherical_anisotropy'])
    np.savez("masked_voi_anisotropy_metrics.npz",
             linear_anisotropy=masked_voi_anisotropy_metrics['linear_anisotropy'],
             planar_anisotropy=masked_voi_anisotropy_metrics['planar_anisotropy'],
             spherical_anisotropy=masked_voi_anisotropy_metrics['spherical_anisotropy'])
    print("Saved sub-VOI anisotropy metrics (.npz)")

    # Bin sub-VOI anisotropy metrics
    voi_linear_binned = _bin_to_supervoxels(voi_anisotropy_metrics['linear_anisotropy'], voxel_ratio)
    masked_voi_linear_binned = _bin_to_supervoxels(masked_voi_anisotropy_metrics['linear_anisotropy'], voxel_ratio)
    voi_planar_binned = _bin_to_supervoxels(voi_anisotropy_metrics['planar_anisotropy'], voxel_ratio)
    masked_voi_planar_binned = _bin_to_supervoxels(masked_voi_anisotropy_metrics['planar_anisotropy'], voxel_ratio)

    # Statistical tests on sub-VOI
    print("\n--- Sub-VOI statistical tests ---")
    mean_voi_linear = np.mean(voi_linear_binned, axis=(1, 3, 5)).flatten()
    masked_mean_voi_linear = np.mean(masked_voi_linear_binned, axis=(1, 3, 5)).flatten()
    _run_statistical_battery(mean_voi_linear, masked_mean_voi_linear, "sub-VOI linear anisotropy")

    mean_voi_planar = np.mean(voi_planar_binned, axis=(1, 3, 5)).flatten()
    masked_mean_voi_planar = np.mean(masked_voi_planar_binned, axis=(1, 3, 5)).flatten()
    _run_statistical_battery(mean_voi_planar, masked_mean_voi_planar, "sub-VOI planar anisotropy")

    mean_voi_FA = np.mean(voi_FA_binned, axis=(1, 3, 5)).flatten()
    masked_mean_voi_FA = np.mean(masked_voi_FA_binned, axis=(1, 3, 5)).flatten()
    _run_statistical_battery(mean_voi_FA, masked_mean_voi_FA, "sub-VOI FA")

    # Sub-VOI fODF and ACC
    print("\n--- Sub-VOI fODF estimation ---")
    from fiberorient.odf import ODF
    from fiberorient.util import make_sphere

    sphere = make_sphere(N_SPHERE_POINTS)

    for tag, evec, save_dir, prefix in [
        ("unmasked", voi_eigen_vectors, UNMASKED_ODF_DIR, "sub_voi"),
        ("masked", masked_voi_eigen_vectors, MASKED_ODF_DIR, "masked_sub_voi"),
    ]:
        print(f"\nFitting {tag} sub-VOI fODF...")
        evec_reordered = np.stack([evec[RGB_ORDER[0]], evec[RGB_ORDER[1]], evec[RGB_ORDER[2]]], axis=0)
        evec_xyz = np.transpose(evec_reordered, (1, 2, 3, 0))
        if np.any(np.isnan(evec_xyz)) or np.any(np.isinf(evec_xyz)):
            evec_xyz = _drop_nans_infs(evec_xyz)

        odf = ODF(degree=SH_DEGREE, method='precompute').fit(evec_xyz)
        odf2sphere = odf.to_sphere(sphere)
        np.save(save_dir / f"{prefix}_odf_coef.npy", odf.coef)
        np.save(save_dir / f"{prefix}_odf_array.npy", odf2sphere)
        print(f"  Saved {prefix}_odf_coef.npy (shape: {odf.coef.shape})")

    sub_voi_odf_coef = np.load(UNMASKED_ODF_DIR / "sub_voi_odf_coef.npy")
    masked_sub_voi_odf_coef = np.load(MASKED_ODF_DIR / "masked_sub_voi_odf_coef.npy")
    sub_voi_acc = compute_ACC(sub_voi_odf_coef, masked_sub_voi_odf_coef)
    print(f"\nSub-VOI ACC (masked vs unmasked): {sub_voi_acc:.6f}")


# ===========================================================================
# Stage 4: Per-supervoxel ACC computation
# ===========================================================================
if RUN_PER_SUPERVOXEL_ACC:
    print("\n" + "=" * 60)
    print("Stage 4: Per-supervoxel ACC computation")
    print("=" * 60)

    odf_coef = np.load(UNMASKED_ODF_DIR / "odf_coef.npy")
    masked_odf_coef = np.load(MASKED_ODF_DIR / "odf_coef.npy")
    print(f"ODF coef shape: {odf_coef.shape}")
    print(f"Masked ODF coef shape: {masked_odf_coef.shape}")

    acc_array = np.zeros((odf_coef.shape[0], odf_coef.shape[1], odf_coef.shape[-1]))
    for z in range(acc_array.shape[2]):
        for y in range(acc_array.shape[1]):
            for x in range(acc_array.shape[0]):
                u = odf_coef[x, y, :, z]
                v = masked_odf_coef[x, y, :, z]
                acc_array[x, y, z] = compute_ACC(u, v)

    print(f"ACC array shape: {acc_array.shape}")
    print(f"ACC range: [{acc_array.min():.4f}, {acc_array.max():.4f}]")
    print(f"ACC mean: {acc_array.mean():.4f}")

    vessel_volume = np.load(UNMASKED_ODF_DIR / "vessel_volume.npy")
    vessel_fraction = np.load(UNMASKED_ODF_DIR / "vessel_frac.npy")
    print(f"Vessel fraction shape: {vessel_fraction.shape}")


# ===========================================================================
# Stage 5: ACC vs vessel fraction scatter plot
# ===========================================================================
if RUN_ACC_SCATTER:
    print("\n" + "=" * 60)
    print("Stage 5: ACC vs vessel fraction scatter plot")
    print("=" * 60)

    _set_nature_comms_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        vessel_fraction,
        acc_array,
        s=20,
        color=WONG_BLUE,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("ACC Values vs Vessel Fractions", fontsize=24)
    ax.set_xlabel("Supervoxel vessel fraction", fontsize=24)
    ax.set_ylabel("ACC", fontsize=24)
    ax.set_xscale("log")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.tick_params(axis='x', which='major', bottom=True, width=2.0, length=5.5)
    ax.tick_params(axis='y', which='major', left=True, width=2.0, length=5.5)

    plt.tight_layout()
    out_path = "whole_voi_acc_vs_vessel_fraction.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to {out_path}")
    plt.close(fig)


# ===========================================================================
# Stage 6: Vessel fraction distribution
# ===========================================================================
if RUN_VESSEL_FRACTION_DIST:
    print("\n" + "=" * 60)
    print("Stage 6: Vessel fraction distribution")
    print("=" * 60)

    _set_nature_comms_style()

    vf_data = vessel_fraction.flatten()
    vf_data = vf_data[~np.isnan(vf_data)]

    fig, ax = plt.subplots(figsize=(8, 6))

    set2_colors = sns.color_palette("Set2")
    sns.violinplot(
        x=vf_data,
        palette=[set2_colors[1], set2_colors[0]],
        linewidth=0.8,
        ax=ax,
    )

    ax.set_xlabel("Supervoxel vessel fraction", fontsize=24)
    ax.set_title("Distribution of Vessel Fractions", fontsize=24)
    ax.set_yticks([])
    ax.set_ylabel("Density")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

    ax.set_xlim(-0.025, 0.2)
    ax.set_xticks([0.0, 0.025, 0.050, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    ax.tick_params(axis='x', which='major', bottom=True, width=2.0, length=5.5, labelsize=16)

    plt.tight_layout()
    out_path = "vessel_fraction_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot to {out_path}")
    plt.close(fig)


# ===========================================================================
# Stage 7: ACC between masked and unmasked global fODFs
# ===========================================================================
if RUN_GLOBAL_ACC:
    print("\n" + "=" * 60)
    print("Stage 7: ACC comparison (masked vs unmasked global fODF)")
    print("=" * 60)

    assert UNMASKED_ODF_DIR.exists(), f"Unmasked ODF directory {UNMASKED_ODF_DIR} does not exist"
    assert MASKED_ODF_DIR.exists(), f"Masked ODF directory {MASKED_ODF_DIR} does not exist"

    whole_voi_odf_coef = np.load(UNMASKED_ODF_DIR / "whole_voi_odf_coef.npy")
    masked_whole_voi_odf_coef = np.load(MASKED_ODF_DIR / "whole_voi_odf_coef.npy")
    print(f"Unmasked global fODF coef shape: {whole_voi_odf_coef.shape}")
    print(f"Masked global fODF coef shape: {masked_whole_voi_odf_coef.shape}")

    acc = compute_ACC(whole_voi_odf_coef, masked_whole_voi_odf_coef)
    print(f"ACC (masked vs unmasked): {acc:.6f}")


# ===========================================================================
# Stage 8: Violin plots — FA, linear anisotropy, planar anisotropy
# ===========================================================================
if RUN_VIOLIN_PLOTS:
    print("\n" + "=" * 60)
    print("Stage 8: Violin plots")
    print("=" * 60)

    _set_nature_comms_style()

    # Load sub-VOI data (from stage 3 outputs or pre-computed files)
    if 'voi_FA' not in dir():
        voi_FA = np.load("voi_FA.npy")
        masked_voi_FA = np.load("masked_voi_FA.npy")
    if 'voi_anisotropy_metrics' not in dir():
        voi_anisotropy_metrics = dict(np.load("voi_anisotropy_metrics.npz"))
        masked_voi_anisotropy_metrics = dict(np.load("masked_voi_anisotropy_metrics.npz"))

    _nature_violin(
        data=[voi_FA.flatten(), masked_voi_FA.flatten()],
        labels=LABELS,
        ylabel="Fractional Anisotropy",
        title="Effect of Vasculature on FA",
        out_path="FA_violin.png",
    )

    _nature_violin(
        data=[
            voi_anisotropy_metrics['linear_anisotropy'].flatten(),
            masked_voi_anisotropy_metrics['linear_anisotropy'].flatten(),
        ],
        labels=LABELS,
        ylabel="Fiber-Like Symmetry",
        title="Effect of Vasculature on Fiber-Like Symmetry",
        out_path="linear_anisotropy_violin.png",
    )

    _nature_violin(
        data=[
            voi_anisotropy_metrics['planar_anisotropy'].flatten(),
            masked_voi_anisotropy_metrics['planar_anisotropy'].flatten(),
        ],
        labels=LABELS,
        ylabel="Planar-Like Symmetry",
        title="Effect of Vasculature on Planar-Like Symmetry",
        out_path="planar_anisotropy_violin.png",
    )
