"""Shared analysis utilities for the dMRI–HiP-CT pipeline."""

import numpy as np


def get_analysis_shape(X, Y, Z, voxel_ratio):
    """Crop dimensions to the largest multiple of voxel_ratio.

    Parameters
    ----------
    X, Y, Z : int
        Spatial dimensions.
    voxel_ratio : int
        Number of HiP-CT voxels per dMRI voxel.

    Returns
    -------
    tuple of (int, int, int)
    """
    return (
        X - (X % voxel_ratio),
        Y - (Y % voxel_ratio),
        Z - (Z % voxel_ratio),
    )


def compute_gfa_sh(coef, sh0_index=0):
    """Compute generalised fractional anisotropy from SH coefficients.

    Parameters
    ----------
    coef : ndarray
        SH coefficients (last dimension = coefficients). Must use a
        normalised SH basis.
    sh0_index : int, optional
        Index of the 0th-order SH coefficient.

    Returns
    -------
    gfa : ndarray
        GFA values in [0, 1].
    """
    coef_sq = coef ** 2
    numer = coef_sq[..., sh0_index]
    denom = coef_sq.sum(-1)
    allzero = denom == 0
    numer = numer + allzero
    denom = denom + allzero
    return np.sqrt(1.0 - (numer / denom))


def get_voxel_ratio(hipct_resolution, dmri_resolution):
    """Return the number of HiP-CT voxels that fit in one dMRI voxel."""
    voxel_ratio = int(dmri_resolution // hipct_resolution)
    print(f"Number of HiP-CT voxels in 1 dMRI voxel: {voxel_ratio}")
    return voxel_ratio


def compute_FA(eigenvalues):
    """Compute fractional anisotropy from eigenvalues.

    Parameters
    ----------
    eigenvalues : ndarray, shape (3, x, y, z)
        Sorted eigenvalues where eigenvalues[0] >= eigenvalues[1] >= eigenvalues[2].

    Returns
    -------
    FA : ndarray, shape (x, y, z)
        Fractional anisotropy in [0, 1]. NaN/Inf values are replaced with 0.
    """
    assert eigenvalues.shape[0] == 3 and eigenvalues.ndim == 4, \
        "Eigenvalues should be of shape (3, x, y, z)"

    lambda_mean = np.mean(eigenvalues, axis=0)
    numerator = np.sum((eigenvalues - lambda_mean) ** 2, axis=0)
    denominator = np.sum(eigenvalues ** 2, axis=0)

    FA = np.sqrt((3 / 2) * (numerator / denominator))
    FA = np.nan_to_num(FA, nan=0.0, posinf=0.0, neginf=0.0)

    return FA


def validate_eigenvalue_order(eigenvalues):
    """Check whether eigenvalues are ordered lambda_1 >= lambda_2 >= lambda_3 everywhere.

    Parameters
    ----------
    eigenvalues : ndarray, shape (3, X, Y, Z)

    Returns
    -------
    bool
        True if the ordering holds for all voxels.
    """
    return bool(
        np.all(eigenvalues[0] >= eigenvalues[1])
        and np.all(eigenvalues[1] >= eigenvalues[2])
    )


def extract_rotation_matrix(affine_matrix):
    """Extract the pure rotation component from a 4x4 affine matrix via SVD.

    Decomposes the upper-left 3x3 block into U @ diag(S) @ Vt, then returns
    R = U @ Vt with sign correction to ensure det(R) > 0.

    Parameters
    ----------
    affine_matrix : ndarray, shape (4, 4)

    Returns
    -------
    R : ndarray, shape (3, 3)
        Orthogonal rotation matrix.
    """
    A = affine_matrix[0:3, 0:3].copy()
    U, Svals, Vt = np.linalg.svd(A)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def rotate_vector_field(rot_mat, vector_field):
    """Rotate a 3D vector field using a rotation matrix.

    Parameters
    ----------
    rot_mat : ndarray, shape (3, 3)
    vector_field : ndarray, shape (3, x, y, z)

    Returns
    -------
    rotated : ndarray, shape (3, x, y, z)
    """
    return np.einsum('ij,jxyz->ixyz', rot_mat, vector_field)
