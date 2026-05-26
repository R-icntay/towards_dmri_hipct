"""Adapted from the structure-tensor package: https://github.com/Skielex/structure-tensor"""

import numpy as np


def eig_special_3d(S, full=False):
    """Eigensolution for symmetric real 3-by-3 matrices.

    Arguments:
        S: ndarray
            A floating point array with shape (6, ...) containing structure tensor.
            Use float64 to avoid numerical errors. When using lower precision, ensure
            that the values of S are not very small/large.
        full: bool, optional
            A flag indicating that all three eigenvalues should be returned.

    Returns:
        val: ndarray
            An array with shape (3, ...) containing sorted eigenvalues
        vec: ndarray
            An array with shape (3, ...) containing eigenvector corresponding to
            the smallest eigenvalue. If full, vec has shape (3, 3, ...) and contains
            all three eigenvectors.

    More:
        An analytic solution of eigenvalue problem for real symmetric matrix,
        using an affine transformation and a trigonometric solution of third
        order polynomial. See https://en.wikipedia.org/wiki/Eigenvalue_algorithm
        which refers to Smith's algorithm https://dl.acm.org/citation.cfm?id=366316.

    Authors: vand@dtu.dk, 2019; niejep@dtu.dk, 2019-2020
    """
    S = np.asarray(S)

    if not np.issubdtype(S.dtype, np.floating):
        raise ValueError('S must be floating point type.')

    input_shape = S.shape
    S = S.reshape(6, -1)

    v = np.array([[2 * np.pi / 3], [4 * np.pi / 3]], dtype=S.dtype)

    if full:
        val = np.empty((3, ) + S.shape[1:], dtype=S.dtype)
        vec = np.empty((9, ) + S.shape[1:], dtype=S.dtype)
        tmp = np.empty((4, ) + S.shape[1:], dtype=S.dtype)
        B03 = val
        B36 = vec[:3]
    else:
        val = np.empty((3, ) + S.shape[1:], dtype=S.dtype)
        vec = np.empty((3, ) + S.shape[1:], dtype=S.dtype)
        tmp = np.empty((4, ) + S.shape[1:], dtype=S.dtype)
        B03 = val
        B36 = vec

    B0 = B03[0]
    B1 = B03[1]
    B2 = B03[2]
    B3 = B36[0]
    B4 = B36[1]
    B5 = B36[2]

    # Compute q, mean of diagonal (np.mean has precision issues).
    q = np.add(S[0], S[1], out=tmp[0])
    q += S[2]
    q /= 3

    Sq = np.subtract(S[:3], q, out=B03)

    s = np.einsum('ij,ij->j', S[3:], S[3:], out=tmp[1])
    s *= 2

    p = np.einsum('ij,ij->j', Sq, Sq, out=tmp[2])
    del Sq
    p += s

    p *= 1 / 6
    np.sqrt(p, out=p)

    p_inv = s
    del s
    p_inv[:] = 0
    np.divide(1, p, out=p_inv, where=p != 0)

    B03 *= p_inv
    np.multiply(S[3:], p_inv, out=B36)

    d = np.prod(B03, axis=0, out=tmp[3])

    d_tmp = p_inv
    del p_inv
    np.multiply(B2, B3, d_tmp)
    d_tmp *= B3
    d -= d_tmp
    np.multiply(B4, B4, out=d_tmp)
    d_tmp *= B1
    d -= d_tmp
    np.prod(B36, axis=0, out=d_tmp)
    d_tmp *= 2
    d += d_tmp
    np.multiply(B5, B5, out=d_tmp)
    d_tmp *= B0
    d -= d_tmp
    d *= 0.5
    np.clip(d, -1, 1, out=d)

    phi = d
    del d
    phi = np.arccos(phi, out=phi)
    phi /= 3

    del B03, B36, B0, B1, B2, B3, B4, B5

    np.add(v, phi[np.newaxis], out=val[:2])
    val[2] = phi
    np.cos(val, out=val)
    p *= 2
    val *= p
    val += q

    del q
    del p
    del phi
    del d_tmp

    if full:
        l = val
        vec = vec.reshape(3, 3, -1)
        vec_tmp = tmp[:3]
    else:
        l = val[0]
        vec_tmp = tmp[2]

    u = np.subtract(S[2], l, out=vec[0])
    np.multiply(u, S[3], out=u)
    u_tmp = np.multiply(S[4], S[5], out=tmp[3])
    np.subtract(u_tmp, u, out=u)

    v = np.subtract(S[1], l, out=vec_tmp)
    np.multiply(v, S[4], out=v)
    v_tmp = np.multiply(S[3], S[5], out=tmp[3])
    np.subtract(v_tmp, v, out=v)

    w = np.subtract(S[0], l, out=vec[2])
    np.multiply(w, S[5], out=w)
    w_tmp = np.multiply(S[3], S[4], out=tmp[3])
    np.subtract(w_tmp, w, out=w)

    vec[1] = u
    np.multiply(u, v, out=vec[0])
    u = vec[1]
    np.multiply(u, w, out=vec[1])
    np.multiply(v, w, out=vec[2])

    del u
    del v
    del w
    del l

    if full:
        l = np.einsum('ijk,ijk->jk', vec, vec, out=vec_tmp)[:, np.newaxis]
        vec = np.swapaxes(vec, 0, 1)
    else:
        l = np.einsum('ij,ij->j', vec, vec, out=vec_tmp)

    np.sqrt(l, out=l)
    vec /= l

    return val.reshape(val.shape[:-1] + input_shape[1:]), vec.reshape(vec.shape[:-1] + input_shape[1:])
