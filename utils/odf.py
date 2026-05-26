"""Spherical harmonic representation of fiber orientation distribution functions.

Projects 3D vector fields (e.g. eigenvectors from structure tensor analysis)
onto a real spherical harmonic basis to produce fODF coefficients compatible
with dipy's SH conventions.
"""

import numpy as np
import scipy.special as sps
from dipy.core.sphere import Sphere
from dipy.core.geometry import cart2sphere
from dipy.utils.compatibility import check_max_version
from dipy.reconst.shm import calculate_max_order
from sklearn.neighbors import NearestNeighbors
from fiberorient.util import make_sphere
from typing import Literal


class vector_field_SH:
    def __init__(
        self,
        sh_order_max: int = 2,
        use_descoteaux07: bool = True
    ):
        assert sh_order_max >= 0 and sh_order_max % 2 == 0, "sh_order_max must be an even integer >= 0"
        self.sh_order_max = sh_order_max
        self.use_descoteaux07 = use_descoteaux07

        m_values, l_values = self.sph_harm_ind_list(sh_order_max)
        self.m_values, self.l_values = m_values, l_values

    @staticmethod
    def sph_harm_ind_list(sh_order_max: int) -> (np.ndarray, np.ndarray):
        assert sh_order_max >= 0 and sh_order_max % 2 == 0, "sh_order_max must be an even integer >= 0"

        l_values = np.arange(0, sh_order_max + 1, 2, dtype=int)
        ncoef = int(((sh_order_max + 2) * (sh_order_max + 1)) // 2)

        l_list = np.repeat(l_values, repeats=2 * l_values + 1)
        m_list = []

        for ll in l_values:
            m_vals = np.arange(-ll, ll + 1)
            m_list.extend(m_vals.tolist())

        m_list = np.array(m_list, dtype=int)

        assert len(l_list) == ncoef and len(m_list) == ncoef, \
            f"l_list {len(l_list)} and m_list {len(m_list)} must have length {ncoef}"
        return m_list, l_list

    def fit(self, vector_field: np.ndarray,
            method: Literal["spherical_histogram", "dirac_delta"] = "spherical_histogram",
            K=None, n_bins=6500):
        """Compute SH coefficients by projecting 3D vectors onto the SH basis.

        Supports histogram-based or Dirac-sum projection.
        """
        if (vector_field.ndim != 2) or (vector_field.shape[1] != 3):
            vector_field = vector_field.reshape(-1, 3)

        if K is None:
            K = vector_field.shape[0]

        if method == "spherical_histogram":
            self.shm_coeffs = self.fit_spherical_histogram(vector_field, K, n_bins)
        elif method == "dirac_delta":
            self.shm_coeffs = self.fit_dirac_delta(vector_field, K)
        else:
            raise ValueError(f"Unknown method {method}. Supported: 'spherical_histogram', 'dirac_delta'")

        return self

    @staticmethod
    def make_sphere(n):
        """Return a dipy Sphere with *n* Fibonacci-sampled points.

        Parameters
        ----------
        n : int
            Number of points on the sphere.

        Returns
        -------
        sphere : dipy Sphere
        """
        z = np.linspace(1 - 1 / n, -1 + 1 / n, num=n)
        polar = np.arccos(z)
        azim = np.mod((np.pi * (3.0 - np.sqrt(5.0))) *
                       np.arange(n), 2 * np.pi) - np.pi
        azim[azim < 0] += 2 * np.pi

        sphere = Sphere(theta=polar, phi=azim)
        return sphere

    @staticmethod
    def vector_to_sph_hist(vectors, n_bins):
        sphere = make_sphere(n_bins)
        hist_points = np.stack((sphere.x, sphere.y, sphere.z), axis=-1)
        nbrs = NearestNeighbors(n_neighbors=1,
                                algorithm='ball_tree',
                                leaf_size=5).fit(hist_points)

        indices = nbrs.kneighbors(vectors, return_distance=False)
        hist = np.bincount(indices.flatten(), minlength=sphere.theta.size)
        return hist

    @staticmethod
    def spherical_harmonics(m_values, l_values, azimuth, polar):
        """Compute complex spherical harmonics Y_l^m at given angles.

        Parameters
        ----------
        m_values : array, |m| <= l
        l_values : array, l >= 0
        azimuth : array, [0, 2*pi]
        polar : array, [0, pi]

        Returns
        -------
        y_mn : complex ndarray
        """
        if check_max_version("scipy", "1.15.0", strict=True):
            sph_harmonics = sps.sph_harm(m_values, l_values, azimuth, polar).astype(complex)
        else:
            degree = (
                l_values.astype(int) if isinstance(l_values, np.ndarray) else int(l_values)
            )
            order = (
                m_values.astype(int) if isinstance(m_values, np.ndarray) else int(m_values)
            )
            sph_harmonics = sps.sph_harm_y(degree, order, polar, azimuth).astype(complex)

        return sph_harmonics

    def real_sh_descoteaux_from_index(self, m_values, l_values, azimuth, polar, legacy=True):
        r"""Compute real spherical harmonics using the Descoteaux et al. (2007) basis.

        .. math::

            Y_l^m =
            \begin{cases}
                \sqrt{2} \cdot \Im(Y_l^m) & \text{if } m > 0 \\
                Y^0_l & \text{if } m = 0 \\
                \sqrt{2} \cdot \Re(Y_l^m) & \text{if } m < 0
            \end{cases}

        Parameters
        ----------
        m_values : array, |m| <= l
        l_values : array, l >= 0
        azimuth : array, [0, 2*pi]
        polar : array, [0, pi]
        legacy : bool
            If True, uses DIPY's legacy descoteaux07 implementation (|m| for m < 0).

        Returns
        -------
        real_sh : float ndarray
        """
        if legacy:
            sh = self.spherical_harmonics(m_values=np.abs(m_values), l_values=l_values,
                                          azimuth=azimuth, polar=polar)
        else:
            raise NotImplementedError("The non-legacy descoteaux07 basis is not implemented yet.")

        real_sh = np.where(m_values > 0, sh.imag, sh.real)
        real_sh *= np.where(m_values == 0, 1.0, np.sqrt(2))

        return real_sh

    def real_sh_tournier_from_index(self, m_values, l_values, azimuth, polar):
        r"""Compute real spherical harmonics using the Tournier et al. (2007) basis.

        .. math::

            Y_l^m =
            \begin{cases}
                \sqrt{2} \cdot \Re(Y_l^m) & \text{if } m > 0 \\
                Y^0_l & \text{if } m = 0 \\
                \sqrt{2} \cdot \Im(Y_l^{|m|}) & \text{if } m < 0
            \end{cases}

        Parameters
        ----------
        m_values : array, |m| <= l
        l_values : array, l >= 0
        azimuth : array, [0, 2*pi]
        polar : array, [0, pi]

        Returns
        -------
        real_sh : float ndarray
        """
        sh = self.spherical_harmonics(m_values=np.abs(m_values), l_values=l_values,
                                      azimuth=azimuth, polar=polar)

        real_sh = np.where(m_values < 0, sh.imag, sh.real)
        real_sh *= np.where(m_values == 0, 1, np.sqrt(2))

        return real_sh

    def fit_spherical_histogram(self, vector_field, K, n_bins):
        if K is None:
            K = vector_field.shape[0]

        spherical_hist = self.vector_to_sph_hist(vector_field, n_bins)
        spherical_hist = spherical_hist[:, None]

        sphere = self.make_sphere(n_bins)
        azimuth, polar = sphere.phi[:, None], sphere.theta[:, None]
        assert azimuth.max() <= 2 * np.pi and polar.max() <= np.pi, \
            "Azimuth must be in range [0, 2pi] and polar in range [0, pi]"
        assert polar.min() >= 0, "Polar must be in range [0, pi]"

        if self.use_descoteaux07:
            sph_harmonics = self.real_sh_descoteaux_from_index(
                m_values=self.m_values, l_values=self.l_values,
                azimuth=azimuth, polar=polar)
        else:
            sph_harmonics = self.real_sh_tournier_from_index(
                m_values=self.m_values, l_values=self.l_values,
                azimuth=azimuth, polar=polar)

        sph_harmonics = sph_harmonics.transpose(1, 0)

        sh_coeffs = np.matmul(sph_harmonics, spherical_hist) / K
        return sh_coeffs.flatten()

    def fit_dirac_delta(self, vector_field, K):
        """Treat each vector as a Dirac mass on the sphere and project onto the SH basis."""
        if K is None:
            K = vector_field.shape[0]

        r, polar, azimuth = self._cart2sphere(vector_field[:, 0], vector_field[:, 1], vector_field[:, 2])
        azimuth = np.mod(azimuth, 2 * np.pi)
        assert azimuth.max() <= 2 * np.pi and azimuth.min() >= 0, "Azimuth must be in range [0, 2pi]"
        assert polar.max() <= np.pi and polar.min() >= 0, "Polar must be in range [0, pi]"
        azimuth, polar = azimuth[:, None], polar[:, None]

        if self.use_descoteaux07:
            sph_harmonics = self.real_sh_descoteaux_from_index(
                m_values=self.m_values, l_values=self.l_values,
                azimuth=azimuth, polar=polar)
        else:
            sph_harmonics = self.real_sh_tournier_from_index(
                m_values=self.m_values, l_values=self.l_values,
                azimuth=azimuth, polar=polar)

        sph_harmonics = sph_harmonics.transpose(1, 0)

        sh_coeffs = np.matmul(sph_harmonics, np.ones((sph_harmonics.shape[1], 1))) / K
        return sh_coeffs.flatten()

    @staticmethod
    def _cart2sphere(x, y, z):
        r, polar, azimuth = cart2sphere(x, y, z)
        return r, polar, azimuth

    @staticmethod
    def _odf_on_sphere(shm_coeffs, sampling_matrix):
        """Evaluate the ODF on a sphere given SH coefficients and a sampling matrix.

        Parameters
        ----------
        shm_coeffs : ndarray, shape (n_coeffs,)
        sampling_matrix : ndarray, shape (n_points, n_coeffs)

        Returns
        -------
        odf : ndarray, shape (n_points,)
        """
        shm_coeffs = shm_coeffs.reshape(-1, 1)
        assert sampling_matrix.shape[1] == shm_coeffs.shape[0], \
            f"sampling_matrix shape {sampling_matrix.shape} and shm_coeffs shape {shm_coeffs.shape} are not compatible"
        odf = np.matmul(sampling_matrix, shm_coeffs)
        return odf.flatten()

    def sample_odf_on_sphere(self, sphere):
        """Sample the ODF on the given sphere.

        Parameters
        ----------
        sphere : dipy Sphere

        Returns
        -------
        odf : ndarray, shape (n_points,)
        """
        azimuth, polar = sphere.phi[:, None], sphere.theta[:, None]
        assert azimuth.max() <= 2 * np.pi and polar.max() <= np.pi, \
            "Azimuth must be in range [0, 2pi] and polar in range [0, pi]"

        if self.use_descoteaux07:
            sampling_matrix = self.real_sh_descoteaux_from_index(
                m_values=self.m_values, l_values=self.l_values,
                azimuth=azimuth, polar=polar)
        else:
            sampling_matrix = self.real_sh_tournier_from_index(
                m_values=self.m_values, l_values=self.l_values,
                azimuth=azimuth, polar=polar)

        return self._odf_on_sphere(self.shm_coeffs, sampling_matrix)

    @property
    def shm_coeff(self):
        """The spherical harmonics coefficients computed after calling fit()."""
        assert hasattr(self, 'shm_coeffs'), "You must call fit() before accessing this property"
        return self.shm_coeffs

    @property
    def descoteaux_to_tournier_sh_coeffs(self):
        """Convert SH coefficients from legacy-descoteaux07 to normalized-tournier07 basis."""
        assert hasattr(self, 'shm_coeffs'), "You must call fit() before accessing this property"
        sh_coeffs = self.shm_coeffs.copy().reshape(1, -1)
        return self.convert_sh_descoteaux_tournier(sh_coeffs).flatten()

    @property
    def tournier_to_descoteaux_sh_coeffs(self):
        """Convert SH coefficients from normalized-tournier07 to legacy-descoteaux07 basis."""
        assert hasattr(self, 'shm_coeffs'), "You must call fit() before accessing this property"
        sh_coeffs = self.shm_coeffs.copy().reshape(1, -1)
        return self.convert_sh_descoteaux_tournier(sh_coeffs).flatten()

    def convert_sh_descoteaux_tournier(self, sh_coeffs):
        """Convert SH coefficients between legacy-descoteaux07 and tournier07.

        This conversion is its own inverse: it can convert in either direction
        (legacy-descoteaux to non-legacy-tournier or vice versa). Useful for
        converting SH representations between DIPY and MRtrix3.

        Parameters
        ----------
        sh_coeffs : ndarray
            Array where the last dimension contains SH coefficients.

        Returns
        -------
        out_sh_coeffs : ndarray
            Coefficients expressed in the other SH basis.

        References
        ----------
        .. [1] Descoteaux et al. 2007
        .. [2] Tournier et al. 2019
        .. [3] https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html
        .. [4] https://github.com/dipy/dipy/discussions/2959#discussioncomment-7481675
        """
        sh_order_max = calculate_max_order(sh_coeffs.shape[-1])
        m_values, l_values = self.sph_harm_ind_list(sh_order_max)

        basis_indices = list(zip(m_values, l_values))
        basis_indices_permuted = list(zip(-m_values, l_values))

        permutation = [
            basis_indices.index(basis_indices_permuted[i])
            for i in range(len(basis_indices))
        ]

        return sh_coeffs[..., permutation]
