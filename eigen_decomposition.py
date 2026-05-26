import numpy as np

# Define function to compute eigenvalues and eigenvectors of structure tensor
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

    # Check data type. Must be floating point.
    if not np.issubdtype(S.dtype, np.floating):
        raise ValueError('S must be floating point type.')

    # Flatten S.
    input_shape = S.shape
    S = S.reshape(6, -1)

    # Create v vector.
    v = np.array([[2 * np.pi / 3], [4 * np.pi / 3]], dtype=S.dtype)

    # Computing eigenvalues.

    # Allocate vec and val. We will use them for intermediate computations as well.
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

    # Views for B.
    B0 = B03[0]
    B1 = B03[1]
    B2 = B03[2]
    B3 = B36[0]
    B4 = B36[1]
    B5 = B36[2]

    # Compute q, mean of diagonal. We need to use q multiple times later.
    # Using np.mean has precision issues.
    q = np.add(S[0], S[1], out=tmp[0])
    q += S[2]
    q /= 3

    # Compute S minus q. Insert it directly into B where it'll stay.
    Sq = np.subtract(S[:3], q, out=B03)

    # Compute s, off-diagonal elements. Store in part of B not yet used.
    s = np.einsum('ij,ij->j', S[3:], S[3:], out=tmp[1])
    s *= 2

    # Compute p.
    p = np.einsum('ij,ij->j', Sq, Sq, out=tmp[2])
    del Sq  # Last use of Sq.
    p += s

    p *= 1 / 6
    np.sqrt(p, out=p)

    # Compute inverse p, while avoiding 0 division.
    # Reuse s allocation and delete s to ensure we don't efter it's been reused.
    p_inv = s
    del s
    p_inv[:] = 0
    np.divide(1, p, out=p_inv, where=p != 0)

    # Compute B. First part is already filled.
    B03 *= p_inv
    np.multiply(S[3:], p_inv, out=B36)

    # Compute d, determinant of B.
    d = np.prod(B03, axis=0, out=tmp[3])

    # Reuse allocation for p_inv and delete variable.
    d_tmp = p_inv
    del p_inv
    # Computation of d.
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
    # Ensure -1 <= d/2 <= 1.
    np.clip(d, -1, 1, out=d)

    # Compute phi. Beware that we reuse d variable!
    phi = d
    del d
    phi = np.arccos(phi, out=phi)
    phi /= 3

    # Compute val, ordered eigenvalues. Resuing B allocation.
    del B03, B36, B0, B1, B2, B3, B4, B5

    np.add(v, phi[np.newaxis], out=val[:2])
    val[2] = phi
    np.cos(val, out=val)
    p *= 2
    val *= p
    val += q

    # Remove all variable using tmp allocation.
    del q
    del p
    del phi
    del d_tmp

    # Computing eigenvectors -- either only one or all three.
    if full:
        l = val
        vec = vec.reshape(3, 3, -1)
        vec_tmp = tmp[:3]
    else:
        l = val[0]
        vec_tmp = tmp[2]

    # Compute vec. The tmp variable can be reused.

    # u = S[4] * S[5] - (S[2] - l) * S[3]
    u = np.subtract(S[2], l, out=vec[0])
    np.multiply(u, S[3], out=u)
    u_tmp = np.multiply(S[4], S[5], out=tmp[3])
    np.subtract(u_tmp, u, out=u)
    # Put values of u into vector 2 aswell.

    # v = S[3] * S[5] - (S[1] - l) * S[4]
    v = np.subtract(S[1], l, out=vec_tmp)
    np.multiply(v, S[4], out=v)
    v_tmp = np.multiply(S[3], S[5], out=tmp[3])
    np.subtract(v_tmp, v, out=v)

    # w = S[3] * S[4] - (S[0] - l) * S[5]
    w = np.subtract(S[0], l, out=vec[2])
    np.multiply(w, S[5], out=w)
    w_tmp = np.multiply(S[3], S[4], out=tmp[3])
    np.subtract(w_tmp, w, out=w)

    vec[1] = u
    np.multiply(u, v, out=vec[0])
    u = vec[1]
    np.multiply(u, w, out=vec[1])
    np.multiply(v, w, out=vec[2])

    # Remove u, v, w and l variables.
    del u
    del v
    del w
    del l

    # Normalizing -- depends on number of vectors.
    if full:
        # vec is [x1 x2 x3, y1 y2 y3, z1 z2 z3]
        l = np.einsum('ijk,ijk->jk', vec, vec, out=vec_tmp)[:, np.newaxis]
        vec = np.swapaxes(vec, 0, 1)
    else:
        # vec is [x1 y1 z1] = v1
        l = np.einsum('ij,ij->j', vec, vec, out=vec_tmp)

    np.sqrt(l, out=l)
    vec /= l

    return val.reshape(val.shape[:-1] + input_shape[1:]), vec.reshape(vec.shape[:-1] + input_shape[1:])




# def sta_analysis(memmap_volume, z_idx, y_idx, x_idx, padding = pad, sigma = sigma,
#     rho = rho, truncate = truncate, save_dir = None):

#     """
#     Convert a directory of images to a nifti file

#     Parameters
#     ----------
#     image_dir : str
#         Path to the directory containing the images/memmap file
#     extension : str
#         Extension of the images to be converted

#     Returns
#     -------
#     None


#     """

#     ## Assert save_dir is Path object
#     # assert isinstance(save_dir, Path), "save_dir must be a pathlib Path object"
#     # assert save_dir.exists(), "save_dir must exist"
    

#     ## Assert eig_special_3d is callable
#     assert callable(eig_special_3d), "eig_special_3d has to be a callable function"

#     # Extract start and end indices
#     start_z, end_z = max(0, z_idx[0] - padding), min(memmap_volume.shape[0], z_idx[1] + padding)
#     start_y, end_y = max(0, y_idx[0] - padding), min(memmap_volume.shape[1], y_idx[1] + padding)
#     start_x, end_x = max(0, x_idx[0] - padding), min(memmap_volume.shape[2], x_idx[1] + padding)

#     # Create expected dummy image for verification
#     dummy_image = np.zeros((x_idx[1]-x_idx[0], y_idx[1]-y_idx[0], z_idx[1]-z_idx[0]))
#     print(f"Expected dummy image shape for verification: {dummy_image.shape}")

#     # Extract the chunk from the memmap
#     print(f"Extracting chunk with padding: Z({start_z}:{end_z}), Y({start_y}:{end_y}), X({start_x}:{end_x})")
#     chunk_volume = memmap_volume[start_z:end_z, start_y:end_y, start_x:end_x].copy()
#     print(f"Extracted chunk shape ZYX: {chunk_volume.shape}")


#     # # Print shape of the numpy array
#     # print(f"\nShape of volumes ZYX before re-orientation to match dmri: {volume.shape}")

#     ## Transpose the volume to match dmri orientation
#     ## Transpose axes from zyx to xyz
#     chunk_volume = chunk_volume.transpose(2, 1, 0)

#     # Print shape of the numpy array
#     print(f"Shape of volumes after re-orientation to XYZ: {chunk_volume.shape}")

#     ##### ---------------------------------------------- Compute the structure tensor ---------------------------------------------------------- #####



#     print(f"\nStarting Structure Tensor computation")

#     print(f"Computing gradient by convolution with Gaussian derivative")
            

#     t_st = perf_counter()
#     # # Computing derivatives (scipy implementation truncates filter at 4 sigma).
#     Vx = filters.gaussian_filter(chunk_volume, sigma, order=[1, 0, 0], mode='nearest', truncate=truncate) # X, y, z
#     Vy = filters.gaussian_filter(chunk_volume, sigma, order=[0, 1, 0], mode='nearest', truncate=truncate) # x, Y, z
#     Vz = filters.gaussian_filter(chunk_volume, sigma, order=[0, 0, 1], mode='nearest', truncate=truncate) # x, y, Z
#     t_end = perf_counter()


#     print(f"Time taken to compute gradients: {(t_end - t_st)/60:.2f} minutes")
#     chunk_volume_shape = chunk_volume.shape
#     chunk_volume = None  # Free memory

#     t_st = perf_counter()
#     print(f"Computing the structure tensor")
#     S = np.empty((6, ) + chunk_volume_shape, dtype = np.float64)

#     # Integrating elements of structure tensor (scipy uses sequence of 1D).

#     np.multiply(Vx, Vx, out = S[0])
#     filters.gaussian_filter(S[0], rho, mode='nearest', output=S[0], truncate=truncate) # S[0] = Vx*Vx

#     np.multiply(Vy, Vy, out = S[1])
#     filters.gaussian_filter(S[1], rho, mode='nearest', output=S[1], truncate=truncate) # S[1] = Vy*Vy

#     np.multiply(Vz, Vz, out = S[2])
#     filters.gaussian_filter(S[2], rho, mode='nearest', output=S[2], truncate=truncate) # S[2] = Vz*Vz

#     np.multiply(Vx, Vy, out = S[3])
#     filters.gaussian_filter(S[3], rho, mode='nearest', output=S[3], truncate=truncate) # S[3] = Vx*Vy

#     np.multiply(Vx, Vz, out = S[4])
#     filters.gaussian_filter(S[4], rho, mode='nearest', output=S[4], truncate=truncate) # S[4] = Vx*Vz

#     np.multiply(Vy, Vz, out = S[5])
#     filters.gaussian_filter(S[5], rho, mode='nearest', output=S[5], truncate=truncate) # S[5] = Vy*Vz

    

#     t_end = perf_counter()
#     print(f"Time taken to compute all S components: {(t_end - t_st)/60:.2f} minutes")
#     print("All S components computed.")
#     Vx, Vy, Vz = None, None, None  # Free memory


   

#     # Extracting non-overlap portion of the structure tensor
#     print(f"Extracting non-overlap portion of chunk")
#     print(f"Shape before removing padding: {S.shape}")
#     S = S[:, (x_idx[0] - start_x): -(end_x - x_idx[1]), (y_idx[0] - start_y): -(end_y - y_idx[1]), (z_idx[0] - start_z): -(end_z - z_idx[1])]
#     print(f"Shape of S after removing padding: {S.shape}")
    
    

 

#     assert S.shape == (6, dummy_image.shape[0], dummy_image.shape[1], dummy_image.shape[2]), f"Unexpected S shape: {S.shape}"
#     assert S.dtype == np.float64, "S dtype should be float64"





#     #### -------------------------------------------------- End of structure tensor computation -------------------------------------------------####
#     ## Compute eigenvalues and eigenvectors of structure tensor for each voxel in chunk
#     print(f"Computing eigenvalues and eigenvectors for chunk")
#     t_st = perf_counter()
#     eigen_values, eigen_vectors = eig_special_3d(S, full = False)
#     t_end = perf_counter()
#     print(f"Time taken for eigen computation for chunk: {(t_end - t_st)/60:.2f} minutes")


   

#     # Save the eigen values and eigen vectors
#     print(f"Saving eigen values and eigen vectors for chunk...")
#     # eval_memmap[:, x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], z_idx[0]:z_idx[1]] = eigen_values.astype(np.float32) # *
#     # evec_memmap[:, x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], z_idx[0]:z_idx[1]] = eigen_vectors.astype(np.float32) # *
#     t_st = perf_counter()
#     eval_memmap[:, x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], :] = eigen_values.astype(np.float32) # *
#     evec_memmap[:, x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], :] = eigen_vectors.astype(np.float32) # *
    
#     eval_memmap.flush()
#     evec_memmap.flush()
#     t_end = perf_counter()
#     print(f"Time taken to save eigenvalues and eigenvectors for chunk: {(t_end - t_st)/60:.2f} minutes")
    

#     print(f"Finished processing chunk X[{x_idx[0]}:{x_idx[1]}] Y[{y_idx[0]}:{y_idx[1]}] Z[{z_idx[0]}:{z_idx[1]}].\n")
#     # print('_ '*50)
#     gc.collect()

#     return

