"""Visualization utilities for tractography and fODF rendering.

Provides streamline coloring, 2D overlay rendering with fury/VTK,
and streamline sampling for figure generation.
"""

import numpy as np
import vtk
from fury import actor, window
from nibabel.streamlines import ArraySequence


# ---------------------------------------------------------------------------
# Coloring utilities
# ---------------------------------------------------------------------------

def orient2rgb(v):
    """Map vector directions to RGB colours via absolute orientation.

    Parameters
    ----------
    v : ndarray, shape (3,) or (N, 3)

    Returns
    -------
    orient : ndarray, same shape as *v*, values in [0, 1]
    """
    if v.ndim == 1:
        r = np.linalg.norm(v)
        orient = np.abs(np.divide(v, r, where=r != 0))
    elif v.ndim == 2:
        orientn = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
        orientn.shape = orientn.shape + (1,)
        orient = np.abs(np.divide(v, orientn, where=orientn != 0))
    else:
        raise ValueError("Expected array with shape (3,) or (N, 3)")
    return orient


def orient2rbg(v):
    """Like :func:`orient2rgb` but with green and blue channels swapped."""
    orient = orient2rgb(v)
    if orient.ndim == 1:
        orient[1], orient[2] = orient[2], orient[1]
    else:
        orient[:, [1, 2]] = orient[:, [2, 1]]
    return orient


def line_colors(streamlines, *, cmap="rgb_standard"):
    """Compute per-streamline RGB colours from endpoint directions.

    Parameters
    ----------
    streamlines : sequence of ndarrays
    cmap : {'rgb_standard', 'rbg_standard', 'boys_standard'}

    Returns
    -------
    colors : ndarray, shape (N_total_points, 3)
    """
    if cmap == "rgb_standard":
        col_list = [orient2rgb(s[-1] - s[0]) for s in streamlines]
    elif cmap == "rbg_standard":
        col_list = [orient2rbg(s[-1] - s[0]) for s in streamlines]
    elif cmap == "boys_standard":
        from dipy.viz.colormap import boys2rgb
        col_list = [boys2rgb(s[-1] - s[0]) for s in streamlines]
    else:
        raise ValueError(f"Unknown cmap: {cmap}")
    return np.vstack(col_list)


def numpy_to_vtk_lookup_table(cmap_array):
    """Convert a numpy colormap array to a VTK lookup table.

    Parameters
    ----------
    cmap_array : ndarray, shape (N, 3), dtype uint8
        RGB values in [0, 255].

    Returns
    -------
    lut : vtkLookupTable
    """
    ncolors = cmap_array.shape[0]
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(ncolors)
    lut.Build()
    for i in range(ncolors):
        r, g, b = cmap_array[i] / 255.0
        lut.SetTableValue(i, r, g, b, 1.0)
    return lut


# ---------------------------------------------------------------------------
# Streamline sampling
# ---------------------------------------------------------------------------

def sample_streamlines(streamlines, sample_percentage=0.10):
    """Deterministically subsample streamlines by a fixed percentage.

    Parameters
    ----------
    streamlines : ArraySequence or list
    sample_percentage : float
        Fraction of streamlines to keep (0, 1].

    Returns
    -------
    sampled : ArraySequence
    """
    remaining = int(len(streamlines) * sample_percentage)
    step = max(1, len(streamlines) // remaining)
    indices = np.arange(0, len(streamlines), step=step)
    sampled = ArraySequence([streamlines[i] for i in indices if len(streamlines[i])])
    print(f"Sampled {len(sampled)} streamlines from {len(streamlines)} "
          f"({(1 - sample_percentage) * 100:.0f}% reduction)")
    return sampled


# ---------------------------------------------------------------------------
# 2D overlay rendering
# ---------------------------------------------------------------------------

def show_tracts_img_2d(streamlines, opacity, dmri_supervoxel_dim=(13, 13),
                       img=None, display_image=True, min_pix=1000,
                       offset=0.25, interactive=False, colors=None,
                       cmap="rgb_standard", lookup_colormap=None,
                       magnification=1.0, out_path=None):
    """Render streamlines overlaid on a 2D image using fury/VTK.

    Parameters
    ----------
    streamlines : list of ndarrays
    opacity : float
    dmri_supervoxel_dim : tuple
        (nx, ny) grid size for the supervoxel field.
    img : ndarray or None
        2D image (X, Y) to display as background.
    display_image : bool
    min_pix : int
        Minimum pixel dimension for the output window.
    offset : float
        Z-offset for the background image plane.
    interactive : bool
        If True, open an interactive window.
    colors : ndarray or None
        Pre-computed per-point colours; if None, computed from *cmap*.
    cmap : str
    lookup_colormap : vtkLookupTable or None
    magnification : float
    out_path : str or None
        If provided (and not interactive), save a screenshot.

    Returns
    -------
    scene : fury Scene
    """
    if not isinstance(min_pix, int):
        min_pix = int(min_pix)

    scene = window.Scene()
    scene.SetBackground(0, 0, 0)

    odf_nx, odf_ny = dmri_supervoxel_dim

    if img is not None:
        if img.ndim != 2:
            raise ValueError("img must be 2D")
        nx, ny = img.shape
        img = np.expand_dims(img, -1)
        aff = np.eye(4)
        aff[0, 0] = -odf_nx / nx
        aff[1, 1] = -odf_ny / ny
        aff[0, 3] = odf_nx - 0.5
        aff[1, 3] = odf_ny - 0.5
        aff[2, 3] = offset

        image_actor = actor.slicer(img, interpolation='nearest',
                                   affine=aff, lookup_colormap=lookup_colormap)
        image_actor.display(None, None, 0)
        if display_image:
            scene.add(image_actor)
    else:
        nx, ny = odf_nx, odf_ny

    line_actor = actor.line(
        streamlines, opacity=opacity,
        colors=line_colors(streamlines, cmap=cmap) if colors is None else colors,
    )
    scene.add(line_actor)
    scene.yaw(180)

    center_x = (odf_nx - 1) / 2.0
    center_y = (odf_ny - 1) / 2.0
    flip_mat = vtk.vtkMatrix4x4()
    flip_mat.Identity()
    flip_mat.SetElement(0, 0, -1)
    flip_mat.SetElement(0, 3, 2 * center_x)
    flip_mat.SetElement(1, 1, -1)
    flip_mat.SetElement(1, 3, 2 * center_y)
    line_actor.SetUserMatrix(flip_mat)

    if ny <= nx:
        outy = min_pix
        outx = int(nx / ny * min_pix)
    else:
        outx = min_pix
        outy = int(ny / nx * min_pix)
    size = (outx, outy)

    if img is not None:
        bounds = image_actor.GetBounds()
        x_center = (bounds[0] + bounds[1]) / 2.0
        y_center = (bounds[2] + bounds[3]) / 2.0
        z_center = (bounds[4] + bounds[5]) / 2.0
        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]
        max_extent = max(width, height)
        view_angle_rad = np.deg2rad(scene.camera().GetViewAngle())
        distance = max_extent / (2 * np.tan(view_angle_rad / 2))
        scene.set_camera(focal_point=[x_center, y_center, z_center],
                         view_up=[0, 1, 0])
        scene.set_camera(position=[x_center, y_center, z_center - distance],
                         focal_point=[x_center, y_center, z_center],
                         view_up=[0, 1, 0])

    if interactive:
        window.show(scene, size=size, reset_camera=False,
                    png_magnify=magnification, order_transparent=True)
    elif out_path is not None:
        window.record(scene, size=size, reset_camera=False,
                      out_path=out_path, magnification=int(magnification))

    return scene


def show_odf_img_2d(odf_array, sphere, img=None, min_pix=1000,
                    slicer_args=None, offset=0.25, interactive=False,
                    lookup_colormap=None, magnification=1.0, out_path=None):
    """Render ODF glyphs overlaid on a 2D image using fury/VTK.

    Parameters
    ----------
    odf_array : ndarray, shape (nx, ny, n_points)
    sphere : dipy Sphere
    img : ndarray or None
        2D background image (X, Y).
    min_pix : int
    slicer_args : dict or None
    offset : float
    interactive : bool
    lookup_colormap : vtkLookupTable or None
    magnification : float
    out_path : str or None

    Returns
    -------
    scene : fury Scene
    """
    if not isinstance(min_pix, int):
        min_pix = int(min_pix)

    if odf_array.ndim != 3:
        raise ValueError("odf_array must have shape (nx, ny, n_points)")

    if slicer_args is None:
        slicer_args = {'norm': False, 'colormap': None, 'scale': 0.28}

    scene = window.Scene()
    scene.SetBackground(0, 0, 0)

    odf_nx, odf_ny = odf_array.shape[:2]

    if img is not None:
        if img.ndim != 2:
            raise ValueError("img must be 2D")
        nx, ny = img.shape
        img = np.expand_dims(img, -1)
        aff = np.eye(4)
        aff[0, 0] = -odf_nx / nx
        aff[1, 1] = -odf_ny / ny
        aff[0, 3] = odf_nx
        aff[1, 3] = odf_ny
        aff[2, 3] = offset

        image_actor = actor.slicer(img, interpolation='nearest',
                                   affine=aff, lookup_colormap=lookup_colormap)
        image_actor.display(None, None, 0)
        scene.add(image_actor)
    else:
        nx, ny = odf_nx, odf_ny

    odf_array = np.expand_dims(odf_array, 2)
    odf_actor = actor.odf_slicer(odf_array, sphere=sphere, **slicer_args)
    scene.add(odf_actor)

    center_x = (odf_nx - 1) / 2.0
    center_y = (odf_ny - 1) / 2.0
    flip_mat = vtk.vtkMatrix4x4()
    flip_mat.Identity()
    flip_mat.SetElement(0, 0, -1)
    flip_mat.SetElement(0, 3, 2 * center_x)
    flip_mat.SetElement(1, 1, -1)
    flip_mat.SetElement(1, 3, 2 * center_y)
    odf_actor.SetUserMatrix(flip_mat)

    if ny <= nx:
        outy = min_pix
        outx = int(nx / ny * min_pix)
    else:
        outx = min_pix
        outy = int(ny / nx * min_pix)
    size = (outx, outy)

    if img is not None:
        bounds = image_actor.GetBounds()
        x_center = (bounds[0] + bounds[1]) / 2.0
        y_center = (bounds[2] + bounds[3]) / 2.0
        z_center = (bounds[4] + bounds[5]) / 2.0
        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]
        max_extent = max(width, height)
        view_angle_rad = np.deg2rad(scene.camera().GetViewAngle())
        distance = max_extent / (2 * np.tan(view_angle_rad / 2))
        scene.set_camera(focal_point=[x_center, y_center, z_center],
                         view_up=[0, 1, 0])
        scene.set_camera(position=[x_center, y_center, z_center - distance],
                         focal_point=[x_center, y_center, z_center],
                         view_up=[0, 1, 0])

    if interactive:
        window.show(scene, size=size, reset_camera=False,
                    png_magnify=magnification, order_transparent=False)
    elif out_path is not None:
        window.record(scene, size=size, reset_camera=False,
                      out_path=out_path, magnification=int(magnification))

    return scene
