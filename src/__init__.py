from ._spreadinterp import interpolateField, spreadParticles
import cupy as cp
from typing import List, Optional
import numpy as np


def create_kernel(type: str, **kwargs) -> dict:
    """Create a kernel for interpolation or spreading.

    Parameters
    ----------
    type : str
        The type of kernel to create. Options are **peskin3pt** or **gaussian**.
    kwargs : dict
        Additional parameters for the kernel. For `gaussian`, `width` and `cutoff` must be provided.

    Returns
    -------

    dict
        A dictionary representing the kernel with its type and parameters that can be passed to interpolate or spread functions.
    """

    if type == "peskin3pt":
        assert (
            len(kwargs) == 0
        ), "No additional parameters are required for peskin3pt kernel"
        return {"type": "peskin3pt"}
    elif type == "gaussian":
        assert "width" in kwargs, "width must be provided for gaussian kernel"
        gaussian_width = kwargs["width"]
        assert isinstance(
            gaussian_width, (int, float)
        ), "gaussian_width must be a number"
        assert gaussian_width > 0, "gaussian_width must be positive"
        cutoff = kwargs.get("cutoff", 4.0 * gaussian_width)
        assert isinstance(cutoff, (int, float)), "cutoff must be a number if provided"
        return {"type": "gaussian", "width": gaussian_width, "cutoff": cutoff}


def interpolate(
    pos: cp.ndarray,
    grid_data: cp.ndarray,
    L: List,
    gradient: bool = False,
    gradient_direction: Optional[List] = None,
    kernel: Optional[dict] = None,
) -> cp.ndarray:
    """Interpolate a field defined on a grid to a set of points.
       Field is assumed to be defined on a regular grid with periodic boundary conditions.
       The grid points are defined in the range [-L+h, L+h]*0.5 in each direction
       Positions and field must be in cuda memory.

    Parameters
    ----------
    pos : ndarray
        The positions of the points to interpolate to. Shape (N, 3)
    field : ndarray
        The field to interpolate. Shape (n[0], n[1], n[2], nf).
    L : ndarray
        The box size. Shape (3,).
    gradient : bool
        Whether to interpolate using the gradient of the kernel.
    gradient_direction : ndarray
        The direction of the gradient. Shape (3,).
    kernel : dict, optional
        The kernel to use for interpolation. If None, the default kernel is used.
        The kernel must be created using :py:mod:`spreadinterp.create_kernel` function.

    Returns
    -------

    ndarray
        The interpolated field at the points. Shape (N, nf)
    """
    assert grid_data.ndim >= 3, "grid_data must have at least 3 dimensions"
    assert grid_data.ndim <= 4, "grid_data must have at most 4 dimensions"
    if kernel is None:
        kernel = create_kernel("peskin3pt")
    if isinstance(pos, np.ndarray):
        pos = cp.array(pos)
    if isinstance(grid_data, np.ndarray):
        grid_data = cp.array(grid_data)
    if grid_data.ndim == 3:
        grid_data = cp.ascontiguousarray(grid_data[:, :, :, cp.newaxis])
    nf = grid_data.shape[3]
    result = cp.zeros((pos.shape[0], nf), dtype=pos.dtype)
    L = np.array(L, dtype=pos.dtype)
    interpolateField(pos, grid_data, result, L, gradient, gradient_direction, kernel)
    return result


def spread(
    pos: cp.ndarray,
    quantity: cp.ndarray,
    L: List,
    n: List,
    gradient=False,
    gradient_direction: Optional[List] = None,
    kernel: Optional[dict] = None,
) -> cp.ndarray:
    """
    Spread a quantity defined at a set of points to a grid.
    Quantity is assumed to be defined at a set of points.
    Positions and quantity must be in cuda memory.

    Parameters
    ----------
    pos : ndarray
        The positions of the points to spread from. Shape (N, 3)
    quantity : ndarray
        The quantity to spread. Shape (N, Nq)
    L : ndarray
        The box size. Shape (3,)
    n : ndarray
        The number of grid points in each direction. Shape (3,)
    gradient : bool
        Whether to spread using the gradient of the kernel.
    gradient_direction : ndarray
        The direction of the gradient. Shape (3,).
    kernel : dict, optional
        The kernel to use for spreading. If None, the default kernel is used.
        The kernel must be created using :py:mod:`spreadinterp.create_kernel` function.
    Returns
    -------
    ndarray
        The spread quantity on the grid. Shape (n[0], n[1], n[2], Nq).
    """
    assert quantity.ndim <= 2, "quantity must have at most 2 dimensions"
    if quantity.ndim == 1:
        quantity = quantity.reshape(-1, 1)
    if isinstance(pos, np.ndarray):
        pos = cp.array(pos)
    if isinstance(quantity, np.ndarray):
        quantity = cp.array(quantity)
    if kernel is None:
        kernel = create_kernel("peskin3pt")
    L = np.array(L, dtype=pos.dtype)
    n = np.array(n, dtype=np.int32)
    result = cp.zeros((n[0], n[1], n[2], quantity.shape[-1]), dtype=pos.dtype)
    spreadParticles(pos, quantity, result, L, n, gradient, gradient_direction, kernel)
    return result
