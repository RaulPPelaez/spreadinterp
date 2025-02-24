from ._spreadinterp import interpolateField, spreadParticles
import cupy as cp
from typing import List, Optional


def interpolate(
    pos: cp.ndarray,
    grid_data: cp.ndarray,
    L: List,
    gradient: bool = False,
    gradient_direction: Optional[List] = None,
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

    Returns
    -------

    ndarray
        The interpolated field at the points. Shape (N, nf)
    """
    assert grid_data.ndim >= 3, "grid_data must have at least 3 dimensions"
    assert grid_data.ndim <= 4, "grid_data must have at most 4 dimensions"
    if grid_data.ndim == 3:
        grid_data = cp.ascontiguousarray(grid_data[:, :, :, cp.newaxis])
    nf = grid_data.shape[3]
    result = cp.zeros((pos.shape[0], nf), dtype=cp.float32)
    interpolateField(pos, grid_data, result, L, gradient, gradient_direction)
    return result


def spread(
    pos: cp.ndarray,
    quantity: cp.ndarray,
    L: List,
    n: List,
    gradient=False,
    gradient_direction: Optional[List] = None,
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
    Returns
    -------
    ndarray
        The spread quantity on the grid. Shape (n[0], n[1], n[2], Nq).
    """
    assert quantity.ndim <= 2, "quantity must have at most 2 dimensions"
    if quantity.ndim == 1:
        quantity = cp.ascontiguousarray(quantity[:, cp.newaxis]).astype(cp.float32)
    result = cp.zeros((n[0], n[1], n[2], quantity.shape[-1]), dtype=cp.float32)
    spreadParticles(pos, quantity, result, L, n, gradient, gradient_direction)
    return result
