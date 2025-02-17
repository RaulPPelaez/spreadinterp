from ._spreadinterp import interpolateField, spreadParticles
import cupy as cp
from typing import List


def interpolate(pos: cp.ndarray, grid_data: cp.ndarray, L: List) -> cp.ndarray:
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

    Returns
    -------

    ndarray
        The interpolated field at the points. Shape (N, nf)
    """
    assert grid_data.ndim >= 3, "grid_data must have at least 3 dimensions"
    assert grid_data.ndim <= 4, "grid_data must have at most 4 dimensions"
    if len(grid_data.shape) > 3:
        result = cp.empty((pos.shape[0], grid_data.shape[-1]), dtype=cp.float32)
        for i in range(grid_data.shape[-1]):
            gi = cp.ascontiguousarray(grid_data[:, :, :, i], dtype=cp.float32)
            ri = cp.zeros((pos.shape[0]), dtype=cp.float32)
            interpolateField(pos, gi, ri, L)
            result[:, i] = ri
        return result
    else:
        result = cp.zeros((pos.shape[0]), dtype=cp.float32)
        interpolateField(pos, grid_data, result, L)
        return result


def spread(pos: cp.ndarray, quantity: cp.ndarray, L: List, n: List) -> cp.ndarray:
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

    Returns
    -------
    ndarray
        The spread quantity on the grid. Shape (n[0], n[1], n[2], Nq).
    """
    assert quantity.ndim <= 2, "quantity must have at most 2 dimensions"
    if len(quantity.shape) > 1:
        result = cp.empty((n[0], n[1], n[2], quantity.shape[-1]), dtype=cp.float32)
        for i in range(quantity.shape[-1]):
            qi = cp.ascontiguousarray(quantity[:, i], dtype=cp.float32)
            ri = cp.zeros((n[0], n[1], n[2]), dtype=cp.float32)
            spreadParticles(pos, qi, ri, L, n)
            result[:, :, :, i] = ri
        return result
    else:
        result = cp.ascontiguousarray(cp.zeros((n[0], n[1], n[2]), dtype=cp.float32))
        spreadParticles(pos, quantity, result, L, n)
        return result
