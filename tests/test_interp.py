import spreadinterp
import numpy as np
import cupy as cp
import pytest


@pytest.mark.parametrize("n", [8, 16, 32])
@pytest.mark.parametrize("numberParticles", [1, 2, 1024])
def test_interp(n, numberParticles):
    L = 16
    pos = cp.array((np.random.rand(numberParticles, 3) - 0.5) * (L - 1))
    field = cp.ones((n, n, n, 3))
    L = np.array([L, L, L])
    n = np.array([n, n, n])
    assert pos.shape == (numberParticles, 3)
    assert field.shape == (n[0], n[1], n[2], 3)
    res = spreadinterp.interpolateField(pos, field, L, n)

    assert res.shape == (numberParticles, 3)
    assert res.dtype == cp.float32
    assert res.device == pos.device
    assert cp.allclose(res, 1.0)
