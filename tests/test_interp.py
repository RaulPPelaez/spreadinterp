import spreadinterp
import numpy as np
import cupy as cp
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)


def peskin_3pt(rp, h):
    """Computes the 3-point Peskin kernel for a given r."""
    rp = cp.asarray(rp)  # Ensure input is a NumPy array
    # return 1 / cp.sqrt(2 * np.pi) * cp.exp(-(rp**2) / 2.0)
    r = cp.abs(rp) / h
    phi = cp.zeros_like(r, dtype=np.float64)
    mask1 = r < 0.5
    phi[mask1] = (1 / 3.0) * (1 + cp.sqrt(1 - 3 * r[mask1] ** 2))
    mask2 = (r >= 0.5) & (r < 1.5)
    phi[mask2] = (1 / 6.0) * (5 - 3 * r[mask2] - cp.sqrt(1 - 3 * (1 - r[mask2]) ** 2))
    return phi / h


def manual_spread(pos, quantity, L, n):
    h = L[0] / n[0]
    field = cp.zeros((n[0], n[1], n[2], quantity.shape[1]), dtype=cp.float32)
    for i in range(pos.shape[0]):
        node_centers = cp.linspace(0, L[0], n[0], endpoint=False) - L[0] / 2.0 + h / 2.0
        x, y, z = cp.meshgrid(node_centers, node_centers, node_centers)
        xp = x - pos[i, 0]
        yp = y - pos[i, 1]
        zp = z - pos[i, 2]
        delta = peskin_3pt(xp, h) * peskin_3pt(yp, h) * peskin_3pt(zp, h)
        field += (
            delta[:, :, :, cp.newaxis]
            * quantity[i, :][cp.newaxis, cp.newaxis, cp.newaxis]
        )
    return field


@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("numberParticles", [1, 32])
@pytest.mark.parametrize("nquantities", [1, 3])
def test_spread(n, numberParticles, nquantities):
    L = 16
    pos = cp.zeros((numberParticles, 3), dtype=cp.float32)
    # pos = cp.array(
    #     (np.random.rand(numberParticles, 3) - 0.5) * (L * 0.5), dtype=cp.float32
    # )
    quantity = cp.ones((numberParticles, nquantities), dtype=cp.float32)
    L = np.array([L, L, L])
    n = np.array([n, n, n])
    field = spreadinterp.spread(pos, quantity, L, n)
    field_expected = manual_spread(pos, quantity, L, n)
    assert field.shape == (n[0], n[1], n[2], nquantities)
    assert field.dtype == cp.float32
    assert field.device == pos.device
    assert np.allclose(field.get(), field_expected.get()), np.max(
        field.get() - field_expected.get()
    )


import scipy.integrate as integrate


def peskin_integral(h):
    res = integrate.quad(lambda x: (peskin_3pt(x, h) ** 2), -1.5, 1.5)[0]
    return res


@pytest.mark.parametrize("is2D", [False, True])
def test_spreadinterp(is2D):
    # JS1 = 1/dV
    # Where dV is the integral of the kernel squared: \int \delta_a(\vec{r})^2 dr^3
    # This test checks that the spread and interpolate functions are adjoint
    L = 16
    n = 64
    h = L / n
    pos = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    quantity = cp.ones((1, 1), dtype=cp.float32)
    L = np.array([L, L, L])
    n = np.array([n, n, n])
    if is2D:
        pos[:, 2] = 0
        L[2] = 0
        n[2] = 1
    field = spreadinterp.spread(pos, quantity, L, n)
    dV = peskin_integral(h) ** (2 if is2D else 3)

    quantity_reconstructed = spreadinterp.interpolate(pos, field, L) / dV
    assert cp.allclose(
        quantity.get(), quantity_reconstructed.get(), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("n", [8, 16, 32])
@pytest.mark.parametrize("numberParticles", [1, 2, 1024])
@pytest.mark.parametrize("is2D", [False, True])
def test_interp(n, numberParticles, is2D):
    L = 16
    pos = cp.array(
        (np.random.rand(numberParticles, 3) - 0.5) * (L - 1.0), dtype=cp.float32
    )
    field = cp.ones((n, n, n, 3), dtype=pos.dtype)
    L = np.array([L, L, L])
    assert pos.shape == (numberParticles, 3)
    if is2D:
        pos[:, 2] = 0
        L[2] = 0
        field = field[:, :, 0, :]
        field = field[:, :, np.newaxis, :]
        assert field.shape == (n, n, 1, 3)
    res = spreadinterp.interpolate(pos, field, L)
    assert res.shape == (numberParticles, 3)
    assert res.dtype == cp.float32
    assert res.device == pos.device
    assert cp.allclose(res.get(), 1.0)
