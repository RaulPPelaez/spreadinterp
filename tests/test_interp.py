import spreadinterp
import numpy as np
import cupy as cp
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)


def peskin_3pt(rp, h):
    """Computes the 3-point Peskin kernel for a given r."""
    rp = cp.asarray(rp)
    r = cp.abs(rp) / h
    phi = cp.zeros_like(r, dtype=np.float64)
    mask1 = r < 0.5
    phi[mask1] = (1 / 3.0) * (1 + cp.sqrt(1 - 3 * r[mask1] ** 2))
    mask2 = (r >= 0.5) & (r < 1.5)
    phi[mask2] = (1 / 6.0) * (5 - 3 * r[mask2] - cp.sqrt(1 - 3 * (1 - r[mask2]) ** 2))
    return phi / h


def manual_spread(pos, quantity, L, n):
    h = L / n
    field = cp.zeros((n[0], n[1], n[2], quantity.shape[1]), dtype=cp.float32)
    n_x = cp.linspace(0, L[0], n[0], endpoint=False) - L[0] / 2.0 + h[0] / 2.0
    n_y = cp.linspace(0, L[1], n[1], endpoint=False) - L[1] / 2.0 + h[1] / 2.0
    n_z = cp.linspace(0, L[2], n[2], endpoint=False) - L[2] / 2.0 + h[2] / 2.0
    x, y, z = cp.meshgrid(n_x, n_y, n_z, indexing="ij")
    for i in range(pos.shape[0]):
        xp = x - pos[i, 0]
        yp = y - pos[i, 1]
        zp = z - pos[i, 2]
        delta = peskin_3pt(xp, h[0]) * peskin_3pt(yp, h[1]) * peskin_3pt(zp, h[2])
        field += (
            delta[..., cp.newaxis] * quantity[i][cp.newaxis, cp.newaxis, cp.newaxis]
        )
    return field


def manual_interp(pos, field, L):
    h = L / field.shape[:3]
    n_x = cp.linspace(0, L[0], field.shape[0], endpoint=False) - L[0] / 2.0 + h[0] / 2.0
    n_y = cp.linspace(0, L[1], field.shape[1], endpoint=False) - L[1] / 2.0 + h[1] / 2.0
    n_z = cp.linspace(0, L[2], field.shape[2], endpoint=False) - L[2] / 2.0 + h[2] / 2.0
    x, y, z = cp.meshgrid(n_x, n_y, n_z, indexing="ij")
    quantity = cp.zeros((pos.shape[0], field.shape[3]), dtype=cp.float32)
    qw = h[0] * h[1]
    if field.shape[2] > 1:
        qw *= h[2]
    for i in range(pos.shape[0]):
        xp = x - pos[i, 0]
        yp = y - pos[i, 1]
        zp = z - pos[i, 2]
        delta = peskin_3pt(xp, h[0]) * peskin_3pt(yp, h[1])
        if field.shape[2] > 1:
            delta *= peskin_3pt(zp, h[2])
        quantity[i] = cp.sum(delta[..., cp.newaxis] * field, axis=(0, 1, 2)) * qw
    return quantity


@pytest.mark.parametrize("n", [3, 8, 16])
@pytest.mark.parametrize("numberParticles", [1, 32])
@pytest.mark.parametrize("nquantities", [1, 3, 4, 5, 10])
@pytest.mark.parametrize("nonregular", [False, True])
def test_spread(n, numberParticles, nquantities, nonregular):
    L = 16
    # Factor is a number between 1.0 and 2.0
    factor = (np.random.rand(3) + 1.0) if nonregular else 1.0
    L = np.array([L, L, L]) * factor
    n = (np.array([n, n, n]) * factor).astype(int)
    h = L / n.astype(float)
    pos = cp.array(
        (np.random.rand(numberParticles, 3) - 0.5) * (L - 2 * h), dtype=cp.float32
    ) * (0 if n.min() == 3 else 1)
    quantity = cp.random.rand(numberParticles, nquantities).astype(cp.float32)
    field = spreadinterp.spread(pos, quantity, L, n)
    field_expected = manual_spread(pos, quantity, L, n)
    assert field.shape == (n[0], n[1], n[2], nquantities)
    assert field.dtype == cp.float32
    assert field.device == pos.device
    assert np.allclose(field.get(), field_expected.get(), atol=1e-5, rtol=1e-5), np.max(
        field.get() - field_expected.get()
    )


import scipy.integrate as integrate


def peskin_integral(h):
    res = integrate.quad(lambda x: (peskin_3pt(x, h) ** 2), -1.5, 1.5)[0]
    return res


@pytest.mark.parametrize("is2D", [False, True])
@pytest.mark.parametrize("nonregular", [False, True])
def test_spreadinterp(is2D, nonregular):
    # JS1 = 1/dV
    # Where dV is the integral of the kernel squared: \int \delta_a(\vec{r})^2 dr^3
    # This test checks that the spread and interpolate functions are adjoint
    L = 16
    n = 64
    pos = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    quantity = cp.ones((1, 1), dtype=cp.float32)
    factor = (np.random.rand() * 1.5 + 0.5) if nonregular else 1.0
    L = np.array([L, L * factor, L])
    n = np.array([n, int(n * factor), n])
    h = L / n
    if is2D:
        pos[:, 2] = 0
        L[2] = 0
        n[2] = 1
    field = spreadinterp.spread(pos, quantity, L, n)
    dV = peskin_integral(h[0]) * peskin_integral(h[1])
    if not is2D:
        dV *= peskin_integral(h[2])

    quantity_reconstructed = spreadinterp.interpolate(pos, field, L) / dV
    assert cp.allclose(
        quantity.get(), quantity_reconstructed.get(), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("n", [3, 8, 17, 32])
@pytest.mark.parametrize("numberParticles", [1, 2, 256])
@pytest.mark.parametrize("is2D", [False, True])
@pytest.mark.parametrize("nquantities", [1, 3, 4, 5, 10])
def test_interp(n, numberParticles, is2D, nquantities):
    L = 16
    h = L / n
    pos = cp.array(
        (np.random.rand(numberParticles, 3) - 0.5) * (L - 2 * h), dtype=cp.float32
    ) * (0 if n == 3 else 1)
    field = cp.random.rand(n, n, n, nquantities).astype(cp.float32) * 0 + 1
    L = np.array([L, L, L])
    if is2D:
        pos[:, 2] = 0
        L[2] = 0
        field = field[:, :, 0, :]
        field = field[:, :, np.newaxis, :]
        assert field.shape == (n, n, 1, nquantities)
    res = spreadinterp.interpolate(pos, field, L)
    expected = manual_interp(pos, field, L)
    assert res.shape == (numberParticles, nquantities)
    assert res.dtype == cp.float32
    assert res.device == pos.device
    assert cp.allclose(res.get(), expected.get(), atol=1e-5, rtol=1e-5), cp.max(
        res.get() - expected.get()
    )
