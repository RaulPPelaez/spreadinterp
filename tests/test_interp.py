import spreadinterp
import numpy as np
import cupy as cp
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)


def gen_grid_positions(n, L):
    h = L / n
    n_x = cp.linspace(0, L[0], n[0], endpoint=False) - L[0] / 2.0 + h[0] / 2.0
    n_y = cp.linspace(0, L[1], n[1], endpoint=False) - L[1] / 2.0 + h[1] / 2.0
    n_z = cp.linspace(0, L[2], n[2], endpoint=False) - L[2] / 2.0 + h[2] / 2.0
    x, y, z = cp.meshgrid(n_x, n_y, n_z, indexing="ij")
    return x, y, z


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


def peskin_3pt_derivative(rp, h):
    """Computes the derivative of the 3-point Peskin kernel for a given r."""
    rp = cp.asarray(rp)
    r = cp.abs(rp) / h
    sgn = cp.sign(rp)
    phi = cp.zeros_like(r, dtype=np.float64)
    mask1 = r < 0.5
    phi[mask1] = -1 / h**2 * r[mask1] * sgn[mask1] / cp.sqrt(1 - 3 * r[mask1] ** 2)
    mask2 = (r >= 0.5) & (r < 1.5)
    phi[mask2] = (
        -1
        / h**2
        * (1 / 2)
        * (1 + (1 - r[mask2]) / cp.sqrt(1 - 3 * (1 - r[mask2]) ** 2))
        * sgn[mask2]
    )
    return phi


def manual_spread(pos, quantity, L, n):
    h = L / n
    field = cp.zeros((n[0], n[1], n[2], quantity.shape[1]), dtype=cp.float32)
    x, y, z = gen_grid_positions(n, L)
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
    x, y, z = gen_grid_positions(field.shape[:3], L)
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


def test_interp_gradient_constant():
    # Interpolate a constant field on a particle located at the center. The result should be zero
    L = 16
    n = 64
    pos = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    field = cp.ones((n, n, n, 3), dtype=cp.float32)
    direction = cp.ones_like(pos) * cp.array([1, 0, 0])
    L = np.array([L, L, L])
    gradient = spreadinterp.interpolate(
        pos, field, L, gradient=True, gradient_direction=direction
    )
    assert gradient.shape == pos.shape
    assert cp.allclose(gradient, 0.0, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_interp_gradient_linear(direction):
    # Interpolate a field f(x,y,z) = x on a particle located at the center. The gradient should be [1, 0, 0]
    L = np.array([16, 16, 16])
    n = np.array([64, 64, 64])
    pos = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    x, y, z = gen_grid_positions(n, L)
    field = cp.zeros((n[0], n[1], n[2], 3), dtype=cp.float32)
    f = x if direction == 0 else y if direction == 1 else z
    field[..., direction] = f
    d = cp.zeros(3)
    d[direction] = 1
    direction = cp.ones_like(pos) * d
    assert direction.shape == pos.shape
    assert field.shape == (n[0], n[1], n[2], 3)
    gradient = spreadinterp.interpolate(
        pos, field, L, gradient=True, gradient_direction=direction
    )
    assert gradient.shape == pos.shape
    assert cp.allclose(gradient, d, atol=1e-5, rtol=1e-5)


def test_interp_gradient_radial():
    # Interpolate a field f(x,y,z) = sqrt(x^2 + y^2 + z^2) on a particle located at the center. The gradient should be [x, y, z]/r
    L = np.array([16, 16, 16])
    n = np.array([64, 64, 64])
    h = L / n
    pos = cp.array([[-5, -5, -5]], dtype=cp.float32)
    x, y, z = gen_grid_positions(n, L)
    f = cp.sqrt(x**2 + y**2 + z**2)
    field = cp.zeros((n[0], n[1], n[2], 3), dtype=cp.float32)
    field[..., 0] = f
    d = cp.array([1, 1, 1])
    direction = cp.ones_like(pos) * d
    assert direction.shape == pos.shape
    assert field.shape == (n[0], n[1], n[2], 3)
    gradient = spreadinterp.interpolate(
        pos, field, L, gradient=True, gradient_direction=direction
    )
    assert gradient.shape == pos.shape
    r = cp.linalg.norm(pos, axis=1)
    expected = cp.dot((pos / r), d)
    assert gradient.shape == (1, 3)
    assert cp.allclose(gradient[:, 0], expected, atol=1e-2), np.max(
        np.abs(gradient - expected)
    )


@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("dimensions", [1, 3])
def test_spread_gradient(direction, dimensions):
    # Spread a quantity on a particle located at (0,0,0). The resulting field should be:
    # (dS_x*d_x + dS_y*d_y + dS_z*d_z)*quantity
    # dS_alpha = \prod_{\beta \neq \alpha} \delta_\beta * d\delta_\alpha
    # Where d\delta is the derivative of the Peskin kernel
    L = np.array([16, 16, 16])
    n = np.array([64, 64, 64])
    pos = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    quantity = cp.ones((1, dimensions), dtype=cp.float32)
    x, y, z = gen_grid_positions(n, L)
    Sx = peskin_3pt(x, L[0] / n[0])
    Sy = peskin_3pt(y, L[1] / n[1])
    Sz = peskin_3pt(z, L[2] / n[2])
    dSx = peskin_3pt_derivative(pos[0, 0] - x, L[0] / n[0]) * Sy * Sz
    dSy = peskin_3pt_derivative(pos[0, 1] - y, L[1] / n[1]) * Sx * Sz
    dSz = peskin_3pt_derivative(pos[0, 2] - z, L[2] / n[2]) * Sx * Sy
    d = cp.zeros((1, 3))
    d[:, direction] = 1
    delta = dSx * d[0, 0] + dSy * d[0, 1] + dSz * d[0, 2]
    expected = cp.array(delta[..., None] * quantity)
    assert expected.shape == (n[0], n[1], n[2], dimensions)
    gradient = spreadinterp.spread(
        pos, quantity, L, n, gradient=True, gradient_direction=d
    )
    assert gradient.shape == (n[0], n[1], n[2], dimensions)
    max_dev = cp.max(cp.abs(gradient - expected))
    assert cp.allclose(gradient, expected, atol=1e-5, rtol=1e-5), max_dev
