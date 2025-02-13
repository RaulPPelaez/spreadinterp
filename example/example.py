import spreadinterp
import numpy as np
import cupy as cp

L = 16
numberParticles = 1000
pos = cp.array((np.random.rand(numberParticles, 3) - 0.5) * L)
n = 32
field = cp.array(np.random.rand(n, n, n, 3) * 0 + 1)
L = np.array([L, L, L])
n = np.array([n, n, n])

assert pos.shape == (numberParticles, 3)
assert field.shape == (n[0], n[1], n[2], 3)
res = spreadinterp.interpolateField(pos, field, L, n)

print(res)
assert res.shape == (numberParticles, 3)
