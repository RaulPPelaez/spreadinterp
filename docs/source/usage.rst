Usage
-----

   
Example
~~~~~~~

.. code:: python


	  import spreadinterp
	  import numpy as np
	  import cupy as cp

	  L = 16
	  numberParticles = 1000
	  pos = (cp.random.rand(numberParticles, 3) - 0.5) * L
	  n = 32
	  field = cp.ones((n, n, n, 3))
	  L = np.array([L, L, L])
	  n = np.array([n, n, n])
	  res = spreadinterp.interpolate(pos, field, L)

	  print(res)
	  assert res.shape == (numberParticles, 3)
	  assert cp.allclose(res, 1)

