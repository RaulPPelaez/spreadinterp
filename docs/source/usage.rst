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


Choosing a different kernel
===========================
You can choose a different kernel by passing the `kernel` argument to the other functions.
The available kernels are:

- **peskin3pt**: Peskin 3-point kernel
- **gaussian**: Gaussian kernel


The kernel can be specified as follows:

.. code:: python
	  
	  kernel = spreadinterp.create_kernel('peskin3pt')
	  # kernel = spreadinterp.create_kernel('gaussian', width=1.0, cutoff=4.0)
	  res = spreadinterp.interpolate(pos, field, L, kernel=kernel)
	  res = spreadinterp.interpolate(pos, field, L, kernel=kernel)

	  
