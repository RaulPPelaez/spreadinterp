spreadinterp: GPU Spreading and interpolation
=============================================

spreadinterp allows to transform between Eulerian (grid based) and Lagrangian (particle based) descriptions by making use of the Immersed Boundary Method.

spreadinterp exposes part of UAMMD's GPU spreading and interpolation capabilities to Python.

Functionality
-------------

Given a set of particles with positions and properties, spreadinterp allows to apply the following operations:

**Spreading:**

.. math::

    \boldsymbol{f}(\boldsymbol{x}) = \mathcal{S}(\boldsymbol{x})\{\boldsymbol{p}\} = \sum_{i=1}^{N} \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) \boldsymbol{p}_i

**Interpolation:**

.. math::

    \boldsymbol{p}_i = \mathcal{J}_{\boldsymbol{x}_i}\{\boldsymbol{f}(\boldsymbol{x})\} = \int_V \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) \boldsymbol{f}(\boldsymbol{x}) d\boldsymbol{x}

Where:

- :math:`\boldsymbol{f}(\boldsymbol{x})` is a field at position :math:`\boldsymbol{x}`
- :math:`\boldsymbol{p}_i` is the property of particle :math:`i`
- :math:`\mathcal{S}(\boldsymbol{x})` is the spreading operator
- :math:`\mathcal{J}_{\boldsymbol{x}_i}` is the interpolation operator  
- :math:`\boldsymbol{x}_i` is the position of particle :math:`i`
- :math:`\delta_a(\boldsymbol{x} - \boldsymbol{x}_i)` is the spreading kernel, which can be chosen.
- :math:`N` is the number of particles
- :math:`V` is the volume of a particle

These operator are related by the following identity:

.. math::

    \mathcal{J}\mathcal{S} 1 = \int \delta_a^2(\boldsymbol{r})d\boldsymbol{r} = \Delta V^{-1}

where :math:`\Delta V` is an effective "volume" for the particles.    

And the spreading kernel follows the familiar Immersed Boundary rules:

.. math::
   
   \int_V \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) d\boldsymbol{x} = 1
       

Currently, :math:`\delta_a` can chosen as a Gaussian or a 3-point Peskin kernel, defined as:

.. math::

     \phi_{p_3}(|r|) =  \left\{
     \begin{aligned}
     & \frac{1}{3}\left( 1 + \sqrt{1-3r^2}\right)& r < 0.5\\
     & \frac{1}{6}\left(5-3r-\sqrt{1-3(1-r)^2}\right)& r < 1.5\\
     & 0 & r>1.5 
     \end{aligned}\right.


Gradient spreading and interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library also allows to spread dipoles and interpolate gradients of fields by making use of the following operations:

**Gradient Interpolation:**

.. math::

    \boldsymbol{p'}_i = \mathcal{J}_{\boldsymbol{x}_i}\left(\partial_\boldsymbol{x}\boldsymbol{f}(\boldsymbol{x})\cdot\boldsymbol{d}_i\right) = \int_V \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) \partial_{\boldsymbol{x}} \boldsymbol{f}(\boldsymbol{x})\cdot \boldsymbol{d}_i d\boldsymbol{x} = \int_V \left[\partial_\boldsymbol{x}\delta_a(\boldsymbol{x} - \boldsymbol{x}_i)\cdot \boldsymbol{d}_i\right] \boldsymbol{f}(\boldsymbol{x}) d\boldsymbol{x}
    
Where:

- :math:`\boldsymbol{d}_i` is a direction vector
- :math:`\partial_\boldsymbol{x}\delta_a(\boldsymbol{x} - \boldsymbol{x}_i)` is the gradient of the spreading kernel in the direction of :math:`\boldsymbol{x}`.

.. note::
   
   This evaluates the gradient of a vector field in a particle position for a given direction. The gradient of a vector field is a tensor, which is then multiplied by a direction tensor to obtain a vector, in particular:

   .. math::

      \left(\partial_\boldsymbol{x}\boldsymbol{f}(\boldsymbol{x})\cdot \boldsymbol{d}_i\right)_{\alpha} = \partial_{x} f_\alpha d^x_i + \partial_{y} f_\alpha d^y_i + \partial_{z} f_\alpha d^z_i


**Dipole spreading**

.. math::

    \boldsymbol{f'}(\boldsymbol{x}) = \sum_{i=1}^{N} \left(\partial_{\boldsymbol{x}}\delta_a(\boldsymbol{x} - \boldsymbol{x}_i)\cdot\boldsymbol{d}_i\right) \boldsymbol{p}_i

.. note::
   
   This spreads a particle quantity into a dipole field at a given position and direction. The gradient of the spreading kernel (a vector) is multiplied by the dipole direction to obtain an scalar, which is then multiplied by the particle property to obtain the dipole field. In particular

   .. math::

      \left(\partial_{\boldsymbol{x}}\delta_a(\boldsymbol{x} - \boldsymbol{x}_i)\cdot\boldsymbol{d}_i\right) = \partial_{x} \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) d^x_i + \partial_{y} \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) d^y_i + \partial_{z} \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) d^z_i


.. toctree::
   :hidden:
      
   installation
   usage
   api

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
