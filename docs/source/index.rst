spreadinterp: GPU Spreading and interpolation
=============================================

spreadinterp allows to transform between Eulerian (grid based) and Lagrangian (particle based) descriptions by making use of the Immersed Boundary Method.

spreadinterp exposes part of UAMMD's GPU spreading and interpolation capabilities to Python.

Functionality
-------------

Given a set of particles with positions and properties, spreadinterp allows to apply the following operations:

*Spreading:*

.. math::

    \boldsymbol{f}(\boldsymbol{x}) = \mathcal{S}(\boldsymbol{x})\{\boldsymbol{p}\} = \sum_{i=1}^{N} \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) \boldsymbol{p}_i

*Interpolation:*

.. math::

    \boldsymbol{p}_i = \mathcal{J}_{\boldsymbol{x}_i}\{\boldsymbol{f}(\boldsymbol{x})\} = \int_V \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) \boldsymbol{f}(\boldsymbol{x}) d\boldsymbol{x}

Where:

- :math:`\boldsymbol{f}(\boldsymbol{x})` is a field at position :math:`\boldsymbol{x}`
- :math:`\boldsymbol{p}_i` is the property of particle :math:`i`
- :math:`\mathcal{S}(\boldsymbol{x})` is the spreading operator
- :math:`\mathcal{J}_{\boldsymbol{x}_i}` is the interpolation operator  
- :math:`\boldsymbol{x}_i` is the position of particle :math:`i`
- :math:`\delta_a(\boldsymbol{x} - \boldsymbol{x}_i)` is the spreading kernel
- :math:`N` is the number of particles
- :math:`V` is the volume of a particle

These operator are related by the following identity:

.. math::

    \mathcal{J}\mathcal{S} 1 = \int \delta_a^2(\boldsymbol{r})d\boldsymbol{r} = \Delta V^{-1}

where :math:`\Delta V` is an effective "volume" for the particles.    

And the spreading kernel follows the familiar Immersed Boundary rules:

.. math::
   
   \int_V \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) d\boldsymbol{x} = 1
       

Currently, :math:`\delta_a` is chosen as a 3-point Peskin kernel, defined as:

.. math::

     \phi_{p_3}(|r|) =  \left\{
     \begin{aligned}
     & \frac{1}{3}\left( 1 + \sqrt{1-3r^2}\right)& r < 0.5\\
     & \frac{1}{6}\left(5-3r-\sqrt{1-3(1-r)^2}\right)& r < 1.5\\
     & 0 & r>1.5 
     \end{aligned}\right.


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
