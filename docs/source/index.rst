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

    \mathcal{J}\mathcal{S} = \mathcal{I}

And the spreading kernel follows the familiar Immersed Boundary rules:
.. math::
   
   \int_V \delta_a(\boldsymbol{x} - \boldsymbol{x}_i) d\boldsymbol{x} = 1
       

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
