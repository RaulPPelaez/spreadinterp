Installation
============

We recommend working with a `conda <https://docs.conda.io/en/latest/>`_ environment.

Install from conda (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the library from the ``stochasticHydroTools`` Anaconda channel with:

.. code-block:: shell

	$ conda install -c stochasticHydroTools spreadinterp

Compilation from source
~~~~~~~~~~~~~~~~~~~~~~~

If you want to compile the library from source, you can clone the repository with:

.. code-block:: shell

    $ git clone https://github.com/RaulPPelaez/spreadinterp

Getting dependencies
--------------------

The file ``environment.yml`` contains the necessary dependencies to compile and use the library.

You can create the environment with:

.. code-block:: shell

    $ conda env create -f environment.yml

Then, activate the environment with:

.. code-block:: shell

    $ conda activate spreadinterp

	  
Installing via pip
----------------------

After installing the dependencies, you can install the library with pip. Go to the root of the repository and run:

.. code-block:: shell

    $ pip install .
    
   

