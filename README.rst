Package for analyzing central moments
=====================================

For example usage, look
at\ `example\ usage.ipynb <examples/example_usage.ipynb>`__

Installation
============

Install needed packages
-----------------------

If using conda, then

.. code:: bash

   conda install numpy xarray numba 

From source
-----------

.. code:: bash

   cd directory/path
   git clone https://github.com/wpk-nist-gov/cmomy.git

   # full install (note that versioning is iffy.  -I forces reinstall)
   pip install -I -U .

   # for editable
   pip install -e .

To run tests:

.. code:: bash

   pytest -v

Note that tests are a bit slow (about 1.5 min). This is due to numba jit
compiler having to compile all underlying functions for all cases.

pip install
-----------

.. code:: bash

   pip install -I -U git+https://github.com/wpk-nist-gov/cmomy.git@develop

Note on caching
===============

This code makes extensive use of the numba python package. This uses a
jit compiler to speed up vital code sections. This means that the first
time a funciton called, it has to compile the underlying code. However,
caching has been implemented. Therefore, the very first time you run a
function, it may be slow. But all subsequent uses (including other
sessions) will be already compiled.
