=====
cmomy
=====

Central (co)moment calculation/manipulation


* Free software: NIST license

Overview
--------
``cmomy`` is an open source package to calculate central moments and co-moments in a numerical stable and direct way.
Behind the scenes, ``cmomy`` makes use of Numba_ to rapidly calculate moments.  A good introduction to the type of formulas used can
be found `here <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_.



Features
--------

* Fast calculation of central moments and central co-moments with weights
* Support for scalor or vector inputs
* numpy and xarray api's


Links
-----

* `Github <https://github.com/usnistgov/cmomy>`__
* `Documentation <https://pages.nist.gov/cmomy/index.html>`__



Installation
------------
Use one of the following

.. code:: bash

          pip install cmomy

.. code:: bash

          conda install -c wpk-nist cmomy


Basic Usage
-----------

For a quick introduction to the usage of ``cmomy``, please see the `basic usage <https://github.com/usnistgov/cmomy/blob/master/docs/notebooks/docs/notebooks/usage_notebook.ipynb>`__


Note on caching
---------------

This code makes extensive use of the numba python package. This uses a
jit compiler to speed up vital code sections. This means that the first
time a funciton called, it has to compile the underlying code. However,
caching has been implemented. Therefore, the very first time you run a
function, it may be slow. But all subsequent uses (including other
sessions) will be already compiled.

Testing
-------
Tests are packaged with the distribution intentionally. To test code
run:

.. code:: bash

   pytest --pyargs cmomy

By running the tests once, you create a cache of the numba code for most
cases. The first time you run the tests, it will take a while (about 1.5
min on my machine). However, subsequent runs will be much faster (about
3 seconds on my machine).

Credits
-------

This package was created with Cookiecutter_ and the `wpk-nist-gov/cookiecutter-pypackage`_ Project template forked from `audreyr/cookiecutter-pypackage`_.

.. _Numba: https://numba.pydata.org/
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`wpk-nist-gov/cookiecutter-pypackage`: https://github.com/wpk-nist-gov/cookiecutter-pypackage
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
