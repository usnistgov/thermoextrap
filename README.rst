=====
cmomy
=====


.. image:: https://img.shields.io/pypi/v/cmomy.svg
        :target: https://pypi.python.org/pypi/cmomy

.. image:: https://img.shields.io/travis/wpk-nist-gov/cmomy.svg
        :target: https://travis-ci.com/wpk-nist-gov/cmomy

..
   .. image:: https://readthedocs.org/projects/cmomy/badge/?version=latest
           :target: https://cmomy.readthedocs.io/en/latest/?badge=latest
           :alt: Documentation Status


Central (co)moment calculation/manipulation


* Free software: NIST license
* Documentation: https://cmomy.readthedocs.io.


Features
--------

* Fast calculation of central moments and central co-moments with weights
* Support for scalor or vector inputs
* numpy and xarray api's


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

   pytest -x -v --pyargs cmomy

By running the tests once, you create a cache of the numba code for most
cases. The first time you run the tests, it will take a while (about 1.5
min on my machine). However, subsequent runs will be much faster (about
3 seconds on my machine).

Credits
-------

This package was created with Cookiecutter_ and the `wpk-nist-gov/cookiecutter-pypackage`_ Project template forked from `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`wpk-nist-gov/cookiecutter-pypackage`: https://github.com/wpk-nist-gov/cookiecutter-pypackage
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
