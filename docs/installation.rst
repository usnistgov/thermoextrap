.. highlight:: shell

============
Installation
============


Stable release
--------------

To install thermodynamic-extrapolation, run this command in your terminal::

  $ pip install thermoextrap

or, if you use conda, run::

  $ conda install c wpk-nist thermoextrap


Additional dependencies
-----------------------

To utilize the full potential of `thermoextrap`, additional dependencies are needed.  This can be done via pip by using::

  $ pip install thermoextrap[all]

If using conda, then you'll have to manually install some dependencies.  For example, you can run::

  $ conda install bottleneck dask pymbar<4.0

At this time, it is recommended to install the Gaussian Process Regression (GPR) dependencies via pip, as the conda-forge recipes are slightly out of date::

  $ pip install tensorflow tensorflow-probability gpflow


From sources
------------

The sources for thermodynamic-extrapolation can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/wpk-nist-gov/thermoextrap

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/wpk-nist-gov/thermoextrap/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/wpk-nist-gov/thermoextrap
.. _tarball: https://github.com/wpk-nist-gov/thermoextrap/tarball/master
