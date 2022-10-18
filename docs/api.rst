API Reference
=============


.. currentmodule:: thermoextrap



Data Models
-----------

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst
   :recursive:

   DataCentralMoments
   DataCentralMomentsVals


   DataValues
   DataValuesCentral


.. autosummary::
   :toctree: generated/

   factory_data_values
   resample_indices


General extrapolation and interpolation models
----------------------------------------------

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst


   ExtrapModel
   ExtrapWeightedModel
   InterpModel
   InterpModelPiecewise
   MBARModel
   PerturbModel
   StateCollection



Specific models
---------------

Inverse temperature extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: custom-module-single.rst
   :recursive:

   beta


Volume extrapolation
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: custom-module-single.rst
   :recursive:

   volume


TMMC :math:`\ln \Pi(N)` extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: custom-module-single.rst
   :recursive:

   lnpi
