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

.. currentmodule:: thermoextrap.beta


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst
   :recursive:

   SymDerivBeta

.. autosummary::
   :toctree: generated/

   factory_derivatives
   factory_extrapmodel
   factory_perturbmodel



Volume extrapolation
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: thermoextrap.volume

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst
   :recursive:

   VolumeDerivFuncs
   VolumeDataCallback

.. autosummary::
   :toctree: generated/

   factory_derivatives
   factory_extrapmodel

Ideal gas volume extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: thermoextrap.volume_idealgas

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst
   :recursive:

   VolumeDerivFuncsIG

.. autosummary::
   :toctree: generated/

   factory_derivatives
   factory_extrapmodel
   factory_extrapmodel_data


TMMC :math:`\ln \Pi(N)` extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: thermoextrap.lnpi


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst
   :recursive:

   lnPiDataCallback

.. autosummary::
   :toctree: generated/

   factory_derivatives
   factory_extrapmodel_lnPi
