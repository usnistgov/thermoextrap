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
   ..
      ~core.data.DataCentralMomentsBase


   DataValues
   DataValuesCentral

   AbstractData


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
   Derivatives




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



Gaussian Process Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: thermoextrap.gpr_active


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst
   :recursive:

   ig_active.IG_DataWrapper
   ig_active.SimulateIG

   active_utils.DataWrapper
   active_utils.SimWrapper
   active_utils.ChangeInnerOuterRBFDerivKernel
   active_utils.UpdateALMbrute
   active_utils.UpdateSpaceFill
   active_utils.UpdateFuncBase
   active_utils.MaxRelGlobalVar
   active_utils.MaxAbsRelGlobalDeviation
   active_utils.MaxRelVar
   active_utils.StopCriteria



.. autosummary::
   :toctree: generated/

   active_utils.active_learning
   active_utils.get_logweights
   active_utils.input_GP_from_state
   active_utils.create_base_GP_model
   active_utils.train_GPR
   active_utils.create_GPR
   active_utils.make_poly_expr
   active_utils.make_matern_expr
   active_utils.make_rbf_expr

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst
   :recursive:



   gp_models.DerivativeKernel
   gp_models.HetGaussianDeriv
   gp_models.HeteroscedasticGPR
