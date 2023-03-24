# Top level API ({mod}`thermoextrap`)


:::{eval-rst}

.. automodule:: thermoextrap
   :no-members:
   :no-inherited-members:
   :no-special-members:

:::


## Data Models


These classes/routines are made available at the top level by importing from {mod}`thermoextrap.data`


:::{eval-rst}

.. currentmodule:: thermoextrap.data

.. autosummary::

    DataCentralMoments
    DataCentralMomentsVals
    DataValues
    DataValuesCentral

    factory_data_values
    resample_indices

:::

## General extrapolation and interpolation models

These classes/routines are made available at the top level by importing from {mod}`thermoextrap.models`

```{eval-rst}
.. currentmodule:: thermoextrap.models

.. autosummary::

   ExtrapModel
   ExtrapWeightedModel
   InterpModel
   InterpModelPiecewise
   MBARModel
   PerturbModel
   StateCollection
   Derivatives
```


<!-- ## Specific models -->

<!-- ### Inverse temperature extrapolation -->

<!-- ```{eval-rst} -->
<!-- .. currentmodule:: thermoextrap.beta -->

<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->
<!--    :template: custom-class.rst -->
<!--    :recursive: -->

<!--    SymDerivBeta -->
<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->

<!--    factory_derivatives -->
<!--    factory_extrapmodel -->
<!--    factory_perturbmodel -->


<!-- ``` -->

<!-- ### Volume extrapolation -->

<!-- ```{eval-rst} -->
<!-- .. currentmodule:: thermoextrap.volume -->
<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->
<!--    :template: custom-class.rst -->
<!--    :recursive: -->

<!--    VolumeDerivFuncs -->
<!--    VolumeDataCallback -->
<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->

<!--    factory_derivatives -->
<!--    factory_extrapmodel -->
<!-- ``` -->

<!-- ### Ideal gas volume extrapolation -->

<!-- ```{eval-rst} -->
<!-- .. currentmodule:: thermoextrap.volume_idealgas -->
<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->
<!--    :template: custom-class.rst -->
<!--    :recursive: -->

<!--    VolumeDerivFuncsIG -->
<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->

<!--    factory_derivatives -->
<!--    factory_extrapmodel -->
<!--    factory_extrapmodel_data -->

<!-- ``` -->

<!-- ### TMMC $\ln \Pi(N)$ extrapolation -->

<!-- ```{eval-rst} -->
<!-- .. currentmodule:: thermoextrap.lnpi -->

<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->
<!--    :template: custom-class.rst -->
<!--    :recursive: -->

<!--    lnPiDataCallback -->
<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->

<!--    factory_derivatives -->
<!--    factory_extrapmodel_lnPi -->


<!-- ``` -->

<!-- ### Gaussian Process Regression -->

<!-- ```{eval-rst} -->
<!-- .. currentmodule:: thermoextrap.gpr_active -->

<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->
<!--    :template: custom-class.rst -->
<!--    :recursive: -->

<!--    ig_active.IG_DataWrapper -->
<!--    ig_active.SimulateIG -->

<!--    active_utils.DataWrapper -->
<!--    active_utils.SimWrapper -->
<!--    active_utils.ChangeInnerOuterRBFDerivKernel -->
<!--    active_utils.UpdateALMbrute -->
<!--    active_utils.UpdateSpaceFill -->
<!--    active_utils.UpdateFuncBase -->
<!--    active_utils.MaxRelGlobalVar -->
<!--    active_utils.MaxAbsRelGlobalDeviation -->
<!--    active_utils.MaxRelVar -->
<!--    active_utils.StopCriteria -->


<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->

<!--    active_utils.active_learning -->
<!--    active_utils.get_logweights -->
<!--    active_utils.input_GP_from_state -->
<!--    active_utils.create_base_GP_model -->
<!--    active_utils.train_GPR -->
<!--    active_utils.create_GPR -->
<!--    active_utils.make_poly_expr -->
<!--    active_utils.make_matern_expr -->
<!--    active_utils.make_rbf_expr -->
<!-- ``` -->

<!-- ```{eval-rst} -->
<!-- .. autosummary:: -->
<!--    :toctree: generated/ -->
<!--    :template: custom-class.rst -->
<!--    :recursive: -->



<!--    gp_models.DerivativeKernel -->
<!--    gp_models.HetGaussianDeriv -->
<!--    gp_models.HeteroscedasticGPR -->
<!-- ``` -->
