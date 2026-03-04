"""
Adaptive interpolation (:mod:`~thermoextrap.adaptive_interp`)
=============================================================

Holds recursive interpolation class.
This includes the recursive training algorithm and consistency checks.

See :ref:`examples/usage/basic/temperature_interp:adaptive interpolation` for example usage.

"""
# pyright: reportMissingTypeStubs=false, reportMissingImports=false, reportUnknownVariableType=warning, reportUnknownMemberType=warning

from __future__ import annotations

import logging
from itertools import chain, islice, pairwise
from typing import TYPE_CHECKING, Generic

import numpy as np
import xarray as xr

from .core.typing import DataT
from .core.typing_compat import TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

    from .core.typing import (
        OptionalKwsAny,
        OptionalRng,
        SupportsModelDerivs,
        SupportsModelDerivsDataArrayT,
    )
    from .core.typing_compat import Concatenate, NotRequired, TypeAlias, TypeVar
    from .models import ExtrapModel, StateCollection

    _T = TypeVar("_T")

    FactoryState: TypeAlias = Callable[
        Concatenate[float, ...], SupportsModelDerivsDataArrayT
    ]

    FactoryStateCollection: TypeAlias = Callable[
        Concatenate[Sequence[SupportsModelDerivsDataArrayT], ...],
        StateCollection[xr.DataArray, SupportsModelDerivsDataArrayT],
    ]

logging.basicConfig()
logger = logging.getLogger(__name__)


def window(seq: Iterable[_T], n: int = 2) -> Iterator[tuple[_T, ...]]:
    """
    Returns a sliding window (of width n) over data from `seq`.

    ``s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...``
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = (*result[1:], elem)
        yield result


def relative_fluctuations(da: DataT, dim: str) -> tuple[DataT, DataT]:
    """Calculate relative mean and relative error of DataArray along dimension."""
    ave = da.mean(dim)
    err = da.std(dim) / np.abs(ave)
    err = err.where(~np.isinf(err))

    return ave, err


class _InfoDict(TypedDict, Generic[DataT], total=True):
    """Info dictionary."""

    alpha0: list[float]
    err: DataT
    ave: DataT
    alpha_new: NotRequired[float]
    err_max: NotRequired[float]
    depth: NotRequired[int]


def _check_relative_fluctuations(
    alphas: NDArray[Any],
    model: StateCollection[xr.DataArray, SupportsModelDerivs[xr.DataArray]],
    states: Sequence[SupportsModelDerivs[xr.DataArray]],
    reduce_dim: str = "rep",
    # states_avail=None,
    predict_kws: OptionalKwsAny = None,
    tol: float = 0.003,
    alpha_tol: float = 0.01,
) -> tuple[float | None, _InfoDict[xr.DataArray]]:
    """Test relative fluctuations of model."""
    if predict_kws is None:
        predict_kws = {}

    alpha_name = model.alpha_name
    alphas_states_dim = f"_{alpha_name}_states"

    ave, err_rel = relative_fluctuations(
        da=model.predict(alphas, **predict_kws),
        dim=reduce_dim,
    )

    # take maximum over all dimenensions but alpha_name
    max_dims = set(err_rel.dims) - {alpha_name}
    if len(max_dims) > 0:
        err_rel = err_rel.max(dims=max_dims)

    # collect info before reduce err_rel below
    info: _InfoDict[xr.DataArray] = {"alpha0": model.alpha0, "err": err_rel, "ave": ave}

    # only consider values > tol
    err_rel = err_rel.where(err_rel > tol, drop=True)

    # only consider values sufficiently far from current states
    if len(err_rel) > 0 and len(states) > 0 and alpha_tol > 0:
        alphas_states = xr.DataArray([s.alpha0 for s in states], dims=alphas_states_dim)
        err_rel = err_rel.where(
            np.abs(err_rel[alpha_name] - alphas_states).min(alphas_states_dim)  # type: ignore[arg-type]  # pyright: ignore[reportCallIssue, reportArgumentType]
            > alpha_tol,
            drop=True,
        )

    if len(err_rel) > 0:
        alpha_new = float(err_rel.idxmax(alpha_name))
        info["alpha_new"] = alpha_new
        info["err_max"] = float(err_rel.max())
    else:
        alpha_new = None

    return alpha_new, info


def train_iterative(
    alphas: ArrayLike,
    factory_state: FactoryState[SupportsModelDerivsDataArrayT],
    factory_statecollection: FactoryStateCollection[SupportsModelDerivsDataArrayT],
    states: Sequence[SupportsModelDerivsDataArrayT] | None = None,
    reduce_dim: str = "rep",
    maxiter: int = 10,
    state_kws: OptionalKwsAny = None,
    statecollection_kws: OptionalKwsAny = None,
    predict_kws: OptionalKwsAny = None,
    tol: float = 0.003,
    alpha_tol: float = 0.01,
    callback: Callable[..., Any] | None = None,
    callback_kws: OptionalKwsAny = None,
) -> tuple[
    StateCollection[xr.DataArray, SupportsModelDerivsDataArrayT],
    list[_InfoDict[xr.DataArray]],
]:
    """
    Add states to satisfy some tolerance.

    Each iteration calculates the relative error, then adds a state
    where error is largest.

    NOTE:  The big difference between this and the recursive interpolation
    is that a single set of alphas is passed, and always considered.  That is, each
    iteration considers the whole range of alphas.  If this is undesirable, we go back to
    a recursive?


    Parameters
    ----------
    alphas : array-like
        values of alpha to calculate along
    factory_state : callable
        state creation factory function.
        `state = factory_state(alpha, **state_kws)`.
        This state must have a dimension `reduce_dim`.
    factory_statecollection : callable
        state collection factory.
        `model = factory_statecollection(states)`
    states : list of object, optional
        initial states list.  If not passed, first guess at states is
        `[factory_state(alphas[0]), factory_state(alphas[-1])]`
    reduce_dim : str
        dimension to calculate statistics along.
    maxiter : int, default=10
        number of iterations
    state_kws : dict, optional
        extra arguments to `factory_state`
    statecollection_kws : dict, optional
        extra arguments to `factory_statecollection`
    predict_kws : dict, optional
        extra arguments to `model.predict(alphas, **predict_kws)`
    tol : float, default=0.003
        relative tolerance.  If `max err_rel < tol` then not new state added
    alpha_tol : float, default=0.01
        new states must have `abs(alpha_new - alpha) > alpha_tol` for all existing states.
    callback : callable
        `stop = callback(model, alphas, info_dict, **callback_kws)`.
        If callback returns something that evaluates True, then the iteration stops.
        * model : current model.
        * alphas : sequence of alphas
        * info_dict : dictionary containing current estimate information
        * info_dict['alpha0'] : alpha0 values in the model
        * info_dict['err'] : relative error in the model
        * info_dict['ave'] : average estimate of the model
        * info_dict['depth'] : depth of interaction
    callback_kws : dict, optional
        extra arguments to `callback`


    Returns
    -------
    model : :class:`thermoextrap.models.StateCollection` instance
        final output of `factory_statecollection`
    info : list of dict
        Information from each iteration

    """
    if state_kws is None:
        state_kws = {}
    if statecollection_kws is None:
        statecollection_kws = {}
    if predict_kws is None:
        predict_kws = {}
    if callback_kws is None:
        callback_kws = {}

    alphas = np.atleast_1d(alphas)
    if states is None:
        states = [
            factory_state(alphas[0], **state_kws),
            factory_state(alphas[-1], **state_kws),
        ]

    # assert maxiter > 0
    if maxiter <= 0:
        msg = f"{maxiter=} must be positive"
        raise ValueError(msg)

    # work with copy
    states = list(states)
    info: list[_InfoDict[xr.DataArray]] = []

    for depth in range(maxiter):
        model = factory_statecollection(states, **statecollection_kws)

        alpha_new, info_dict = _check_relative_fluctuations(
            alphas=alphas,
            model=model,
            states=states,
            reduce_dim=reduce_dim,
            # states_avail=states_avail,
            predict_kws=predict_kws,
            tol=tol,
            alpha_tol=alpha_tol,
        )

        info_dict["depth"] = depth
        info.append(info_dict)

        if callback is not None and callback(model, alphas, info_dict, **callback_kws):
            break

        if alpha_new is not None:
            state_new = factory_state(alpha_new, **state_kws)
            states = sorted([*states, state_new], key=lambda x: x.alpha0)
        else:
            break

    return model, info  # pyright: ignore[reportPossiblyUnboundVariable]


def train_recursive(  # noqa: C901,PLR0913,PLR0914,PLR0917
    alphas: ArrayLike,
    factory_state: FactoryState[SupportsModelDerivsDataArrayT],
    factory_statecollection: FactoryStateCollection[SupportsModelDerivsDataArrayT,],
    state0: SupportsModelDerivsDataArrayT | None = None,
    state1: SupportsModelDerivsDataArrayT | None = None,
    states: Sequence[SupportsModelDerivsDataArrayT] | None = None,
    info: list[_InfoDict[xr.DataArray]] | None = None,
    reduce_dim: str = "rep",
    depth: int = 0,
    maxiter: int = 10,
    state_kws: OptionalKwsAny = None,
    statecollection_kws: OptionalKwsAny = None,
    predict_kws: OptionalKwsAny = None,
    tol: float = 0.003,
    alpha_tol: float = 0.01,
    callback: Callable[
        Concatenate[
            StateCollection[xr.DataArray, SupportsModelDerivsDataArrayT],
            ArrayLike,
            Mapping[str, Any],
            ...,
        ],
        Any,
    ]
    | None = None,
    callback_kws: OptionalKwsAny = None,
) -> tuple[list[SupportsModelDerivsDataArrayT], list[_InfoDict[xr.DataArray]]]:
    """
    Add states to satisfy some tolerance.

    Each iteration calculates the relative error, then adds a state
    where error is largest.

    NOTE:  The big difference between this and the recursive interpolation
    is that a single set of alphas is passed, and always considered.  That is, each
    iteration considers the whole range of alphas.  If this is undesirable, we go back to
    a recursive?


    Parameters
    ----------
    alphas : array-like
        values of alpha to calculate along
    factory_state : callable
        state creation factory function.
        `state = factory_state(alpha, **state_kws)`.
        This state must have a dimension `reduce_dim`.
    factory_statecollection : callable
        state collection factory.
        `model = factory_statecollection(states)`
    state0, state1 : object
        states to be used for building model.
        defaults to building states at `alphas[0]` and `alphas[-1]`
    states : list of object, optional
        initial states list.  If not passed, first guess at states is
        `[factory_state(alphas[0]), factory_state(alphas[-1])]`
    reduce_dim : str
        dimension to calculate statistics along.
    maxiter : int, default=10
        number of iterations
    state_kws : dict, optional
        extra arguments to `factory_state`
    statecollection_kws : dict, optional
        extra arguments to `factory_statecollection`
    predict_kws : dict, optional
        extra arguments to `model.predict(alphas, **predict_kws)`
    tol : float, default=0.003
        relative tolerance.  If `max err_rel < tol` then not new state added
    alpha_tol : float, default=0.01
        new states must have `abs(alpha_new - alpha) > alpha_tol` for all existing states.
    callback : callable
        `stop = callback(model, alphas, info_dict, **callback_kws)`.
        If callback returns something that evaluates True, then the iteration stops.
        `model` is the current model.
        `alphas` is the sequence of alphas
        `info_dict` dictionary containing
        `info_dict['alpha0']` the alpha0 values in the model
        `info_dict['err']` the normalized error in the model
        `info_dict['depth']` the depth of interaction
    callback_kws : dict, optional
        extra arguments to `callback`
    depth : int
        Internal variable used during recursion.
    info : list
        Internal variable used during recursion.



    Returns
    -------
    states : list of object
        list of states
    info : list of dict
        Information from each iteration

    """
    states = [] if states is None else list(states)
    info = [] if info is None else list(info)

    if depth >= maxiter:
        return states, info

    if state_kws is None:
        state_kws = {}
    if statecollection_kws is None:
        statecollection_kws = {}
    if predict_kws is None:
        predict_kws = {}
    if callback_kws is None:
        callback_kws = {}

    alphas = np.atleast_1d(alphas)

    def get_state(
        alpha: float, states: Iterable[SupportsModelDerivsDataArrayT]
    ) -> SupportsModelDerivsDataArrayT:
        states_dict = {s.alpha0: s for s in states}
        if alpha in states_dict:
            return states_dict[alpha]
        return factory_state(alpha, **state_kws)

    if state0 is None:
        state0 = get_state(alphas[0], states)
    if state1 is None:
        state1 = get_state(alphas[-1], states)

    # alpha_name = state0.alpha_name
    # alphas_states_dim = f"_{alpha_name}_states"

    model = factory_statecollection([state0, state1], **statecollection_kws)
    alpha0, alpha1 = model.alpha0

    alpha_new, info_dict = _check_relative_fluctuations(
        alphas=alphas,
        model=model,
        states=states,
        reduce_dim=reduce_dim,
        # states_avail=states_avail,
        predict_kws=predict_kws,
        tol=tol,
        alpha_tol=alpha_tol,
    )

    info_dict["depth"] = depth
    info = [*info, info_dict]

    if callback is not None and callback(model, alphas, info_dict, **callback_kws):
        alpha_new = None

    if alpha_new is not None:
        state_new = get_state(alpha_new, states)

        alphas_left = alphas[(alpha0 <= alphas) & (alphas < alpha_new)]
        states, info = train_recursive(
            alphas=alphas_left,
            state0=state0,
            state1=state_new,
            factory_state=factory_state,
            factory_statecollection=factory_statecollection,
            states=states,
            info=info,
            reduce_dim=reduce_dim,
            depth=depth + 1,
            maxiter=maxiter,
            # states_avail=states_avail,
            state_kws=state_kws,
            statecollection_kws=statecollection_kws,
            predict_kws=predict_kws,
            tol=tol,
            alpha_tol=alpha_tol,
            callback=callback,
            callback_kws=callback_kws,
        )

        alphas_right = alphas[(alpha_new <= alphas) & (alphas <= alpha1)]
        states, info = train_recursive(
            alphas=alphas_right,
            state0=state_new,
            state1=state1,
            factory_state=factory_state,
            factory_statecollection=factory_statecollection,
            states=states,
            info=info,
            reduce_dim=reduce_dim,
            depth=depth + 1,
            maxiter=maxiter,
            # states_avail=states_avail,
            state_kws=state_kws,
            statecollection_kws=statecollection_kws,
            predict_kws=predict_kws,
            tol=tol,
            alpha_tol=alpha_tol,
            callback=callback,
            callback_kws=callback_kws,
        )

    else:
        alphas_states = {s.alpha0 for s in states}
        for alpha, state in zip([alpha0, alpha1], [state0, state1], strict=True):
            if alpha not in alphas_states:
                states.append(state)
        states = sorted(states, key=lambda x: x.alpha0)

    return states, info


def check_polynomial_consistency(
    states: Sequence[SupportsModelDerivsDataArrayT],
    factory_statecollection: FactoryStateCollection[SupportsModelDerivsDataArrayT],
    reduce_dim: str = "rep",
    statecollection_kws: OptionalKwsAny = None,
) -> tuple[
    dict[Any, xr.DataArray],
    dict[Any, StateCollection[xr.DataArray, SupportsModelDerivsDataArrayT]],
]:
    """
    Check polynomial consistency across subsegments.

    Parameters
    ----------
    states : sequence of object
        sequence of states
    factory_statecollection : callable
        `model = factory_statecollection(states, **statecollection_kws)`
    reduce_dim : str, default="rep"
        dimension to reduce along
    statecollection_kws : dict, optional
        extra arguments to `factory_statecollection`

    Returns
    -------
    p_values : dict
        p value for pairs of models.  Keys will be of the form
        ((alpha0, alpha1), (alpha2, alpha3))
    models : dict
        collection of models created.  Keys are of the form
        (alpha0, alpha1)
    """
    from scipy import stats

    ave = {}
    var = {}
    models: dict[Any, StateCollection[xr.DataArray, SupportsModelDerivsDataArrayT]] = {}

    if statecollection_kws is None:
        statecollection_kws = {}

    key: tuple[Any, ...]
    for state_pair in chain(
        pairwise(states), zip(states[:-2], states[2:], strict=True)
    ):
        model = factory_statecollection(list(state_pair), **statecollection_kws)
        key = tuple(model.alpha0)
        coef = model.coefs(order=None)

        ave[key] = coef.mean(reduce_dim)
        var[key] = coef.var(reduce_dim)

        models[key] = model

    # build up p values
    ps: dict[Any, xr.DataArray] = {}
    for keys in window((s.alpha0 for s in states), n=3):
        keys01 = keys[0], keys[1]
        keys12 = keys[1], keys[2]
        keys02 = keys[0], keys[2]

        for key0, key1 in ((keys01, keys12), (keys01, keys02), (keys12, keys02)):
            if (key := key0, key1) not in ps:
                z: xr.DataArray = (ave[key0] - ave[key1]) / np.sqrt(  # pyright: ignore[reportUnknownVariableType]
                    var[key0] + var[key1]  # pyright: ignore[reportUnknownArgumentType]
                )
                zv: NDArray[Any] = np.abs(z.to_numpy())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                p: xr.DataArray = z.copy(data=stats.norm.cdf(zv) - stats.norm.cdf(-zv))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                ps[key] = p

    return ps, models


# Utility functions for Examples/testing
# need a function to create states
def factory_state_idealgas(
    beta: float,
    order: int,
    nrep: int = 100,
    rep_dim: str = "rep",
    nconfig: int = 10_000,
    npart: int = 1_000,
    rng: OptionalRng = None,
) -> ExtrapModel[xr.DataArray]:
    """
    Example factory function to create single state.

    This particular state function returns the a `beta` extrapolation model for the position
    of an ideal gas particle in an external field.

    This can be as complicated as you want.  It just needs to return a state at the given value of alpha.
    It also should have a dimension with the same name as `reduce_dim` below
    Here, that dimension is 'rep', the resampled/replicated dimension

    Extra arguments can be passed via the state_kws dictionary

    Parameters
    ----------
    rng: :class:`numpy.random.Generator`, optional

    See Also
    --------
    thermoextrap.idealgas
    thermoextrap.beta.factory_extrapmodel
    """
    from cmomy.random import validate_rng

    from . import beta as xpan_beta
    from . import idealgas
    from .data import DataCentralMomentsVals

    rng = validate_rng(rng)

    xdata, udata = (
        xr.DataArray(_, dims="rec")
        for _ in idealgas.generate_data(shape=(nconfig, npart), beta=beta, rng=rng)
    )
    data = DataCentralMomentsVals.from_vals(xv=xdata, uv=udata, order=order)

    # use indices for reproducibility
    nrec = len(xdata)
    indices = rng.choice(nrec, (nrep, nrec))
    return xpan_beta.factory_extrapmodel(beta=beta, data=data).resample(
        sampler={"indices": indices}, rep_dim=rep_dim
    )


def callback_plot_progress(
    model: StateCollection[xr.DataArray, SupportsModelDerivsDataArrayT],
    alphas: ArrayLike,  # noqa: ARG001
    info_dict: _InfoDict[xr.DataArray],
    verbose: bool = True,
    maxdepth_stop: int | None = None,
    ax: Any = None,
) -> bool:
    """
    The callback function is called each iteration after model is created.

    Optionally, it can return value `True` to stop iteration


    Parameters
    ----------
    verbose : bool, default=True
    maxdepth_stop : int, optional
        Note that this is redundant with `maxdepth`, but for demonstration
        purposes
    ax : :class:`matplotlib.axes.Axes`, optional
    """
    import matplotlib.pyplot as plt

    from . import idealgas

    if (depth := info_dict.get("depth")) is None:
        msg = "Must have `depth` parameter in info_dict"
        raise ValueError(msg)

    if verbose:
        logger.setLevel(logging.INFO)

    logger.info("depth: %s", depth)
    logger.info("alphas: %s", model.alpha0)

    if ax is None:
        _, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]

    pred = info_dict["ave"]
    pred.plot(ax=ax)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]

    # absolute:
    idealgas.x_ave(pred.beta).plot(ls=":", color="k", ax=ax)

    if (alpha_new := info_dict.get("alpha_new")) is not None:
        logger.info("alpha_new: %s", alpha_new)
        ax.axvline(x=alpha_new, ls=":")
    plt.show()  # pyright: ignore[reportUnknownMemberType]

    # demo of coding in stop criteria
    if maxdepth_stop is not None:
        if stop := depth > maxdepth_stop:
            logger.info("reached maxdepth_stop in callback")
        return stop
    return False


# def plot_polynomial_consistency(alphas, states, factory_statecollection):
#     """Plotter for polynomial consistency."""
#     import matplotlib.pyplot as plt

#     p_values, models_dict = check_polynomial_consistency(
#         states, factory_statecollection
#     )

#     hit = set()
#     for (key0, key1), p in p_values.items():
#         print(
#             "range0: {} range1:{} p01: {}".format(
#                 *(np.round(x, 3) for x in [key0, key1, p.values])
#             )
#         )

#         lb = min(k[0] for k in (key0, key1))
#         ub = max(k[1] for k in (key0, key1))

#         alphas_lim = alphas[(lb <= alphas) & (alphas <= ub)]

#         for key in key0, key1:
#             if key not in hit:
#                 models_dict[key].predict(alphas_lim).mean("rep").plot(
#                     label=str(np.round(key, 3))
#                 )
#                 hit.add(key)

#     plt.legend()
#     return p_values, models_dict
