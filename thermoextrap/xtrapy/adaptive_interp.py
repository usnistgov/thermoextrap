"""Holds recursive interpolation class.
This includes the recursive training algorithm and consistency checks.
"""

from __future__ import absolute_import

from itertools import chain, islice

import numpy as np
import xarray as xr
from scipy import stats

# try:
#     import matplotlib.pyplot as plt
# except ImportError:
#     print(
#         "Could not find matplotlib - plotting will fail, so ensure that all"
#         " doPlot options are set to False, which is the default."
#     )


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def relative_fluctuations(da, dim):
    """
    Calculate relative fluctuations (std / |mean|) of DataArray
    """
    out = da.std(dim) / np.abs(da.mean(dim))
    out = out.where(~np.isinf(out))
    return out


def train_single_interation(
    alphas,
    factory_state,
    factory_statecollection,
    states=None,
    reduce_dim="rep",
    states_avail=None,
    state_kws=None,
    statecollection_kws=None,
    predict_kws=None,
    tol=0.003,
    alpha_tol=0.01,
):

    if state_kws is None:
        state_kws = {}
    if statecollection_kws is None:
        statecollection_kws = {}
    if predict_kws is None:
        predict_kws = {}

    if states is None:
        states = [
            factory_state(alphas[0], **state_kws),
            factory_state(alphas[-1], **state_kws),
        ]

    alpha_name = states[0].alpha_name
    alphas_states_dim = f"_{alpha_name}_states"

    model = factory_statecollection(states, **statecollection_kws)

    err_rel = model.predict(alphas, **predict_kws).pipe(
        relative_fluctuations, dim=reduce_dim
    )

    # take maximum over all dimenensions but alpha_name
    max_dims = set(err_rel.dims) - {alpha_name}
    if len(max_dims) > 0:
        err_rel = err_rel.max(dims=max_dims)

    # collect info before reduce err_rel below
    info = {"alpha0": model.alpha0, "err": err_rel}

    # only consider values > tol
    err_rel = err_rel.where(err_rel > tol, drop=True)

    # only consider values sufficiently far from current states
    alphas_states = xr.DataArray([s.alpha0 for s in states], dims=alphas_states_dim)
    err_rel = err_rel.where(
        np.abs(err_rel[alpha_name] - alphas_states).min(alphas_states_dim) > alpha_tol,
        drop=True,
    )

    if len(err_rel) > 0:
        alpha_new = err_rel.idxmax(alpha_name).values[()]
        state_new = factory_state(alpha_new, **state_kws)

        info["alpha_new"] = alpha_new
        info["err_rel_max"] = err_rel.max().values[()]

    else:
        state_new = None

    return model, state_new, info


def train_iterative(
    alphas,
    factory_state,
    factory_statecollection,
    states=None,
    reduce_dim="rep",
    maxiter=10,
    states_avail=None,
    state_kws=None,
    statecollection_kws=None,
    predict_kws=None,
    tol=0.003,
    alpha_tol=0.01,
):
    """
    add states to satisfy some tolerance.

    Each iteration calculates the relative error, then adds a state
    where error is largest.

    NOTE:  The big difference between this and the recursive interopolation
    is that a single set of alphas is passed, and always considered.  That is, each
    iteration considers the whole range of alphas.  If this is undeirable, we go back to
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
    states : list of states, optional
        initial states list.  If not passed, first guess at states is
        `[factory_state(alphas[0]), factory_state(alphas[-1])]`
    reduce_dim : str
        dimension to calculate statistics along.
    maxiter : int, default=10
        number of iterations
    states_avail : list, optional
        Not implemented yet
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


    Returns
    -------
    model : statecollection
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

    if states is None:
        states = [
            factory_state(alphas[0], **state_kws),
            factory_state(alphas[-1], **state_kws),
        ]

    assert maxiter > 0

    alpha_name = states[0].alpha_name
    alphas_states_dim = f"_{alpha_name}_states"

    # work with copy
    states = list(states)
    info = []

    for count in range(maxiter):
        model = factory_statecollection(states, **statecollection_kws)

        err_rel = model.predict(alphas, **predict_kws).pipe(
            relative_fluctuations, dim=reduce_dim
        )

        # take maximum over all dimenensions but alpha_name
        max_dims = set(err_rel.dims) - {alpha_name}
        if len(max_dims) > 0:
            err_rel = err_rel.max(dims=max_dims)

        # collect info before reduce err_rel below
        # info.append({"alpha0": model.alpha0, "err": err_rel})
        info.append({"alpha0": model.alpha0, "err": err_rel})

        # only consider values > tol
        err_rel = err_rel.where(err_rel > tol, drop=True)

        # only consider values sufficiently far from current states
        alphas_states = xr.DataArray([s.alpha0 for s in states], dims=alphas_states_dim)
        err_rel = err_rel.where(
            np.abs(err_rel[alpha_name] - alphas_states).min(alphas_states_dim)
            > alpha_tol,
            drop=True,
        )

        if len(err_rel) > 0:
            alpha_new = err_rel.idxmax(alpha_name).values[()]
            state_new = factory_state(alpha_new, **state_kws)
            info[-1]["alpha_new"] = alpha_new

            states = sorted(states + [state_new], key=lambda x: x.alpha0)

        else:
            break

    return model, info


def train_recursive(
    alphas,
    factory_state,
    factory_statecollection,
    state0=None,
    state1=None,
    states=None,
    info=None,
    reduce_dim="rep",
    depth=0,
    maxiter=10,
    states_avail=None,
    state_kws=None,
    statecollection_kws=None,
    predict_kws=None,
    tol=0.003,
    alpha_tol=0.01,
):
    """
    add states to satisfy some tolerance.

    Each iteration calculates the relative error, then adds a state
    where error is largest.

    NOTE:  The big difference between this and the recursive interopolation
    is that a single set of alphas is passed, and always considered.  That is, each
    iteration considers the whole range of alphas.  If this is undeirable, we go back to
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
    states : list of states, optional
        initial states list.  If not passed, first guess at states is
        `[factory_state(alphas[0]), factory_state(alphas[-1])]`
    reduce_dim : str
        dimension to calculate statistics along.
    maxiter : int, default=10
        number of iterations
    states_avail : list, optional
        Not implemented yet
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


    Returns
    -------
    model : statecollection
        final output of `factory_statecollection`
    info : list of dict
        Information from each iteration

    """

    if states is None:
        states = []
    else:
        states = list(states)

    if info is None:
        info = []
    else:
        info = list(info)

    if depth >= maxiter:
        return states, info

    if state_kws is None:
        state_kws = {}
    if statecollection_kws is None:
        statecollection_kws = {}
    if predict_kws is None:
        predict_kws = {}

    def get_state(alpha, states):
        states_dict = {s.alpha0: s for s in states}
        if alpha in states_dict:
            return states_dict[alpha]
        else:
            return factory_state(alpha, **state_kws)

    if state0 is None:
        state0 = get_state(alphas[0], states)
    if state1 is None:
        state1 = get_state(alphas[-1], states)

    alpha_name = state0.alpha_name
    alphas_states_dim = f"_{alpha_name}_states"

    model = factory_statecollection([state0, state1], **statecollection_kws)
    alpha0, alpha1 = model.alpha0

    err_rel = model.predict(alphas, **predict_kws).pipe(
        relative_fluctuations, dim=reduce_dim
    )

    # take maximum over all dimenensions but alpha_name
    max_dims = set(err_rel.dims) - {alpha_name}
    if len(max_dims) > 0:
        err_rel = err_rel.max(dims=max_dims)

    # collect info before reduce err_rel below
    info.append({"depth": depth, "alpha0": model.alpha0, "err": err_rel})

    # only consider values > tol
    err_rel = err_rel.where(err_rel > tol, drop=True)

    if len(err_rel) > 0 and len(states) > 0:
        # only consider values sufficiently far from current states
        alphas_states = xr.DataArray([s.alpha0 for s in states], dims=alphas_states_dim)
        err_rel = err_rel.where(
            np.abs(err_rel[alpha_name] - alphas_states).min(alphas_states_dim)
            > alpha_tol,
            drop=True,
        )

    if len(err_rel) > 0:
        alpha_new = err_rel.idxmax(alpha_name).values[()]
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
            states_avail=states_avail,
            state_kws=state_kws,
            statecollection_kws=statecollection_kws,
            predict_kws=predict_kws,
            tol=tol,
            alpha_tol=alpha_tol,
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
            states_avail=states_avail,
            state_kws=state_kws,
            statecollection_kws=statecollection_kws,
            predict_kws=predict_kws,
            tol=tol,
            alpha_tol=alpha_tol,
        )

    else:
        alphas_states = set([s.alpha0 for s in states])
        for alpha, state in zip([alpha0, alpha1], [state0, state1]):
            if alpha not in alphas_states:
                states.append(state)

        states = sorted(states, key=lambda x: x.alpha0)

    return states, info


def check_polynomial_consistency(
    states, factory_statecollection, reduce_dim="rep", order=None
):
    ave = {}
    var = {}
    models = {}

    for state_pair in chain(zip(states[:-1], states[1:]), zip(states[:-2], states[2:])):

        model = factory_statecollection(list(state_pair))
        key = tuple(model.alpha0)
        coef = model.xcoefs(order=None)

        ave[key] = coef.mean(reduce_dim)
        var[key] = coef.var(reduce_dim)

        models[key] = model

    # build up p values
    ps = {}
    for keys in window((s.alpha0 for s in states), n=3):
        keys01 = keys[0], keys[1]
        keys12 = keys[1], keys[2]
        keys02 = keys[0], keys[2]

        for key0, key1 in [(keys01, keys12), (keys01, keys02), (keys12, keys02)]:
            z = (ave[key0] - ave[key1]) / np.sqrt(var[key0] + var[key1])
            p = xr.DataArray(
                stats.norm.cdf(np.abs(z)) - stats.norm.cdf(-np.abs(z)),
                dims=z.dims,
                coords=z.coords,
            )

            ps[key0, key1] = p

    return ps, models
