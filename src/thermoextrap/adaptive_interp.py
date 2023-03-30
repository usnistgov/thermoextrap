"""
Adaptive interpolation (:mod:`~thermoextrap.adaptive_interp`)
=============================================================

Holds recursive interpolation class.
This includes the recursive training algorithm and consistency checks.

See :ref:`notebooks/temperature_interp:adaptive interpolation` for example usage.

"""


from itertools import chain, islice

import numpy as np
import xarray as xr
from scipy import stats


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from `seq`.

    ``s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...``
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def relative_fluctuations(da, dim):
    """Calculate relative mean and relative error of DataArray along dimension."""
    ave = da.mean(dim)
    err = da.std(dim) / np.abs(ave)
    err = err.where(~np.isinf(err))

    return ave, err


def test_relative_fluctuations(
    alphas,
    model,
    states,
    reduce_dim="rep",
    states_avail=None,
    predict_kws=None,
    tol=0.003,
    alpha_tol=0.01,
):
    """Test relative fluctuations of model."""

    if predict_kws is None:
        predict_kws = {}

    alpha_name = model.alpha_name
    alphas_states_dim = f"_{alpha_name}_states"

    ave, err_rel = model.predict(alphas, **predict_kws).pipe(
        relative_fluctuations, dim=reduce_dim
    )

    # take maximum over all dimenensions but alpha_name
    max_dims = set(err_rel.dims) - {alpha_name}
    if len(max_dims) > 0:
        err_rel = err_rel.max(dims=max_dims)

    # collect info before reduce err_rel below
    info = {"alpha0": model.alpha0, "err": err_rel, "ave": ave}

    # only consider values > tol
    err_rel = err_rel.where(err_rel > tol, drop=True)

    # only consider values sufficiently far from current states
    if len(err_rel) > 0 and len(states) > 0 and alpha_tol > 0:
        alphas_states = xr.DataArray([s.alpha0 for s in states], dims=alphas_states_dim)
        err_rel = err_rel.where(
            np.abs(err_rel[alpha_name] - alphas_states).min(alphas_states_dim)
            > alpha_tol,
            drop=True,
        )

    if len(err_rel) > 0:
        alpha_new = err_rel.idxmax(alpha_name).values[()]
        info["alpha_new"] = alpha_new
        info["err_max"] = err_rel.max().values[()]
    else:
        alpha_new = None

    return alpha_new, info


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
    callback=None,
    callback_kws=None,
):
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
    if callback is not None and callback_kws is None:
        callback_kws = {}

    if states is None:
        states = [
            factory_state(alphas[0], **state_kws),
            factory_state(alphas[-1], **state_kws),
        ]

    assert maxiter > 0

    # work with copy
    states = list(states)
    info = []

    for depth in range(maxiter):
        model = factory_statecollection(states, **statecollection_kws)

        alpha_new, info_dict = test_relative_fluctuations(
            alphas=alphas,
            model=model,
            states=states,
            reduce_dim=reduce_dim,
            states_avail=states_avail,
            predict_kws=predict_kws,
            tol=tol,
            alpha_tol=alpha_tol,
        )

        info_dict["depth"] = depth
        info.append(info_dict)

        if callback is not None:
            if callback(model, alphas, info_dict, **callback_kws):
                break

        if alpha_new is not None:
            state_new = factory_state(alpha_new, **state_kws)
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
    callback=None,
    callback_kws=None,
):
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
    if callback is not None and callback_kws is None:
        callback_kws = {}

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

    # alpha_name = state0.alpha_name
    # alphas_states_dim = f"_{alpha_name}_states"

    model = factory_statecollection([state0, state1], **statecollection_kws)
    alpha0, alpha1 = model.alpha0

    alpha_new, info_dict = test_relative_fluctuations(
        alphas=alphas,
        model=model,
        states=states,
        reduce_dim=reduce_dim,
        states_avail=states_avail,
        predict_kws=predict_kws,
        tol=tol,
        alpha_tol=alpha_tol,
    )

    info_dict["depth"] = depth
    info = info + [info_dict]

    if callback is not None:
        if callback(model, alphas, info_dict, **callback_kws):
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
            states_avail=states_avail,
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
            states_avail=states_avail,
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
        for alpha, state in zip([alpha0, alpha1], [state0, state1]):
            if alpha not in alphas_states:
                states.append(state)
        states = sorted(states, key=lambda x: x.alpha0)

    return states, info


def check_polynomial_consistency(
    states,
    factory_statecollection,
    reduce_dim="rep",
    order=None,
    statecollection_kws=None,
):
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
    order : int, optional
        order passed to `model.predict`
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

    ave = {}
    var = {}
    models = {}

    for state_pair in chain(zip(states[:-1], states[1:]), zip(states[:-2], states[2:])):
        model = factory_statecollection(list(state_pair))
        key = tuple(model.alpha0)
        coef = model.coefs(order=None)

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
            key = key0, key1
            if key not in ps:
                z = (ave[key0] - ave[key1]) / np.sqrt(var[key0] + var[key1])
                p = xr.DataArray(
                    stats.norm.cdf(np.abs(z)) - stats.norm.cdf(-np.abs(z)),
                    dims=z.dims,
                    coords=z.coords,
                )
                ps[key] = p

    return ps, models


# Utility functions for Examples/testing
# need a function to create states
def factory_state_idealgas(
    beta,
    order,
    nrep=100,
    rep_dim="rep",
    seed_from_beta=True,
    nconfig=10_000,
    npart=1_000,
):  # noqa: D417
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
    seed_from_beta : bool, default=True
        If `True`, then set `np.random.seed` based on beta value.
        For testing purposes

    See Also
    --------
    thermoextrap.idealgas
    thermoextrap.beta.factory_extrapmodel
    """

    from . import beta as xpan_beta
    from .core import idealgas
    from .core.data import DataCentralMomentsVals

    # NOTE: this is for reproducible results.
    if seed_from_beta:
        np.random.seed(int(beta * 1000))

    xdata, udata = idealgas.generate_data(shape=(nconfig, npart), beta=beta)
    data = DataCentralMomentsVals.from_vals(xv=xdata, uv=udata, order=order)

    # use indices for reproducibility
    nrec = len(xdata)
    indices = np.random.choice(nrec, (nrep, nrec))
    return xpan_beta.factory_extrapmodel(beta=beta, data=data).resample(
        indices=indices, rep_dim=rep_dim
    )


def callback_plot_progress(  # noqa: D417
    model, alphas, info_dict, verbose=True, maxdepth_stop=None, ax=None
):
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

    from .core import idealgas

    if verbose:
        print("depth:", info_dict["depth"])
        print("alphas:", model.alpha0)

    if ax is None:
        _, ax = plt.subplots()

    pred = info_dict["ave"]
    pred.plot(ax=ax)

    # absolute:
    idealgas.x_ave(pred.beta).plot(ls=":", color="k", ax=ax)

    alpha_new = info_dict.get("alpha_new", None)
    if alpha_new is not None:
        if verbose:
            print("alpha_new:", alpha_new)
        ax.axvline(x=alpha_new, ls=":")
    plt.show()

    # demo of coding in stop criteria
    if maxdepth_stop is not None:
        stop = info_dict["depth"] > maxdepth_stop
        if stop and verbose:
            print("reached maxdepth_stop in callback")
    else:
        stop = False
    return stop


def plot_polynomial_consistency(alphas, states, factory_statecollection):
    """Plotter for polynomial consistency."""
    import matplotlib.pyplot as plt

    P, models_dict = check_polynomial_consistency(states, factory_statecollection)

    hit = set()
    for (key0, key1), p in P.items():
        print(
            "range0: {} range1:{} p01: {}".format(
                *map(lambda x: np.round(x, 3), [key0, key1, p.values])
            )
        )

        lb = min(k[0] for k in (key0, key1))
        ub = max(k[1] for k in (key0, key1))

        alphas_lim = alphas[(lb <= alphas) & (alphas <= ub)]

        for key in key0, key1:
            if key not in hit:
                models_dict[key].predict(alphas_lim).mean("rep").plot(
                    label=str(np.round(key, 3))
                )
                hit.add(key)

    plt.legend()
    return P, models_dict
