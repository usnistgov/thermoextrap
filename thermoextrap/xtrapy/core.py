from __future__ import absolute_import

from functools import lru_cache

import numpy as np
import sympy as sp
import xarray as xr
from scipy.special import factorial as sp_factorial

from .cached_decorators import gcached
from .data import xrwrap_alpha

try:
    from pymbar import mbar

    _HAS_PYMBAR = True
except ImportError:
    _HAS_PYMBAR = False


################################################################################
# Structure(s) to deal with analytic derivatives, etc
################################################################################


@lru_cache(100)
def _get_default_symbol(*args):
    return sp.symbols(",".join(args))


@lru_cache(100)
def _get_default_indexed(*args):
    out = [sp.IndexedBase(key) for key in args]
    if len(out) == 1:
        out = out[0]
    return out


# @lru_cache(100)
# def _get_default_function(*args):
#     out = [sp.Function(key) for key in args]
#     if len(out) == 1:
#         out = out[0]
#     return out


class SymSubs(object):
    def __init__(
        self,
        funcs,
        subs=None,
        subs_final=None,
        subs_all=None,
        recursive=True,
        simplify=False,
        expand=True,
    ):
        """
        perform substitution on stuff
        """

        self.funcs = funcs
        self.subs = subs
        self.subs_final = subs_final
        self.subs_all = subs_all

        self.recursive = recursive
        self.simplify = simplify
        self.expand = expand

    @gcached(prop=False)
    def __getitem__(self, order):
        func = self.funcs[order]

        if self.subs is not None:
            if self.recursive:
                for o in range(order, -1, -1):
                    func = func.subs(self.subs[o])
            else:
                func = func.subs(self.subs[order])

        if self.subs_final is not None:
            func = func.subs(self.subs_final[order])

        if self.subs_all is not None:
            func = func.subs(self.subs_all)

        if self.simplify:
            func = func.simplify()

        if self.expand:
            func = func.expand()

        return func


class Lambdify(object):
    """
    create python function from list of sympy expression
    """

    def __init__(self, exprs, args=None, **opts):
        """
        Parameters
        ----------
        exprs : array-like
            array of sympy expressions to lambdify
        args : array-like
            array of sympy symbols which will be in args of the resulting function
        opts : dict
            extra arguments to sympy.lambdify
        """

        self.exprs = exprs
        self.args = args
        self.opts = opts

    @gcached(prop=False)
    def __getitem__(self, order):
        return sp.lambdify(self.args, self.exprs[order], **self.opts)

    @classmethod
    def from_u_xu(cls, exprs, **opts):
        """factory for u/xu args"""
        u, xu = _get_default_indexed("u", "xu")
        # args = (u, xu)
        return cls(exprs=exprs, args=(u, xu), **opts)

    @classmethod
    def from_du_dxdu(cls, exprs, xalpha=False, **opts):
        """factory for du/dxdu args"""
        if xalpha:
            x1 = _get_default_indexed("x1")
        else:
            x1 = _get_default_symbol("x1")
        du, dxdu = _get_default_indexed("du", "dxdu")
        return cls(exprs=exprs, args=(x1, du, dxdu), **opts)


# -log<X>
class SymMinusLog(object):
    """class to take -log(X)"""

    X, dX = _get_default_indexed("X", "dX")

    @gcached(prop=False)
    def __getitem__(self, order):

        if order == 0:
            return -sp.log(self.X[0])

        expr = 0
        for k in range(1, order + 1):
            expr += (
                sp.factorial(k - 1) * (-1 / self.X[0]) ** k * sp.bell(order, k, self.dX)
            )
        # subber
        subs = {self.dX[j]: self.X[j + 1] for j in range(order + 1)}
        return expr.subs(subs).expand().simplify()


@lru_cache(5)
def factory_minus_log():
    s = SymMinusLog()
    return Lambdify(s, (s.X,))


class Coefs(object):
    """class to handle coefficients in taylor expansion"""

    def __init__(self, funcs, exprs=None):
        """
        Parameters
        ----------
        funcs : array-like
            array of functions.
            funcs[n] is the nth derivative of function
        exprs : array-like, optional
            optional placeholder for sympy expressions corresponding to the
            lambdified functions in `funcs`
        """
        self.funcs = funcs
        self.exprs = exprs

    def _apply_minus_log(self, X, order):
        func = factory_minus_log()
        return [func[i](X) for i in range(order + 1)]

    def coefs(self, *args, order, norm=True, minus_log=False):
        out = [self.funcs[i](*args) for i in range(order + 1)]
        if minus_log:
            out = self._apply_minus_log(X=out, order=order)

        if norm:
            out = [x / np.math.factorial(i) for i, x in enumerate(out)]
        return out

    def xcoefs(self, data, order=None, norm=True, minus_log=False, order_name="order"):
        if order is None:
            order = data.order
        out = self.coefs(*data.xcoefs_args, order=order, norm=norm, minus_log=minus_log)
        return xr.concat(out, dim=order_name)

    @classmethod
    def from_sympy(cls, exprs, args):
        funcs = Lambdify(exprs, args=args)
        return cls(funcs=funcs, exprs=exprs)


class ExtrapModel(object):
    """
    apply taylor series extrapolation
    """

    def __init__(
        self, alpha0, data, coefs, order=None, minus_log=False, alpha_name="alpha"
    ):
        self.alpha0 = alpha0
        self.data = data
        self.coefs = coefs

        if order is None:
            order = data.order
        self.order = order

        if minus_log is None:
            minus_log = False
        self.minus_log = minus_log

        if alpha_name is None:
            alpha_name = "alpha"
        self.alpha_name = alpha_name

    @gcached(prop=False)
    def xcoefs(self, order=None, order_name="order", norm=True, minus_log=None):
        if minus_log is None:
            minus_log = self.minus_log
        if order is None:
            order = self.order
        return self.coefs.xcoefs(
            self.data,
            order=order,
            order_name=order_name,
            norm=norm,
            minus_log=minus_log,
        )

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(
        self,
        alpha,
        order=None,
        order_name="order",
        cumsum=False,
        minus_log=None,
        alpha_name=None,
    ):
        if order is None:
            order = self.order

        if alpha_name is None:
            alpha_name = self.alpha_name

        xcoefs = self.xcoefs(
            order=order, order_name=order_name, norm=True, minus_log=minus_log
        )

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        dalpha = alpha - self.alpha0
        p = xr.DataArray(np.arange(order + 1), dims=order_name)
        prefac = dalpha ** p

        # TODO : this should be an option, same for xcoefs
        coords = {"dalpha": dalpha, alpha_name + "0": self.alpha0}

        out = (prefac * xcoefs.sel(**{order_name: prefac[order_name]})).assign_coords(
            **coords
        )

        if cumsum:
            out = out.cumsum(order_name)
        else:
            out = out.sum(order_name)

        return out

    def resample(self, indices=None, nrep=None, **kws):
        return self.__class__(
            order=self.order,
            alpha0=self.alpha0,
            coefs=self.coefs,
            data=self.data.resample(nrep=nrep, indices=indices, **kws),
            minus_log=self.minus_log,
            alpha_name=self.alpha_name,
        )


class StateCollection(object):
    def __init__(self, states):  # , **kws):
        """
        Parameters
        ----------
        states : list
            list of states to consider
            Note that some subclasses require this list to be sorted
        """

        self.states = states
        # self.kws = kws

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

    @property
    def alpha_name(self):
        try:
            alpha_name = self[0].alpha_name
        except Exception:
            alpha_name = "alpha"
        return alpha_name

    def resample(self, indices=None, nrep=None, **kws):
        """
        resample things
        """
        if indices is None:
            indices = [None] * len(self)

        assert len(indices) == len(self)

        if "freq" in kws:
            freq = kws.pop("freq")
            if freq is None:
                freq = [None] * len(self)
            assert len(freq) == len(self)

            return type(self)(
                states=tuple(
                    state.resample(indices=idx, nrep=nrep, freq=fq, **kws)
                    for state, idx, fq in zip(self.states, indices, freq)
                )
            )

        else:
            return self.__class__(
                states=tuple(
                    state.resample(indices=idx, nrep=nrep, **kws)
                    for state, idx in zip(self.states, indices)
                )
            )

    @gcached()
    def order(self):
        return min([m.order for m in self])

    @gcached()
    def alpha0(self):
        return [m.alpha0 for m in self]


def xr_weights_minkowski(deltas, m=20, dim="state"):
    deltas_m = deltas ** m
    return 1.0 - deltas_m / deltas_m.sum(dim)


class ExtrapWeightedModel(StateCollection):
    def _states_between_alpha(self, alpha):
        idx = np.digitize(alpha, self.alpha0, right=False) - 1
        if idx < 0:
            idx = 0
        elif idx == len(self) - 1:
            idx = len(self) - 2

        return self.states[idx : idx + 2]

    def _states_nearest_alpha(self, alpha):
        dalpha = np.abs(np.array(self.alpha0) - alpha)
        # two lowest
        idx = np.argsort(dalpha)[:2]
        return [self[i] for i in idx]

    def _states_alpha(self, alpha, method):
        if method is None or method == "between":
            return self._states_between_alpha(alpha)
        elif method == "nearest":
            return self._states_nearest_alpha(alpha)
        else:
            raise ValueError("unknown method {}".format(method))

    def predict(
        self,
        alpha,
        order=None,
        order_name="order",
        cumsum=False,
        minus_log=None,
        alpha_name=None,
        method=None,
    ):
        """
        Parameters
        ----------
        method : {None, 'between', 'nearest'}
            method to select which models are chosen to predict value for given
            value of alpha.
            * None or between: use states such that `state[i].alpha0 <= alpha < states[i+1]`
              if alpha < state[0].alpha0 use first two states
              if alpha > states[-1].alpha0 use last two states
            * nearest: use two states with minimum `abs(state[k].alpha0 - alpha)`

        Notes
        -----
        This requires that self.states are ordered in ascending alpha0 order
        """

        if alpha_name is None:
            alpha_name = self.alpha_name

        if order is None:
            order = self.order
        if alpha_name is None:
            alpha_name = self.alpha_name

        if len(self) == 2:
            states = self.states

        else:
            # multiple states
            if np.array(alpha).ndim > 0:
                # have multiple alphas
                # recursively call
                return xr.concat(
                    (
                        self.predict(
                            alpha=a,
                            order=order,
                            order_name=order_name,
                            cumsum=cumsum,
                            minus_log=minus_log,
                            alpha_name=alpha_name,
                            method=method,
                        )
                        for a in alpha
                    ),
                    dim=alpha_name,
                )

            states = self._states_alpha(alpha, method)

        out = xr.concat(
            [
                m.predict(
                    alpha,
                    order=order,
                    order_name=order_name,
                    cumsum=cumsum,
                    minus_log=minus_log,
                    alpha_name=alpha_name,
                )
                for m in states
            ],
            dim="state",
        )

        w = xr_weights_minkowski(np.abs(out.dalpha))
        out = (out * w).sum("state") / w.sum("state")
        return out


class InterpModel(StateCollection):
    @gcached(prop=False)
    def xcoefs(self, order=None, order_name="porder", minus_log=None):

        if order is None:
            order = self.order

        porder = len(self) * (order + 1) - 1

        # keep track of these to reconstruct index later
        states = []
        orders = []

        # construct mat[porder, porder]
        # by stacking
        mat = []
        power = np.arange(porder + 1)
        num = sp_factorial(np.arange(porder + 1))

        for istate, m in enumerate(self.states):
            alpha = m.alpha0
            for j in range(order + 1):
                with np.errstate(divide="ignore"):
                    val = (
                        (alpha ** (power - j))
                        * num
                        / sp_factorial(np.arange(porder + 1) - j)
                    )
                mat.append(val)
                states.append(istate)
                orders.append(j)

        mat = np.array(mat)
        mat[np.isinf(mat)] = 0.0

        mat_inv = np.linalg.inv(mat)
        mat_inv = (
            xr.DataArray(mat_inv, dims=[order_name, "state_order"])
            .assign_coords(state=("state_order", states))
            .assign_coords(order=("state_order", orders))
            .set_index(state_order=["state", "order"])
            .unstack()
        )

        coefs = xr.concat(
            [m.xcoefs(order, norm=False, minus_log=minus_log) for m in self.states],
            dim="state",
        )
        if isinstance(coefs, xr.Dataset):
            coefs = xr.Dataset({k: xr.dot(mat_inv, v) for k, v in coefs.items()})
        else:
            coefs = xr.dot(mat_inv, coefs)

        return coefs

    def predict(
        self, alpha, order=None, order_name="porder", minus_log=None, alpha_name=None
    ):

        if order is None:
            order = self.order
        if alpha_name is None:
            alpha_name = self.alpha_name

        xcoefs = self.xcoefs(order=order, order_name=order_name, minus_log=minus_log)
        alpha = xrwrap_alpha(alpha, name=alpha_name)

        porder = len(xcoefs[order_name]) - 1

        p = xr.DataArray(np.arange(porder + 1), dims=order_name)
        prefac = alpha ** p

        out = (prefac * xcoefs).sum(order_name)
        return out


class PerturbModel(object):
    def __init__(self, alpha0, data, alpha_name="alpha"):

        self.alpha0 = alpha0
        self.data = data

        if alpha_name is None:
            alpha_name = "alpha"
        self.alpha_name = alpha_name

    def predict(self, alpha, alpha_name=None):

        if alpha_name is None:
            alpha_name = self.alpha_name

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        uv = self.data.uv
        xv = self.data.xv

        alpha0 = self.alpha0

        rec = self.data.rec
        dalpha = alpha - alpha0

        dalpha_uv = (-1.0) * dalpha * uv
        dalpha_uv_diff = dalpha_uv - dalpha_uv.max(rec)
        expvals = np.exp(dalpha_uv_diff)

        num = xr.dot(expvals, xv, dims="rec") / len(xv[rec])
        den = expvals.mean("rec")

        return num / den

    def resample(self, nrep, idx=None, **kws):
        return self.__class__(
            alpha0=self.alpha0,
            data=self.data.resample(nrep=nrep, idx=idx, **kws),
            alpha_name=self.alpha_name,
        )


class MBARModel(StateCollection):
    """
    Sadly, this doesn't work as beautifully.
    """

    def __init__(self, states):
        if not _HAS_PYMBAR:
            raise ImportError("need pymbar to use this")
        super(MBARModel, self).__init__(states)

    @gcached(prop=False)
    def _default_params(self, state_name="state", alpha_name="alpha"):

        # all xvalues:
        xv = xr.concat([m.data.xv for m in self], dim=state_name)
        uv = xr.concat([m.data.uv for m in self], dim=state_name)
        alpha0 = xrwrap_alpha([m.alpha0 for m in self], name=alpha_name)

        # make sure uv, xv in correct orde
        rec = self[0].data.rec
        xv = xv.transpose(state_name, rec, ...)
        uv = uv.transpose(state_name, rec, ...)

        # alpha[alpha] * uv[state, rec] = out[alpha, state, rec]
        Ukn = (alpha0 * uv).values.reshape(len(self), -1)
        N = np.ones(len(self)) * len(xv["rec"])
        mbar_obj = mbar.MBAR(Ukn, N)

        return uv, xv, alpha0, mbar_obj

    def predict(self, alpha, alpha_name=None):
        if alpha_name is None:
            alpha_name = self.alpha_name

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        if alpha.ndim == 0:
            alpha = alpha.expand_dims(alpha.name)

        uv, xv, alpha0, mbar_obj = self._default_params("state", alpha.name)

        dims = xv.dims
        x = np.array(xv, order="c")
        x_flat = x.reshape(x.shape[0] * x.shape[1], -1)

        U = uv.values.reshape(-1)

        out = []
        for b in alpha.values:
            out.append(mbar_obj.computeMultipleExpectations(x_flat.T, b * U)[0])

        out = np.array(out)
        # reshape
        shape = (out.shape[0],) + x.shape[2:]
        out = xr.DataArray(
            out.reshape(shape), dims=(alpha.name,) + dims[2:]
        ).assign_coords(alpha=alpha)

        return out

    def resample(self, *args, **kwargs):
        raise NotImplementedError("resample not implemented for this class")
