from __future__ import absolute_import

from functools import lru_cache

import numpy as np
import sympy as sp
import xarray as xr

from scipy.special import factorial as sp_factorial

from .cached_decorators import gcached

try:
    from pymbar import mbar

    _HAS_PYMBAR = True
except ImportError:
    _HAS_PYMBAR = False

# NOTE: General scheme:
# uv, xv -> samples (values) for u, x
# u, xu -> averages of u and x*u
# u[i] = <u**i>
# xu[i] = <x * u**i>
# xu[i, j] = <d^i x/d beta^i * u**j>


###############################################################################
# Structure(s) to handle data
###############################################################################
def _check_xr(x, dims, strict=True, name=None):
    if isinstance(x, xr.Dataset):
        # don't do anything to datasets
        pass

    elif not isinstance(x, xr.DataArray):
        x = np.array(x)
        if isinstance(dims, dict):
            dims = dims[x.ndim]
        x = xr.DataArray(x, dims=dims, name=name)
    elif strict:
        if isinstance(dims, dict):
            dims = dims[x.ndim]
        for d in dims:
            if d not in x.dims:
                raise ValueError("{} not in dims".format(d))
    return x


def xrwrap_uv(uv, dims=None, rec="rec", rep="rep", name="u", stict=True):
    """
    wrap uv (energy values) array

    assumes uv[rec], or uv[rep, rec] where rec is recored (or time) and rep is replicate
    """
    if dims is None:
        dims = {1: [rec], 2: [rep, rec]}
    return _check_xr(uv, dims, strict=stict, name=name)


def xrwrap_xv(
    xv,
    dims=None,
    rec="rec",
    rep="rep",
    deriv="deriv",
    val="val",
    xalpha=False,
    name="x",
    strict=None,
):
    """
    wraps xv (x values) array

    if xalpha is False, assumes xv[rec], xv[rec, val], xv[rep, rec, val]
    if xalpha is True, assumes xv[rec, deriv], xv[rec,deriv, val], xv[rep,rec,deriv,val]
    """
    if not xalpha:
        if strict is None:
            strict = False
        if dims is None:
            dims = {1: [rec], 2: [rec, val], 3: [rep, rec, val]}

    else:
        if strict is None:
            strict = False
        if dims is None:
            dims = {2: [rec, deriv], 3: [rec, deriv, val], 4: [rep, rec, deriv, val]}
    return _check_xr(xv, dims=dims, strict=strict, name=name)


def xrwrap_alpha(alpha, dims=None, stict=False, name="alpha"):
    """
    wrap alpha values
    """
    if isinstance(alpha, xr.DataArray):
        pass
    else:
        alpha = np.array(alpha)
        if dims is None:
            dims = name

        if alpha.ndim == 0:
            alpha = xr.DataArray(alpha, coords={dims: alpha}, name=name)
        elif alpha.ndim == 1:
            alpha = xr.DataArray(alpha, dims=dims, coords={dims: alpha}, name=name)
        else:
            alpha = xr.DataArray(alpha, dims=dims, name=name)
    return alpha


def resample_indicies(size, nrep, rec="rec", rep="rep"):
    """
    get indexing DataArray
    """
    return (
        xr.DataArray(
            [np.random.choice(size, size=size, replace=True) for _ in range(nrep)],
            dims=[rep, rec],
        )
        # things are faster with
        .transpose("rec", ...)
    )


class DatasetSelector(object):
    """
    wrap dataset so can index like ds[i, j]

    Needed for calling sympy.lambdify functions
    """

    def __init__(self, data, dims=None, moment="moment", deriv="deriv"):

        # Default dims
        if dims is None:
            if deriv in data.dims:
                dims = [moment, deriv]
            else:
                dims = [moment]

        if isinstance(dims, str):
            dims = [dims]

        self.data = data
        self.dims = dims

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        assert len(idx) == len(self.dims)
        selector = dict(zip(self.dims, idx))
        return self.data.isel(**selector)


class DataBase(object):
    def __init__(
        self,
        uv,
        xv,
        order,
        skipna=False,
        xalpha=False,
        rec="rec",
        moment="moment",
        val="val",
        rep="rep",
        deriv="deriv",
        chunk=None,
        compute=None,
        **kws
    ):

        uv = xrwrap_uv(uv, rec=rec, rep=rep)
        xv = xrwrap_xv(xv, rec=rec, rep=rep, deriv=deriv, val=val, xalpha=xalpha)

        if chunk is not None:
            if isinstance(chunk, int):
                chunk = {rec: chunk}

            uv = uv.chunk(chunk)
            xv = xv.chunk(chunk)

        if compute is None:
            # default compute
            # if not chunk, default compute is False
            # if chunk, default compute is True
            if chunk is None:
                compute = False
            else:
                compute = True

        self.uv = uv
        self.xv = xv
        self.chunk = chunk
        self.compute = compute

        self.order = order
        self.skipna = skipna
        self.xalpha = xalpha

        self._rec = rec
        self._rep = rep
        self._val = val
        self._moment = moment
        self._deriv = deriv
        self._kws = kws

    def __len__(self):
        return len(self.uv["rec"])

    def resample(self, nrep, idx=None, chunk=None, compute="None"):

        if chunk is None:
            chunk = self.chunk

        if compute is "None":
            compute = None
        elif compute is None:
            compute = self.compute

        shape = len(self.uv[self._rec])

        if idx is None:
            idx = resample_indicies(shape, nrep, rec=self._rec, rep=self._rep)

        uv = self.uv.compute()[idx]
        # allow for Dataset
        xv = self.xv.compute().isel(**{self._rec: idx})

        return self.__class__(
            uv=uv,
            xv=xv,
            order=self.order,
            rec=self._rec,
            rep=self._rep,
            val=self._val,
            moment=self._moment,
            deriv=self._deriv,
            xalpha=self.xalpha,
            skipna=self.skipna,
            chunk=chunk,
            compute=compute,
            **self._kws
        )


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


@lru_cache(100)
def _get_default_function(*args):
    out = [sp.Function(key) for key in args]
    if len(out) == 1:
        out = out[0]
    return out


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
        args = (u, xu)
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
        out = self.coefs(
            *data._xcoefs_args, order=order, norm=norm, minus_log=minus_log
        )
        return xr.concat(out, dim=order_name)

    @classmethod
    def from_sympy(cls, exprs, args):
        funcs = Lambdify(exprs, args=args)
        return cls(funcs=funcs, exprs=exprs)


class ExtrapModel(object):
    """
    apply taylor series extrapolation
    """
    def __init__(self, alpha0, data, coefs, order=None, minus_log=False, alpha_name='alpha'):
        self.alpha0 = alpha0
        self.data = data
        self.coefs = coefs

        if order is None:
            order = self.order
        if minus_log is None:
            minus_log = False

        self.minus_log = minus_log
        self.order = order

        if alpha_name is None:
            alpha_name = 'alpha'
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
            self, alpha, order=None, order_name="order", cumsum=False, minus_log=None,
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

        coords = {'dalpha': dalpha, alpha_name + '0': self.alpha0}

        out = (
            (prefac * xcoefs.sel(**{order_name: prefac[order_name]}))
            .assign_coords(**coords)
        )

        if cumsum:
            out = out.cumsum(order_name)
        else:
            out = out.sum(order_name)

        return out

    def resample(self, nrep, idx=None, **kws):
        return self.__class__(
            order=self.order,
            alpha0=self.alpha0,
            coefs=self.coefs,
            data=self.data.resample(nrep=nrep, idx=idx, **kws),
            minus_log=self.minus_log,
            alpha_name=self.alpha_name,
        )

    # @classmethod
    # def from_values_beta(
    #     cls, order, alpha0, uv, xv, xalpha=False, central=False, minus_log=False, **kws
    # ):
    #     """
    #     build a model from beta extraploation from data
    #     """

    #     data = factory_data(
    #         uv=uv, xv=xv, order=order, xalpha=xalpha, central=central, **kws
    #     )

    #     coefs = factory_coefs_beta(xalpha=xalpha, central=central)

    #     return cls(
    #         order=order, alpha0=alpha0, coefs=coefs, data=data, minus_log=minus_log
    #     )


class StateCollection(object):
    def __init__(self, states):
        self.states = states

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
        except:
            alpha_name = 'alpha'
        return alpha_name

    def resample(self, nrep, idxs=None, **kws):
        if idxs is None:
            idxs = [None] * len(self)
        assert len(idxs) == len(self)

        return self.__class__(
            states=tuple(
                state.resample(nrep=nrep, idx=idx, **kws)
                for state, idx in zip(self.states, idxs)
            )
        )

    @gcached()
    def order(self):
        return min([m.order for m in self])


def xr_weights_minkowski(deltas, m=20, dim="state"):
    deltas_m = deltas ** m
    return 1.0 - deltas_m / deltas_m.sum(dim)


class ExtrapWeightedModel(StateCollection):
    def predict(
            self, alpha, order=None, order_name="order", cumsum=False, minus_log=None,
            alpha_name=None,
    ):

        if order is None:
            order = self.order
        if alpha_name is None:
            alpha_name = self.alpha_name

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
                for m in self.states
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

    def predict(self, alpha, order=None, order_name="porder", minus_log=None, alpha_name=None):

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
    def __init__(self, alpha0, data, alpha_name='alpha'):

        self.alpha0 = alpha0
        self.data = data

        if alpha_name is None:
            alpha_name = 'alpha'
        self.alpha_name = alpha_name

    def predict(self, alpha, alpha_name=None):

        if alpha_name is None:
            alpha_name = self.alpha_name

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        uv = self.data.uv
        xv = self.data.xv

        alpha0 = self.alpha0

        rec = self.data._rec
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

        # make sure uv, xv in correct order
        rec = self[0].data._rec
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
