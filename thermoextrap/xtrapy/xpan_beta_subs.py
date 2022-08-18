from __future__ import absolute_import

from functools import lru_cache

# import numpy as np
import sympy as sp

from .cached_decorators import gcached
from .models import (
    Derivs,
    SymSubs,
    _get_default_function,
    _get_default_indexed,
    _get_default_symbol,
)

# from scipy.special import factorial as sp_factorial


###############################################################################
# Central moment classes:
###############################################################################


class _Central_u_dxdu(object):
    """
    u = _Central_u_dxdu()

    u[i] = u({du}, {dxdu})
    """

    def __init__(self, use_u1=False, **kwargs):
        """
        Parameters
        kwargs : dict
            optional values for u, du, u1

        use_u1 : bool, default=False
            if True, substitue u[1] = u1
        """
        self.k = _get_default_symbol("k")
        for key in ["u", "du"]:
            if key in kwargs:
                val = kwargs[key]
            else:
                val = _get_default_indexed(key)
            setattr(self, key, val)

        if use_u1:
            key = "u1"
            if key in kwargs:
                val = kwargs[key]
            else:
                val = _get_default_symbol(key)
            setattr(self, key, val)

        else:
            self.u1 = self.u[1]

    @gcached(prop=False)
    def _get_ubar_of_dubar(self, n):
        expr = (
            sp.Sum(
                sp.binomial(n, self.k) * self.du[self.k] * self.u1 ** (n - self.k),
                (self.k, 0, n),
            )
            .doit()
            .subs({self.du[0]: 1, self.du[1]: 0})
            # .simplify()
            .expand()
        )
        return expr

    def __getitem__(self, n):
        return self._get_ubar_of_dubar(n)


@lru_cache(5)
def factory_central_u_dxdu(use_u1=False, **kwargs):
    return _Central_u_dxdu(use_u1=use_u1, **kwargs)


class _Central_xu_dxdu(object):
    """
    xu = _Central_xu_dxdu()

    xu[i] = xu({x1}, {du}, {dxdu})
    """

    def __init__(self, use_u1=False, use_x1=True, **kwargs):
        """
        Parameters
        kwargs : dict
            optional values for u, x, du, dxdu, u1, x1
        """
        self.u = factory_central_u_dxdu(use_u1=use_u1, **kwargs)

        # pointer to self.u attributes
        for attr in ["u1", "du", "k"]:
            setattr(self, attr, getattr(self.u, attr))

        for key in ["x", "dxdu"]:
            if key in kwargs:
                val = kwargs[key]
            else:
                val = _get_default_indexed(key)
            setattr(self, key, val)

        if use_x1:
            key = "x1"
            if key in kwargs:
                val = kwargs[key]
            else:
                val = _get_default_symbol(key)
            setattr(self, key, val)

        else:
            self.x1 = self.x[1]

    @gcached(prop=False)
    def _get_xubar_of_dxdubar(self, n):
        expr = (
            sp.Sum(
                sp.binomial(n, self.k) * self.u1 ** (n - self.k) * self.dxdu[self.k],
                (self.k, 0, n),
            )
            + self.x1 * self.u[n]
        )
        return (
            expr.doit()
            .subs({self.dxdu[0]: 0})
            .expand()
            # .simplify()
        )

    def __getitem__(self, n):
        return self._get_xubar_of_dxdubar(n)


class _Central_xu_dxdu_xalpha(object):
    """
    xu = _Central_xu_dxdu()

    xu[i, n] = < d^n x / dalpha^n u**i>
    """

    def __init__(self, use_u1=False, use_x1=True, **kwargs):
        """
        Parameters
        kwargs : dict
            optional values for u, x, du, dxdu, u1, x1
        """
        self.u = factory_central_u_dxdu(use_u1=use_u1, **kwargs)

        # pointer to self.u attributes
        for attr in ["u1", "du", "k"]:
            setattr(self, attr, getattr(self.u, attr))

        for key in ["x", "dxdu"]:
            if key in kwargs:
                val = kwargs[key]
            else:
                val = _get_default_indexed(key)
            setattr(self, key, val)

        if use_x1:
            key = "x1"
            if key in kwargs:
                val = kwargs[key]
            else:
                val = _get_default_indexed(key)

            setattr(self, key, val)

            # NOTE: because could be using x1[deriv] or x[1, deriv]
            # use a function to wrap this behaviour.
            self.x1_func = lambda deriv: self.x1[deriv]
        else:
            self.x1_func = lambda deriv: self.x[1, deriv]

    x1 = _get_default_indexed("x1")

    @gcached(prop=False)
    def _get_xubar_of_dxdubar(self, n, deriv):
        expr = (
            sp.Sum(
                sp.binomial(n, self.k)
                * self.u1 ** (n - self.k)
                * self.dxdu[self.k, deriv],
                (self.k, 0, n),
            )
            + self.x1_func(deriv) * self.u[n]
        )
        return (
            expr.doit()
            .subs({self.dxdu[0, deriv]: 0})
            .expand()
            # .simplify()
        )

    def __getitem__(self, idx):
        return self._get_xubar_of_dxdubar(*idx)


@lru_cache(5)
def factory_central_xu_dxdu(xalpha=False, use_u1=False, use_x1=True, **kwargs):
    if xalpha:
        cls = _Central_xu_dxdu_xalpha
    else:
        cls = _Central_xu_dxdu

    return cls(use_u1=use_u1, use_x1=use_x1, **kwargs)


def factory_central_u_xu(xalpha=False, use_u1=False, use_x1=True, **kwargs):
    u = factory_central_u_dxdu(use_u1=use_u1, **kwargs)
    xu = factory_central_xu_dxdu(xalpha=xalpha, use_u1=use_u1, use_x1=use_x1, **kwargs)
    return u, xu


class _SymDerivBeta(object):
    """
    Analytic derivative of canonical partition function wrt beta

    Q = f / z
    """

    b = _get_default_symbol("b")
    f, z = _get_default_function("f", "z")
    Q = f(b) / z(b)

    # def __init__(self, **kwargs):
    #     f, z = _get_default_function('f', 'z')
    #     self.b = _get_default_symbol('b')
    #     self.f = f(self.b)
    #     self.z = z(self.b)
    #     self.Q = self.f / self.z

    @gcached(prop=False)
    def __getitem__(self, order):
        # recusive get derivative

        if order == 0:
            return self.Q
        else:
            return self[order - 1].diff(self.b, 1)


class _SubsBeta(object):
    """
    d = _SymDerivBeta()
    s = _SubsBeta()
    d[k].subs(s[k]) -> derivative in terms of u[i], xu[i]

    As constructed, u, xu can be symbols or something else (to sub in values)
    """

    b = _get_default_symbol("b")
    f, z = _get_default_function("f", "z")
    f = f(b)
    z = z(b)

    def __init__(self, u=None, xu=None):

        f, z = _get_default_function("f", "z")
        self.b = _get_default_symbol("b")
        self.f = f(self.b)
        self.z = z(self.b)
        for key, val in zip(["u", "xu"], [u, xu]):
            if val is None:
                val = _get_default_indexed(key)
            setattr(self, key, val)

        self._init_data()

    @classmethod
    def from_central_moments(cls, **kwargs):
        u, xu = factory_central_u_xu(xalpha=False, **kwargs)
        return cls(u=u, xu=xu)

    def _init_data(self):
        self._data = [[(self.f, self.xu[0] * self.z)]]

    @property
    def order(self):
        return len(self._data) - 1

    def _add_order(self):
        order = self.order + 1
        new = []
        # f derivative:
        lhs = self.f.diff(self.b, order)
        rhs = (-1) ** order * self.xu[order] * self.z
        new.append((lhs, rhs))

        # z deriative:
        lhs = self.z.diff(self.b, order)
        rhs = (-1) ** order * self.u[order] * self.z
        new.append((lhs, rhs))
        self._data.append(new)

    #    @gcached(prop=False)
    def __getitem__(self, order):
        assert order >= 0
        while order > self.order:
            self._add_order()
        return sum(self._data[: order + 1], [])[-1::-1]


class _SubsBeta_xalpha(_SubsBeta):
    """
    substitutions with beta dependencie
    """

    k = _get_default_symbol("k")

    def _init_data(self):
        self._data = [[(self.f, self.xu[0, 0] * self.z)]]

    @classmethod
    def from_central_moments(cls, **kwargs):
        u, xu = factory_central_u_xu(xalpha=True, **kwargs)
        return cls(u=u, xu=xu)

    def _add_order(self):
        order = self.order + 1

        new = []

        # f deriv:
        lhs = self.f.diff(self.b, order)

        # Note: sp.Sum doesn't work
        # right with user defined u/xu
        rhs = 0
        for j in range(order + 1):
            rhs += (-1) ** j * sp.binomial(order, j) * self.xu[j, order - j]
        rhs *= self.z

        # NOTE: This doesn't work for non-sympy xu
        # rhs = (
        #     sp.Sum(
        #         (
        #             (-1) ** self.k
        #             * sp.binomial(order, self.k)
        #             * self.xu[order - self.k, self.k]
        #         ),
        #         (self.k, 0, order),
        #     ).doit()
        #     * self.z
        # )

        new.append((lhs, rhs))

        # z deriv:
        lhs = self.z.diff(self.b, order)
        rhs = (-1) ** order * self.u[order] * self.z
        new.append((lhs, rhs))
        self._data.append(new)


@lru_cache(5)
def factory_derivs(xalpha=False, central=False):
    derivs = _SymDerivBeta()

    if xalpha:
        cls = _SubsBeta_xalpha
    else:
        cls = _SubsBeta

    if central:
        subs = cls.from_central_moments()
        args = (subs.xu.x1, subs.xu.du, subs.xu.dxdu)
    else:
        subs = cls()
        args = (subs.u, subs.xu)

    exprs = SymSubs(derivs, subs, recursive=False, simplify=False, expand=True)

    return Derivs.from_sympy(exprs, args=args)
