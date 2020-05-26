from __future__ import absolute_import

from functools import lru_cache

import numpy as np
import sympy as sp


from .cached_decorators import gcached
from .core import _get_default_symbol, _get_default_indexed, _get_default_function

from .core import SymSubs, Coefs


###############################################################################
# Central moments
###############################################################################

class du_func(sp.Function):
    """
    du_func(beta, n) = <(u - <u>)**n>

    d du_func / d(beta) = -(du_func(beta, n+1) - n * du_func(beta, n-1) * du_func(beta, 2))
    """

    nargs = 2

    du = _get_default_indexed("du")

    def fdiff(self, argindex=1):
        beta, n = self.args
        out = du_func(beta, n + 1) - n * du_func(beta, n - 1) * du_func(beta, 2)

        # (-beta) -> beta
        out = -1 * out
        return out

    @classmethod
    def eval(cls, beta, n):
        if n == 0:
            return 1
        elif n == 1:
            return 0
        elif beta is None:
            return cls.du[n]
        else:
            return


class dxdu_func(sp.Function):
    """

    dxdu_func(beta, n, d) = <du**n * (x^(d) - <x^(d)>)> = dxdu[d, n]

    or (if x != x(alpha))

    dxdu_func(beta, n) = <du**n * (x - <x>)>

    """

    nargs = (2, 3)
    dxdu = _get_default_indexed("dxdu")

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            beta, n = self.args

            out = (
                -dxdu_func(beta, n + 1)
                + n * dxdu_func(beta, n - 1) * du_func(beta, 2)
                + dxdu_func(beta, 1) * du_func(beta, n)
            )
            return out

        else:
            # xalpha
            beta, n, d = self.args
            out = (
                -dxdu_func(beta, n + 1, d)
                + n * dxdu_func(beta, n - 1, d) * du_func(beta, 2)
                + dxdu_func(beta, n, d + 1)
                + dxdu_func(beta, 1, d) * du_func(beta, n)
            )

            return out

    @classmethod
    def eval(cls, beta, n, deriv=None):
        if n == 0:
            return 0

        elif beta is None:
            if deriv is None:
                return cls.dxdu[n]
            else:
                return cls.dxdu[deriv, n]
        else:
            return


class x_central_func(sp.Function):
    nargs = (1, 2)

    x1_indexed = _get_default_indexed("x1")
    x1_symbol = _get_default_symbol("x1")

    def fdiff(self, argindex=1):

        if len(self.args) == 1:
            (beta,) = self.args
            return -dxdu_func(beta, 1)

        else:
            # xalpha
            beta, d = self.args
            out = -dxdu_func(beta, 1, d) + x_central_func(beta, d + 1)
            return out

    @classmethod
    def eval(cls, beta, deriv=None):

        if beta is None:
            if deriv is None:
                return cls.x1_symbol
            else:
                return cls.x1_indexed[deriv]
        else:
            return


###############################################################################
# raw moments
###############################################################################

class u_func(sp.Function):
    """
    calculate derivatives of <U**n>
    """

    nargs = 2
    u = _get_default_indexed("u")

    def fdiff(self, argindex=1):
        beta, n = self.args
        return -(u_func(beta, n + 1) - u_func(beta, n) * u_func(beta, 1))

    @classmethod
    def eval(cls, beta, n):
        if n == 0:
            return 1

        elif beta is None:
            return cls.u[n]

        else:
            return

class xu_func(sp.Function):
    """
    calculate derivatives of

    xu_func(beta, n) = <x * u**n> = xu[n]

    or (xalpha)

    xu_func(beta, n, d) = <x^(d) * u**n> = xu[d, n]

    """

    nargs = (2, 3)

    xu = _get_default_indexed("xu")

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            beta, n = self.args
            return -xu_func(beta, n + 1) + xu_func(beta, n) * u_func(beta, 1)

        else:
            beta, n, d = self.args

            return (
                -xu_func(beta, n + 1, d)
                + xu_func(beta, n, d + 1)
                + xu_func(beta, n, d) * u_func(beta, 1)
            )

    @classmethod
    def eval(cls, beta, n, deriv=None):
        if beta is None:
            if deriv is None:
                return cls.xu[n]
            else:
                return cls.xu[deriv, n]
        else:
            return


class SymDerivBeta(object):
    """
    provide expressions for d^n <x> / d(beta)^n
    """

    beta = _get_default_symbol('beta')

    def __init__(self, xalpha=False, central=False, expand=True):
        if central:
            if xalpha:
                x = x_central_func(self.beta, 0)
                args = [x.x1_indexed] 
            else:
                x = x_central_func(self.beta)
                args = [x.x1_symbol]
            args += _get_default_indexed('du','dxdu')


        else:
            if xalpha:
                x = xu_func(self.beta, 0, 0)
            else:
                x = xu_func(self.beta, 0)
            args = _get_default_indexed('u','xu')

        self.xave = x
        self.args = args
        self.expand = expand


    @gcached(prop=False)
    def __getitem__(self, order):
        if order == 0:
            out = self.xave
        else:
            out = self[order-1].diff(self.beta, 1)
            if self.expand:
                out = out.expand()
        return out




@lru_cache(5)
def factory_coefs_beta(xalpha=False, central=False):
    derivs = SymDerivBeta(xalpha=xalpha, central=central)
    exprs = SymSubs(derivs, subs_all={derivs.beta: 'None'}, expand=False, simplify=False)
    return Coefs.from_sympy(exprs, args=derivs.args)





