"""
routines to define a cached class without needing to subclass Cached class
"""
from __future__ import absolute_import

from functools import wraps
from inspect import signature

__all__ = ["cached", "cached_func", "cached_clear", "gcached"]


def gcached(key=None, prop=True):
    def wrapper(func):
        if prop:
            wrapped = property(cached(key)(func))
        else:
            wrapped = cached_func(key)(func)
        return wrapped

    return wrapper


def cached(key=None):
    """Decorator to cache a property within a class

    Requires the Class to have a cache dict called ``_cache``.

    Notes
    -----
    Usage::

        class A(object):
           def__init__(self):
               # this isn't strictly needed as it will be created on demand
               self._cache = dict()

           @property
           @cached('keyname')
           def size(self):
               # This code gets ran only if the lookup of keyname fails
               # After this code has been ran once, the result is stored in
               # _cache with the key: 'keyname'
               size = 10.0

            #no aguments implies give cache function name
            @property
            @cached()
            def myprop(self):
                #results in _cache['myprop']

    See Also
    --------
    cached_clear : corresponding decorator to clear cache
    cached_func : decorator for cache creation of function
    """

    def cached_lookup(func):
        if key is None:
            _key = func.__name__
        else:
            _key = key
        # if len(args) == 0:
        #     key = func.__name__
        # elif len(args) == 1:
        #     key = args[0]
        # else:
        #     raise ValueError('key must be single valued or None')

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return self._cache[_key]
            except AttributeError:
                self._cache = dict()
            except KeyError:
                pass

            self._cache[_key] = ret = func(self, *args, **kwargs)
            return ret

        return wrapper

    return cached_lookup


def cached_func(key=None):
    """Decorator to cache a function within a class

    Requires the Class to have a cache dict called ``_cache``.

    Notes
    -----
    Usage::

        class A(object):
           def __init__(self):
              pass

           @cached_func('keyname')
           def amethod(self, *args):
               # This code gets ran only if the lookup of keyname fails
               # After this code has been ran once, the result is stored in
               # _cache with the key: 'keyname'
               # a long calculation...
               return long_calc(self,val)
               # if already executed result in _cache[('keyname',) + args]

            #no aguments implies give cache function name
            @property
            @cached()
            def myprop(self):
                #results in _cache['myprop']


    See also
    --------
    cached_clear : corresponding decorator to remove cache

    cached : decorator for properties

    """

    def cached_lookup(func):
        if key is None:
            _key = func.__name__
        else:
            _key = key

        # use signature
        bind = signature(func).bind

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # key_func = (_key, args, frozenset(kwargs.items()))
            # params = sig.bind
            params = bind(self, *args, **kwargs)
            params.apply_defaults()
            key_func = (_key, params.args[1:], frozenset(params.kwargs.items()))
            try:
                return self._cache[key_func]
            except TypeError:
                # this means that key_func is bad hash
                return func(self, *args, **kwargs)
            except AttributeError:
                self._cache = dict()
            except KeyError:
                pass

            self._cache[key_func] = ret = func(self, *args, **kwargs)
            return ret

        return wrapper

    return cached_lookup


def cached_clear(*keys):
    """
    Decorator to clear self._cache of specified properties

    Parameters
    ----------
    *keys : arguments
        remove these keys from cache.  if len(keys)==0, remove all keys.


    Examples
    --------
    Usage::

        class tmp(object)
            ...

            @property
            @cached()
            def a(self):
                ...

            @x.setter
            @cached_clear('a','b')
            def x(self,val):
                #....
                #deletes self._cache['a'],self,_cache['b']
                #if args are empty, remove all


    See Also
    --------
    cached : corresponding decorator for cache creation of property
    cached_func : decorator for cache creation of function
    """

    def cached_clear(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # self._clear_caches(*keys)
            # clear out keys
            if len(keys) == 0 or not hasattr(self, "_cache"):
                self._cache = dict()
            else:
                for name in keys:
                    try:
                        del self._cache[name]
                    except KeyError:
                        pass

                # functions
                keys_tuples = [
                    k for k in self._cache if isinstance(k, tuple) and k[0] in keys
                ]
                for name in keys_tuples:
                    del self._cache[name]

            return func(self, *args, **kwargs)

        return wrapper

    return cached_clear
