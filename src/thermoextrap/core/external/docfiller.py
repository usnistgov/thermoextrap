"""Classes/routines to fill common documentation."""
from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from textwrap import dedent
from typing import Any, Callable, TypeVar

from ..cached_decorators import gcached
from .docscrape import NumpyDocString, Parameter

try:
    # Default is DOC_SUB is True
    DOC_SUB = os.getenv("DOCFILLER_SUB", "True").lower() not in (
        "0",
        "f",
        "false",
    )
except KeyError:
    DOC_SUB = True


# Taken directly from pandas.utils._decorators.doc
# See https://github.com/pandas-dev/pandas/blob/main/pandas/util/_decorators.py
# would just use the pandas version, but since it's a private
# module, feel it's better to have static version here.


# to maintain type information across generic functions and parametrization
T = TypeVar("T")
# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def doc(*docstrings: str | Callable, **params) -> Callable[[F], F]:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : str or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if hasattr(docstring, "_docstring_components"):
                # error: Item "str" of "Union[str, Callable[..., Any]]" has no attribute
                # "_docstring_components"
                # error: Item "function" of "Union[str, Callable[..., Any]]" has no
                # attribute "_docstring_components"
                docstring_components.extend(
                    docstring._docstring_components  # type: ignore[union-attr]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        # formatting templates and concatenating docstring
        decorated.__doc__ = "".join(
            [
                # change to format_map to avoid converting to a dict.
                # component.format(**params)
                component.format_map(params)
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in docstring_components
            ]
        )

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator


# Factory method to create docfiller
def docfiller(*templates, **params):
    """
    To fill common docs.

    Taken from pandas.utils._decorators
    """

    if DOC_SUB:
        return doc(*templates, **params)
    else:

        def decorated(func):
            return func

        return decorated


def _get_nested_values(d, join_string="\n"):
    out = []
    for k in d:
        v = d[k]
        if isinstance(v, str):
            out.append(v)
        else:
            out.extend(_get_nested_values(v, join_string=None))

    if join_string is not None:
        out = join_string.join(out)

    return out


class AttributeDict(Mapping):
    """
    Dictionary with recursive attribute like access.

    To be used in str.format calls, so can expand on fields like
    `{name.property}` in a nested manner.

    Parameters
    ----------
    entries : dict
    recursive : bool, default=True
        If True, recursively return ``AttributeDict`` for nested dicts.

    Example
    -------
    >>> d = AttributeDict({"a": 1, "b": {"c": 2}})
    >>> d.a
    1
    >>> d.b
    AttributeDict({'c': 2})
    >>> d.b.c
    2
    """

    __slots__ = ("_entries", "_recursive", "_allow_missing")

    def __init__(self, entries=None, recursive=True, allow_missing=True):
        """Init."""
        if entries is None:
            entries = {}
        self._entries = entries
        self._recursive = recursive
        self._allow_missing = allow_missing

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _get_nested_values(self._getslice(key), join_string="\n")

        if key == ":":
            return _get_nested_values(self._entries, join_string="\n")

        elif "," in key:
            return "\n".join(self[x] for x in (x.strip() for x in key.split(",")))

        if self._allow_missing and key not in self._entries:
            return f"{{{key}}}"
        else:
            return self._entries[key]

    def _getslice(self, s):
        start = s.start
        stop = s.stop

        keys = list(self._entries.keys())

        if isinstance(s.start, int) or s.start is None and isinstance(s.stop, int):
            slc = s

        else:
            if s.start is None:
                start = 0
            else:
                start = keys.index(s.start)

            if s.stop is None:
                stop = len(self) + 1
            else:
                stop = keys.index(s.stop) + 1

            slc = slice(start, stop, s.step)

        subset = {k: self._entries[k] for k in keys[slc]}
        return subset

    def __setitem__(self, key, value):
        self._entries[key] = value

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def _items(self):
        yield from self._entries.items()

    def __dir__(self):
        return list(self.keys()) + super().__dir__()

    def _update(self, *args, **kwargs):
        self._entries.update(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in self._entries:
            out = self._entries[attr]
            if self._recursive and isinstance(out, dict):
                out = type(self)(out)
            return out
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                # If Python is run with -OO, it will strip docstrings and our lookup
                # from self._entries will fail. We check for __debug__, which is actually
                # set to False by -O (it is True for normal execution).
                # But we only want to see an error when building the docs;
                # not something users should see, so this slight inconsistency is fine.
                if __debug__:
                    raise err
                else:
                    pass

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._entries)})"

    @classmethod
    def _from_dict(cls, params, max_level=1, level=0, recursive=True):
        # to hide level parameter
        out = cls(recursive=recursive)
        for k in params:
            v = params[k]
            if isinstance(v, dict) and level < max_level:
                v = cls._from_dict(
                    v, max_level=max_level, level=level + 1, recursive=recursive
                )
            out[k] = v
        return out

    @classmethod
    def from_dict(cls, params, max_level=1, recursive=True, level=0):
        """
        Create AttributeDict recursively for nested dictionaries.

        To be used in cases where need to apply AttibuteDict to parameters
        passed with ``func(**params)``.


        Parameters
        ----------
        params : mapping
            Mapping to apply cls to.
        max_level : int, default=1
            How deep to apply cls to.
        recursive : bool, default=True
            ``recursive`` parameter of resulting object.

        Returns
        -------
        AttributeDict

        Examples
        --------
        >>> d = {"a": "p0", "b": {"c": {"d": "d0"}}}
        >>> AttributeDict.from_dict(d, max_level=1)
        AttributeDict({'a': 'p0', 'b': AttributeDict({'c': {'d': 'd0'}})})

        >>> AttributeDict.from_dict(d, max_level=2)
        AttributeDict({'a': 'p0', 'b': AttributeDict({'c': AttributeDict({'d': 'd0'})})})

        """
        return cls._from_dict(
            params=params, max_level=max_level, level=0, recursive=recursive
        )


def _build_param_docstring(name, ptype, desc):
    """
    Create multiline documentation of single name, type, desc.

    Parameters
    ----------
    name : str
        Parameter Name
    ptype : str
        Parameter type
    desc : list of str
        Parameter description

    Returns
    -------
    output : string
        Multiline string for output.

    """

    no_name = name is None or name == ""
    no_type = ptype is None or ptype == ""

    if no_name and no_type:
        raise ValueError("must specify name or ptype")
    elif no_type:
        s = f"{name}"
    elif no_name:
        s = f"{ptype}"
    else:
        s = f"{name} : {ptype}"

    if isinstance(desc, str):
        if desc == "":
            desc = []
        else:
            desc = [desc]

    if len(desc) > 0:
        desc = "\n    ".join(desc)
        s += f"\n    {desc}"
    return s


def _params_to_string(params, key_char="|"):
    """
    Parse list of Parameters objects to string.

    Examples
    --------
    >>> from module_utilities.docscrape import Parameter
    >>> p = Parameter(name="a", type="int", desc=["A parameter"])
    >>> out = _parse_params_to_string(p)
    >>> print(out["a"])
    a : int
        A parameter

    """

    if len(params) == 0:
        return ""

    if isinstance(params, Parameter):
        params = [params]

    if not isinstance(params, (list, tuple)):
        params = [params]

    if isinstance(params[0], str):
        return "\n".join(params)

    out = {}
    for p in params:
        name = p.name
        if key_char in name:
            key, name = (x.strip() for x in name.split(key_char))
        else:
            key = name

        out[key] = _build_param_docstring(name=name, ptype=p.type, desc=p.desc)

    return out


def parse_docstring(func_or_doc, key_char="|", expand=True):
    """
    Parse numpy style docstring from function or string to dictionary.

    Parameters
    ----------
    func_or_doc : callable or str
        If function, extract docstring from function.
    key_char : str, default='|'
        Character to split key_name/name
    expand : bool, default=False

    Returns
    -------
    parameters : dict
        Dictionary with keys from docstring sections.

    Notes
    -----
    the keys of ``parameters`` have the are the those of the numpy docstring,
    but lowercase, and spaces replaced with underscores.  The sections parsed are:
    Summary, Extended Summary, Parameters, Returns, Yields, Notes,
    Warnings, Other Parameters, Attributes, Methods, References, and Examples.

    Examples
    --------
    >>> doc_string = '''
    ... Parameters
    ... ----------
    ... x : int
    ...     x parameter
    ... y_alt | y : float
    ...     y parameter
    ...
    ... Returns
    ... -------
    ... output : float
    ...     an output
    ... '''

    >>> p = parse_docstring(doc_string)
    >>> print(p["parameters"]["x"])
    x : int
        x parameter
    >>> print(p["parameters"]["y_alt"])
    y : float
        y parameter
    >>> print(p["returns"]["output"])
    output : float
        an output


    """

    if callable(func_or_doc):
        doc = inspect.getdoc(func_or_doc)
    else:
        doc = func_or_doc

    parsed = NumpyDocString(doc)._parsed_data

    if expand:
        parsed = {
            k.replace(" ", "_").lower(): _params_to_string(parsed[k], key_char=key_char)
            for k in [
                "Summary",
                "Extended Summary",
                "Parameters",
                "Returns",
                "Yields",
                "Notes",
                "Warnings",
                "Other Parameters",
                "Attributes",
                "Methods",
                "References",
                "Examples",
            ]
        }
    return parsed


def dedent_recursive(data):
    """Dedent nested mapping of strings."""
    out = type(data)()
    for k in data:
        v = data[k]
        if isinstance(v, str):
            v = dedent(v).strip()
        else:
            v = dedent_recursive(v)
        out[k] = v
    return out


def _recursive_keys(data):
    keys = []
    for k, v in data.items():
        if isinstance(v, dict):
            key_list = [f"{k}.{x}" for x in _recursive_keys(v)]
        elif isinstance(v, str):
            key_list = [k]
        else:
            raise ValueError(f"unknown type {type(v)}")

        keys.extend(key_list)

    return keys


class DocFiller:
    """
    Parameters
    ----------
    func_or_doc : callable or str
        Docstring to parse.  If callable, extract from function signature.
    key_char : str, default='|'
        Optional string to split name into key/name pair.
    """

    def __init__(self, params=None):
        """Init."""
        if params is None:
            params = {}
        self.data = params

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __getitem__(self, key):
        val = self.data[key]
        if isinstance(val, dict):
            return type(self)(val)
        else:
            return val

    def dedent(self):
        return type(self)(dedent_recursive(self.data))

    @gcached(prop=False)
    def keys(self):
        return _recursive_keys(self.data)

    def assign_combined_key(self, new_key, keys):
        """Combine multiple keys into single key."""

        data = self.data.copy()

        data[new_key] = "\n".join([self.data[k] for k in keys])
        return type(self)(data)

    @classmethod
    def concat(cls, *args, **kwargs):
        """
        Create new object from multiple DocFiller or dict objects.

        Parameters
        ----------
        *args : dict or Docfiller objects
        **kwargs : named dict or Docfiller objects

        Returns
        -------
        DocFiller
            combined DocFiller object.

        Notes
        -----
        Use unnamed `args` to pass in underlying data.
        Use names ``kwargs`` to add namespace.
        """

        # create

        data = {}

        def _update_data(x):
            if isinstance(x, DocFiller):
                x = x.params
            data.update(**x)

        for a in args:
            _update_data(a)

        for k, v in kwargs.items():
            if isinstance(v, cls):
                kwargs[k] = v.data

        _update_data(kwargs)
        return cls(data)

    def append(self, *args, **kwargs):
        """Calls ``concat`` method with ``self`` as first argument."""

        return type(self).concat(self, *args, **kwargs)

    @gcached()
    def params(self):
        return AttributeDict.from_dict(self.data, max_level=1)

    @gcached()
    def default_decorator(self):
        return docfiller(**self.params)

    def update(self, *args, **kwargs):
        data = self.data.copy()
        data.update(*args, **kwargs)
        return type(self)(params=data)

        # self.data.update(*args, **kwargs)
        # return self

    def decorate(self, *templates, **params):
        nparams, ntemplates = len(params), len(templates)
        if ntemplates == nparams == 0:
            return self.default_decorator

        elif ntemplates == 1 and nparams == 0:
            func = templates[0]
            if callable(func):
                return self.default_decorator(func)
            else:
                return docfiller(*templates)

        elif nparams > 0:
            params = AttributeDict.from_dict({**self.data, **params}, max_level=1)
            return docfiller(*templates, **params)

        else:
            return docfiller(*templates)

    def __call__(self, *templates, **params):
        if len(params) > 0:
            params = AttributeDict.from_dict({**self.data, **params}, max_level=1)
            return docfiller(*templates, **params)

        elif len(templates) > 0:
            return docfiller(*templates, **self.params)

        else:
            return self.default_decorator

    @classmethod
    def from_dict(
        cls,
        params,
        namespace=None,
        combine_keys=None,
        keep_keys=True,
        key_map=None,
    ):
        """
        Create a Docfiller instance from a dictionary.

        Parameters
        ----------
        params : mapkping
        namespace : str, optional
            Top level namespace for DocFiller.
        combine_keys : str, sequence of str, mapping, optional.
            If str or sequence of str, Keys of ``params`` to at the top level.
            If mapping, should be of form {namespace: key(s)}

        keep_keys : str, sequence of str, bool
            If False, then don't keep any keys at top level.  This is useful with the ``combine_keys`` parameter.
            If True, keep all keys, regardless of combine_keys.
            If str or sequence of str, keep these keys in output.
        key_map : mapping or callable
            Function or mapping to new keys in resulting dict.

        Returns
        -------
        DocFiller
        """
        if not keep_keys:
            keep_keys = []
        elif keep_keys is True:
            keep_keys = params.keys()
        elif isinstance(keep_keys, str):
            keep_keys = [keep_keys]

        if combine_keys:
            if isinstance(combine_keys, str):
                combine_keys = [combine_keys]

            updated_params = {k: params[k] for k in keep_keys}

            # if isinstance(combine_keys, (list, tuple)):
            #     combine_keys = {'': combine_keys}

            # for k, v in combine_keys.items():

            #     if k in updated_params:
            #         updated_params =

            for k in combine_keys:
                updated_params.update(**params[k])

        else:
            updated_params = {k: params[k] for k in keep_keys}

        if key_map:
            if not callable(key_map):
                key_map = lambda x: key_map[x]

            updated_params = {key_map(k): updated_params[k] for k in updated_params}

        if namespace:
            updated_params = {namespace: updated_params}

        return cls(params=updated_params)

    @classmethod
    def from_docstring(
        cls,
        func_or_doc,
        namespace=None,
        combine_keys=None,
        key_char="|",
        keep_keys=True,
        key_map=None,
    ):
        """
        Create a Docfiller instance from a function or docstring.

        Parameters
        ----------
        func_or_doc
        """
        params = parse_docstring(
            func_or_doc=func_or_doc, key_char=key_char, expand=True
        )
        return cls.from_dict(
            params=params,
            namespace=namespace,
            combine_keys=combine_keys,
            keep_keys=keep_keys,
            key_map=key_map,
        )
