import numpy as np

# from .pushers import factory_pushers


def verify_value(
    x,
    val_shape,
    mom_shape=None,
    axis=None,
    nrec=None,
    expand=False,
    broadcast=False,
    flatten=True,
    verify=False,
    dtype=None,
    order=None,
    shape=None,
    shape_flat=None,
    first=True,
):
    """
    given an array `x` verify that it conforms to a target shape

    Parameters
    ----------
    x: array-like
        array to consider
    val_shape : tuple
        shape of value part of target
    mom_shape : tuple, optional
        shape of moment part of target.  If None, then target shape
        excludes moments
    verify: callable, optional
        if present, `x = verify(x)`.  For example, can pass
        verify=numpy.asarray


    axis: int, optional
        if specified, then this is the axis that will be reduced over.
        This axis will be moved to the first axis of the output
    nrec: int, optional
        number of records.  That is, x.shape[axis]
        Pass this to perform a check on sizes.
    broadcast: bool, default=False
        if True, then broadcast to target shape

    """

    if verify:
        x = np.asarray(x, dtype=dtype, order=order)

    ndim = len(val_shape)
    axis = np.core.numeric.normalize_axis_index(axis, ndim + 1)

    if shape is None or shape_flat is None:
        if ndim == 0:
            shape_flat = ()
        else:
            shape_flat = (np.prod(shape),)

        if axis is not None:
            shape = shape[:axis] + (nrec,) + shape[axis:]
            shape_flat = (nrec,) + shape_flat

        if mom_shape is not None:
            shape = shape + mom_shape
            shape_flat = shape_flat + mom_shape

    if expand:
        if x.ndim == 1 and x.ndim != len(shape) and len(x) == shape[axis]:

            reshape = [1] * len(shape)
            reshape[axis] = -1
            x = x.reshape(*reshape)

    if broadcast and x.shape != shape:
        x = np.broadcast_to(x, shape)
    else:
        assert x.shape == shape

    if first or flatten and axis != 0:
        x = np.moveaxis(x, axis, 0)

    if flatten:
        x = x.reshape(shape_flat)

    if x.ndim == 0:
        x = x[()]

    return x, shape, shape_flat


# def push_val(x,
#              mom,
#              w=None,
#              verify=True,
#              broadcast=False,
#              data=None,
#              data_flat=None,
#              val_shape=None,
# ):


#     if isinstance(mom, int):
#         mom = (mom,)
#     mom_ndim = len(mom)
#     assert mom_ndim in (1, 2)
#     mom_shape = tuple(x+1 for x in mom)


#     if mom_ndim == 1:
#         x = (x,)

#     if verify:
#         x = (np.asarray(_) for _ in x)

#     if val_shape is None:
#         val_shape = x[0].shape

#     if val_shape == ():
#         val_shape_flat = ()
#     else:
#         val_shape_flat = (np.prod(val_shape), )

#     shape = val_shape + mom_shape
#     shape_flat = val_shape_flat + mom_shape

#     # verify x[0]
#     x = verify_value(x, val_shape=val_shape, mom_shape=)
