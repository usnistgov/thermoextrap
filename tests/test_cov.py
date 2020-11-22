import cmomy.accumulator as accumulator
import cmomy.central as central

import numpy as np
import pytest

# central moments single variable
def _get_cmom_single(u, x, nu, nx, last=True):
    test = np.zeros((nu+1, nx+1))
    dx = x - x.mean()
    du = u - u.mean()
    for i in range(nu+1):
        for j in range(nx+1):
            if i == 0 and j == 0:
                other = len(x)
            elif i + j == 1:
                other = (u**i * x**j).mean()
            else:
                other = (du**i * dx**j).mean()
            test[i, j] = other


    return test

def _get_data_single(nrec=100, weighted=False):
    x0 = np.random.rand(nrec)
    x1 = np.random.rand(nrec)

    if weighted is None:
        w = None
    elif weighted:
        w = np.random.rand(nrec)
    else:
        w = np.ones_like(x0)

    return w, x0, x1



@pytest.mark.parametrize("nrec", [100])
@pytest.mark.parametrize("moments", [5, (5,5), (1,1)])
@pytest.mark.parametrize("weighted", [False, True])
def test_vals(nrec, moments, weighted):

    w, x0, x1 = _get_data_single(nrec, weighted)

    data = central.central_comoments(x0, x1, moments, w, axis=0)

    if not weighted:
        if isinstance(moments, int):
            nu = moments
            nx = moments
        else:
            nu, nx = moments

        dataA = _get_cmom_single(x0, x1, nu, nx)
        np.testing.assert_allclose(data, dataA)


    s = central.StatsAccumCov.zeros(mom=moments)
    s.push_vals(x0, x1, w)
    np.testing.assert_allclose(s.data, data)

    s = central.StatsAccumCov.from_vals(x=x0, y=x1, w=w, mom=moments)
    np.testing.assert_allclose(s.data, data)



@pytest.mark.parametrize("nrec", [100])
@pytest.mark.parametrize("moments", [5])
@pytest.mark.parametrize("weighted", [False, True])
def test_StatsAccum_stats(nrec, moments, weighted):
    # unweighted
    w, x0, x1 = _get_data_single(nrec, weighted)

    data = central.central_comoments(x0, x1, moments, w)
    if not weighted:
        if isinstance(moments, int):
            nu = moments
            nx = moments
        else:
            nu, nx = moments

        dataA = _get_cmom_single(x0, x1, nu, nx)
        np.testing.assert_allclose(data, dataA)


    splits = [len(x0) // 3, len(x0) // 3 * 2]

    ws = np.split(w, splits)
    x0s = np.split(x0, splits)
    x1s = np.split(x1, splits)

    datas = []
    for ww, xx0, xx1 in zip(ws, x0s, x1s):
        datas.append(central.central_comoments(xx0, xx1, moments, ww))

    datas = np.array(datas)

    # factory
    s = central.StatsAccumCov.from_datas(datas, mom=moments)
    np.testing.assert_allclose(s.data, data)

    # pushs
    s = central.StatsAccumCov.zeros(mom=moments)

    for d in datas:
        s.push_data(d)
    np.testing.assert_allclose(s.data, data)

    s.zero()
    s.push_datas(datas)
    np.testing.assert_allclose(s.data, data)

    # addition
    S = [central.StatsAccumCov.from_data(d, mom=moments) for d in datas]
    out = S[0]
    for s in S[1:]:
        out = out + s
    np.testing.assert_allclose(out.data, data)

    out = sum(S, central.StatsAccumCov.zeros(mom=moments))
    np.testing.assert_allclose(out.data, data)

    out = central.StatsAccumCov.zeros(mom=moments)
    for s in S:
        out += s
    np.testing.assert_allclose(out.data, data)

    # subtraction
    out = S[0] + S[1] - S[0]
    np.testing.assert_allclose(out.data, datas[1])

    # iadd/isub
    out = central.StatsAccumCov.zeros(mom=moments)
    out += S[0]
    np.testing.assert_allclose(out.data, S[0].data)

    out += S[1]
    np.testing.assert_allclose(out.data, (S[0] + S[1]).data)

    out -= S[0]
    np.testing.assert_allclose(out.data, S[1].data)

    # mult
    out1 = S[0] * 2
    out2 = S[0] + S[0]
    np.testing.assert_allclose(out1.data, out2.data)

    # imul
    out = central.StatsAccumCov.from_vals(x0s[0], x1s[0], ws[0], mom=moments)
    out *= 2
    np.testing.assert_allclose(out.data, (S[0] + S[0]).data)



def _get_data_vec(shape, weighted=False):
    x0 = np.random.rand(*shape)
    x1 = np.random.rand(*shape)

    if weighted is None:
        w = None
    elif weighted:
        w = np.random.rand(*shape)
    else:
        w = np.ones_like(x0)

    return w, x0, x1


@pytest.mark.parametrize("dshape,axis", [
    ((100,1), 0),
    ((100,10), 0),
    ((10,100), 1),
    ((100, 5, 5), 0), ((5, 100, 5), 1), ((5, 5, 100), 2)
])
@pytest.mark.parametrize("moments", [5, (5, 5), (1, 1)])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize('broadcast', [False, True])
def test_vec_vals(dshape, axis, moments, weighted, broadcast):
    # unweighted
    wt, x0, x1 = _get_data_vec(dshape, weighted)

    if broadcast:
        index = [0] * x1.ndim
        index[axis] = slice(None)
        x1 = x1[tuple(index)]
        x1b = central._axis_expand_broadcast(x1, x0.shape, axis, expand=True, roll=False, broadcast=True)
    else:
        x1b = None


    # single weight
    slicer = [0] * wt.ndim
    slicer[axis] = slice(None)
    ws = wt[tuple(slicer)]

    # push
    shape = list(dshape)
    shape.pop(axis)
    shape = tuple(shape)


    for w in (wt, ws):
        data = central.central_comoments(x0, x1, moments, w, axis=axis, broadcast=broadcast)



        s = central.StatsAccumCov.zeros(shape=shape, mom=moments)
        # push_vals
        s.push_vals(x0, x1, w, axis=axis, broadcast=broadcast)
        np.testing.assert_allclose(s.data, data)

        # from vals
        s = central.StatsAccumCov.from_vals(x0, x1, w, mom=moments, axis=axis, broadcast=broadcast)
        np.testing.assert_allclose(s.data, data)


        # testing broadcasting
        if x1b is not None:
            tmp = central.central_comoments(x0, x1b, moments, w, axis=axis, broadcast=False)
            np.testing.assert_allclose(tmp, data)

            s = central.StatsAccumCov.from_vals(x0, x1b, w, mom=moments, axis=axis, broadcast=False)
            np.testing.assert_allclose(s.data, data)


@pytest.mark.parametrize("dshape,axis", [
    ((100,1), 0),
    ((100, 10), 0),
    ((10,100), 1),
    ((100, 5, 5), 0), ((5, 100, 5), 1), ((5, 5, 100), 2)
])
@pytest.mark.parametrize("moments", [5, (5,5), (1,1)])
@pytest.mark.parametrize("weighted", [False, True])
def test_Vec_stats(dshape, axis, moments, weighted):
    # unweighted
    wt, x0, x1 = _get_data_vec(dshape, weighted)

    # single weight
    slicer = [0] * wt.ndim
    slicer[axis] = slice(None)
    ws = wt[tuple(slicer)]

    # push
    shape = list(dshape)
    shape.pop(axis)
    shape = tuple(shape)

    n = x0.shape[axis]
    splits = [n //3, n //3*2]
    x0split = np.split(x0, splits, axis)
    x1split = np.split(x1, splits, axis)

    for w in (wt, ws):
        data = central.central_comoments(x0, x1, moments, w, axis=axis)
        if w.ndim == 1:
            wsplit = np.split(w, splits)
        else:
            wsplit = np.split(w, splits, axis=axis)

        datas = []
        for ww, xx0, xx1 in zip(wsplit, x0split, x1split):
            datas.append(central.central_comoments(xx0, xx1, moments, ww, axis=axis))
        datas = np.array(datas)

 
        # factory
        s = central.StatsAccumCov.from_datas(datas, mom=moments, axis=0)
        np.testing.assert_allclose(s.data, data)

        # pushs
        s = central.StatsAccumCov.zeros(mom=moments, shape=shape)

        for d in datas:
            s.push_data(d)
        np.testing.assert_allclose(s.data, data)

        s.zero()
        s.push_datas(datas)
        np.testing.assert_allclose(s.data, data)


        # addition
        S = [central.StatsAccumCov.from_data(d, mom=moments) for d in datas]
        out = S[0]
        for s in S[1:]:
            out = out + s
        np.testing.assert_allclose(out.data, data)

        out = sum(S, s.zeros_like())
        np.testing.assert_allclose(out.data, data)

        out = s.zeros_like()
        for s in S:
            out += s
        np.testing.assert_allclose(out.data, data)

        # subtraction
        out = S[0] + S[1] - S[0]
        np.testing.assert_allclose(out.data, datas[1])

        # iadd/isub
        out = s.zeros_like()
        out += S[0]
        np.testing.assert_allclose(out.data, S[0].data)

        out += S[1]
        np.testing.assert_allclose(out.data, (S[0] + S[1]).data)

        out -= S[0]
        np.testing.assert_allclose(out.data, S[1].data)

        # mult
        out1 = S[0] * 2
        out2 = S[0] + S[0]
        np.testing.assert_allclose(out1.data, out2.data)

        # imul
        out = S[0].copy()
        out *= 2
        np.testing.assert_allclose(out.data, (S[0] + S[0]).data)



