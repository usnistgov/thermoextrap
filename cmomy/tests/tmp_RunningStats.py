from builtins import *
import pavey as RS
import numpy as np


def assert_rs(rs, ave, std0, std1, var0, var1, cov0=None, cov1=None):
    np.testing.assert_almost_equal(ave, rs.mean())
    np.testing.assert_almost_equal(std0, rs.std(0))
    np.testing.assert_almost_equal(std1, rs.std(1))

    np.testing.assert_almost_equal(var0, rs.var(0))
    np.testing.assert_almost_equal(var1, rs.var(1))


    if cov0 is not None:
        np.testing.assert_almost_equal(cov0, rs.cov(0))

    if cov1 is not None:
        np.testing.assert_almost_equal(cov1, rs.cov(1))
    
    

def test_RunningStats(n0=10, n1=5, n2=100):

    L = [np.random.randn(np.random.randint(n1, n2)) for _ in range(n0)]
    Lc = np.concatenate(L)

    ave = Lc.mean()
    std0 = Lc.std(ddof=0)
    std1 = Lc.std(ddof=1)
    var0 = Lc.var(ddof=0)
    var1 = Lc.var(ddof=1)

    # pushing
    rs0 = RS.RunningStats()
    rs1 = RS.RunningStats()
    rs2 = RS.RunningStats()

    for x in L:
        rs0.push_vals(x)
        rs1.push_stat(w=len(x), a=x.mean(), v=x.var(ddof=0))
        for xx in x:
            rs2.push_val(xx)

    for rs in [rs0, rs1, rs2]:
        assert_rs(rs, ave, std0, std1, var0, var1)

    # froming
    rs0 = RS.RunningStats.from_vals(Lc)
    rs1 = RS.RunningStats.from_stat(len(Lc), ave, var0)
    ave_s = np.array([np.mean(x) for x in L])
    var_s = np.array([np.var(x) for x in L])
    wt_s = np.array([len(x) for x in L])
    rs2 = RS.RunningStats.from_stats(wt_s, ave_s, var_s)
    for rs in [rs0, rs1, rs2]:
        assert_rs(rs, ave, std0, std1, var0, var1)


    #test addition
    rs0 = RS.RunningStats()
    rs1 = RS.RunningStats()
    for x in L[:n0 // 2]:
        rs0.push_vals(x)
    for x in L[n0 // 2:]:
        rs1.push_vals(x)

    rs2 = rs0 + rs1
    assert_rs(rs2, ave, std0, std1, var0, var1)

    #test subtraction
    rs3 = rs2 - rs1
    assert_rs(rs3, rs0.mean(), rs0.std(0), rs0.std(1), rs0.var(0), rs0.var(1))

    rs0 += rs1
    assert_rs(rs0, ave, std0, std1, var0, var1)


def test_RunningStatsVec(n0=10, n1=5, n2=100, shape=(10, 2)):

    L0 = []
    for _ in range(n0):
        n = np.random.randint(n1, n2)
        L0.append(np.random.randn(*((n, ) + shape)))
    X0 = np.concatenate(L0, axis=0)

    L1 = []
    for _ in range(n0):
        n = np.random.randint(n1, n2)
        L1.append(np.random.randn(*(shape + (n, ))))
    X1 = np.concatenate(L1, axis=-1)

    for L, X, axis in zip([L0, L1], [X0, X1], [0, -1]):

        ave = X.mean(axis=axis)
        std0 = X.std(axis=axis, ddof=0)
        std1 = X.std(axis=axis, ddof=1)
        var0 = X.var(axis=axis, ddof=0)
        var1 = X.var(axis=axis, ddof=1)

        assert ave.shape == shape
        rs0 = RS.RunningStatsVec(shape=shape)
        rs1 = RS.RunningStatsVec(shape=shape)
        rs2 = RS.RunningStatsVec(shape=shape)
        for x in L:
            rs0.push_vals(x, axis=axis)
            rs1.push_stat(w=x.shape[axis],
                           a=x.mean(axis=axis),
                           v=x.var(ddof=0, axis=axis))

            if axis == 0:
                for xx in x:
                    rs2.push_val(xx)
            elif axis == -1:
                for xx in np.rollaxis(x,x.ndim-1,0):
                    rs2.push_val(xx)

        for rs in [rs0, rs1, rs2]:
            assert_rs(rs, ave, std0, std1, var0, var1)

        # test addition
        rs0 = RS.RunningStatsVec(shape)
        rs1 = RS.RunningStatsVec(shape)

        for x in L[:n0 // 2]:
            rs0.push_vals(x, axis=axis)
        for x in L[n0 // 2:]:
            rs1.push_vals(x, axis=axis)

        rs2 = rs0 + rs1
        assert_rs(rs2, ave, std0, std1, var0, var1)

        #test subtraction
        rs3 = rs2 - rs1
        assert_rs(rs3, rs0.mean(), rs0.std(0), rs0.std(1), rs0.var(0), rs0.var(1))

        #iadd
        rs0 += rs1
        assert_rs(rs0, ave, std0, std1, var0, var1)


def test_cov(n0=10, n1=5, n2=100, shape=None):

    if shape is None:
        #shape = (10,2)
        shape = tuple(np.random.randint(1,10,3))

    L0 = []
    L1 = []
    for _ in range(n0):
        n = np.random.randint(n1, n2)
        L0.append(np.random.randn(*((n,) + shape)))
        L1.append(np.random.randn(*(shape +(n,))))

    X0 = np.concatenate(L0, axis=0)
    X1 = np.concatenate(L1, axis=-1)

    for L, X, axis in zip([L0, L1], [X0, X1], [0, -1]):
        ave = X.mean(axis=axis)
        std0 = X.std(axis=axis, ddof=0)
        std1 = X.std(axis=axis, ddof=1)
        var0 = X.var(axis=axis, ddof=0)
        var1 = X.var(axis=axis, ddof=1)

        cov0 = RS.cov_nd(X, axis=axis, ddof=0)
        cov1 = RS.cov_nd(X, axis=axis, ddof=1)

        
        rs0 = RS.RunningStatsVecCov(shape)
        rs2 = RS.RunningStatsVecCov(shape)

        for x in L:
            rs0.push_vals(x, axis=axis)

            if axis == 0:
                for xx in x:
                    rs2.push_val(xx)
            elif axis == -1:
                for xx in np.rollaxis(x,x.ndim-1,0):
                    rs2.push_val(xx)


        for rs in [rs0, rs2]:
            assert_rs(rs, ave, std0, std1, var0, var1, cov0, cov1)


        # test addition
        rs0 = RS.RunningStatsVecCov(shape)
        rs1 = RS.RunningStatsVecCov(shape)

        for x in L[:n0 // 2]:
            rs0.push_vals(x, axis=axis)
        for x in L[n0 // 2:]:
            rs1.push_vals(x, axis=axis)

        rs2 = rs0 + rs1
        assert_rs(rs2, ave, std0, std1, var0, var1, cov0, cov1)


        #test subtraction
        rs3 = rs2 - rs1
        assert_rs(rs3, rs0.mean(), rs0.std(0), rs0.std(1), rs0.var(0), rs0.var(1), rs0.cov(0), rs0.cov(1))

        rs0 += rs1
        assert_rs(rs0, ave, std0, std1, var0, var1, cov0, cov1)
