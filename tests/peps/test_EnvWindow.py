# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Test PEPS measurments with MpsBoundary in a product state. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps
try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg

tol = 1e-12

def init_peps(Dphys=()):
    """ initialized PEPS with mixed bond dimensions for testing. """
    geometry = fpeps.SquareLattice(dims=(2, 3))  # boundary='infinite'
    psi = fpeps.Peps(geometry)
    s = (-1, 1, 1, -1) + (1,) * len(Dphys)
    psi[0, 0] = yastn.rand(cfg, s=s, D=(2, 3, 4, 5) + Dphys, dtype='complex128')
    psi[1, 0] = yastn.rand(cfg, s=s, D=(4, 6, 2, 4) + Dphys, dtype='complex128')
    psi[0, 1] = yastn.rand(cfg, s=s, D=(3, 5, 5, 2) + Dphys, dtype='complex128')
    psi[1, 1] = yastn.rand(cfg, s=s, D=(5, 4, 3, 6) + Dphys, dtype='complex128')
    psi[0, 2] = yastn.rand(cfg, s=s, D=(2, 2, 3, 3) + Dphys, dtype='complex128')
    psi[1, 2] = yastn.rand(cfg, s=s, D=(3, 6, 2, 6) + Dphys, dtype='complex128')
    return psi


def test_window_shapes():
    """ Initialize a product PEPS and perform a set of measurment. """
    for Dphys in [(), (2,)]:  # Dphys = () gives single-layer PEPS; (2,) gives double-layer PEPS
        psi = init_peps(Dphys)
        #
        env_ctm = fpeps.EnvCTM(psi, init='eye')
        #
        #  test contractions of < mps | mpo | mps > in different configurations
        #
        for xrange in [(0, 1), (1, 2), (2, 3), (0, 3)]:
            for yrange in [(0, 5), (1, 3), (2, 6)]:
                env_win = fpeps.EnvWindow(env_ctm, xrange=xrange, yrange=yrange)
                for ix in range(*xrange):
                    top = env_win[ix, 't']
                    TMh = env_win[ix, 'h']
                    btm = env_win[ix, 'b']
                    mps.vdot(btm, TMh, top)
        #
        for yrange in [(0, 1), (1, 2), (2, 3), (1, 4)]:
            for xrange in [(0, 5), (1, 3), (2, 6)]:
                env_win = fpeps.EnvWindow(env_ctm, xrange=xrange, yrange=yrange)
                for iy in range(*yrange):
                    rht = env_win[iy, 'r']
                    TMv = env_win[iy, 'v']
                    lft = env_win[iy, 'l']
                    mps.vdot(rht, TMv, lft)

    with pytest.raises(yastn.YastnError):
        env_win[10, 'l']
        # n=10 not within self.yrange=(0, 3)
    with pytest.raises(yastn.YastnError):
        env_win[-2, 't']
        # n=-2 not within self.xrange=(0, 4)
    with pytest.raises(yastn.YastnError):
        env_win[2, 'none']
        # dirn='none' not recognized. Should be 't', 'h' 'b', 'r', 'v', or 'l'.


def test_window_measure():
    """ checks syntax of sample and measure_2site, but not values. """
    # for Dphys = 2
    psi = init_peps(Dphys=(2,))
    D_total = 8
    opts_svd = {'D_total': D_total, 'tol': 1e-10}
    env_ctm = fpeps.EnvCTM(psi, init='rand')
    #
    for _ in range(5):
        env_ctm.update_(opts_svd=opts_svd)  # single sweep
    env_win = fpeps.EnvWindow(env_ctm, xrange=(0, 4), yrange=(0, 3))
    #
    # test sample
    #
    ops = yastn.operators.Spin12(sym='dense')
    vecs = [ops.vec_z(val=v) for v in [-1, 1]]
    projs = [yastn.tensordot(vec, vec.conj(), axes=((), ())) for vec in vecs]
    #
    number = 4
    out = env_win.sample(projs, number=number, return_info=True, progressbar=True)
    info = out.pop('info')
    assert info['error'] > 1e-3
    assert info['opts_svd']['D_total'] == D_total
    assert len(out) == 12
    for ny in range(0, 3):
        for nx in range(0, 4):
            assert len(out[nx, ny]) == number
            assert all(x in [0, 1] for x in out[nx, ny])

    env_ctm.ctmrg_(opts_svd, max_sweeps=20, corner_tol=1e-3)
    projs = {k: v for k, v in zip('tb', projs)}
    out = env_win.sample(projs, number=number, return_info=True)
    info = out.pop('info')
    assert info['error'] < 1e-4
    assert info['opts_svd']['D_total'] == D_total
    assert len(out) == 12
    for ny in range(0, 3):
        for nx in range(0, 4):
            assert len(out[nx, ny]) == number
            assert all(x in 'tb' for x in out[nx, ny])
    #
    with pytest.raises(yastn.YastnError):
        env_win.sample(projectors={(0, 0): projs, (1, 0): projs})
        # projectors not defined for some sites in xrange=(0, 4), yrange=(0, 3).
    #
    # test measure_2site
    #
    out = env_win.measure_2site(ops.sz(), ops.sz(), opts_svd={'D_total':2})


if __name__ == '__main__':
    test_window_shapes()
    test_window_measure()
