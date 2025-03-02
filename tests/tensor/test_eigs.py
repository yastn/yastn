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
""" yastn.compress_to_1d() yastn.decompress_from_1d()  in combination with scipy LinearOperator and eigs """
import pytest
from scipy.sparse.linalg import eigs
import yastn


def test_eigs_arnoldi(config_kwargs):
    """
    An example that is drawn for torch is challenging, with degeneracies and near-degeneracies.
    The current version of yastn.eigs has only very basic eigs implemented at the moment,
    which is however what is needed in typical dmrg.
    This test should be updated if more general eigs gets implemented.
    """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    config_U1.backend.random_seed(seed=0)  # fix seed for testing

    legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(1, 1)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, legs=legs)  # could be an MPS tensor

    tm = yastn.ncon([a, a.conj()], [(-1, 1, -3), (-2, 1, -4)])
    tm = tm.fuse_legs(axes=((2, 3), (0, 1)), mode='hard')
    tmn = tm.to_numpy()

    f = lambda t: yastn.ncon([t, a, a.conj()], [(1, 3, -3), (1, 2, -1), (3, 2, -2)])

    legs = [a.get_legs(0).conj(),
            a.get_legs(0),
            yastn.Leg(a.config, s=1, t=(-1, 0, 1, -2, 2), D=(1, 1, 1, 1, 1))]

    mp = {'LM': lambda x: abs(x),
          'LR': lambda x: x.real,
          'SR': lambda x: x.real,}

    for which in ('LM', 'LR', 'SR'):
        w_ref, _ = eigs(tmn, k=1, which=which)  # use scipy.sparse.linalg.eigs
        v0 = yastn.randC(config=a.config, legs=legs)
        v0 = [v0 / v0.norm()]
        w_old = 100
        for ii in range(10):  # no restart in yastn.eigs
            w, v0 = yastn.eigs(f, v0=v0[0], k=1, which=which, ncv=25, hermitian=False)
            if abs(w - w_old) < 1e-8:
                break
            w_old = w
        print(which, ii, abs(mp[which](w_ref) - mp[which](w.item())))
        assert (f(v0[0]) - w[0] * v0[0]).norm() < 1e-4
        assert abs(mp[which](w_ref) - mp[which](w.item())) < 1e-4


def test_eigs_lanczos(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    config_U1.backend.random_seed(seed=0)  # fix seed for testing

    legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(1, 1)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, legs=legs)  # could be an MPS tensor

    tm = yastn.ncon([a, a.conj()], [(-1, 1, -3), (-2, 1, -4)])
    tm = tm.fuse_legs(axes=((2, 3), (0, 1)), mode='hard')
    tmn = tm.to_numpy()
    f = lambda t: yastn.ncon([t, a, a.conj()], [(1, 3, -3), (1, 2, -1), (3, 2, -2)])

    legs = [a.get_legs(0).conj(),
            a.get_legs(0),
            yastn.Leg(a.config, s=1, t=(-1, 0, 1, -2, 2), D=(1, 1, 1, 1, 1))]

    tmn = tmn + tmn.T
    f = lambda t: yastn.ncon([t, a, a.conj()], [(1, 3, -3), (1, 2, -1), (3, 2, -2)]) + \
                  yastn.ncon([t, a.conj(), a], [(1, 3, -3), (-1, 2, 1), (-2, 2, 3)])

    for which in ('LM', 'LR', 'SR'):
        w_ref, _ = eigs(tmn, k=1, which=which)  # use scipy.sparse.linalg.eigs
        v0 = yastn.randC(config=a.config, legs=legs)
        v0 = [v0 / v0.norm()]
        w_old = 100
        for ii in range(100):  # no restart in yastn.eigs
            w, v0 = yastn.eigs(f, v0=v0[0], k=1, which=which, ncv=4, hermitian=True)
            if abs(w - w_old) < 1e-9:
                break
            w_old = w
        print(which,  ii, abs(w_ref - w.item()))
        assert abs(w_ref - w.item()) < 1e-8


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
