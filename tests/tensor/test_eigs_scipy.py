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
""" yastn.to_dict() yastn.from_dict() yastn.split_data_and_meta()
    yastn.combine_data_and_meta(), in combination with scipy LinearOperator and eigs """
import numpy as np
import pytest
from scipy.sparse.linalg import eigs, LinearOperator
import yastn

tol = 1e-8  #pylint: disable=invalid-name

numpy_test = pytest.mark.skipif("'np' not in config.getoption('--backend')",
                                reason="using scipy procedures for raw data requires np")


@numpy_test
def test_eigs_simple(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 2)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(1, 1)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 2))]
    a = yastn.rand(config=config_U1, legs=legs)  # e.g., it could be an MPS tensor
    a, _ = yastn.qr(a, axes=((0, 1), 2), sQ=-1)  # orthonormalize

    # Dense transfer matrix build from a; reference solution
    tm = yastn.ncon([a, a.conj()], [(-1, 1, -3), (-2, 1, -4)])
    tm = tm.fuse_legs(axes=((2, 3), (0, 1)), mode='hard')
    tmn = tm.to_numpy()
    w_ref, v_ref = eigs(tmn, k=1, which='LM')  # use scipy.sparse.linalg.eigs

    # Initializing random tensor matching tm from left.
    # We add an extra 3-rd leg carrying charges -1, 0, 1
    # to calculate eigs over those 3 subspaces in one go.
    legs = [a.get_legs(0).conj(),
            a.get_legs(0),
            yastn.Leg(a.config, s=1, t=(-1, 0, 1), D=(1, 1, 1))]
    v0 = yastn.rand(config=a.config, legs=legs)
    # Define a wrapper that goes r1d -> yastn.tensor -> tm @ yastn.tensor -> r1d
    r1d, meta = yastn.split_data_and_meta(v0.to_dict(level=0), squeeze=True)
    def f(x):
        t = yastn.Tensor.from_dict(yastn.combine_data_and_meta(x, meta))
        t2 = yastn.ncon([t, a, a.conj()], [(1, 3, -3), (1, 2, -1), (3, 2, -2)])
        t3, _ = yastn.split_data_and_meta(t2.to_dict(level=0, meta=meta), squeeze=True)
        return t3
    ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)
    # scipy.sparse.linalg.eigs that goes though yastn symmetric tensor.
    wa, va1d = eigs(ff, v0=r1d, k=1, which='LM', tol=1e-10)
    # Transform eigenvectors into yastn tensors
    va = [yastn.Tensor.from_dict(yastn.combine_data_and_meta(x, meta)) for x in va1d.T]
    # We can remove zero blocks now, as there are eigenvectors with well defined charge
    # (though we might get superposition of symmetry sectors in case of degeneracy).
    va = [x.remove_zero_blocks() for x in va]

    # we can also limit ourselves directly to eigenvectors with desired charge, here n=0.
    legs = [a.get_legs(0).conj(),
            a.get_legs(0)]
    v0 = yastn.rand(config=a.config, legs=legs, n=0)
    r1d, meta = yastn.split_data_and_meta(v0.to_dict(level=0), squeeze=True)
    def f(x):
        t = yastn.Tensor.from_dict(yastn.combine_data_and_meta(x, meta))
        t2 = yastn.ncon([t, a, a.conj()], [(1, 3), (1, 2, -1), (3, 2, -2)])
        t3, _ = yastn.split_data_and_meta(t2.to_dict(level=0, meta=meta), squeeze=True)
        return t3
    ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)
    wb, vb1d = eigs(ff, v0=r1d, k=1, which='LM', tol=1e-10)  # scipy.sparse.linalg.eigs
    vb = [yastn.Tensor.from_dict(yastn.combine_data_and_meta(x, meta)) for x in vb1d.T]  # eigenvectors as yastn tensors

    # dominant eigenvalue should have amplitude 1 (likely degenerate in our example)
    assert all(pytest.approx(abs(x), rel=1e-10) == 1.0 for x in (w_ref, wa, wb))
    print("va -> ", va.pop())
    print("vb -> ", vb.pop())


@numpy_test
def test_eigs_mismatches(config_kwargs):
    #
    # here define a problem in a way that there are some mismatches in legs to be resolved
    #
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg0 = yastn.Leg(config_U1, s=1, t=(-2, -1, 0, 1), D=(1, 2, 3 ,4))
    leg1 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(2, 3 ,4, 5))

    a = yastn.rand(config=config_U1, legs=(leg0, leg1, leg2.conj()), n=0)
    # will be treated as mps tensor

    # dense transfer matrix build from a -- here a has some un-matching blocks between first and last legs
    tm = yastn.ncon([a, a], [(-1, 1, -3), (-2, 1, -4)], conjs=(0, 1))
    tm = tm.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    # make sure to fill-in zero blocks, as in this example tm is not a square matrix
    legs_for_tm = {0: tm.get_legs(1).conj(), 1: tm.get_legs(0).conj()}
    tmn = tm.to_numpy(legs=legs_for_tm)
    wn, vn = eigs(tmn, k=5, which='LM')  # scipy

    ## initializing random tensor matching TM, with 3-rd leg extra carrying charges -1, 0, 1
    leg02 = yastn.legs_union(leg0, leg2)
    leg_aux = yastn.Leg(a.config, s=1, t=(-1, 0, 1), D=(1, 1, 1))
    vv = yastn.rand(config=a.config, legs=(leg02, leg02.conj(), leg_aux), dtype='float64')
    r1d, meta = yastn.split_data_and_meta(vv.to_dict(level=0), squeeze=True)
    def f(x):  # change all that into a wrapper around ncon part?
        t = yastn.Tensor.from_dict(yastn.combine_data_and_meta(x, meta))
        t2 = yastn.ncon([a, a.conj(), t], [(-1, 1, 2), (-2, 1, 3), (2, 3, -3)])
        t3, _ = yastn.split_data_and_meta(t2.to_dict(level=0, meta=meta), squeeze=True)
        return t3
    ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)

    # eigs going though yastn.tensor
    wy1, vy1d = eigs(ff, v0=r1d, k=5, which='LM', tol=1e-10)  # scipy going though yastn.tensor

    # transform eigenvectors into yastn tensors
    vy = [yastn.Tensor.from_dict(yastn.combine_data_and_meta(x, meta)) for x in vy1d.T]
    # remove zero blocks and checks if that was correct
    vyr = [yastn.remove_zero_blocks(a, rtol=1e-12) for a in vy]
    assert all((yastn.norm(x - y) < tol for x, y in zip(vy, vyr)))
    # display charges of eigenvectors (only charge on last leg)
    print(vy[0].get_legs(2))
    print(vyr[0].get_legs(2))
    # for others there might be superposition between +1 and -1


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
