
# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" yastn.truncation_mask() """
import pytest
import yastn

tol = 1e-12


def test_svd_multiplets(config_kwargs):
    #
    # fixing tensor for testing
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    S0 = yastn.Tensor(config_U1, isdiag=True)
    S1 = yastn.Tensor(config_U1, isdiag=True)
    v0  = [1, -1, 0.1001, 0.1000, 0.1000, 0.0999, 0.001001, 0.001000, 0, 0, 0]
    vp1 = [1,  1, 0.1000, 0.1000, 0.0010, 0]
    vm1 = [1, -0.98, 0.1000, -0.0999, 0]
    vm2 = [-0.1000, -0.0998, 0.0010]
    S0.set_block(ts=(0, 0), Ds=11, val=v0)
    S0.set_block(ts=(1, 1), Ds=6, val=vp1)
    S0.set_block(ts=(-1, -1), Ds=5, val=vm1)
    S0.set_block(ts=(-2, -2), Ds=3, val=vm2)
    #
    v0  = [-1, 0.001001, 1, 0.001000, 0, 0, 0, 0.1001, 0.1000, 0.1000, 0.0999]
    vp1 = [1, -0.0999, 0, -0.98, 0.1000]
    vm1 = [1, 0.0010, 1, 0.1000, 0.1000, 0]
    vm2 = [-0.0998, 0.0010, -0.1000]
    S1.set_block(ts=(0, 0), Ds=11, val=v0)
    S1.set_block(ts=(1, 1), Ds=5, val=vp1)
    S1.set_block(ts=(-1, -1), Ds=6, val=vm1)
    S1.set_block(ts=(-2, -2), Ds=3, val=vm2)
    #
    Db0 = {(0,): 3, (1,): 4,  (-2,): 5}
    tb0 = {(0,): 0.01, (-1,): 0.5, (-2,): 0.1}
    Db1 = {(0,): 3, (-1,): 4, (-2,): 5}
    tb1 = {(0,): 0.01, (1,): 0.5,  (-2,): 0.1}

    for S, Db, tb in [(S0, Db0, tb0), (S1, Db1, tb1)]:
        #
        #  D_total
        Smask = yastn.truncation_mask(S, which='LM', D_total=12)
        assert sum(Smask.data) == 12 and abs(sum(Smask.data * S.data).item() - 2.4201) < tol
        Smask = yastn.truncation_mask(S, which='LR', D_total=12)
        assert sum(Smask.data) == 12 and abs(sum(Smask.data * S.data).item() - 4.701001) < tol
        Smask = yastn.truncation_mask(S, which='SM', D_total=13)
        assert sum(Smask.data) == 13 and abs(sum(Smask.data * S.data).item() - 0.004201) < tol
        Smask = yastn.truncation_mask(S, which='SR', D_total=11)
        assert sum(Smask.data) == 11 and abs(sum(Smask.data * S.data).item() + 2.2787) < tol
        #
        #  tol
        Smask = yastn.truncation_mask(S, which='LM', tol=0.09999999)
        assert sum(Smask.data) == 13 and abs(sum(Smask.data * S.data).item() - 2.5201) < tol
        Smask = yastn.truncation_mask(S, which='LR', tol=0.09999999)
        assert sum(Smask.data) == 10 and abs(sum(Smask.data * S.data).item() - 4.6001) < tol
        #
        #  D_block
        Smask = yastn.truncation_mask(S, which='LM', D_block=5)
        assert sum(Smask.data) == 18 and abs(sum(Smask.data * S.data).item() - 2.3224) < tol
        Smask = yastn.truncation_mask(S, which='SR', D_block=Db)
        assert sum(Smask.data) == 10 and abs(sum(Smask.data * S.data).item() + 0.9978) < tol
        #
        #  tol_block
        Smask = yastn.truncation_mask(S, which='LM', tol_block=0.09999999)
        assert sum(Smask.data) == 14 and abs(sum(Smask.data * S.data).item() - 2.4203) < tol
        Smask = yastn.truncation_mask(S, which='LR', tol_block=tb)
        assert sum(Smask.data) == 6 and abs(sum(Smask.data * S.data).item() - 2.4000) < tol
        #
        #  hermitian
        Smask = yastn.truncation_mask(S, which='LR', tol=0.09999999, hermitian=True)
        assert sum(Smask.data) == 8 and abs(sum(Smask.data * S.data).item() - 4.4001) < tol
        #
        #  largest_gap
        Smask = yastn.truncation_mask(S, which='LM', D_total=12, largest_gap=True)
        assert sum(Smask.data) == 16 and abs(sum(Smask.data * S.data).item() - 2.4203) < tol
        Smask = yastn.truncation_mask(S, which='LR', D_total=12, largest_gap=True)
        assert sum(Smask.data) == 15 and abs(sum(Smask.data * S.data).item() - 4.704001) < tol
        Smask = yastn.truncation_mask(S, which='LR', D_total=15, largest_gap=True)
        assert sum(Smask.data) == 15 and abs(sum(Smask.data * S.data).item() - 4.704001) < tol
        #
        #  eps_multiplet
        Smask = yastn.truncation_mask(S, which='LM', D_total=12, eps_multiplet=1e-2)
        assert sum(Smask.data) == 6 and abs(sum(Smask.data * S.data).item() - 2.0200) < tol
        Smask = yastn.truncation_mask(S, which='LM', D_total=6, eps_multiplet=1e-2)
        assert sum(Smask.data) == 6 and abs(sum(Smask.data * S.data).item() - 2.0200) < tol
        Smask = yastn.truncation_mask(S, which='LR', D_total=17, eps_multiplet=1e-6)
        assert sum(Smask.data) == 15 and abs(sum(Smask.data * S.data).item() - 4.704001) < tol
        #
        with pytest.raises(yastn.YastnError,
                        match="Truncation by tolerance with which='SR' or 'SM' is not supported."):
            yastn.truncation_mask(S, which="SR", tol=1e-2)
        #
        with pytest.raises(yastn.YastnError,
                        match="Truncation by tolerance with which='SR' or 'SM' is not supported."):
            yastn.truncation_mask(S, which="SM", tol=1e-2)
        #
        with pytest.raises(yastn.YastnError,
                        match="Truncation by block cannot be used when multiplet-related schmes are invoked."):
            yastn.truncation_mask(S, tol_block=1e-2, largest_gap=True)
        #
        with pytest.raises(yastn.YastnError,
                        match="Only which = 'LM' or 'LR' are supported when multiplet-related schmes are invoked."):
            yastn.truncation_mask(S, which='SR', eps_multiplet=1e-2)
        #
        with pytest.raises(yastn.YastnError,
                        match="Truncation multiplets cannot perform both schemes largest_gap and eps_multiplets simultaneously."):
            yastn.truncation_mask(S, eps_multiplet=1e-2, largest_gap=True)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
