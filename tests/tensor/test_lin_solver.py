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
import pytest
import yastn

numtol = 1e-16
errtol = 1e-12
pinv_tol = 1e-13

def test_dense(config_kwargs):
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=0)  # fix seed for testing
    
    D = 10
    a = yastn.rand(config=config, s=(-1, 1), D=[D, D])
    b = yastn.rand(config=config, s=(-1,), D=[D])
    f = lambda x: a @ x
    
    v0 = b.copy() * 0
    _, res = yastn.lin_solver(f, b, v0, ncv=D, tol=numtol, pinv_tol=pinv_tol, hermitian=False)
    assert res < errtol

    d1, d2 = 10, 2
    a = yastn.rand(config=config, s=(-1, 1, 1, -1), D=[d1, d2, d1, d2])
    b = yastn.rand(config=config, s=(-1, 1), D=[d1, d2])
    f = lambda x: a.tensordot(x, axes=((2,3), (0,1)))
    
    v0 = b.copy() * 0
    _, res = yastn.lin_solver(f, b, v0, ncv=d1*d2, tol=numtol, pinv_tol=pinv_tol, hermitian=False)
    assert res < errtol


def test_U1(config_kwargs):
    config = yastn.make_config(sym='U1', **config_kwargs)
    config.backend.random_seed(seed=0)  # fix seed for testing
    
    leg = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yastn.rand(config=config, legs=[leg, leg.conj()])
    b = yastn.rand(config=config, legs=[leg])
    f = lambda x: a @ x
    v0 = b.copy() * 0
    _, res = yastn.lin_solver(f, b, v0, ncv=sum(leg.D), tol=numtol, pinv_tol=pinv_tol, hermitian=False)
    assert res < errtol
    
    leg1 = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(2, 3, 4))
    leg2 = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(3, 2, 2))
    a = yastn.rand(config=config, legs=[leg, leg.conj(), leg.conj(), leg])
    b = yastn.rand(config=config, legs=[leg, leg.conj()])
    f = lambda x: a.tensordot(x, axes=((2,3), (0,1)))
    
    v0 = b.copy() * 0
    _, res = yastn.lin_solver(f, b, v0, ncv=sum(leg1.D)*sum(leg2.D), tol=numtol, pinv_tol=pinv_tol, hermitian=False)
    assert res < errtol

if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "np"])
