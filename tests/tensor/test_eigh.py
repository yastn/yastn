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
""" yastn.linalg.eigh() """
from itertools import product
import pytest
import yastn
from ._nonabelian_utils import matrix_tensor

tol = 1e-9  #pylint: disable=invalid-name


def _run_eigh_nonabelian(config_kwargs, sym):
    config = yastn.make_config(sym=sym, **config_kwargs)
    a = matrix_tensor(config)
    a = a + a.H
    S, U = yastn.eigh(a, axes=(0, 1))
    assert (U @ S @ U.H - a).norm() < tol


def test_eigh_SU2(config_kwargs):
    _run_eigh_nonabelian(config_kwargs, 'SU2')


def test_eigh_SU2xU1(config_kwargs):
    _run_eigh_nonabelian(config_kwargs, 'SU2xU1')


def test_eigh_block_lanczos_zero_block(config_kwargs):
    """Block Lanczos does not pass an all-zero sector to ARPACK."""
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config_U1, s=1, t=(0, 1), D=(72, 72))
    a = yastn.rand(config=config_U1, legs=[leg.conj(), leg])
    a = a + a.transpose(axes=(1, 0)).conj()
    a.set_block(ts=(0, 0), val='zeros')

    S, U = yastn.eigh(a, axes=(0, 1), policy='block_lanczos', D_block=1)

    assert S.get_blocks_charge() == ((1, 1),)
    assert U.is_consistent()
    assert S.is_consistent()


def eigh_combine(a,which='SR'):
    """ decompose and contracts Hermitian tensor using eigh decomposition """
    a2 = yastn.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))  # makes Hermitian matrix from a
    S, U = yastn.linalg.eigh(a2, axes=((0, 1), (2, 3)), which=which)
    US = yastn.tensordot(U, S, axes=(2, 0))
    USU = yastn.tensordot(US, U, axes=(2, 2), conj=(0, 1))
    assert yastn.norm(a2 - USU) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()

    # changes signature of new leg; and position of new leg
    S, U = yastn.eigh(a2, axes=((0, 1), (2, 3)), Uaxis=0, sU=-1, which=which)
    US = yastn.tensordot(S, U, axes=(0, 0))
    USU = yastn.tensordot(US, U, axes=(0, 0), conj=(0, 1))
    assert yastn.norm(a2 - USU) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()


@pytest.mark.parametrize("which", ['SR', 'LR', 'LM', 'SM'])
def test_eigh_basic(config_kwargs,which):
    """ test eigh decomposition for various symmetries """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    eigh_combine(a,which=which)

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(5, 6, 7)),
            yastn.Leg(config_U1, s=1, t=(-2, -1, 0, 1, 2), D=(6, 5, 4, 3, 2)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))]
    a = yastn.rand(config=config_U1, n=1, legs=legs)
    eigh_combine(a,which=which)

    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(2, 3, 4, 5)),
            yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(5, 4, 3, 2)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(3, 4, 5, 6)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 4))]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    eigh_combine(a,which=which)

    # test eigh of empty Tensor
    for config in [config_dense, config_U1, config_Z2xU1]:
        a = yastn.Tensor(config, s=(1, -1, -1, 1))
        S, U = yastn.linalg.eigh(a, axes=((0, 1), (2, 3)))
        assert S.size == U.size == 0


@pytest.mark.parametrize("which", ['SR', 'LR', 'LM', 'SM'])
def test_eigh_Z3(config_kwargs,which):
    # Z3
    config_Z3 = yastn.make_config(sym='Z3', **config_kwargs)
    s0set = (-1, 1)
    sUset = (-1, 1)
    for s, sU in product(s0set, sUset):
        leg = yastn.Leg(config_Z3, s=s, t=(0, 1, 2), D=(2, 5, 3))
        a = yastn.rand(config=config_Z3,legs=[leg, leg.conj()], dtype='complex128')
        a = a + a.transpose(axes=(1, 0)).conj()
        S, U = yastn.linalg.eigh(a, axes=(0, 1), sU=sU, which=which)
        assert yastn.norm(a - U @ S @ U.transpose(axes=(1, 0)).conj()) < tol  # == 0.0
        assert U.is_consistent()
        assert S.is_consistent()


def test_eigh_transpose_meta(config_kwargs):
    """ test eigh decomposition with meta-fuse and transpose """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8)))
    #
    aH = yastn.tensordot(a, a.conj(), axes=(3, 3))
    aHf = aH.fuse_legs(axes=((0, 2), 1, (3, 5), 4), mode='meta')
    aHff = aHf.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
    assert aHff.trans == (0, 2, 1, 3, 5, 4)
    Sff, Uff = yastn.linalg.eigh(aHff, axes=(0, 1))
    USUff = Uff @ Sff @ Uff.H
    assert yastn.norm(USUff - aHff) < tol  # == 0.0


def test_eigh_exceptions(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0), D=(2, 3)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0), D=(5, 6))]
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config_U1, n=2, legs=legs)
        _ = yastn.eigh(a, axes=(0, 1))
        # eigh requires tensor charge to be zero.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config_U1, n=0, legs=legs)
        _ = yastn.eigh(a, axes=(0, 1))
        # Tensor likely not Hermitian. Legs of effective square blocks not match.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])


def test_eigh_retries_transient_cusolver_failure(config_kwargs, monkeypatch):
    """ A transient cuSOLVER failure must be retried, not abort a long evolution run. """
    if config_kwargs.get('backend') != 'torch':
        pytest.skip('backend_torch.eigh is exercised with the torch backend')
    import torch

    genuine = torch.linalg.eigh
    calls = []

    def flaky(x, *args, **kwargs):
        calls.append(tuple(x.shape))
        if len(calls) == 1:
            raise RuntimeError('cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling '
                               'cusolverDnXsyevd')
        return genuine(x, *args, **kwargs)

    config = yastn.make_config(sym='SU2xU1', **config_kwargs)
    a = matrix_tensor(config)
    a = a + a.H

    monkeypatch.setattr(torch.linalg, 'eigh', flaky)
    S, U = yastn.eigh(a, axes=(0, 1))
    assert len(calls) > 1, 'the failing solve was not retried'
    assert (U @ S @ U.H - a).norm() < tol

    def broken(x, *args, **kwargs):
        raise RuntimeError('some unrelated failure')

    monkeypatch.setattr(torch.linalg, 'eigh', broken)
    with pytest.raises(RuntimeError, match='some unrelated failure'):
        yastn.eigh(a, axes=(0, 1))


def test_with_cusolver_retry_fallback_chain():
    """ The fallback chain is walked: failing call -> fresh copy -> (MAGMA) -> CPU. """
    torch = pytest.importorskip('torch')  # the np-only CI environment has no torch
    from yastn.backend.backend_torch import _with_cusolver_retry

    x = torch.eye(2)
    calls = []

    def flaky(t):
        calls.append(t)
        if len(calls) == 1:
            raise RuntimeError('cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR')
        return t

    assert _with_cusolver_retry(flaky, x) is not None
    assert len(calls) == 2  # the failing solve was retried on a copy

    def always_broken(t):
        calls.append(t)
        raise RuntimeError('cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR')

    calls.clear()
    with pytest.raises(RuntimeError):
        _with_cusolver_retry(always_broken, x)
    assert len(calls) == 2  # original and fresh copy; a CPU input has no CPU fallback left

    calls.clear()

    def unrelated(t):
        calls.append(t)
        raise RuntimeError('some unrelated failure')

    with pytest.raises(RuntimeError, match='some unrelated failure'):
        _with_cusolver_retry(unrelated, x)
    assert len(calls) == 1  # not a cuSOLVER failure: no retry


def test_eigh_recovers_from_cusolver_failure():
    """
    The recorded gauge metric of a U1 fPEPS run: rank 326/696, condition number 7e17 and 609
    eigenvalue gaps below 1e-10. cuSOLVER's syevd returns CUSOLVER_STATUS_INTERNAL_ERROR for
    it on Hopper (deterministically, also after a fresh copy) while MAGMA and LAPACK solve the
    same block, so backend_torch's fallback must return a usable decomposition.
    """
    import os

    import numpy as np

    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('the recorded failure is a CUDA cuSOLVER failure')
    from yastn.backend.backend_torch import _with_cusolver_retry

    fname = os.path.join(os.path.dirname(__file__), 'inputs/cusolver_break_696.npy')
    a = torch.from_numpy(np.load(fname)).cuda()

    # Whether the raw cuSOLVER call fails depends on the CUDA stack; when it does, it must be
    # a cuSOLVER error (that is what the fallback keys on), and either way the guarded call
    # has to return a valid decomposition.
    try:
        torch.linalg.eigh(a)
        print('note: raw torch.linalg.eigh succeeded on this block in this environment')
    except RuntimeError as exc:
        assert 'cusolver' in str(exc).lower()

    S, U = _with_cusolver_retry(torch.linalg.eigh, a)
    assert S.shape == (a.shape[0],) and U.shape == a.shape
    assert torch.isfinite(S).all() and torch.isfinite(U).all()
    assert torch.all(S[1:] >= S[:-1])  # eigenvalues are returned in ascending order
    assert torch.allclose(S.cpu(), torch.linalg.eigvalsh(a.cpu()), atol=1e-12)
