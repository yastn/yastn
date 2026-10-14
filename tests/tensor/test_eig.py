# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
""" yastn.linalg.svd() and truncation of its singular values """
import pytest
import yastn

tol = 1e-10  #pylint: disable=invalid-name
seed = 22

torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                reason="Uses torch.autograd.gradcheck().")

def eig_combine(a):
    """ decompose and contracts tensor using svd decomposition """
    U, S, V = yastn.linalg.eig(a, axes=((3, 1), (2, 0)), sU=-1)
    US = yastn.tensordot(U, S, axes=(2, 0))
    USV = yastn.tensordot(US, V, axes=(2, 0))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert yastn.norm(a - USV) < tol  # == 0.0
    assert all(x.is_consistent() for x in (a, U, S, V))

    onlyS = yastn.linalg.eig(a, axes=((3, 1), (2, 0)), sU=-1, compute_uv=False)
    assert yastn.norm(S - onlyS) < tol

    # changes signature of new leg; and position of new leg
    U, S, V = yastn.linalg.eig(a, axes=((3, 1), (2, 0)), sU=1, nU=False, Uaxis=0, Vaxis=-1, fix_signs=True)
    US = yastn.tensordot(S, U, axes=(0, 0))
    USV = yastn.tensordot(US, V, axes=(0, 2))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert yastn.norm(a - USV) < tol  # == 0.0
    assert all(x.is_consistent() for x in (U, S, V))

    onlyS = yastn.linalg.eig(a, axes=((3, 1), (2, 0)), sU=1, nU=False, compute_uv=False)
    assert yastn.norm(S - onlyS) < tol


def test_eig_basic(config_kwargs):
    """ test svd decomposition for various symmetries """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    config_dense.backend.random_seed(seed=seed)

    a = yastn.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 11, 13, 13])
    eig_combine(a)

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, n=0, legs=legs)
    eig_combine(a)

    # test eig of empty Tensor
    for config in [config_dense, config_U1]:
        a = yastn.Tensor(config, s=(-1, 1))
        U, S, V = yastn.linalg.eig(a, axes=(0, 1))
        assert U.size == S.size == V.size == 0


def test_eig_transpose_meta(config_kwargs):
    """ test eig decomposition with meta-fuse and transpose """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (1, 2), (3, 4)))
    #
    af = a.fuse_legs(axes=((1, 2), (3, 0)), mode='meta')
    assert af.trans == (1, 2, 3, 0)
    #
    aft = af.transpose(axes=(1, 0))
    Uf, Sf, Vf = yastn.linalg.eig(aft, axes=(1, 0))
    USVf = Uf @ Sf @ Vf
    assert yastn.norm(USVf - af) < tol  # == 0.0
    #
    U, S, V = yastn.linalg.eig(a, axes=((1, 2), (3, 0)))
    Um = U.fuse_legs(axes=((0, 1), 2), mode='meta')
    Vm = V.fuse_legs(axes=(0, (1, 2)), mode='meta')
    assert yastn.norm(Uf - Um) < tol  # == 0.0
    assert yastn.norm(Sf - S) < tol  # == 0.0
    assert yastn.norm(Vf - Vm) < tol  # == 0.0


@pytest.mark.xfail(reason='Raising error might depend on linear algebra backend', strict=False)
def test_eig_degeneracy_fail(config_kwargs):
    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 1)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 1)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 1)),
            yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 1))]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    with pytest.raises(ValueError):
        eig_combine(a)
        # fails to biorthogonalize due to degeneracies in eigenspaces


def test_eig_complex(config_kwargs):
    """ test eig decomposition and dtype propagation """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    config_dense.backend.random_seed(seed=seed)

    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=[11, 11, 11, 11], dtype='complex128')
    U, S, V = yastn.linalg.eig(a, axes=((0, 1), (2, 3)), sU=-1)
    assert U.yastn_dtype == 'complex128'
    assert S.yastn_dtype == 'complex128'

    US = yastn.tensordot(U, S, axes=(2, 0))  # here tensordot goes though broadcasting
    USV = yastn.tensordot(US, V, axes=(2, 0))
    assert yastn.norm(a - USV) < tol  # == 0.0

    SS = yastn.diag(S)
    assert SS.yastn_dtype == 'complex128'
    US = yastn.tensordot(U, SS, axes=(2, 0))
    USV = yastn.tensordot(US, V, axes=(2, 0))
    assert yastn.norm(a - USV) < tol  # == 0.0

    eig_combine(a)


def test_eig_exceptions(config_kwargs):
    """ raising exceptions by eig(), and some corner cases. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(0, 1), D=(5, 6)),
            yastn.Leg(config_U1, s=-1, t=(0, 1), D=(5, 6)),]
    a = yastn.rand(config=config_U1, legs=legs)

    with pytest.raises(yastn.YastnError):
        _ = yastn.eig(a, axes=((0, 1), 2), policy='wrong_policy')
        # svd policy should be one of (`lowrank`, `fullrank`)
    with pytest.raises(yastn.YastnError):
        _ = yastn.eig(a, axes=((0, 1), 2), policy='lowrank')
        # lowrank policy in svd requires passing argument D_block

if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"]) #, "--backend", "torch"])
