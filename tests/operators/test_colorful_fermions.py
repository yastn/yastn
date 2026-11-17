# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" Predefined operators for fermions with three color species. """
import itertools
import numpy as np
import pytest
import yastn

tol = 1e-12  # pylint: disable=invalid-name

COLORS = ('r', 'g', 'b')
SYMS = ('Z2', 'U1', 'U1xU1xU1', 'U1xU1xU1xZ2')
# common dense order of |n_r n_g n_b> used by ops.key()
ORDER = ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))


def su3_structure_constants():
    """ Totally antisymmetric structure constants f_abc of SU(3), indexed 1..8. """
    f = np.zeros((9, 9, 9))
    s3 = np.sqrt(3) / 2
    for (a, b, c), v in {(1, 2, 3): 1, (1, 4, 7): 0.5, (1, 5, 6): -0.5, (2, 4, 6): 0.5, (2, 5, 7): 0.5,
                         (3, 4, 5): 0.5, (3, 6, 7): -0.5, (4, 5, 8): s3, (6, 7, 8): s3}.items():
        for p, sgn in zip(itertools.permutations((a, b, c)), (1, -1, -1, 1, 1, -1)):
            f[p] = sgn * v
    return f


def test_colorful_fermions(config_kwargs):
    """ Generate standard operators in 8-dimensional Hilbert space for various symmetries. """
    ops_Z2 = yastn.operators.ColorfulFermions(sym='Z2', **config_kwargs)
    ops_U1xU1xU1_ind = yastn.operators.ColorfulFermions(sym='U1xU1xU1xZ2', **config_kwargs)
    ops_U1xU1xU1_dis = yastn.operators.ColorfulFermions(sym='U1xU1xU1', **config_kwargs)
    # other way to initialize
    config_U1 = yastn.make_config(fermionic=True, sym="U1", **config_kwargs)
    ops_U1 = yastn.operators.ColorfulFermions(**config_U1._asdict())

    opss = [ops_Z2, ops_U1, ops_U1xU1xU1_ind, ops_U1xU1xU1_dis]
    Is = [ops.I() for ops in opss]
    legs = [ops.space() for ops in opss]

    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
    assert all(np.allclose(I.to_numpy(), np.eye(8)) for I in Is)
    assert all(ops.config.fermionic == fs for ops, fs in zip(opss, (True, True, (False, False, False, True), True)))

    lss = [dict(enumerate(ops.I().get_legs())) for ops in opss]
    for ls, ops in zip(lss, opss):
        for ic, color in enumerate(COLORS):
            assert np.allclose(ops.n(color).to_numpy(legs=ls, key=ops.key()), np.diag([st[ic] for st in ORDER]))
    # indistinguishable species give identical dense creation operators in all symmetries
    for color in COLORS:
        ref = ops_Z2.cp(color).to_numpy(legs=lss[0], key=ops_Z2.key())
        for ls, ops in zip(lss[1:3], opss[1:3]):
            assert np.allclose(ops.cp(color).to_numpy(legs=ls, key=ops.key()), ref)

    for ops, inter_sgn in [(ops_Z2, 1), (ops_U1, 1), (ops_U1xU1xU1_ind, 1), (ops_U1xU1xU1_dis, -1)]:
        for s in COLORS:
            assert yastn.norm(ops.cp(s) @ ops.c(s) - ops.n(s)) < tol
            assert yastn.norm(ops.c(s) @ ops.c(s)) < tol
            assert yastn.norm(ops.cp(s) @ ops.cp(s)) < tol
            assert yastn.norm(ops.c(s) @ ops.cp(s) + ops.cp(s) @ ops.c(s) - ops.I()) < tol

        # anticommutator for indistinguishable; commutator for distinguishable
        for s1, s2 in itertools.combinations(COLORS, 2):
            for o1, o2 in itertools.product((ops.c, ops.cp), repeat=2):
                assert yastn.norm(o1(s1) @ o2(s2) + inter_sgn * o2(s2) @ o1(s1)) < tol

        vs = {st: ops.vec_n(st) for st in ORDER}
        for st, v in vs.items():
            for ic, color in enumerate(COLORS):
                assert yastn.norm(ops.n(color) @ v - st[ic] * v) < tol

        # |rgb> = cp_r cp_g cp_b |000>
        assert yastn.norm(ops.cp('r') @ ops.cp('g') @ ops.cp('b') @ vs[(0, 0, 0)] - vs[(1, 1, 1)]) < tol
        assert yastn.norm(ops.c('r') @ vs[(1, 1, 1)] - vs[(0, 1, 1)]) < tol
        assert yastn.norm(ops.c('g') @ vs[(1, 1, 1)] + inter_sgn * vs[(1, 0, 1)]) < tol
        assert yastn.norm(ops.c('b') @ vs[(1, 1, 1)] - vs[(1, 1, 0)]) < tol
        assert yastn.norm(ops.cp('b') @ vs[(0, 1, 0)] + inter_sgn * vs[(0, 1, 1)]) < tol
        assert yastn.norm(ops.cp('g') @ vs[(1, 0, 1)] + inter_sgn * vs[(1, 1, 1)]) < tol

    with pytest.raises(yastn.YastnError):
        yastn.operators.ColorfulFermions(sym='U1xU1')
        # For ColorfulFermions sym should be in ('Z2', 'U1', 'U1xU1xU1', 'U1xU1xU1xZ2').
    with pytest.raises(yastn.YastnError):
        yastn.operators.ColorfulFermions(sym='U1', fermionic=False)
        # For ColorfulFermions config.sym does not match config.fermionic.
    with pytest.raises(yastn.YastnError):
        yastn.operators.ColorfulFermions(sym='U1xU1xU1xZ2', fermionic=True)
        # For ColorfulFermions config.sym does not match config.fermionic.
    with pytest.raises(yastn.YastnError):
        ops_Z2.c(color='red')
        # color should be equal 'r', 'g', or 'b'.
    with pytest.raises(yastn.YastnError):
        ops_Z2.n(color=1)
        # color should be equal 'r', 'g', or 'b'.
    with pytest.raises(yastn.YastnError):
        ops_Z2.vec_n((1, 0))
        # Occupations given by val should be a tuple (n_r, n_g, n_b) with elements 0 or 1.


def test_colorful_fermions_su3(config_kwargs):
    """ Gell-Mann operators and SU(3) generators. """
    f = su3_structure_constants()
    for sym in SYMS:
        ops = yastn.operators.ColorfulFermions(sym=sym, **config_kwargs)
        ls = dict(enumerate(ops.I().get_legs()))
        ks = range(1, 9) if sym in ('Z2', 'U1') else (3, 8)
        G = {k: ops.gell_mann(k) for k in ks}
        T = {k: ops.su3(k) for k in ks}
        for k in ks:
            assert yastn.norm(G[k] - G[k].H) < tol
            assert yastn.norm(T[k] - G[k] / 2) < tol
            assert G[k].is_consistent()
        for a, b in itertools.product(ks, repeat=2):
            # [T_a, T_b] = i f_abc T_c
            diff = T[a] @ T[b] - T[b] @ T[a]
            for c in ks:
                if abs(f[a, b, c]) > 0:
                    diff = diff - 1j * f[a, b, c] * T[c]
            assert yastn.norm(diff) < tol
        # diagonal generators
        n = {c: ops.n(c) for c in COLORS}
        assert yastn.norm(G[3] - n['r'] + n['g']) < tol
        assert yastn.norm(G[8] - (n['r'] + n['g'] - 2 * n['b']) / np.sqrt(3)) < tol
        nr, ng, nb = (np.array([st[i] for st in ORDER]) for i in range(3))
        assert np.allclose(G[8].to_numpy(legs=ls, key=ops.key()), np.diag(nr + ng - 2 * nb) / np.sqrt(3))
        # Casimir sum_k T_k^2: 4/3 on singly-occupied states (fundamental representation),
        # 4/3 on doubly-occupied states (anti-fundamental), and 0 on |000> and |111> (singlets)
        if sym in ('Z2', 'U1'):
            C2 = T[1] @ T[1]
            for k in range(2, 9):
                C2 = C2 + T[k] @ T[k]
            for st in ORDER:
                v = ops.vec_n(st)
                c2 = 0 if sum(st) in (0, 3) else 4 / 3
                assert yastn.norm(C2 @ v - c2 * v) < tol
            assert G[2].yastn_dtype == 'complex128'
        else:
            for k in (1, 2, 4, 5, 6, 7):
                with pytest.raises(yastn.YastnError):
                    ops.gell_mann(k)
                    # Cannot define Gell-Mann operator for U1xU1xU1 symmetry.
        with pytest.raises(yastn.YastnError):
            ops.su3(0)
            # Gell-Mann index k should be in 1, 2, ..., 8.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
