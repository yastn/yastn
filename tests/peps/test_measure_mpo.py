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
""" MPO tensors in the n-site OE measurement: fermionic signs without any model. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tensor.oe_blocksparse import make_sliced_legs
from yastn.tn.fpeps.envs._env_ctm_measure import _eval_projectors

tol = 1e-10  # pylint: disable=invalid-name


def _fermions(config_kwargs):
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    return ops, ops.c(), ops.cp(), ops.n(), ops.I()  # noqa: E741 (I: identity)


def test_mpo_repr(config_kwargs):
    """
    The MPO built by mpo_from_products from a sum of plain outer products holds
    the bare (interleaved-word) entries.  Contracting the chain with exactly one
    swap pair per bond, bond_k x in_k, reproduces the tensor whose entries are
    the true Fock matrix elements, i.e. the interleaved tensor reordered to the
    nested word (out_0, .., out_3, in_3, .., in_0) with explicit swap gates.
    """
    _, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    terms = [(1.0, [cp, I, c, n]), (0.7, [cp, n, c, I]),
             (-0.3, [c, cp, c, cp]), (0.5, [n, c, cp, I])]
    T = fpeps.sum_of_products(terms)  # (o0, i0, o1, i1, o2, i2, o3, i3)
    mpo, bond_dims = fpeps.mpo_from_products(terms)
    assert len(mpo) == 4 and len(bond_dims) == 3
    assert all(op.n == (0,) for op in mpo)
    # a bond carrying both parities: the per-block crossings are not a
    # single fixed-charge string
    assert any(len({t[0] % 2 for t in op.get_legs(op.ndim - 1).t}) == 2 for op in mpo[:-1])

    N = T
    for k in range(3):  # move in_k past every later (out_j, in_j) pair
        for j in range(2 * k + 2, 8):
            N = N.swap_gate(axes=(2 * k + 1, j))
    N = N.transpose(axes=(0, 2, 4, 6, 7, 5, 3, 1))

    labels = [[-1, -8, 1], [-2, -7, 1, 2], [-3, -6, 2, 3], [-4, -5, 3]]

    def contract(swap):
        return yastn.ncon(mpo, labels, swap=swap)

    # open legs in the nested word (o0, .., o3, i3, .., i0) = labels -1 .. -8,
    # bonds 1, 2, 3; bond k crosses only the in leg of tensor k, i.e. the swap
    # pairs are (1, -8), (2, -7), (3, -6) (diagram: "Swap gates of MPO" in
    # docs/source/fpeps/measurement_oe.rst)
    assert (contract([(1, -8), (2, -7), (3, -6)]) - N).norm() < tol


def _random_fermionic_peps(ops, seed=0):
    """2x2 unit cell of random U(1) tensors on the infinite lattice."""
    ops.config.backend.random_seed(seed)
    g = fpeps.SquareLattice(dims=(2, 2), boundary='infinite')
    v = yastn.Leg(ops.config, s=1, t=(-1, 0, 1), D=(1, 2, 1))
    legs = [v.conj(), v, v, v.conj(), ops.space()]  # t, l, b, r, p
    tensors = {}
    for site in g.sites():
        A = yastn.rand(config=ops.config, legs=legs, n=0)
        tensors[site] = A / A.norm()
    psi = fpeps.Peps(g, tensors=tensors)
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 8}, max_sweeps=2)
    return env


@pytest.mark.parametrize('sites', [
    [(0, 0), (0, 1), (1, 1)],            # three sites, 2x2 window, canonical order
    [(1, 1), (0, 0), (0, 1)],            # same sites, scrambled listing order
    [(0, 0), (1, 1)],                    # diagonal pair
    [(0, 0), (0, 1), (1, 0), (1, 1)],    # full 2x2
    [(1, 1), (0, 0), (1, 0), (0, 1)],    # full 2x2, scrambled listing order
    [(0, 0), (1, 2)],                    # 2x3 window, sites at opposite corners
])
def test_measure_mpo(config_kwargs, sites):
    """
    A sum of operator products measured two ways gives the same number:
    (i) term by term with plain, possibly charged, operators (Jordan-Wigner
    strings drawn by the measurement), through the OE contraction;
    (ii) as one MPO, for every slicing of the MPO bonds.
    The chain must run in the lattice's fermionic order of the sites, so each
    term is permuted with canonical_order first, the commutation sign going
    into its coefficient; a chain in any other order is rejected.
    """
    ops, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    env = _random_fermionic_peps(ops)
    sites = [fpeps.Site(*s) for s in sites]
    nsite = len(sites)

    pool = {2: [[cp, c], [c, cp], [n, n], [n, I], [I, n]],
            3: [[cp, c, n], [c, cp, I], [n, n, n], [cp, n, c], [c, n, cp], [I, cp, c]],
            4: [[cp, c, n, I], [c, cp, cp, c], [n, I, n, n], [cp, n, I, c], [I, cp, c, n]]}[nsite]
    coeffs = [1.0, 0.7, -0.3, 0.5, 0.2, -0.6][:len(pool)]
    terms = list(zip(coeffs, pool))

    kw = dict(sites=sites, separate_layers=True)
    norm = env.measure_nsite_norm_exact_oe(**kw)

    # (i) plain operators, one contraction per term
    plain = sum(coeff * env.measure_nsite_numerator_exact_oe(*term_ops, **kw)
                for coeff, term_ops in terms)
    # (ii) one MPO on the canonically ordered chain
    perm = None
    ordered = []
    for coeff, term_ops in terms:
        # permute the operators into its canonical ordering, recording the extra sign
        sign, perm = fpeps.canonical_order(term_ops, sites, env.f_ordered)
        ordered.append((sign * coeff, [term_ops[p] for p in perm]))
    csites = [sites[p] for p in perm]
    mpo, bond_dims = fpeps.mpo_from_products(ordered)
    kw = dict(sites=csites, separate_layers=True)

    slicings = {'none': None,
                'sector': {('opb', k + 1): make_sliced_legs(op.get_legs(op.ndim - 1))
                           for k, op in enumerate(mpo[:-1])},
                'index': {('opb', k + 1): 1 for k in range(nsite - 1)}}
    for name, unroll in slicings.items():
        val = env.measure_nsite_numerator_exact_oe(*mpo, unroll=unroll, **kw)
        assert abs(val - plain) < tol * max(1.0, abs(plain)), (name, val, plain)
        # the norm network has no operator bonds and ignores their labels
        assert abs(env.measure_nsite_norm_exact_oe(unroll=unroll, **kw) - norm) < tol * abs(norm)

    # the contracted builder absorbs the operator into the site
    # tensor and cannot draw the bond crossings: MPO tensors fall back to
    # separate layers with a warning
    with pytest.warns(UserWarning, match="separate_layers"):
        val = env.measure_nsite_numerator_exact_oe(*mpo, sites=csites, separate_layers=False)
    assert abs(val - plain) < tol * max(1.0, abs(plain))


@pytest.mark.parametrize('probe, opened', [
    (((0, 0), 'hlb'), ((1, 0), 'hlt')),  # the string of (1, 1) crosses an open leg
    (((1, 0), 'hlt'), ((0, 0), 'hlb')),  # ... and a leg renamed by the probe
])
def test_cut_map_mpo(config_kwargs, probe, opened):
    """
    The cut map of a window with one half-projector replaced by a probe and its
    partner side left open is linear in the operator: for an MPO it equals the
    sum of the plain per-term cut maps.  Closing it with the partner half gives
    the numerator with the pair inserted.  Probing both sides of the cut puts
    the MPO bond crossings once on the probe side and once on the open legs.
    """
    ops, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    env = _random_fermionic_peps(ops)
    for move in 'hv':  # projectors matching the current environment
        _eval_projectors(env, move, {'D_total': 8})
    sites = [fpeps.Site(0, 0), fpeps.Site(1, 1)]  # fermionic order
    terms = [(1.0, [cp, c]), (0.7, [c, cp]), (-0.3, [n, n]), (0.5, [n, I])]
    mpo, _ = fpeps.mpo_from_products(terms)
    (ps, psl), (os_, osl) = [(fpeps.Site(*s), slot) for s, slot in (probe, opened)]
    kw = dict(sites=sites, probe_site=ps, probe_slot=psl, probe=getattr(env.proj[ps], psl))

    Y_plain = None
    for coeff, term_ops in terms:
        Y = coeff * env.measure_nsite_cut_map_oe(*term_ops, **kw)
        Y_plain = Y if Y_plain is None else Y_plain + Y
    for unroll in (None, {('opb', 1): 1}):
        Y_mpo = env.measure_nsite_cut_map_oe(*mpo, unroll=unroll, **kw)
        assert (Y_mpo - Y_plain).norm() < tol * Y_plain.norm(), unroll

    P = getattr(env.proj[os_], osl).unfuse_legs(axes=(1,))  # (env, ket, bra, thin)
    closed = yastn.tensordot(Y_mpo, P, axes=((0, 1, 2, 3), (0, 1, 2, 3))).to_number()
    num = env.measure_nsite_numerator_exact_oe(*mpo, sites=sites, projectors={ps: (psl,), os_: (osl,)})
    assert abs(closed - num) < tol * abs(num)
