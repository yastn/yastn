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
import yastn.tn.mps as mps
from yastn.tensor.oe_blocksparse import make_sliced_legs
from yastn.tn.fpeps.envs._env_ctm_measure import _eval_projectors

tol = 1e-10  # pylint: disable=invalid-name


def _fermions(config_kwargs):
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    return ops, ops.c(), ops.cp(), ops.n(), ops.I()  # noqa: E741 (I: identity)


def _mpo(terms, I):  # noqa: E741 (I: identity)
    """MPO of ``[(coeff, ops), ...]`` on a chain of ``len(ops)`` sites."""
    nsite = len(terms[0][1])
    return mps.generate_mpo(mps.product_mpo(I, N=nsite),
                            [mps.Hterm(co, list(range(nsite)), list(ops)) for co, ops in terms])


def test_mpo_repr(config_kwargs):
    """
    The MPO from mps.generate_mpo holds the Fock matrix elements: contracting
    its chain with no swap gate reproduces the interleaved-word tensor of the
    same sum of products, reordered to the nested word
    (out_0, .., out_3, in_3, .., in_0) with explicit swap gates.
    """
    _, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    terms = [(1.0, [cp, I, c, n]), (0.7, [cp, n, c, I]),
             (-0.3, [c, cp, c, cp]), (0.5, [n, c, cp, I])]
    H = _mpo(terms, I)
    assert len(H) == 4
    # a bond carrying both parities: the crossings are not a single fixed-charge string
    assert any(len({t[0] % 2 for t in H[k].get_legs(2).t}) == 2 for k in range(3))

    # the same operator as a plain outer product: the interleaved word, no swap gates
    T = None
    for coeff, ops_ in terms:
        P = ops_[0]
        for op in ops_[1:]:
            P = yastn.tensordot(P, op, axes=((), ()))
        T = coeff * P if T is None else T + coeff * P

    N = T  # to the nested word: move in_k past every later (out_j, in_j) pair
    for k in range(3):
        for j in range(2 * k + 2, 8):
            N = N.swap_gate(axes=(2 * k + 1, j))
    N = N.transpose(axes=(0, 2, 4, 6, 7, 5, 3, 1))

    # mps leg order (left, out, right, in); the boundary bonds are dimension one
    labels = [[5, -1, 1, -8], [1, -2, 2, -7], [2, -3, 3, -6], [3, -4, 5, -5]]
    assert (yastn.ncon([H[k] for k in range(4)], labels) - N).norm() < tol


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
    (i) term by term with plain, possibly charged, operators, through the OE
    contraction;
    (ii) as one MPO, its chain the sites as listed, for every slicing of the MPO
    bonds.
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

    kw = dict(sites=sites)
    norm = env.measure_nsite_norm_exact_oe(**kw)

    # (i) plain operators, one contraction per term
    plain = sum(coeff * env.measure_nsite_numerator_exact_oe(*term_ops, **kw)
                for coeff, term_ops in terms)
    # (ii) one MPO, its chain the sites as listed
    H = _mpo(terms, I)

    slicings = {'none': None,
                'sector': {('opb', k + 1): make_sliced_legs(H[k].get_legs(2))
                           for k in range(nsite - 1)},
                'index': {('opb', k + 1): 1 for k in range(nsite - 1)}}
    for name, unroll in slicings.items():
        val = env.measure_nsite_numerator_exact_oe(H, unroll=unroll, **kw)
        assert abs(val - plain) < tol * max(1.0, abs(plain)), (name, val, plain)
        # the norm network has no operator bonds and ignores their labels
        assert abs(env.measure_nsite_norm_exact_oe(unroll=unroll, **kw) - norm) < tol * abs(norm)

    # the chain is the sites as listed: the same operator on a rotated chain, its
    # terms placed by the positions they take there
    shuffle = (nsite - 1, *range(nsite - 1))
    pos = [shuffle.index(k) for k in range(nsite)]
    Hs = mps.generate_mpo(mps.product_mpo(I, N=nsite), [mps.Hterm(co, pos, list(t)) for co, t in terms])
    val = env.measure_nsite_numerator_exact_oe(Hs, sites=[sites[p] for p in shuffle])
    assert abs(val - plain) < tol * max(1.0, abs(plain))

    # a canonical form keeps part of the value in H.factor and in a central block
    Hc = H.shallow_copy()
    Hc.canonize_(to='last', normalize=False)
    assert Hc.factor != 1
    assert abs(env.measure_nsite_numerator_exact_oe(Hc, **kw) - plain) < tol * max(1.0, abs(plain))
    Ho = H.shallow_copy()
    Ho.orthogonalize_site_(0, to='last', normalize=False)
    assert Ho.pC is not None
    assert abs(env.measure_nsite_numerator_exact_oe(Ho, **kw) - plain) < tol * max(1.0, abs(plain))


@pytest.mark.parametrize('chain', [
    [(0, 1), (0, 0)],                  # adjacent, against the fermionic order
    [(0, 0), (1, 1)],                  # diagonal: the walk passes (1, 0)
    [(1, 0), (0, 1)],                  # anti-diagonal
    [(0, 2), (0, 0)],                  # long range, backwards
    [(1, 1), (0, 0), (0, 1)],          # permuted, with a gap
    [(0, 0), (0, 2), (0, 1)],          # the walk to (0, 2) passes the chain's own (0, 1)
    [(0, 2), (1, 0), (0, 1), (1, 2)],  # gaps, the walk passing chain sites
])
def test_mpo_chain_order(config_kwargs, chain):
    """
    The chain of an MPO is its sites as listed: in any order, not necessarily
    adjacent, the walk from one to the next passing other sites, chain sites
    included.  The value is that of the same terms as plain operators.
    """
    ops, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    env = _random_fermionic_peps(ops)
    chain = [fpeps.Site(*s) for s in chain]
    pool = {2: [(1.0, [cp, c]), (0.7, [c, cp]), (-0.3, [n, n])],
            3: [(1.0, [cp, n, c]), (0.5, [c, cp, n]), (0.2, [n, n, n])],
            4: [(1.0, [cp, c, n, n]), (0.5, [c, n, cp, n]), (0.3, [cp, cp, c, c])]}[len(chain)]
    kw = dict(sites=chain)
    for terms in [[t] for t in pool] + [pool]:  # each term alone, then all together
        plain = sum(co * env.measure_nsite_numerator_exact_oe(*t, **kw) for co, t in terms)
        val = env.measure_nsite_numerator_exact_oe(_mpo(terms, I), **kw)
        assert abs(val - plain) < tol * max(1.0, abs(plain)), terms


def test_mpo_term_order(config_kwargs):
    """A term whose operators are not listed in the chain's order: generate_mpo
    signs it from the positions, or the sign goes into the coefficient."""
    ops, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    env = _random_fermionic_peps(ops)
    term_sites = [fpeps.Site(0, 1), fpeps.Site(0, 0)]  # not in fermionic order
    term_ops = [cp, c]
    plain = env.measure_nsite_numerator_exact_oe(*term_ops, sites=term_sites)

    csites = term_sites[::-1]  # the chain in fermionic order
    Imp = mps.product_mpo(I, N=len(csites))
    kw = dict(sites=csites)

    # the term as written, its operators placed on the sorted chain
    H = mps.generate_mpo(Imp, [mps.Hterm(1.0, [1, 0], term_ops)])
    assert abs(env.measure_nsite_numerator_exact_oe(H, **kw) - plain) < tol * abs(plain)

    # the operators permuted, the sign of commuting two odd operators carried by the coefficient
    H = mps.generate_mpo(Imp, [mps.Hterm(-1.0, [0, 1], term_ops[::-1])])
    assert abs(env.measure_nsite_numerator_exact_oe(H, **kw) - plain) < tol * abs(plain)


def test_mpo_purification(config_kwargs):
    """
    Ancilla legs: on a purification the physical leg is a fusion of system and
    ancilla.  The state is cooled from infinite temperature by a few gates, so that
    system and ancilla are entangled; an MPO on any chain must give the plain sum.
    """
    ops, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    g = fpeps.SquareLattice(dims=(2, 2), boundary='infinite')
    psi = fpeps.product_peps(g, I)
    for bond in g.bonds():
        psi.apply_gate_(fpeps.gates.gate_nn_hopping(1.0, 0.2, I, c, cp, bond))
    for site in g.sites():
        psi.apply_gate_(fpeps.gates.gate_local_occupation(0.3, 0.2, I, n, site))
    assert psi[fpeps.Site(0, 0)].get_legs(axes=4).is_fused()
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 16}, max_sweeps=4)

    for chain, terms in [([(0, 0), (0, 1)], [(1.0, [cp, c]), (0.5, [n, n]), (-0.3, [c, cp])]),
                         ([(0, 1), (0, 0)], [(1.0, [cp, c]), (0.5, [n, n])]),
                         ([(0, 0), (1, 1)], [(1.0, [cp, c]), (0.7, [c, cp]), (0.5, [n, n])]),
                         ([(0, 0), (0, 2)], [(1.0, [cp, c]), (0.5, [c, cp])]),
                         ([(1, 1), (0, 0), (0, 1)], [(1.0, [cp, n, c]), (0.5, [c, cp, n]), (0.2, [n, n, n])]),
                         ([(0, 0), (1, 0), (1, 1), (0, 1)], [(1.0, [cp, c, n, n]), (0.5, [c, n, cp, n])])]:
        kw = dict(sites=[fpeps.Site(*s) for s in chain])
        plain = sum(co * env.measure_nsite_numerator_exact_oe(*t, **kw) for co, t in terms)
        val = env.measure_nsite_numerator_exact_oe(_mpo(terms, I), **kw)
        assert abs(val - plain) < tol * max(1.0, abs(plain)), chain

    # a one-site value, normalized, against measure_1site
    site = fpeps.Site(0, 0)
    num = env.measure_nsite_numerator_exact_oe(n, sites=[site])
    den = env.measure_nsite_norm_exact_oe(sites=[site])
    assert abs(num / den - env.measure_1site(n)[site]) < tol


@pytest.mark.parametrize('sites, probe, opened', [
    ([(0, 0), (1, 1)], ((0, 0), 'hlb'), ((1, 0), 'hlt')),  # the bond runs down the cut bond (0,0)-(1,0)
    ([(0, 0), (1, 1)], ((1, 0), 'hlt'), ((0, 0), 'hlb')),  # ... probed on its other side
    ([(0, 0), (1, 1)], ((1, 1), 'vbl'), ((1, 0), 'vbr')),  # ... right along the cut bond (1,0)-(1,1)
    ([(1, 1), (0, 0)], ((1, 1), 'hrt'), ((0, 1), 'hrb')),  # ... up the cut bond (1,1)-(0,1), backward
    ([(0, 1), (1, 1)], ((1, 1), 'hrt'), ((0, 1), 'hrb')),  # ... down it, forward
    ([(1, 0), (0, 0)], ((0, 0), 'hlb'), ((1, 0), 'hlt')),  # ... up the cut bond (1,0)-(0,0), backward
    ([(0, 0), (1, 1)], ((1, 1), 'hrt'), ((0, 1), 'hrb')),  # ... from (0,0) down the cut bond (0,1)-(1,1)
])
def test_cut_map_mpo(config_kwargs, sites, probe, opened):
    """
    The cut map of a window with one half-projector replaced by a probe and its
    partner side left open is linear in the operator: for an MPO it equals the
    sum of the plain per-term cut maps.  Closing it with the partner half gives
    the numerator with the pair inserted.  The cut lattice bond lies on the
    MPO bond's path or off it, walked forward or backward.
    """
    ops, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    env = _random_fermionic_peps(ops)
    for move in 'hv':  # projectors matching the current environment
        _eval_projectors(env, move, {'D_total': 8})
    sites = [fpeps.Site(*s) for s in sites]
    terms = [(1.0, [cp, c]), (0.7, [c, cp]), (-0.3, [n, n]), (0.5, [n, I])]
    H = _mpo(terms, I)
    (ps, psl), (os_, osl) = [(fpeps.Site(*s), slot) for s, slot in (probe, opened)]
    kw = dict(sites=sites, probe_site=ps, probe_slot=psl, probe=getattr(env.proj[ps], psl))

    Y_plain = None
    for coeff, term_ops in terms:
        Y = coeff * env.measure_nsite_cut_map_oe(*term_ops, **kw)
        Y_plain = Y if Y_plain is None else Y_plain + Y
    for unroll in (None, {('opb', 1): 1}):
        Y_mpo = env.measure_nsite_cut_map_oe(H, unroll=unroll, **kw)
        assert (Y_mpo - Y_plain).norm() < tol * Y_plain.norm(), unroll

    P = getattr(env.proj[os_], osl).unfuse_legs(axes=(1,))  # (env, ket, bra, thin)
    closed = yastn.tensordot(Y_mpo, P, axes=((0, 1, 2, 3), (0, 1, 2, 3))).to_number()
    num = env.measure_nsite_numerator_exact_oe(H, sites=sites, projectors={ps: (psl,), os_: (osl,)})
    assert abs(closed - num) < tol * abs(num)


def test_cylinder_two_rows(config_kwargs):
    """
    On a cylinder with two rows, a step between the rows inside the window is
    also a step across the periodic boundary; the window's own coordinates
    tell which one the measurement takes.
    """
    ops, c, cp, n, I = _fermions(config_kwargs)  # noqa: E741
    ops.config.backend.random_seed(seed=0)
    g = fpeps.SquareLattice(dims=(2, 3), boundary='cylinder')
    v = yastn.Leg(ops.config, s=1, t=(-1, 0, 1), D=(1, 2, 1))
    one = yastn.Leg(ops.config, s=1, t=(0,), D=(1,))
    tensors = {}
    for site in g.sites():
        left, right = (one if site[1] == 0 else v), (one if site[1] == 2 else v)
        A = yastn.rand(config=ops.config, legs=[v.conj(), left, v, right.conj(), ops.space()], n=0)
        tensors[site] = A / A.norm()
    env = fpeps.EnvCTM(fpeps.Peps(g, tensors=tensors), init='eye')
    env.ctmrg_(opts_svd={'D_total': 16}, max_sweeps=4)

    terms = [(1.0, [cp, c]), (0.7, [c, cp]), (-0.3, [n, n])]
    H = _mpo(terms, I)
    for sites in [[(1, 0), (0, 1)], [(1, 1), (0, 2)], [(0, 2), (1, 0)]]:
        sites = [fpeps.Site(*s) for s in sites]
        for coeff, term_ops in terms:
            ref = env.measure_nsite_exact(*term_ops, sites=sites)
            assert abs(env.measure_nsite_exact_oe(*term_ops, sites=sites) - ref) < tol
        ref = sum(coeff * env.measure_nsite_exact(*t, sites=sites) for coeff, t in terms)
        assert abs(env.measure_nsite_exact_oe(H, sites=sites) - ref) < tol
