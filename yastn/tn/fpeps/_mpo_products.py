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
""" MPO decomposition of a sum of operator products, for the n-site OE measurement. """
from functools import cmp_to_key, reduce
from ...tensor import add, tensordot, sign_canonical_order
from ...tensor.linalg import svd_with_truncation


def canonical_order(operators, sites, f_ordered):
    r"""
    Permutation bringing ``sites`` into the lattice's fermionic order and the
    sign of commuting ``operators`` along with it.

    Returns ``(sign, perm)`` with ``[sites[p] for p in perm]`` fermionically
    ordered and ``sign * O_{sites[perm[0]]} O_{sites[perm[1]]} ... ==
    O_{sites[0]} O_{sites[1]} ...``, the same sign as
    :func:`yastn.sign_canonical_order`.  Sites are picked first-in-order
    one at a time, so operators at the same site keep their relative order.
    ``perm`` depends only on ``sites``; ``sign`` also on the operators' charges.
    """
    sites = list(sites)
    before = cmp_to_key(lambda a, b: 0 if sites[a] == sites[b] else (-1 if f_ordered(sites[a], sites[b]) else 1))
    perm = tuple(sorted(range(len(sites)), key=before))  # stable: same-site operators keep their order
    sign = sign_canonical_order(*operators, sites=sites, f_ordered=f_ordered)
    return sign, perm


def sum_of_products(terms):
    r"""
    ``sum_i coeff_i * O_i0 (x) O_i1 (x) ...`` as one tensor with legs
    ``(out_0, in_0, out_1, in_1, ...)``.

    Plain outer products, no swap gates.  The operators of every term must be
    listed in the lattice's fermionic order of their sites (use
    :func:`canonical_order` to permute a term and pick up the commutation
    sign); every product must carry the same total charge.

    Parameters
    ----------
    terms : Sequence[tuple[number, Sequence[yastn.Tensor]]]
        ``(coeff, ops)`` pairs, ``ops`` one two-leg operator per site.
    """
    prods = [reduce(lambda a, b: tensordot(a, b, axes=((), ())), ops) for _, ops in terms]
    return add(*prods, amplitudes=[coeff for coeff, _ in terms])


def mpo_from_products(terms, tol=1e-12):
    r"""
    MPO decomposition of ``sum_i coeff_i * O_i0 (x) O_i1 (x) ...``.

    Sums the terms into one operator (:func:`sum_of_products`) and splits it
    site by site with truncated SVDs, so the bond between sites ``k-1`` and
    ``k`` has the true operator rank and, being a symmetric leg, carries the
    cumulative charge ``sum_{j<k} q_j`` of every summand in its sectors.
    Parity classes may be mixed freely; the measurement handles each block by
    its own bond charges.  Every MPO tensor is charge-neutral when the sum is.

    Leg order of an MPO tensor is ``(phys_out, phys_in, left_bond, right_bond)``,
    absent bonds omitted, with signature ``+1`` on the left and ``-1`` on the
    right bond.  This is the outer-product order of :func:`sum_of_products`
    re-indexed; the sweep applies no swap gate, and the measurement's crossing
    rules (``_mpo_bond_swaps`` in ``envs._env_ctm_oe_measure_network``) are
    derived for exactly this order and for a chain that runs in the lattice's
    fermionic order of the sites -- the convention :func:`sum_of_products`
    requires of ``terms`` and that the measurement enforces.  Such tensors are
    accepted by :meth:`EnvCTM.measure_nsite_exact_oe` and its norm/numerator
    variants, one per site, in place of plain two-leg operators.

    Returns ``(ops, bond_dims)``; ``bond_dims[k]`` is the total dimension of
    the bond between sites k and k+1.
    """
    nsite = len(terms[0][1])
    R = sum_of_products(terms)
    if nsite == 1:
        return [R], []
    ops, bond_dims = [], []
    for k in range(nsite - 1):
        nl = 1 if k else 0  # a left bond leg once past the first site
        # R legs: (L?, out_k, in_k, out_{k+1}, in_{k+1}, ...)
        R = R.fuse_legs(axes=(tuple(range(nl + 2)), tuple(range(nl + 2, R.ndim))))
        U, S, V = svd_with_truncation(R, axes=(0, 1), sU=-1, tol=tol)
        U = U.unfuse_legs(axes=0)  # (L?, out, in, R)
        ops.append(U.transpose(axes=(nl, nl + 1) + ((0,) if nl else ()) + (nl + 2,)))
        bond_dims.append(sum(U.get_legs(axes=U.ndim - 1).D))
        R = (S @ V).unfuse_legs(axes=1)  # (L, out_{k+1}, in_{k+1}, ...)
    ops.append(R.transpose(axes=(1, 2, 0)))
    return ops, bond_dims
