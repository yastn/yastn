from __future__ import annotations
import numpy as np
import numbers
from typing import NamedTuple
from ... import ones, zeros, ncon, Leg, YastnError, Tensor, block, svd_with_truncation
from ._mps import Mpo


class Hterm(NamedTuple):
    r"""
    Defines a product operator :math:`O = amplitude \times \bigotimes_i o_i` of local operators :math:`o_i`.
    Local operators that are not explicitly specified are assumed to be identity operators.

    If operators are fermionic, execution of :meth:`swap gates<yastn.swap_gate>` enforces fermionic order,
    with the last operator in `operators` acting first.

    Parameters
    ----------
    amplitude: number
        numerical multiplier in front of the operator product.
    positions: Sequence[int]
        positions of the local operators :math:`o_i` in the product different than identity.
    operators: Sequence[yastn.Tensor]
        local operators in the product that are different than the identity.
        *i*-th operator is acting at `positions[i]`.
        Each operator should have `ndim=2` and signature `s=(+1, -1).`
    """
    amplitude: float = 1.0
    positions: tuple = ()
    operators: tuple = ()


def generate_product_mpo_from_Hterm(I, term, amplitude=True) -> yastn.tn.mps.MpsMpo:
    r"""
    Apply local operators specified by term in :class:`Hterm` to the MPO `I`.

    MPO `I` is presumed to be an identity.
    Apply swap_gates to introduce fermionic degrees of freedom
    (fermionic order is the same as the order of sites in Mps).
    With this respect, local operators specified in term.operators are applied starting with the last element,
    i.e., from right to left.

    Parameters
    ----------
    term: yastn.tn.mps.Hterm

    amplitude: bool
        if True, includes term.amplitude in MPO.
    """
    single_mpo = I.copy()

    try:
        if len(term.positions) != len(term.operators):
            raise YastnError("Hterm: numbers of positions and operators do not match. ")
    except TypeError:
        raise YastnError("Hterm: positions and operators should be provided as lists or tuples.")

    for site, op in zip(term.positions[::-1], term.operators[::-1]):
        if site < 0 or site > I.N or not isinstance(site, numbers.Integral):
            raise YastnError("position in Hterm should be in 0, 1, ..., N-1 ")
        if not op.s == (1, -1):
            raise YastnError("operator in Hterm should be a matrix with signature (1, -1)")
        op = op.add_leg(axis=0, s=-1)
        leg = op.get_legs(axes=0)
        one = ones(config=op.config, legs=(leg, leg.conj()))
        temp = ncon([op, single_mpo[site]], [(-1, -2, 1), (-0, 1, -3, -4)])
        single_mpo[site] = temp.fuse_legs(axes=((0, 1), 2, 3, 4), mode='hard')
        for n in range(site):
            temp = ncon([single_mpo[n], one], [(-0, -2, -3, -5), (-1, -4)])
            temp = temp.swap_gate(axes=(1, 2))
            single_mpo[n] = temp.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='hard')
    for n in single_mpo.sweep():
        single_mpo[n] = single_mpo[n].drop_leg_history(axes=(0, 2))
    if amplitude:
        single_mpo[0] = term.amplitude * single_mpo[0]
    return single_mpo


class GenerateMpoTemplate(NamedTuple):
    config: NamedTuple = None
    basis: list = None
    trans: list = None
    tleft: list = None


def generate_mpo_preprocessing(I, terms, return_amplitudes=False) -> GenerateMpoTemplate | tuple[GenerateMpoTemplate, list[float]]:
    r"""
    Precompute an amplitude-independent template that is used
    to generate MPO with :meth:`mps.generate_mpo_fast<yastn.tn.mps.generate_mpo_fast>`

    Parameters
    ----------
    terms: list of :class:`Hterm`
        product operators making up the MPO.
    I: yastn.tn.mps.MpsMpo
        identity MPO.
    return_amplitudes: bool
        If True, apart from template return also amplitudes = [term.amplitude for term in terms].
    """
    H1s = [generate_product_mpo_from_Hterm(I, term, amplitude=False) for term in terms]
    cfg = H1s[0][0].config
    mapH = np.zeros((len(H1s), I.N), dtype=int)

    basis, t1bs, t2bs, tfbs, ifbs = [], [], [], [], []
    for n in I.sweep():
        base = []
        for m, H1 in enumerate(H1s):
            ind = next((ind for ind, v in enumerate(base) if (v - H1[n]).norm() < 1e-13), None)  # site-tensors differing by less then 1e-13 are considered identical
            if ind is None:
                ind = len(base)
                base.append(H1[n])
            mapH[m, n] = ind

        t1bs.append([ten.get_legs(axes=0).t[0] for ten in base])
        t2bs.append([ten.get_legs(axes=2).t[0] for ten in base])
        base = [ten.fuse_legs(axes=((0, 2), 1, 3)).drop_leg_history() for ten in base]
        tfb = [ten.get_legs(axes=0).t[0] for ten in base]
        tfbs.append(tfb)
        ifbs.append([sum(x == y for x in tfb[:i]) for i, y in enumerate(tfb)])
        base = block(dict(enumerate(base)), common_legs=(1, 2)).drop_leg_history()
        basis.append(base)

    tleft = [t1bs[0][i] for i in mapH[:, 0]]

    trans = []
    for n in I.sweep():
        mapH0 = mapH[:, 0]
        mapH, rind, iind = np.unique(mapH[:, 1:], axis=0, return_index=True, return_inverse=True)

        i2bs = {t: {} for t in t2bs[n]}
        for ii, rr in enumerate(rind):
            i2bs[t2bs[n][mapH0[rr]]][ii] = len(i2bs[t2bs[n][mapH0[rr]]])

        i1bs = {t: 0 for t in t1bs[n]}
        for ii, rr in enumerate(mapH0):
            i1bs[t1bs[n][rr]] += 1

        leg1 = Leg(cfg, s=-1, t=list(i1bs.keys()), D=list(i1bs.values()))
        leg2 = basis[n].get_legs(axes=0).conj()
        leg3 = Leg(cfg, s=1, t=list(i2bs.keys()), D=[len(x) for x in i2bs.values()])
        tran = zeros(config=cfg, legs=[leg1, leg2, leg3])

        li = {x: -1 for x in t1bs[n]}
        for bl, br in zip(mapH0, iind):
            lt = t1bs[n][bl]
            li[lt] += 1
            ft = tfbs[n][bl]
            fi = ifbs[n][bl]
            rt = t2bs[n][bl]
            ri = i2bs[rt][br]
            tran[lt + ft + rt][li[lt], fi, ri] += 1
        trans.append(tran)

    template = GenerateMpoTemplate(config=cfg, basis=basis, trans=trans, tleft=tleft)
    if return_amplitudes:
        amplitudes = [term.amplitude for term in terms]
        return template, amplitudes
    return template


def generate_mpo_fast(template, amplitudes, opts=None) -> yastn.tn.mps.MpsMpo:
    r"""
    Fast generation of MPOs representing `Sequence[Hterm]` that differ only in amplitudes.

    Preprocessing in :meth:`yastn.tn.mps.generate_mpo` might be slow.
    When only amplitudes in Hterms are changing, e.g., for time-dependent Hamiltonian,
    MPO generation can be significantly speeded up by precalculating and reusing amplitude-independent `template`.
    The latter is done with :meth:`yastn.tn.mps.generate_mpo_preprocessing`.

    Parameters
    ----------
    template: NamedTuple
        Calculated with :meth:`mps.generate_mpo_preprocessing<yastn.tn.mps.generate_mpo_preprocessing>`.
    amplidutes: Sequence[numbers]
        List of amplitudes that would appear in :class:`Hterm`.
        The order of the list should match the order of `Sequence[Hterm]`
        supplemented to :meth:`mps.generate_mpo_preprocessing<yastn.tn.mps.generate_mpo_preprocessing>`.
    opts: dict
        Options passed to :meth:`yastn.linalg.svd_with_truncation`.
        The function employs SVD while compressing the MPO bond dimensions.
        Default `None` sets truncation `tol` close to the numerical precision,
        which typically results in lossless compression.
    """
    if opts is None:
        opts = {'tol': 1e-13}

    Js = {}
    for a, t in zip(amplitudes, template.tleft):
        if t in Js:
            Js[t].append(a)
        else:
            Js[t] = [a]

    dtype = 'complex128' if any(isinstance(a, complex) for a in amplitudes) else template.config.default_dtype
    J = Tensor(config=template.config, s=(-1, 1), dtype=dtype)
    for t, val in Js.items():
        J.set_block(ts=(t, t), Ds=(1, len(val)), val=val)

    M = Mpo(len(template.basis))
    for n in M.sweep():
        nJ = J @ template.trans[n]
        nJ = ncon([nJ, template.basis[n]], [[0, 1, -3], [1, -1, -2]])
        if n < M.last:
            nJ, S, V = svd_with_truncation(nJ, axes=((0, 1, 2), 3), sU=1, **opts)
            nS = S.norm()
            nJ = nS * nJ
            J = (S / nS) @ V
        M[n] = nJ.transpose(axes=(0, 1, 3, 2))
    return M


def generate_mpo(I, terms, opts=None) -> yastn.tn.mps.MpsMpo:
    r"""
    Generate MPO provided a list of :class:`Hterm`\-s and identity MPO `I`.

    It employs :meth:`mps.generate_mpo_preprocessing<yastn.tn.mps.generate_mpo_preprocessing>`
    and :meth:`mps.generate_mpo_fast<yastn.tn.mps.generate_mpo_fast>`,
    but without bookkeeping the template obtained in the preprocessing step.
    The latter would speed up the generation of MPOs for different amplitudes
    in front of the same set of product operators.

    Parameters
    ----------
    terms: Sequence[yastn.tn.mps.Hterm]
        Product operators making up MPO.
    I: yastn.tn.mps.MpsMpo
        Identity MPO.
    opts: dict
        Options passed to :meth:`yastn.linalg.svd_with_truncation`.
        The function employs SVD while compressing the MPO bond dimensions.
        Default `None` sets truncation `tol` close to the numerical precision,
        which typically results in lossless compression.
    """
    template, amplitudes = generate_mpo_preprocessing(I, terms, return_amplitudes=True)
    return generate_mpo_fast(template, amplitudes, opts=opts)
