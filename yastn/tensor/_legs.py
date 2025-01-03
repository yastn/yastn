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
""" class yastn.Leg """
from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import product, groupby
from operator import itemgetter
import numpy as np
from ._auxliary import _flatten
from ._tests import YastnError
from ..sym import sym_none
from ._merging import _Fusion, _pure_hfs_union, _fuse_hfs, _unfuse_Fusion

__all__ = ['Leg', 'leg_union', 'random_leg', 'leg_product', 'leg_undo_product']


@dataclass(frozen=True, repr=False)
class Leg:
    r"""
    :meth:`Leg` is a hashable `dataclass <https://docs.python.org/3/library/dataclasses.html>`_,
    defining a vector space.

    An abelian symmetric vector space can be specified as a direct
    sum of *plain* vector spaces (sectors), each labeled by charge :math:`t`

    .. math::
        V = \bigoplus_t V_t

    The action of abelian symmetry on elements of such space
    depends only on the charge :math:`t` of the element

    .. math::
        g \in G:\quad U(g) V = \bigoplus_t U(g)_t V_t.

    The size of individual sectors :math:`{\rm dim}(V_t)` is arbitrary.

    Parameters
    ----------
    sym : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    s : int
        Signature of the leg. Either 1 (ingoing) or -1 (outgoing).
    t : Sequence[int] | Sequence[Sequence[int]]
        List of charge sectors.
    D : Sequence[int]
        List of dimensions of corresponding charge sectors.
        The lengths :code:`len(D)` and :code:`len(t)` must be equal.
    legs: Sequence[yastn.Leg]
        Includes information about fused (sub-)legs.
    fusion: str
        Specification of how the fusion was performed.
    """
    sym: any = sym_none
    s: int = 1  # leg signature in (1, -1)
    t: tuple = ()  # leg charges
    D: tuple = ()  # and their dimensions
    fusion: str = "hard"  # 'hard', 'meta' -- tuple of meta_fusions, (in the future also None, 'sum')
    legs: tuple = () # sub-legs
    _verified: bool = False

    def __post_init__(self):
        if not self._verified:
            if not hasattr(self.sym, 'SYM_ID'):  # if config is provided
                object.__setattr__(self, "sym", self.sym.sym)  # replace is with config.sym
            if self.s not in (-1, 1):
                raise YastnError('Signature of Leg should be 1 or -1')
            object.__setattr__(self, "s", int(self.s))
            D = list(_flatten(self.D))
            t = list(_flatten(self.t))
            if not all(int(x) == x and x > 0 for x in D):
                raise YastnError('D should be a tuple of positive ints')
            if not all(int(x) == x for x in t):
                raise YastnError('Charges should be ints')
            lD, nsym = len(D), self.sym.NSYM
            if lD * nsym != len(t) or (nsym == 0 and lD != 1):
                raise YastnError('Number of provided charges and bond dimensions do not match sym.NSYM')
            #
            t = np.array(t, dtype=np.int64)
            newt = list(map(tuple, self.sym.fuse(t.reshape(lD, 1, nsym), (self.s,), self.s).tolist()))
            oldt = list(map(tuple, t.reshape(lD, nsym).tolist()))
            D = np.array(D, dtype=np.int64).reshape(lD).tolist()
            if oldt != newt:
                raise YastnError('Provided charges are outside of the natural range for specified symmetry.')
            if len(set(newt)) != len(newt):
                raise YastnError('Repeated charge index.')

            tD = dict(sorted(zip(newt, D)))
            object.__setattr__(self, "t", tuple(tD.keys()))
            object.__setattr__(self, "D", tuple(tD.values()))

            if len(self.legs) == 0:
                legs = (_Fusion(s=(self.s,)),)
                object.__setattr__(self, "legs", legs)
            object.__setattr__(self, "_verified", True)

    def __repr__(self):
        return ("Leg(sym={}, s={}, t={}, D={}, hist={})".format(self.sym, self.s, self.t, self.D, self.history()))

    def __str__(self):
        return ("Leg(sym={}, s={}, t={}, D={}, hist={})".format(self.sym, self.s, self.t, self.D, self.history()))

    def conj(self) -> yastn.Leg:
        r""" New :class:`yastn.Leg` with switched signature. """
        legs_conj = tuple(leg.conj() for leg in self.legs)
        return replace(self, s=-self.s, legs=legs_conj)

    def drop_history(self) -> yastn.Leg:
        r""" New :class:`yastn.Leg` with no information on merging history. """
        return Leg(self.sym, self.s, self.t, self.D)

    def __getitem__(self, t) -> int:
        r"""
        Size of a charge sector.

        Parameters
        ----------
        t : int | Sequence[int]
            selected charge sector
        """
        return self.D[self.t.index(t)]

    @property
    def tD(self) -> dict[tuple, int]:
        r"""
        Return charge sectors `t` and their sizes `D` as a dictionary ``{t: D}``.
        """
        return dict(zip(self.t, self.D))

    def history(self) -> str:
        """
        Show linearized representation of Leg fusion history.

        ::

            'o' marks original legs
            's' is for sum, i.e. block
            'p' is for product, i.e., fuse_legs(..., mode='hard')
            'm' is for meta-fusion

        Example
        -------
        'p(p(oo)p(oo))' corresponds to 4 original spaces.
        Two pairs are fused first, then the result gets fused.
        """
        if isinstance(self.fusion, tuple):  # meta fused
            tree = self.fusion
            op=''.join('m' if x > 1 else 'X' for x in tree)
            tmp  = _str_tree(tree, op).split('X')
            st = tmp[0]
            for leg_native, sm in zip(self.legs, tmp[1:]):
                st = st + leg_native.history()  + sm
            return st
        hf = self.legs[0]  # hard fusion
        return _str_tree(hf.tree, hf.op)

    def is_fused(self) -> bool:
        """ Return :code:`True` if the leg is a result of some fusion, and :code:`False` is it is elementary. """
        return len(self.legs) > 1 or self.legs[0].tree[0] > 1

    def unfuse_leg(self):
        return leg_undo_product(self)


def random_leg(config, s=1, n=None, sigma=1, D_total=8, legs=None, nonnegative=False) -> yastn.Leg:
    """
    Create :class:`yastn.Leg` randomly distributing bond dimensions to sectors
    according to Gaussian distribution.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
    s : int
        Signature of the leg. Either 1 (ingoing) or -1 (outgoing).
    n : int or tuple[int, ...]
        mean charge of the distribution.
    sigma : number
        standard deviation of the distribution.
    D_total : int
        total bond dimension of the leg, to be distributed to sectors.
    nonnegative : bool
        If :code:`True`, cut off negative charges.
    legs : Sequence[yastn.Leg]
        limits charges to match provided legs (e.g., in tensor with zero charge).
    """
    if config.sym.NSYM == 0:
        return Leg(config, s=s, D=(D_total,))

    if n is None:
        n = config.sym.zero()
    try:  # handle int input
        n = tuple(n)
    except TypeError:
        n = (n,)
    if len(n) != config.sym.NSYM:
        raise YastnError('len(n) is not consistent with provided symmetry.')

    an = np.array(n, dtype=np.int64)
    spanning_vectors = np.eye(len(n)) if not hasattr(config.sym, 'spanning_vectors') \
                        else np.array(config.sym.spanning_vectors)

    nvec = len(spanning_vectors)
    maxr = np.ceil(3 * sigma).astype(dtype=np.int64)

    if legs is None:
        shifts = np.zeros((2 * maxr + 1,) * nvec + (nvec,))
        for i in range(nvec):
            dims = (1,) * i + (2 * maxr + 1,) + (1,) * (nvec - i - 1)
            shifts[(slice(None),) * nvec + (i,)] = np.reshape(np.arange(-maxr, maxr+1), dims)
        ts = shifts.reshape(-1, nvec) @ spanning_vectors + an
        ts = np.round(ts).astype(dtype=np.int64)
        ts = config.sym.fuse(ts.reshape(-1, 1, config.sym.NSYM), (1,), 1)
    else:
        ss = tuple(leg.s for leg in legs)
        comb_t = list(product(*(leg.t for leg in legs)))
        lcomb_t = len(comb_t)
        comb_t = list(_flatten(comb_t))
        comb_t = np.array(comb_t, dtype=np.int64).reshape((lcomb_t, len(ss), len(n)))
        ts = config.sym.fuse(comb_t, ss, -s)

    if nonnegative:
        ts = ts[np.all(ts >= 0, axis=1)]

    uts = sorted(set(map(tuple, ts.tolist())))
    ts = np.array(uts, dtype=np.int64)

    distance = np.linalg.norm((ts - an.reshape(1, len(n))) @ spanning_vectors.T, axis=1)
    pd = np.exp(- (distance ** 2) / (2 * sigma ** 2))
    pd = pd / sum(pd)

    Ds = np.zeros(len(ts), dtype=np.int64)
    cdf = np.add.accumulate(pd).reshape(1, -1)
    # backend.rand gives distribution in [-1, 1]; subjected to backend seed fixing
    samples = (config.backend.rand(D_total, dtype='float64') + 1.) / 2.
    samples = np.array(samples).reshape(D_total, 1)
    inds = np.sum(samples > cdf, axis=1, dtype=np.int64)
    for i in inds:
        Ds[i] += 1
    Ds = Ds.tolist()
    tnonzero, Dnonzero = zip(*[(t, D) for t, D in zip(uts, Ds) if D > 0])
    return Leg(config, s=s, t=tnonzero, D=Dnonzero)


def _leg_fusions_need_mask(*legs):
    legs = list(legs)
    if all(leg.fusion == 'hard' for leg in legs):
        return any(legs[0].legs[0] != leg.legs[0] for leg in legs)
    if all(isinstance(leg.fusion, tuple) for leg in legs):
        mf = legs[0].fusion
        return any(_leg_fusions_need_mask(*(mleg.legs[n] for mleg in legs)) for n in range(mf[0]))
    raise YastnError("Mixing meta- and hard-fused legs")


def leg_product(*legs, t_allowed=None) -> yastn.Leg:
    """
    Output Leg being an outer product of a list of legs.

    Equivalent to result of :meth:`yastn.Tensor.get_legs` from tensor with fused legs
    (up to possibility that not all possible effective charges have to appear in fused tensor).

    Parameters
    ----------
    legs : yastn.Leg
        legs to compute outer product from.

    t_allowed : Sequence[Sequence[int]]
        limit effective charges to the ones provided in the list.
    """
    seff = legs[0].s
    sym = legs[0].sym

    comb_t = tuple(product(*(leg.t for leg in legs)))
    comb_t = np.array(comb_t, dtype=np.int64).reshape((len(comb_t), len(legs), sym.NSYM))
    comb_D = tuple(product(*(leg.D for leg in legs)))
    comb_D = np.array(comb_D, dtype=np.int64).reshape((len(comb_D), len(legs)))
    teff = sym.fuse(comb_t, tuple(leg.s for leg in legs), seff).tolist()
    Deff = np.prod(comb_D, axis=1, dtype=np.int64).tolist()
    tDs = sorted((tuple(x), y) for x, y in zip(teff, Deff))
    #
    tnew, Dnew = [], []
    for t, group in groupby(tDs, key=itemgetter(0)):
        if t_allowed is None or t in t_allowed:
            tnew.append(t)
            Dnew.append(sum(tD[1] for tD in group))
    tnew = tuple(tnew)
    Dnew = tuple(Dnew)
    hfs = tuple(leg.legs[0] for leg in legs)  # here assumes that all legs are 'hard fused'
    ts = tuple(leg.t for leg in legs)
    Ds = tuple(leg.D for leg in legs)
    hf = _fuse_hfs(hfs, ts, Ds, seff)
    return Leg(sym=sym, s=seff, t=tnew, D=Dnew, legs=(hf,))


def leg_undo_product(leg) -> Sequence[yastn.Leg]:
    """
    Reverse the operation of :meth:`yastn.leg_product`.

    Parameters
    ----------
    leg : yastn.Leg
        legs to compute outer product from.
    """
    hst = leg.history()
    if hst[0] in ('o', 's'):
        raise YastnError('Leg is not a result of outer_product.')
    # else hst[0] == 'p':
    ts, Ds, ss, hfs = _unfuse_Fusion(leg.legs[0])
    return tuple(Leg(sym=leg.sym, s=s, t=t, D=D, legs=(hf,))
                    for s, t, D, hf in zip(ss, ts, Ds, hfs))


def leg_union(*legs) -> yastn.Leg:
    """
    Output Leg that represent space being an union of spaces of a list of legs.

    It collects charges appearing in all provided legs.
    Their dimensions and fusion history have to match.
    """
    legs = list(legs)
    if len(legs) == 1:
        return legs.pop()
    if all(leg.fusion == 'hard' for leg in legs):
        return _leg_union(*legs)
    if all(isinstance(leg.fusion, tuple) for leg in legs):
        mf = legs[0].fusion
        if any(mf != leg.fusion for leg in legs):
            raise YastnError('Meta-fusions do not match.')
        new_nlegs = tuple(_leg_union(*(mleg.legs[n] for mleg in legs)) for n in range(mf[0]))
        nsym = legs[0].sym.NSYM
        t = tuple(sorted(set.union(*(set(leg.t) for leg in legs))))
        Dt = [tuple(leg[x[n * nsym : (n + 1) * nsym]] for n, leg in enumerate(new_nlegs)) for x in t]
        D = tuple(np.prod(Dt, axis=1, dtype=np.int64).tolist())
        return replace(legs[0], t=t, D=D, legs=new_nlegs)
    raise YastnError('All arguments of leg_union should have consistent fusions.')


def _leg_union(*legs) -> yastn.Leg:
    """
    Output _Leg that represent space being an union of spaces of a list of legs.
    """
    legs = list(legs)
    if any(leg.sym.SYM_ID != legs[0].sym.SYM_ID for leg in legs):
        raise YastnError('Provided legs have different symmetries.')
    if any(leg.s != legs[0].s for leg in legs):
        raise YastnError('Provided legs have different signatures.')
    if any(leg.legs != legs[0].legs for leg in legs):
        t, D, hf = _pure_hfs_union(legs[0].sym, [leg.t for leg in legs] ,[leg.legs[0] for leg in legs])
    else:
        tD = {}
        for leg in legs:
            for t, D in zip(leg.t, leg.D):
                if t in tD and tD[t] != D:
                    raise YastnError('Legs have inconsistent dimensions.')
                tD[t] = D
        tD = dict(sorted(tD.items()))
        t = tuple(tD.keys())
        D = tuple(tD.values())
        hf = legs[0].legs[0]
    return Leg(sym=legs[0].sym, s=legs[0].s, t=t, D=D, legs=(hf,))


def _str_tree(tree, op) -> str:
    if len(tree) == 1:
        return op
    st, op, tree = op[0] + '(', op[1:], tree[1:]
    while len(tree) > 0:
        slc = [pos for pos, node in enumerate(tree) if node == 1][tree[0] - 1] + 1
        st = st + _str_tree(tree[:slc], op[:slc])
        tree, op = tree[slc:], op[slc:]
    return st  + ')'
