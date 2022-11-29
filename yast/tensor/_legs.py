""" class yast.Leg """
from dataclasses import dataclass, replace
from itertools import product, groupby
import numpy as np
from ._auxliary import _flatten
from ._tests import YastError
from ..sym import sym_none
from ._merging import _Fusion, _pure_hfs_union, _fuse_hfs, _unfuse_Fusion

__all__ = ['Leg', 'leg_union', 'random_leg', 'leg_outer_product', 'leg_undo_product']


@dataclass(frozen=True, repr=False)
class Leg:
    r"""
    `Leg` is a hashable `dataclass <https://docs.python.org/3/library/dataclasses.html>`_,
    defining a vector space.

    An abelian symmetric vector space can be specified as a direct
    sum of `plain` vector spaces (sectors), each labeled by charge `t`

    .. math::
        V = \oplus_t V_t

    The action of abelian symmetry on elements of such space
    depends only on the charge `t` of the element

    .. math::
        g \in G:\quad U(g)V = \oplus_t U(g)_t V_t.

    The size of individual sectors :math:`dim(V_t)` is arbitrary.

    Parameters
    ----------
    sym : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    s : int
        Signature of the leg. Either 1 (ingoing) or -1 (outgoing).
    t : iterable[int] or iterable[iterable[int]]
        List of charge sectors.
    D : iterable[int]
        List of corresponding charge sector dimensions.
        The lengths `len(D)` and `len(t)` must be equal.
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
            if not hasattr(self.sym, 'SYM_ID'):
                object.__setattr__(self, "sym", self.sym.sym)
            if self.s not in (-1, 1):
                raise YastError('Signature of Leg should be 1 or -1')
            D = tuple(_flatten(self.D))
            t = tuple(_flatten(self.t))
            if not all(int(x) == x and x > 0 for x in D):
                raise YastError('D should be a tuple of positive ints')
            if not all(int(x) == x for x in t):
                raise YastError('Charges should be ints')
            if len(D) * self.sym.NSYM != len(t) or (self.sym.NSYM == 0 and len(D) != 1):
                raise YastError('Number of provided charges and bond dimensions do not match sym.NSYM')
            newt = tuple(tuple(x.flat) for x in self.sym.fuse(np.array(t).reshape((len(D), 1, self.sym.NSYM)), (self.s,), self.s))
            oldt = tuple(tuple(x.flat) for x in np.array(t).reshape(len(D), self.sym.NSYM))
            if oldt != newt:
                raise YastError('Provided charges are outside of the natural range for specified symmetry.')
            if len(set(newt)) != len(newt):
                raise YastError('Repeated charge index.')
            tD = dict(zip(newt, D))
            t =  tuple(sorted(newt))
            object.__setattr__(self, "t", t)
            object.__setattr__(self, "D", tuple(tD[x] for x in t))

            if len(self.legs) == 0:
                legs = (_Fusion(s=(self.s,)),)
                object.__setattr__(self, "legs", legs)
            object.__setattr__(self, "_verified", True)

    def __repr__(self):
        return ("Leg(sym={}, s={}, t={}, D={}, hist={})".format(self.sym, self.s, self.t, self.D, self.history()))

    def conj(self):
        r"""
        Switch the signature of Leg.

        Returns
        -------
        Leg
            Returns a new Leg with opposite signature.
        """
        legs_conj = tuple(leg.conj() for leg in self.legs)
        return replace(self, s=-self.s, legs=legs_conj)

    def __getitem__(self, t):
        r"""
        Size of a charge sector

        Parameters
        ----------
        t : int or tuple(int)
            selected charge sector

        Returns
        -------
        int
            size of the charge sector
        """
        return self.D[self.t.index(t)]

    @property
    def tD(self):
        r"""
        Returns
        -------
        dict
            charge sectors `t` and their sizes `D` as dictionary ``{t: D}``.
        """
        return dict(zip(self.t, self.D))

    def history(self):
        """
        Show str representation of Leg fusion history.
        'o' marks original legs,
        's' is for sum (block),
        'p' is for product fuse(..., mode='hard'),
        'm' is for meta-fusion.
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


def random_leg(config, s=1, n=None, sigma=1, D_total=8, legs=None, positive=False):
    """
    Creat :class:`yast.Leg`. Randomly distribute bond dimensions to sectors according to Gaussian distribution.

    Parameters
    ----------
    module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YAST configuration <tensor/configuration:yast configuration>`
    s : int
        Signature of the leg. Either 1 (ingoing) or -1 (outgoing).
    n : int or tuple
        mean charge of the distribution
    sigma : number
        standard deviation of the distribution
    D_total : int
        total bond dimension of the leg, to be distributed to sectors
    positive : bool
        If true, cut off negative charges

    Returns
    -------
    Leg : :class:`yast.Leg`
        Returns a new Leg with randomly distributed bond dimensions.
    """
    if config.sym.NSYM == 0:
        return Leg(config, s=s, D=(D_total,))

    if n is None:
        n = (0,) * config.sym.NSYM
    try:  # handle int input
        n = tuple(n)
    except TypeError:
        n = (n,)
    if len(n) != config.sym.NSYM:
        raise YastError('len(n) is not consistent with provided symmetry.')

    an = np.array(n)
    spanning_vectors = np.eye(len(n)) if not hasattr(config.sym, 'spanning_vectors') \
                        else np.array(config.sym.spanning_vectors)

    nvec = len(spanning_vectors)
    maxr = np.ceil(3 * sigma).astype(dtype=int)

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
        comb_t = np.array(comb_t, dtype=int).reshape((lcomb_t, len(ss), len(n)))
        ts = config.sym.fuse(comb_t, ss, -s)
    if positive:
        ts = ts[np.all(ts >= 0, axis=1)]

    uts = tuple(set(tuple(x.flat) for x in ts))
    ts = np.array(uts)
    distance = np.linalg.norm((ts - an.reshape(1, -1)) @ spanning_vectors.T, axis=1)

    pd = np.exp(- (distance ** 2) / (2 * sigma ** 2))
    pd = pd / sum(pd)

    Ds = np.zeros(len(ts), dtype=int)
    cdf = np.add.accumulate(pd).reshape(1, -1)
    # backend.rand gives distribution in [-1, 1]; subjected to backend seed fixing
    samples = (config.backend.rand(D_total, dtype='float64') + 1.) / 2.
    samples = np.array(samples).reshape(-1, 1)
    inds = np.sum(samples > cdf, axis=1)
    for i in inds:
        Ds[i] += 1
    tnonzero, Dnonzero = zip(*[(t, D) for t, D in zip(uts, Ds) if D > 0])

    return Leg(config, s=s, t=tnonzero, D=Dnonzero)


def _leg_fusions_need_mask(*legs):
    legs = list(legs)
    if all(leg.fusion == 'hard' for leg in legs):
        return any(legs[0].legs[0] != leg.legs[0] for leg in legs)
    if all(isinstance(leg.fusion, tuple) for leg in legs):
        mf = legs[0].fusion
        return any(_leg_fusions_need_mask(*(mleg.legs[n] for mleg in legs)) for n in range(mf[0]))


def leg_outer_product(*legs, t_allowed=None):
    """
    Output Leg being an outer product of a list of legs.
    
    Equivalent to result of :meth:`yast.Tensor.get_legs` from tensor with fused legs -
    up to possibility, that in fused tensor not all possible effective charges have to appear.

    Parameters
    ----------
    *legs : :class:`yast.Leg`
        legs to compute outer product from.

    t_allowed : list(tuple)
        limit effective charges to the ones provided in the list.

    Returns
    -------
    Leg : :class:`yast.Leg`
        Leg representing outer product of provided legs.
    """
    seff = legs[0].s
    sym = legs[0].sym

    comb_t = tuple(product(*(leg.t for leg in legs)))
    comb_t = np.array(comb_t, dtype=int).reshape((len(comb_t), len(legs), sym.NSYM))
    comb_D = tuple(product(*(leg.D for leg in legs)))
    comb_D = np.array(comb_D, dtype=int).reshape((len(comb_D), len(legs)))
    teff = sym.fuse(comb_t, tuple(leg.s for leg in legs), seff)
    Deff = np.prod(comb_D, axis=1, dtype=int)
    tDs = sorted((tuple(t.flat), D) for t, D in zip(teff, Deff))
    tnew, Dnew = [], []
    for t, group in groupby(tDs, key = lambda x: x[0]):
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


def leg_undo_product(leg):
    """
    Output Leg being an outer product of a list of legs.
    
    Equivalent to result of :meth:`yast.Tensor.get_legs` from tensor with fused legs -
    up to possibility, that in fused tensor not all possible effective charges have to appear.

    Parameters
    ----------
    leg : :class:`yast.Leg`
        legs to compute outer product from.

    Returns
    -------
    list(:class:`yast.Leg`)
        Returns a new Leg with randomly distributed bond dimensions.
    """
    hst = leg.history()
    if hst[0] in ('o', 's'):
        raise YastError('Leg is not a result of outer_product.')
    if hst[0] == 'p':
        ts, Ds, ss, hfs = _unfuse_Fusion(leg.legs[0])
        return tuple(Leg(sym=leg.sym, s=s, t=t, D=D, legs=(hf,))
                        for s, t, D, hf in zip(ss, ts, Ds, hfs))
    pass


def leg_union(*legs):
    """
    Output Leg that represent space being an union of spaces of a list of legs.
    """
    legs = list(legs)
    if len(legs) == 1:
        return legs.pop()
    if all(leg.fusion == 'hard' for leg in legs):
        return _leg_union(*legs)
    if all(isinstance(leg.fusion, tuple) for leg in legs):
        mf = legs[0].fusion
        if any(mf != leg.fusion for leg in legs):
            raise YastError('Meta-fusions do not match.')
        new_nlegs = tuple(_leg_union(*(mleg.legs[n] for mleg in legs)) for n in range(mf[0]))
        nsym = legs[0].sym.NSYM
        t = tuple(sorted(set.union(*(set(leg.t) for leg in legs))))
        Dt = [tuple(leg[x[n * nsym : (n + 1) * nsym]] for n, leg in enumerate(new_nlegs)) for x in t]
        D = tuple(np.prod(Dt, axis=1))
        return replace(legs[0], t=t, D=D, legs=new_nlegs)
    raise YastError('All arguments of leg_union should have consistent fusions.')


def _leg_union(*legs):
    """
    Output _Leg that represent space being an union of spaces of a list of legs.
    """
    legs = list(legs)
    if any(leg.sym.SYM_ID != legs[0].sym.SYM_ID for leg in legs):
        raise YastError('Provided legs have different symmetries.')
    if any(leg.s != legs[0].s for leg in legs):
        raise YastError('Provided legs have different signatures.')
    if any(leg.legs != legs[0].legs for leg in legs):
        t, D, hf = _pure_hfs_union(legs[0].sym, [leg.t for leg in legs] ,[leg.legs[0] for leg in legs])
    else:
        tD = {}
        for leg in legs:
            for t, D in zip(leg.t, leg.D):
                if t in tD and tD[t] != D:
                    raise YastError('Legs have inconsistent dimensions.')
                tD[t] = D
        t = tuple(sorted(tD.keys()))
        D = tuple(tD[x] for x in t)
        hf = legs[0].legs[0]
    return Leg(sym=legs[0].sym, s=legs[0].s, t=t, D=D, legs=(hf,))


def _str_tree(tree, op):
    if len(tree) == 1:
        return op
    st, op, tree = op[0] + '(', op[1:], tree[1:]
    while len(tree) > 0:
        slc = [pos for pos, node in enumerate(tree) if node == 1][tree[0] - 1] + 1
        st = st + _str_tree(tree[:slc], op[:slc])
        tree, op = tree[slc:], op[slc:]
    return st  + ')'
