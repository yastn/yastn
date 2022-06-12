from dataclasses import dataclass, replace
import numpy as np
from ._auxliary import _flatten
from ._tests import YastError
from ..sym import sym_none
from ._merging import _Fusion, _hfs_union
#from ...Initialize import zeros

__all__ = ['Leg', 'leg_union']

    # r"""
    # Abelian symmetric vector space - leg of a tensor.
    # .. py:attribute:: sym : symmetry class or compatible object
    #     Specify abelian symmetry. To see how YAST defines symmetries,
    #     see :class:`yast.sym.sym_abelian`.
    #     Defaults to ``yast.sym.sym_none``, effectively a dense tensor.
    # .. py:attribute:: s
    #         Signature of the leg. Either 1 (ingoing) or -1 (outgoing).
    # .. py:attribute:: t : iterable[int] or iterable[iterable[int]]
    #     List of charge sectors
    # .. py:attribute:: D : iterable[int]
    #     List of corresponding charge sector dimensions.
    # .. py:attribute:: fusion
    #     Information about type of fusion
    # """

@dataclass(frozen=True)
class Leg:
    r"""
    `Leg` is a hashable `dataclass <https://docs.python.org/3/library/dataclasses.html>`_,
    defining a vector space.

    An abelian symmetric vector space can be specified as a direct
    sum of `plain` vector spaces (sectors), each labeled by charge `t`

    .. math::
        V = \oplus_t V_t

    The action of abelian symmetry on elements of such space
    depends only on the charge `t` of the element.

    .. math::
        g \in G:\quad U(g)V = \oplus_t U(g)_t V_t

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

    def conj(self):
        """
        Switch signature of Leg.

        Returns
        -------
        leg : Leg
            Returns new Leg with opposite signature.
        """
        legs_conj = tuple(leg.conj() for leg in self.legs)
        return replace(self, s=-self.s, legs=legs_conj)
    
    def __getitem__(self, key):
        """ Dimension of the space with charge given by key."""
        return self.D[self.t.index(key)]


def _leg_fusions_need_mask(*legs):
    legs = list(legs)
    if all(leg.fusion == 'hard' for leg in legs):
        return any(legs[0].legs[0] != leg.legs[0] for leg in legs)
    if all(isinstance(leg.fusion, tuple) for leg in legs):
        mf = legs[0].fusion
        if any(mf != leg.fusion for leg in legs):
            raise YastError('Meta-fusions do not match.')
        return any(_leg_fusions_need_mask(*(mleg.legs[n] for mleg in legs)) for n in range(mf[0]))


def leg_union(*legs):
    """
    Output Leg that represent space being an union of spaces of a list of legs.
    """
    legs = list(legs)
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
        t, D, hf = _hfs_union(legs[0].sym, [leg.t for leg in legs] ,[leg.legs[0] for leg in legs])
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


def _unpack_legs(legs):
    """ return native legs and mfs. """
    ulegs, mfs = [], []
    for leg in legs:
        if isinstance(leg.fusion, tuple):  # meta-fused
            ulegs.extend(leg.legs)
            mfs.append(leg.fusion)
        else:  #_Leg
            ulegs.append(leg)
            mfs.append((1,))
    return tuple(ulegs), tuple(mfs)
