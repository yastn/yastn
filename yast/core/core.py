r"""
Yet another symmetric tensor

This class defines generic arbitrary-rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used (including no symmetries).

An instance of a Tensor is specified by a list of blocks (dense tensors) labeled by symmetries' charges on each leg.
"""

from collections import namedtuple
from functools import lru_cache
from itertools import product, groupby
from operator import itemgetter
import numpy as np
from ._auxliary import _unpack_axes, _clear_axes, _common_keys, _indices_common_rows
from ..sym import sym_none

__all__ = ['Tensor', 'YastError', 'check_signatures_match', 'check_consistency', 'allow_cache_meta']


_config = namedtuple('_config', ('backend', 'sym', 'dtype', 'device'), \
                    defaults = (None, sym_none, 'float64', 'cpu'))

_struct = namedtuple('_struct', ('t', 'D', 's', 'n'))

_check = {"signatures_match":True, "consistency":True, "cache_meta":True}

def check_signatures_match(value=True):
    """Set the value of the flag check_signatures_match."""
    _check["signatures_match"] = bool(value)

def check_consistency(value=True):
    """Set the value of the flag check_consistency."""
    _check["consistency"] = bool(value)

def allow_cache_meta(value=True):
    """Set the value of the flag that permits to reuses some metadata."""
    _check["cache_meta"] = bool(value)

class YastError(Exception):
    """Errors cought by checks in yast."""

class Tensor:
    """ Class defining a tensor with abelian symmetries, and operations on such tensor(s). """

    def __init__(self, config=None, s=(), n=None, isdiag=False, **kwargs):
        self.config = config if isinstance(config, _config) else _config(**{a:getattr(config, a) for a in _config._fields if hasattr(config, a)})
        self.isdiag = isdiag
        self.nlegs = 1 if isinstance(s, int) else len(s)  # number of native legs
        self.s = np.array(s, dtype=int).reshape(self.nlegs)
        self.n = np.zeros(self.config.sym.nsym, dtype=int) if n is None else np.array(n, dtype=int).reshape(self.config.sym.nsym)
        if self.isdiag:
            if len(self.s) == 0:
                self.s = np.array([1, -1], dtype=int)
                self.nlegs = 2
            if not np.sum(self.s) == 0:
                raise YastError("Signature should be (-1, 1) or (1, -1) in diagonal tensor")
            if not np.sum(np.abs(self.n)) == 0:
                raise YastError("Tensor charge should be 0 in diagonal tensor")
            if not self.nlegs == 2:
                raise YastError("Diagonal tensor should have ndim == 2")
        self.A = {}  # dictionary of blocks
        # (meta) fusion tree for each leg: (encodes number of fused legs e.g. 5 2 1 1 3 1 2 1 1 -- 5 legs fused and history); is immutable
        self.meta_fusion = tuple(kwargs['meta_fusion']) if ('meta_fusion' in kwargs and kwargs['meta_fusion'] is not None) else ((1,),) * self.nlegs
        self.mlegs = len(self.meta_fusion)  # number of meta legs
        self.struct = _struct((), (), tuple(self.s), tuple(self.n))

    ######################
    #     fill tensor    #
    ######################

    def fill_tensor(self, t=(), D=(), val='rand'):
        r"""
        Create all possible blocks based on s, n and list of charges for all legs.

        Brute-force check all possibilities and select the ones satisfying f(t@s) == n for each symmetry generator f.
        Initialize each possible block with sizes given by D.

        Parameters
        ----------
        t : list
            All possible combination of charges for each leg:
            t = [[(leg1sym1, leg1sym2), ... ], [(leg2sym1, leg2sym2), ... )]
            If nsym is 0, it is not taken into account.
            When somewhere there is only one value and it is unambiguous, tuple can typically be replaced by int, see examples.

        D : tuple
            list of bond dimensions on all legs
            If nsym == 0, D = [leg1, leg2, leg3]
            If nsym >= 1, it should match the form of t
            When somewhere there is only one value tuple can typically be replaced by int.

        val : str
            'randR', 'rand' (use current dtype float or complex), 'ones', 'zeros'

        Examples
        --------
        D = 5 # ndim = 1)
        D = (1, 2, 3) # nsym = 0, ndim = 3
        t = [0, (-2, 0), (2, 0)] D=[1, (1, 2), (1, 3)] # nsym = 1 ndim = 3
        t = [[(0, 0)], [(-2, -2), (0, 0), (-2, 0), (0, -2)], [(2, 2), (0, 0), (2, 0), (0, 2)]] D=[1, (1, 4, 2, 2), (1, 9, 3, 3)] # nsym = 2 ndim = 3
        """
        D = (D,) if isinstance(D, int) else D
        t = (t,) if isinstance(t, int) else t

        if self.config.sym.nsym == 0:
            if self.isdiag and len(D) == 1:
                D = D + D
            if len(D) != self.nlegs:
                raise YastError("Number of elements in D does not match tensor rank.")
            tset = np.zeros((1, self.nlegs, self.config.sym.nsym))
            Dset = np.array(D, dtype=int).reshape(1, self.nlegs)
        else:  # self.config.sym.nsym >= 1
            D = (D,) if (self.nlegs == 1 or self.isdiag) and isinstance(D[0], int) else D
            t = (t,) if (self.nlegs == 1 or self.isdiag) and isinstance(t[0], int) else t
            D = D + D if self.isdiag and len(D) == 1 else D
            t = t + t if self.isdiag and len(t) == 1 else t

            D = list((x,) if isinstance(x, int) else x for x in D)
            t = list((x,) if isinstance(x, int) else x for x in t)

            if len(D) != self.nlegs:
                raise YastError("Number of elements in D does not match tensor rank.")
            if len(t) != self.nlegs:
                raise YastError("Number of elements in t does not match tensor rank.")
            for x, y in zip(D, t):
                if len(x) != len(y):
                    raise YastError("Elements of t and D do not match")

            comb_t = list(product(*t))
            comb_D = list(product(*D))
            lcomb_t = len(comb_t)
            comb_t = np.array(comb_t, dtype=int).reshape(lcomb_t, self.nlegs, self.config.sym.nsym)
            comb_D = np.array(comb_D, dtype=int).reshape(lcomb_t, self.nlegs)
            ind = np.all(self.config.sym.fuse(comb_t, self.s, 1) == self.n, axis=1)
            tset = comb_t[ind]
            Dset = comb_D[ind]

        for ts, Ds in zip(tset, Dset):
            self.set_block(tuple(ts.flat), tuple(Ds), val)

    def set_block(self, ts=(), Ds=None, val='zeros'):
        """
        Add new block to tensor or change the existing one.

        This is the intended way to add new blocks by hand.
        Checks if bond dimensions of the new block are consistent with the existing ones.
        Updates meta-data.

        Parameters
        ----------
        ts : tuple
            charges identifing the block, t = (sym1leg1, sym2leg1, sym1leg2, sym2leg2, ...)
            If nsym == 0, it is not taken into account.

        Ds : tuple
            bond dimensions of the block. Ds = (leg1, leg2, leg3)
            If Ds not given, tries to read it from existing blocks.

        val : str, nparray, list
            'randR', 'rand' (use current dtype float or complex), 'ones', 'zeros'
            for nparray setting Ds is needed.
        """
        if isinstance(Ds, int):
            Ds = (Ds,)
        if isinstance(ts, int):
            ts = (ts,)
        if self.isdiag and Ds is not None and len(Ds) == 1:
            Ds = Ds + Ds
        if self.isdiag and len(ts) == self.config.sym.nsym:
            ts = ts + ts

        if len(ts) != self.nlegs * self.config.sym.nsym:
            raise YastError('Wrong size of ts.')
        if Ds is not None and len(Ds) != self.nlegs:
            raise YastError('Wrong size of Ds.')

        ats = np.array(ts, dtype=int).reshape(1, self.nlegs, self.config.sym.nsym)
        if not np.all(self.config.sym.fuse(ats, self.s, 1) == self.n):
            raise YastError('Charges ts are not consistent with the symmetry rules: t @ s - n != 0')

        if isinstance(val, str):
            if Ds is None:  # attempt to read Ds from existing block
                Ds = []
                tD = [self.get_leg_structure(n, native=True) for n in range(self.nlegs)]
                for n in range(self.nlegs):
                    try:
                        Ds.append(tD[n][tuple(ats[0, n, :].flat)])
                    except KeyError:
                        raise YastError('Provided Ds. Cannot infer all bond dimensions from existing blocks.')
                Ds = tuple(Ds)

            if val == 'zeros':
                self.A[ts] = self.config.backend.zeros(Ds, dtype=self.config.dtype, device=self.config.device)
            elif val == 'randR':
                self.A[ts] = self.config.backend.randR(Ds, dtype=self.config.dtype, device=self.config.device)
            elif val == 'rand':
                self.A[ts] = self.config.backend.rand(Ds, dtype=self.config.dtype, device=self.config.device)
            elif val == 'ones':
                self.A[ts] = self.config.backend.ones(Ds, dtype=self.config.dtype, device=self.config.device)
            if self.isdiag:
                self.A[ts] = self.config.backend.diag_get(self.A[ts])
                self.A[ts] = self.config.backend.diag_create(self.A[ts])
        else:  # enforce that Ds is provided to increase clarity of the code
            if self.isdiag and val.ndim == 1 and np.prod(Ds)==(val.size**2):
                self.A[ts] = self.config.backend.to_tensor(np.diag(val), Ds, dtype=self.config.dtype, device=self.config.device)
            else:
                self.A[ts] = self.config.backend.to_tensor(val, Ds=Ds, dtype=self.config.dtype, device=self.config.device)
        self.update_struct()
        tD = [self.get_leg_structure(n, native=True) for n in range(self.nlegs)]  # here checks the consistency of bond dimensions

    #######################
    #     new tensors     #
    #######################

    def copy(self):
        """ Return a copy of the tensor.

            Warning: this might break autograd if you are using it.
        """
        a = Tensor(config=self.config, s=self.s, n=self.n, isdiag=self.isdiag, meta_fusion=self.meta_fusion)
        a.A = {ts: self.config.backend.copy(x) for ts, x in self.A.items()}
        a.struct = self.struct
        return a

    @property
    def real(self):
        if not self.is_complex():
            raise RuntimeError("Supported only for complex tensors.")
        config_real= self.config._replace(dtype="float64")
        a = Tensor(config=config_real, s=self.s, n=self.n, isdiag=self.isdiag, meta_fusion=self.meta_fusion)
        a.A = {ts: self.config.backend.real(x) for ts, x in self.A.items()}
        a.struct = self.struct
        return a

    @property
    def imag(self):
        if not self.is_complex():
            raise RuntimeError("Supported only for complex tensors.")
        config_real= self.config._replace(dtype="float64")
        a = Tensor(config=config_real, s=self.s, n=self.n, isdiag=self.isdiag, meta_fusion=self.meta_fusion)
        a.A = {ts: self.config.backend.imag(x) for ts, x in self.A.items()}
        a.struct = self.struct
        return a

    def clone(self):
        """ Return a copy of the tensor, tracking gradients. """
        a = Tensor(config=self.config, s=self.s, n=self.n, isdiag=self.isdiag, meta_fusion=self.meta_fusion)
        a.A = {ts: self.config.backend.clone(x) for ts, x in self.A.items()}
        a.struct = self.struct
        return a

    def detach(self, inplace=False):
        """ Detach tensor from autograd; Can be called inplace (?) """
        if inplace:
            for x in self.A.values(): self.config.backend.detach_(x)
            return self
        a = Tensor(config=self.config, s=self.s, n=self.n, isdiag=self.isdiag, meta_fusion=self.meta_fusion)
        a.struct = self.struct
        a.A = {ts: self.config.backend.detach(x) for ts, x in self.A.items()}
        return a

    def to(self, device):
        r"""
        Move the ``Tensor`` to ``device``. Returns a copy of the tensor on `device``.

        If the tensor already resides on ``device``, return self

        Parameters
        ----------
        device: str
            device identifier
        """
        if self.config.device == device:
            return self
        config_d = self.config._replace(device=device)
        a = Tensor(config=config_d, s=self.s, n=self.n, isdiag=self.isdiag, meta_fusion=self.meta_fusion)
        a.struct = self.struct
        a.A = self.config.backend.move_to_device(self.A, device)
        return a

    def copy_empty(self):
        """ Return a copy of the tensor, but without copying blocks. """
        return Tensor(config=self.config, s=self.s, n=self.n, isdiag=self.isdiag, meta_fusion=self.meta_fusion)

    #######################
    #    export tensor    #
    #######################

    def export_to_dict(self):
        r"""
        Export relevant information about tensor to dictionary --  it can be saved using numpy.save

        Returns
        -------
        d: dict
            dictionary containing all the information needed to recreate the tensor.
        """
        AA = {ind: self.config.backend.to_numpy(self.A[ind]) for ind in self.A}
        if self.isdiag:
            AA = {t: np.diag(x) for t, x in AA.items()}
        out = {'A': AA, 's': tuple(self.s), 'n': tuple(self.n), 'isdiag': self.isdiag, 'meta_fusion': self.meta_fusion}
        return out

    def compress_to_1d(self, meta=None):
        """
        Store each block as 1D array within r1d in contiguous manner; outputs meta-information to reconstruct the original tensor

        Parameters
        ----------
            meta: dict
                If not None, uses this metainformation to merge into 1d structure (filling-in zeros if tensor does not have some blocks).
                Raise error, if tensor has some blocks which are not included in meta; or otherwise meta does not match the tensor.
        """
        if meta is None:
            D_rsh = np.prod(self._Darray(), axis=1)
            aD_rsh = np.cumsum(D_rsh)
            D_tot = np.sum(D_rsh)
            meta_new = (((), D_tot),)
            # meta_merge = ((tn, Ds, to, Do), ...)
            meta_merge = tuple(((), (aD-D, aD), t, (D,)) for t, D, aD in zip(self.struct.t, D_rsh, aD_rsh))
            # (told, tnew, Dsl, Dnew)
            meta_unmerge = tuple((told, tnew, Dsl, Dnew) for (told, Dsl, tnew, _), Dnew in zip(meta_merge, self.struct.D))
            meta = {'s': tuple(self.s), 'n': tuple(self.n), 'isdiag': self.isdiag, \
                'meta_fusion': self.meta_fusion, 'meta_unmerge':meta_unmerge, 'meta_merge':meta_merge}
        else:
            if tuple(self.s) != meta['s'] or tuple(self.n) != meta['n'] or self.isdiag != meta['isdiag'] or self.meta_fusion != meta['meta_fusion']:
                raise YastError("Tensor do not match provided metadata.")
            meta_merge = meta['meta_merge']
            D_tot = meta_merge[-1][1][1]
            meta_new = (((), D_tot),)
            if len(self.A) != sum(ind in self.A for (_, _, ind, _) in meta_merge):
                raise YastError("Tensor has blocks that do not appear in meta.")

        A = self.config.backend.merge_one_leg(self.A, 0, tuple(range(self.nlegs)), \
            meta_new, meta_merge, self.config.dtype, self.config.device)
        return A[()], meta

    ############################
    #    output information    #
    ############################

    def show_properties(self):
        """ Display basic properties of the tensor. """
        print("Symmetry    :", self.config.sym.name)
        print("signature   :", self.s)  # signature
        print("charge      :", self.n)  # total charge of tensor
        print("isdiag      :", self.isdiag)
        print("dim meta    :", self.mlegs)  # number of meta legs
        print("dim native  :", self.nlegs)  # number of native legs
        print("shape meta  :", self.get_shape(native=False))
        print("shape native:", self.get_shape(native=True))
        print("no. blocks  :", len(self.A))  # number of blocks
        print("size        :", self.get_size())  # total number of elements in all blocks
        print("meta fusion :", self.meta_fusion, "\n")  # encoding meta fusion tree for each leg

    def __str__(self):
        # return str(self.A)
        ts, Ds= self.get_leg_charges_and_dims(native=False)
        s=f"{self.config.sym.name} s= {self.s} n= {self.n}\n"
        # s+=f"charges      : {self.ts}\n"
        s+=f"leg charges  : {ts}\n"
        s+=f"dimensions   : {Ds}"
        return s

    def print_blocks(self):
        for ind, x in self.A.items():
            print(f"{ind} {self.config.backend.get_shape(x)}")

    def is_complex(self):
        """ Returns True if all blocks are complex. """
        return all([self.config.backend.is_complex(x) for x in self.A.values()])

    def get_size(self):
        """ Total number of elements in the tensor. """
        return sum(np.prod(self._Darray(), axis=1))

    def get_tensor_charge(self):
        """ Global charge of the tensor. """
        return tuple(self.n)

    def get_signature(self, native=False):
        """ Tensor signatures. If not native, returns the signature of the first leg in each group."""
        if native:
            return tuple(self.s)
        pn = tuple((n,) for n in range(self.mlegs)) if self.mlegs > 0 else ()
        un = tuple(_unpack_axes(self, *pn))
        return tuple(self.s[p[0]] for p in un)

    def get_blocks_charges(self):
        """ Charges of all native blocks. """
        return self.struct.t

    def get_blocks_shapes(self):
        """ Shapes fo all native blocks. """
        return self.struct.D

    def get_leg_fusion(self, axes=None):
        """
        Fusion trees for meta legs.

        Parameters
        ----------
        axes : Int or tuple of ints
            indices of legs; If axes is None returns all (default).
        """
        if axes is None:
            return self.meta_fusion
        if isinstance(axes, int):
            return self.meta_fusion(axes)
        return tuple(self.meta_fusion(n) for n in axes)

    def get_leg_structure(self, axis, native=False):
        r"""
        Find all charges and the corresponding bond dimension for n-th leg.

        Parameters
        ----------
        axis : int
            Index of a leg.

        native : bool
            consider native legs if True; otherwise meta/fused legs (default).

        Returns
        -------
            tDn : dict of {tn: Dn}
        """
        axis, = _clear_axes(axis)
        if not native:
            axis, = _unpack_axes(self, axis)
        tset = self._tarray()
        Dset = self._Darray()
        tset = tset[:, axis, :]
        Dset = Dset[:, axis]
        tset = tset.reshape(len(tset), len(axis) * self.config.sym.nsym)
        Dset = np.prod(Dset, axis=1) if len(axis) > 1 else Dset.reshape(-1)

        tDn = {tuple(tn.flat): Dn for tn, Dn in zip(tset, Dset)}
        if _check["consistency"]:
            for tn, Dn in zip(tset, Dset):
                if tDn[tuple(tn.flat)] != Dn:
                    raise YastError('Inconsistend bond dimension of charge.')
        return tDn

    def get_leg_charges_and_dims(self, native=False):
        """ collect information about charges and dimensions on all legs into two lists. """
        _tmp= [self.get_leg_structure(n, native=native) for n in range(self.get_ndim(native))]
        ts, Ds= tuple(zip(*[tuple(zip(*l.items())) for l in _tmp]))
        return ts, Ds

    def get_shape(self, axes=None, native=False):
        r"""
        Return total bond dimension of meta legs.

        Parameters
        ----------
        axes : Int or tuple of ints
            indices of legs; If axes is None returns all (default).

        Returns
        -------
        shape : Int or tuple of ints
            shapes of legs specified by axes
        """
        if axes is None:
            axes = tuple(n for n in range(self.nlegs if native else self.mlegs))
        if isinstance(axes, int):
            return sum(self.get_leg_structure(axes, native=native).values())
        return tuple(sum(self.get_leg_structure(ii, native=native).values()) for ii in axes)

    def get_ndim(self, native=False):
        """ Number of: meta legs if not native else native legs. """
        return self.nlegs if native else self.mlegs

    def __getitem__(self, key):
        """ Returns block based on its charges. """
        return self.A[key]

    #########################
    #    output tensors     #
    #########################

    def to_dense(self, leg_structures=None, native=False):
        r"""
        Create full tensor corresponding to the symmetric tensor.

        Blockes are ordered according to increasing charges on each leg.
        It is possible to supply a list of charges and bond dimensions to be included
        (should be consistent with the tensor). This allows to fill in some zero blocks.

        Parameters
        ----------
        leg_structures : dict
            {n: {tn: Dn}} specify charges and dimensions to include on some legs (indicated by keys n).

        native: bool
            output native tensor (neglecting meta fusions).

        Returns
        -------
        out : tensor of the type used by backend
        """
        nlegs = self.get_ndim(native=native)
        tD = [self.get_leg_structure(n, native=native) for n in range(nlegs)]
        if leg_structures is not None:
            for n, tDn in leg_structures.items():
                if (n < 0) or (n >= nlegs):
                    raise YastError('Specified leg out of ndim')
                for tn, Dn in tDn.items():
                    if (tn in tD[n]) and tD[n][tn] != Dn:
                        raise YastError('Specified bond dimensions inconsistent with tensor.')
                    tD[n][tn] = Dn
        Dtot = [sum(tDn.values()) for tDn in tD]
        for tDn in tD:
            tns = sorted(tDn.keys())
            Dlow = 0
            for tn in tns:
                Dhigh = Dlow + tDn[tn]
                tDn[tn] = (Dlow, Dhigh)
                Dlow = Dhigh
        axes = tuple((n,) for n in range(nlegs))
        if not native:
            axes = tuple(_unpack_axes(self, *axes))
        meta = []
        tset = self._tarray()
        for tind, tt in zip(self.struct.t, tset):
            meta.append((tind, tuple(tD[n][tuple(tt[m, :].flat)] for n, m in enumerate(axes))))
        return self.config.backend.merge_to_dense(self.A, Dtot, meta, self.config.dtype, self.config.device)

    def to_numpy(self, leg_structures=None, native=False):
        r"""
        Create full nparray corresponding to the symmetric tensor. See `yast.to_dense`
        """
        return self.config.backend.to_numpy(self.to_dense(leg_structures, native))

    def to_raw_tensor(self):
        """
        For tensor with a single block, return raw tensor representing that block.
        """
        if len(self.A) == 1:
            key = next(iter(self.A))
            return self.A[key]
        raise YastError('Only tensor with a single block can be converted to raw tensor')

    def to_nonsymmetric(self, leg_structures=None, native=False):
        r"""
        Create full tensor corresponding to the symmetric tensor. Output it as yast tensor with no symmetry.

        Blockes are ordered according to increasing charges on each leg.
        It is possible to supply a list of charges and bond dimensions to be included
        (should be consistent with the tensor). This allows to fill in some zero blocks.

        Parameters
        ----------
        leg_structures : dict
            {n: {tn: Dn}} specify charges and dimensions to include on some legs (indicated by keys n).

        native: bool
            output native tensor (neglecting meta fusions).

        Returns
        -------
        out : tensor of the type used by backend
        """
        config_dense = self.config._replace(sym=sym_none)
        news = self.get_signature(native)
        T = Tensor(config=config_dense, s=news, n=None, isdiag=self.isdiag)
        T.set_block(val=self.to_dense(leg_structures, native))
        return T

    #########################
    #    output numbers     #
    #########################

    def zero_of_dtype(self):
        return self.config.backend.zero_scalar(dtype=self.config.dtype, device=self.config.device)

    def to_number(self):
        """
        Return an element of the size-one tensor as a scalar of the same type as the
        type use by backend.

        For empty tensor, returns 0
        """
        size = self.get_size()
        if size == 1:
            return self.config.backend.first_element(next(iter(self.A.values())))
        if size == 0:
            return self.zero_of_dtype()
            # is there a better solution for torch autograd?
        raise YastError('Specified bond dimensions inconsistent with tensor.')

    def item(self):
        """
        Return an element of the size-one tensor as a standard Python scalar.

        For empty tensor, returns 0
        """
        size = self.get_size()
        if size == 1:
            return self.config.backend.item(next(iter(self.A.values())))
        if size == 0:
            return 0
        raise YastError("only single-element (symmetric) Tensor can be converted to scalar")

    def norm(self, p='fro'):
        r"""
        Norm of the rensor.

        Parameters
        ----------
        p: str
            'fro' = Frobenious; 'inf' = max(abs())

        Returns
        -------
        norm : float64
        """
        if len(self.A) == 0:
            return self.zero_of_dtype()
        return self.config.backend.norm(self.A, p)

    # def max_abs(self):
    #     """
    #     Largest element by magnitude.  THIS IS OBSOLATE norm(ord = 'inf') DOES THE SAME

    #     Returns
    #     -------
    #     max_abs : scalar
    #     """
    #     return self.zero_of_dtype() if len(self.A) == 0 else self.config.backend.max_abs(self.A)

    def norm_diff(self, other, p='fro'):
        """
        Norm of the difference of two tensors.

        Parameters
        ----------
        other: Tensor

        ord: str
            'fro' = Frobenious; 'inf' = max(abs())

        Returns
        -------
        norm : float64
        """
        self._test_tensors_match(other)
        if (len(self.A) == 0) and (len(other.A) == 0):
            return self.zero_of_dtype()
        meta = _common_keys(self.A, other.A)
        return self.config.backend.norm_diff(self.A, other.A, meta, p)

    #############################
    #     linear operations     #
    #############################

    def __mul__(self, other):
        """
        Multiply tensor by a number, use: number * tensor.

        Parameters
        ----------
        other: number

        Returns
        -------
        tensor : Tensor
            result of multipcilation as a new tensor
        """
        a = self.copy_empty()
        a.A= {ind: other * x for ind, x in self.A.items()}
        a.struct = self.struct
        return a

    def __rmul__(self, other):
        """
        Multiply tensor by a number, use: tensor * number.

        Parameters
        ----------
        other: number

        Returns
        -------
        tensor : Tensor
            result of multipcilation as a new tensor
        """
        return self.__mul__(other)

    def __pow__(self, exponent):
        """
        Element-wise exponent of tensor, use: tensor ** exponent.

        Parameters
        ----------
        exponent: number

        Returns
        -------
        tensor : Tensor
            result of element-wise exponentiation as a new tensor
        """
        a = self.copy_empty()
        a.A= {ind: x**exponent for ind, x in self.A.items()}
        a.struct = self.struct
        return a

    def __truediv__(self, other):
        """
        Divide tensor by a scalar, use: tensor / scalar.

        Parameters
        ----------
        other: scalar

        Returns
        -------
        tensor : Tensor
            result of element-wise division  as a new tensor
        """
        a = self.copy_empty()
        a.A= {ind: x / other for ind, x in self.A.items()}
        a.struct = self.struct
        return a

    def __add__(self, other):
        """
        Add two tensors, use: tensor + tensor.

        Signatures and total charges should match.

        Parameters
        ----------
        other: Tensor

        Returns
        -------
        tensor : Tensor
            result of addition as a new tensor
        """
        self._test_configs_match(other)
        self._test_tensors_match(other)
        meta = _common_keys(self.A, other.A)
        a = self.copy_empty()
        a.A = a.config.backend.add(self.A, other.A, meta)
        a.update_struct()
        return a

    def __sub__(self, other):
        """
        Subtract two tensors, use: tensor - tensor.

        Both signatures and total charges should match.

        Parameters
        ----------
        other: Tensor

        Returns
        -------
        tensor : Tensor
            result of subtraction as a new tensor
        """
        self._test_configs_match(other)
        self._test_tensors_match(other)
        meta = _common_keys(self.A, other.A)
        a = self.copy_empty()
        a.A = a.config.backend.sub(self.A, other.A, meta)
        a.update_struct()
        return a

    def apxb(self, other, x=1):
        """
        Directly calculate tensor + x * other tensor

        Signatures and total charges should match.

        Parameters
        ----------
        other: Tensor
        x : number

        Returns
        -------
        tensor : Tensor
            result of addition as a new tensor
        """
        self._test_configs_match(other)
        self._test_tensors_match(other)
        meta = _common_keys(self.A, other.A)
        a = self.copy_empty()
        a.A = a.config.backend.apxb(self.A, other.A, x, meta)
        a.update_struct()
        return a

    #############################
    #     tensor operations     #
    #############################

    def conj(self, inplace=False):
        """
        Return conjugated tensor.

        Changes sign of signature s and total charge n, as well as complex conjugate each block.

        Returns
        -------
        tensor : Tensor
        """
        newn = self.config.sym.fuse(self.n.reshape(1, 1, -1), np.array([1], dtype=int), -1)[0]
        if inplace:
            a = self
            a.n = newn
            a.s *= -1
        else:
            a = Tensor(config=self.config, s=-self.s, n=newn, isdiag=self.isdiag, meta_fusion=self.meta_fusion)
        
        a.struct = self.struct._replace(s=tuple(-self.s))
        a.A = a.config.backend.conj(self.A, inplace)
        return a

    def conj_blocks(self, inplace=False):
        """
        Conjugated each block, leaving symmetry structure unchanged.

        Returns
        -------
        tensor : Tensor
        """
        if inplace:
            a = self
        else:
            a = self.copy_empty()
            a.struct = self.struct
        a.A = a.config.backend.conj(self.A, inplace)
        return a

    def flip_signature(self, inplace=False):
        """
        Conjugated each block, leaving symmetry structure unchanged.

        Returns
        -------
        tensor : Tensor
        """
        newn = self.config.sym.fuse(self.n.reshape(1, 1, -1), np.array([1], dtype=int), -1)[0]
        if inplace:
            self.n = newn
            self.s = -self.s
            return self
        a = Tensor(config=self.config, s=-self.s, n=newn, isdiag=self.isdiag, meta_fusion=self.meta_fusion)
        a.struct = self.struct
        a.A= {ind: self.config.backend.clone(self.A[ind]) for ind in self.A}
        return a

    def transpose(self, axes, inplace=False):
        r"""
        Return transposed tensor.

        Operation can be done in-place, in which case copying of the data is not forced.
        Othersiwe, new tensor is created and the data are copied.

        Parameters
        ----------
        axes: tuple of ints
            New order of the legs. Should be a permutation of (0, 1, ..., ndim-1)
        """
        uaxes, = _unpack_axes(self, axes)
        order = np.array(uaxes, dtype=np.intp)
        new_meta_fusion = tuple(self.meta_fusion[ii] for ii in axes)
        news = self.s[order]
        if inplace:
            self.s = news
            self.meta_fusion = new_meta_fusion
            a = self
        else:
            a = Tensor(config=self.config, s=news, n=self.n, isdiag=self.isdiag, meta_fusion=new_meta_fusion)
        tset = self._tarray()
        newt = tset[:, order, :]
        meta_transpose = tuple((told, tuple(tnew.flat)) for told, tnew in zip(self.struct.t, newt))
        a.A = a.config.backend.transpose(self.A, uaxes, meta_transpose, inplace)
        a.update_struct()
        return a

    def moveaxis(self, source, destination, inplace=False):
        r"""
        Change the position of an axis (or a group of axes) of the tensor.

        Operation can be done in-place, in which case copying of the data is not forced.
        Othersiwe, new tensor is created and the data are copied. Calls transpose.

        Parameters
        ----------
        source, destination: ints
        """
        lsrc, ldst = _clear_axes(source, destination)
        lsrc = tuple(xx + self.mlegs if xx < 0 else xx for xx in lsrc)
        ldst = tuple(xx + self.mlegs if xx < 0 else xx for xx in ldst)
        if lsrc == ldst:
            return self if inplace else self.copy()
        axes = [ii for ii in range(self.mlegs) if ii not in lsrc]
        ds = sorted(((d, s) for d, s in zip(ldst, lsrc)))
        for d, s in ds:
            axes.insert(d, s)
        return self.transpose(axes, inplace)

    def diag(self):
        """Select diagonal of 2d tensor and output it as a diagonal tensor, or vice versa. """
        if self.isdiag:
            a = Tensor(config=self.config, s=self.s, n=self.n, isdiag=False, meta_fusion=self.meta_fusion)
            a.A = {ind: self.config.backend.diag_diag(self.A[ind]) for ind in self.A}
        elif self.nlegs == 2 and sum(np.abs(self.n)) == 0 and sum(self.s) == 0:
            a = Tensor(config=self.config, s=self.s, isdiag=True, meta_fusion=self.meta_fusion)
            a.A = {ind: self.config.backend.diag_diag(self.A[ind]) for ind in self.A}
        else:
            raise YastError('Tensor cannot be changed into a diagonal one')
        a.update_struct()
        return a

    def abs(self):
        """
        Return element-wise absolut value.

        Returns
        -------
        tansor: Tensor
        """
        a = self.copy_empty()
        a.A = self.config.backend.absolute(self.A)
        a.struct = self.struct
        return a

    def rsqrt(self, cutoff=0):
        """
        Return element-wise 1/sqrt(A).

        The tensor elements with absolut value below the cutoff are set to zero.

        Parameters
        ----------
            cutoff: float64
            Cut-off for (elementwise) pseudo-inverse.

        Returns
        -------
        tansor: Tensor
        """
        a = self.copy_empty()
        if not a.isdiag:
            a.A = self.config.backend.rsqrt(self.A, cutoff=cutoff)
        else:
            a.A = self.config.backend.rsqrt_diag(self.A, cutoff=cutoff)
        a.struct = self.struct
        return a

    def reciprocal(self, cutoff=0):
        """
        Return element-wise 1/A.

        The tensor elements with absolut value below the cutoff are set to zero.

        Parameters
        ----------
        cutoff: float64
            Cut-off for (elementwise) pseudo-inverse.

        Returns
        -------
        tansor: Tensor
        """
        a = self.copy_empty()
        if not a.isdiag:
            a.A = self.config.backend.reciprocal(self.A, cutoff=cutoff)
        else:
            a.A = self.config.backend.reciprocal_diag(self.A, cutoff=cutoff)
        a.struct = self.struct
        return a

    def exp(self, step=1.):
        """
        Return element-wise exp(step * A).

        This is calculated for existing blocks only.

        Parameters
        ----------
        step: number

        Returns
        -------
        tansor: Tensor
        """
        a = self.copy_empty()
        if not a.isdiag:
            a.A = self.config.backend.exp(self.A, step)
        else:
            a.A = self.config.backend.exp_diag(self.A, step)
        a.struct = self.struct
        return a

    def sqrt(self):
        """
        Return element-wise sqrt(A).

        Parameters
        ----------
        step: number

        Returns
        -------
        tansor: Tensor
        """
        a = self.copy_empty()
        a.A = self.config.backend.sqrt(self.A)
        a.struct = self.struct
        return a

    ##################################
    #     contraction operations     #
    ##################################

    def tensordot(a, b, axes, conj=(0, 0)):
        r"""
        Compute tensor dot product of two tensor along specified axes.

        Outgoing legs are ordered such that first ones are the remaining legs of the first tensor in the original order,
        and than those of the second tensor.

        Parameters
        ----------
        a, b: Tensors to contract

        axes: tuple
            legs of both tensors (for each it is specified by int or tuple of ints)
            e.g. axes=(0, 3), axes=((0, 3), (1, 2))

        conj: tuple
            shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1).
            Defult is (0, 0), i.e. no conjugation

        Returns
        -------
        tansor: Tensor
        """
        a._test_configs_match(b)
        la_con, lb_con = _clear_axes(*axes)  # contracted meta legs
        la_out = tuple(ii for ii in range(a.mlegs) if ii not in la_con)  # outgoing meta legs
        lb_out = tuple(ii for ii in range(b.mlegs) if ii not in lb_con)  # outgoing meta legs

        a_con, a_out = _unpack_axes(a, la_con, la_out)  # actual legs of a
        b_con, b_out = _unpack_axes(b, lb_con, lb_out)  # actual legs of b

        na_con, na_out = np.array(a_con, dtype=np.intp), np.array(a_out, dtype=np.intp)
        nb_con, nb_out = np.array(b_con, dtype=np.intp), np.array(b_out, dtype=np.intp)

        conja, conjb = (1 - 2 * conj[0]), (1 - 2 * conj[1])
        if not all(a.s[na_con] == (-conja * conjb) * b.s[nb_con]):
            if a.isdiag:  # if tensor is diagonal, than freely changes the signature by a factor of -1
                a.s *= -1
            elif b.isdiag:
                b.s *= -1
            elif _check["signatures_match"]:
                raise YastError('Signs do not match')

        c_n = np.vstack([a.n, b.n]).reshape(1, 2, -1)
        c_s = np.array([conja, conjb], dtype=int)
        c_n = a.config.sym.fuse(c_n, c_s, 1)

        inda, indb = _indices_common_rows(a._tarray()[:, na_con, :], b._tarray()[:, nb_con, :])

        Am, ls_l, ls_ac, ua_l, ua_r = a.merge_to_matrix(a_out, a_con, conja, -conja, inda, sort_r=True)
        Bm, ls_bc, ls_r, ub_l, ub_r = b.merge_to_matrix(b_con, b_out, conjb, -conjb, indb)

        meta_dot = tuple((al + br, al + ar, bl + br)  for al, ar, bl, br in zip(ua_l, ua_r, ub_l, ub_r))

        if _check["consistency"] and not (ua_r == ub_l and ls_ac.match(ls_bc)):
            raise YastError('Something went wrong in matching the indices of the two tensors')

        c_s = np.hstack([conja * a.s[na_out], conjb * b.s[nb_out]])
        c_meta_fusion = [a.meta_fusion[ii] for ii in la_out] + [b.meta_fusion[ii] for ii in lb_out]
        c = Tensor(config=a.config, s=c_s, n=c_n, meta_fusion=c_meta_fusion)

        Cm = c.config.backend.dot(Am, Bm, conj, meta_dot)
        c.A = c.unmerge_from_matrix(Cm, ls_l, ls_r)
        c.update_struct()
        return c

    def vdot(a, b, conj=(1, 0)):
        r"""
        Compute scalar product x = <a|b> of two tensors. a is conjugated.

        Parameters
        ----------
        other: Tensor

        Returns
        -------
        x: number
        """
        a._test_configs_match(b)
        a._test_tensors_match(b)
        k12, _, _ = _common_keys(a.A, b.A)
        if len(k12) > 0:
            return a.config.backend.vdot(a.A, b.A, k12)
        return a.zero_of_dtype()

    def trace(self, axes=(0, 1)):
        """
        Compute trace of legs specified by axes.

        Parameters
        ----------
            axes: tuple
            Legs to be traced out, e.g axes=(0, 1); or axes=((2, 3, 4), (0, 1, 5))

        Returns
        -------
            tensor: Tensor
        """
        lin1, lin2 = _clear_axes(*axes)  # contracted legs
        lin12 = lin1 + lin2
        lout = tuple(ii for ii in range(self.mlegs) if ii not in lin12)
        in1, in2, out = _unpack_axes(self, lin1, lin2, lout)

        if len(in1) != len(in2) or len(lin1) != len(lin2):
            raise YastError('Number of axis to trace should be the same')
        if len(in1) == 0:
            return self

        order = in1 + in2 + out
        ain1 = np.array(in1, dtype=np.intp)
        ain2 = np.array(in2, dtype=np.intp)
        aout = np.array(out, dtype=np.intp)

        if not all(self.s[ain1] == -self.s[ain2]):
            raise YastError('Signs do not match')

        tset = self._tarray()
        Dset = self._Darray()
        lt = len(tset)
        t1 = tset[:, ain1, :].reshape(lt, -1)
        t2 = tset[:, ain2, :].reshape(lt, -1)
        to = tset[:, aout, :].reshape(lt, -1)
        D1 = Dset[:, ain1]
        D2 = Dset[:, ain2]
        D3 = Dset[:, aout]
        pD1 = np.prod(D1, axis=1).reshape(lt, 1)
        pD2 = np.prod(D2, axis=1).reshape(lt, 1)
        ind = (np.all(t1==t2, axis=1)).nonzero()[0]
        Drsh = np.hstack([pD1, pD2, D3])

        if not np.all(D1[ind] == D2[ind]):
            raise YastError('Not all bond dimensions of the traced legs match')

        meta = [(tuple(to[n]), tuple(tset[n].flat), tuple(Drsh[n])) for n in ind]
        a = Tensor(config=self.config, s=self.s[aout], n=self.n, meta_fusion=tuple(self.meta_fusion[ii] for ii in lout))
        a.A = a.config.backend.trace(self.A, order, meta)
        a.update_struct()
        return a

    #############################
    #        swap gate          #
    #############################

    def swap_gate(self, axes, inplace=True):
        """
        Return tensor after application of the swap gate.

        Multiply the block with odd charges on swaped legs by -1.
        If one of the provided axes is -1, then swap with the charge n.

        Parameters
        ----------
        axes: tuple
            two groups of legs to be swaped

        Returns
        -------
        tensor : Tensor
        """
        try:
            fss = self.config.sym.fermionic  # fermionic symmetry sectors
        except NameError:
            return self
        if any(fss):
            a = self if inplace else self.clone()
            tset = a._tarray()
            l1, l2 = _clear_axes(*axes)  # swaped groups of legs
            if len(set(l1) & set(l2)) > 0:
                raise YastError('Cannot sweep the same index')
            if l2 == (-1,):
                l1, l2 = l2, l1
            if l1 == (-1,):
                l2, = _unpack_axes(self, l2)
                t1 = a.n
            else:
                l1, l2 = _unpack_axes(self, l1, l2)
                al1 = np.array(l1, dtype=np.intp)
                t1 = np.sum(tset[:, al1, :], axis=1)
            al2 = np.array(l2, dtype=np.intp)
            t2 = np.sum(tset[:, al2, :], axis=1)
            tp = np.sum(t1 * t2, axis=1) % 2 == 1
            for ind, odd in zip(self.struct.t, tp):
                if odd:
                    a.A[ind] = -a.A[ind]
            return a
        return self

    ##############################
    #     merging operations     #
    ##############################

    def merge_to_matrix(self, out_l, out_r, news_l, news_r, inds=None, sort_r=False):
        order = out_l + out_r
        meta_new, meta_mrg, ls_l, ls_r, u_new_l, u_new_r = _meta_merge_to_matrix(self.config, self.struct, out_l, out_r, news_l, news_r, inds, sort_r)
        Anew = self.config.backend.merge_to_matrix(self.A, order, meta_new, meta_mrg, self.config.dtype, self.config.device)
        return Anew, ls_l, ls_r, u_new_l, u_new_r

    def unmerge_from_matrix(self, A, ls_l, ls_r):
        meta = []
        for il, ir in product(ls_l.dec, ls_r.dec):
            ic = il + ir
            if ic in A:
                for (tl, (sl, _, Dl)), (tr, (sr, _, Dr)) in product(ls_l.dec[il].items(), ls_r.dec[ir].items()):
                    meta.append((tl + tr, ic, sl, sr, Dl+Dr))
        return self.config.backend.unmerge_from_matrix(A, meta)

    def unmerge_from_diagonal(self, A, ls):
        meta = tuple((ta + ta, ia, sa) for ia in ls.dec for ta, (sa, _, _) in ls.dec[ia].items())
        Anew = self.config.backend.unmerge_from_diagonal(A, meta)
        return {ind: self.config.backend.diag_create(Anew[ind]) for ind in Anew}

    ##########################
    #    fusing operations   #
    ##########################

    def fuse_legs(self, axes, inplace=False):
        r"""
        Permutes tensor legs. Next, fuse groups of consecutive legs into new meta legs.

        Parameters
        ----------
        axes: tuple
            tuple of leg's indices for transpose. Groups of legs to be fused together form inner tuples.

        Returns
        -------
        tensor : Tensor

        Example
        -------
        tensor.fuse_legs(axes=(2, 0, (1, 4), 3)) gives 4 efective legs from original 5; with one metaly non-trivial one
        tensor.fuse_legs(axes=((2, 0), (1, 4), (3, 5))) gives 3 effective legs from original 6
        """
        if self.isdiag:
            raise YastError('Cannot group legs of a diagonal tensor')

        meta_fusion, order = [], []
        for group in axes:
            if isinstance(group, int):
                order.append(group)
                meta_fusion.append(self.meta_fusion[group])
            else:
                if not all(isinstance(x, int) for x in group):
                    raise YastError('Inner touples of axes can only contain integers')
                order.extend(group)
                nlegs = [sum(self.meta_fusion[ii][0] for ii in group)]
                for ii in group:
                    nlegs.extend(self.meta_fusion[ii])
                meta_fusion.append(tuple(nlegs))
        order = tuple(order)
        if inplace and order == tuple(ii for ii in range(self.mlegs)):
            a = self
        else:
            a = self.transpose(axes=order, inplace=inplace)
        a.meta_fusion = tuple(meta_fusion)
        a.mlegs = len(a.meta_fusion)
        return a

    def unfuse_legs(self, axes, inplace=False):
        """
        Unfuse meta legs reverting one layer of fusion. Operation can be done in-place.

        New legs are inserted in place of the unfused one.

        Parameters
        ----------
        axis: int or tuple of ints
            leg(s) to ungroup.

        Returns
        -------
        tensor : Tensor
        """
        if isinstance(axes, int):
            axes = (axes,)
        a = self if inplace else self.clone()
        new_meta_fusion = []
        for ii in range(a.mlegs):
            if ii not in axes or a.meta_fusion[ii][0] == 1:
                new_meta_fusion.append(a.meta_fusion[ii])
            else:
                stack = a.meta_fusion[ii]
                lstack = len(stack)
                pos_init, cum = 1, 0
                for pos in range(1, lstack):
                    if cum == 0:
                        cum = stack[pos]
                    if stack[pos] == 1:
                        cum = cum - 1
                        if cum == 0:
                            new_meta_fusion.append(stack[pos_init : pos + 1])
                            pos_init = pos + 1
                # new_meta_fusion.extend(a.meta_fusion[ii][1])
        a.meta_fusion = tuple(new_meta_fusion)
        a.mlegs = len(a.meta_fusion)
        return a

    #################
    #     tests     #
    #################

    def is_independent(self, other):
        """
        Test if all elements of two tensors are independent objects in memory.
        """
        test = []
        test.append(self is other)
        test.append(self.A is other.A)
        test.append(self.n is other.n)
        test.append(self.s is other.s)
        for key in self.A.keys():
            if key in other.A:
                test.append(self.config.backend.is_independent(self.A[key], other.A[key]))
        return not any(test)

    def is_consistent(self):
        """
        Test:
        1) tset and Dset correspond to A
        2) tset follow symmetry rule f(s@t)==n
        3) block dimensions are consistent (this requires config.test=True)
        """
        test = []
        for ind, D in zip(self.struct.t, self.struct.D):
            test.append(ind in self.A)
            test.append(self.config.backend.get_shape(self.A[ind]) == D)
        test.append(len(self.struct.t) == len(self.A))
        test.append(len(self.struct.D) == len(self.A))

        tset = self._tarray()
        test.append(np.all(self.config.sym.fuse(tset, self.s, 1) == self.n))
        for n in range(self.nlegs):
            self.get_leg_structure(n, native=True)

        return all(test)

    ########################
    #     aux function     #
    ########################

    def _tarray(self):
        return np.array(self.struct.t, dtype=int).reshape(len(self.struct.t), self.nlegs, self.config.sym.nsym)

    def _Darray(self):
        return np.array(self.struct.D, dtype=int).reshape(len(self.struct.D), self.nlegs)

    def _test_configs_match(self, other):
        # if self.config != other.config:
        if not (self.config.dtype== other.config.dtype \
            and self.config.dtype== other.config.dtype \
            and self.config.sym.name== other.config.sym.name \
            and self.config.backend._backend_id== other.config.backend._backend_id):
            raise YastError('configs do not match')

    def _test_tensors_match(self, other):
        if _check["signatures_match"] and (not all(self.s == other.s) or not all(self.n == other.n)):
            raise YastError('Tensor signatures do not match')
        if _check["consistency"] and not self.meta_fusion == other.meta_fusion:
            raise YastError('Fusion trees do not match')

    def _test_axes_split(self, out_l, out_r):
        if _check["consistency"]:
            if not self.nlegs == len(out_l) + len(out_r):
                raise YastError('Two few indices in axes')
            if not sorted(set(out_l+out_r)) == list(range(self.nlegs)):
                raise YastError('Repeated axis')

    def update_struct(self):
        """Updates meta-information about charges and dimensions of all blocks."""
        d = self.A
        self.A = {k: d[k] for k in sorted(d)}
        t = tuple(self.A.keys())
        D = tuple(self.config.backend.get_shape(x) for x in self.A.values())
        self.struct = _struct(t, D, tuple(self.s), tuple(self.n))


# @lru_cache(maxsize=256)
def _meta_merge_to_matrix(config, struct, out_l, out_r, news_l, news_r, inds, sort_r):
    legs_l = np.array(out_l, np.int)
    legs_r = np.array(out_r, np.int)
    nsym = len(struct.n)
    nleg = len(struct.s)
    told = struct.t if inds is None else [struct.t[ii] for ii in inds]
    Dold = struct.D if inds is None else [struct.D[ii] for ii in inds]
    tset = np.array(told, dtype=int).reshape(len(told), nleg, nsym)
    Dset = np.array(Dold, dtype=int).reshape(len(Dold), nleg)
    t_l = tset[:, legs_l, :]
    t_r = tset[:, legs_r, :]
    D_l = Dset[:, legs_l]
    D_r = Dset[:, legs_r]
    s_l = np.array([struct.s[ii] for ii in out_l], dtype=int)
    s_r = np.array([struct.s[ii] for ii in out_r], dtype=int)
    Deff_l = np.prod(D_l, axis=1)
    Deff_r = np.prod(D_r, axis=1)
    
    te_l = config.sym.fuse(t_l, s_l, news_l)
    te_r = config.sym.fuse(t_r, s_r, news_r)
    tnew = np.hstack([te_l, te_r])

    tnew = tuple(tuple(t.flat) for t in tnew)
    te_l = tuple(tuple(t.flat) for t in te_l)
    te_r = tuple(tuple(t.flat) for t in te_r)
    t_l = tuple(tuple(t.flat) for t in t_l)
    t_r = tuple(tuple(t.flat) for t in t_r)
    D_l = tuple(tuple(x) for x in D_l)
    D_r = tuple(tuple(x) for x in D_r)
    dec_l, Dtot_l = _leg_structure_merge(te_l, t_l, Deff_l, D_l)
    dec_r, Dtot_r = _leg_structure_merge(te_r, t_r, Deff_r, D_r)

    ls_l = _LegDecomposition(config, s_l, news_l)
    ls_r = _LegDecomposition(config, s_r, news_r)
    ls_l.dec = dec_l
    ls_r.dec = dec_r
    ls_l.D = Dtot_l
    ls_r.D = Dtot_r

    meta_mrg = tuple((tn, to, *dec_l[tel][tl][:2], *dec_r[ter][tr][:2]) for tn, to, tel, tl, ter, tr in zip(tnew, told, te_l, t_l, te_r, t_r))
    # meta_mrg = ((tnew, told, Dslc_l, D_l, Dslc_r, D_r), ...)

    if sort_r:
        tt = sorted(set(zip(te_r, te_l, tnew)))
        unew_r, unew_l, unew = zip(*tt)  if len(tt) > 0 else ((), (), ())
    else:
        tt = sorted(set(zip(tnew, te_l, te_r)))
        unew, unew_l, unew_r = zip(*tt) if len(tt) > 0 else ((), (), ())

    meta_new = tuple((u, (ls_l.D[l], ls_r.D[r])) for u, l, r in zip(unew, unew_l, unew_r))
    # meta_new = ((unew, Dnew), ...)
    return meta_new, meta_mrg, ls_l, ls_r, unew_l, unew_r


def _leg_structure_merge(teff, tlegs, Deff, Dlegs):
    tt = sorted(set(zip(teff, tlegs, Deff, Dlegs)))
    dec, Dtot = {}, {}
    for te, grp in groupby(tt, key=itemgetter(0)):
        Dlow = 0
        dec[te] = {}
        for _, tl, De, Dl in grp:
            Dtop = Dlow + De
            dec[te][tl] = ((Dlow, Dtop), De, Dl)
            Dlow = Dtop
        Dtot[te] = Dtop
    return dec, Dtot

class _LegDecomposition:
    """Information about internal structure of leg resulting from fusions."""
    def __init__(self, config=None, s=(), news=1):
        self.s, = _clear_axes(s)  # signature of fused legs
        self.nlegs = len(self.s)  # number of fused legs
        self.config = config
        self.news = news # signature of effective leg
        self.D = {}
        self.dec = {}  # leg's structure/ decomposition

    def match(self, other):
        """ Compare if decomposition match. This does not include signatures."""
        return self.nlegs == other.nlegs and self.D == other.D and self.dec == other.dec

    def copy(self):
        """ Copy leg structure. """
        ls = _LegDecomposition(s=self.s, news=self.news)
        for te, de in self.dec.items():
            ls.dec[te] = de.copy()
        ls.D = self.D.copy()
        return ls

    def show(self):
        """ Print information about leg structure. """
        print("Leg structure: fused = ", self.nlegs)
        for te, de in self.dec.items():
            print(te, ":")
            for to, Do in de.items():
                print("   ",to, ":", Do)

    def leg_struct_for_merged(self, teff, tlegs, Deff, Dlegs):
        """ Calculate meta-information about bond dimensions for merging into one leg. """
        shape_t = list(tlegs.shape)
        shape_t[1] = shape_t[1] + 1
        tcom = np.empty(shape_t, dtype=int)
        tcom[:, 0, :] = teff
        tcom[:, 1:, :] = tlegs
        tcom = tcom.reshape((shape_t[0], shape_t[1]*shape_t[2]))
        ucom, icom = np.unique(tcom, return_index=True, axis=0)
        Dlow = 0
        for ii, tt in zip(icom, ucom):
            t0 = tuple(tt[:self.config.sym.nsym])
            t1 = tuple(tt[self.config.sym.nsym:])
            if t0 not in self.dec:
                self.dec[t0] = {}
                Dlow = 0
            Dtop = Dlow + Deff[ii]
            self.dec[t0][t1] = ((Dlow, Dtop), Deff[ii], tuple(Dlegs[ii]))
            Dlow = Dtop
            self.D[t0] = Dtop

    def leg_struct_trivial(self, A, axis):
        """ Meta-information for single leg. """
        nsym = self.config.sym.nsym
        for ind, val in A.items():
            t = ind[nsym * axis: nsym*(axis + 1)]
            D = self.config.backend.get_shape(val)[axis]
            self.dec[t] = {t: ((0, D), D, (D,))}

    def leg_struct_for_truncation(self, A, opts, sorting='svd'):
        r"""Gives slices for truncation of 1d matrices according to tolerance, D_block, D_total.

        A should be dict of ordered 1d arrays.
        Sorting gives information about ordering outputed by a particular splitting funcion:
        Usual convention is that for svd A[ind][0] is largest; and for eigh A[ind][-1] is largest.
        """
        maxS = 0 if len(A) == 0 else self.config.backend.maximum(A)
        Dmax, D_keep = {}, {}
        for ind in A:
            Dmax[ind] = self.config.backend.get_size(A[ind])
            D_keep[ind] = min(opts['D_block'], Dmax[ind])
        if (opts['tol'] > 0) and (maxS > 0):  # truncate to relative tolerance
            for ind in D_keep:
                D_keep[ind] = min(D_keep[ind], self.config.backend.count_greater(A[ind], maxS * opts['tol']))
        if sum(D_keep[ind] for ind in D_keep) > opts['D_total']:  # truncate to total bond dimension
            if 'keep_multiplets' in opts.keys():
                order = self.config.backend.select_global_largest(A, D_keep, opts['D_total'], \
                    sorting, keep_multiplets=opts['keep_multiplets'], eps_multiplet=opts['eps_multiplet'])
            else:
                order = self.config.backend.select_global_largest(A, D_keep, opts['D_total'], sorting)
            low = 0
            for ind in D_keep:
                high = low + D_keep[ind]
                D_keep[ind] = sum((low <= order) & (order < high))
                low = high

        # check symmetry related blocks and truncate to equal length
        if 'keep_multiplets' in opts.keys() and opts['keep_multiplets']:
            ind_list= [np.asarray(k) for k in D_keep]
            for ind in ind_list:
                t= tuple(ind)
                tn= tuple(-ind)
                minD_sector= min(D_keep[t],D_keep[tn])
                D_keep[t]=D_keep[tn]= minD_sector
                if -ind in ind_list:
                    ind_list.remove(-ind)

        for ind in D_keep:
            if D_keep[ind] > 0:
                Dslc = self.config.backend.range_largest(D_keep[ind], Dmax[ind], sorting)
                self.dec[ind] = {ind: (Dslc, D_keep[ind], (D_keep[ind],))}



# def group_legs(self, axes, new_s=None):
#     """
#     Permutes tensor legs. Next, fuse a specified group of legs into a new single leg.

#     Parameters
#     ----------
#     axes: tuple
#         tuple of leg indices for transpose. Group of legs to be fused forms inner tuple.
#         If there is not internal tuple, fuse given indices and a new leg is placed at the position of the first fused oned.

#     new_s: int
#         signature of a new leg. If not given, the signature of the first fused leg is given.

#     Returns
#     -------
#     tensor : Tensor

#     Example
#     -------
#     For tensor with 5 legs: tensor.fuse_legs1(axes=(2, 0, (1, 4), 3))
#     tensor.fuse_legs1(axes=(2, 0)) is equivalent to tensor.fuse_legs1(axes=(1, (2, 0), 3, 4))
#     """
#     if self.isdiag:
#         raise YastError('Cannot group legs of a diagonal tensor')

#     ituple = [ii for ii, ax in enumerate(axes) if isinstance(ax, tuple)]
#     if len(ituple) == 1:
#         ig = ituple[0]
#         al, ag, ar = axes[:ig], axes[ig], axes[ig+1:]
#     elif len(ituple) == 0:
#         al = tuple(ii for ii in range(axes[0]) if ii not in axes)
#         ar = tuple(ii for ii in range(axes[0]+1, self.nlegs) if ii not in axes)
#         ag = axes
#         ig = len(al)
#     else:
#         raise YastError('Too many groups to fuse')
#     if len(ag) < 2:
#         raise YastError('Need at least two legs to fuse')

#     order = al+ag+ar  # order for permute
#     legs_l = np.array(al, dtype=np.intp)
#     legs_r = np.array(ar, dtype=np.intp)
#     legs_c = np.array(ag, dtype=np.intp)

#     if new_s is None:
#         new_s = self.s[ag[0]]

#     new_ndim = len(al) + 1 + len(ar)

#     t_grp = self.tset[:, legs_c, :]
#     D_grp = self.Dset[:, legs_c]
#     s_grp = self.s[legs_c]
#     t_eff = self.config.sym.fuse(t_grp, s_grp, new_s)
#     D_eff = np.prod(D_grp, axis=1)

#     D_rsh = np.empty((len(self.A), new_ndim), dtype=int)
#     D_rsh[:, :ig] = self.Dset[:, legs_l]
#     D_rsh[:, ig] = D_eff
#     D_rsh[:, ig+1:] = self.Dset[:, legs_r]

#     ls_c = _LegDecomposition(self.config, s_grp, new_s)
#     ls_c.leg_struct_for_merged(t_eff, t_grp, D_eff, D_grp)

#     t_new = np.empty((len(self.A), new_ndim, self.config.sym.nsym), dtype=int)
#     t_new[:, :ig, :] = self.tset[:, legs_l, :]
#     t_new[:, ig, :] = t_eff
#     t_new[:, ig+1:, :] = self.tset[:, legs_r, :]

#     t_new = t_new.reshape(len(t_new), -1)
#     u_new, iu_new = np.unique(t_new, return_index=True, axis=0)
#     Du_new = D_rsh[iu_new]
#     Du_new[:, ig] = np.array([ls_c.D[tuple(t_eff[ii].flat)] for ii in iu_new], dtype=int)

#     # meta_new = ((u, Du), ...)
#     meta_new = tuple((tuple(u.flat), tuple(Du)) for u, Du in zip(u_new, Du_new))
#     # meta_mrg = ((tn, Ds, to, Do), ...)
#     meta_mrg = tuple((tuple(tn.flat), ls_c.dec[tuple(te.flat)][tuple(tg.flat)][0], tuple(to.flat), tuple(Do))
#         for tn, te, tg, to, Do in zip(t_new, t_eff, t_grp, self.tset, D_rsh))

#     c = self.empty(s=tuple(self.s[legs_l]) + (new_s,) + tuple(self.s[legs_r]), n=self.n, isdiag=self.isdiag)
#     c.A = self.config.backend.merge_one_leg(self.A, ig, order, meta_new , meta_mrg, self.config.dtype)
#     c.update_struct()
#     c.lss[ig] = ls_c
#     for nnew, nold in enumerate(al+ (-1,) + ar):
#         if nold in self.lss:
#             c.lss[nnew] = self.lss[nold].copy()
#     return c

# def ungroup_leg(self, axis):
#     """
#     Unfuse a single tensor leg.

#     New legs are inserted in place of the unfused one.

#     Parameters
#     ----------
#     axis: int
#         index of leg to ungroup.

#     Returns
#     -------
#     tensor : Tensor
#     """
#     try:
#         ls = self.lss[axis]
#     except KeyError:
#         return self

#     meta = []
#     for tt, DD in zip(self.tset, self.Dset):
#         tl = tuple(tt[:axis, :].flat)
#         tc = tuple(tt[axis, :].flat)
#         tr = tuple(tt[axis+1:, :].flat)
#         told = tuple(tt.flat)
#         Dl = tuple(DD[:axis])
#         Dr = tuple(DD[axis+1:])
#         for tcom, (Dsl, _, Dc) in ls.dec[tc].items():
#             tnew = tl + tcom + tr
#             Dnew = Dl + Dc + Dr
#             meta.append((told, tnew, Dsl, Dnew))
#     meta = tuple(meta)
#     s = tuple(self.s[:axis]) + ls.s + tuple(self.s[axis+1:])

#     c = self.empty(s=s, n=self.n, isdiag=self.isdiag)
#     c.A = self.config.backend.unmerge_one_leg(self.A, axis, meta)
#     c.update_struct()
#     for ii in range(axis):
#         if ii in self.lss:
#             c.lss[ii]=self.lss[ii].copy()
#     for ii in range(axis+1, self.nlegs):
#         if ii in self.lss:
#             c.lss[ii+ls.nlegs]=self.lss[ii].copy()
#     return c


