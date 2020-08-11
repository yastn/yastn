r"""
Tensor with abelian symmetries.

This class defines generic arbitrary rank tensor supporting abelian symmetries.
In principle, any number of symmetries can be used -- including no symmetries.
An instance of a Tensor is defined by list of blocks (dense tensors)
with their individual dimensions labeled by the symmetries' charges.
"""

import numpy as np
import itertools
from functools import wraps

# defaults and defining symmetries
_large_int = 1073741824


def _sym_U1(x):
    """
    Defines function f for U1 symmetry, where allowed charges should satisfy f(s * t - n) == 0.
    """
    return x


def _sym_Z2(x):
    """
    Defines function f for Z2 symmetry, where allowed charges should satisfy f(s * t - n) == 0.
    """
    return np.mod(x, 2)


def _argsort(tset, nsym):
    """
    Auxliary function supporting sorting of tset.
    """
    if nsym == 0:
        return np.array([0], dtype=int)
    elif nsym == 1:
        return tset[:, 0].argsort()
    # else:
    return np.lexsort(tset.T[::-1])


_tmod = {'U1': _sym_U1,
         'Z2': _sym_Z2}


# dtype used during conversion to numpy array
_select_dtype = {'float64': np.float64,
                 'complex128': np.complex128}

#######################################################
#     Functions creating and filling in new tensor    #
#######################################################


def rand(settings=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with random numbers from [-1,1].

    Initialize tensor and call :meth:`Tensor.reset_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.reset_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        a random instance of a tensor
    """
    a = Tensor(settings=settings, s=s, n=n, isdiag=isdiag)
    a.reset_tensor(t=t, D=D, val='rand')
    return a


def randR(settings=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with complex random numbers from [-1,1] + 1j * [-1,1].

    Initialize tensor and call :meth:`Tensor.reset_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.reset_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        a random instance of a tensor
    """
    a = Tensor(settings=settings, s=s, n=n, isdiag=isdiag)
    a.reset_tensor(t=t, D=D, val='randR')
    return a


def zeros(settings=None, s=(), n=None, t=(), D=(), **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with zeros.

    Initialize tensor and call :meth:`Tensor.reset_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.reset_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        an instance of a tensor filled with zeros
    """
    a = Tensor(settings=settings, s=s, n=n, isdiag=False)
    a.reset_tensor(t=t, D=D, val='zeros')
    return a


def ones(settings=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.reset_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.reset_tensor` for description.
    D : list
        a list of corresponding bond dimensions

    Returns
    -------
    tensor : tensor
        an instance of a tensor filled with ones
    """
    a = Tensor(settings=settings, s=s, n=n, isdiag=isdiag)
    a.reset_tensor(t=t, D=D, val='ones')
    return a


def eye(settings=None, t=(), D=(), **kwargs):
    r"""
    Initialize diagonal tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.reset_tensor`.

    Parameters
    ----------
    t : list
        a list of charges for each leg, see :meth:`Tensor.reset_tensor` for description.
    D : list
        a list of corresponding bond dimensions

    Returns
    -------
    tensor : tensor
        an instance of diagonal tensor filled with ones
    """
    a = Tensor(settings=settings, isdiag=True)
    a.reset_tensor(t=t, D=D, val='ones')
    return a


def from_dict(settings=None, d={'s': (), 'n': None, 'isdiag': False, 'A': {}}):
    """
    Generate tensor based on information in dictionary d.

    Parameters
    ----------
    settings: module
            configuration with backend, symmetry, etc.

    d : dict
        information about tensor stored with :meth:`Tensor.to_dict`
    """
    a = Tensor(settings=settings, s=d['s'], n=d['n'], isdiag=d['isdiag'])
    for ind in d['A']:
        a.set_block(ts=ind, val=d['A'][ind])
    return a


def match_legs(tensors=None, legs=None, conjs=None, val='ones', isdiag=False):
    r"""
    Initialize tensor matching legs of existing tensors, so that it can be contracted with those tensors.

    Finds all matching symmetry sectors and their bond dimensions and passes it to :meth:`Tensor.reset_tensor`.
    Can creat diagonal tensor by matching to one leg of one other tensor.

    Parameters
    ----------
    tensors: list
        list of tensors -- they should not be diagonal to properly identify signature.
    legs: list
        and their corresponding legs to match
    conjs: list
        if tensors are entering dot as conjugated
    val: str
        'randR', 'rand', 'ones', 'zeros'
    """
    t, D, s = [], [], []
    if conjs is None:
        conjs = len(tensors) * [0]
    for ii, te, cc in zip(legs, tensors, conjs):
        if te.isdiag:
            raise TensorError("One of the tensors is diagonal.")
        lts, lDs = te.get_tD()
        t.append(lts[ii])
        D.append(lDs[ii])
        s.append(te.s[ii] * (2 * cc - 1))
    a = Tensor(settings=tensors[0].conf, s=s, isdiag=isdiag)
    a.reset_tensor(t=t, D=D, val=val)
    return a


def block(td, common_legs):
    """ Assemble new tensor by blocking a set of tensors.

        Parameters
        ----------
        td : dict
            dictionary of tensors {(k,l): tensor at position k,l}.
            Length of tuple should be equall to tensor.ndim - len(common_legs)

        common_legs : list
            Legs which are not blocked

        ndim : int
            All tensor should have the same rank ndim
    """
    try:
        ls = len(common_legs)
        out_s = tuple(common_legs)
    except TypeError:
        out_s = (common_legs,)  # indices going u
        ls = 1

    a = next(iter(td.values()))  # first tensor, used to initialize and retrive common values
    ndim = a._ndim
    nsym = a.nsym

    out_m = tuple(ii for ii in range(ndim) if ii not in out_s)
    out_ma = np.array(out_m, dtype=int)
    li = ndim - ls
    pos = []
    for ind in td:
        if li != len(ind):
            raise TensorError('block: wrong tensors rank or placement')
        pos.append(ind)
    pos.sort()

    # all charges and bond dimensions
    tlist, Dlist = {}, {}
    for ind in pos:
        tt, DD = td[ind].get_tD()
        tlist[ind] = tt
        Dlist[ind] = DD

    # all unique blocks to block
    t_out_unique = sorted(set([ind for x in td.values() for ind in x.A]))
    to_execute = []
    for t in t_out_unique:
        ta = np.array(t, dtype=np.int).reshape(ndim, nsym)
        legs_ind = []  # indices on specific legs
        legs_D = []  # and corresponding D
        kk = -1
        for ii in range(ndim):
            tl = tuple(ta[ii].flat)
            relevant_pos = []
            relevant_D = []
            for ind in pos:
                if tl in tlist[ind][ii]:
                    relevant_pos.append(ind)
                    ind_tl = tlist[ind][ii].index(tl)
                    relevant_D.append(Dlist[ind][ii][ind_tl])
            if ii in out_m:
                kk += 1
                posa = np.array(relevant_pos, dtype=int)
                x, y = np.unique(posa[:, kk], return_index=True)
                legs_ind.append(list(x))
                legs_D.append([relevant_D[ll] for ll in y])  # all D's for unique positions should be identical -- does not check this
            else:
                legs_D.append([relevant_D[0]])  # all should be identical -- does not check this
        to_execute.append((t, legs_ind, legs_D))

    c = Tensor(settings=a.conf, s=a.s, isdiag=a.isdiag)
    c.A = c.conf.back.block(td, to_execute, dtype=a.conf.dtype)
    c.tset = np.array([ind for ind in c.A], dtype=np.int).reshape(len(c.A), c._ndim, c.nsym)
    return c


class TensorError(Exception):
    pass


class Tensor:
    """
    Class defining a tensor with abelian symmetries and main operations acting on such tensor(s).
    """

    def __init__(self, settings=None, s=(), n=None, isdiag=False, **kwargs, dtype=None):
        r"""
        Initialize empty Tensor with abelian symmetries.

        Parameters
        ----------
        settings: module
            configuration with backend, symmetry, etc.
        s : tuple, int
            a signature of the tensor; define rank of the tensor; can provide int for ndim == 1
        n : tuple, int
            total charge for each symmetry sectors; default is 0; can provide int for nsym == 1
        isdiag : bool
            makes tensor diagonal: no signature nor charge; keep only diagonal; Defined and stored using _ndim=1,
            but acts like a rank-2 tensor.
        """
        self.conf = settings
        self.isdiag = isdiag
        self.nsym = self.conf.nsym  # number of symmetries
        if not isdiag:
            self._ndim = 1 if isinstance(s, int) else len(s)  # number of legs
            self.s = np.array(s, dtype=np.int).reshape(self._ndim)
            self.n = np.zeros(self.nsym, dtype=np.int) if n is None else np.array(n, dtype=np.int).reshape(self.nsym)
        else:
            self._ndim = 1  # number of legs
            self.s = np.zeros(self._ndim, dtype=np.int)
            self.n = np.zeros(self.nsym, dtype=np.int)
        self.tset = np.empty((0, self._ndim, self.nsym), dtype=np.int)  # list of blocks; 3d nparray of ints
        self.A = {}  # dictionary of blocks
        if dtype:
            self.conf.dtype = dtype

    def copy(self):
        r"""
        Return a copy of the tensor.
        """
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        a.tset = self.tset.copy()
        for ind in self.A:
            a.A[ind] = self.conf.back.copy(self.A[ind])
        return a

    ################
    #  fill tensor #
    ################

    def reset_tensor(self, t=(), D=(), val='rand'):
        r"""
        Create all possible blocks based on s, n and list of charges for all legs.

        Brute-force check all possibilities and select the ones satisfying s * t = n for each symmetry.
        Initialize each possible block with sizes given by D.

        Parameters
        ----------
        t : list
            list of charges on all legs
            If nsym == 0, it is not taken into account.
            Otherwise, two formats are accepted:
            1) All possible charges for leg and symmetry
            t = [(leg1sym1, ...), (leg1sym2, ...), (leg2sym1, ...), (leg2sym2, ...), ... ]
            2) All possible combination of charges for each leg:
            t = [[(leg1sym1, leg1sym2), ... ], [(leg2sym1, leg2sym2), ... )]
            They are equivalent for nsym = 1.
            When somewhere there is only one value and it is unambiguous, tuple can typically be replaced by int, see examples.

        D : tuple
            list of bond dimensions on all legs
            If nsym == 0, D = [leg1, leg2, leg3]
            If nsym >= 1 (it should match the form of t)
            When somewhere there is only one value tuple can typically be replaced by int.

        val : str
            'randR', 'rand' (use current dtype float or complex), 'ones', 'zeros'

        Examples
        --------
        D=5 (ndim = 1)
        D=(1, 2, 3) (nsym = 0 ndim = 3)
        t=[0, (-2, 0), (2, 0)] D=[1, (1, 2), (1, 3)] (nsym = 1 ndim = 3)
        t=[0, 0, (-2, 0), (-2, 0), (2, 0), (2, 0)], D=[1, 1, (1, 2), (1, 2), (1, 3), (1, 3)] (nsym = 2 ndim = 3)
        t=[[(0, 0)], [(-2, -2), (0, 0), (-2, 0), (0, -2)], [(2, 2), (0, 0), (2, 0), (0, 2)]] D=[1, (1, 4, 2, 2), (1, 9, 3, 3)] (nsym = 2 ndim = 3)
        The last two give the same structure.
        """
        if isinstance(D, int):
            D = (D,)
        if isinstance(t, int):
            t = (t,)

        if self.nsym == 0:
            if len(D) != self._ndim:
                raise TensorError("Wrong number of elements in D")
            tset = np.zeros((1, self._ndim, self.nsym))
            Dset = np.array(D, dtype=np.int).reshape(1, self._ndim, 1)
        elif self.nsym >= 1:
            if (self._ndim == 1) and isinstance(D[0], int):
                D = (D,)
            if (self._ndim == 1) and isinstance(t[0], int):
                t = (t,)
            D = list(x if isinstance(x, tuple) or isinstance(x, list) else (x, ) for x in D)
            t = list(x if isinstance(x, tuple) or isinstance(x, list) else (x, ) for x in t)

            if len(D) != self._ndim and len(D) != self._ndim * self.nsym:
                print(t, D, self._ndim, self.nsym, len(D))
                raise TensorError("Wrong number of elements in D")
            if len(t) != self._ndim and len(t) != self._ndim * self.nsym:
                raise TensorError("Wrong number of elements in t")
            for x, y in zip(D, t):
                if len(x) != len(y):
                    raise TensorError("t and D do not match")

            all_t = []
            all_D = []
            if len(t) > self._ndim:
                for ss in range(self.nsym):
                    rr = slice(ss, self.nsym * self._ndim, self.nsym)
                    comb_t = np.array(list(itertools.product(*t[rr])), dtype=np.int)
                    ind = (_tmod[self.conf.sym[ss]](comb_t @ self.s - self.n[ss]) == 0)
                    all_t.append(comb_t[ind])
                    comb_D = np.array(list(itertools.product(*D[rr])), dtype=np.int)
                    all_D.append(comb_D[ind])
                tset = [ins for ins in itertools.product(*all_t)]
                Dset = [ins for ins in itertools.product(*all_D)]
                tset = np.array(tset, dtype=np.int).transpose(0, 2, 1)
                Dset = np.array(Dset, dtype=np.int).transpose(0, 2, 1)
            else:
                comb_t = list(itertools.product(*t))
                comb_D = list(itertools.product(*D))
                lcomb_t = len(comb_t)
                comb_t = np.array(comb_t, dtype=np.int).reshape(lcomb_t, self._ndim, self.nsym)
                comb_D = np.array(comb_D, dtype=np.int).reshape(lcomb_t, self._ndim, 1)
                t_cut = np.zeros((lcomb_t, self.nsym), dtype=np.int)
                for ss in range(self.nsym):
                    t_cut[:, ss] = _tmod[self.conf.sym[ss]](comb_t[:, :, ss] @ self.s - self.n[ss])
                ind = (np.sum(np.abs(t_cut), axis=1) == 0)
                tset = comb_t[ind]
                Dset = comb_D[ind]

        for ind, Ds in zip(tset, Dset):
            ind, Ds = tuple(ind.flat), tuple(np.prod(Ds, axis=1))
            if val == 'zeros':
                self.A[ind] = self.conf.back.zeros(Ds, dtype=self.conf.dtype)
            elif val == 'randR':
                self.A[ind] = self.conf.back.randR(Ds, dtype=self.conf.dtype)
            elif val == 'rand':
                self.A[ind] = self.conf.back.rand(Ds, dtype=self.conf.dtype)
            elif val == 'ones':
                self.A[ind] = self.conf.back.ones(Ds, dtype=self.conf.dtype)
        self.tset = tset

    def set_block(self, ts=(), Ds=None, val='zeros'):
        """
        Add new block to tensor or change the existing one.

        Checks if bond dimensions of the new block are consistent with the existing ones.

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

        if (len(ts) != self._ndim * self.nsym) or (Ds is not None and len(Ds) != self._ndim):
            raise TensorError('Wrong size of input')

        ats = np.array(ts, dtype=np.int).reshape(1, self._ndim, self.nsym)
        for ss in range(self.nsym):
            if not (_tmod[self.conf.sym[ss]](ats[0, :, ss] @ self.s - self.n[ss]) == 0):
                raise TensorError('Charges do not fit the tensor: t @ s != n')

        lts, lDs = self.get_tD()
        existing_D = []
        no_existing_D = False
        for ii in range(self._ndim):
            try:
                existing_D.append(lDs[ii][lts[ii].index(tuple(ats[0, ii, :].flat))])
            except ValueError:
                existing_D.append(-1)
                no_existing_D = True

        if ts not in self.A:
            self.tset = np.vstack([self.tset, ats])

        if isinstance(val, str):
            if Ds is None:
                if no_existing_D:
                    raise TensorError('Not all dimensions specify')
                Ds = existing_D
            else:
                for D1, D2 in zip(Ds, existing_D):
                    if (D1 != D2) and (D2 != -1):
                        raise TensorError('Dimension of the new block does not match the existing ones')
            Ds = tuple(Ds)
            if val == 'zeros':
                self.A[ts] = self.conf.back.zeros(Ds, dtype=self.conf.dtype)
            elif val == 'randR':
                self.A[ts] = self.conf.back.randR(Ds, dtype=self.conf.dtype)
            elif val == 'rand':
                self.A[ts] = self.conf.back.rand(Ds, dtype=self.conf.dtype)
            elif val == 'ones':
                self.A[ts] = self.conf.back.ones(Ds, dtype=self.conf.dtype)
        else:
            self.A[ts] = self.conf.back.to_tensor(val, Ds, dtype=self.conf.dtype)
            Ds = self.conf.back.get_shape(self.A[ts])
            for D1, D2 in zip(Ds, existing_D):
                if (D1 != D2) and (D2 != -1):
                    raise TensorError('Dimension of a new block does not match the existing ones')

    ###########################
    #       new tensors       #
    ###########################

    def empty(self, s=None, n=None, isdiag=False, **kwargs):
        r"""
        Initialize a new tensor using the same settings and symmetries as the current one.

        Parameters
        ----------
        settings: module
            include backend and define symmetry
        s : tuple
            a signature of tensor
        n : int
            total charges in all symmetry sectors
        isdiag : bool
            makes tensor diagonal
        Returns
        -------
        tensor : Tensor
            empty tensor
        """
        return Tensor(settings=self.conf, s=s, n=n, isdiag=isdiag)

    def match_legs(self, **kwargs):
        """
        Wraper to :meth:`match_legs`.
        """
        return match_legs(**kwargs)

    def from_dict(self, **kwargs):
        r"""
        Wraper to :meth:`from_dict`, passing the settings.
        """
        return from_dict(settings=self.conf, **kwargs)

    def rand(self, **kwargs):
        r"""
        Wraper to :meth:`rand`, passing the settings.
        """
        return rand(settings=self.conf, **kwargs)

    def zeros(self, **kwargs):
        r"""
        Wraper to :meth:`zeros`, passing the settings.
        """
        return zeros(settings=self.conf, **kwargs)

    def ones(self, **kwargs):
        r"""
        Wraper to :meth:`ones`, passing the settings.
        """
        return ones(settings=self.conf, **kwargs)

    def eye(self, **kwargs):
        r"""
        Wraper to :meth:`eye`, passing the settings.
        """
        return eye(settings=self.conf, **kwargs)

    ###########################
    #     output functions    #
    ###########################

    def to_dict(self):
        r"""
        Export relevant information about tensor to dictionary -- so that it can be saved using numpy.save

        Returns
        -------
        d: dict
            dictionary containing all the information needed to recreate the tensor.
        """
        AA = {ind: self.conf.back.to_numpy(self.A[ind]) for ind in self.A}
        out = {'A': AA, 's': self.s, 'n': self.n, 'isdiag': self.isdiag}
        return out

    def __str__(self):
        return self.conf.sym + ' s=' + str(self.s) + ' n=' + str(self.n)

    def show_properties(self):
        r"""
        Display basic properties of the tensor.
        """
        print("symmetry     : ", self.conf.sym)
        print("ndim         : ", self.ndim)
        print("sign         : ", self.s)
        print("charge       : ", self.n)
        print("isdiag       : ", self.isdiag)
        print("no. of blocks: ", len(self.A))
        print("size         : ", self.get_size())
        print("tset shape   : ", self.tset.shape)
        print("charges      : ", self.get_t())
        lts, lDs = self.get_tD()
        print("leg charges  : ", lts)
        print("dimensions   : ", lDs)
        print("total dim    : ", [sum(xx) for xx in lDs], "\n")

    def get_size(self):
        """ Total number of elements in tensor."""
        return sum(self.conf.back.get_size(A) for A in self.A.values())

    def get_total_charge(self):
        """ Global charges of the tensor tensor."""
        return self.n

    def get_signature(self):
        """ Tensor signature."""
        return self.s

    def get_charges(self):
        """ Charges of all blocks."""
        return self.tset

    def get_shape(self):
        """ Shapes fo all blocks. """
        shape = []

        for ind in self.tset:
            shape.append(self.conf.back.get_shape(self.A[tuple(ind.flat)]))
        return shape

    @property
    def ndim(self):
        """
        Number of legs.

        Since diagonal tensor is kept using 1d structure self._ndim is dimension of blocks keeping data.
        """
        return 2 if self.isdiag else self._ndim

    def get_nsym(self):
        """ Number of symmetries"""
        return self.nsym

    def get_t(self):
        """
        Find all unique charges for each symmetry on all legs.

        Returns
        -------
            ts : list
                format is [[(sym1 leg1), (sym1 leg2), ...], [(sym2 leg1), ... ], ...]
        """
        ts = []
        for ss in range(self.nsym):
            ts.append([sorted(set(self.tset[:, ll, ss])) for ll in range(self._ndim)])
        return ts

    def get_tD(self):
        """
        Find all charges configurations for each legs and the corresponding block dimensions.

        Returns
        -------
            lts : list
                format is [[charges on leg1], [charges on leg2], ...]

            lDs : list
                format is [[block dimensions on leg1], [block dimensions on leg1], ...]
        """
        tset, Dset = [], []

        Dset = []
        for ind in self.tset:
            ind = tuple(ind.flat)
            Dset.append(self.conf.back.get_shape(self.A[ind]))

        lts, lDs = [], []
        for ii in range(self._ndim):
            tt = [tuple(t[ii].flat) for t in self.tset]
            DD = [D[ii] for D in Dset]
            TD = sorted(set(zip(tt, DD)))
            lts.append(tuple(ss[0] for ss in TD))
            lDs.append(tuple(ss[1] for ss in TD))
        return lts, lDs

    def to_numpy(self):
        """
        Create full nparray corresponding to the tensor.

        Returns
        -------
            out : nparray.
        """
        lts, lDs = self.get_tD()
        Dtotal = [sum(Ds) for Ds in lDs]
        a = np.zeros(Dtotal, dtype=_select_dtype[self.conf.dtype])
        for ind in self.tset:  # fill in the blocks
            sl = []
            for leg, t in enumerate(ind):
                t = tuple(t)
                ii = lts[leg].index(t)
                Dleg = sum(lDs[leg][:ii])
                sl.append(slice(Dleg, Dleg + lDs[leg][ii]))
            temp = self.conf.back.to_numpy(self.A[tuple(ind.flat)])
            if sl:
                a[tuple(sl)] = temp
            else:  # should only happen for 0-dim tensor -- i.e.  a scalar
                a = temp
        if self.isdiag:
            a = np.diag(a)
        return a

    def to_number(self):
        """
        Return first number in the first (unsorted) block.
        Mainly used for rank-0 tensor with 1 block of size 1.

        Return 0 if there are no blocks.
        """
        if len(self.A) > 0:
            key = next(iter(self.A))
            return self.conf.back.first_el(self.A[key])
        else:
            return 0.

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
            new tensor with the result of multipcilation.
        """
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        a.tset = self.tset.copy()
        for ind in self.A:
            a.A[ind] = other * self.A[ind]
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
            new tensor with the result of multipcilation.
        """
        return self.__mul__(other)

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
            the result of addition as a new tensor.
        """
        if not all(self.s == other.s) or not all(self.n == other.n):
            raise TensorError('Tensor signatures do not match')
        to_execute = []
        tset = self.tset.copy()
        new_tset = []
        for ind in self.A:
            if ind in other.A:
                to_execute.append((ind, 0))
            else:
                to_execute.append((ind, 1))
        for ind in other.A:
            if ind not in self.A:
                to_execute.append((ind, 2))
                new_tset.append(ind)
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        if len(new_tset) > 0:
            tset = np.vstack([tset, np.array(new_tset, dtype=np.int).reshape((-1, self._ndim, self.nsym))])
        a.tset = tset
        a.A = a.conf.back.add(self.A, other.A, to_execute)
        return a

    def axpb(self, other, x=1):
        """
        Directly calculate tensor + x * tensor

        Signatures and total charges should match.

        Parameters
        ----------
        other: Tensor
        x : number

        Returns
        -------
        tensor : Tensor
            the result of addition as a new tensor.
        """
        if not all(self.s == other.s) or not all(self.n == other.n):
            raise TensorError('Tensors do not match')
        to_execute = []
        tset = self.tset.copy()
        new_tset = []
        for ind in self.A:
            if ind in other.A:
                to_execute.append((ind, 0))
            else:
                to_execute.append((ind, 1))
        for ind in other.A:
            if ind not in self.A:
                to_execute.append((ind, 2))
                new_tset.append(ind)
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        if len(new_tset) > 0:
            tset = np.vstack([tset, np.array(new_tset, dtype=np.int).reshape((-1, self._ndim, self.nsym))])
        a.tset = tset
        a.A = a.conf.back.add(self.A, other.A, to_execute, x)
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
            the result of subtraction as a new tensor.
        """
        if not all(self.s == other.s) or not all(self.n == other.n):
            raise TensorError('Tensors do not match')
        to_execute = []
        tset = self.tset.copy()
        new_tset = []
        for ind in self.A:
            if ind in other.A:
                to_execute.append((ind, 0))
            else:
                to_execute.append((ind, 1))
        for ind in other.A:
            if ind not in self.A:
                to_execute.append((ind, 2))
                new_tset.append(ind)
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        if len(new_tset) > 0:
            tset = np.vstack([tset, np.array(new_tset, dtype=np.int).reshape((-1, self._ndim, self.nsym))])
        a.tset = tset
        a.A = a.conf.back.sub(self.A, other.A, to_execute)
        return a

    def norm(self, ord='fro'):
        """
        Norm of the rensor

        Parameters
        ----------
        ord: str
            'fro' = Frobenious; 'inf' = max(abs())

        Returns
        -------
        norm : float64
        """
        return self.conf.back.norm(self.A, ord=ord)

    def norm_diff(self, other, ord='fro'):
        """
        Norm of the difference of the two tensors

        Parameters
        ----------
        other: Tensor

        ord: str
            'fro' = Frobenious; 'inf' = max(abs())

        Returns
        -------
        norm : float64
        """
        if not all(self.s == other.s):
            raise TensorError('Signs do not match')
        return self.conf.back.norm_diff(self.A, other.A, ord)

    ############################
    #     tensor functions     #
    ############################

    def diag(self, s0=1):
        """
        Select diagonal of 2d tensor and output it as a diagonal tensor, or vice versa.

        Parameters
        ----------
            s0: +1 or -1
                while transforming diagonal tensor into 2d tensor, one has to select signature (s0, -s0)
        """
        if self.isdiag:
            a = Tensor(settings=self.conf, s=(s0, -s0), n=self.n, isdiag=False)
            for ind in self.A:
                nind = ind + ind
                a.set_block(ts=nind, val=self.conf.back.diag_create(self.A[ind]))
        elif self._ndim == 2 and sum(np.abs(self.n)) == 0 and sum(self.s) == 0:
            a = Tensor(settings=self.conf, isdiag=True)
            for ind in self.tset:
                if np.all(ind[0, :] == ind[1, :]):
                    nind = tuple(ind[0, :].flat)
                    a.set_block(ts=nind, val=self.conf.back.diag_get(self.A[tuple(ind.flat)]))
        else:
            raise TensorError('Tensor cannot be changed into a diagonal one')
        return a

    def conj(self):
        """
        Return conjugated tensor.

        Changes sign of signature s and total charge n, as well as complex conjugate each block.

        Returns
        -------
        tensor : Tensor
        """
        a = Tensor(settings=self.conf, s=-self.s, n=-self.n, isdiag=self.isdiag)
        a.tset = self.tset.copy()
        a.A = a.conf.back.conj(self.A)
        return a

    def _transpose_local(self, axes):
        """
        Transpose tensor in place. Intended for internal use and ncon.

        Parameters
        ----------
        axes: tuple
            New order of the legs.
        """
        if (not self.isdiag) and (axes != tuple(range(self._ndim))):
            order = np.array(axes, dtype=np.int)
            self.s = self.s[order]
            tset = self.tset[:, order, :]
            to_execute = []
            for old, new in zip(self.tset, tset):
                to_execute.append((tuple(old.flat), tuple(new.flat)))
            cA = self.conf.back.transpose_local(self.A, axes, to_execute)
            self.tset = tset
            self.A = cA
        return self

    def transpose(self, axes):
        """
        Return transposed tensor.

        Parameters
        ----------
        axes: tuple
            New order of the legs.

        Returns
        -------
        tensor : Tensor
        """
        if not self.isdiag:
            order = np.array(axes, dtype=np.int)
            a = Tensor(settings=self.conf, s=self.s[order], n=self.n, isdiag=self.isdiag)
            a.tset = self.tset[:, order, :]
            to_execute = []
            for old, new in zip(self.tset, a.tset):
                to_execute.append((tuple(old.flat), tuple(new.flat)))
            a.A = a.conf.back.transpose(self.A, axes, to_execute)
            return a
        else:
            return self.copy()

    def group_legs(self, axes, new_s=1):
        """
        Group tensor legs into a single leg.

        Create one new leg, which is placed on the position of the first index in axes. The rest is shifted accordingly.

        Parameters
        ----------
        axes: tuple
            tuple of legs to combine.

        new_s: int
            signeture of a new leg.

        Returns
        -------
        tensor : Tensor
        """
        if self.isdiag:
            raise TensorError('Cannot group legs of diagonal tensor')

        nin = np.array(axes, dtype=np.int)

        rest = tuple(ii for ii in range(self._ndim) if ii not in axes)  # other legs
        nrest = np.array(rest, dtype=np.int)

        axes1 = axes[1:]
        rest1 = tuple(ii for ii in range(self._ndim) if ii not in axes1)  # other legs
        nrest1 = np.array(rest1, dtype=np.int)
        new_axis = rest1.index(axes[0])

        t_cuts = self.tset[:, nin, :].swapaxes(1, 2) @ self.s[nin]
        for ss in range(self.nsym):
            t_cuts[:, ss] = _tmod[self.conf.sym[ss]](new_s * t_cuts[:, ss])

        t_new = self.tset[:, nrest1, :]
        t_new[:, new_axis, :] = t_cuts
        t_in = self.tset[:, nin, :]

        xx_list = sorted((tuple(t1.flat), tuple(t2.flat), tuple(t3.flat), tuple(t4.flat))
                         for t1, t2, t3, t4 in zip(t_cuts, t_in, t_new, self.tset))
        to_execute = []
        for t1, tgroup in itertools.groupby(xx_list, key=lambda x: x[0]):
            to_execute.append((t1, sorted(tgroup)))

        Agrouped, leg_order = self.conf.back.group_legs(self.A, to_execute, axes, rest, new_axis, self.conf.dtype)

        s = self.s[nrest1]
        s[new_axis] = new_s

        leg_order['s'] = tuple(self.s[nin])
        c = Tensor(settings=self.conf, s=s, n=self.n, isdiag=self.isdiag)
        c.A = Agrouped
        c.tset = np.array([ind for ind in c.A], dtype=np.int).reshape(len(c.A), c._ndim, c.nsym)
        return c, leg_order

    def ungroup_leg(self, axis, leg_order):
        """
        Ungroup a single tensor leg.

        New legs are inserted in place of the ungrouped one.

        Parameters
        ----------
        axis: int
            index of leg to ungroup.

        leg_order: tuple
            information about grouped indices generetat by :meth:`Tensor.group_legs`

        Returns
        -------
        tensor : Tensor
        """

        if self.isdiag:
            raise TensorError('Cannot group legs of diagonal tensor')

        to_execute = []
        for t in self.tset:
            tcut = tuple(t[axis, :].flat)
            tl = tuple(t[:axis, :].flat)
            tr = tuple(t[axis + 1:, :].flat)
            t = tuple(t.flat)
            for tout in leg_order[tcut]:
                to_execute.append((tcut, tout, t, tl + tout + tr))

        s = tuple(self.s[:axis]) + leg_order['s'] + tuple(self.s[axis + 1:])
        c = Tensor(settings=self.conf, s=s, n=self.n, isdiag=self.isdiag)
        c.A = self.conf.back.ungroup_leg(self.A, axis, self._ndim, leg_order, to_execute)
        c.tset = np.array([ind for ind in c.A], dtype=np.int).reshape(len(c.A), c._ndim, c.nsym)
        return c

    def swap_gate(self, axes, fermionic=[]):
        """
        Return tensor after application of the swap gate.

        Multiply the block with odd charges on swaped legs by -1.
        If one of the axes is -1, then swap with charge n.

        TEST IT

        Parameters
        ----------
        axes: tuple
            Two legs to swap

        fermionic: tuple
            which symmetries are fermionic.

        Returns
        -------
        tensor : Tensor
        """
        if any(fermionic):
            fermionic = np.array(fermionic, dtype=bool)
            axes = sorted(list(axes))
            a = self.copy()
            if axes[0] == axes[1]:
                raise TensorError('Cannot sweep the same index')
            if not self.isdiag:
                if (axes[0] == -1) and (np.sum(a.n[fermionic]) % 2 == 1):  # swap gate with local a.n
                    for ind in a.tset:
                        if np.sum(ind[axes[1], fermionic]) % 2 == 1:
                            ind = tuple(ind.flat)
                            a.A[ind] = -a.A[ind]
                else:  # axes[0] != axes[1]:  # swap gate on 2 legs
                    for ind in a.tset:
                        if (np.sum(ind[axes[0], fermionic]) % 2 == 1) and (np.sum(ind[axes[1], fermionic]) % 2 == 1):
                            a.A[ind] = -a.A[ind]
            else:
                for ind in a.tset:
                    if (np.sum(ind[axes[0], fermionic]) % 2 == 1):
                        a.A[ind] = -a.A[ind]
            return a
        else:
            return self

    def invsqrt(self):
        """ Element-wise 1/sqrt(A)"""
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        a.A = self.conf.back.invsqrt(self.A, self.isdiag)
        a.tset = self.tset.copy()
        return a

    def inv(self):
        """ Element-wise 1/sqrt(A)"""
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        a.A = self.conf.back.inv(self.A, self.isdiag)
        a.tset = self.tset.copy()
        return a

    def exp(self, step=1.):
        """ Element-wise exp(step*A)"""
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        a.A = self.conf.back.exp(self.A, step, self.isdiag)
        a.tset = self.tset.copy()
        return a

    def sqrt(self):
        """ Element-wise sqrt"""
        a = Tensor(settings=self.conf, s=self.s, n=self.n, isdiag=self.isdiag)
        a.A = self.conf.back.sqrt(self.A, self.isdiag)
        a.tset = self.tset.copy()
        return a

    def entropy(self, axes=(0, 1), alpha=1):
        """
        Calculate entropy from tensor.

        If diagonal, calculates entropy treating S^2 as probabilities. Normalizes S^2 if neccesary.
        If not diagonal, calculates svd first to get the diagonal S.
        Use log base 2.

        Parameters
        ----------
        axes: tuple
            how to split the tensor for svd

        alpha: float
            Order of Renyi entropy.
            alpha=1 is von Neuman: Entropy -Tr(S^2 log2(S^2))
            otherwise: 1/(1-alpha) log2(Tr(S^(2 alpha)))

        Returns
        -------
            entropy : float64
        """

        if self.isdiag is False:
            try:
                ll = len(axes[0])
                out_l = tuple(axes[0])
            except TypeError:
                out_l = (axes[0],)  # indices going u
                ll = 1
            try:
                lr = len(axes[1])
                out_r = tuple(axes[1])
            except TypeError:
                out_r = (axes[1],)  # indices going v
                lr = 1

            if not (self._ndim == ll + lr):
                raise TensorError('Two few indices in axes')

            # divide charges between l and r
            # order formation of blocks
            nout_r, nout_l = np.array(out_r, dtype=np.int), np.array(out_l, dtype=np.int)

            t_cuts = self.tset[:, nout_l, :].swapaxes(1, 2) @ self.s[nout_l]
            for ss in range(self.nsym):
                t_cuts[:, ss] = _tmod[self.conf.sym[ss]](t_cuts[:, ss])

            t_l = self.tset[:, np.append(nout_l, 0), :]
            t_l[:, -1, :] = t_cuts
            t_r = self.tset[:, np.append(0, nout_r), :]
            t_r[:, 0, :] = t_cuts

            t_sort = _argsort(t_cuts, self.nsym)
            t_cuts = t_cuts[t_sort]
            t_full = self.tset[t_sort]
            t_l = t_l[t_sort]
            t_r = t_r[t_sort]

            xx_list = ((tuple(t2.flat), tuple(t3.flat), tuple(t4.flat)) for t2, t3, t4 in zip(t_l, t_r, t_full))
            to_execute = []
            for t1, tgroup in itertools.groupby(xx_list, key=lambda _, it=iter(t_cuts): tuple(next(it).flat)):
                to_execute.append((t1, sorted(tgroup)))

            # merged blocks; do not need information for unmerging
            Amerged, _, _ = self.conf.back.merge_blocks(self.A, to_execute, out_l, out_r, self.conf.dtype)
            Smerged = self.conf.back.svd_no_uv(Amerged)
        else:
            Smerged = self.A

        ent, Smin, no = self.conf.back.entropy(Smerged, alpha=alpha)
        return ent, Smin, no

    ##################################
    #     contraction operations     #
    ##################################

    def scalar(self, other):
        """ Compute scalar product x=(a, b) of two tensor.

            Note that the first one is conjugated.

            Parameters
            ----------
            other: Tensor

            Returns
            -------
            x: number
        """
        if not all(self.s == other.s):
            raise TensorError('Signs do not match')

        a_set = set([tuple(t.flat) for t in self.tset])
        b_set = set([tuple(t.flat) for t in other.tset])
        common = a_set & b_set
        to_execute = [(t, t, ()) for t in common]
        if len(to_execute) > 0:
            a_out = ()
            a_con = tuple(range(self._ndim))
            b_out = a_out
            b_con = a_con
            A = self.conf.back.dot(self.A, other.A, (1, 0), to_execute, a_out, a_con, b_con, b_out, self.conf.dtype)
            return self.conf.back.first_el(A[()])
        else:
            return 0.

    def trace(self, axes=(0, 1)):
        """
        Compute trace of legs specified by axes.

        For diagonal tensor, return 0-rank tensor
        """
        try:
            in1 = tuple(axes[0])
        except TypeError:
            in1 = (axes[0],)  # indices going u
        try:
            in2 = tuple(axes[1])
        except TypeError:
            in2 = (axes[1],)  # indices going v

        if len(in1) != len(in2):
            raise TensorError('Number of axis to trace should be the same')

        if self.isdiag:
            if in1 == in2 == ():
                return self.copy()
            elif in1 + in2 == (0, 1) or in1 + in2 == (1, 0):
                a = Tensor(settings=self.conf)
                to_execute = []
                for tt in self.tset:
                    to_execute.append((tuple(tt.flat), ()))  # old, new
                a.A = a.conf.back.trace_axis(A=self.A, to_execute=to_execute, axis=0)
            else:
                raise TensorError('Wrong axes for diagonal tensor')
        else:
            out = tuple(ii for ii in range(self._ndim) if ii not in in1 + in2)
            nout = np.array(out, dtype=np.int)
            nin1 = np.array(in1, dtype=np.int)
            nin2 = np.array(in2, dtype=np.int)
            if not all(self.s[nin1] == -self.s[nin2]):
                raise TensorError('Signs do not match')

            to_execute = []
            for tt in self.tset:
                if np.all(tt[nin1, :] == tt[nin2, :]):
                    to_execute.append((tuple(tt.flat), tuple(tt[nout].flat)))  # old, new
            a = Tensor(settings=self.conf, s=self.s[nout], n=self.n)
            a.A = a.conf.back.trace(A=self.A, to_execute=to_execute, in1=in1, in2=in2, out=out)

        a.tset = np.array([ind for ind in a.A], dtype=np.int).reshape(len(a.A), a._ndim, a.nsym)
        return a

    def dot_diag(self, other, axis, conj=(0, 0)):
        r""" Compute dot product of a tensor with a diagonal tensor.

        Other tensors should be diagonal.
        Legs of a new tensor are ordered in the same way as those of the first one.
        Produce diagonal tensor if both are diagonal.

        Parameters
        ----------
        other: Tensor
            diagonal tensor

        axis: int or tuple
            leg of non-diagonal tensor to be multiplied by the diagonal one.

        conj: tuple
            shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1)
        """

        if not other.isdiag:
            raise TensorError('Other tensor should be diagonal.')

        conja = (1 - 2 * conj[0])
        na_con = 0 if self.isdiag else axis if isinstance(axis, int) else axis[0]

        t_a_con = self.tset[:, na_con, :]
        block_a = sorted([(tuple(x.flat), tuple(y.flat)) for x, y in zip(t_a_con, self.tset)], key=lambda x: x[0])
        block_b = sorted([tuple(x.flat) for x in other.tset])
        block_a = itertools.groupby(block_a, key=lambda x: x[0])
        block_b = iter(block_b)

        to_execute = []
        try:
            tta, ga = next(block_a)
            ttb = next(block_b)
            while True:
                if tta == ttb:
                    for ta in ga:
                        to_execute.append((ta[1], ttb, ta[1]))
                    tta, ga = next(block_a)
                    ttb = next(block_b)
                elif tta < ttb:
                    tta, ga = next(block_a)
                elif tta > ttb:
                    ttb = next(block_b)
        except StopIteration:
            pass

        c = Tensor(settings=self.conf, s=conja * self.s, n=conja * self.n, isdiag=self.isdiag)
        c.A = self.conf.back.dot_diag(self.A, other.A, conj, to_execute, na_con, self._ndim)
        c.tset = np.array([ind for ind in c.A], dtype=np.int).reshape(len(c.A), c._ndim, c.nsym)
        return c

    def trace_dot_diag(self, other, axis1=0, axis2=1, conj=(0, 0)):
        r""" Contract two legs of a tensor with a diagonal tensor -- equivalent to dot_diag and trace of the result.

        Other tensor should be diagonal.
        Legs of a new tensor are ordered in the same way as those of the first one.

        Parameters
        ----------
        other: Tensor
            diagonal tensor

        axis1, axis2: int
            2 legs of the first tensor to be multiplied by the diagonal one.
            they are ignored if the first tensor is diagonal.

        conj: tuple
            shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1)
        """
        if self.isdiag:
            return self.dot_diag(other, axis=0, conj=conj).trace()

        if self.s[axis1] != -self.s[axis2]:
            raise TensorError('Signs do not match')

        conja = (1 - 2 * conj[0])
        na_con = np.array([axis1, axis2], dtype=np.int)
        out = tuple(ii for ii in range(self._ndim) if ii != axis1 and ii != axis2)
        nout = np.array(out, dtype=np.int)

        ind = np.all(self.tset[:, axis1, :] == self.tset[:, axis2, :], axis=1)
        tset = self.tset[ind]
        t_a_con = tset[:, axis1, :]
        t_a_out = tset[:, nout, :]

        block_a = sorted([(tuple(x.flat), tuple(y.flat), tuple(z.flat)) for x, y, z in zip(t_a_con, tset, t_a_out)], key=lambda x: x[0])
        block_b = sorted([tuple(x.flat) for x in other.tset])
        block_a = itertools.groupby(block_a, key=lambda x: x[0])
        block_b = iter(block_b)

        to_execute = []
        try:
            tta, ga = next(block_a)
            ttb = next(block_b)
            while True:
                if tta == ttb:
                    for ta in ga:
                        to_execute.append((ta[1], ttb, ta[2]))
                    tta, ga = next(block_a)
                    ttb = next(block_b)
                elif tta < ttb:
                    tta, ga = next(block_a)
                elif tta > ttb:
                    ttb = next(block_b)
        except StopIteration:
            pass

        c = Tensor(settings=self.conf, s=conja * self.s[nout], n=conja * self.n, isdiag=False)
        c.A = self.conf.back.trace_dot_diag(self.A, other.A, conj, to_execute, axis1, axis2, self._ndim)
        c.tset = np.array([ind for ind in c.A], dtype=np.int).reshape(len(c.A), c._ndim, c.nsym)
        return c

    # @profile
    def dot(self, other, axes, conj=(0, 0)):
        r""" Compute dot product of two tensor along specified axes.

            Outgoing legs ordered such that first come remaining legs of the first tensor in the original order,
            and than those of the second tensor.

            Parameters
            ----------
            other: Tensor

            axes: tuple
                legs of both tensors (for each it is specified by int or tuple of ints)
                e.g. axes=(0, 3), axes=((0, 3), (1, 2))

            conj: tuple
                shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1).
        """

        try:
            a_con = tuple(axes[0])  # contracted legs
        except TypeError:
            a_con = (axes[0],)
        try:
            b_con = tuple(axes[1])
        except TypeError:
            b_con = (axes[1],)

        if other.isdiag:
            if len(b_con) == 1:
                c = self.dot_diag(other, axis=a_con[0], conj=conj)
                order = tuple((ii for ii in range(self._ndim) if ii != a_con[0])) + (a_con[0],)
                c._transpose_local(axes=order)
            elif len(b_con) == 2:
                c = self.trace_dot_diag(other, axis1=a_con[0], axis2=a_con[1], conj=conj)
            elif len(b_con) == 0:
                raise TensorError('len(axes[0]) == 0. Do not support outer product with diagonal tensor. Use diag() first.')
            else:
                raise TensorError('To many axis to contract.')
            return c

        if self.isdiag:
            if len(a_con) == 1:
                c = other.dot_diag(self, axis=b_con[0], conj=conj[::-1])
                order = (b_con[0],) + tuple((ii for ii in range(other._ndim) if ii != b_con[0]))
                c._transpose_local(axes=order)
            elif len(a_con) == 2:
                c = other.trace_dot_diag(self, axis1=b_con[0], axis2=b_con[1], conj=conj[::-1])
            elif len(a_con) > 2:
                raise TensorError('len(axes[0]) == 0. Do not support outer product with diagonal tensor. Use diag() first.')
            else:
                raise TensorError('To many axis to contract.')
            return c

        conja = (1 - 2 * conj[0])
        conjb = (1 - 2 * conj[1])

        a_out = tuple(ii for ii in range(self._ndim) if ii not in a_con)  # outgoing legs
        b_out = tuple(ii for ii in range(other._ndim) if ii not in b_con)

        na_con = np.array(a_con, dtype=np.int)
        nb_con = np.array(b_con, dtype=np.int)
        na_out = np.array(a_out, dtype=np.int)
        nb_out = np.array(b_out, dtype=np.int)

        if not all(self.s[na_con] == -other.s[nb_con] * conja * conjb):
            raise TensorError('Signs do not match')

        c_n = conja * self.n + conjb * other.n
        for ss in range(self.nsym):
            c_n[ss] = _tmod[self.conf.sym[ss]](c_n[ss])

        if self.conf.dot_merge:
            t_a_con = self.tset[:, na_con, :]
            t_b_con = other.tset[:, nb_con, :]
            t_a_out = self.tset[:, na_out, :]
            t_b_out = other.tset[:, nb_out, :]
            t_a_cuts = t_a_con.swapaxes(1, 2) @ self.s[na_con]
            t_b_cuts = -t_b_con.swapaxes(1, 2) @ other.s[nb_con] * conja * conjb
            for ss in range(self.nsym):
                t_a_cuts[:, ss] = _tmod[self.conf.sym[ss]](t_a_cuts[:, ss])
                t_b_cuts[:, ss] = _tmod[self.conf.sym[ss]](t_b_cuts[:, ss])

            a_sort = _argsort(t_a_cuts, self.nsym)
            b_sort = _argsort(t_b_cuts, self.nsym)

            t_a_full = self.tset[a_sort]
            t_a_con = t_a_con[a_sort]
            t_a_out = t_a_out[a_sort]
            t_a_cuts = t_a_cuts[a_sort]

            t_b_full = other.tset[b_sort]
            t_b_con = t_b_con[b_sort]
            t_b_out = t_b_out[b_sort]
            t_b_cuts = t_b_cuts[b_sort]

            a_list = [(tuple(t2.flat), tuple(t3.flat), tuple(t4.flat)) for t2, t3, t4 in zip(t_a_out, t_a_con, t_a_full)]
            b_list = [(tuple(t2.flat), tuple(t3.flat), tuple(t4.flat)) for t2, t3, t4 in zip(t_b_con, t_b_out, t_b_full)]

            a_iter = itertools.groupby(a_list, key=lambda _, ia=iter(t_a_cuts): tuple(next(ia).flat))
            b_iter = itertools.groupby(b_list, key=lambda _, ib=iter(t_b_cuts): tuple(next(ib).flat))

            to_merge_a, to_merge_b = [], []
            try:
                cut_a, ga = next(a_iter)
                cut_b, gb = next(b_iter)
                while True:
                    if cut_a < cut_b:
                        cut_a, ga = next(a_iter)
                    elif cut_a > cut_b:
                        cut_b, gb = next(b_iter)
                    else:
                        ga_iter = itertools.groupby(sorted(ga, key=lambda x: x[1]), key=lambda x: x[1])
                        gb_iter = itertools.groupby(sorted(gb, key=lambda x: x[0]), key=lambda x: x[0])
                        try:
                            con_a, la = next(ga_iter)
                            con_b, lb = next(gb_iter)
                            lla, llb = [], []
                            while True:
                                if con_a < con_b:
                                    con_a, la = next(ga_iter)
                                elif con_a > con_b:
                                    con_b, lb = next(gb_iter)
                                else:
                                    lla += list(la)
                                    llb += list(lb)
                                    con_a, la = next(ga_iter)
                                    con_b, lb = next(gb_iter)
                        except StopIteration:
                            pass
                        if lla:
                            to_merge_a.append((cut_a, sorted(lla)))
                            to_merge_b.append((cut_b, sorted(llb)))
                        cut_a, ga = next(a_iter)
                        cut_b, gb = next(b_iter)
            except StopIteration:
                pass

            # merged blocks and information for un-merging
            Amerged, order_l, _ = self.conf.back.merge_blocks(self.A, to_merge_a, a_out, a_con, self.conf.dtype)
            # merged blocks and information for un-merging
            Bmerged, _, order_r = self.conf.back.merge_blocks(other.A, to_merge_b, b_con, b_out, self.conf.dtype)

            Cmerged = self.conf.back.dot_merged(Amerged, Bmerged, conj)

            c_s = np.hstack([conja * self.s[na_out], conjb * other.s[nb_out]])
            c = Tensor(settings=self.conf, s=c_s, n=c_n)
            c.A = self.conf.back.unmerge_blocks(Cmerged, order_l, order_r)
            c.tset = np.array([ind for ind in c.A], dtype=np.int).reshape(len(c.A), c._ndim, c.nsym)
            return c
        else:
            t_a_con = self.tset[:, na_con, :]
            t_b_con = other.tset[:, nb_con, :]
            t_a_out = self.tset[:, na_out, :]
            t_b_out = other.tset[:, nb_out, :]

            block_a = sorted([(tuple(x.flat), tuple(y.flat), tuple(z.flat)) for x, y, z in zip(t_a_con, t_a_out, self.tset)], key=lambda x: x[0])
            block_b = sorted([(tuple(x.flat), tuple(y.flat), tuple(z.flat)) for x, y, z in zip(t_b_con, t_b_out, other.tset)], key=lambda x: x[0])

            block_a = itertools.groupby(block_a, key=lambda x: x[0])
            block_b = itertools.groupby(block_b, key=lambda x: x[0])

            to_execute = []
            try:
                tta, ga = next(block_a)
                ttb, gb = next(block_b)
                while True:
                    if tta == ttb:
                        for ta, tb in itertools.product(ga, gb):
                            to_execute.append((ta[2], tb[2], ta[1] + tb[1]))
                        tta, ga = next(block_a)
                        ttb, gb = next(block_b)
                    elif tta < ttb:
                        tta, ga = next(block_a)
                    elif tta > ttb:
                        ttb, gb = next(block_b)
            except StopIteration:
                pass

            c_s = np.hstack([conja * self.s[na_out], conjb * other.s[nb_out]])
            c = Tensor(settings=self.conf, s=c_s, n=c_n)
            c.A = self.conf.back.dot(self.A, other.A, conj, to_execute, a_out, a_con, b_con, b_out, self.conf.dtype)
            c.tset = np.array([ind for ind in c.A], dtype=np.int).reshape(len(c.A), c._ndim, c.nsym)
            return c

    ###########################
    #     spliting tensor     #
    ###########################

    def split_svd(self, axes, sU=1, Uaxis=-1, Vaxis=0, tol=0, D_block=_large_int, D_total=_large_int, truncated_svd=False, truncated_nbit=60, truncated_kfac=6):
        r"""
        Split tensor using svd, tensor = U * S * V. Truncate smallest singular values if neccesary.

        Truncate using (whichever gives smaller bond dimension) relative tolerance, bond dimension of each block, and total bond dimension from all blocks.
        By default do not truncate. Charge divided between U and V as n_u = n+1//2 and n_v = n//2, respectively.

        Parameters
        ----------
        axes: tuple
            Specify two groups of legs between which to perform svd, as well as their final order.

        sU: int
            signature of connecting leg in U equall 1 or -1. Default is 1.

        Uaxis, Vaxis: int
            specify which leg of U and V tensors are connecting with S. By delault it is the last leg of U and the first of V.

        tol: float
            relative tolerance of singular values below which to truncate.

        D_block: int
            largest number of singular values to keep in a single block.

        D_total: int
            largest total number of singular values to keep.

        truncated_svd: bool
            flag to employ truncated-svd algorithm.

        truncated_nbit, truncated_kfac: int
            parameters of the truncated-svd algorithm.

        Returns
        -------
            U, S, V: Tensor
                U and V are unitary projectors. S is diagonal.
        """
        try:
            ll = len(axes[0])
            out_l = tuple(axes[0])
        except TypeError:
            out_l = (axes[0],)  # indices going u
            ll = 1
        try:
            lr = len(axes[1])
            out_r = tuple(axes[1])
        except TypeError:
            out_r = (axes[1],)  # indices going v
            lr = 1

        if not (self._ndim == ll + lr):
            raise TensorError('Two few indices in axes')
        elif not (sorted(set(out_l + out_r)) == list(range(self._ndim))):
            raise TensorError('Repeated axis')

        # divide charges between l and r
        n_l, n_r = np.zeros(self.nsym, dtype=int), np.zeros(self.nsym, dtype=int)
        for ss in range(self.nsym):
            n_l[ss] = _tmod[self.conf.sym[ss]]((self.n[ss] + 1) // 2)
            n_r[ss] = _tmod[self.conf.sym[ss]](self.n[ss] // 2)

        # order formation of blocks
        nout_r, nout_l = np.array(out_r, dtype=np.int), np.array(out_l, dtype=np.int)

        t_cuts = self.tset[:, nout_l, :].swapaxes(1, 2) @ self.s[nout_l]
        for ss in range(self.nsym):
            t_cuts[:, ss] = _tmod[self.conf.sym[ss]](sU * (n_l[ss] - t_cuts[:, ss]))

        t_l = self.tset[:, np.append(nout_l, 0), :]
        t_l[:, -1, :] = t_cuts
        t_r = self.tset[:, np.append(0, nout_r), :]
        t_r[:, 0, :] = t_cuts

        t_sort = _argsort(t_cuts, self.nsym)
        t_cuts = t_cuts[t_sort]
        t_full = self.tset[t_sort]
        t_l = t_l[t_sort]
        t_r = t_r[t_sort]

        xx_list = ((tuple(t2.flat), tuple(t3.flat), tuple(t4.flat)) for t2, t3, t4 in zip(t_l, t_r, t_full))
        to_execute = []
        for t1, tgroup in itertools.groupby(xx_list, key=lambda _, it=iter(t_cuts): tuple(next(it).flat)):
            to_execute.append((t1, sorted(tgroup)))

        # merged blocks and information for un-merging
        Amerged, order_l, order_r = self.conf.back.merge_blocks(self.A, to_execute, out_l, out_r, self.conf.dtype)
        Umerged, Smerged, Vmerged = self.conf.back.svd(Amerged, truncated=truncated_svd, Dblock=D_block, nbit=truncated_nbit, kfac=truncated_kfac)

        U = Tensor(settings=self.conf, s=np.append(self.s[nout_l], sU), n=n_l)
        S = Tensor(settings=self.conf, isdiag=True)
        V = Tensor(settings=self.conf, s=np.append(-sU, self.s[nout_r]), n=n_r)

        Dcut = self.conf.back.slice_S(Smerged, tol=tol, Dblock=D_block, Dtotal=D_total)

        U.A = self.conf.back.unmerge_blocks_left(Umerged, order_l, Dcut)
        S.A = self.conf.back.unmerge_blocks_diag(Smerged, Dcut)
        V.A = self.conf.back.unmerge_blocks_right(Vmerged, order_r, Dcut)

        U.tset = np.array([ind for ind in U.A], dtype=np.int).reshape((len(U.A), U._ndim, U.nsym))
        S.tset = np.array([ind for ind in S.A], dtype=np.int).reshape((len(S.A), S._ndim, S.nsym))
        V.tset = np.array([ind for ind in V.A], dtype=np.int).reshape((len(V.A), V._ndim, V.nsym))

        if Uaxis != -1:
            order = list(range(U.ndim - 1))
            if Uaxis >= 0:
                order.insert(Uaxis, U.ndim - 1)
            else:
                order.insert(Uaxis + 1, U.ndim - 1)
            U._transpose_local(axes=order)

        if Vaxis != 0:
            order = list(range(1, V.ndim))
            if Vaxis >= 0:
                order.insert(Vaxis, 0)
            elif Vaxis == -1:
                order.append(0)
            else:
                order.insert(Vaxis + 1, 0)
            V._transpose_local(axes=order)

        return U, S, V

    def split_qr(self, axes, sQ=1, Qaxis=-1, Raxis=0):
        r"""
        Split tensor using qr decomposition, tensor = Q * R.

        Charge of R is zero.

        Parameters
        ----------
        axes: tuple
            Specify two groups of legs between which to perform svd, as well as their final order.

        sQ: int
            signature of connecting leg in Q equall 1 or -1. Default is 1.

        Qaxis, Raxis: int
            specify which leg of Q and R tensors are connecting to the other tensor. By delault it is the last leg of Q and the first of R.

        Returns
        -------
            Q, R: Tensor
        """
        try:
            ll = len(axes[0])
            out_l = tuple(axes[0])
        except TypeError:
            out_l = (axes[0],)  # indices going u
            ll = 1
        try:
            lr = len(axes[1])
            out_r = tuple(axes[1])
        except TypeError:
            out_r = (axes[1],)  # indices going v
            lr = 1
        out_all = out_l + out_r  # order for transpose

        if not (self._ndim == ll + lr):
            raise TensorError('Two few indices in axes')
        elif not (sorted(set(out_all)) == list(range(self._ndim))):
            raise TensorError('Repeated axis')

        # divide charges between Q=l and R=r
        n_l, n_r = self.n, np.zeros(self.nsym, dtype=int)

        # order formation of blocks
        nout_r, nout_l = np.array(out_r, dtype=np.int), np.array(out_l, dtype=np.int)

        t_cuts = self.tset[:, nout_l, :].swapaxes(1, 2) @ self.s[nout_l]
        for ss in range(self.nsym):
            t_cuts[:, ss] = _tmod[self.conf.sym[ss]](sQ * (n_l[ss] - t_cuts[:, ss]))

        t_l = self.tset[:, np.append(nout_l, 0), :]
        t_l[:, -1, :] = t_cuts
        t_r = self.tset[:, np.append(0, nout_r), :]
        t_r[:, 0, :] = t_cuts

        t_sort = _argsort(t_cuts, self.nsym)
        t_cuts = t_cuts[t_sort]
        t_full = self.tset[t_sort]
        t_l = t_l[t_sort]
        t_r = t_r[t_sort]

        xx_list = ((tuple(t2.flat), tuple(t3.flat), tuple(t4.flat)) for t2, t3, t4 in zip(t_l, t_r, t_full))
        to_execute = []
        for t1, tgroup in itertools.groupby(xx_list, key=lambda _, it=iter(t_cuts): tuple(next(it).flat)):
            to_execute.append((t1, sorted(tgroup)))

        # merged blocks and information for un-merging
        Amerged, order_l, order_r = self.conf.back.merge_blocks(self.A, to_execute, out_l, out_r, self.conf.dtype)
        Qmerged, Rmerged = self.conf.back.qr(Amerged)
        Dcut = self.conf.back.slice_none(Amerged)

        Q = Tensor(settings=self.conf, s=np.append(self.s[nout_l], sQ), n=n_l)
        R = Tensor(settings=self.conf, s=np.append(-sQ, self.s[nout_r]), n=n_r)

        Q.A = self.conf.back.unmerge_blocks_left(Qmerged, order_l, Dcut)
        R.A = self.conf.back.unmerge_blocks_right(Rmerged, order_r, Dcut)

        Q.tset = np.array([ind for ind in Q.A], dtype=np.int).reshape(len(Q.A), Q._ndim, Q.nsym)
        R.tset = np.array([ind for ind in R.A], dtype=np.int).reshape(len(R.A), R._ndim, R.nsym)

        if Qaxis != -1:
            order = list(range(Q.ndim - 1))
            if Qaxis >= 0:
                order.insert(Qaxis, Q.ndim - 1)
            else:
                order.insert(Qaxis + 1, Q.ndim - 1)
            Q._transpose_local(axes=order)

        if Raxis != 0:
            order = list(range(1, R.ndim))
            if Raxis > 0:
                order.insert(Raxis, 0)
            elif Raxis == -1:
                order.append(0)
            else:
                order.insert(Raxis + 1, 0)
            R._transpose_local(axes=order)

        return Q, R

    def split_eigh(self, axes=(0, 1), sU=1, Uaxis=-1, tol=0, D_block=_large_int, D_total=_large_int):
        r"""
        Split tensor using eig, tensor = U * S * U^dag. Truncate smallest eigenvalues if neccesary.

        Tensor should be hermitian and has charge 0.
        Truncate using (whichever gives smaller bond dimension) relative tolerance, bond dimension of each block, and total bond dimension from all blocks.
        By default do not truncate. Truncate based on tolerance only if some eigenvalue is positive -- than all negative ones are discarded.
        Function primarly intended to be used for positively defined tensors.

        Parameters
        ----------
        axes: tuple
            Specify two groups of legs between which to perform svd, as well as their final order.

        sU: int
            signature of connecting leg in U equall 1 or -1. Default is 1.

        Uaxis: int
            specify which leg of U is the new connecting leg. By delault it is the last leg.

        tol: float
            relative tolerance of singular values below which to truncate.

        D_block: int
            largest number of singular values to keep in a single block.

        D_total: int
            largest total number of singular values to keep.

        Returns
        -------
            S, U: Tensor
                U is unitary projector. S is diagonal.
        """
        try:
            ll = len(axes[0])
            out_l = tuple(axes[0])
        except TypeError:
            out_l = (axes[0],)  # indices going u
            ll = 1
        try:
            lr = len(axes[1])
            out_r = tuple(axes[1])
        except TypeError:
            out_r = (axes[1],)  # indices going v
            lr = 1

        out_all = out_l + out_r  # order for transpose
        if not (self._ndim == ll + lr):
            raise TensorError('Two few indices in axes')
        elif not (sorted(set(out_all)) == list(range(self._ndim))):
            raise TensorError('Repeated axis')
        elif np.any(self.n != 0):
            raise TensorError('Charge should be zero')

        nout_r = np.array(out_r, dtype=np.int)
        nout_l = np.array(out_l, dtype=np.int)

        t_cuts = self.tset[:, nout_l, :].swapaxes(1, 2) @ self.s[nout_l]
        for ss in range(self.nsym):
            t_cuts[:, ss] = _tmod[self.conf.sym[ss]](-sU * t_cuts[:, ss])

        t_l = self.tset[:, np.append(nout_l, 0), :]
        t_l[:, -1, :] = t_cuts
        t_r = self.tset[:, np.append(0, nout_r), :]
        t_r[:, 0, :] = t_cuts

        t_sort = _argsort(t_cuts, self.nsym)
        t_cuts = t_cuts[t_sort]
        t_full = self.tset[t_sort]
        t_l = t_l[t_sort]
        t_r = t_r[t_sort]

        xx_list = ((tuple(t2.flat), tuple(t3.flat), tuple(t4.flat)) for t2, t3, t4 in zip(t_l, t_r, t_full))
        to_execute = []
        for t1, tgroup in itertools.groupby(xx_list, key=lambda _, it=iter(t_cuts): tuple(next(it).flat)):
            to_execute.append((t1, sorted(tgroup)))

        # merged blocks and information for un-merging
        Amerged, order_l, _ = self.conf.back.merge_blocks(self.A, to_execute, out_l, out_r, self.conf.dtype)
        Smerged, Umerged = self.conf.back.eigh(Amerged)

        # order formation of blocks
        S = Tensor(settings=self.conf, isdiag=True)
        U = Tensor(settings=self.conf, s=np.append(self.s[nout_l], sU))

        Dcut = self.conf.back.slice_S(Smerged, tol=tol, Dblock=D_block, Dtotal=D_total, decrease=False)

        S.A = self.conf.back.unmerge_blocks_diag(Smerged, Dcut)
        U.A = self.conf.back.unmerge_blocks_left(Umerged, order_l, Dcut)
        S.tset = np.array([ind for ind in S.A], dtype=np.int).reshape(len(S.A), S._ndim, S.nsym)
        U.tset = np.array([ind for ind in U.A], dtype=np.int).reshape(len(U.A), U._ndim, U.nsym)

        if Uaxis != -1:
            order = list(range(U.ndim - 1))
            if Uaxis >= 0:
                order.insert(Uaxis, U.ndim - 1)
            else:
                order.insert(Uaxis + 1, U.ndim - 1)
            U._transpose_local(axes=order)

        return S, U

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
                test.append(self.conf.back.is_independent(self.A[key], other.A[key]))
        return not any(test)

    def is_symmetric(self):
        """
        Test if tset corresponds to indices of A; and is f(s * t - n) == 0 is satisfied for all blocks.
        """
        test = []
        for ind in self.tset:
            test.append(tuple(ind.flat) in self.A)
        test.append(len(self.tset) == len(self.A))

        tcut = self.tset.swapaxes(1, 2) @ self.s
        for ss in range(self.nsym):
            tcut[:, ss] = _tmod[self.conf.sym[ss]](tcut[:, ss] - self.n[ss])
        test.append(not np.any(tcut))
        ts = self.get_t()
        for ss in range(self.nsym):
            for nn in range(self._ndim):
                cleared_t = [_tmod[self.conf.sym[ss]](x) for x in ts[ss][nn]]
                test.append(len(set(cleared_t)) == len(ts[ss][nn]))
        return all(test)
