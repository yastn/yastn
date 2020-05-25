r"""
Tensor with one abelian symmetry.

This class defines generic arbitrary rank tensor supporting abelian
symmetry. An instance of a Tensor is defined by list of blocks (dense tensors)
with their individual dimensions labeled by the symmetry charges.
"""

import numpy as np
import itertools

# defaults
_large_int = 1073741824
_opts_split_svd = {'tol': 0, 'D_block': _large_int, 'D_total': _large_int, 'truncated_svd': False, 'truncated_nbit': 60, 'truncated_kfac': 6}
_opts_split_eigh = {'tol': 0, 'D_block': _large_int, 'D_total': _large_int}
_ind_type = np.int  # type for charges handling
_tmod = {'U1': lambda x: x,
         'Z2': lambda x: np.mod(x, 2)}


class TensorShapeError(Exception):
    pass


# creating tensors
def rand(settings=None, s=[], n=0, t=[], D=[], isdiag=False, dtype='float64'):
    r"""
    Initialize tensor with all possible blocks filled with random numbers from [-1,1].

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal; s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
    dtype : str
         floa64 or complex128

    Returns
    -------
    tensor : tensor
        a random instance of a tens tensor
    """
    a = Tensor(settings=settings, s=s, n=n, isdiag=isdiag, dtype=dtype)
    a.reset_tensor(t=t, D=D, val='rand')
    return a


def zeros(settings=None, s=[], n=0, t=[], D=[], dtype='float64'):
    r"""
    Initialize tensor with all possible blocks filled with zeros.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal; s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
    dtype : str
         floa64 or complex128

    Returns
    -------
    tensor : tensor
        an instance of a tens tensor filled with zeros
    """
    a = Tensor(settings=settings, s=s, n=n, isdiag=False, dtype=dtype)
    a.reset_tensor(t=t, D=D, val='zeros')
    return a


def ones(settings=None, s=[], n=0, t=[], D=[], dtype='float64'):
    r"""
    Initialize tensor with all possible blocks filled with ones.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal; s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
    dtype : str
         floa64 or complex128

    Returns
    -------
    tensor : tensor
        an instance of a tens tensor filled with ones
    """
    a = Tensor(settings=settings, s=s, n=n, isdiag=False, dtype=dtype)
    a.reset_tensor(t=t, D=D, val='ones')
    return a


def eye(settings=None, t=[], D=[], dtype='float64'):
    """ Initialize diagonal identity tensor
        s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
        dtype = floa64/complex128 """
    a = Tensor(settings=settings, s=(1, -1), n=0, isdiag=True, dtype=dtype)
    a.reset_tensor(t=t, D=D, val='ones')
    return a


def from_dict(settings=None, d={'s': [], 'n': 0, 'isdiag': False, 'dtype': 'float64', 'A': {}}):
    """ Load tensor from dictionary """
    a = Tensor(settings=settings, s=d['s'], n=d['n'], isdiag=d['isdiag'], dtype=d['dtype'])
    # lookup table of possible blocks (combinations of t) numpy.array
    for ind in d['A']:
        a.set_block(ts=ind, val=d['A'][ind])
    return a


def match_legs(tensors, legs, conjs=None, isdiag=False):
    """ Find initialisation for tensor matching existing legs """
    t, D, s = [], [], []
    if conjs is None:
        conjs = len(tensors) * [0]
    for ii, te, cc in zip(legs, tensors, conjs):
        t_list, D_list = te.get_tD_list()
        t.append(t_list[ii])
        D.append(D_list[ii])
        if cc == 0:
            s.append(-te.s[ii])
        else:
            s.append(te.s[ii])
    if isdiag:
        return {'t': t[0], 'D': D[0]}
    else:
        return {'t': tuple(t), 'D': tuple(D), 's': tuple(s)}


def block(td, common_legs, ndim):
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

    out_m = tuple(ii for ii in range(ndim) if ii not in out_s)
    out_ma = np.array(out_m, dtype=int)
    li = ndim - ls
    pos = []
    newdtype = 'float64'
    for ind, ten in td.items():
        if li != len(ind):
            raise TensorShapeError('block: wrong tensors rank or placement')
        pos.append(ind)
        if ten.dtype == 'complex128':
            newdtype = 'complex128'
    pos.sort()

    # all charges and bond dimensions
    tlist, Dlist = {}, {}
    for ind in pos:
        tt, DD = td[ind].get_tD_list()
        tlist[ind] = tt
        Dlist[ind] = DD

    # combinations of charges on legs to merge
    t_out_m = [np.unique(td[ind].tset[:, out_ma], axis=0) for ind in pos]
    t_out_unique = np.unique(np.vstack(t_out_m), axis=0)

    # positions including those charges
    t_out_pos = []
    for tt in t_out_unique:
        t_out_pos.append([ind for ind, tm in zip(pos, t_out_m) if not np.any(np.sum(np.abs(tt - tm), axis=1))])

    # print(t_out_m)
    # print(t_out_unique)
    # print(t_out_pos)

    for tt, pos_tt in zip(t_out_unique, t_out_pos):
        for ind in pos_tt:
            for kk in td[ind].tset:
                if np.all(kk[out_ma] == tt):
                    pass
        #pos_tt

    posa = np.array(pos, dtype=int)
    legs_ind = []  # indices on specific legs
    legs_D = []  # and corresponding keys
    kk = -1
    for ii in range(ndim):
        if ii in out_m:
            kk += 1
            x, y = np.unique(posa[:, kk], return_index=True)
            legs_ind.append(list(x))
            legs_D.append([td[pos[ll]].get_shape()[ii] for ll in y])
        else:
            legs_D.append([td[pos[0]].get_shape()[ii]])

    Ad = {key: td[key].A for key in pos}
    to_execute = [(0, pos, legs_ind, legs_D)]

    c = Tensor(td[pos[0]].settings, dtype=newdtype)
    c.A = c.backend.block(Ad, to_execute)
    return c


# Main class defining operations on tensor
class Tensor:
    """
    Class defining an abelian tensor and all related operations.
    """

    # input #
    def __init__(self, settings=None, s=[], n=0, isdiag=False, dtype='float64'):
        r"""
        Initialize empty Tensor with abelian symmetry.

        Parameters
        ----------
        settings:
            settings
        s : tuple
            a signature of tensor
        n : int
            total charge
        isdiag : bool
            makes tensor diagonal; s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
        dtype : str
            floa64 or complex128
        """
        self.symmetry = settings.symmetry
        self.backend = settings.backend
        self.settings = settings
        self.isdiag = isdiag  # diagonal
        self.dtype = dtype  # float64 or complex128
        if not isdiag:
            self.s = np.array(s, dtype=_ind_type)  # signature
            self.n = n  # tensor charge (sum(s*t) = n)
        else:
            self.s = np.array([1, -1], dtype=_ind_type)
            self.n = 0
        self.ndim = len(self.s)  # number of legs
        self.tset = np.empty((0, self.ndim), dtype=_ind_type)  # list of blocks
        self.A = {}  # dictionary of blocks

    def copy(self):
        """ Make a copy of a tensor """
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.tset = self.tset.copy()
        for ind in self.A:
            a.A[ind] = self.backend.copy(self.A[ind])
        return a

    def reset_tensor(self, t=[], D=[], val='rand'):
        """ Create all possible blocks based on t, s, n
            brute-force check all possibilities select the ones satisfying s*t=n
            initialise tensors in each possible block based on given bond dimensions"""
        if self.isdiag:
            t, D = (t, t), (D, D)

        all_combinations_t = np.array(list(itertools.product(*t)), dtype=_ind_type)
        ind = (_tmod[self.symmetry](all_combinations_t @ self.s) == _tmod[self.symmetry](self.n))
        all_combinations_t = all_combinations_t[ind]
        all_combinations_D = np.array(list(itertools.product(*D)), dtype=_ind_type)
        all_combinations_D = all_combinations_D[ind]
        self.tset = all_combinations_t

        for ind, Ds in zip(all_combinations_t, all_combinations_D):
            ind, Ds = tuple(ind), tuple(Ds)
            if val == 'zeros':
                self.A[ind] = self.backend.zeros(Ds, self.isdiag, self.dtype)
            elif val == 'rand':
                self.A[ind] = self.backend.rand(Ds, self.isdiag, self.dtype)
            elif val == 'ones':
                self.A[ind] = self.backend.ones(Ds, self.isdiag, self.dtype)

    def set_block(self, ts=[], Ds=None, val='zeros'):
        """ Add new block to tensor.
            ts = charges of the block.
            Ds = its bond dimension. If Ds not given, tries to read it from existing block with ts.
            val = 'zeros','rand','ones', else assume it is an array of dimension (Ds is compulsory!).
        """
        # if not((val == 'zeros') or not(val == 'rand') or (val == 'ones')) and (Ds is None):
        #     raise TensorShapeError('Provide dimensions for given tensor')

        if len(ts) != self.ndim:
            raise TensorShapeError('Number of charges does not match ndim')
        ts = np.array(ts, dtype=_ind_type)
        if not (_tmod[self.symmetry](ts @ self.s) == _tmod[self.symmetry](self.n)):
            raise TensorShapeError('Charges do not fit the tensor: t @ s != n')
        ts = tuple(ts)
        if ts not in self.A:
            self.tset = np.vstack([self.tset, np.array([ts], dtype=_ind_type)])

        t_list, D_list = self.get_tD_list()
        existing_D = []
        no_existing_D = False
        for ii in range(self.ndim):
            try:
                existing_D.append(D_list[ii][t_list[ii].index(ts[ii])])
            except ValueError:
                existing_D.append(-1)
                no_existing_D = True

        if isinstance(val, str):
            if Ds is None:
                if no_existing_D:
                    raise TensorShapeError('Not all dimensions specify')
                Ds = existing_D
            else:
                for D1, D2 in zip(Ds, existing_D):
                    if (D1 != D2) and (D2 != -1):
                        raise TensorShapeError('Dimension of a new block does not match the existing ones')
            Ds = tuple(Ds)
            if val == 'zeros':
                self.A[ts] = self.backend.zeros(Ds, self.isdiag, self.dtype)
            elif val == 'rand':
                self.A[ts] = self.backend.rand(Ds, self.isdiag, self.dtype)
            elif val == 'ones':
                self.A[ts] = self.backend.ones(Ds, self.isdiag, self.dtype)
        else:
            if Ds is not None:
                val = np.reshape(np.array(val), Ds)
            self.A[ts] = self.backend.to_tensor(val, isdiag=self.isdiag, dtype=self.dtype)
            Ds = self.backend.get_shape(self.A[ts])
            for D1, D2 in zip(Ds, existing_D):
                if (D1 != D2) and (D2 != -1):
                    raise TensorShapeError('Dimension of a new block does not match the existing ones')

    ###########################
    #       new tensors       #
    ###########################

    def rand(self, s=[], n=0, t=[], D=[], isdiag=False, dtype='float64'):
        r"""
        Initialize a new tensor with all possible blocks filled with random numbers from [-1,1].
        Use the same settings as self.

        Parameters
        ----------
        s : tuple
            a signature of tensor
        n : int
            total charge
        t : list
            a list of charges for each leg
        D : list
            a list of corresponding bond dimensions
        isdiag : bool
            makes tensor diagonal; s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
        dtype : str
            floa64 or complex128

        Returns
        -------
        tensor : tensor
            a random instance of a tens tensor
        """
        a = Tensor(settings=self.settings, s=s, n=n, isdiag=isdiag, dtype=dtype)
        a.reset_tensor(t=t, D=D, val='rand')
        return a

    def zeros(self, s=[], n=0, t=[], D=[], dtype='float64'):
        r"""
        Initialize a new tensor with all possible blocks filled with zeros.
        Use the same settings as self.

        Parameters
        ----------
        s : tuple
            a signature of tensor
        n : int
            total charge
        t : list
            a list of charges for each leg
        D : list
            a list of corresponding bond dimensions
        isdiag : bool
            makes tensor diagonal; s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
        dtype : str
            floa64 or complex128

        Returns
        -------
        tensor : tensor
            an instance of a tens tensor filled with zeros
        """
        a = Tensor(settings=self.settings, s=s, n=n, isdiag=False, dtype=dtype)
        a.reset_tensor(t=t, D=D, val='zeros')
        return a

    def ones(self, s=[], n=0, t=[], D=[], dtype='float64'):
        r"""
        Initialize a new tensor with all possible blocks filled with ones.
        Use the same settings as self.

        Parameters
        ----------
        s : tuple
            a signature of tensor
        n : int
            total charge
        t : list
            a list of charges for each leg
        D : list
            a list of corresponding bond dimensions
        isdiag : bool
            makes tensor diagonal; s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
        dtype : str
            floa64 or complex128

        Returns
        -------
        tensor : tensor
            an instance of a tens tensor filled with ones
        """
        a = Tensor(settings=self.settings, s=s, n=n, isdiag=False, dtype=dtype)
        a.reset_tensor(t=t, D=D, val='ones')
        return a

    def eye(self, t=[], D=[], dtype='float64'):
        """ Initialize a new diagonal identity tensor using the same settings as self
            s=(1, -1); n=0; t/D is a list of charges/dimensions of one leg
            dtype = floa64/complex128 """
        a = Tensor(settings=self.settings, s=(1, -1), n=0, isdiag=True, dtype=dtype)
        a.reset_tensor(t=t, D=D, val='ones')
        return a

    def from_dict(self, d={'s': [], 'n': 0, 'isdiag': False, 'dtype': 'float64', 'A': {}}):
        """ Load tensor from dictionary """
        a = Tensor(settings=self.settings, s=d['s'], n=d['n'], isdiag=d['isdiag'], dtype=d['dtype'])
        # lookup table of possible blocks (combinations of t) numpy.array
        for ind in d['A']:
            a.set_block(ts=ind, val=d['A'][ind])
        return a

    def match_legs(self, tensors, legs, conjs=None, isdiag=False):
        """ Find initialisation for tensor matching existing legs """
        return match_legs(tensors, legs, conjs, isdiag)

    ###########################
    #     output functions    #
    ###########################

    def to_dict(self):
        """ Export relevant information about tensor to dictionary -- to be saved """
        if self.isdiag:
            AA = {ind: self.backend.to_numpy(self.backend.diag_get(self.A[ind])) for ind in self.A}
        else:
            AA = {ind: self.backend.to_numpy(self.A[ind]) for ind in self.A}
        out = {'A': AA, 's': self.s, 'n': self.n, 'isdiag': self.isdiag, 'dtype': self.dtype}
        return out

    def __str__(self):
        return self.symmetry + ' s=' + str(self.s) + ' n=' + str(self.n)

    def show_properties(self):
        r""" Display basic properties of the tensor """
        print("symmetry     : ", self.symmetry)
        print("ndim         : ", self.ndim)
        print("sign         : ", self.s)
        print("charge       : ", self.n)
        print("isdiag       : ", self.isdiag)
        print("dtype        : ", self.dtype)
        print("no. of blocks: ", len(self.A))
        list_t, list_D = self.get_tD_list()
        print("charges      : ", list_t)
        print("dimensions   : ", list_D)
        print("total dim    : ", [sum(xx) for xx in list_D])

    def get_total_charge(self):
        return self.n

    def get_signature(self):
        return self.s

    def get_charges(self):
        return self.tset

    def get_shape(self):
        shape = []
        for A in self.A.values():
            shape.append(self.backend.get_shape(A))
        return shape

    def get_ndim(self):
        return self.ndim

    def get_dtype(self):
        return self.dtype

    def get_t_list(self):
        """ All charges on all legs"""
        list_t = []
        for ii in range(self.ndim):
            tt = [t[ii] for t in self.tset]
            list_t.append(sorted(set(tt)))
        return list_t

    def get_tD_list(self):
        """ All charges and corresponding dimensions on all legs"""
        tset, Dset = [], []
        for ind in self.A:
            tset.append(ind)
            Dset.append(self.backend.get_shape(self.A[ind]))
        list_t, list_D = [], []
        for ii in range(self.ndim):
            tt = [t[ii] for t in tset]
            DD = [D[ii] for D in Dset]
            TD = sorted(set(zip(tt, DD)))
            list_t.append(tuple(ss[0] for ss in TD))
            list_D.append(tuple(ss[1] for ss in TD))
        return list_t, list_D

    def to_numpy(self):
        """ Create full np.array corresponding to the tensor"""
        list_t, list_D = self.get_tD_list()
        Dtotal = [sum(Ds) for Ds in list_D]
        a = np.zeros(Dtotal, self.dtype)
        for ind in self.A:  # fill in the blocks
            sl = []
            for leg, t in enumerate(ind):
                ii = list_t[leg].index(t)
                Dleg = sum(list_D[leg][:ii])
                sl.append(slice(Dleg, Dleg + list_D[leg][ii]))
            if sl:
                a[tuple(sl)] = self.A[ind]
            else:  # should only happen for 0-dim tensor -- i.e.  a scalar
                a = self.A[ind]
        return a

    def to_number(self):
        """ First number in the first block (unsorted)
            0 if no blocks = empty tensor """
        if len(self.A) > 0:
            key = next(iter(self.A))
            return self.backend.first_el(self.A[key])
        else:
            return 0.

    #############################
    #     linear operations     #
    #############################

    def __mul__(self, other):
        """ Multiply tensor by number"""
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.tset = self.tset.copy()
        for ind in self.A:
            a.A[ind] = other * self.A[ind]
        return a

    def __rmul__(self, other):
        """ Multiply tensor by number"""
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.tset = self.tset.copy()
        for ind in self.A:
            a.A[ind] = other * self.A[ind]
        return a

    def __add__(self, other):
        """Add two tensors"""
        if not all(self.s == other.s) or (self.n != other.n):
            raise TensorShapeError('Tensors do not match')
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
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        if len(new_tset) > 0:
            tset = np.vstack([tset, np.array(new_tset, dtype=_ind_type)])
        a.tset = tset
        a.A = a.backend.add(self.A, other.A, to_execute)
        return a

    def apxb(self, other, x=1):
        """ Add two tensors
            c = self + x * other
            [default x=1] """
        if not all(self.s == other.s) or (self.n != other.n):
            raise TensorShapeError('Tensors do not match')
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
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        if len(new_tset) > 0:
            tset = np.vstack([tset, np.array(new_tset, dtype=_ind_type)])
        a.tset = tset
        a.A = a.backend.add(self.A, other.A, to_execute, x)
        return a

    def __sub__(self, other):
        """ Subtract two tensors
            c = self - x * other
            [default x=1] """
        if not all(self.s == other.s) or (self.n != other.n):
            raise TensorShapeError('Tensors do not match')
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
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        if len(new_tset) > 0:
            tset = np.vstack([tset, np.array(new_tset, dtype=_ind_type)])
        a.tset = tset
        a.A = a.backend.sub(self.A, other.A, to_execute)
        return a

    def norm(self, ord='fro', round2=False):
        """ Norm of tensor
            'fro': Frobenious
            'inf': max(abs())"""
        return self.backend.norm(self.A, ord=ord, round2=round2)

    def norm_diff(self, b, ord='fro'):
        """ Norm of tensor a-b
            'fro': Frobenious
            'inf': max(abs())"""
        return self.backend.norm_diff(self.A, b.A, ord)

    ############################
    #     tensor functions     #
    ############################

    def conj(self):
        """ Conjugate """
        a = Tensor(settings=self.settings, s=-self.s, n=-self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.tset = self.tset.copy()
        a.A = a.backend.conj(self.A)
        return a

    def transpose(self, axes, local=True):
        """ tranpose tensor """
        order = np.array(axes, dtype=_ind_type)
        a = Tensor(settings=self.settings, s=self.s[order], n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.tset = self.tset[:, order]
        to_execute = []
        for old, new in zip(self.tset, a.tset):
            to_execute.append((tuple(old), tuple(new)))
        a.A = a.backend.transpose(self.A, axes, to_execute)
        return a

    def swap_gate(self, axes, fermionic=False):
        """ Apply swap gate between axes[1] and axes[2].
            if axes[1] is -1, swap gate axes[2] with charge n."""
        if fermionic:
            axes = sorted(list(axes))
            a = self.copy()
            if (axes[0] == -1) and (a.n % 2 == 1):  # swap gate with local a.n
                for ind in self.A:
                    if ind[axes[1]] % 2 == 1:
                        a.A[ind] = -a.A[ind]
            elif axes[0] != axes[1]:  # swap gate on 2 legs
                for ind in a.A:
                    if (ind[axes[0]] % 2 == 1) and (ind[axes[1]] % 2 == 1):
                        a.A[ind] = -a.A[ind]
            else:
                raise TensorShapeError('Cannot sweep the same index')
            return a
        else:
            return self

    def invsqrt(self):
        """ element-wise 1/sqrt(A)"""
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.A = self.backend.invsqrt(self.A, self.isdiag)
        a.tset = self.tset.copy()
        return a

    def inv(self):
        """ element-wise 1/sqrt(A)"""
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.A = self.backend.inv(self.A, self.isdiag)
        a.tset = self.tset.copy()
        return a

    def exp(self, step=1.):
        """element-wise exp(step*A)"""
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.A = self.backend.exp(self.A, step, self.isdiag)
        a.tset = self.tset.copy()
        return a

    def sqrt(self):
        """element-wise sqrt"""
        a = Tensor(settings=self.settings, s=self.s, n=self.n, isdiag=self.isdiag, dtype=self.dtype)
        a.A = self.backend.sqrt(self.A, self.isdiag)
        a.tset = self.tset.copy()
        return a

    def entropy(self, axes=(0, 1), alpha=1):
        """
        Calculate entropy from tensor.

        If diagonal, calculates entropy treating S^2 as probabilities,
        where self = U*S*V. It normalizes S^2 if neccesary.
        If not diagonal, calculates svd first to get the diagonal S.
        Base of log is 2.

        Parameters
        ----------
        axes: tuple
            how to split the tensor for svd

        alpha: float
            Order of Renyi entropy.
            alpha=1 is von Neuman: Entropy -Tr(S^2 log2(S^2))
            otherwise: 1/(1-alpha) log2(Tr(S^(2 alpha)))

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

            if not (self.ndim == ll + lr):
                raise TensorShapeError('Two few indices in axes')

            # divide charges between l and r
            n_l, n_r = _tmod[self.symmetry]((self.n + 1) // 2), _tmod[self.symmetry](self.n // 2)

            # order formation of blocks
            nout_r, nout_l = np.array(out_r, dtype=_ind_type), np.array(out_l, dtype=_ind_type)
            t_cuts = _tmod[self.symmetry](n_r - self.tset[:, nout_r] @ self.s[nout_r])
            t_l = self.tset[:, np.append(nout_l, 0)]
            t_l[:, -1] = t_cuts
            t_r = self.tset[:, np.append(0, nout_r)]
            t_r[:, 0] = t_cuts

            xx_list = sorted((t1, tuple(t2), tuple(t3), tuple(t4)) for t1, t2, t3, t4 in zip(t_cuts, t_l, t_r, self.tset))
            to_execute = []
            for t1, tgroup in itertools.groupby(xx_list, key=lambda x: x[0]):
                to_execute.append((t1, list(tgroup)))

            # merged blocks; do not need information for unmerging
            Amerged, _, _ = self.backend.merge_blocks(self.A, to_execute, out_l, out_r)
            Smerged = self.backend.svd_no_uv(Amerged)
        else:
            Smerged = self.A

        ent, Smin, no = self.backend.entropy(Smerged, alpha=alpha)
        return ent, Smin, no

    ##################################
    #     contraction operations     #
    ##################################

    def scalar(self, b):
        """ scalar product: (a,b) """
        self_dim = len(self.get_shape()[0])
        b_dim = len(b.get_shape()[0])
        ids = tuple(range(self_dim))
        if ids != tuple(range(b_dim)):
            raise TensorShapeError('tensor.abelian.scalar:: Legs` dimensions do not match')
        out = self.dot(b, axes=(ids, ids), conj=(1, 0))
        return out.to_number()

    def trace(self, axes=(0, 1)):
        """ Compute trace of legs specified by axes"""
        try:
            in1 = tuple(axes[0])
        except TypeError:
            in1 = (axes[0],)  # indices going u
        try:
            in2 = tuple(axes[1])
        except TypeError:
            in2 = (axes[1],)  # indices going v
        out = tuple(ii for ii in range(self.ndim) if ii not in in1 + in2)

        nout = np.array(out, dtype=_ind_type)
        nin1 = np.array(in1, dtype=_ind_type)
        nin2 = np.array(in2, dtype=_ind_type)

        if not all(self.s[nin1] == -self.s[nin2]):
            raise TensorShapeError('Signs do not match')

        to_execute = []
        for tt in self.tset:
            if all(tt[nin1] == tt[nin2]):
                to_execute.append((tuple(tt), tuple(tt[nout])))  # old, new

        a = Tensor(settings=self.settings, s=self.s[nout], n=self.n, dtype=self.dtype)
        a.A = a.backend.trace(A=self.A, to_execute=to_execute, in1=in1, in2=in2, out=out)
        a.tset = np.array([ind for ind in a.A], dtype=_ind_type).reshape(len(a.A), a.ndim)
        return a

    def dot(self, b, axes, conj=(0, 0)):
        """ Compute tensor dot product a*b along specified axes
            order outgoing legs like in np.tensordot
            conj = tuple showing which tensor is conjugate;
            default is conj=(0, 0) [no conjugation]"""
        try:
            a_con = tuple(axes[0])  # contracted legs
        except TypeError:
            a_con = (axes[0],)
        try:
            b_con = tuple(axes[1])
        except TypeError:
            b_con = (axes[1],)
        conja = (1 - 2 * conj[0])
        conjb = (1 - 2 * conj[1])

        a_out = tuple(ii for ii in range(self.ndim) if ii not in a_con)  # outgoing legs
        b_out = tuple(ii for ii in range(b.ndim) if ii not in b_con)

        na_con = np.array(a_con, dtype=_ind_type)
        nb_con = np.array(b_con, dtype=_ind_type)
        na_out = np.array(a_out, dtype=_ind_type)
        nb_out = np.array(b_out, dtype=_ind_type)

        if not all(self.s[na_con] == -b.s[nb_con] * conja * conjb):
            if b.isdiag:  # added for diagonal tensor to efectively reverse the order of signs
                if all(self.s[na_con] == b.s[nb_con] * conja * conjb):
                    conjb *= -1
                else:
                    raise TensorShapeError('Signs do not match')
            elif self.isdiag:
                if all(self.s[na_con] == b.s[nb_con] * conja * conjb):
                    conja *= -1
                else:
                    raise TensorShapeError('Signs do not match')
            else:
                raise TensorShapeError('Signs do not match')

        if self.settings.dot_merge:
            t_a_con = self.tset[:, na_con]
            t_b_con = b.tset[:, nb_con]
            t_a_out = self.tset[:, na_out]
            t_b_out = b.tset[:, nb_out]
            t_a_cuts = _tmod[self.symmetry](t_a_con @ self.s[na_con])
            t_b_cuts = _tmod[self.symmetry](-t_b_con @ b.s[nb_con] * conja * conjb)

            a_sort = t_a_cuts.argsort()
            b_sort = t_b_cuts.argsort()

            t_a_full = self.tset[a_sort]
            t_a_con = t_a_con[a_sort]
            t_a_out = t_a_out[a_sort]
            t_a_cuts = t_a_cuts[a_sort]

            t_b_full = b.tset[b_sort]
            t_b_con = t_b_con[b_sort]
            t_b_out = t_b_out[b_sort]
            t_b_cuts = t_b_cuts[b_sort]

            a_list = [(t1, tuple(t2), tuple(t3), tuple(t4)) for t1, t2, t3, t4 in zip(t_a_cuts, t_a_out, t_a_con, t_a_full)]
            b_list = [(t1, tuple(t2), tuple(t3), tuple(t4)) for t1, t2, t3, t4 in zip(t_b_cuts, t_b_con, t_b_out, t_b_full)]
            a_iter = itertools.groupby(a_list, key=lambda x: x[0])
            b_iter = itertools.groupby(b_list, key=lambda x: x[0])

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
                        ga_iter = itertools.groupby(sorted(ga, key=lambda x: x[2]), key=lambda x: x[2])
                        gb_iter = itertools.groupby(sorted(gb, key=lambda x: x[1]), key=lambda x: x[1])
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
            Amerged, order_l, _ = self.backend.merge_blocks(self.A, to_merge_a, a_out, a_con)
            # merged blocks and information for un-merging
            Bmerged, _, order_r = self.backend.merge_blocks(b.A, to_merge_b, b_con, b_out)

            Cmerged = self.backend.dot_merged(Amerged, Bmerged, conj)

            newdtype = 'complex128' if (self.dtype == 'complex128' or b.dtype == 'complex128') else 'float64'
            c_s = np.hstack([conja * self.s[na_out], conjb * b.s[nb_out]])
            c = Tensor(settings=self.settings, s=c_s, n=conja * self.n + conjb * b.n, dtype=newdtype)
            c.A = self.backend.unmerge_blocks(Cmerged, order_l, order_r)
            c.tset = np.array([ind for ind in c.A], dtype=_ind_type).reshape(len(c.A), c.ndim)
            return c
        else:
            t_a_con = self.tset[:, na_con]
            t_b_con = b.tset[:, nb_con]
            t_a_out = self.tset[:, na_out]
            t_b_out = b.tset[:, nb_out]

            block_a = sorted([(tuple(x), tuple(y), tuple(z)) for x, y, z in zip(t_a_con, t_a_out, self.tset)], key=lambda x: x[0])
            block_b = sorted([(tuple(x), tuple(y), tuple(z)) for x, y, z in zip(t_b_con, t_b_out, b.tset)], key=lambda x: x[0])

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
            newdtype = 'complex128' if (self.dtype == 'complex128' or b.dtype == 'complex128') else 'float64'
            c_s = np.hstack([conja * self.s[na_out], conjb * b.s[nb_out]])
            c = Tensor(settings=self.settings, s=c_s, n=conja * self.n + conjb * b.n, dtype=newdtype)
            c.A = self.backend.dot(self.A, b.A, conj, to_execute, a_out, a_con, b_con, b_out)
            c.tset = np.array([ind for ind in c.A], dtype=_ind_type).reshape(len(c.A), c.ndim)
            return c

    ###########################
    #     spliting tensor     #
    ###########################

    def split_svd(self, axes=(0, 1), opts={}):
        """ Split tensor using svd:  a = u * s * v
            axes specifies legs and their final order
            s is diagonal tensor with signature (1, -1)
            Charge divided between u [n_u = n+1//2] and v [n_v = n//2]
            opts = {'tol':0, 'D_block':_large_int, 'D_total':_large_int, 'truncated_svd':False, 'truncated_nbit':60, 'truncated_kfac':6}
            Truncate using (whichever gives smaller bond dimension):
            relative tolerance tol, bond dimension of each block D_block, total bond dimension D_total
            By default do not truncate.
            Can use truncated_svd """
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
        try:
            tol = opts['tol']
        except KeyError:
            tol = _opts_split_svd['tol']
        try:
            D_block = opts['D_block']
        except KeyError:
            D_block = _opts_split_svd['D_block']
        try:
            D_total = opts['D_total']
        except KeyError:
            D_total = _opts_split_svd['D_total']
        try:
            truncated_svd = opts['truncated_svd']
        except KeyError:
            truncated_svd = _opts_split_svd['truncated_svd']
        try:
            truncated_nbit = opts['truncated_nbit']
        except KeyError:
            truncated_nbit = _opts_split_svd['truncated_nbit']
        try:
            truncated_kfac = opts['truncated_kfac']
        except KeyError:
            truncated_kfac = _opts_split_svd['truncated_kfac']

        if not (self.ndim == ll + lr):
            raise TensorShapeError('Two few indices in axes')
        elif not (sorted(set(out_l + out_r)) == list(range(self.ndim))):
            raise TensorShapeError('Repeated axis')

        # divide charges between l and r
        n_l, n_r = _tmod[self.symmetry]((self.n + 1) // 2), _tmod[self.symmetry](self.n // 2)

        # order formation of blocks
        nout_r, nout_l = np.array(out_r, dtype=_ind_type), np.array(out_l, dtype=_ind_type)
        t_cuts = _tmod[self.symmetry](n_r - self.tset[:, nout_r] @ self.s[nout_r])
        t_l = self.tset[:, np.append(nout_l, 0)]
        t_l[:, -1] = t_cuts
        t_r = self.tset[:, np.append(0, nout_r)]
        t_r[:, 0] = t_cuts

        xx_list = sorted((t1, tuple(t2), tuple(t3), tuple(t4)) for t1, t2, t3, t4 in zip(t_cuts, t_l, t_r, self.tset))
        to_execute = []
        for t1, tgroup in itertools.groupby(xx_list, key=lambda x: x[0]):
            to_execute.append((t1, list(tgroup)))

        # merged blocks and information for un-merging
        Amerged, order_l, order_r = self.backend.merge_blocks(self.A, to_execute, out_l, out_r)
        Umerged, Smerged, Vmerged = self.backend.svd(Amerged, truncated=truncated_svd, Dblock=D_block, nbit=truncated_nbit, kfac=truncated_kfac)

        U = Tensor(settings=self.settings, s=np.append(self.s[nout_l], -1), n=n_l, dtype=self.dtype)
        S = Tensor(settings=self.settings, isdiag=True, dtype='float64')
        V = Tensor(settings=self.settings, s=np.append(1, self.s[nout_r]), n=n_r, dtype=self.dtype)

        Dcut = self.backend.slice_S(Smerged, tol=tol, Dblock=D_block, Dtotal=D_total)
        order_s = [(tcut, (tcut, tcut)) for tcut in Dcut]

        U.A = self.backend.unmerge_blocks_left(Umerged, order_l, Dcut)
        S.A = self.backend.unmerge_blocks_diag(Smerged, order_s, Dcut)
        V.A = self.backend.unmerge_blocks_right(Vmerged, order_r, Dcut)

        U.tset = np.array([ind for ind in U.A], dtype=_ind_type)
        S.tset = np.array([ind for ind in S.A], dtype=_ind_type)
        V.tset = np.array([ind for ind in V.A], dtype=_ind_type)
        return U, S, V

    def split_qr(self, axes):
        """ Split tensor using qr: a = q * r
            axes specifies legs and their final order.
            connecting signs -1 (in q), 1 (in r)
            Charge of r is zero
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

        if not (self.ndim == ll + lr):
            raise TensorShapeError('Two few indices in axes')
        elif not (sorted(set(out_all)) == list(range(self.ndim))):
            raise TensorShapeError('Repeated axis')

        # divide charges between Q=l and R=r
        n_l, n_r = self.n, 0

        # order formation of blocks
        nout_r, nout_l = np.array(out_r, dtype=_ind_type), np.array(out_l, dtype=_ind_type)

        t_cuts = _tmod[self.symmetry](n_r - self.tset[:, nout_r] @ self.s[nout_r])
        t_l = self.tset[:, np.append(nout_l, 0)]
        t_l[:, -1] = t_cuts
        t_r = self.tset[:, np.append(0, nout_r)]
        t_r[:, 0] = t_cuts

        xx_list = sorted((t1, tuple(t2), tuple(t3), tuple(t4)) for t1, t2, t3, t4 in zip(t_cuts, t_l, t_r, self.tset))
        to_execute = []

        for t1, tgroup in itertools.groupby(xx_list, key=lambda x: x[0]):
            to_execute.append((t1, list(tgroup)))

        # merged blocks and information for un-merging
        Amerged, order_l, order_r = self.backend.merge_blocks(self.A, to_execute, out_l, out_r)
        Qmerged, Rmerged = self.backend.qr(Amerged)
        Dcut = self.backend.slice_none(Amerged)

        Q = Tensor(settings=self.settings, s=np.append(self.s[nout_l], -1), n=n_l, dtype=self.dtype)
        R = Tensor(settings=self.settings, s=np.append(1, self.s[nout_r]), n=n_r, dtype=self.dtype)

        Q.A = self.backend.unmerge_blocks_left(Qmerged, order_l, Dcut)
        R.A = self.backend.unmerge_blocks_right(Rmerged, order_r, Dcut)

        Q.tset = np.array([ind for ind in Q.A], dtype=_ind_type)
        R.tset = np.array([ind for ind in R.A], dtype=_ind_type)
        return Q, R

    def split_rq(self, axes):
        """ Split tensor using qr: a = r * q
            axes specifies legs and their final order.
            connecting signs -1 (in r), 1 (in q)
            Charge of r is zero
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

        if not (self.ndim == ll + lr):
            raise TensorShapeError('Two few indices in axes')
        elif not (sorted(set(out_all)) == list(range(self.ndim))):
            raise TensorShapeError('Repeated axis')

        # divide charges between R=l and Q=r
        n_l, n_r = 0, self.n

        # order formation of blocks
        nout_r, nout_l = np.array(out_r, dtype=_ind_type), np.array(out_l, dtype=_ind_type)

        t_cuts = _tmod[self.symmetry](n_r - self.tset[:, nout_r] @ self.s[nout_r])
        t_l = self.tset[:, np.append(nout_l, 0)]
        t_l[:, -1] = t_cuts
        t_r = self.tset[:, np.append(0, nout_r)]
        t_r[:, 0] = t_cuts

        xx_list = sorted((t1, tuple(t2), tuple(t3), tuple(t4)) for t1, t2, t3, t4 in zip(t_cuts, t_l, t_r, self.tset))
        to_execute = []

        for t1, tgroup in itertools.groupby(xx_list, key=lambda x: x[0]):
            to_execute.append((t1, list(tgroup)))

        # merged blocks and information for un-merging
        Amerged, order_l, order_r = self.backend.merge_blocks(self.A, to_execute, out_l, out_r)

        Rmerged, Qmerged = self.backend.rq(Amerged)
        Dcut = self.backend.slice_none(Amerged)

        R = Tensor(settings=self.settings, s=np.append(self.s[nout_l], -1), n=n_l, dtype=self.dtype)
        Q = Tensor(settings=self.settings, s=np.append(1, self.s[nout_r]), n=n_r, dtype=self.dtype)

        R.A = self.backend.unmerge_blocks_left(Rmerged, order_l, Dcut)
        Q.A = self.backend.unmerge_blocks_right(Qmerged, order_r, Dcut)

        R.tset = np.array([ind for ind in R.A], dtype=_ind_type)
        Q.tset = np.array([ind for ind in Q.A], dtype=_ind_type)
        return R, Q

    def split_eigh(self, axes=(0, 1), opts={}):
        """ Split tensor using eigh: a = u * s * u^dag
            axes specifies legs and their final order.
            tensor should be hermitian and has charge n=0
            s is diagonal with signature (1, -1)
            opts = {'tol':0, 'D_block':_large_int, 'D_total':_large_int}
            Truncate using (whichever gives smaller bond dimension):
            relative tolerance tol, bond dimension of each block D_block, total bond dimension D_total
            By default do not truncate.
            Truncate on tolerance only if some eigenvalue is positive [primarly used for tensors which whould be positively defined] """
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
        try:
            tol = opts['tol']
        except KeyError:
            tol = _opts_split_eigh['tol']
        try:
            D_block = opts['D_block']
        except KeyError:
            D_block = _opts_split_eigh['D_block']
        try:
            D_total = opts['D_total']
        except KeyError:
            D_total = _opts_split_eigh['D_total']

        out_all = out_l + out_r  # order for transpose
        if not (self.ndim == ll + lr):
            raise TensorShapeError('Two few indices in axes')
        elif not (sorted(set(out_all)) == list(range(self.ndim))):
            raise TensorShapeError('Repeated axis')
        elif self.n != 0:
            raise TensorShapeError('Charge should be zero')

        nout_r = np.array(out_r, dtype=_ind_type)
        nout_l = np.array(out_l, dtype=_ind_type)

        t_cuts = _tmod[self.symmetry](-self.tset[:, nout_r] @ self.s[nout_r])
        t_l = self.tset[:, np.append(nout_l, 0)]
        t_l[:, -1] = t_cuts
        t_r = self.tset[:, np.append(0, nout_r)]
        t_r[:, 0] = t_cuts

        xx_list = sorted((t1, tuple(t2), tuple(t3), tuple(t4)) for t1, t2, t3, t4 in zip(t_cuts, t_l, t_r, self.tset))
        to_execute = []
        for t1, tgroup in itertools.groupby(xx_list, key=lambda x: x[0]):
            to_execute.append((t1, list(tgroup)))

        # merged blocks and information for un-merging
        Amerged, order_l, _ = self.backend.merge_blocks(self.A, to_execute, out_l, out_r)
        Smerged, Umerged = self.backend.eigh(Amerged)

        # order formation of blocks
        S = Tensor(settings=self.settings, isdiag=True, dtype='float64')
        U = Tensor(settings=self.settings, s=np.append(self.s[nout_l], -1), n=0, dtype=self.dtype)

        Dcut = self.backend.slice_S(Smerged, tol=tol, Dblock=D_block, Dtotal=D_total, decrease=False)
        order_s = [(tcut, (tcut, tcut)) for tcut in Dcut]

        S.A = self.backend.unmerge_blocks_diag(Smerged, order_s, Dcut)
        U.A = self.backend.unmerge_blocks_left(Umerged, order_l, Dcut)
        U.tset = np.array([ind for ind in U.A], dtype=_ind_type)
        S.tset = np.array([ind for ind in S.A], dtype=_ind_type)
        return S, U

    #################
    #     tests     #
    #################

    def is_independent(self, other):
        """ Test if two tensors are independent objects in memory."""
        t = []
        t.append(self is other)
        t.append(self.A is other.A)
        for key in self.A.keys():
            if key in other.A:
                t.append(self.A[key] is other.A[key])
        for key in other.A.keys():
            if key in self.A:
                t.append(self.A[key] is other.A[key])
        if not any(t):
            return True
        else:
            return False
