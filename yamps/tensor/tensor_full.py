import numpy as np


# defaults
_large_int = 1073741824
_opts_split_svd = {'tol': 0, 'D_block': _large_int, 'D_total': _large_int, 'truncated_svd': False, 'truncated_nbit': 60, 'truncated_kfac': 6}
_opts_split_eigh = {'tol': 0, 'D_block': _large_int, 'D_total': _large_int}
# auxliary
_to_execute_merge = [(0, ((0, 0, 0, 0),))]
_to_execute_dot = [(0, 0, 0)]


def rand(settings=None, D=[], isdiag=False, dtype='float64'):
    """ Initialize tensor with random numbers from [-1,1].
        D = a list of bond dimensions
        isdiag makes tensor diagonal: D is a single dimension
        dtype = floa64/complex128 """
    a = Tensor(settings=settings, isdiag=isdiag, dtype=dtype)
    a.set_block(Ds=D, val='rand')
    return a


def zeros(settings=None, D=[], dtype='float64'):
    """ Initialize tensor with zeros
        D = a list of bond dimensions
        dtype = floa64/complex128 """
    a = Tensor(settings=settings, isdiag=False, dtype=dtype)
    a.set_block(Ds=D, val='zeros')
    return a


def ones(settings=None, D=[], dtype='float64'):
    """ Initialize tensor with ones
        D = a list of bond dimensions
        dtype = floa64/complex128 """
    a = Tensor(settings=settings, isdiag=False, dtype=dtype)
    a.set_block(Ds=D, val='ones')
    return a


def eye(settings=None, D=[], dtype='float64'):
    """ Initialize diagonal identity tensor
        D is a single dimension
        dtype = floa64/complex128 """
    a = Tensor(settings=settings, isdiag=True, dtype=dtype)
    a.set_block(Ds=D, val='ones')
    return a


def from_dict(settings=None, d={'A': {}, 'dtype': 'float64', 'isdiag': False}):
    """ Load tensor from dictionary """
    a = Tensor(settings=settings, isdiag=d['isdiag'], dtype=d['dtype'])
    a.set_block(val=d['A'])
    return a


def match_legs(tensors, legs, conjs=None, isdiag=False):
    """ Find initialisation for tensor matching existing legs """
    D = []
    for ii, te in zip(legs, tensors):
        D_list = te.get_shape()
        D.append(D_list[ii])
    if isdiag:
        return {'D': D[0]}
    else:
        return {'D': tuple(D)}


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


class Tensor:
    """
    =========================================
        Full tensor (wraper to np.array)
    =========================================
    """

    # Initialization #

    def __init__(self, settings=None, isdiag=False, dtype='float64'):
        """Initialize empty tensor"""
        self.tensor_type = 'full'
        self.backend = settings.backend
        self.settings = settings
        self.isdiag = isdiag  # diagonal
        self.dtype = dtype  # float64 or complex128
        self.A = {}

    def copy(self):
        """ Make a copy of a tensor """
        a = Tensor(settings=self.settings, isdiag=self.isdiag, dtype=self.dtype)
        a.A[0] = self.backend.copy(self.A[0])
        return a

    def set_block(self, Ds=None, val='zeros'):
        """ Set values of the tensor"""
        if val is 'zeros':
            if Ds is None:
                Ds = self.backend.get_shape(self.A[0])
            elif self.isdiag:
                Ds = (Ds, Ds)
            self.A[0] = self.backend.zeros(Ds, self.isdiag, self.dtype)
        elif val is 'rand':
            if Ds is None:
                Ds = self.backend.get_shape(self.A[0])
            elif self.isdiag:
                Ds = (Ds, Ds)
            self.A[0] = self.backend.rand(Ds, self.isdiag, self.dtype)
        elif val is 'ones':
            if Ds is None:
                Ds = self.backend.get_shape(self.A[0])
            elif self.isdiag:
                Ds = (Ds, Ds)
            self.A[0] = self.backend.ones(Ds, self.isdiag, self.dtype)
        else:
            self.A[0] = self.backend.to_tensor(val, isdiag=self.isdiag, dtype=self.dtype, Ds=Ds)

    #     New tensor of the same class

    def ones(self, D=[], dtype='float64'):
        return ones(settings=self.settings, D=D, dtype=Dtype)

    def rand(self, D=[], isdiag=False, dtype='float64'):
        return rand(settings=self.settings, D=D, isdiag=isdiag, dtype=dtype)

    def zeros(self, D=[], dtype='float64'):
        return zeros(settings=self.settings, D=D, dtype=dtype)

    def ones(self, D=[], dtype='float64'):
        return ones(settings=self.settings, D=D, dtype=dtype)

    def eye(self, D=[], dtype='float64'):
        return eye(settings=self.settings, D=D, dtype=dtype)

    def match_legs(self, tensors, legs, conjs=None, isdiag=False):
        return match_legs(tensors, legs, conjs, isdiag)

    #     Output

    def to_dict(self):
        """ Export relevant information about tensor to dictionary -- to be saved."""
        if self.isdiag:
            out = {'A': self.backend.to_numpy(self.backend.diag_get(self.A[0])), 'isdiag': self.isdiag, 'dtype': self.dtype}
        else:
            out = {'A': self.backend.to_numpy(self.A[0]), 'isdiag': self.isdiag, 'dtype': self.dtype}
        return out

    def __str__(self):
        return self.tensor_type+' A='+self.backend.get_str(self.A)

    def show_properties(self):
        """ Display basic properties of tensor"""
        print("type : ", self.tensor_type)
        if len(self.A) == 0:
            print("tensor not initialized")
        else:
            print("ndim : ", self.get_ndim())
            print("isdiag: ", self.isdiag)
            print("dtype: ", self.dtype)
            print("shape: ", self.get_shape())

    def get_shape(self):
        return self.backend.get_shape(self.A[0])

    def get_ndim(self):
        return self.backend.get_ndim(self.A[0])

    def get_dtype(self):
        return self.backend.get_dtype(self.A[0])

    def to_numpy(self):
        """ Create np.array corresponding to tensor"""
        return self.backend.to_numpy(self.A[0])

    def to_number(self):
        """ First number"""
        return self.backend.first_el(self.A[0])

    ### linear operations ###
    def __mul__(self, other):
        """Multiply tensor by number"""
        a = Tensor(self.settings, self.isdiag, self.dtype)
        a.A[0] = other*self.A[0]
        return a

    def __rmul__(self, other):
        """Multiply tensor by number"""
        a = Tensor(self.settings, self.isdiag, self.dtype)
        a.A[0] = other*self.A[0]
        return a

    def __add__(self, other):
        """Add two tensors"""
        a = Tensor(self.settings, self.isdiag, self.dtype)
        a.A[0] = self.A[0] + other.A[0]
        return a

    def apxb(self, other, x=1):
        """ Subtract two tensors
            c = self - x * other
            [default x=1] """
        a = Tensor(self.settings, self.isdiag, self.dtype)
        a.A[0] = self.A[0] + x * other.A[0]
        return a

    def __sub__(self, other):
        """ Subtract two tensors
            c = self - x * other
            [default x=1] """
        a = Tensor(self.settings, self.isdiag, self.dtype)
        a.A[0] = self.A[0] - other.A[0]
        return a

    def norm(self, ord='fro', round2=False):
        """ Calculate norm of tensor
            'fro': 2 norm of a tensor reshaped to a vector
            'inf': max(abs())"""
        return self.backend.norm(self.A, ord=ord, round2=round2)

    def norm_diff(self, b, ord='fro'):
        """ Calculate norm of tensor
            'fro': 2 norm of a tensor reshaped to a vector
            'inf': max(abs())"""
        return self.backend.norm_diff(self.A, b.A, ord=ord)

    ### tensor functions ###
    def conj(self):
        """ Conjugate """
        a = Tensor(self.settings, self.isdiag)
        a.A = self.backend.conj(self.A)
        return a

    def transpose(self, axes=None):
        """ tranpose tensor """
        a = Tensor(self.settings, self.isdiag)
        a.A = a.backend.transpose(self.A, axes, to_execute=[(0, 0)])
        return a

    def invsqrt(self):
        """ element-wise 1/sqrt(A)"""
        a = Tensor(self.settings, self.isdiag)
        a.A = self.backend.invsqrt(self.A, self.isdiag)
        return a

    def inv(self):
        """ element-wise 1/sqrt(A)"""
        a = Tensor(self.settings, self.isdiag)
        a.A = self.backend.inv(self.A, self.isdiag)
        return a

    def exp(self, step=1.):
        """element-wise exp(step*A)"""
        a = Tensor(self.settings, self.isdiag)
        a.A = self.backend.exp(self.A, step, self.isdiag)
        return a

    def sqrt(self):
        """element-wise sqrt(A)"""
        a = Tensor(self.settings, self.isdiag)
        a.A = self.backend.sqrt(self.A, self.isdiag)
        return a

    def entropy(self, axes=(0, 1), alpha=1):
        """
        Calculate entropy from tensor.

        If diagonal, calculates entropy treating S^2 as probabilities 
        It normalizes S^2 if neccesary.
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

            # merged blocks; do not need information for unmerging
            Amerged, _, _ = self.backend.merge_blocks(self.A, to_execute=_to_execute_merge, out_l=out_l, out_r=out_r)
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
        ids = tuple(range(len(self.get_shape())))
        if len(self.get_shape()) != len(b.get_shape()):
            print('Number of legs does not match.')
            exit()
        else:
            if ids != tuple(range(len(b.get_shape()))):
                print('Legs` dimensions do not match.')
                exit()
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
            in2 = (axes[1],) # indices going v
        out = tuple(ii for ii in range(self.get_ndim()) if ii not in in1+in2)
        a = Tensor(self.settings, self.isdiag, dtype=self.dtype)
        a.A = self.backend.trace(A=self.A, to_execute=[(0, 0)], in1=in1, in2=in2, out=out)
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
        a_out = tuple(ii for ii in range(self.get_ndim()) if ii not in a_con)  # outgoing legs
        b_out = tuple(ii for ii in range(b.get_ndim()) if ii not in b_con)

        newdtype = 'complex128' if (self.dtype == 'complex128' or b.dtype == 'complex128') else 'float64'
        c = Tensor(self.settings, dtype=newdtype)
        c.A = self.backend.dot(self.A, b.A, conj, _to_execute_dot, a_out, a_con, b_con, b_out)
        return c

    ###########################
    #     spliting tensor     #
    ###########################

    def split_svd(self, axes=(0, 1), opts={}):
        """Split tensor using svd: a = u * s * v
            axes specifies legs and their final order.
            s is diagonal tensor
            opts = {'tol':0, 'D_block':_large_int, 'D_total':_large_int, 'truncated_svd':False, 'truncated_nbit':60, 'truncated_kfac':6}
            Truncate using (whichever gives smaller bond dimension):
            relative tolerance tol, bond dimension of each block D_block, total bond dimension D_total
            By default do not truncate
            Can use truncated_svd """
        try:
            out_l = tuple(axes[0])
        except TypeError:
            out_l = (axes[0],)  # indices going u
        try:
            out_r = tuple(axes[1])
        except TypeError:
            out_r = (axes[1],)  # indices going v

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

        Amerged, order_l, order_r = self.backend.merge_blocks(self.A, to_execute=_to_execute_merge, out_l=out_l, out_r=out_r) # merged blocks and information for un-merging
        Umerged, Smerged, Vmerged = self.backend.svd(Amerged, truncated=truncated_svd, Dblock=D_block, nbit=truncated_nbit, kfac=truncated_kfac)

        U = Tensor(self.settings)
        V = Tensor(self.settings)
        S = Tensor(self.settings, isdiag = True)
        Dcut = self.backend.slice_S(Smerged, tol=tol, Dblock=D_block, Dtotal=D_total)

        U.A = self.backend.unmerge_blocks_left(Umerged, order_l, Dcut)
        S.A = self.backend.unmerge_blocks_diag(Smerged, order_s=[(0,0)], Dcut=Dcut)
        V.A = self.backend.unmerge_blocks_right(Vmerged, order_r, Dcut)
        return U, S, V

    def split_qr(self, axes=(0, 1)):
        """ Split tensor using qr: a = q * r
            axes specifies legs and their final order """
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
        out_all = out_l + out_r

        if not (self.get_ndim() == ll + lr):
            raise Exception('Two few indices in axes')
        elif not (sorted(set(out_all)) == list(range(self.get_ndim()))):
            raise Exception('Repeated axis')

        Amerged, order_l, order_r = self.backend.merge_blocks(self.A, to_execute=_to_execute_merge, out_l=out_l, out_r=out_r) # merged blocks and information for un-merging
        Qmerged, Rmerged = self.backend.qr(Amerged)
        Dcut = self.backend.slice_none(Amerged)

        Q = Tensor(self.settings, dtype=self.dtype)
        R = Tensor(self.settings, dtype=self.dtype)

        Q.A = self.backend.unmerge_blocks_left(Qmerged, order_l, Dcut)
        R.A = self.backend.unmerge_blocks_right(Rmerged, order_r, Dcut)
        return Q, R

    def split_rq(self, axes=(0, 1)):
        """ Split tensor using qr: a = q * r
            axes specifies legs and their final order """
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
            out_r = (axes[1],) # indices going v
            lr = 1
        out_all = out_l + out_r

        if not (self.get_ndim() == ll + lr):
            raise Exception('Two few indices in axes')
        elif not (sorted(set(out_all)) == list(range(self.get_ndim()))):
            raise Exception('Repeated axis')

        Amerged, order_l, order_r = self.backend.merge_blocks(self.A, to_execute=_to_execute_merge, out_l=out_l, out_r=out_r) # merged blocks and information for un-merging
        Rmerged, Qmerged = self.backend.rq(Amerged)
        Dcut = self.backend.slice_none(Amerged)

        R = Tensor(self.settings, dtype=self.dtype)
        Q = Tensor(self.settings, dtype=self.dtype)

        R.A = self.backend.unmerge_blocks_left(Rmerged, order_l, Dcut)
        Q.A = self.backend.unmerge_blocks_right(Qmerged, order_r, Dcut)
        return R, Q

    def split_eigh(self, axes = (0, 1), opts={}):
        """ Split tensor using eigh: a = u * s * u^dag
            Axes specifies legs and their final order
            Tensor should be hermitian
            s is diagonal
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
            out_r = (axes[1],) # indices going v
            lr = 1
        out_all = out_l + out_r

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

        if not (self.get_ndim() == ll + lr):
            raise Exception('Two few indices in axes')
        elif not (sorted(set(out_all)) == list(range(self.get_ndim()))):
            raise Exception('Repeated axis')

        Amerged, order_l, _ = self.backend.merge_blocks(self.A, to_execute=_to_execute_merge, out_l=out_l, out_r=out_r) # merged blocks and information for un-merging
        Smerged, Umerged = self.backend.eigh(Amerged)

        S = Tensor(self.settings, isdiag=True, dtype='float64')
        U = Tensor(self.settings, dtype=self.dtype)

        Dcut = self.backend.slice_S(Smerged, tol=tol, Dblock=D_block, Dtotal=D_total, decrease=False)
        S.A = self.backend.unmerge_blocks_diag(Smerged, order_s=[(0, 0)], Dcut=Dcut)
        U.A = self.backend.unmerge_blocks_left(Umerged, order_l, Dcut)
        return S, U

    ### statistics ###
    def swap_gate(self, axes, fermionic=False):
        """ Apply swap gate """
        return self

    ### tests ###
    def is_independent(self, other):
        """ Test if two tensors are independent objects in memory."""
        t  = []
        t.append(self is other)
        t.append(self.A is other.A)
        if not any(t):
            return True
        else:
            return False
