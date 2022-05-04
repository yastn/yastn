""" Mps structure and its basic manipulations. """
from numpy import array, nonzero
from yast.tensor import block, entropy
from yast.tensor import save_to_hdf5 as Tensor_to_hdf5
from yast.tensor import save_to_dict as Tensor_to_dict
from yast import load_from_dict as Tensor_from_dict
from yast import load_from_hdf5 as Tensor_from_hdf5


class YampsError(Exception):
    pass


def apxb(a, b, common_legs, x=1):
    r"""
    Adds two Mps-s with multiplicative prefactor x, and creates a new object as an output.
    In short: c = a + x * b


    Parameters
    ----------
        a, b : Mps
            matrix products to be added

        common_legs : tuple
            which legs of individual tensors in Mps objects are common and are expanded by an addition. The addition is done on remaining legs

        x : float/complex
            multiplicative prefactor for tensor b

    Returns
    -------
        c : Mps
            new Mps, sum of a and b which is independent of them
    """
    if a.N is not b.N:
        YampsError('Mps-s must have equal number of Tensor-s.')

    c = a.copy()
    for n in range(c.N):
        if n == 0:
            if x != 1:
                d = {(0,): x*a.A[n], (1,): x*b.A[n]}
            else:
                d = {(0,): a.A[n], (1,): b.A[n]}
            common_lgs = (0,)+common_legs
        elif n == a.N-1:
            d = {(0,): a.A[n], (1,): b.A[n]}
            common_lgs = common_legs+(common_legs[-1]+1,)
        else:
            d = {(0, 0): a.A[n], (1, 1): b.A[n]}
            common_lgs = common_legs
        c.A[n] = block(d, common_lgs)
    return c


def add(tens, amp, common_legs):
    r"""
    Adds any number of Mps-s stored in tens with multiplicative prefactors specified in amp. It creates a new Mps as an output.
    In short: c = \sum_j amp[j] * tens[j]
    Number of Mps-s should be the same as the number of prefactors.


    Parameters
    ----------
        tens : list of Mps-s
            Each element of the list should contain a single Mps.

        tens : list of float/complex-s
            Each element of the list should contain a single number.

        common_legs : tuple
            which legs of individual tensors in Mps objects are common and are expanded by an addition. The addition is done on remaining legs

    Returns
    -------
        c : Mps
            new Mps, sum of all Mps-s in tens. It is independent of them
    """
    if len(tens) is not len(amp):
        raise YampsError('Number of Mps-s must be equal to number of coefficients in amp.')

    elif sum([tens[j].N-tens[0].N for j in range(len(tens))]) != 0:
        raise YampsError('Mps-s must have equal lengths.')

    c = tens[0].copy()
    N = c.N
    for n in range(N):
        d = {}
        if n == 0:
            for j in range(len(tens)):
                d[(j,)] = amp[j]*tens[j].A[n] if amp[j] != 1. else tens[j].A[n]
            common_lgs = (0,) + common_legs
        elif n == N-1:
            for j in range(len(tens)):
                d[(j,)] = tens[j].A[n]
            common_lgs = common_legs+(common_legs[-1]+1,)
        else:
            for j in range(len(tens)):
                d[(j, j)] = tens[j].A[n]
            common_lgs = common_legs
        c.A[n] = block(d, common_lgs)
    return c


def x_a_times_b(a, b, axes, axes_fin, conj=(0, 0), x=1, mode='hard'):
    r"""
    Multiplies two Mps-s with additional prefector which is an number. In short: c = x * a * b


    Parameters
    ----------
        a, b : Mps
            matrix products to be added

        axes: tuple
            an argument for tensordot, legs of both tensors (for each it is specified by int or tuple of ints)
            e.g. axes=(0, 3) to contract 0th leg of `a` with 3rd leg of `b`
            axes=((0, 3), (1, 2)) to contract legs 0 and 3 of `a` with 1 and 2 of `b`, respectively.

        axes_fin: tuple
            an argument for fuse_legs, happening after performing tensordot along Mps-s,        
            legs of both tensors (for each it is specified by int or tuple of ints)
            e.g. (0, (1,2,), 3) means fusing leg 1 and 2 into one leg
            The fusion in performed inplace=True

        conj: tuple
            an argument for tensordot, shows which Mps to conjugate: (0, 0), (0, 1), (1, 0), or (1, 1).
            Default is (0, 0), i.e. neither is conjugated

        x : float/complex
            multiplicative prefactor

        mode : str
            an argument for fuse_legs, mode for the fusion of legs

    Returns
    -------
        c : Mps
            new Mps, product of Mps-s a and b, is independent of them
    """
    if a.N is not b.N:
        YampsError('Mps-s must have equal number of Tensor-s.')
    c = a.copy()
    for n in range(c.N):
        if n == 0:
            c.A[n] = x*a.A[n].tensordot(b.A[n], axes, conj).fuse_legs(axes_fin, True, mode)
        else:
            c.A[n] = a.A[n].tensordot(b.A[n], axes, conj).fuse_legs(axes_fin, True, mode)
    return c


def load_from_dict(config, nr_phys, in_dict):
    r"""
    Reads Tensor-s of Mps from a dictionary into an Mps object

    Returns
    -------
    out_Mps : Mps
    """
    N = len(in_dict)
    out_Mps = Mps(N, nr_phys=nr_phys)
    for n in range(out_Mps.N):
        out_Mps.A[n] = Tensor_from_dict(config=config, d=in_dict[n])
    return out_Mps


def load_from_hdf5(config, nr_phys, file, in_file_path):
    r"""
    Reads Tensor-s of Mps from a HDF5 file into an Mps object

    Parameters
    -----------
    config: config
        Configuration of Tensors' symmetries

    nr_phys: int
        number of physical legs

    file: File
        A 'pointer' to a file opened by a user

    in_file_path: File
        Name of a group in the file, where the Mps saved

    Returns
    -------
    out_Mps : Mps
    """
    N = len(file[in_file_path].keys())
    out_Mps = Mps(N, nr_phys=nr_phys)
    for n in range(out_Mps.N):
        out_Mps.A[n] = Tensor_from_hdf5(config, file, in_file_path+str(n))
    return out_Mps


def generate_Mij(amp, connect, N, nr_phys):
    x_from = connect['from']
    x_to = connect['to']
    x_conn = connect['conn']
    x_else = connect['else']
    jL, T_from = x_from
    jR, T_to = x_to
    T_conn = x_conn
    T_else = x_else

    M = Mps(N, nr_phys=nr_phys)
    for n in range(M.N):
        if jL == jR:
            M.A[n] = amp*T_from.copy() if n == jL else T_else.copy()
        else:
            if n == jL:
                M.A[n] = amp*T_from.copy()
            elif n == jR:
                M.A[n] = T_to.copy()
            elif n > jL and n < jR:
                M.A[n] = T_conn.copy()
            else:
                M.A[n] = T_else.copy()
        if n == 0:
            tt = (0,) * len(M.A[n].n)
        M.A[n] = M.A[n].add_leg(axis=0, t=tt, s=1)
        M.A[n] = M.A[n].add_leg(axis=-1, s=-1)
        tD = M.A[n].get_leg_structure(axis=-1)
        tt = next(iter(tD))
    return M


def automatic_Mps(amplitude, from_it, to_it, permute_amp, Tensor_from, Tensor_to, Tensor_conn, Tensor_other, N, nr_phys, common_legs, opts={'tol': 1e-14}):
    r"""
    Generate Mps representuing sum of two-point operators M=\sum_i,j Mij Op_i Op_j with possibility to include jordan-Wigner chain for these.

    Parameters
    ----------
    amplitude : iterable list of numbers
        Mij, amplitudes for an operator
    from_it : int iterable list
        first index of Mij
    to_it : int iterable list
        second index of Mij
    permute_amp : iterable list of numbers
        accounds for commuation/anticommutation rule while Op_j, Opj have to be permuted.
    Tensor_from: list of Tensor-s
        list of Op_i for Mij-th element
    Tensor_to: list of Tensor-s
        list of Op_j for Mij-th element
    Tensor_conn: list of Tensor-s
        list of operators to put in cetween Op_i and Opj for Mij-th element
    Tensor_other: list of Tensor-s
        list of operators outside i-j for Mij-th element
    N : int
        number of sites of Mps
    nr_phys : int
        number of physical legs: _1_ for mps; _2_ for mpo;
    common_legs : tuple of int
        common legs for Tensors
    opts : dict
        Options passed for svd -- including information how to truncate.
    """
    new_common_legs = tuple([n+1 for n in common_legs])
    given = nonzero(array(amplitude))[0]
    bunch_tens, bunch_amp = [None]*len(given), [None]*len(given)
    for ik in range(len(given)):
        it = given[ik]
        if from_it[it] > to_it[it]:
            conn, other = Tensor_conn[it], Tensor_other[it]
            if nr_phys > 1:
                left, right = Tensor_to[it].tensordot(conn, axes=common_legs[::-1]), Tensor_from[it]
            else:
                left, right = Tensor_to[it], Tensor_from[it]
            il, ir = to_it[it], from_it[it]
            amp = amplitude[it]*permute_amp[it]
        else:
            conn, other = Tensor_conn[it], Tensor_other[it]
            left, right = Tensor_from[it], Tensor_to[it]
            il, ir = from_it[it], to_it[it]
            amp = amplitude[it]

            if il == ir and right:
                left, right = left.tensordot(right, axes=common_legs[::-1]), None

        connect = {'from': (il, left),
                'to': (ir, right),
                'conn': conn,
                'else': other}

        bunch_tens[ik] = generate_Mij(1., connect, N, nr_phys)
        bunch_amp[ik] = amp

    M = add(bunch_tens, bunch_amp, new_common_legs)
    M.canonize_sweep(to='last', normalize=False)
    M.truncate_sweep(to='first', opts=opts, normalize=False)
    return M


def apxb(a, b, common_legs, x=1):
    """
    if inplace=false a+a*b will be a new Mps otherwise I will replace mb and delete b
    """
    if a.N is not b.N:
        YampsError('Mps-s must have equal number of Tensor-s.')

    c = a.copy()
    for n in range(c.N):
        if n == 0:
            if x != 1:
                d = {(0,): x*a.A[n], (1,): x*b.A[n]}
            else:
                d = {(0,): a.A[n], (1,): b.A[n]}
            common_lgs = (0,)+common_legs
        elif n == a.N-1:
            d = {(0,): a.A[n], (1,): b.A[n]}
            common_lgs = common_legs+(common_legs[-1]+1,)
        else:
            d = {(0, 0): a.A[n], (1, 1): b.A[n]}
            common_lgs = common_legs
        c.A[n] = block(d, common_lgs)
    return c


def x_a_times_b(a, b, axes, axes_fin, conj=(0, 0), x=1, mode='hard'):
    # make multiplication x*a*b, with conj if necessary
    if a.N is not b.N:
        YampsError('Mps-s must have equal number of Tensor-s.')
    c = a.copy()
    for n in range(c.N):
        if n == 0:
            c.A[n] = x*a.A[n].tensordot(b.A[n], axes, conj).fuse_legs(axes_fin, mode)
        else:
            c.A[n] = a.A[n].tensordot(b.A[n], axes, conj).fuse_legs(axes_fin, mode)
    return c


###################################
#     basic operations on MPS     #
###################################


class Mps:
    """
    The basic structure of mps (for nr_phys=1) and mpo (for nr_phys=2) and some basic operations on a single mps.
    Order of legs for a single mps tensor is (left virtual, 1st physical, 2nd physical, right virtual).
    Mps tensors are index with :math:`0, 1, 2, 3, \\ldots, N-1` (with :math:`0` corresponding to the first site).
    A central block (associated with a bond) is indexed using ordered tuple (n, n+1).
    Maximally one central block is allowed.
    """

    def __init__(self, N, nr_phys=1):
        r"""
        Initialize basic structure for matrix product state/operator/purification.

        Parameters
        ----------
        N : int
            number of sites of mps, or directly an instance of geometry class
        nr_phys : int
            number of physical legs: _1_ for mps; _2_ for mpo;
        """
        if not isinstance(N, int) or N <= 0:
            raise YampsError("Number of Mps sites N should be a positive integer.")
        if nr_phys not in (1, 2):
            raise YampsError("Number of physical legs of Mps, nr_phys, should be equal to 1 or 2.")
        self.N = N
        self.nr_phys = nr_phys
        self.A = {}  # dict of mps tensors; indexed by integers
        self.pC = None  # index of the central site, None if it does not exist
        self.left = (0,)  # convention which leg is left virtual(connected to site with smaller index)
        self.right = (nr_phys + 1,)  # convention which leg is a right virtual leg (connected to site with larger index)
        self.phys = (1,) if nr_phys == 1 else (1, 2)  # convention which legs are physical
        self.first = 0
        self.last = self.N - 1

    def sweep(self, to='last', df=0, dl=0):
        r"""
        Generator of indices of all sites going from the first site to the last site, or vice-versa.

        Parameters
        ----------
        to : str
            'first' or 'last'.
        df, dl : int
            shift iterator by df >= 0 and dl >= 0 from the first and the last site, respectively.
        """
        if to == 'last':
            return range(df, self.N - dl)
        if to == 'first':
            return range(self.N - 1 - dl, df - 1, -1)
        raise YampsError('Argument "to" should be in ("first", "last")')

    def clone(self):
        r"""
        Makes a copy of mps. Copy all mps tensors, tracking gradients.

        Use when retaining "old" mps is neccesary -- other operations on mps are often done in-place.

        Returns
        -------
        Cloned mps : Mps
        """
        phi = Mps(N=self.N, nr_phys=self.nr_phys)
        phi.A = {ind: ten.clone() for ind, ten in self.A.items()}
        phi.pC = self.pC
        return phi

    def copy(self):
        r"""
        Makes a copy of mps. Copy all mps tensors.

        Warning, this might break autograd if you are using it.
        Use when retaining "old" mps is neccesary -- other operations on mps are often done in-place.

        Returns
        -------
        Copied mps : Mps
        """
        phi = Mps(N=self.N, nr_phys=self.nr_phys)
        phi.A = {ind: ten.copy() for ind, ten in self.A.items()}
        phi.pC = self.pC
        return phi

    def orthogonalize_site(self, n, to='last', normalize=True):
        r"""
        Orthogonalize n-th site to the first site.

        Parameters
        ----------
            n : int
                index of site to be ortogonalized

            to : str
                'last' or 'first'.

            normalize : bool
                If true, central sites is normalized to 1 according to standard 2-norm.
        """
        if self.pC is not None:
            raise YampsError('Only one central block is possible. Attach the existing central block first.')

        if to == 'first':
            self.pC = (n - 1, n)
            self.A[n], R = self.A[n].qr(axes=(self.phys + self.right, self.left), sQ=1, Qaxis=0, Raxis=-1)
            self.A[self.pC] = R / R.norm() if normalize else R
        elif to == 'last':
            self.pC = (n, n + 1)
            self.A[n], R = self.A[n].qr(axes=(self.left + self.phys, self.right), sQ=-1)
            self.A[self.pC] = R / R.norm() if normalize else R
        else:
            raise YampsError('Argument "to" should be in ("first", "last")')

    def diagonalize_central(self, opts=None, normalize=True):
        r"""
        Perform svd of the central site C = U S V -- truncating according to opts which is pased into svd.

        Attach U and V respective to the left and right sites.

        Parameters
        ----------
        opts : dict
            Options passed for svd -- including information how to truncate.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.

        Return
        ------
        discarded : float
            norm of discarded singular values normalized by the remining ones
        """
        if self.pC is not None:
            normC = self.A[self.pC].norm()
            if opts is None:
                opts = {'tol': 1e-12}
            U, S, V = self.A[self.pC].svd_with_truncation(axes=(0, 1), sU=-1, **opts)

            normS = S.norm()
            self.A[self.pC] = S / normS if normalize else S
            n1, n2 = self.pC

            if n1 >= 0:
                self.A[n1] = self.A[n1].tensordot(U, axes=(self.right, 0))
            else:
                self.A[self.pC] = U.tensordot(self.A[self.pC], axes=(1, 0))

            if n2 <= self.N - 1:
                self.A[n2] = V.tensordot(self.A[n2], axes=(1, self.left))
            else:
                self.A[self.pC] = self.A[self.pC].tensordot(V, axes=(1, 0))

            return (normC -  normS) / normS
        return 0.

    def remove_central(self):
        r"""
        Removes (ignores) the central site. Do nothing if is does not exist.
        """
        if self.pC is not None:
            del self.A[self.pC]
            self.pC = None

    def absorb_central(self, to='last'):
        r"""
        Absorb central site towards the first or the last site.

        If the central site is outside of the chain, it is goes in the one direction possible.
        Do nothing if central does not exist.

        Parameters
        ----------
        to : str
            'last' or 'first'.
        """
        if self.pC is not None:
            C = self.A.pop(self.pC)
            n1, n2 = self.pC
            self.pC = None

            if (to == 'first' and n1 >= 0) or n2 >= self.N:
                self.A[n1] = self.A[n1].tensordot(C, axes=(self.right, 0))
            else:  # (to == 'last' and n2 < self.N) or n1 < 0
                self.A[n2] = C.tensordot(self.A[n2], axes=(1, self.left))

    def canonize_sweep(self, to='last', normalize=True):
        r"""
        Sweep though the mps and cannonize it toward the first of last site.

        At the end, attach the trivial central block to the end of the chain, so that there is no central block left.

        Parameters
        ----------
        to : str
            'last' or 'first'.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.
        """
        self.absorb_central(to=to)
        for n in self.sweep(to=to):
            self.orthogonalize_site(n=n, to=to, normalize=normalize)
            self.absorb_central(to=to)
        return self

    def truncate_sweep(self, to='last', normalize=True, opts=None):
        r"""
        Sweep though the mps, cannonize it toward the first or last site truncating via svd.

        Truncation makes sense if mps is in the cannonical form in the oposite direction to that of the current sweep.

        Parameters
        ----------
        to : str
            'last' or 'first'.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.

        opts : dict
            Options passed for svd; includes information how to truncate.

        Return
        ------
        discarded : float
            maximal norm of discarded singular values normalized by the remining ones
        """
        discarded_max = 0.
        if opts is None:
            opts = {'tol': 1e-12}
        for n in self.sweep(to=to):
            self.orthogonalize_site(n=n, to=to, normalize=normalize)
            discarded = self.diagonalize_central(opts=opts, normalize=normalize)
            discarded_max = max(discarded_max, discarded)
            self.absorb_central(to=to)
        return discarded_max

    def merge_two_sites(self, bd):
        r"""
        Merge two neighbouring mps sites and return the resulting tensor. 

        Fuse physical indices.

        Parameters
        ----------
        bd : tuple
            (n, n + 1), index of two sites to merge.

        Returns
        ----------
        out : Tensor
            tensor formed from A[n] and A[n + 1]
        """
        nl, nr = bd
        AA = self.A[nl].tensordot(self.A[nr], axes=(self.right, self.left))
        axes = (0, (1, 2), 3) if self.nr_phys == 1 else (0, (1, 3), (2, 4), 5)
        return AA.fuse_legs(axes=axes)

    def unmerge_two_sites(self, AA, bd, opts_svd):
        r"""
        Unmerge tensor into two neighbouring mps sites and a central block using svd to trunctate the bond dimension.

        Provided tensor should be fused consistently with `merge_two_sites`.

        Parameters
        ----------
        AA : Tensor
            Tensor to be unmerged. It gets unfused in place during the operation.

        bd : tuple
            (n, n + 1), index of two sites to merge.

        Returns
        ----------
        out : Tensor
            tensor formed from A[n] and A[n + 1]
        """
        nl, nr = bd
        axes = (1,) if self.nr_phys == 1 else (1, 2)
        AA = AA.unfuse_legs(axes=axes)
        axes = ((0, 1), (2, 3)) if self.nr_phys == 1 else ((0, 1, 3), (2, 4, 5))
        self.pC = bd
        self.A[nl], self.A[bd], self.A[nr] = AA.svd_with_truncation(axes=axes, sU=-1, **opts_svd)

    def get_bond_dimensions(self):
        r"""
        Returns bond dimensions of mps.

        Returns
        -------
        Ds : list
            list of bond dimensions on virtual legs from first to last,
            including "trivial" leftmost and rightmost virtual indices.
        """
        Ds = [self.A[n].get_shape(self.left[0]) for n in self.sweep(to='last')]
        Ds.append(self.A[self.last].get_shape(self.right[0]))
        return Ds

    def get_bond_charges_dimensions(self):
        r"""
        Returns charges and dimensions of all virtual mps bonds.

        Returns
        -------
        tDs : list
            list of charges and corresponding dimensions on virtual mps bonds from first to last,
            including "trivial" leftmost and rightmost virtual indices.
        """
        tDs = [self.A[n].get_leg_structure(self.left[0]) for n in self.sweep(to='last')]
        tDs.append(self.A[self.last].get_leg_structure(self.right[0]))
        return tDs

    def get_entropy(self, alpha=1):
        r"""
        Entropy for a bipartition on each bond

        Returns
        -------
        Entropy : list
            list of bond entropies on virtual legs.
        """
        Entropy = [0]*self.N
        self.canonize_sweep(to='last', normalize=False)
        self.absorb_central(to='first')
        for n in self.sweep(to='first'):
            self.orthogonalize_site(n=n, to='first')
            Entropy[n] = entropy(self.A[self.pC], alpha=alpha)[0]
            self.absorb_central(to='first')
        return Entropy

    def get_Schmidt_values(self):
        r"""
        Schmidt values for a bipartition on each bond

        Returns
        -------
        SV : dictionary
            each element of dictionary is a Schmidt values for a bipartition on this bond
        """
        SV = {}
        self.canonize_sweep(to='last', normalize=False)
        self.absorb_central(to='first')
        for n in self.sweep(to='first'):
            self.orthogonalize_site(n=n, to='first')
            _, sv, _ = self.A[self.pC].svd()
            SV[n] = sv
            self.absorb_central(to='first')
        return SV

    def save_to_dict(self):
        r"""
        Writes Tensor-s of Mps into a dictionary

        Returns
        -------
        out_dict : dictionary of dictionaries
            each element represents a tensor in the chain from first to last.
        """
        out_dict = {}
        for n in self.sweep(to='last'):
            out_dict[n] = Tensor_to_dict(self.A[n])
        return out_dict

    def save_to_hdf5(self, file, in_file_path):
        r"""
        Writes Tensor-s of Mps into a HDF5 file

        Parameters
        -----------
        file: File
            A 'pointer' to a file opened by a user

        in_file_path: File
            Name of a group in the file, where the Mps will be saved
        """
        for n in self.sweep(to='last'):
            Tensor_to_hdf5(self.A[n], file, in_file_path+str(n))
