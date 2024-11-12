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
""" Mps structure and its basic manipulations. """
from __future__ import annotations
from ... import tensor, initialize, YastnError
from ._mps_parent import _MpsMpoParent

###################################
#   auxiliary for basic algebra   #
###################################

def Mps(N) -> yastn.tn.mps.MpsMpoOBC:
    r""" Generate empty MPS for system of `N` sites, fixing :code:`nr_phys=1`. """
    return MpsMpoOBC(N, nr_phys=1)


def Mpo(N, periodic=False) -> yastn.tn.mps.MpsMpoOBC | yastn.tn.mps.MpoPBC:
    r"""
    Generate empty MPO for system of `N` sites, fixing :code:`nr_phys=2`.

    A flag :code:`periodic` allows initializing periodic MPO,
    which is a special class supported as an operator in MPS environments.
    """
    if periodic:
        return MpoPBC(N, nr_phys=2)
    return MpsMpoOBC(N, nr_phys=2)


def add(*states, amplitudes=None) -> yastn.tn.mps.MpsMpoOBC:
    r"""
    Linear superposition of several MPS/MPOs with specific amplitudes, i.e., :math:`\sum_j \textrm{amplitudes[j]}{\times}\textrm{states[j]}`.

    Compression (truncation of bond dimensions) is not performed.

    Parameters
    ----------
    states: Sequence[yastn.tn.mps.MpsMpoOBC]

    amplitudes: Sequence[scalar]
        If :code:`None`, all amplitudes are assumed to be 1.
    """
    if amplitudes is None:
        amplitudes = [1] * len(states)

    if len(states) != len(amplitudes):
        raise YastnError('Number of MpsMpoOBC-s to add must be equal to the number of coefficients in amplitudes.')

    if any(not isinstance(state, MpsMpoOBC) for state in states):
        raise YastnError('All added states should be MpsMpoOBC-s.')

    phi = MpsMpoOBC(N=states[0].N, nr_phys=states[0].nr_phys)

    if any(psi.N != phi.N for psi in states):
        raise YastnError('All MpsMpoOBC to add must have equal number of sites.')
    if any(psi.nr_phys != phi.nr_phys for psi in states):
        raise YastnError('All states to add should be either Mps or Mpo.')
    if any(psi.pC != None for psi in states):
        raise YastnError('Absorb central block of MpsMpoOBC-s before calling add.')
    # legf = states[0][phi.first].get_legs(axes=0)
    # legl = states[0][phi.last].get_legs(axes=2)
    #if any(psi.virtual_leg('first') != legf or psi.virtual_leg('last') != legl for psi in states):
    #    raise YastnError('MPS: Addition')

    amplitudes = [x * psi.factor for x, psi in zip(amplitudes, states)]

    n = phi.first
    d = {(j,): psi.A[n] * amplitudes[j] for j, psi in enumerate(states)}
    common_legs = (0, 1) if phi.nr_phys == 1 else (0, 1, 3)
    phi.A[n] = initialize.block(d, common_legs)

    common_legs = (1,) if phi.nr_phys == 1 else (1, 3)
    for n in phi.sweep(to='last', df=1, dl=1):
        d = {(j, j): psi.A[n] for j, psi in enumerate(states)}
        phi.A[n] = initialize.block(d, common_legs)

    n = phi.last
    d = {(j,): psi.A[n] for j, psi in enumerate(states)}
    common_legs = (1, 2) if phi.nr_phys == 1 else (1, 2, 3)
    phi.A[n] = initialize.block(d, common_legs)

    return phi


def multiply(a, b, mode=None) -> yastn.tn.mps.MpsMpoOBC:
    r"""
    Performs MPO-MPS product resulting in a new MPS, or
    MPO-MPO product resulting in a new MPO.

    For MPS/MPO with no symmetry, the bond dimensions
    of the result are given by the product of the bond dimensions of the inputs.
    For symmetric MPS/MPO the virtual spaces of the result are given by
    :ref:`fusion <tensor/algebra:fusion of legs (reshaping)>`
    (i.e. resolving tensor product into direct sum) of the virtual spaces
    of the inputs.

    .. math::
        V^{\textrm{result}}_{j,j+1} = V^{a}_{j,j+1} \otimes V^{b}_{j,j+1}
        = \sum_{c} V^{\textrm{result},c}_{j,j+1},

    where the charge sectors :math:`V^{\textrm{result},c}` are computed during fusion.

    .. note::
        One can equivalently call :code:`a @ b`.

    Parameters
    ----------
        a, b: yastn.tn.mps.MpsMpoOBC, yastn.tn.mps.MpsMpoOBC
            a pair of MPO and MPS or two MPO's to be multiplied

        mode: str
           mode for :meth:`Tensor.fuse_legs()<yastn.Tensor.fuse_legs>`; If :code:`None` (default)
           use default setting from YASTN tensor's
           :ref:`configuration<tensor/configuration:yastn configuration>`.
    """
    if not isinstance(a, MpsMpoOBC) or not isinstance(b, MpsMpoOBC):
        raise YastnError('multiply requres two MpsMpoOBC-s.')

    if a.N != b.N:
        raise YastnError('MpsMpoOBC-s to multiply must have equal number of sites.')

    if a.pC is not None or b.pC is not None:
        raise YastnError('Absorb central blocks of MpsMpoOBC-s before calling multiply.')

    nr_phys = a.nr_phys + b.nr_phys - 2

    if a.nr_phys == 1:
        raise YastnError(' Multiplication by MPS from left is not supported.')

    phi = MpsMpoOBC(N=a.N, nr_phys=nr_phys)

    axes_fuse = ((0, 3), 1, (2, 4)) if b.nr_phys == 1 else ((0, 3), 1, (2, 4), 5)
    for n in phi.sweep():
        phi.A[n] = tensor.tensordot(a.A[n], b.A[n], axes=(3, 1)).fuse_legs(axes_fuse, mode)
    phi.A[phi.first] = phi.A[phi.first].drop_leg_history(axes=0)
    phi.A[phi.last] = phi.A[phi.last].drop_leg_history(axes=2)
    phi.factor = a.factor * b.factor
    return phi


class MpoPBC(_MpsMpoParent):
    # The basic structure of MPO with `N` sites and PBC.

    def __init__(self, N=1, nr_phys=2):
        r"""
        Class to isolate a special case of periodic Mpo,
        which in some functions can be applied on Mps with OBC.
        """
        super().__init__(N=N, nr_phys=2)
        self.tol = None



class MpsMpoOBC(_MpsMpoParent):
    # The basic structure of MPS/MPO with `N` sites and OBC

    def __init__(self, N=1, nr_phys=1):
        r"""
        Initialize empty MPS (:code:`nr_phys=1`) or MPO (:code:`nr_phys=2`)
        for system of `N` sites. Empty MPS/MPO has no tensors assigned.

        MpsMpoOBC tensors (sites) are indexed by integers :math:`0, 1, 2, \ldots, N-1`,
        where :math:`0` corresponds to the `'first'` site.
        They can be accessed with ``[]`` operator.
        MPS/MPO can contain a central block associated with a bond and indexed by a tuple :math:`(n, n+1)`.
        At most one central block is allowed.

        :ref:`The legs' order <mps/properties:index convention>` of each tensor is:
        (0) virtual leg pointing towards the first site, (1) 1st physical leg, i.e., :math:`|\textrm{ket}\rangle`,
        (2) virtual leg pointing towards the last site, and, in case of MPO, (3) 2nd physical leg, i.e., :math:`\langle \textrm{bra}|`.

        The canonical form can be used with the MPS/MPO freely.
        If the MPS/MPO is not normalized the amplitude is kept as :code:`self.factor` paramter of the object.
        If the object is normalized, then :code:`self.factor=1`.
        """
        super().__init__(N=N, nr_phys=nr_phys)
        self.pC = None  # index of the central block, None if it does not exist

    def __add__(self, phi) -> yastn.tn.mps.MpsMpoOBC:
        """ Sum of two Mps's or two Mpo's. """
        return add(self, phi)

    def __sub__(self, phi) -> yastn.tn.mps.MpsMpoOBC:
        """ Subtraction of two Mps's or two Mpo's. """
        return add(self, phi, amplitudes=(1, -1))

    def __matmul__(self, phi) -> yastn.tn.mps.MpsMpoOBC:
        """ Multiply Mpo by Mpo or Mps. """
        return multiply(self, phi)

    def orthogonalize_site_(self, n, to='first', normalize=True) -> None:
        r"""
        Performs QR (or RQ) decomposition of on-site tensor at :code:`n`-th position.
        Two modes of usage are

            * ``to='last'`` - Advance left canonical form: Assuming first `n - 1` sites are already
              in the left canonical form, brings n-th site to left canonical form,
              i.e., extends left canonical form by one site towards :code:`'last'` site::

                 --A---            --
                |  |    = Identity|
                 --A*--            --

            * ``to='first'`` - Advance right canonical form: Assuming all `m > n` sites are already
              in right canonical form, brings n-th site to right canonical form, i.e.,
              extends right canonical form by one site towards :code:`'first'` site::

                --A---    --
                  |   | =   |Identity
                --A*--    --

        Parameters
        ----------
            n: int
                index of site to be orthogonalized

            to: str
                a choice of canonical form: :code:`'last'` or :code:`'first'`.

            normalize: bool
                Whether to keep track of the norm by accumulating it in self.factor;
                default is True, i.e., sets the norm to unity.
                The central blocks at the end of the procedure is normalized to unity.
        """
        if self.pC is not None:
            raise YastnError('Only one central block is allowed. Attach the existing central block before orthogonalizing site.')

        if to == 'first':
            self.pC = (n - 1, n)
            ax = (1, 2) if self.nr_phys == 1 else (1, 2, 3)
            # A = ax(A)--Q--0 1--R--0(0A) => 0--Q*--    --
            #                                   1   2 =   |Identity
            #                                0--Q---    --
            self.A[n], R = self.A[n].qr(axes=(ax, 0), sQ=-1, Qaxis=0, Raxis=1)
        elif to == 'last':
            self.pC = (n, n + 1)
            ax = (0, 1) if self.nr_phys == 1 else (0, 1, 3)
            # A = ax(A)--Q--2 0--R--1(2A) =>  --Q*--2            --
            #                                0  1     = Identity|
            #                                 --Q---2            --
            self.A[n], R = self.A[n].qr(axes=(ax, 2), sQ=1, Qaxis=2)
        else:
            raise YastnError('"to" should be in "first" or "last"')
        nR = R.norm()
        self.A[self.pC] = R / nR
        self.factor = 1 if normalize else self.factor * nR

    def diagonalize_central_(self, opts_svd, normalize=True) -> number:
        r"""
        Use svd() to perform SVD of the central block C = U @ S @ V, where S contains singular (Schmidt) values.
        Truncation is done based on opts_svd.

        Attach U and V respectively to the left and right sites
        (or to the central site at the edges when there are no left or right sites).

        Returns the norm of truncated Schmidt values, normalized by the norm of all Schmidt values.

        Parameters
        ----------
        opts_svd: dict
            Options passed to :meth:`yastn.linalg.truncation_mask`. It includes information on how to truncate the Schmidt values.

        normalize: bool
            Whether to keep track of the norm of retained Schmidt values
            by accumulating it in self.factor; default is True, i.e., sets the norm to unity.
            The truncated Schmidt values (central block) at the end of the procedure are normalized to unity.
        """
        # if self.pC is None:  does not happen now
        #     return 0
        discarded = 0.
        if self.pC is not None:

            U, S, V = tensor.svd(self.A[self.pC], axes=(0, 1), sU=1)
            nSold = S.norm()

            mask = tensor.truncation_mask(S, **opts_svd)
            nSout = tensor.bitwise_not(mask).apply_mask(S, axes=0).norm()
            discarded = nSout / nSold

            U, S, V = mask.apply_mask(U, S, V, axes=(1, 0, 0))
            nS = S.norm()
            self.A[self.pC] = S / nS
            self.factor = 1 if normalize else self.factor * nS

            n1, n2 = self.pC

            if n1 >= self.first:
                ax = (-0, -1, 1) if self.nr_phys == 1 else (-0, -1, 1, -3)
                self.A[n1] = tensor.ncon([self.A[n1], U], (ax, (1, -2)))
            else:
                self.A[self.pC] = U @ self.A[self.pC]

            if n2 <= self.last:
                self.A[n2] = V @ self.A[n2]
            else:
                self.A[self.pC] = self.A[self.pC] @ V
        return discarded

    def remove_central_(self) -> None:
        """ Removes (ignores) the central block. Do nothing if is does not exist. """
        if self.pC is not None:
            del self.A[self.pC]
            self.pC = None

    def absorb_central_(self, to='last') -> None:
        r"""
        Absorb central block towards the first or the last site.

        If the central block is outside of the chain, it goes in the one direction possible.
        Do nothing if central does not exist.

        Parameters
        ----------
        to: str
            'last' or 'first'.
        """
        if self.pC is not None:
            C = self.A.pop(self.pC)
            n1, n2 = self.pC
            self.pC = None

            if (to == 'first' and n1 >= self.first) or n2 > self.last:
                ax = (-0, -1, 1) if self.nr_phys == 1 else (-0, -1, 1, -3)
                self.A[n1] = tensor.ncon([self.A[n1], C], (ax, (1, -2)))
            else:  # (to == 'last' and n2 <= self.last) or n1 < self.first
                self.A[n2] = C @ self.A[n2]

    def norm(self) -> number:
        """ Calculate norm of MPS/MPO via canonization. """
        phi = self.shallow_copy()
        #if not phi.is_canonical(to='first'):
        phi.canonize_(to='first', normalize=False)
        return phi.factor

    def canonize_(self, to='first', normalize=True) -> yastn.tn.mps.MpsMpoOBC:
        r"""
        Sweep through the MPS/MPO and put it in right/left canonical form
        (:code:`to='first'` or :code:`to='last'`, respectively)
        using :meth:`QR decomposition<yastn.linalg.qr>`.
        It is assumed that tensors are enumerated
        by index increasing from 0 (:code:`first`) to N-1 (:code:`last`).

        Finally, the trivial central block of dimension *1* obtained for terminal sites
        is attached at the end of the chain.

        It updates MPS/MPO in place.

        Parameters
        ----------
        to: str
            :code:`'first'` (default) or :code:`'last'`.

        normalize: bool
            Whether to keep track of the norm of the state by accumulating it in self.factor;
            default is True, i.e., sets the norm to unity.
            The individual tensors at the end of the procedure are in a proper canonical form.
        """
        self.absorb_central_(to=to)
        for n in self.sweep(to=to):
            self.orthogonalize_site_(n=n, to=to, normalize=normalize)
            self.absorb_central_(to=to)
        return self

    def is_canonical(self, to='first', n=None, tol=1e-12) -> bool:
        r"""
        Check if MPS/MPO is in a canonical form.

        Parameters
        ----------
        to: str
            :code:`'first'` (default) or code:`'last'`.

        n: int
            Can check a single site if int provided. If None, check all sites.

        tol: float
            Tolerance of the check. The default is 1e-12.
        """
        if to == 'first':
            # 0--A*--
            #    1   2
            # 0--A---
            cl = (1, 2) if self.nr_phys == 1 else (1, 2, 3)
        else: # to == 'last':
            #  --A*--2
            # 0  1
            #  --A---2
            cl = (0, 1) if self.nr_phys == 1 else (0, 1, 3)
        it = self.sweep(to=to) if n is None else [n]
        for n in it:
            x = tensor.tensordot(self.A[n], self.A[n].conj(), axes=(cl, cl))
            x = x.drop_leg_history()
            x0 = initialize.eye(config=x.config, legs=x.get_legs((0, 1)))
            if (x - x0.diag()).norm() > tol:  # == 0
                return False
        return self.pC is None  # True if no central block

    def truncate_(self, to='last', opts_svd=None, normalize=True) -> number:
        r"""
        Sweep through the MPS/MPO and put it in right/left canonical form
        (:code:`to='first'` or :code:`to='last'`, respectively)
        using :meth:`yastn.linalg.svd_with_truncation` decomposition.
        It is assumed that tensors are enumerated
        by index increasing from 0 (:code:`first`) to N-1 (:code:`last`).

        Access to singular values during sweeping allows truncation of virtual spaces.

        The truncation is effective when it is done in the canonical form.
        The canonical form has to be ensured prior truncation by setting it opposite to the sweep of :code:`truncate_`.
        I.e., prepare MPS/MPO in the right canonical form (:code:`to='first'`) for truncation with :code:`to='last'`
        and left canonical form (:code:`to='last'`) for truncation with :code:`to='first'`.

        The MPS/MPO is updated in place.

        Returns the norm of truncated elements normalized by the norm of the untruncated state.

        Parameters
        ----------
        to: str
            :code:`'last'` (default) or :code:`'first'`.

        normalize: bool
            Whether to keep in self.factor the norm of the initial state projected on
            the direction of the truncated state; default is True, i.e., sets the norm to unity.
            The individual tensors at the end of the procedure are in a proper canonical form.

        opts_svd: dict
            options passed to :meth:`yastn.linalg.svd_with_truncation`,
            including options governing truncation.
        """
        discarded2_total = 0.
        if opts_svd is None:
            raise YastnError("truncate_: provide opts_svd.")

        for n in self.sweep(to=to):
            self.orthogonalize_site_(n=n, to=to, normalize=normalize)
            discarded_local = self.diagonalize_central_(opts_svd=opts_svd, normalize=normalize)
            discarded2_local = discarded_local ** 2
            discarded2_total = discarded2_local + discarded2_total - discarded2_total * discarded2_local
            self.absorb_central_(to=to)
        return self.config.backend.sqrt(discarded2_total)

    def merge_two_sites(self, bd) -> yastn.Tensor:
        r"""
        Merge two neighbouring mps sites and return the resulting tensor.

        Fuse physical indices.

        Parameters
        ----------
        bd: tuple
            (n, n + 1), index of two sites to merge.

        Returns
        -------
        tensor formed from A[n] and A[n + 1]
        """
        nl, nr = bd
        return tensor.tensordot(self.A[nl], self.A[nr], axes=(2, 0))

    def unmerge_two_sites_(self, AA, bd, opts_svd) -> number:
        r"""
        Unmerge rank-4 tensor into two neighbouring MPS sites and a central block
        using :meth:`yastn.linalg.svd_with_truncation` to trunctate the bond dimension.

        Input tensor should be a result of :meth:`merge_two_sites` (or fused consistently).

        Parameters
        ----------
        AA: Tensor
            Tensor to be unmerged. It gets unfused in place during the operation.

        bd: tuple
            (n, n + 1), index of two sites to merge.

        Returns
        -------
        normalized discarded weight :math:`\sum_{i\in\textrm{discarded}}\lambda_i/\sum_i\lambda_i`,
        where :math:`\lambda_i` are singular values across the bond.
        """
        nl, nr = bd
        axes = ((0, 1), (2, 3)) if self.nr_phys == 1 else ((0, 1, 2), (3, 4, 5))
        self.pC = bd
        U, S, V = tensor.svd(AA, axes=axes, sU=1, Uaxis=2, **opts_svd)
        mask = tensor.truncation_mask(S, **opts_svd)
        self.A[nl], self.A[bd], self.A[nr] = mask.apply_mask(U, S, V, axes=(2, 0, 0))
        return tensor.bitwise_not(mask).apply_mask(S, axes=0).norm() / S.norm()  # discarded weight


    def get_entropy(self, alpha=1) -> Sequence[number]:
        r"""
        Entropy of bipartition across each bond, along MPS/MPO from the first to the last site,
        including "trivial" leftmost and rightmost cuts.  This gives a list with N+1 elements.

        Parameters
        ----------
        alpha: int
            Order of Renyi entropy.
            alpha == 1 is von Neuman entropy: -Tr(a log2(a))
            otherwise: 1/(1-alpha) log2(Tr(a ** alpha))
            The default value is 1, which corresponds to the von Neumann entropy.
        """
        schmidt_spectra = self.get_Schmidt_values()
        return [tensor.entropy(spectrum ** 2, alpha=alpha) for spectrum in schmidt_spectra]

    def get_Schmidt_values(self) -> Sequence[yastn.Tensor]:
        r"""
        Schmidt values for bipartition across all bonds along MPS/MPO from the first to the last site,
        including "trivial" leftmost and rightmost cuts. This gives a list with N+1 elements.
        Schmidt values are stored as diagonal tensors and are normalized.
        """
        SV = []
        psi = self.shallow_copy()
        psi.canonize_(to='first')
        # canonize_ attaches trivial central block at the end
        axes = (0, (1, 2)) if self.nr_phys == 1 else (0, (1, 2, 3))
        _, sv, _ = tensor.svd(psi.A[self.first], axes=axes, sU=1)
        SV.append(sv)
        for n in psi.sweep(to='last'):
            psi.orthogonalize_site_(n=n, to='last')
            _, sv, _ = tensor.svd(psi.A[psi.pC], sU=1)
            SV.append(sv)
            psi.absorb_central_(to='last')
        return SV
