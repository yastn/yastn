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
from __future__ import annotations

import numpy as np
from ._mps_obc import Mpo, Mps, MpsMpoOBC
from ...initialize import load_from_hdf5 as load_from_hdf5_tensor
from ...initialize import rand
from ...operators import Qdit
from ...tensor import Leg, YastnError, gaussian_leg


def product_mps(vectors, N=None) -> MpsMpoOBC:
    r"""
    Generate an MPS with bond-dimension 1 from a list of vectors that get assigned to consecutive MPS sites.

    If ``N`` is provided, vectors are cyclicly iterated to fill in ``N`` MPS sites.

    Parameters
    ----------
    vectors: Sequence[yastn.Tensor] | yastn.Tensor
        Tensors will be attributed to consecutive MPS sites.
        They can have non-zero charges that will be converted into matching MPS virtual legs.
        Each tensor should have ``ndim=1``.

    N: Optional[int]
        number of MPS sites. By default, it is equal to the number of provided `vectors`.
    """
    return _product_MpsMpoOBC(vectors, N=N, nr_phys=1)


def product_mpo(operators, N=None) -> MpsMpoOBC:
    r"""
    Generate an MPO with bond-dimension 1 from a list of operators that get assigned to consecutive MPO sites.

    If ``N`` is provided, operators are cyclicly iterated to fill in ``N`` MPO sites.

    Parameters
    ----------
    operators: Sequence[yastn.Tensor] | yastn.Tensor
        Tensors will be attributed to consecutive MPS sites.
        They can have non-zero charges, that will be converted into matching MPO virtual legs.
        Each tensor should have ``ndim=2``.

    N: Optional[int]
        number of MPO sites. By default, it is equal to the number of provided `operators`.

    Example
    -------

    ::

        # This function can help set up an identity MPO,
        # which is the base ingredient for a few other functions
        # generating more complicated MPOs and MPSs.

        import yastn
        import yastn.tn.mps as mps

        ops = yastn.operators.Spin12(sym='Z2')
        I = mps.product_mpo(ops.I(), N=8)

        # Here, each site has the same local physical Hilbert space
        # of dimension 2, consistent with predefined spin-1/2 operators.
        # The MPO I uniquely identifies those local Hilbert spaces.
    """
    return _product_MpsMpoOBC(operators, N=N, nr_phys=2)


def _product_MpsMpoOBC(vectors, N=None, nr_phys=1) -> MpsMpoOBC:
    """ handles product mpo and mps"""
    try:  # handle inputing single bare Tensor
        vectors = list(vectors)
    except TypeError:
        vectors = [vectors]

    if N is None:
        N = len(vectors)

    psi = MpsMpoOBC(N=N, nr_phys=nr_phys)

    if nr_phys == 1 and any(vec.ndim != 1 for vec in vectors):
        raise YastnError("Vector should have ndim = 1.")
    if nr_phys == 2 and any(vec.ndim != 2 for vec in vectors):
        raise YastnError("Operator should have ndim = 2.")

    Nv = len(vectors)
    if Nv != N:
        vectors = [vectors[n % Nv] for n in psi.sweep(to='last')]

    rt = vectors[0].config.sym.zero()
    for n, vec in zip(psi.sweep(to='first'), vectors[::-1]):
        vec = vec.add_leg(axis=1, s=1, t=rt)
        rt = vec.n
        psi[n] = vec.add_leg(axis=0, s=-1)
    return psi


def random_mps(I, n=None, D_total=8, sigma=1, distribution=(-1, 1), dtype='float64', **kwargs) -> MpsMpoOBC:
    r"""
    Generate a random MPS of total charge ``n`` and bond dimension ``D_total``.

    Local Hilbert spaces are read from ket spaces
    of provided MPS or MPO ``I``. For instance, ``I`` can be an identity MPO.
    The number of sites and Tensor config is also inherited from ``I``.

    Parameters
    ----------
    I: MpsMpoOBC
        MPS or MPO that defines local Hilbert spaces.
    n: int | tuple[int] | Sequence[number] | Sequence[tulpe[numbers]] | None
        Total charge of MPS, which equalls the charge on the first virtual leg of MPS.
        Virtual charge bond dimensions along the MPS are drawn from a normal distribution with specified mean.
        ``n`` can be a list that provides the means for all virtual legs (with zero charge on the last leg).
        If ``n`` is a single charge, the means change linearly along the chain from ``n`` to 0.
        If None, which is the default, the MPS charge is set to zero.
    D_total: int
        Largest bond dimension. Note that due to the local (and potentially random) nature of the procedure,
        the desired total bond dimension might not be reached on some bonds, in particular, for higher symmetries.
    sigma: int
        The standard deviation of the normal distribution.
    distribution: tuple[float, float] | str
        Passed to :meth:`yastn.rand`. Range of random numbers, or a string 'normal' for normal distribution.
        The default is (-1, 1).
    dtype: str
        Passed to :meth:`yastn.rand`. Number format, i.e., ``'float64'`` or ``'complex128'``.
    kwargs: dict
        Further parameters passed to :meth:`yastn.gaussian_leg`.

    Example
    -------

    ::

        import numpy as np
        import yastn
        import yastn.tn.mps as mps

        ops = yastn.operators.SpinlessFermions(sym='U1')
        N = 13
        I = mps.product_mpo(ops.I(), N=N)

        psi = mps.random_mps(I, n=6, D_total=8)
        # Random MPS with 13 sites occupied by 6 fermions (fixed by U1 symmetry),
        # and maximal bond dimension 8.

        n_profile = np.cos(np.linspace(0, np.pi / 2, N + 1)) * 7
        phi = mps.random_mps(I, n=n_profile, D_total=32, sigma=2)
        # Here mean virtual charges along the chain are distributed according to n_profile,
        # sigma = 2 gives a broader spread of charges on virtual legs.
        # MPS encodes a state with 7 particles.

    """
    config = I.config

    if n is None:
        n = config.sym.zero()
    an = np.array(n)
    if an.size == config.sym.NSYM:
        n_profile = np.linspace(1, 0, I.N + 1).reshape(-1, 1) * an.reshape(1, config.sym.NSYM)
    elif an.size == config.sym.NSYM * (I.N + 1):
        n_profile = an.reshape(I.N + 1, config.sym.NSYM)
    else:
        raise YastnError("Wrong number of elements in 'n'. It should be a charge on the first virtual leg, or list of charges on all len(I) + 1 virtual legs.")

    nr = tuple(np.round(n_profile[I.N]).astype(np.int64).tolist())
    if nr != config.sym.zero():
        raise YastnError("The charge on the last virtual leg should be zero.")
    n0 = np.round(n_profile[0]).astype(np.int64)
    fn0 = config.sym.fuse(n0.reshape(1, 1, config.sym.NSYM), (1,), 1).ravel()
    n0, fn0 = tuple(n0.tolist()), tuple(fn0.tolist())
    if n0 != fn0:
        raise YastnError("Charge on the first virtual leg is not consistent with tensor symmetry.")

    psi = Mps(I.N)
    lr = Leg(config, s=1, t=(nr,), D=(1,))
    for site in psi.sweep(to='first'):
        lp = I[site].get_legs(axes=1)  # ket leg of MPS/MPO
        nl = n_profile[site]  # mean n changes linearly along the chain
        if site != psi.first:
            ll = gaussian_leg(config, s=-1, n=nl, D_total=D_total, sigma=sigma, legs=[lp, lr], **kwargs)
        else:
            ll = Leg(config, s=-1, t=(n0,), D=(1,))
        psi.A[site] = rand(config, distribution=distribution, legs=[ll, lp, lr], dtype=dtype)
        lr = psi.A[site].get_legs(axes=0).conj()
    if sum(lr.D) == 1:
        return psi
    raise YastnError("Random mps is a zero state. Check parameters, or try running again in this is due to randomness of the initialization.")


def random_mpo(I, D_total=8, sigma=1, distribution=(-1, 1), dtype='float64', **kwargs) -> MpsMpoOBC:
    r"""
    Generate a random MPO with bond dimension ``D_total``.

    The number of sites and local bra and ket spaces of MPO follow
    from provided MPO ``I``, e.g., an identity MPO.
    ``I`` can be an MPS, in which case its ket spaces are used and conjugated for bra spaces.

    Parameters
    ----------
    I: MpsMpoOBC
        MPS or MPO that defines local spaces.
    D_total: int
        Largest bond dimension. Note that due to the random and local nature of the procedure,
        the desired total bond dimension might not be reached on some bonds,
        in particular, for higher symmetries.
    sigma: int
        Standard deviation of a normal distribution from which dimensions of charge sectors are drawn.
    distribution: tuple[float, float] | str
        Passed to :meth:`yastn.rand`. Range of random numbers, or a string 'normal' for normal distribution.
        The default is (-1, 1).
    dtype: str
        Passed to :meth:`yastn.rand`. Number format, i.e., ``'float64'`` or ``'complex128'``.
    kwargs: dict
        Further parameters passed to :meth:`yastn.gaussian_leg`.
    """
    config = I.config
    n0 = config.sym.zero()
    psi = Mpo(I.N)

    lr = Leg(config, s=1, t=(n0,), D=(1,),)
    for site in psi.sweep(to='first'):
        lp = I[site].get_legs(axes=1)
        lpc = I[site].get_legs(axes=3) if I[site].ndim == 4 else lp.conj()
        if site != psi.first:
            ll = gaussian_leg(config, s=-1, n=n0, D_total=D_total, sigma=sigma, legs=[lp, lr, lpc], **kwargs)
        else:
            ll = Leg(config, s=-1, t=(n0,), D=(1,),)
        psi.A[site] = rand(config, distribution=distribution, legs=[ll, lp, lr, lpc], dtype=dtype)
        lr = psi.A[site].get_legs(axes=0).conj()
    if sum(lr.D) == 1:
        return psi
    raise YastnError("Random mpo is a zero state. Check parameters, or try running again in this is due to randomness of the initialization.")


def random_dense_mps(N, D, d, **kwargs) -> MpsMpoOBC:
    r"""Generate random MPS with physical dimension d and virtual dimension D."""
    I = product_mpo(Qdit(d=d, **kwargs).I(), N)
    return random_mps(I, D_total=D)


def random_dense_mpo(N, D, d, **kwargs) -> MpsMpoOBC:
    r"""Generate random MPO with physical dimension d and virtual dimension D."""
    I = product_mpo(Qdit(d=d, **kwargs).I(), N)
    return random_mpo(I, D_total=D)


def load_from_dict(config, in_dict) -> MpsMpoOBC:
    r"""
    Create MPS/MPO from dictionary.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    in_dict: dict
        dictionary containing serialized MPS/MPO, i.e.,
        a result of :meth:`yastn.tn.mps.MpsMpoOBC.save_to_dict`.
    """
    return MpsMpoOBC.from_dict(in_dict, config)


def load_from_hdf5(config, file, my_address) -> MpsMpoOBC:
    r"""
    Create MPS/MPO from HDF5 file.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    file: File
        A `pointer` to a file opened by the user

    my_address: str
        Name of a group in the file, where the Mps is saved, e.g., './state/'
    """

    nr_phys = int(file[my_address].get('nr_phys')[()])
    N = file[my_address].get('N')
    N = len(file[my_address+'/A'].keys()) if N is None else int(N[()])
    out_Mps = MpsMpoOBC(N, nr_phys=nr_phys)

    factor = file[my_address].get('factor')
    if factor:
        out_Mps.factor = factor[()]
    for n in range(out_Mps.N):
        out_Mps.A[n] = load_from_hdf5_tensor(config, file, my_address+'/A/'+str(n))
    return out_Mps


def mps_from_tensor(ten, nr_phys=1, canonize='last', opts_svd=None) -> MpsMpoOBC:
    r"""
    Generate MPS from a tensor if ``nr_phys=1`` (the default)
    and MPO for ``nr_phys=2``

    ::

        ┌─────────── ... ──────┐
        |                      |  ==>  MPS of N sites
        └──┬─────┬── ... ───┬──┘
           |     |          |
           0     1         N-1

           1     3        2N-1
           |     |          |
        ┌──┴─────┴── ... ───┴──┐
        |                      |  ==>  MPO of N sites
        └──┬─────┬── ... ───┬──┘
           |     |          |
           0     2        2N-2

    Parameters
    ----------
    canonize: str
        'first', 'last', or 'balance'. The default is 'first'.
        Canonize to 'first' or 'last' site.
        For 'balance', square roots of singular values are attached symmetrically to left and right tensors.
    opts_svd: dict | None
        Truncate MPS/MPO by passing svd_opts to :meth:`yastn.linalg.truncation_mask`.
        The default is None, which sets truncation tolerance to 1e-14.
    """
    N = ten.ndim // nr_phys
    psi = MpsMpoOBC(N, nr_phys=nr_phys)
    ten = ten.add_leg(axis=0, s=-1).add_leg(axis=-nr_phys, s=1)

    if opts_svd is None:
        opts_svd = {'tol': 1e-14}

    for n in psi.sweep(to='last', dl=1):
        axes = (list(range(nr_phys + 1)), list(range(nr_phys + 1, ten.ndim)))
        psi.A[n], S, V = ten.svd_with_truncation(axes=axes, sU=1, Uaxis=2, **opts_svd)
        ten = S @ V

    psi.factor = ten.norm()
    psi.A[psi.last] = ten / psi.factor

    if canonize == 'first':
        psi.canonize_(to='first', normalize=False)
    elif canonize == 'balance':
        psi.absorb_central_(to='first')
        for n in psi.sweep(to='first', df=1):
            psi.orthogonalize_site_(n=n, to='first', normalize=False)
            pC = psi.pC
            U, S, V = psi[pC].svd(axes=(0, 1))
            S2 = S.sqrt()
            psi.A[pC] = S2 @ V
            psi.absorb_central_(to='last')
            psi.pC = pC
            psi.A[pC] = U @ S2
            psi.absorb_central_(to='first')
    return psi


def mpo_from_tensor(ten, canonize='balance', opts_svd=None) -> MpsMpoOBC:
    r"""
    Generate MPO from a tensor.
    Shortcut for :meth:`yastn.tn.mps.mps_from_tensor` with ``nr_phys=2``.
    """
    return mps_from_tensor(ten, nr_phys=2, canonize=canonize, opts_svd=opts_svd)
