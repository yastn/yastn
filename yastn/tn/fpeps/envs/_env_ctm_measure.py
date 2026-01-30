# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" Common measure functions for EnvCTM and EnvBoudndaryMPS """

import scipy.sparse.linalg as sla

from ._env_window import EnvWindow, _measure_2site, _measure_nsite, _sample
from .._gates_auxiliary import fkron, gate_fix_swap_gate, clear_operator_input
from .._doublePepsTensor import DoublePepsTensor
from .._geometry import Site, is_bond, is_site
from ... import mps
from ....initialize import rand
from ....operators import sign_canonical_order
from ....tensor import YastnError, Tensor, tensordot, vdot, split_data_and_meta, combine_data_and_meta



def measure_1site(self, O, site=None) -> dict:
    r"""
    Calculate local expectation values within CTM environment.

    Returns a number if ``site`` is provided.
    If ``None``, returns a dictionary {site: value} for all unique lattice sites.

    Parameters
    ----------
    env: EnvCtm
        Class containing CTM environment tensors along with lattice structure data.

    O: Tensor
        Single-site operator
    """
    return_one = False
    if site is None:
        sites = self.sites()
    elif is_site(site):  # single site
        return_one = True
        sites = [site]
    elif all(is_site(ss) for ss in site):
        sites = site  # a few sites
    else:
        raise YastnError("site should be None, Site, or Sequence[Site]")
    opdict = clear_operator_input(O, sites)
    if return_one and len(opdict[site]) > 1:
        return_one = False

    out = {}
    for site, ops in opdict.items():
        lenv = self[site]
        ten = self.psi[site]
        vect = (lenv.l @ lenv.tl) @ (lenv.t @ lenv.tr)
        vecb = (lenv.r @ lenv.br) @ (lenv.b @ lenv.bl)

        tmp = tensordot(vect, ten, axes=((2, 1), (0, 1)))
        val_no = tensordot(vecb, tmp, axes=((0, 1, 2, 3), (1, 3, 2, 0))).to_number()

        for nz, op in ops.items():
            if isinstance(ten, DoublePepsTensor):  # 2-layers PEPS
                ten.set_operator_(op)
            else:  # for a single-layer Peps, replace with new peps tensor
                ten = op
            tmp = tensordot(vect, ten, axes=((2, 1), (0, 1)))
            val_op = tensordot(vecb, tmp, axes=((0, 1, 2, 3), (1, 3, 2, 0))).to_number()
            out[site + nz] = val_op / val_no

    return out[site + nz] if return_one else out


def measure_nn(self, O, P, bond=None) -> dict:
    r"""
    Calculate nearest-neighbor expectation values within CTM environment.

    Return a number if the nearest-neighbor ``bond`` is provided.
    If ``None``, returns a dictionary {bond: value} for all unique lattice bonds.

    Parameters
    ----------
    O, P: yastn.Tensor
        Calculate <O_s0 P_s1>.
        P is applied first, which might matter for fermionic operators.

    bond: yastn.tn.fpeps.Bond | tuple[tuple[int, int], tuple[int, int]]
        Bond of the form (s0, s1). Sites s0 and s1 should be nearest-neighbors on the lattice.
    """
    return_one = False
    if bond is None:
        bonds = self.bonds()
    elif is_bond(bond):  # single bond
        return_one = True
        bonds = [bond]
    elif all(is_bond(bb) for bb in bond):
        bonds = bond  # a few sites
    else:
        raise YastnError("bond should be None, Bond, or Sequence[Bond]")
    Osites = list(set(bond[0] for bond in bonds))
    Psites = list(set(bond[1] for bond in bonds))
    Odict = clear_operator_input(O, Osites)
    Pdict = clear_operator_input(P, Psites)

    if return_one and (len(Odict[bond[0]]) > 1 or len(Pdict[bond[1]]) > 1):
        return_one = False

    out = {}
    for bond in bonds:
        for nz0, O in Odict[bond[0]].items():
            for nz1, P in Pdict[bond[1]].items():

                if O.ndim == 2 and P.ndim == 2:
                    O, P = fkron(O, P, sites=(0, 1), merge=False)

                dirn = self.nn_bond_dirn(*bond)
                if O.ndim == 3 and P.ndim == 3:
                    O, P = gate_fix_swap_gate(O, P, dirn, self.f_ordered(*bond))

                s0, s1 = bond if dirn in ('lr', 'tb') else bond[::-1]
                G0, G1 = (O, P) if dirn in ('lr', 'tb') else (P, O)
                env0, env1 = self[s0], self[s1]
                ten0, ten1 = self.psi[s0], self.psi[s1]

                if dirn in ('lr', 'rl'):
                    vecl = (env0.bl @ env0.l) @ (env0.tl @ env0.t)
                    vecr = (env1.tr @ env1.r) @ (env1.br @ env1.b)

                    tmp0 = tensordot(ten0, vecl, axes=((0, 1), (2, 1)))
                    tmp0 = tensordot(env0.b, tmp0, axes=((1, 2), (0, 2)))
                    tmp1 = tensordot(vecr, ten1, axes=((2, 1), (2, 3)))
                    tmp1 = tensordot(tmp1, env1.t, axes=((2, 0), (1, 2)))
                    val_no = vdot(tmp0, tmp1, conj=(0, 0))

                    if isinstance(ten0, DoublePepsTensor):  # 2-layers PEPS
                        ten0 = ten0.apply_gate_on_ket(G0, dirn='l')
                        ten1 = ten1.apply_gate_on_ket(G1, dirn='r')
                    else:  # 1-layer PEPS
                        ten0, ten1 = G0, G1

                    tmp0 = tensordot(ten0, vecl, axes=((0, 1), (2, 1)))
                    tmp0 = tensordot(env0.b, tmp0, axes=((1, 2), (0, 2)))
                    tmp1 = tensordot(vecr, ten1, axes=((2, 1), (2, 3)))
                    tmp1 = tensordot(tmp1, env1.t, axes=((2, 0), (1, 2)))
                    val_op = vdot(tmp0, tmp1, conj=(0, 0))
                else:  # dirn in ('tb', 'bt'):
                    vect = (env0.l @ env0.tl) @ (env0.t @ env0.tr)
                    vecb = (env1.r @ env1.br) @ (env1.b @ env1.bl)

                    tmp0 = tensordot(vect, ten0, axes=((2, 1), (0, 1)))
                    tmp0 = tensordot(tmp0, env0.r, axes=((1, 3), (0, 1)))
                    tmp1 = tensordot(ten1, vecb, axes=((2, 3), (2, 1)))
                    tmp1 = tensordot(env1.l, tmp1, axes=((0, 1), (3, 1)))
                    val_no = vdot(tmp0, tmp1, conj=(0, 0))

                    if isinstance(ten0, DoublePepsTensor):  # 2-layers PEPS
                        ten0 = ten0.apply_gate_on_ket(G0, dirn='t')
                        ten1 = ten1.apply_gate_on_ket(G1, dirn='b')
                    else: # 1-layer PEPS
                        ten0, ten1 = G0, G1

                    tmp0 = tensordot(vect, ten0, axes=((2, 1), (0, 1)))
                    tmp0 = tensordot(tmp0, env0.r, axes=((1, 3), (0, 1)))
                    tmp1 = tensordot(ten1, vecb, axes=((2, 3), (2, 1)))
                    tmp1 = tensordot(env1.l, tmp1, axes=((0, 1), (3, 1)))
                    val_op = vdot(tmp0, tmp1, conj=(0, 0))

                out[bond[0] + nz0, bond[1] + nz1] = val_op / val_no

    return out[bond[0] + nz0, bond[1] + nz1] if return_one else out


def measure_2x2(self, *operators, sites=None) -> float:
    r"""
    Calculate expectation value of a product of local operators
    in a :math:`2 \times 2` window within the CTM environment.
    Perform exact contraction of the window.

    Parameters
    ----------
    operators: Sequence[yastn.Tensor]
        List of local operators to calculate <O0_s0 O1_s1 ...>.

    sites: Sequence[tuple[int, int]]
        A list of sites [s0, s1, ...] matching corresponding operators.
    """
    if sites is None or len(operators) != len(sites):
        raise YastnError("Number of operators and sites should match.")

    # unpack operators if operators provided as a Lattice or dict
    operators = [op[site] if not isinstance(op, Tensor) else op for op, site in zip(operators, sites)]

    sign = sign_canonical_order(*operators, sites=sites, f_ordered=self.f_ordered)
    ops = {}
    for n, op in zip(sites, operators):
        ops[n] = ops[n] @ op if n in ops else op

    minx = min(site[0] for site in sites)  # tl corner
    miny = min(site[1] for site in sites)

    maxx = max(site[0] for site in sites)  # br corner
    maxy = max(site[1] for site in sites)

    if minx == maxx and self.nn_site((minx, miny), 'b') is None:
        minx -= 1  # for a finite system
    if miny == maxy and self.nn_site((minx, miny), 'r') is None:
        miny -= 1  # for a finite system

    tl = Site(minx, miny)
    tr = self.nn_site(tl, 'r')
    br = self.nn_site(tl, 'br')
    bl = self.nn_site(tl, 'b')
    window = [tl, tr, br, bl]

    if any(site not in window for site in sites):
        raise YastnError("Sites do not form a 2x2 window.")

    ten_tl = self.psi[tl]
    ten_tr = self.psi[tr]
    ten_br = self.psi[br]
    ten_bl = self.psi[bl]

    vec_tl = self[tl].l @ (self[tl].tl @ self[tl].t)
    vec_tr = self[tr].t @ (self[tr].tr @ self[tr].r)
    vec_br = self[br].r @ (self[br].br @ self[br].b)
    vec_bl = self[bl].b @ (self[bl].bl @ self[bl].l)

    cor_tl = tensordot(vec_tl, ten_tl, axes=((2, 1), (0, 1)))
    cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))
    cor_tr = tensordot(vec_tr, ten_tr, axes=((1, 2), (0, 3)))
    cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))
    cor_br = tensordot(vec_br, ten_br, axes=((2, 1), (2, 3)))
    cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))
    cor_bl = tensordot(vec_bl, ten_bl, axes=((2, 1), (1, 2)))
    cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))

    val_no = vdot(cor_tl @ cor_tr, tensordot(cor_bl, cor_br, axes=(0, 1)), conj=(0, 0))

    up_tl, up_bl, up_tr, up_br = False, False, False, False
    if isinstance(ten_tl, DoublePepsTensor):
        if tl in ops:
            ten_tl.set_operator_(ops[tl])
        if bl in ops:
            ten_bl.set_operator_(ops[bl])
            ten_bl.add_charge_swaps_(ops[bl].n, axes='k1')
            ten_tl.add_charge_swaps_(ops[bl].n, axes=['b3', 'k4'])
        if tr in ops:
            ten_tr.set_operator_(ops[tr])
            ten_tr.add_charge_swaps_(ops[tr].n, axes='b0')
            ten_tl.add_charge_swaps_(ops[tr].n, axes=['k2', 'k4'])
        if br in ops:
            ten_br.set_operator_(ops[br])
            ten_br.add_charge_swaps_(ops[br].n, axes='k1')
            ten_tr.add_charge_swaps_(ops[br].n, axes=['b3', 'b0', 'k4'])
            ten_tl.add_charge_swaps_(ops[br].n, axes=['k2', 'k4'])
        up_tl = ten_tl.has_operator_or_swap()
        up_bl = ten_bl.has_operator_or_swap()
        up_tr = ten_tr.has_operator_or_swap()
        up_br = ten_br.has_operator_or_swap()
    else:  # single-layer Peps
        if tl in ops:
            ten_tl, up_tl = ops[tl], True
        if bl in ops:
            ten_bl, up_bl = ops[bl], True
        if tr in ops:
            ten_tr, up_tr = ops[tr], True
        if br in ops:
            ten_br, up_br = ops[br], True

    if up_tl:
        cor_tl = tensordot(vec_tl, ten_tl, axes=((2, 1), (0, 1)))
        cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))
    if up_bl:
        cor_bl = tensordot(vec_bl, ten_bl, axes=((2, 1), (1, 2)))
        cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))
    if up_tr:
        cor_tr = tensordot(vec_tr, ten_tr, axes=((1, 2), (0, 3)))
        cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))
    if up_br:
        cor_br = tensordot(vec_br, ten_br, axes=((2, 1), (2, 3)))
        cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))

    val_op = vdot(cor_tl @ cor_tr, tensordot(cor_bl, cor_br, axes=(0, 1)), conj=(0, 0))
    return sign * val_op / val_no


def measure_line(self, *operators, sites=None) -> float:
    r"""
    Calculate expectation value of a product of local opertors
    along a horizontal or vertical line within CTM environment.
    Perform exact contraction of a width-one window.

    Parameters
    ----------
    operators: Sequence[yastn.Tensor]
        List of local operators to calculate <O0_s0 O1_s1 ...>.

    sites: Sequence[tuple[int, int]]
        List of sites that should match operators.
    """
    if sites is None or len(operators) != len(sites):
        raise YastnError("Number of operators and sites should match.")

    # unpack operators if operators provided as a Lattice or dict
    operators = [op[site] if not isinstance(op, Tensor) else op for op, site in zip(operators, sites)]

    sign = sign_canonical_order(*operators, sites=sites, f_ordered=self.f_ordered)
    ops = {}
    for n, op in zip(sites, operators):
        ops[n] = ops[n] @ op if n in ops else op

    xs = sorted(set(site[0] for site in sites))
    ys = sorted(set(site[1] for site in sites))
    if len(xs) > 1 and len(ys) > 1:
        raise YastnError("Sites should form a horizontal or vertical line.")

    env_win = EnvWindow(self, (xs[0], xs[-1] + 1), (ys[0], ys[-1] + 1))
    horizontal = (len(xs) == 1)
    if horizontal:
        vr = env_win[xs[0], 't']
        tm = env_win[xs[0], 'h']
        vl = env_win[xs[0], 'b'].conj()
        axes_op = 'b0'
        axes_string = ('b0', 'k2', 'k4')
    else:  # vertical
        vr = env_win[ys[0], 'l']
        tm = env_win[ys[0], 'v']
        vl = env_win[ys[0], 'r'].conj()
        axes_op = 'k1'
        axes_string = ('k1', 'k4', 'b3')

    val_no = mps.vdot(vl, tm, vr)

    for site, op in ops.items():
        ind = site[0] - xs[0] + site[1] - ys[0] + 1
        if isinstance(tm[ind], DoublePepsTensor):  # 2-layers PEPS
            tm[ind].set_operator_(op)
            tm[ind].add_charge_swaps_(op.n, axes=axes_op)
            for ii in range(1, ind):
                tm[ii].add_charge_swaps_(op.n, axes=axes_string)
        else:  # 1-layer PEPS
            axes = (1, 2, 3, 0) if horizontal else (0, 3, 2, 1)
            tm[ind] = op.transpose(axes=axes)

    val_op = mps.vdot(vl, tm, vr)
    return sign * val_op / val_no


def measure_nsite(self, *operators, sites=None) -> float:
    r"""
    Calculate expectation value of a product of local operators.
    Perform approximate contraction of a windows of PEPS sites
    within CTM environment using boundary MPS.
    The size of the window is taken to include provided sites.

    Parameters
    ----------
    operators: Sequence[yastn.Tensor]
        List of local operators to calculate <O0_s0 O1_s1 ...>.

    sites: Sequence[int]
        A list of sites [s0, s1, ...] matching corresponding operators.
    """
    xrange = (min(site[0] for site in sites), max(site[0] for site in sites) + 1)
    yrange = (min(site[1] for site in sites), max(site[1] for site in sites) + 1)
    env_win = EnvWindow(self, xrange, yrange)
    dirn = 'lr' if (xrange[1] - xrange[0]) >= (yrange[1] - yrange[0]) else 'tb'
    return _measure_nsite(env_win, *operators, sites=sites, dirn=dirn)


def measure_2site(self, O, P, xrange=None, yrange=None, pairs='corner <=', dirn='v', opts_svd=None, opts_var=None) -> dict[Site, float]:
    r"""
    Calculate expectation values :math:`\langle \textrm{O}_i \textrm{P}_j \rangle`
    of local operators :code:`O` and :code:`P` for pairs of lattice sites :math:`i, j`.

    Parameters
    ----------
    O, P: yastn.Tensor
        one-site operators. It is possible to provide a dict of :class:`yastn.tn.fpeps.Lattice` object
        mapping operators to sites.
        For each site, it is possible to provide a list or dict of operators, where the expectation value is calculated
        for each combination of those operators


    xrange: None | tuple[int, int]
        range of rows forming a window, [r0, r1); r0 included, r1 excluded.
        For None, takes a single unit cell of the lattice, which is the default.

    yrange: None | tuple[int, int]
        range of columns forming a window.
        For None, takes a single unit cell of the lattice, which is the default.

    pairs: str | list[tuple[tule[int, int], tuple[int, int]]]
        Limits the pairs of sites to calculate the expectation values.
        If 'corner' in pairs, O is limited to top-left corner of the lattice
        If 'row' in pairs, O is limited to top row of the lattice

    dirn: str
        'h' or 'v', where the boundary MPSs used for truncation are, respectively, horizontal or vertical.
        The default is 'v'.

    opts_svd: dict
        Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces of boundary MPSs used in sampling.
        The default is ``None``, in which case take ``D_total`` as the largest dimension from CTM environment.

    opts_svd: dict
        Options passed to :meth:`yastn.tn.mps.compression_` used in the refining of boundary MPSs.
        The default is ``None``, in which case make 2 variational sweeps.
    """
    if xrange is None:
        xrange = [0, self.Nx]
    if yrange is None:
        yrange = [0, self.Ny]
    env_win = EnvWindow(self, xrange, yrange)
    return _measure_2site(env_win, O, P, xrange, yrange, offset=1, pairs=pairs, dirn=dirn, opts_svd=opts_svd, opts_var=opts_var)


def transfer_matrix_spectrum(env, k=2, n=None, dirn='h', i=0, L=None, dtype='float64'):
    r"""
    Calculate dominant transfer matrix eigenvalues.
    Employs scipy.sparse.linalg.eigs for eigenvalue solver -- as such, works with numpy backend only.

    Parameters
    ----------
    k: int
        number of eigenvalues to recover; The default is 2.

    n: tuple[int] | int | None
        charge of eigenvector. The default None gives zero charge.

    dirn: str
        'h' or 'v'. Vertical of horizontal transfer matrix.

    i: int
        index of row or column from which the transfer matrix is build.

    L: None | int
        The length of transfer matrix.
        The default is None, for which it corresponds to the size of the unit cell.

    dtype: str
        'float64' or 'complex128', dtype used in initializing random starting vector and used in eigensolver.
    """
    if L is None:
        L = env.Nx if dirn == 'v' else env.Ny

    xrange, yrange = ([0, L], [i, i+1]) if dirn == 'v' else ([i, i+1], [0, L])
    env_win = EnvWindow(env, xrange, yrange)
    lvr = 'lvr' if dirn == 'v' else 'thb'
    vr = env_win[i, lvr[0]]
    tm = env_win[i, lvr[1]]
    vl = env_win[i, lvr[2]].conj()
    env_mps = mps.Env(vl, [tm, vr])
    env_mps.update_env_(0, to='last')
    legs = list(env_mps.F[0, 1].get_legs())

    v0 = rand(env.config, legs=legs, n=n, dtype=dtype)

    r1d, meta = split_data_and_meta(v0.to_dict(level=0), squeeze=True)
    def f(x):
        tin = Tensor.from_dict(combine_data_and_meta(x, meta))
        env_mps.F[0, 1] = tin
        for j in range(1, L + 1):
            env_mps.update_env_(j, to='last')
        tout = env_mps.F[L, L+1]
        tout, _ = split_data_and_meta(tout.to_dict(level=0, meta=meta), squeeze=True)
        return tout

    ff = sla.LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=v0.data.dtype)
    eigenvalues, vs1d = sla.eigs(ff, v0=r1d, k=k, which='LM', tol=1e-10)
    return eigenvalues



def sample(env, projectors, number=1, xrange=None, yrange=None, dirn='v', opts_svd=None, opts_var=None, progressbar=False, return_probabilities=False, flatten_one=True, **kwargs) -> dict[Site, list]:
    r"""
    Sample random configurations from PEPS.
    Output a dictionary linking sites with lists of sampled projectors` keys for each site.
    Projectors should be summing up to identity -- this is not checked.

    Parameters
    ----------
    projectors: Dict[Any, yast.Tensor] | Sequence[yast.Tensor] | Dict[Site, Dict[Any, yast.Tensor]]
        Projectors to sample from. We can provide a dict(key: projector), where the sampled results will be given as keys,
        and the same set of projectors is used at each site. For a list of projectors, the keys follow from enumeration.
        Finally, we can provide a dictionary between each site and sets of projectors.

    number: int
        Number of independent samples.

    xrange: None | tuple[int, int]
        range of rows forming a window, [r0, r1); r0 included, r1 excluded.
        For None, takes a single unit cell of the lattice, which is the default.

    yrange: None | tuple[int, int]
        range of columns forming a window.
        For None, takes a single unit cell of the lattice, which is the default.

    dirn: str
        'h' or 'v', where the boundary MPSs used for truncation are, respectively, horizontal or vertical.
        The default is 'v'.

    opts_svd: dict
        Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces of boundary MPSs used in sampling.
        The default is ``None``, in which case take ``D_total`` as the largest dimension from CTM environment.

    opts_var: dict
        Options passed to :meth:`yastn.tn.mps.compression_` used in the refining of boundary MPSs.
        The default is ``None``, in which case make 2 variational sweeps.

    progressbar: bool
        Whether to display progressbar. The default is ``False``.

    return_probabilities: bool
        Whether to return a tuple (samples, probabilities). The default is ``False``, where a dict samples is returned.

    flatten_one: bool
        Whether, for number==1, pop one-element lists for each lattice site to return samples={site: ind, } instead of {site: [ind]}.
        The default is ``True``.
    """
    assert type(env).__name__ in ('EnvCTM', ), "sample only implemented for EnvCTM."
    if xrange is None:
        xrange = [0, env.Nx]
    if yrange is None:
        yrange = [0, env.Ny]
    env_win = EnvWindow(env, xrange, yrange)
    return _sample(env_win, projectors, xrange, yrange, dirn=dirn, offset=1,
                   number=number, opts_svd=opts_svd, opts_var=opts_var,
                   progressbar=progressbar, return_probabilities=return_probabilities, flatten_one=flatten_one)
