# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" Common measure functions for EnvCTM and EnvBoundaryMPS """

import warnings
import scipy.sparse.linalg as sla

from ._env_window import EnvWindow, _measure_2site, _measure_nsite, _sample
from .._gates_auxiliary import gate_fix_swap_gate, clear_operator_input
from .._doublePepsTensor import DoublePepsTensor
from .._geometry import Site, is_bond, is_site
from ... import mps
from ....initialize import rand
from ....tensor import YastnError, Tensor, tensordot, vdot, split_data_and_meta, combine_data_and_meta, sign_canonical_order
from ....tensor.oe_blocksparse import contract_with_unroll
from ._env_ctm_oe_measure_network import (_translate_unroll, _build_ketbra_contracted, _build_ketbra_separate,
                                          _window_bounds, _charge_strings, _mpo_bond_swaps, _build_fused)


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

        if isinstance(ten, DoublePepsTensor):
            ten.del_operator_()
            ten.del_charge_swaps_()

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
                    O = O.add_leg(s=1, axis=2)
                    P = P.add_leg(s=-1, axis=2)
                    O = O.swap_gate(axes=(1, 2))

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

    if isinstance(self.psi[tl], DoublePepsTensor):
        for s in (tl, tr, bl, br):
            self.psi[s].del_operator_()
            self.psi[s].del_charge_swaps_()

    return sign * val_op / val_no

def measure_nsite_exact(self, *operators, sites=None) -> float:
    r"""
    Calculate expectation value of a product of local operators
    in a :math:`Nx \times Ny` window (determined by sites) within the CTM environment.
    Perform exact contraction of the window.
    If Nx <= Ny, contract from left to right. Otherwise, contract from top to bottom.

    Note: use with caution for large windows, as the computational cost grows exponentially with the window size.

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

    Nx, Ny = maxx - minx + 1, maxy - miny + 1

    # four corners of the patch
    tl = Site(minx, miny)
    tr = Site(minx, maxy)
    br = Site(maxx, maxy)
    bl = Site(maxx, miny)
    window = [Site(x, y) for x in range(minx, maxx+1) for y in range(miny, maxy+1)]
    tens = {site: self.psi[site] for site in window}

    def _contract_patch_horz(tens):
        # Form the left boundary
        bdy_left = self[tl].tl
        site = tl
        for i, x in enumerate(range(minx, maxx+1)):
            l = self[site].l
            bdy_left = tensordot(bdy_left, l, axes=(i, 2))
            site = self.nn_site(site, (1, 0))
        bdy_left = tensordot(bdy_left, self[bl].bl, axes=(Nx, 1))
        #   |----------|---- 0
        #   |          |---- 1
        #   | bdy_left |---- ...
        #   |          |---- Nx
        #   |----------|---- Nx+1

        # Form the right boundary
        bdy_right = self[tr].tr
        site = tr
        for i, x in enumerate(range(minx, maxx+1)):
            r = self[site].r
            bdy_right = tensordot(bdy_right, r, axes=(i+1, 0))
            site = self.nn_site(site, (1, 0))
        bdy_right = tensordot(bdy_right, self[br].br, axes=(Nx+1, 0))
        #    0  ----|-----------|
        #    1  ----|           |
        #    ...----| bdy_right |
        #    Nx ----|           |
        #   Nx+1----|-----------|

        # Contract from left to right
        for y in range(miny, maxy+1):
            t = self[Site(minx, y)].t
            bdy_left = t.tensordot(bdy_left, axes=(0, 0))
            for i, x in enumerate(range(minx, maxx+1)):
                bdy_left = tensordot(tens[Site(x, y)], bdy_left, axes=((0,1), (0,i+2)))
            bdy_left = tensordot(self[Site(maxx,y)].b, bdy_left, axes=((1, 2), (0, Nx+2)))
            bdy_left = bdy_left.transpose(axes=tuple(range(Nx+1, -1, -1)))

        return vdot(bdy_left, bdy_right, conj=(0, 0))

    def _contract_patch_vert(tens):
        # Form the top boundary
        bdy_top = self[tl].tl
        site = tl
        for i, y in enumerate(range(miny, maxy+1)):
            t = self[site].t
            bdy_top = tensordot(bdy_top, t, axes=(i+1, 0))
            site = self.nn_site(site, (0, 1))
        bdy_top = tensordot(bdy_top, self[tr].tr, axes=(Ny+1, 0))
        #   |-----------|
        #   | bdy_top   |
        #   |-----------|
        #   |  |  |     |
        #   0  1 ...  Ny+1

        # Form the bottom boundary
        bdy_bottom = self[bl].bl
        site = bl
        for i, y in enumerate(range(miny, maxy+1)):
            b = self[site].b
            bdy_bottom = tensordot(bdy_bottom, b, axes=(i, 2))
            site = self.nn_site(site, (0, 1))
        bdy_bottom = tensordot(bdy_bottom, self[br].br, axes=(Ny, 1))
        #   0  1 ...  Ny+1
        #   |  |  |     |
        #   |-----------|
        #   | bdy_bot   |
        #   |-----------|

        # Contract from top to bottom
        for x in range(minx, maxx+1):
            l = self[Site(x, miny)].l
            bdy_top = l.tensordot(bdy_top, axes=(2, 0))
            for i, y in enumerate(range(miny, maxy+1)):
                bdy_top = tensordot(tens[Site(x, y)], bdy_top, axes=((0,1), (i+2,1)))
            bdy_top = tensordot(self[Site(x,maxy)].r, bdy_top, axes=((0,1), (Ny+2,1)))
            bdy_top = bdy_top.transpose(axes=tuple(range(Ny+1, -1, -1)))
        return vdot(bdy_top, bdy_bottom, conj=(0, 0))

    contract_fn = _contract_patch_horz if Nx <= Ny else _contract_patch_vert
    val_no = contract_fn(tens)

    # Insert operators
    axes_string_x = ['b3', 'k4', 'k1']
    axes_string_y = ['k2', 'k4', 'b0']
    if isinstance(tens[tl], DoublePepsTensor):
        for y in range(miny, maxy+1):
            for x in range(minx, maxx+1):
                site = Site(x, y)
                if site in ops:
                    tens[site].set_operator_(ops[site])
                    if x > minx:
                        tens[site].add_charge_swaps_(ops[site].n, axes='k1')
                        for x1 in range(x-1, minx, -1):
                            tens[Site(x1, y)].add_charge_swaps_(ops[site].n, axes=axes_string_x)
                        tens[Site(minx, y)].add_charge_swaps_(ops[site].n, axes=['b3', 'k4'])

                    if y > miny:
                        tens[Site(minx, y)].add_charge_swaps_(ops[site].n, axes='b0')
                        for y1 in range(y-1, miny, -1):
                            tens[Site(minx, y1)].add_charge_swaps_(ops[site].n, axes=axes_string_y)
                        tens[Site(minx, miny)].add_charge_swaps_(ops[site].n, axes=['k2', 'k4'])

    else:  # single-layer Peps
        for y in range(miny, maxy+1):
            for x in range(minx, maxx+1):
                site = Site(x, y)
                if site in ops:
                    tens[site] = ops[site]

    val_op = contract_fn(tens)

    if isinstance(tens[tl], DoublePepsTensor):
        for s in window:
            tens[s].del_operator_()
            tens[s].del_charge_swaps_()

    return sign * val_op / val_no

def measure_line(self, *operators, sites=None) -> float:
    r"""
    Calculate expectation value of a product of local operators
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

    for ind in range(1, len(tm) - 1):
        if isinstance(tm[ind], DoublePepsTensor):
            tm[ind].del_operator_()
            tm[ind].del_charge_swaps_()

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

    pairs: str | list[tuple[tuple[int, int], tuple[int, int]]]
        Limits the pairs of sites to calculate the expectation values.
        If 'corner' in pairs, O is limited to top-left corner of the lattice
        If 'row' in pairs, O is limited to top row of the lattice

    dirn: str
        'h' or 'v', where the boundary MPSs used for truncation are, respectively, horizontal or vertical.
        The default is 'v'.

    opts_svd: dict
        Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces of boundary MPSs used in sampling.
        The default is ``None``, in which case take ``D_total`` as the largest dimension from CTM environment.

    opts_var: dict
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
    if type(env).__name__ not in ('EnvCTM', ):
        raise YastnError("sample only implemented for EnvCTM.")
    if xrange is None:
        xrange = [0, env.Nx]
    if yrange is None:
        yrange = [0, env.Ny]
    env_win = EnvWindow(env, xrange, yrange)
    return _sample(env_win, projectors, xrange, yrange, dirn=dirn, offset=1,
                   number=number, opts_svd=opts_svd, opts_var=opts_var,
                   progressbar=progressbar, return_probabilities=return_probabilities, flatten_one=flatten_one)


def _parse_operators(env, operators, sites):
    """Sort the operators of a measurement into plain operators and MPO tensors.

    Returns ``(ops, bonds, sign)``: ``ops`` maps a site to its operator (plain
    operators on the same site are multiplied), ``bonds`` maps a site to the
    network labels of its MPO tensor's bond legs, ``('opb', k)`` between MPO
    tensors ``k-1`` and ``k`` of the chain (= listing order), and ``sign`` is
    the sign of bringing plain operators into the lattice's fermionic order.
    A measurement uses one kind or the other.  MPO tensors must be listed in
    fermionic order and get ``sign = 1``: they are neutral, and the caller has
    folded the reordering sign of every term into its coefficient.
    """
    if sites is None or len(operators) != len(sites):
        raise YastnError("Number of operators and sites should match.")
    # unpack operators if operators provided as a Lattice or dict
    operators = [op[site] if not isinstance(op, Tensor) else op for op, site in zip(operators, sites)]
    ops, bonds = {}, {}
    nop = len(operators)
    for k, (site, op) in enumerate(zip(sites, operators)):
        if op.ndim > 2:
            labels = (('opb', k),) * (k > 0) + (('opb', k + 1),) * (k < nop - 1)
            if len(labels) != op.ndim - 2:
                raise YastnError(f"operator {k} of {nop} carries {op.ndim - 2} bond legs, "
                                 f"but its position in the chain allows {len(labels)}.")
            bonds[site] = labels
        elif site in ops:
            op = ops[site] @ op
        ops[site] = op
    if bonds and (len(bonds) < nop or len(ops) < nop):
        raise YastnError("MPO tensors: one per site and no plain operators alongside.")
    if bonds and not all(env.f_ordered(s0, s1) for s0, s1 in zip(sites, sites[1:])):
        raise YastnError("MPO tensors must be listed in the lattice's fermionic order of their sites.")
    sign = 1 if bonds else sign_canonical_order(*operators, sites=sites, f_ordered=env.f_ordered)
    return ops, bonds, sign


def _contract_window(self, ops, bonds, sites, unroll=None, separate_layers=True, projectors=None, probe=None,
                     optimizer="default", per_combo_path=False, combo_path_kwargs=None, **kwargs):
    r"""Contract the window of ``sites`` once, with ``ops, bonds`` from :func:`_parse_operators`.

    Empty ``ops`` gives the norm, otherwise the numerator without the
    reordering sign.  Returns a number, or with ``probe=(site, slot, tensor)``
    the open cut map.  ``kwargs`` (``checkpoint_loop``, ``devices``,
    ``mp_workers_per_device``) go to ``contract_with_unroll``.  Bond labels and
    fermionic signs are described in ``docs/source/fpeps/measurement_oe.rst``.
    """
    minx, miny, maxx, maxy = _window_bounds(self, sites)
    Nx, Ny = maxx - minx + 1, maxy - miny + 1
    tl, tr, br, bl = Site(minx, miny), Site(minx, maxy), Site(maxx, maxy), Site(maxx, miny)
    geom = (Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br)
    tens = {Site(x, y): self.psi[Site(x, y)] for x in range(minx, maxx + 1) for y in range(miny, maxy + 1)}
    if unroll and not ops:  # the norm network carries no operator bond legs
        unroll = {k: v for k, v in unroll.items() if not (isinstance(k, tuple) and k[:1] == ('opb',))} or None

    if not isinstance(tens[tl], DoublePepsTensor):
        # single-layer PEPS (psi[site] is already a fused 4-leg double-layer tensor):
        # each 'operator' must be a replacement site tensor of the same kind, i.e.
        # ket, operator and bra contracted and fused by the caller.  No fermionic
        # strings can be applied here; same convention as measure_1site / measure_nsite_exact.
        if bonds or probe:
            raise YastnError("MPO tensors and cut maps require a DoublePepsTensor PEPS.")
        tens.update(ops)
        tn, swap = _build_fused(self, tens, *geom), None
    else:
        if projectors is not None and not separate_layers:
            raise YastnError("projectors-based compression requires separate_layers=True.")
        if bonds and not separate_layers:
            # The contracted builder absorbs the operator into the site tensor, so the
            # bare ket leg the MPO bonds cross is no longer a network leg.
            warnings.warn("MPO tensors need the operator kept as a separate network tensor; "
                          "using separate_layers=True.", stacklevel=3)
            separate_layers = True
        # fresh shells: the operators and strings attached below never touch self.psi
        tens = {s: DoublePepsTensor(bra=t.bra, ket=t.ket, trans=t.trans) for s, t in tens.items()}
        crossings = ()
        if bonds:
            crossings = _mpo_bond_swaps(tens, ops, bonds, minx, miny)
        else:
            _charge_strings(tens, ops, minx, miny)
        if separate_layers:
            tn, swap = _build_ketbra_separate(self, tens, *geom, projectors=projectors, op_bonds=bonds,
                                              bond_crossings=crossings, probe=probe)
        else:
            tn, swap = _build_ketbra_contracted(self, tens, *geom)
        unroll = _translate_unroll(unroll, Nx, Ny)

    if per_combo_path and unroll:  # tunes the path per slice-combo; only meaningful with an unroll
        kwargs.update(per_combo_path=True,
                      combo_path_kwargs={"optimizer": optimizer} if combo_path_kwargs is None else combo_path_kwargs)
    out = contract_with_unroll(*tn, unroll=unroll, swap=swap, optimizer=optimizer, **kwargs)
    return out if probe else out.to_number()


def measure_nsite_exact_oe(self, *operators, sites=None, unroll=None, checkpoint_loop=False, separate_layers=True, optimizer="default", devices=None, mp_workers_per_device=0, projectors=None, per_combo_path=False, combo_path_kwargs=None) -> float:
    r"""
    Memory-efficient version of :meth:`measure_nsite_exact` using opt_einsum
    contraction path optimization, optional block-sparse index unrolling,
    and checkpointing.

    For ``DoublePepsTensor`` PEPS, ket, operator and bra of every site enter
    the network as separate tensors (``separate_layers=True``, the default),
    with the fermionic crossings between them as ``ncon`` swap pairs; edge
    middle legs are unfused to match.  With ``separate_layers=False`` ket and
    bra are pre-contracted on the physical leg into 8-leg site tensors first,
    which is possible for plain two-leg operators only.

    For single-layer PEPS, falls back to the fused double-layer approach.

    Returns ``<psi| O0_s0 ... |psi> / <psi|psi>``.  See
    :func:`measure_nsite_norm_exact_oe` for the norm-only contraction and
    :func:`measure_nsite_numerator_exact_oe` for the unnormalized numerator
    -- callers that share a single norm across multiple numerator
    evaluations should use those split functions to control the autograd
    graph lifetime explicitly.

    Parameters
    ----------
    operators : Sequence[yastn.Tensor]
        List of local operators to calculate <O0_s0 O1_s1 ...>.

    sites : Sequence[tuple[int, int]]
        A list of sites [s0, s1, ...] matching corresponding operators.

    unroll : dict or None
        Dict mapping bond labels to ``int`` (uniform slice size) or
        ``list[SlicedLeg]``.  See :ref:`oe-bond-labels` for the bond-label scheme.

    checkpoint_loop : bool
        If ``True`` and ``unroll`` is not ``None``, each unroll iteration
        is wrapped in :func:`torch.utils.checkpoint.checkpoint`, trading
        recomputation for lower peak memory.

    separate_layers : bool
        If ``True`` (default) and the PEPS uses ``DoublePepsTensor``, keep
        ket, operator and bra as separate tensors in the ncon network; this
        is required for MPO tensors and gives the path optimizer the most
        freedom.  ``False`` pre-contracts ket and bra into 8-leg site tensors
        (plain operators only).

    optimizer : str or opt_einsum.paths.PathOptimizer
        Contraction-path optimizer passed to :func:`opt_einsum.contract_path`.
        ``"default"`` (also ``None``, ``"dp"``, ``"dynamic-programming"``) uses
        ``opt_einsum.DynamicProgramming(minimize="write", search_outer=False,
        cost_cap=True)``, which minimizes the size of the intermediates.  Any
        other value accepted by opt_einsum, e.g. ``"greedy"`` or ``"auto"``, is
        passed through unchanged.

    devices : Sequence[str] or None
        Devices to spread the contraction over, e.g. ``["cuda:0", "cuda:1"]``.
        ``None`` (default) contracts on the device of the PEPS tensors.  With
        ``unroll``, the slice combinations are dispatched across the devices,
        which needs ``mp_workers_per_device >= 1``; a single device with one
        worker contracts serially there and moves the result back.  Without
        ``unroll``, the network is moved to ``devices[0]`` and contracted there.

    mp_workers_per_device : int
        Number of worker processes per device in the multiprocess pool that contracts the
        slice combinations.  ``0`` (default) disables multiprocessing and
        requires ``devices`` to be ``None`` or the PEPS device.

    projectors : dict or None
        CTM half-projectors to insert into the window, which makes the
        measurement approximate but cheaper.  Maps a lattice site to one slot
        name or a tuple of slot names of :class:`EnvCTM_projectors`
        (``"hlt"``, ``"hlb"``, ``"hrt"``, ``"hrb"``, ``"vtl"``, ``"vtr"``,
        ``"vbl"``, ``"vbr"``), e.g. ``{site: ("hrt", "hrb")}``, which reads the
        tensors from ``env.proj[site]``, or to a ``{slot: tensor}`` dict, which
        supplies them directly (a ``None`` tensor reads that slot from
        ``env.proj``).  Each half must come with its partner:
        the slot with the last letter flipped (``t`` with ``b``, ``l`` with
        ``r``) on the neighbouring site in that letter's direction.  A pair
        compresses the two parallel bonds of its cut into one thin bond, as a
        CTM move does.  Requires ``separate_layers=True`` and a
        ``DoublePepsTensor`` PEPS.  ``None`` (default) inserts nothing.

    per_combo_path : bool
        Only used with ``unroll``.  If ``True``, search a separate contraction
        path for every slice combination, tuned to its slice dimensions and
        cached by shape, instead of reusing one path for all combinations.
        Default ``False``.

    combo_path_kwargs : dict or None
        Options of the per-combination path search when
        ``per_combo_path=True``; keys may include ``optimizer``,
        ``memory_limit``, ``names`` and ``who``.  ``None`` (default) means
        ``{"optimizer": optimizer}``.
    """
    ops, bonds, sign = _parse_operators(self, operators, sites)
    kw = dict(unroll=unroll, checkpoint_loop=checkpoint_loop, separate_layers=separate_layers,
              optimizer=optimizer, devices=devices, mp_workers_per_device=mp_workers_per_device,
              projectors=projectors, per_combo_path=per_combo_path, combo_path_kwargs=combo_path_kwargs)
    val_no = _contract_window(self, {}, {}, sites, **kw)
    return sign * _contract_window(self, ops, bonds, sites, **kw) / val_no


def measure_nsite_norm_exact_oe(self, *, sites, unroll=None, checkpoint_loop=False, separate_layers=True, optimizer="default", devices=None, mp_workers_per_device=0, projectors=None, per_combo_path=False, combo_path_kwargs=None):
    """Contract only the norm <psi|psi> over the bounding window of ``sites``.

    Same contraction backend and options as :func:`measure_nsite_exact_oe`,
    with ``operators`` omitted; operator-bond labels ``('opb', k)`` in
    ``unroll`` are ignored.  Use this when sharing a single norm value across
    multiple numerator evaluations (see :func:`measure_nsite_numerator_exact_oe`).
    """
    return _contract_window(self, {}, {}, sites, unroll=unroll, checkpoint_loop=checkpoint_loop,
                            separate_layers=separate_layers, optimizer=optimizer, devices=devices,
                            mp_workers_per_device=mp_workers_per_device, projectors=projectors,
                            per_combo_path=per_combo_path, combo_path_kwargs=combo_path_kwargs)


def measure_nsite_numerator_exact_oe(self, *operators, sites, unroll=None, checkpoint_loop=False, separate_layers=True, optimizer="default", devices=None, mp_workers_per_device=0, projectors=None, per_combo_path=False, combo_path_kwargs=None):
    """Contract only the unnormalized numerator ``sign * <psi| O0_s0 ... |psi>``.

    Same contraction backend and options as :func:`measure_nsite_exact_oe`;
    the result is *not* divided by the norm.  The caller is responsible for
    dividing by ``<psi|psi>`` (typically obtained via
    :func:`measure_nsite_norm_exact_oe`).
    """
    ops, bonds, sign = _parse_operators(self, operators, sites)
    return sign * _contract_window(self, ops, bonds, sites, unroll=unroll, checkpoint_loop=checkpoint_loop,
                                   separate_layers=separate_layers, optimizer=optimizer, devices=devices,
                                   mp_workers_per_device=mp_workers_per_device, projectors=projectors,
                                   per_combo_path=per_combo_path, combo_path_kwargs=combo_path_kwargs)


def measure_nsite_cut_map_oe(self, *operators, sites, probe_site, probe_slot, probe,
                             projectors=None, unroll=None,
                             checkpoint_loop=False, optimizer="default", devices=None,
                             mp_workers_per_device=0, per_combo_path=False,
                             combo_path_kwargs=None):
    r"""
    Contract a measurement window with a probe tensor closing one side of an
    interior cut and the partner side's legs open, returning the cut map
    ``Y = M . Omega`` as a 4-leg tensor.

    With ``operators`` empty this contracts the norm window; with operators
    given it contracts the numerator window (Jordan-Wigner strings or MPO
    bond crossings included, same as :meth:`measure_nsite_numerator_exact_oe`).  The
    ``probe`` tensor, in the stored 3-leg projector form (env chi, fused
    ket-D x bra-D, thin), is inserted at ``(probe_site, probe_slot)`` as one
    half-projector.  Its partner is not inserted: the severed bonds on the
    partner side, i.e. the env bond and the ket and bra D bonds, plus the
    probe's thin label are left open and form the 4 output legs of the
    returned tensor.  Other interior cuts are compressed by ``projectors``.

    Always uses the separate-layers build; the PEPS must use
    ``DoublePepsTensor``.

    Parameters
    ----------
    operators : Sequence[yastn.Tensor]
        Local operators, one per site: plain two-leg operators or MPO tensors,
        under the same rules as :meth:`measure_nsite_numerator_exact_oe`.
        Empty contracts the norm window.
    sites : Sequence[tuple[int, int]]
        Sites of the window, matching ``operators`` when given.
    probe_site, probe_slot
        Lattice site and slot name (``"hrt"``, ``"hrb"``, ``"hlt"``,
        ``"hlb"``, ...) of the inserted probe.
    probe : yastn.Tensor
        Probe tensor in stored 3-leg projector form.
    projectors : dict or None
        Projectors for the *other* cuts, in either form accepted by
        :meth:`measure_nsite_exact_oe`; partner pairs are enforced as usual.
    unroll, checkpoint_loop, optimizer, devices, mp_workers_per_device, per_combo_path, combo_path_kwargs
        As in :meth:`measure_nsite_exact_oe`.

    Returns
    -------
    yastn.Tensor
        The 4-leg cut map ``Y``, with leg order (env chi, ket D, bra D, thin).
    """
    ops, bonds, sign = _parse_operators(self, operators, sites) if operators else ({}, {}, 1)
    return sign * _contract_window(self, ops, bonds, sites, probe=(probe_site, probe_slot, probe),
                                   projectors=projectors, unroll=unroll, checkpoint_loop=checkpoint_loop,
                                   optimizer=optimizer, devices=devices,
                                   mp_workers_per_device=mp_workers_per_device,
                                   per_combo_path=per_combo_path, combo_path_kwargs=combo_path_kwargs)


def _eval_projectors(env, move, opts_svd):
    """Construct the projectors using the converged env.

    ``opts_svd`` carries the truncation options (notably ``D_total``);
    it is passed through to ``_update_projectors_`` unchanged.
    """
    for site in env.sites():
        env._update_projectors_(site, move, opts_svd, method='2x2')
