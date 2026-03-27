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
""" Common measure functions for EnvCTM and EnvBoundaryMPS """

import scipy.sparse.linalg as sla

from ._env_window import EnvWindow, _measure_2site, _measure_nsite, _sample
from .._gates_auxiliary import gate_fix_swap_gate, clear_operator_input
from .._doublePepsTensor import DoublePepsTensor
from .._geometry import Site, is_bond, is_site
from ... import mps
from ....initialize import rand
from ....tensor import YastnError, Tensor, tensordot, vdot, split_data_and_meta, combine_data_and_meta, sign_canonical_order
from ....tensor.oe_blocksparse import get_contraction_path, contract_with_unroll_compute_constants


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


def _translate_unroll(unroll, Nx, Ny):
    """Map user-facing fused bond labels to unfused ket/bra sub-labels.

    Interior PEPS bonds (``('v', i, j)`` with ``0 <= j < Ny`` and
    ``('h', i, j)`` with ``0 <= i < Nx``) are split into ket/bra
    sub-labels.  Boundary (chi) bonds are kept as-is.
    """
    if unroll is None:
        return None
    translated = {}
    for label, val in unroll.items():
        if (label[0] == 'v' and 0 <= label[2] < Ny) or \
           (label[0] == 'h' and 0 <= label[1] < Nx):
            translated[label + ('k',)] = val
            translated[label + ('b',)] = val
        else:
            translated[label] = val
    return translated


def _pad_unfused_edge(edge_uf, peps_ket_leg, peps_bra_leg, ket_ax=1, bra_ax=2):
    r"""
    Pad an unfused edge tensor with zero blocks so that its ket/bra
    sub-legs match the PEPS ket/bra legs.

    After CTM expansion with OBC boundary projectors, unfused edge
    sub-legs can have fewer charge sectors than the PEPS legs.
    Adding zero blocks for the missing sectors restores compatibility
    without affecting the contraction result (zero blocks contribute
    nothing).
    """
    ket_sub = edge_uf.get_legs(axes=ket_ax)
    bra_sub = edge_uf.get_legs(axes=bra_ax)

    existing_ket_t = set(ket_sub.t)
    missing_ket_t = set(peps_ket_leg.t) - existing_ket_t

    existing_bra_t = set(bra_sub.t)
    missing_bra_t = set(peps_bra_leg.t) - existing_bra_t

    if not missing_ket_t and not missing_bra_t:
        return edge_uf

    legs = edge_uf.get_legs()
    sigs = edge_uf.s
    n_total = edge_uf.n
    ndim = edge_uf.ndim_n
    other_axes = [ax for ax in range(ndim) if ax != ket_ax and ax != bra_ax]

    peps_ket_tD = dict(zip(peps_ket_leg.t, peps_ket_leg.D))
    peps_bra_tD = dict(zip(peps_bra_leg.t, peps_bra_leg.D))

    # Collect existing chi charge pairs
    nsym = edge_uf.config.sym.NSYM
    chi_pairs = set()
    for block_t in edge_uf.struct.t:
        charges = tuple(
            tuple(block_t[ax * nsym:(ax + 1) * nsym]) for ax in range(ndim)
        )
        chi_pairs.add(tuple(charges[ax] for ax in other_axes))

    existing_blocks = set(edge_uf.struct.t)

    def _infer_missing_charge(chi_combo, known_charge, known_ax, unknown_ax):
        """Infer the charge on unknown_ax from the symmetry constraint."""
        mb_list = []
        for s in range(nsym):
            partial = sum(sigs[ax] * chi_combo[idx][s]
                          for idx, ax in enumerate(other_axes))
            partial += sigs[known_ax] * known_charge[s]
            remaining = n_total[s] - partial
            if sigs[unknown_ax] == 0:
                return None
            mb_list.append(remaining // sigs[unknown_ax])
        return tuple(mb_list)

    def _add_zero_block(chi_combo, mk, mb):
        """Add a zero block for (chi_combo, mk, mb) if it doesn't exist."""
        D_k = peps_ket_tD[mk]
        D_b = peps_bra_tD[mb]

        ts_list = [None] * ndim
        Ds_list = [None] * ndim
        ts_list[ket_ax] = mk
        ts_list[bra_ax] = mb
        Ds_list[ket_ax] = D_k
        Ds_list[bra_ax] = D_b
        for idx, ax in enumerate(other_axes):
            ts_list[ax] = chi_combo[idx]
            Ds_list[ax] = legs[ax].D[list(legs[ax].t).index(chi_combo[idx])]

        ts_flat = sum(ts_list, ())
        if ts_flat not in existing_blocks:
            edge_uf.set_block(ts=ts_flat, Ds=tuple(Ds_list), val='zeros')
            existing_blocks.add(ts_flat)

    for chi_combo in chi_pairs:
        for mk in missing_ket_t:
            mb = _infer_missing_charge(chi_combo, mk, ket_ax, bra_ax)
            if mb is None or mb not in peps_bra_tD:
                continue
            _add_zero_block(chi_combo, mk, mb)

        for mb in missing_bra_t:
            mk = _infer_missing_charge(chi_combo, mb, bra_ax, ket_ax)
            if mk is None or mk not in peps_ket_tD:
                continue
            _add_zero_block(chi_combo, mk, mb)

    return edge_uf


def _uf_middle_padded(edge_tensor, peps_ket_leg, peps_bra_leg):
    """Unfuse edge middle leg and pad to match PEPS ket/bra legs."""
    uf = edge_tensor.unfuse_legs(axes=(1,))
    uf = uf.drop_leg_history(axes=(1, 2))
    return _pad_unfused_edge(uf, peps_ket_leg, peps_bra_leg)


def _build_interleaved_unfused(env, tens, Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br):
    r"""
    Assemble the full patch tensor network in interleaved format
    without ``fuse_layers()``.

    Each ``DoublePepsTensor`` site is contracted on the physical leg
    to produce an 8-leg tensor (4 ket + 4 bra PEPS legs) with
    fermionic crossings applied as ``swap_gate``.  This avoids
    ``fuse_layers()`` which additionally fuses the ket/bra sub-legs.

    Edge middle legs (PEPS-facing, fused as ``[ket, bra]``) are
    unfused and padded to match the 8-leg site tensors.  Corner legs
    and edge outer legs (chi bonds) are kept as-is.

    Returns ``(tn_args, swap_pairs)`` where *swap_pairs* contains
    same-tensor fermionic crossing pairs for ncon.
    """

    args = []
    swap_pairs = []

    # --- Pre-contract ket/bra into 8-leg tensors with swap gates ---
    site_tensors = {}
    peps_legs = {}
    for i in range(Nx):
        for j in range(Ny):
            s = Site(minx + i, miny + j)
            dpt = tens[s]
            Ab, Ak = dpt.Ab_Ak_with_charge_swap()

            if dpt.op is not None:
                Ak = tensordot(Ak, dpt.op, axes=(4, 1))

            Ab_c = Ab.conj()

            # Contract on physical leg → 8-leg tensor
            # Ak: (t_k, l_k, b_k, r_k, p), Ab_c: (t_b, l_b, b_b, r_b, p)
            # Result: (t_k, l_k, b_k, r_k, t_b, l_b, b_b, r_b)
            tt = tensordot(Ak, Ab_c, axes=(4, 4))

            # Apply fermionic crossings (same as fuse_layers)
            # swap (l_k, l_b) × t_b and (b_k, b_b) × r_b
            tt = tt.swap_gate(axes=((1, 5), 4, (2, 6), 7))  # (l_k, l_b) × t_b, (b_k, b_b) × r_b

            # Transpose to interleave: t_k, t_b, l_k, l_b, b_k, b_b, r_k, r_b
            tt = tt.transpose(axes=(0, 4, 1, 5, 2, 6, 3, 7))

            # Apply DoublePepsTensor transpose (permutes the 4 PEPS directions)
            trans8 = []
            for k in dpt.trans:
                trans8.extend([2 * k, 2 * k + 1])
            tt = tt.transpose(axes=tuple(trans8)).drop_leg_history()

            site_tensors[(i, j)] = tt
            peps_legs[(i, j)] = (
                tuple(tt.get_legs(axes=2 * ax) for ax in range(4)),      # ket
                tuple(tt.get_legs(axes=2 * ax + 1) for ax in range(4)),  # bra
            )

    # --- Corners ---
    args += [env[tl].tl, [('v', 0, -1), ('h', -1, -1)]]
    args += [env[bl].bl, [('h', Nx, -1), ('v', Nx, -1)]]
    args += [env[tr].tr, [('h', -1, Ny - 1), ('v', 0, Ny)]]
    args += [env[br].br, [('v', Nx, Ny), ('h', Nx, Ny - 1)]]

    # --- Left edges ---
    for i in range(Nx):
        k_legs, b_legs = peps_legs[(i, 0)]
        args += [_uf_middle_padded(env[Site(minx + i, miny)].l, k_legs[1], b_legs[1]),
                 [('v', i + 1, -1), ('h', i, -1, 'k'), ('h', i, -1, 'b'), ('v', i, -1)]]

    # --- Right edges ---
    for i in range(Nx):
        k_legs, b_legs = peps_legs[(i, Ny - 1)]
        args += [_uf_middle_padded(env[Site(minx + i, maxy)].r, k_legs[3], b_legs[3]),
                 [('v', i, Ny), ('h', i, Ny - 1, 'k'), ('h', i, Ny - 1, 'b'), ('v', i + 1, Ny)]]

    # --- Top edges ---
    for j in range(Ny):
        k_legs, b_legs = peps_legs[(0, j)]
        args += [_uf_middle_padded(env[Site(minx, miny + j)].t, k_legs[0], b_legs[0]),
                 [('h', -1, j - 1), ('v', 0, j, 'k'), ('v', 0, j, 'b'), ('h', -1, j)]]

    # --- Bottom edges ---
    for j in range(Ny):
        k_legs, b_legs = peps_legs[(Nx - 1, j)]
        args += [_uf_middle_padded(env[Site(maxx, miny + j)].b, k_legs[2], b_legs[2]),
                 [('h', Nx, j), ('v', Nx, j, 'k'), ('v', Nx, j, 'b'), ('h', Nx, j - 1)]]

    # --- PEPS sites: 8-leg tensors [t_k, t_b, l_k, l_b, b_k, b_b, r_k, r_b] ---
    def _bond_labels(i, j):
        return [('v', i, j), ('h', i, j - 1), ('v', i + 1, j), ('h', i, j)]

    for i in range(Nx):
        for j in range(Ny):
            lbls = _bond_labels(i, j)
            args += [site_tensors[(i, j)],
                     [lbls[0] + ('k',), lbls[0] + ('b',),
                      lbls[1] + ('k',), lbls[1] + ('b',),
                      lbls[2] + ('k',), lbls[2] + ('b',),
                      lbls[3] + ('k',), lbls[3] + ('b',)]]

    args.append(())  # scalar output
    return tuple(args), swap_pairs


def _build_separate_unfused(env, tens, Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br):
    r"""
    Assemble the full patch tensor network with separate ket and bra
    tensors (5-leg each) per site.

    Unlike ``_build_interleaved_unfused`` which pre-contracts ket and bra
    on the physical leg into 8-leg tensors, this keeps them separate and
    connects them via a shared physical-leg label.  Fermionic crossings
    are split into intra-bra ``swap_gate`` calls (applied before adding
    to the network) and inter-tensor swap pairs (returned for ncon).

    Edge tensors, corners, and boundary bonds are identical to
    ``_build_interleaved_unfused``.

    Returns ``(tn_args, swap_pairs)`` for use with ncon / opt_einsum.
    """

    args = []
    swap_pairs = []

    # --- Collect peps_legs for edge padding (same as interleaved path) ---
    peps_legs = {}
    for i in range(Nx):
        for j in range(Ny):
            s = Site(minx + i, miny + j)
            dpt = tens[s]
            Ab, Ak = dpt.Ab_Ak_with_charge_swap()

            if dpt.op is not None:
                Ak_tmp = tensordot(Ak, dpt.op, axes=(4, 1))
            else:
                Ak_tmp = Ak

            Ab_c_tmp = Ab.conj()

            # Apply intra-bra swap gates (canonical order)
            Ab_c_tmp = Ab_c_tmp.swap_gate(axes=(0, 1, 2, 3))  # l_b × t_b, b_b × r_b

            # Transpose to position order
            Ak_t = Ak_tmp.transpose(axes=dpt.trans + (4,)).drop_leg_history()
            Ab_c_t = Ab_c_tmp.transpose(axes=dpt.trans + (4,)).drop_leg_history()

            peps_legs[(i, j)] = (
                tuple(Ak_t.get_legs(axes=ax) for ax in range(4)),    # ket
                tuple(Ab_c_t.get_legs(axes=ax) for ax in range(4)),  # bra
            )

    # --- Corners (identical to interleaved) ---
    args += [env[tl].tl, [('v', 0, -1), ('h', -1, -1)]]
    args += [env[bl].bl, [('h', Nx, -1), ('v', Nx, -1)]]
    args += [env[tr].tr, [('h', -1, Ny - 1), ('v', 0, Ny)]]
    args += [env[br].br, [('v', Nx, Ny), ('h', Nx, Ny - 1)]]

    # --- Left edges ---
    for i in range(Nx):
        k_legs, b_legs = peps_legs[(i, 0)]
        args += [_uf_middle_padded(env[Site(minx + i, miny)].l, k_legs[1], b_legs[1]),
                 [('v', i + 1, -1), ('h', i, -1, 'k'), ('h', i, -1, 'b'), ('v', i, -1)]]

    # --- Right edges ---
    for i in range(Nx):
        k_legs, b_legs = peps_legs[(i, Ny - 1)]
        args += [_uf_middle_padded(env[Site(minx + i, maxy)].r, k_legs[3], b_legs[3]),
                 [('v', i, Ny), ('h', i, Ny - 1, 'k'), ('h', i, Ny - 1, 'b'), ('v', i + 1, Ny)]]

    # --- Top edges ---
    for j in range(Ny):
        k_legs, b_legs = peps_legs[(0, j)]
        args += [_uf_middle_padded(env[Site(minx, miny + j)].t, k_legs[0], b_legs[0]),
                 [('h', -1, j - 1), ('v', 0, j, 'k'), ('v', 0, j, 'b'), ('h', -1, j)]]

    # --- Bottom edges ---
    for j in range(Ny):
        k_legs, b_legs = peps_legs[(Nx - 1, j)]
        args += [_uf_middle_padded(env[Site(maxx, miny + j)].b, k_legs[2], b_legs[2]),
                 [('h', Nx, j), ('v', Nx, j, 'k'), ('v', Nx, j, 'b'), ('h', Nx, j - 1)]]

    # --- PEPS sites: separate ket (5-leg) and bra (5-leg) ---
    def _bond_labels(i, j):
        return [('v', i, j), ('h', i, j - 1), ('v', i + 1, j), ('h', i, j)]

    for i in range(Nx):
        for j in range(Ny):
            s = Site(minx + i, miny + j)
            dpt = tens[s]
            Ab, Ak = dpt.Ab_Ak_with_charge_swap()

            if dpt.op is not None:
                Ak = tensordot(Ak, dpt.op, axes=(4, 1))

            Ab_c = Ab.conj()

            # Apply intra-bra fermionic crossings (canonical order)
            Ab_c = Ab_c.swap_gate(axes=(1, 0, 2, 3))  # l_b × t_b, b_b × r_b

            # Transpose to position order
            Ak = Ak.transpose(axes=dpt.trans + (4,)).drop_leg_history()
            Ab_c = Ab_c.transpose(axes=dpt.trans + (4,)).drop_leg_history()

            lbls = _bond_labels(i, j)

            # Ket tensor: 4 bond legs + physical
            args += [Ak,
                     [lbls[0] + ('k',), lbls[1] + ('k',),
                      lbls[2] + ('k',), lbls[3] + ('k',), ('p', i, j)]]

            # Bra tensor: 4 bond legs + physical (shared physical label)
            args += [Ab_c,
                     [lbls[0] + ('b',), lbls[1] + ('b',),
                      lbls[2] + ('b',), lbls[3] + ('b',), ('p', i, j)]]

            # Inter-tensor swap pairs (ket × bra fermionic crossings)
            # In canonical order: l_k × t_b and b_k × r_b
            # inv[d] = position of canonical direction d after transpose
            inv = [dpt.trans.index(d) for d in range(4)]
            swap_pairs.append((lbls[inv[1]] + ('k',), lbls[inv[0]] + ('b',)))  # l_k × t_b
            swap_pairs.append((lbls[inv[2]] + ('k',), lbls[inv[3]] + ('b',)))  # b_k × r_b

    args.append(())  # scalar output
    return tuple(args), swap_pairs


def measure_nsite_exact_oe(self, *operators, sites=None, unroll=None, checkpoint_loop=False, separate_layers=False) -> float:
    r"""
    Memory-efficient version of :meth:`measure_nsite_exact` using opt_einsum
    contraction path optimization, optional block-sparse index unrolling,
    and checkpointing.

    For ``DoublePepsTensor`` PEPS, ket and bra are pre-contracted on
    the physical leg with fermionic crossings applied via
    ``swap_gate``, producing 8-leg site tensors whose ket/bra
    sub-legs are kept separate (no ``fuse_legs``).  Edge middle legs
    are unfused to match.

    For single-layer PEPS, falls back to the fused double-layer approach.

    Parameters
    ----------
    operators : Sequence[yastn.Tensor]
        List of local operators to calculate <O0_s0 O1_s1 ...>.

    sites : Sequence[tuple[int, int]]
        A list of sites [s0, s1, ...] matching corresponding operators.

    unroll : dict or None
        Dict mapping bond labels to ``int`` (uniform slice size) or
        ``list[SlicedLeg]``.  Slicing a bond partitions its charge sectors
        so the full contraction is split into a loop of smaller partial
        contractions, reducing peak memory.

        Bond labels are tuples that identify edges in the tensor network.
        The window has ``Nx`` rows (indexed ``i = 0 … Nx-1``) and ``Ny``
        columns (indexed ``j = 0 … Ny-1``).  Row/column indices refer to
        positions *within the window*, not absolute lattice coordinates.

        All bond labels use ``(row, col)`` ordering, consistent with
        the yastn site convention ``(x, y) = (row, col)``.

        **Horizontal bonds** ``('h', i, j)`` — run left-to-right between
        columns ``j`` and ``j+1`` at row ``i``::

                   j=-1             j=0               j=1          j=Ny-1    j=Ny
                    :               :                 :             :         :
            i=-1   TL --h,-1,-1-- T[0] --h,-1,0-- T[1] -- ... -- h,-1,Ny-1 -- TR
                    |               |               |               |         |
                 v,0,-1             v,0,0          v,0,1          v,0,Ny-1   v,0,Ny
                    |               |               |               |         |
            i=0    L[0]-h,0,-1------*---h,0,0-------*--- ... --h,0,Ny-1------R[0]
                    |               |               |               |         |
                 v,1,-1             v,1,0          v,1,1          v,1,Ny-1  v,1,Ny
                    |               |               |               |         |
            i=1    L[1]-h,1,-1------*---h,1,0-------*--- ... --h,1,Ny-1------R[1]
                    :               :                :              :         :
                    |               |                |              |         |
                 v,Nx,-1             v,Nx,0          v,Nx,1       v,Nx,Ny-1  v,Nx,Ny
                    |               |                |              |         |
            i=Nx   BL --h,Nx,-1-- B[0] --h,Nx,0-- B[1] -- ... -- h,Nx,Ny-1 -- BR

        where ``*`` marks a PEPS site, ``TL/TR/BL/BR`` are CTM corners,
        ``T/B/L/R`` are CTM edges.

        ``i = -1`` and ``i = Nx`` are boundary rows (chi bonds between
        edge tensors and corners).  ``i = 0 … Nx-1`` are PEPS rows
        (physical bonds).  ``j = -1`` is the left-boundary column.

        **Vertical bonds** ``('v', i, j)`` — run top-to-bottom in column ``j``
        between rows ``i-1`` and ``i``.

        Conventions:
            ``i = 0`` connects the top row (``T[j]`` or corner) to the first
            PEPS row; ``i = Nx`` connects the last PEPS row to the bottom row
            (``B[j]`` or corner); ``i = 1 … Nx-1`` are interior vertical bonds.
            ``j = -1`` is the left boundary column; ``j = Ny`` is the right
            boundary column; ``j = 0 … Ny-1`` are PEPS columns.

        For ``DoublePepsTensor`` PEPS, PEPS-row bonds (``('h', i, j)``
        with ``0 <= i < Nx`` and all ``('v', i, j)``) are automatically
        split into ket/bra sub-labels by ``_translate_unroll``.
        Boundary (chi) bonds are kept as-is.

        **Example** — 2×3 window (``Nx=2, Ny=3``)::

            # unroll the horizontal bond between columns 0 and 1
            # at the first PEPS row, one charge sector at a time:
            unroll = {('h', 0, 0): 1}

            # unroll the vertical bond in column 1 between rows 0 and 1:
            unroll = {('v', 1, 1): 1}

            # unroll multiple bonds simultaneously:
            unroll = {('h', 0, 0): 1, ('v', 1, 1): 1}

            # unroll a left-boundary vertical (chi) bond:
            unroll = {('v', 0, -1): 1}

    checkpoint_loop : bool
        If ``True`` and *unroll* is not ``None``, each unroll iteration is
        wrapped in :func:`torch.utils.checkpoint.checkpoint`, trading
        recomputation for lower peak memory.

    separate_layers : bool
        If ``True`` and the PEPS uses ``DoublePepsTensor`` (double-layer),
        keep ket and bra as separate 5-leg tensors in the ncon network
        instead of pre-contracting them into 8-leg tensors.  This doubles
        the tensor count but gives ``opt_einsum`` more freedom to optimize
        the contraction order.  Default is ``False`` (pre-contracted path).
    """
    if sites is None or len(operators) != len(sites):
        raise YastnError("Number of operators and sites should match.")

    # unpack operators if operators provided as a Lattice or dict
    operators = [op[site] if not isinstance(op, Tensor) else op
                 for op, site in zip(operators, sites)]

    sign = sign_canonical_order(*operators, sites=sites, f_ordered=self.f_ordered)
    ops = {}
    for n, op in zip(sites, operators):
        ops[n] = ops[n] @ op if n in ops else op

    minx = min(site[0] for site in sites)
    miny = min(site[1] for site in sites)
    maxx = max(site[0] for site in sites)
    maxy = max(site[1] for site in sites)

    if minx == maxx and self.nn_site((minx, miny), 'b') is None:
        minx -= 1
    if miny == maxy and self.nn_site((minx, miny), 'r') is None:
        miny -= 1

    Nx = maxx - minx + 1
    Ny = maxy - miny + 1

    tl = Site(minx, miny)
    tr = Site(minx, maxy)
    br = Site(maxx, maxy)
    bl = Site(maxx, miny)
    window = [Site(x, y) for x in range(minx, maxx + 1)
                         for y in range(miny, maxy + 1)]
    tens = {site: self.psi[site] for site in window}

    is_double_layer = isinstance(tens[tl], DoublePepsTensor)

    if is_double_layer:
        # --- Unfused path for double-layer PEPS ---
        path_opts = {"optimizer": "default"}
        translated_unroll = _translate_unroll(unroll, Nx, Ny)
        build_fn = _build_separate_unfused if separate_layers else _build_interleaved_unfused

        # norm contraction (no operators)
        tn_no, swap_no = build_fn(
            self, tens, Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br)
        path_no, _ = get_contraction_path(*tn_no, unroll=translated_unroll, **path_opts)
        val_no = contract_with_unroll_compute_constants(
            *tn_no, optimize=path_no, unroll=translated_unroll,
            checkpoint_loop=checkpoint_loop, swap=swap_no, **path_opts).to_number()

        # insert operators and charge swaps (in-place on DoublePepsTensor)
        axes_string_x = ['b3', 'k4', 'k1']
        axes_string_y = ['k2', 'k4', 'b0']
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                site = Site(x, y)
                if site in ops:
                    tens[site].set_operator_(ops[site])
                    if x > minx:
                        tens[site].add_charge_swaps_(ops[site].n, axes='k1')
                        for x1 in range(x - 1, minx, -1):
                            tens[Site(x1, y)].add_charge_swaps_(
                                ops[site].n, axes=axes_string_x)
                        tens[Site(minx, y)].add_charge_swaps_(
                            ops[site].n, axes=['b3', 'k4'])
                    if y > miny:
                        tens[Site(minx, y)].add_charge_swaps_(
                            ops[site].n, axes='b0')
                        for y1 in range(y - 1, miny, -1):
                            tens[Site(minx, y1)].add_charge_swaps_(
                                ops[site].n, axes=axes_string_y)
                        tens[Site(minx, miny)].add_charge_swaps_(
                            ops[site].n, axes=['k2', 'k4'])

        # operator contraction
        tn_op, swap_op = build_fn(
            self, tens, Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br)
        path_op, _ = get_contraction_path(*tn_op, unroll=translated_unroll, **path_opts)
        val_op = contract_with_unroll_compute_constants(
            *tn_op, optimize=path_op, unroll=translated_unroll,
            checkpoint_loop=checkpoint_loop, swap=swap_op, **path_opts).to_number()

        for s in window:
            tens[s].del_operator_()
            tens[s].del_charge_swaps_()

    else:
        # --- Fused path for single-layer PEPS ---
        def _drop(t):
            return t.drop_leg_history() if hasattr(t, 'drop_leg_history') else t

        def _build_interleaved_fused(realized):
            args = []
            args += [_drop(self[tl].tl), [('v', 0, -1), ('h', -1, -1)]]
            args += [_drop(self[bl].bl), [('h', Nx, -1), ('v', Nx, -1)]]
            args += [_drop(self[tr].tr), [('h', -1, Ny - 1), ('v', 0, Ny)]]
            args += [_drop(self[br].br), [('v', Nx, Ny), ('h', Nx, Ny - 1)]]
            for i in range(Nx):
                args += [_drop(self[Site(minx + i, miny)].l),
                         [('v', i + 1, -1), ('h', i, -1), ('v', i, -1)]]
            for i in range(Nx):
                args += [_drop(self[Site(minx + i, maxy)].r),
                         [('v', i, Ny), ('h', i, Ny - 1), ('v', i + 1, Ny)]]
            for j in range(Ny):
                args += [_drop(self[Site(minx, miny + j)].t),
                         [('h', -1, j - 1), ('v', 0, j), ('h', -1, j)]]
            for j in range(Ny):
                args += [_drop(self[Site(maxx, miny + j)].b),
                         [('h', Nx, j), ('v', Nx, j), ('h', Nx, j - 1)]]
            for i in range(Nx):
                for j in range(Ny):
                    s = Site(minx + i, miny + j)
                    args += [realized[s],
                             [('v', i, j), ('h', i, j - 1),
                              ('v', i + 1, j), ('h', i, j)]]
            args.append(())
            return tuple(args)

        realized_no = {s: _drop(t) for s, t in tens.items()}
        tn_no = _build_interleaved_fused(realized_no)
        path_no, _ = get_contraction_path(*tn_no, unroll=unroll)
        val_no = contract_with_unroll_compute_constants(
            *tn_no, optimize=path_no, unroll=unroll,
            checkpoint_loop=checkpoint_loop).to_number()

        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                site = Site(x, y)
                if site in ops:
                    tens[site] = ops[site]

        realized_op = {s: _drop(t) for s, t in tens.items()}
        tn_op = _build_interleaved_fused(realized_op)
        path_op, _ = get_contraction_path(*tn_op, unroll=unroll)
        val_op = contract_with_unroll_compute_constants(
            *tn_op, optimize=path_op, unroll=unroll,
            checkpoint_loop=checkpoint_loop).to_number()

    return sign * val_op / val_no
