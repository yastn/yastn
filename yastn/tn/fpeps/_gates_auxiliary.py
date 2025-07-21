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
from ... import eye, tensordot, qr, YastnError


def match_ancilla(ten, G, swap=False):
    """
    Kronecker product and fusion of local gate with identity for ancilla.

    Identity is read from the ancilla leg of the tensor.
    Can perform a swap gate of the auxiliary operator leg (if present) with an ancilla.
    """
    leg = ten.get_legs(axes=-1)

    if not leg.is_fused():
        return G

    _, leg = leg.unfuse_leg()  # unfuse to get ancilla leg
    one = eye(config=ten.config, legs=[leg, leg.conj()], isdiag=False)
    Gnew = tensordot(G, one, axes=((), ()))

    if G.ndim == 2:
        return Gnew.fuse_legs(axes=((0, 2), (1, 3)))
    elif G.ndim == 3:
        if swap: Gnew = Gnew.swap_gate(axes=(2, 3))
        return Gnew.fuse_legs(axes=((0, 3), (1, 4), 2))
    elif G.ndim == 4:
        if swap: Gnew = Gnew.swap_gate(axes=(3, 4))
        return Gnew.fuse_legs(axes=(0, (1, 4), (2, 5), 3))



def apply_gate_onsite(ten, G, dirn=None):
    """
    Applies operator to the physical leg of (ket) PEPS tensor.

    Operator with auxiliary leg should have dirn in 'l', 'r', 't', 'b', indicating
    the fusion of the auxiliary leg with the corresponding virtual tensor leg and
    application of a proper swap gate.
    For a local operator with no auxiliary index, dirn should be None.
    """
    swap = dirn is not None and dirn in 'tl'
    G = match_ancilla(ten, G, swap=swap)
    tmp = tensordot(ten, G, axes=(2, 1)) # [t l] [b r] [s a] c

    if dirn is None:
        return tmp
    if dirn == 't':
        tmp = tmp.unfuse_legs(axes=1)  # [t l] b r [s a] c
        tmp = tmp.fuse_legs(axes=(0, (1, 4), 2, 3))  # [t l] [b c] r [s a]
        return tmp.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b c] r] [s a]
    if dirn == 'b':
        tmp = tmp.unfuse_legs(axes=0)  # t l [b r] [s a] c
        tmp = tmp.swap_gate(axes=(1, 4))
        tmp = tmp.fuse_legs(axes=((0, 4), 1, 2, 3))  # [t c] l [b r] [s a]
        return tmp.fuse_legs(axes=((0, 1), 2, 3))  # [[t c] l] [b r] [s a]
    if dirn == 'l':
        tmp = tmp.unfuse_legs(axes=1)  # [t l] b r [s a] c
        tmp = tmp.swap_gate(axes=(1, 4))
        tmp = tmp.fuse_legs(axes=(0, 1, (2, 4), 3))  # [t l] b [r c] [s a]
        return tmp.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [b [r c]] [s a]
    if dirn == 'r':
        tmp = tmp.unfuse_legs(axes=0)  # t l [b r] [s a] c
        tmp = tmp.fuse_legs(axes=(0, (1, 4), 2, 3))  # t [l c] [b r] [s a]
        return tmp.fuse_legs(axes=((0, 1), 2, 3))  # [t [l c]] [b r] [s a]
    # raise YastnError("dirn should be equal to 'l', 'r', 't', 'b', or None")


def apply_gate_nn(ten0, ten1, G0, G1, dirn):
    """
    Apply the nearest neighbor gate to a pair of (ket) PEPS tensors.

    The gate should be oriented in accordance with fermionic and lattice orders,
    i.e., here it is assumed we have gate oriented as 'lr' if dirn=='h', and 'tb' if dirn=='v'.
    This gets handled outside of this function.
    """

    G0 = match_ancilla(ten0, G0, swap=True)
    G1 = match_ancilla(ten1, G1, swap=False)

    if dirn == 'h':  # Horizontal gate, "lr" ordered
        tmp0 = tensordot(ten0, G0, axes=(2, 1))  # [t l] [b r] sa c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] sa] [b r] c
        tmp0 = tmp0.unfuse_legs(axes=1)  # [[t l] sa] b r c
        tmp0 = tmp0.swap_gate(axes=(1, 3))  # b X c
        tmp0 = tmp0.fuse_legs(axes=((0, 1), (2, 3)))  # [[[t l] sa] b] [r c]
        Q0f, R0 = qr(tmp0, axes=(0, 1), sQ=-1)  # [[[t l] sa] b] rr @ rr [r c]
        Q0 = Q0f.unfuse_legs(axes=0)  # [[t l] sa] b rr
        Q0 = Q0.fuse_legs(axes=(0, (1, 2)))  # [[t l] sa] [b rr]
        Q0 = Q0.unfuse_legs(axes=0)  # [t l] sa [b rr]
        Q0 = Q0.transpose(axes=(0, 2, 1))  # [t l] [b rr] sa

        tmp1 = tensordot(ten1, G1, axes=(2, 1))  # [t l] [b r] sa c
        tmp1 = tmp1.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        tmp1 = tmp1.unfuse_legs(axes=0)  # t l [[b r] sa] c
        tmp1 = tmp1.fuse_legs(axes=((0, 2), (1, 3)))  # [t [[b r] sa]] [l c]
        Q1f, R1 = qr(tmp1, axes=(0, 1), sQ=1, Qaxis=0, Raxis=-1)  # ll [t [[b r] sa]]  @  [l c] ll
        Q1 = Q1f.unfuse_legs(axes=1)  # ll t [[b r] sa]
        Q1 = Q1.fuse_legs(axes=((1, 0), 2))  # [t ll] [[b r] sa]
        Q1 = Q1.unfuse_legs(axes=1)  # [t ll] [b r] sa
    else: # dirn == 'v':  # Vertical gate, "tb" ordered
        tmp0 = tensordot(ten0, G0, axes=(2, 1))  # [t l] [b r] sa c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] sa] [b r] c
        tmp0 = tmp0.unfuse_legs(axes=1)  # [[t l] sa] b r c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), (1, 3)))  # [[[t l] sa] r] [b c]
        Q0f, R0 = qr(tmp0, axes=(0, 1), sQ=1)  # [[[t l] sa] r] bb  @  bb [b c]
        Q0 = Q0f.unfuse_legs(axes=0)  # [[t l] sa] r bb
        Q0 = Q0.fuse_legs(axes=(0, (2, 1)))  # [[t l] sa] [bb r]
        Q0 = Q0.unfuse_legs(axes=0)  # [t l] sa [bb r]
        Q0 = Q0.transpose(axes=(0, 2, 1))  # [t l] [bb r] sa

        tmp1 = tensordot(ten1, G1, axes=(2, 1))  # [t l] [b r] sa c
        tmp1 = tmp1.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        tmp1 = tmp1.unfuse_legs(axes=0)  # t l [[b r] sa] c
        tmp1 = tmp1.swap_gate(axes=(1, 3))  # l X c
        tmp1 = tmp1.fuse_legs(axes=((1, 2), (0, 3)))  # [l [[b r] sa]] [t c]
        Q1f, R1 = qr(tmp1, axes=(0, 1), sQ=-1, Qaxis=0, Raxis=-1)  # tt [l [[b r] sa]]  @  [t c] tt
        Q1 = Q1f.unfuse_legs(axes=1)  # t l [[b r] sa]
        Q1 = Q1.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] sa]
        Q1 = Q1.unfuse_legs(axes=1)  # [t l] [b r] sa
    return Q0, Q1, R0, R1, Q0f, Q1f

def apply_gate_nnn(ten0, ten1, ten2, G0, G1, G2, dirn, corner):
    """
    Apply the next-nearest-neighbor gate to ket PEPS tensors.

    The gates (G0, G1, G2) acting on the three-site-patch should be in accordance with fermionic and lattice order.
    """

    G0 = match_ancilla(ten0, G0, swap=True)
    G1 = match_ancilla(ten1, G1, swap=True)
    G2 = match_ancilla(ten2, G2, swap=False)

    if dirn == 'br' and corner == "tr":  # diagonal gate, "tl br" ordered, (ten0, ten1, ten2) is at (tl tr br)
        tmp0 = tensordot(ten0, G0, axes=(2, 1))  # [t l] [b r] sa c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] sa] [b r] c
        tmp0 = tmp0.unfuse_legs(axes=1)  # [[t l] sa] b r c
        tmp0 = tmp0.swap_gate(axes=(1, 3))  # b X c
        tmp0 = tmp0.fuse_legs(axes=((0, 1), (2, 3)))  # [[[t l] sa] b] [r c]
        Q0f, R0 = qr(tmp0, axes=(0, 1), sQ=-1)  # [[[t l] sa] b] rr @ rr [r c]
        Q0 = Q0f.unfuse_legs(axes=0)  # [[t l] sa] b rr
        Q0 = Q0.fuse_legs(axes=(0, (1, 2)))  # [[t l] sa] [b rr]
        Q0 = Q0.unfuse_legs(axes=0)  # [t l] sa [b rr]
        Q0 = Q0.transpose(axes=(0, 2, 1))  # [t l] [b rr] sa

        tmp1 = tensordot(ten1, G1, axes=(2, 2)) # [t l] [b r] cl sa cb
        tmp1 = tmp1.unfuse_legs(axes=(0, 1)) # t l b r cl sa cb
        tmp1 = tmp1.swap_gate(axes=(5, (2, 6))) # sa x [b, cb]
        tmp1 = tmp1.fuse_legs(axes=((0, 3, 5), (1, 4, 2, 6))) # [t r sa] [l cl b cb]
        Q1f, R1= qr(tmp1, axes=(0, 1), sQ=1)  # [t r sa] bl  @  bl [l cl b cb]
        R1 = R1.unfuse_legs(axes=1) # bl l cl b cb
        R1 = R1.fuse_legs(axes=(0, (1, 2), (3, 4))) # bl [l cl] [b cb]
        Q1= Q1f.unfuse_legs(axes=0) # t r sa bl
        Q1 = Q1.fuse_legs(axes=((0, 1), 2, 3)) # [t r] sa bl

        tmp2 = tensordot(ten2, G2, axes=(2, 1))  # [t l] [b r] sa c
        tmp2 = tmp2.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        tmp2 = tmp2.unfuse_legs(axes=0)  # t l [[b r] sa] c
        tmp2 = tmp2.swap_gate(axes=(1, 3))  # l X c
        tmp2 = tmp2.fuse_legs(axes=((1, 2), (0, 3)))  # [l [[b r] sa]] [t c]
        Q2f, R2 = qr(tmp2, axes=(0, 1), sQ=-1, Qaxis=0, Raxis=-1)  # tt [l [[b r] sa]]  @  [t c] tt
        Q2 = Q2f.unfuse_legs(axes=1)  # t l [[b r] sa]
        Q2 = Q2.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] sa]
        Q2 = Q2.unfuse_legs(axes=1)  # [t l] [b r] sa

    elif dirn == 'br' and corner == "bl":  # diagonal gate, "tl br" ordered, (ten0, ten1, ten2) is at (tl bl br)
        tmp0 = tensordot(ten0, G0, axes=(2, 1))  # [t l] [b r] sa c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] sa] [b r] c
        tmp0 = tmp0.unfuse_legs(axes=1)  # [[t l] sa] b r c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), (1, 3)))  # [[[t l] sa] r] [b c]
        Q0f, R0 = qr(tmp0, axes=(0, 1), sQ=1)  # [[[t l] sa] r] bb  @  bb [b c]
        Q0 = Q0f.unfuse_legs(axes=0)  # [[t l] sa] r bb
        Q0 = Q0.fuse_legs(axes=(0, (2, 1)))  # [[t l] sa] [bb r]
        Q0 = Q0.unfuse_legs(axes=0)  # [t l] sa [bb r]
        Q0 = Q0.transpose(axes=(0, 2, 1))  # [t l] [bb r] sa

        tmp1 = tensordot(ten1, G1, axes=(2, 2)) # [t l] [b r] ct sa cr
        tmp1 = tmp1.unfuse_legs(axes=(0, 1)) # t l b r ct sa cr
        tmp1 = tmp1.swap_gate(axes=(1, 4, 2, 6)) # l x ct, b x cr
        tmp1 = tmp1.fuse_legs(axes=((1, 2, 5), (0, 4, 3, 6))) # [l b sa] [t ct r cr]
        Q1f, R1= qr(tmp1, axes=(0, 1), sQ=1)  # [l b sa] tr  @  tr [t ct r cr]
        R1 = R1.unfuse_legs(axes=1) # tr t ct r cr
        R1 = R1.fuse_legs(axes=(0, (1, 2), (3, 4))) # tr [t ct] [r cr]
        Q1= Q1f.unfuse_legs(axes=0) # l b sa tr
        Q1 = Q1.fuse_legs(axes=((0, 1), 2, 3)) # [l b] sa tr

        tmp2 = tensordot(ten2, G2, axes=(2, 1))  # [t l] [b r] sa c
        tmp2 = tmp2.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        tmp2 = tmp2.unfuse_legs(axes=0)  # t l [[b r] sa] c
        tmp2 = tmp2.fuse_legs(axes=((0, 2), (1, 3)))  # [t [[b r] sa]] [l c]
        Q2f, R2 = qr(tmp2, axes=(0, 1), sQ=1, Qaxis=0, Raxis=-1)  # ll [t [[b r] sa]]  @  [l c] ll
        Q2 = Q2f.unfuse_legs(axes=1)  # ll t [[b r] sa]
        Q2 = Q2.fuse_legs(axes=((1, 0), 2))  # [t ll] [[b r] sa]
        Q2 = Q2.unfuse_legs(axes=1)  # [t ll] [b r] sa

    elif dirn == 'tr' and corner == "tl":  # anti-diagonal gate, "bl tr" ordered, (ten0, ten1, ten2) is at (tl bl tr)
        tmp0 = tensordot(ten0, G0, axes=(2, 1))  # [t l] [b r] sa cb
        c_leg = G1.get_legs(axes=3)
        I = eye(config=G1.config, legs=(c_leg.conj(), c_leg), isdiag=False)
        tmp0 = tensordot(tmp0, I, axes=((), ())) # [t l] [b r] sa cb cb' cr
        tmp0 = tmp0.unfuse_legs(axes=(0, 1))  # t l b r sa cb cb' cr
        tmp0 = tmp0.fuse_legs(axes=((0, 1, 4), (2, 5, 6, 3, 7))) # [t l sa] [b cb cb' r cr]
        Q0f, R0= qr(tmp0, axes=(0, 1), sQ=1)  # [t l sa] br @  br [b cb cb' r cr]
        R0 = R0.unfuse_legs(axes=1) # br b cb cb' r cr
        R0 = R0.fuse_legs(axes=(0, (1, 2, 3), (4, 5))) # br [b cb cb'] [r cr]
        Q0= Q0f.unfuse_legs(axes=0) # t l sa br
        Q0 = Q0.fuse_legs(axes=((0, 1), 2, 3)) # [t l] sa br

        tmp1 = tensordot(ten1, G1, axes=(2, 2)) # [t l] [b r] ct sa ct'
        tmp1 = tmp1.unfuse_legs(axes=(0, 1)) # t l b r ct sa ct'
        tmp1 = tmp1.swap_gate(axes=(1, 4, (2, 3), 6)) # l x ct, ct'x (b r)
        tmp1 = tmp1.fuse_legs(axes=((1, 2, 3, 5), (0, 4, 6))) # [l b r sa] [t ct ct']
        Q1f, R1 = qr(tmp1, axes=(0, 1), sQ=-1, Qaxis=-1, Raxis=0)  # [l b r sa] tt @  tt [t ct ct']
        Q1 = Q1f.unfuse_legs(axes=0)  # l b r sa tt
        Q1 = Q1.fuse_legs(axes=((4, 0), (1, 2), 3))  # [t l] [b r] sa

        tmp2 = tensordot(ten2, G2, axes=(2, 1))  # [t l] [b r] sa c
        tmp2 = tmp2.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        tmp2 = tmp2.unfuse_legs(axes=0)  # t l [[b r] sa] c
        tmp2 = tmp2.fuse_legs(axes=((0, 2), (1, 3)))  # [t [[b r] sa]] [l c]
        Q2f, R2 = qr(tmp2, axes=(0, 1), sQ=1, Qaxis=0, Raxis=-1)  # ll [t [[b r] sa]]  @  [l c] ll
        Q2 = Q2f.unfuse_legs(axes=1)  # ll t [[b r] sa]
        Q2 = Q2.fuse_legs(axes=((1, 0), 2))  # [t ll] [[b r] sa]
        Q2 = Q2.unfuse_legs(axes=1)  # [t ll] [b r] sa

    elif dirn == 'tr' and corner == "br":  # anti-diagonal gate, "bl tr" ordered, (ten0, ten1, ten2) is at (bl tr br)
        tmp0 = tensordot(ten0, G0, axes=(2, 1))  # [t l] [b r] sa c
        tmp0 = tmp0.unfuse_legs(axes=(0, 1))  # t l b r sa c
        tmp0 = tmp0.swap_gate(axes=((2, 3), 5)) # [b r] x c
        tmp0 = tmp0.fuse_legs(axes=((0, 1, 2, 4), (3, 5)))  # [t l b sa] [r c]
        Q0f, R0 = qr(tmp0, axes=(0, 1), sQ=-1)  # [t l b sa] rr @ rr [r c]
        Q0 = Q0f.unfuse_legs(axes=0)  # t l b sa rr
        Q0 = Q0.fuse_legs(axes=((0, 1), (2, 4), 3))  # [t l] [b r] sa

        tmp1 = tensordot(ten1, G1, axes=(2, 2)) # [t l] [b r] cb sa cb'
        tmp1 = tmp1.swap_gate(axes=(2, 3)) # cb x sa
        tmp1 = tmp1.unfuse_legs(axes=(0, 1)) # t l b r cb sa cb'
        tmp1 = tmp1.fuse_legs(axes=((0, 1, 3, 5), (2, 4, 6))) # [t l r sa] [b cb cb']
        Q1f, R1 = qr(tmp1, axes=(0, 1), sQ=-1, Qaxis=0, Raxis=-1)  # bb [t l r sa] @ [b cb cb'] bb
        Q1 = Q1f.unfuse_legs(axes=1)  # bb t l r sa
        Q1 = Q1.fuse_legs(axes=((1, 2), (0, 3), 4))  # [t r] [bb r] sa

        c_leg = G1.get_legs(axes=0)
        I = eye(config=G1.config, legs=(c_leg.conj(), c_leg), isdiag=False)
        tmp2 = tensordot(ten2, I, axes=((), ())) # [t l] [b r] sa ct cl
        tmp2 = tensordot(tmp2, G2, axes=(2, 1))  # [t l] [b r] ct cl sa ct'
        tmp2 = tmp2.unfuse_legs(axes=(0, 1)) # t l b r ct cl sa ct'
        tmp2 = tmp2.swap_gate(axes=(1, 7)) # l x ct'
        tmp2 = tmp2.fuse_legs(axes=((2, 3, 6), (0, 4, 7, 1, 5))) # [b r sa] [t ct ct' l cl]
        Q2f, R2= qr(tmp2, axes=(0, 1), sQ=1)  # [b r sa] tl @  tl [t ct ct' l cl]
        R2 = R2.unfuse_legs(axes=1) # tl t ct ct' l cl
        R2 = R2.fuse_legs(axes=(0, (4, 5), (1, 2, 3))) # tl [l cl] [t ct ct']
        Q2= Q2f.unfuse_legs(axes=0) # b r sa tl
        Q2 = Q2.fuse_legs(axes=((0, 1), 2, 3)) # [b r] sa tl

    return Q0, Q1, Q2, R0, R1, R2, Q0f, Q1f, Q2f

def apply_bond_tensors(Q0f, Q1f, M0, M1, dirn):
    """
    Combine unitaries in Q0f, Q1f with optimized M0, M1 to form new peps tensors.
    """
    if dirn == "h":
        ten0 = Q0f @ M0  # [[[[t l] sa] b] r
        ten0 = ten0.unfuse_legs(axes=0)  # [[t l] sa] b r
        ten0 = ten0.fuse_legs(axes=(0, (1, 2)))  # [[t l] sa] [b r]
        ten0 = ten0.unfuse_legs(axes=0)  # [t l] sa [b r]
        ten0 = ten0.transpose(axes=(0, 2, 1))  # [t l] [b r] sa

        ten1 = M1 @ Q1f  # l [t [[b r] s]]
        ten1 = ten1.unfuse_legs(axes=1)  # l t [[b r] sa]
        ten1 = ten1.fuse_legs(axes=((1, 0), 2))  # [t l] [[b r] sa]
        ten1 = ten1.unfuse_legs(axes=1)  # [t l] [b r] sa
    else:  # dirn == "v":
        ten0 = Q0f @ M0  # [[[t l] sa] r] b
        ten0 = ten0.unfuse_legs(axes=0)  # [[t l] sa] r b
        ten0 = ten0.fuse_legs(axes=(0, (2, 1)))  # [[t l] sa] [b r]
        ten0 = ten0.unfuse_legs(axes=0)  # [t l] sa [b r]
        ten0 = ten0.transpose(axes=(0, 2, 1))  # [t l] [b r] sa

        ten1 = M1 @ Q1f  # t [l [[b r] sa]]
        ten1 = ten1.unfuse_legs(axes=1)  # t l [[b r] sa]
        ten1 = ten1.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] sa]
        ten1 = ten1.unfuse_legs(axes=1)  # [t l] [b r] sa
    return ten0, ten1

def apply_bond_tensors_nnn(Q0f, Q1f, Q2f, M0, M1, M2, dirn, corner):
    """
    Combine unitaries in Q0f, Q1f, Q2f with optimized M0, M1, M2 to form new peps tensors.
    """
    if dirn == "br":
        if corner == 'tr':
            ten0 = Q0f @ M0  # [[[[t l] sa] b] r
            ten0 = ten0.unfuse_legs(axes=0)  # [[t l] sa] b r
            ten0 = ten0.fuse_legs(axes=(0, (1, 2)))  # [[t l] sa] [b r]
            ten0 = ten0.unfuse_legs(axes=0)  # [t l] sa [b r]
            ten0 = ten0.transpose(axes=(0, 2, 1))  # [t l] [b r] sa

            ten1 = Q1f @ M1 # [t r sa] l b
            ten1 = ten1.unfuse_legs(axes=(0,)) # t r sa l b
            ten1 = ten1.swap_gate(axes=(2, 4)) # sa x b
            ten1 = ten1.fuse_legs(axes=((0, 3), (4, 1), 2)) # [t l] [b r] sa

            ten2 = M2 @ Q2f  # t [l [[b r] sa]]
            ten2 = ten2.unfuse_legs(axes=1)  # t l [[b r] sa]
            ten2 = ten2.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] sa]
            ten2 = ten2.unfuse_legs(axes=1)  # [t l] [b r] sa

        elif corner == 'bl':
            ten0 = Q0f @ M0  # [[[[t l] sa] r] b
            ten0 = ten0.unfuse_legs(axes=0)  # [[t l] sa] r b
            ten0 = ten0.fuse_legs(axes=(0, (2, 1)))  # [[t l] sa] [b r]
            ten0 = ten0.unfuse_legs(axes=0)  # [t l] sa [b r]
            ten0 = ten0.transpose(axes=(0, 2, 1))  # [t l] [b r] sa

            ten1 = Q1f @ M1 # [l b sa] t r
            ten1 = ten1.unfuse_legs(axes=(0,)) # l b sa t r
            ten1 = ten1.fuse_legs(axes=((3, 0), (1, 4), 2)) # [t l] [b r] sa

            ten2 = M2 @ Q2f  # l [t [[b r] sa]]
            ten2 = ten2.unfuse_legs(axes=1)  # l t [[b r] sa]
            ten2 = ten2.fuse_legs(axes=((1, 0), 2))  # [t l] [[b r] sa]
            ten2 = ten2.unfuse_legs(axes=1)  # [t l] [b r] sa
    elif dirn == "tr":
        if corner == 'tl':
            ten0 = Q0f@M0 # [t l sa] b r
            ten0 = ten0.unfuse_legs(axes=0) # t l sa b r
            ten0 = ten0.fuse_legs(axes=((0, 1), (3, 4), 2))  # [t l] [b r] sa


            ten1 = Q1f@M1 # [l b r sa] t
            ten1 = ten1.unfuse_legs(axes=0) # l b r sa t
            ten1 = ten1.fuse_legs(axes=((4, 0), (1, 2), 3)) # [t l] [b r] sa

            ten2 = M2@Q2f # l [t [[b r] sa]]
            ten2 = ten2.unfuse_legs(axes=1) # l t [[b r] sa]
            ten2 = ten2.unfuse_legs(axes=2) # l t [b r] sa
            ten2 = ten2.fuse_legs(axes=((1, 0), 2, 3)) # [t l] [b r] sa
        elif corner == "br":
            ten0 = Q0f@M0 # [t l b sa] r
            ten0 = ten0.unfuse_legs(axes=(0,)) # t l b sa r
            ten0 = ten0.fuse_legs(axes=((0, 1), (2, 4), 3)) # [t l] [b r] sa

            ten1 = M1@Q1f # b [t l r sa]
            ten1 = ten1.unfuse_legs(axes=1) # b t l r sa
            ten1 = ten1.fuse_legs(axes=((1, 2), (0, 3), 4)) # [t l] [b r] sa

            ten2 = Q2f@M2 # [b r sa] l t
            ten2 = ten2.unfuse_legs(axes=0) # b r sa l t
            ten2 = ten2.fuse_legs(axes=((4, 3), (0, 1), 2)) # [t l] [b r] sa

    return ten0, ten1, ten2

def gate_product_operator(O0, O1, l_ordered=True, f_ordered=True, merge=False):
    """
    Takes two ndim=2 local operators O0 O1 to be applied
    on two sites with O1 acting first (relevant for fermionic operators).
    Adds a connecting leg with a swap_gate consistent with fermionic order.
    Orders output to match lattice order (canonical l_order is 'lr' and 'tb').

    If merge, returns equivalent of ncon([O0, O1], [(-0, -2), (-1, -3)]),
    with proper operator order and swap gate applied.
    """
    s = -1 if l_ordered else 1
    O0 = O0.add_leg(s=s, axis=2)
    O1 = O1.add_leg(s=-s, axis=2)

    if f_ordered:
        O0 = O0.swap_gate(axes=(1, 2))
    else:
        O1 = O1.swap_gate(axes=(0, 2))

    G0, G1 = (O0, O1) if l_ordered else (O1, O0)

    if merge:
        return tensordot(G0, G1, axes=(2, 2)).transpose(axes=(0, 2, 1, 3))
    return G0, G1


def fkron(A, B, sites=(0, 1), merge=True):
    """
    Fermionic kron; auxiliary function for gate generation.
    Returns a Kronecker product of two local operators, A and B,
    including swap-gate (fermionic string) to handle fermionic operators.

    Calculate A_0 B_1 for sites==(0, 1), and A_1 B_0 for sites==(1, 0),
    i.e., operator B acts first (relevant when operators are fermionically non-trivial).

    Site 0 is assumed to be first in both lattice and fermionic orders.

    If merge, returns equivalent of ncon([O0, O1], [(-0, -2), (-1, -3)]),
    with proper operator order and swap gate applied;
    O0 corresponds to the operator acting on site 0.
    """
    if sites not in ((0, 1), (1, 0)):
        raise YastnError("sites should be equal to (0, 1) or (1, 0)")
    order = (sites == (0, 1))
    return gate_product_operator(A, B, order, order, merge)


def gate_fix_order(G0, G1, l_ordered=True, f_ordered=True):
    """
    Modifies two gate tensors,
    that were generated consistently with lattice and fermionic orders,
    making them consistent with provided orders.

    l_ordered and f_ordered typically coincide;
    they do not coincide in a special case of
    cylindric lattice geometry across the periodic boundary.
    """
    if not f_ordered:
        G0 = G0.swap_gate(axes=(1, 2))
        G1 = G1.swap_gate(axes=(0, 2))
    if not l_ordered:
        G0, G1 = G1, G0
    return G0, G1
