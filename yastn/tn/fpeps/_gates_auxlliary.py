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
    # else G.ndim == 3:
    if swap:
        Gnew = Gnew.swap_gate(axes=(2, 3))
    return Gnew.fuse_legs(axes=((0, 3), (1, 4), 2))


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
        tmp = tmp.unfuse_legs(axes=1) # [t l] b r [s a] c
        tmp = tmp.fuse_legs(axes=(0, (1, 4), 2, 3)) # [t l] [b c] r [s a]
        return tmp.fuse_legs(axes=(0, (1, 2), 3)) # [t l] [[b c] r] [s a]
    if dirn == 'b':
        tmp = tmp.unfuse_legs(axes=0) # t l [b r] [s a] c
        tmp = tmp.swap_gate(axes=(1, 4))
        tmp = tmp.fuse_legs(axes=((0, 4), 1, 2, 3)) # [t c] l [b r] [s a]
        return tmp.fuse_legs(axes=((0, 1), 2, 3)) # [[t c] l] [b r] [s a]
    if dirn == 'l':
        tmp = tmp.unfuse_legs(axes=1) # [t l] b r [s a] c
        tmp = tmp.swap_gate(axes=(1, 4))
        tmp = tmp.fuse_legs(axes=(0, 1, (2, 4), 3)) # [t l] b [r c] [s a]
        return tmp.fuse_legs(axes=(0, (1, 2), 3)) # [t l] [b [r c]] [s a]
    if dirn == 'r':
        tmp = tmp.unfuse_legs(axes=0) # t l [b r] [s a] c
        tmp = tmp.fuse_legs(axes=(0, (1, 4), 2, 3)) # t [l c] [b r] [s a]
        return tmp.fuse_legs(axes=((0, 1), 2, 3)) # [t [l c]] [b r] [s a]
    raise YastnError("dirn should be equal to 'l', 'r', 't', 'b', or None")


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
        tmp0 = tensordot(ten0, G0, axes=(2, 1)) # [t l] [b r] sa c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] sa] [b r] c
        tmp0 = tmp0.unfuse_legs(axes=1)  # [[t l] sa] b r c
        tmp0 = tmp0.swap_gate(axes=(1, 3))  # b X c
        tmp0 = tmp0.fuse_legs(axes=((0, 1), (2, 3)))  # [[[t l] sa] b] [r c]
        Q0f, R0 = qr(tmp0, axes=(0, 1), sQ=-1)  # [[[t l] sa] b] rr @ rr [r c]
        Q0 = Q0f.unfuse_legs(axes=0)  # [[t l] sa] b rr
        Q0 = Q0.fuse_legs(axes=(0, (1, 2)))  # [[t l] sa] [b rr]
        Q0 = Q0.unfuse_legs(axes=0)  # [t l] sa [b rr]
        Q0 = Q0.transpose(axes=(0, 2, 1))  # [t l] [b rr] sa

        tmp1 = tensordot(ten1, G1, axes=(2, 1)) # [t l] [b r] sa c
        tmp1 = tmp1.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        tmp1 = tmp1.unfuse_legs(axes=0)  # t l [[b r] sa] c
        tmp1 = tmp1.fuse_legs(axes=((0, 2), (1, 3)))  # [t [[b r] sa]] [l c]
        Q1f, R1 = qr(tmp1, axes=(0, 1), sQ=1, Qaxis=0, Raxis=-1)  # ll [t [[b r] sa]]  @  [l c] ll
        Q1 = Q1f.unfuse_legs(axes=1)  # ll t [[b r] sa]
        Q1 = Q1.fuse_legs(axes=((1, 0), 2))  # [t ll] [[b r] sa]
        Q1 = Q1.unfuse_legs(axes=1)  # [t ll] [b r] sa
    else: # dirn == 'v':  # Vertical gate, "tb" ordered
        tmp0 = tensordot(ten0, G0, axes=(2, 1)) # [t l] [b r] sa c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] sa] [b r] c
        tmp0 = tmp0.unfuse_legs(axes=1)  # [[t l] sa] b r c
        tmp0 = tmp0.fuse_legs(axes=((0, 2), (1, 3)))  # [[[t l] sa] r] [b c]
        Q0f, R0 = qr(tmp0, axes=(0, 1), sQ=1)  # [[[t l] sa] r] bb  @  bb [b c]
        Q0 = Q0f.unfuse_legs(axes=0)  # [[t l] sa] r bb
        Q0 = Q0.fuse_legs(axes=(0, (2, 1)))  # [[t l] sa] [bb r]
        Q0 = Q0.unfuse_legs(axes=0)  # [t l] sa [bb r]
        Q0 = Q0.transpose(axes=(0, 2, 1))  # [t l] [bb r] sa

        tmp1 = tensordot(ten1, G1, axes=(2, 1)) # [t l] [b r] sa c
        tmp1 = tmp1.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        tmp1 = tmp1.unfuse_legs(axes=0)  # t l [[b r] sa] c
        tmp1 = tmp1.swap_gate(axes=(1, 3))  # l X c
        tmp1 = tmp1.fuse_legs(axes=((1, 2), (0, 3)))  # [l [[b r] sa]] [t c]
        Q1f, R1 = qr(tmp1, axes=(0, 1), sQ=-1, Qaxis=0, Raxis=-1)  # tt [l [[b r] sa]]  @  [t c] tt
        Q1 = Q1f.unfuse_legs(axes=1)  # t l [[b r] sa]
        Q1 = Q1.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] sa]
        Q1 = Q1.unfuse_legs(axes=1)  # [t l] [b r] sa
    return Q0, Q1, R0, R1, Q0f, Q1f


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


def gate_product_operator(O0, O1, l_ordered=True, f_ordered=True, merge=False):
    """
    Takes two ndim=2 local operators O0 O1 to be applied
    on two sites with O1 acting first (relevant for fermionic operators).
    Adds a connecting leg with a swap_gate consistent with fermionic order.
    Orders output to match lattice order (canonical l_order is 'lr' and 'tb').

    If merge, returns equivalent of ncon([O0, O1], [(-0, -2), (-1, -3)]),
    with proper operator order and swap gate applied.
    """
    G0 = O0.add_leg(s=1)
    G1 = O1.add_leg(s=-1)
    if f_ordered:
        G0 = G0.swap_gate(axes=(0, 2))
    else:
        G1 = G1.swap_gate(axes=(1, 2))

    if not l_ordered:
        G0, G1 = G1, G0

    if merge:
        return tensordot(G0, G1, axes=(2, 2)).transpose(axes=(0, 2, 1, 3))
    return G0, G1


def twosite_operator(A, B, sites=(0, 1), merge=True):
    """
    Auxiliary function for gate generation,
    returning a Kronecker product of two local operators, A and B.

    Calculate A_0 B_1 for sites == (0, 1), and A_1 B_0 for sites = (1, 0),
    i.e., operator B acts first (relevant for fermionic operators).

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
        G0 = G0.swap_gate(axes=(0, 2))
        G1 = G1.swap_gate(axes=(1, 2))
    if not l_ordered:
        G0, G1 = G1, G0
    return G0, G1
