from ... import eye, tensordot


def match_ancilla(ten, G, swap=False):
    """ kron and fusion of local gate with identity for ancilla. Identity is read from ancila of A. """
    leg = ten.get_legs(axes=-1)

    if not leg.is_fused():
        return G

    _, leg = leg.unfuse_leg()  # unfuse to get ancilla leg
    one = eye(config=ten.config, legs=[leg, leg.conj()], isdiag=False)
    Gnew = tensordot(G, one, axes=((), ()))

    if G.ndim == 2:
        return Gnew.fuse_legs(axes=((0, 2), (1, 3)))
    # else gate.ndim == 3:
    if swap:
        Gnew = Gnew.swap_gate(axes=(2, 3))
    return Gnew.fuse_legs(axes=((0, 3), (1, 4), 2))


def apply_gate(ten, G, dirn=None):
    """
    Prepare top and bottom peps tensors for CTM procedures.
    Applies operators on top if provided, with dir = 'l', 'r', 't', 'b'.
    If dirn is None, no auxiliary indices are introduced as the operator is local.
    Spin and ancilla legs of tensors are always fused.
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
    raise RuntimeError("dirn should be equal to 'l', 'r', 't', 'b', or None")


def gate_product_operator(O0, O1, l_ordered=True, f_ordered=True, merge=False):
    """
    Takes two ndim=2 local operators O0 O1, with O1 acting first (relevant for fermionic operators).
    Adds a connecting leg with a swap_gate consistent with fermionic order.
    Orders output to match lattice order.

    If merge, returns equivalnt of ncon([O0, O1], [(-0, -2), (-1, -3)]),
    with proper operator order and swap-gate applied.
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


def gate_fix_order(G0, G1, l_ordered=True, f_ordered=True):
    """
    Modifies two gate tensors, that were generated consitent with lattice and fermionic orders,
    to make them consistent with provided ordere.
    """
    if not f_ordered:
        G0 = G0.swap_gate(axes=(0, 2))
        G1 = G1.swap_gate(axes=(1, 2))
    if not l_ordered:
        G0, G1 = G1, G0
    return G0, G1
