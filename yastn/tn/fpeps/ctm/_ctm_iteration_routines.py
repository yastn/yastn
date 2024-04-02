
import yastn
from yastn import tensordot, ncon
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps._doublePepsTensor import DoublePepsTensor


def fPEPS_l(A, op):
    """
    attaches operator to the tensor located at the left (left according to
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    Aop = tensordot(A, op, axes=(4, 1)) # t l b r [s a] c
    Aop = Aop.swap_gate(axes=(2, 5))
    Aop = Aop.fuse_legs(axes=(0, 1, 2, (3, 5), 4)) # t l b [r c] [s a]
    return Aop


def fPEPS_r(A, op):
    """
    attaches operator to the tensor located at the right (right according to
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.fuse_legs(axes=(0, (1, 5), 2, 3, 4)) # t [l c] b r [s a]
    return Aop


def fPEPS_t(A, op):
    """
    attaches operator to the tensor located at the top (left according to
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.fuse_legs(axes=(0, 1, (2, 5), 3, 4)) # t l [b c] r [s a]
    return Aop


def fPEPS_b(A, op):
    """
    attaches operator to the tensor located at the bottom (right according to
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.swap_gate(axes=(1, 5))
    Aop = Aop.fuse_legs(axes=((0, 5), 1, 2, 3, 4)) # [t c] l b r [s a]
    return Aop

def fPEPS_op1s(A, op):
    """
    attaches operator to the tensor while calulating expectation
    values of local operators, no need to fuse auxiliary legs
    """
    Aop = tensordot(A, op, axes=(4, 1)) # t l b r [s a]
    return Aop

def fuse_ancilla_wos(op, fid):
    """ kron and fusion of local operator with identity for ancilla --- without string """
    op = ncon((op, fid), ((-0, -2), (-1, -3)))
    return op.fuse_legs(axes=((0, 1), (2, 3)))

def fuse_ancilla_ws(op, fid, dirn):
    """ kron and fusion of nn operator with identity for ancilla --- with string """
    if dirn == 'l':
        op= op.add_leg(s=1).swap_gate(axes=(0, 2))
        op = ncon((op, fid), ((-0, -2, -4), (-1, -3)))
        op = op.swap_gate(axes=(3, 4)) # swap of connecting axis with ancilla is always in GA gate
        op = op.fuse_legs(axes=((0, 1), (2, 3), 4))
    elif dirn == 'r':
        op = op.add_leg(s=-1)
        op = ncon((op, fid), ((-0, -2, -4), (-1, -3)))
        op = op.fuse_legs(axes=((0, 1), (2, 3), 4))
    return op

def fPEPS_2layers(A, B=None, op=None, dir=None):
    """
    Prepare top and bottom peps tensors for CTM procedures.
    Applies operators on top if provided, with dir = 'l', 'r', 't', 'b', '1s'
    If dir = '1s', no auxiliary indices are introduced as the operator is local.
    Here spin and ancilla legs of tensors are fused
    """
    leg = A.get_legs(axes=-1)

    if not leg.is_fused():  # when there is no ancilla on A, only the physical index is present
        A = A.add_leg(s=-1)
        A = A.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))  # new ancilla on outgoing leg
        leg = A.get_legs(axes=-1)

    _, leg = yastn.leg_undo_product(leg) # last leg of A should be fused
    fid = yastn.eye(config=A.config, legs=[leg, leg.conj()]).diag()

    if op is not None:
        if dir == 't':
            op_aux = fuse_ancilla_ws(op,fid,dirn='l')
            Ao = fPEPS_t(A,op_aux)
        elif dir == 'b':
            op_aux = fuse_ancilla_ws(op,fid,dirn='r')
            Ao = fPEPS_b(A,op_aux)
        elif dir == 'l':
            op_aux = fuse_ancilla_ws(op,fid,dirn='l')
            Ao = fPEPS_l(A,op_aux)
        elif dir == 'r':
            op_aux = fuse_ancilla_ws(op,fid,dirn='r')
            Ao = fPEPS_r(A,op_aux)
        elif dir == '1s':
            op = fuse_ancilla_wos(op,fid)
            Ao = fPEPS_op1s(A,op)
        else:
            raise RuntimeError("dir should be equal to 'l', 'r', 't', 'b' or '1s'")
    else:
        Ao = A  # t l b r [s a]

    if B is None:
        B = A  # t l b r [s a]

    AAb = DoublePepsTensor(top=Ao, btm=B)
    return AAb


def check_consistency_tensors(A):
    # to check if the A tensors have the appropriate configuration of legs i.e. t l b r [s a]

    Ab = fpeps.Peps(A)
    if A[0, 0].ndim == 6:
        for ms in Ab.sites():
            Ab[ms] = A[ms].fuse_legs(axes=(0, 1, 2, 3, (4, 5)))
    elif A[0, 0].ndim == 3:
        for ms in Ab.sites():
            Ab[ms] = A[ms].unfuse_legs(axes=(0, 1)) # system and ancila are fused by default
    else:
        for ms in Ab.sites():
            Ab[ms] = A[ms]   # system and ancila are fused by default
    return Ab

