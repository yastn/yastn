""" Mps structure and its basic """
from turtle import position
import numpy as np
import yast
from ._mps import MpsMpo, add

class YampsError(Exception):
    pass


def load_from_dict(config, nr_phys, in_dict):
    r"""
    Reads Tensor-s of Mps from a dictionary into an Mps object

    Returns
    -------
    out_Mps : Mps
    """
    N = len(in_dict)
    out_Mps = MpsMpo(N, nr_phys=nr_phys)
    for n in range(out_Mps.N):
        out_Mps.A[n] = yast.load_from_dict(config=config, d=in_dict[n])
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
    out_Mps = MpsMpo(N, nr_phys=nr_phys)
    for n in range(out_Mps.N):
        out_Mps.A[n] = yast.load_from_hdf5(config, file, in_file_path+str(n))
    return out_Mps


# # cp = yast.ones(config=config_U1)
def generate_mpo(N, H, identity, opts={'tol': 1e-14}):
    identity2 = identity.copy().add_leg(axis=0, s=1).add_leg(axis=-1, s=-1)
    id_helper = MpsMpo(N, nr_phys=2)
    # prepare identity
    for n in range(id_helper.N):
        id_helper.A[n] = identity2
    H_tens = [None]*len(H)
    for j1 in range(len(H)):
        op_info = H[j1]
        print(op_info)
        product_tmp = id_helper.copy()
        amplitude, positions = op_info["amp"], op_info.keys()
        for j2 in list(positions)[1::]:
            operator = op_info[j2].add_leg(axis=0, s=1)
            product_tmp.A[j2] = yast.ncon([product_tmp.A[j2], operator], [(-1,1,-4,-5,),(-2,-3,1)])
            product_tmp.A[j2] = product_tmp.A[j2].fuse_legs(axes=((0,1),2,3,(4)), mode='hard')
            print("0:", j2, len(product_tmp.A[j2].get_legs()))
            for j3 in range(j2):
                print("j3:", j3)
                operator = yast.ones(config=op_info[j2].config, legs=[operator.get_legs()[0]], n=op_info[j2].n, isdiag=op_info[j2].isdiag).conj()
                operator = operator.add_leg(axis=0, s=1)
                product_tmp.A[j2-1-j3] = yast.ncon([product_tmp.A[j2-1-j3], operator], [(-1,-3,-4,-5),(-2,-6)])
                product_tmp.A[j2-1-j3] = product_tmp.A[j2-1-j3].swap_gate(axes=(1,2,))
                product_tmp.A[j2-1-j3] = product_tmp.A[j2-1-j3].fuse_legs(axes=((0,1),2,3,(4,5)), mode='hard')
        H_tens[j1] = amplitude * product_tmp
    M = add(*H_tens)
    M.canonize_sweep(to='last', normalize=False)
    M.truncate_sweep(to='first', opts=opts, normalize=False)
    return M


def automatic_Mps(amplitude, from_it, to_it, permute_amp, Tensor_from, Tensor_to, Tensor_conn, Tensor_other, N, nr_phys,  common_legs, opts={'tol': 1e-14}):
    r"""
    Generate Mps representuing sum of two-point operators M=\sum_i,j Mij Op_i Op_j with possibility to include Jordan-Wigner chains for these.

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
    bunch_tens, bunch_amp = [], []
    for it in np.nonzero(np.array(amplitude))[0]:
        if from_it[it] > to_it[it]:
            conn, other = Tensor_conn[it], Tensor_other[it]
            if nr_phys > 1:
                left, right = Tensor_to[it].tensordot(conn, axes=common_legs[::-1]), Tensor_from[it]
            else:
                left, right = Tensor_to[it], Tensor_from[it]
            il, ir = to_it[it], from_it[it]
            amp = amplitude[it] * permute_amp[it]
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

        bunch_tens.append(_generate_Mij(1., connect, N, nr_phys))
        bunch_amp.append(amp)

    M = add(*bunch_tens, amplitudes=bunch_amp)
    M.canonize_sweep(to='last', normalize=False)
    M.truncate_sweep(to='first', opts=opts, normalize=False)
    return M


def _generate_Mij(amp, connect, N, nr_phys):
    jL, T_from = connect['from']
    jR, T_to = connect['to']
    T_conn = connect['conn']
    T_else = connect['else']

    M = MpsMpo(N, nr_phys=nr_phys)
    tt = (0,) * len(T_from.n)
    for n in range(M.N):
        if jL == jR:
            M.A[n] = amp * T_from.copy() if n == jL else T_else.copy()
        else:
            if n == jL:
                M.A[n] = amp * T_from.copy()
            elif n == jR:
                M.A[n] = T_to.copy()
            elif n > jL and n < jR:
                M.A[n] = T_conn.copy()
            else:
                M.A[n] = T_else.copy()
        M.A[n] = M.A[n].add_leg(axis=0, t=tt, s=1)
        M.A[n] = M.A[n].add_leg(axis=-1, s=-1)
        tt = M.A[n].get_legs(axis=-1).t[0]
    return M
