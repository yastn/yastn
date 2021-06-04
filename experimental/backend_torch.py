import torch

def randR(D, device='cpu'):
    return 2 * torch.rand(D, device=device) - 1


def dot(A, B, meta_dot):
    C = {}
    for (out, ina, inb) in meta_dot:
        C[out] = A[ina] @ B[inb]
    return C

def merge_to_matrix(A, order, meta_new, meta_mrg, device='cpu'):
    """ New dictionary of blocks after merging into matrix. """
    Anew = {u: torch.zeros(Du, dtype=torch.float64, device=device) for (u, Du) in meta_new}
    for (tn, to, Dsl, Dl, Dsr, Dr) in meta_mrg:
        Anew[tn][slice(*Dsl), slice(*Dsr)] = A[to].permute(order).reshape(Dl, Dr)
    return Anew
