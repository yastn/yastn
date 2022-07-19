import numpy as np
import yamps
import yast

####### MPO for XX model ##########
def mpo_XX_model_dense(config, N, t, mu):
    cp = np.array([[0, 0], [1, 0]])
    c = np.array([[0, 1], [0, 0]])
    nn = np.array([[0, 0], [0, 1]])
    ee = np.array([[1, 0], [0, 1]])
    oo = np.array([[0, 0], [0, 0]])

    H = yamps.Mpo(N)
    for n in H.sweep(to='last'):  # empty tensors
        H.A[n] = yast.Tensor(config=config, s=(1, 1, -1, -1))
        if n == H.first:
            tmp = np.block([[mu * nn, t * cp, t * c, ee]])
            tmp = tmp.reshape((1, 2, 4, 2))
            Ds = (1, 2, 2, 4)
        elif n == H.last:
            tmp = np.block([[ee], [c], [cp], [mu * nn]])
            tmp = tmp.reshape((4, 2, 1, 2))
            Ds = (4, 2, 2, 1)
        else:
            tmp = np.block([[ee, oo, oo, oo],
                            [c, oo, oo, oo],
                            [cp, oo, oo, oo],
                            [mu * nn, t * cp, t * c, ee]])
            tmp = tmp.reshape((4, 2, 4, 2))
            Ds = (4, 2, 2, 4)
        tmp = np.transpose(tmp, (0, 1, 3, 2))
        H.A[n].set_block(val=tmp, Ds=Ds)
    return H

def mpo_XX_model_Z2(config, N, t, mu):
    H = yamps.Mpo(N)
    for n in H.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config, s=[1, 1, -1, -1], n=0)
        if n == H.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[mu, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, 1), val=[t, 0], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[0, t], Ds=(1, 1, 1, 2))
        elif n == H.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, mu], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(1, 1, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[0, 1], Ds=(2, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, 1), val=[[0, 0], [t, 0]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[[0, 0], [0, t]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(1, 1, 0, 0), val=[[1, 0], [0, 0]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[[0, 0], [1, 0]], Ds=(2, 1, 1, 2))
    return H


def mpo_XX_model_U1(config, N, t, mu):
    H = yamps.Mpo(N)
    for n in H.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config, s=[1, 1, -1, -1], n=0)
        if n == H.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[mu, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, -1), val=[t], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[t], Ds=(1, 1, 1, 1))
        elif n == H.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, mu], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[1], Ds=(1, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, -1), val=[0, t], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[0, t], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[1, 0], Ds=(1, 1, 1, 2))
    return H

def mpo_XX_model(config, N, t, mu):
    if config.sym.SYM_ID == 'dense':
        return mpo_XX_model_dense(config, N, t, mu)
    elif config.sym.SYM_ID == 'Z2':
        return mpo_XX_model_Z2(config, N, t, mu)
    elif config.sym.SYM_ID == 'U(1)':
        return mpo_XX_model_U1(config, N, t, mu)
