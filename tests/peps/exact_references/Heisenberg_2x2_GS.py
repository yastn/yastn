import numpy as np
import scipy.linalg as LA

sigma_x = 0.5*np.array([[0, 1], [1, 0]])
sigma_y = 0.5*np.array([[0, -1j], [1j, 0]])
sigma_z = 0.5*np.array([[1, 0], [0, -1]])
iden = np.eye(2)

J = 0.333333333333333


Nx, Ny = 3, 3

def label(y, x, Nx):
    return Nx*y + x

def init_ops(op):

    def mult(x1, x2, x3, x4, x5, x6, x7, x8, x9):
        listt = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
      
        nn = listt[0]
        for x in range(len(listt)-1):
            nn = np.kron(nn, listt[x+1])
          
        return nn

    Sm = {}
    Sm[0] = mult(op, iden, iden, iden, iden, iden, iden, iden, iden)
    Sm[1] = mult(iden, op, iden, iden, iden, iden, iden, iden, iden)
    Sm[2] = mult(iden, iden, op, iden, iden, iden, iden, iden, iden)
    Sm[3] = mult(iden, iden, iden, op, iden, iden, iden, iden, iden)
    Sm[4] = mult(iden, iden, iden, iden, op, iden, iden, iden, iden)
    Sm[5] = mult(iden, iden, iden, iden, iden, op, iden, iden, iden)
    Sm[6] = mult(iden, iden, iden, iden, iden, iden, op, iden, iden)
    Sm[7] = mult(iden, iden, iden, iden, iden, iden, iden, op, iden)
    Sm[8] = mult(iden, iden, iden, iden, iden, iden, iden, iden, op)
    return Sm


def int_energy(op):

    SS = init_ops(op)
    total_int = np.matmul(SS[0],SS[1]) + np.matmul(SS[1],SS[2]) + np.matmul(SS[3], SS[4]) + np.matmul(SS[4], SS[5]) + np.matmul(SS[6],SS[7]) + np.matmul(SS[7],SS[8]) + np.matmul(SS[0],SS[3]) + np.matmul(SS[1],SS[4]) + np.matmul(SS[2],SS[5]) + np.matmul(SS[3],SS[6]) + np.matmul(SS[4],SS[7]) + np.matmul(SS[5],SS[8])
    return total_int


H = J * (int_energy(sigma_x) + int_energy(sigma_y) + int_energy(sigma_z))
print(np.shape(H))
print(np.sort(LA.eigvals(H)))




