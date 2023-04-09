import numpy as np
import scipy.linalg as LA


def evolve_correlation_matrix(C, J, t):
    """
    Evolve correlation matrix C by time t with hopping Hamiltonian J.
    Diagonal od J gives chemical potential on sites, and upper triangular of J are hopping amplitudes between sites.
    """
    J = np.triu(J, 0) + np.triu(J, 1).T.conj()
    U = LA.expm(1j * J * t)
    return U.conj().T @ C @ U


def gs_correlation_matrix(J, n):
    """
    Correlation matrix for ground state of n particles of hopping Hamiltonian J.

    C[m, n] = <c_n^dag c_m>
    """
    J = np.triu(J, 0) + np.triu(J, 1).T.conj()

    D, U = LA.eigh(J)
    Egs = np.sum(D[:n])

    C0 = np.zeros(len(D))
    C0[:n] = 1
    C = U @ np.diag(C0) @ U.T.conj()

    return C, Egs


if __name__ == "__main__":
    J = np.array([[1, 0.5, 0, 0.3, 0.1], [0, -1, 0.5, 0, 0.3], [0, 0, 1, 0.5, 0], [0, 0, 0, -1, 0.5], [0, 0, 0, 0, 1]])
    C, Egs = gs_correlation_matrix(J, 3)
    print(Egs, C)
