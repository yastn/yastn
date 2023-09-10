from ...tn.mps import Mps, Mpo
from ._doublePepsTensor import DoublePepsTensor
from ... import tensor, initialize
from ._geometry import Lattice

def mpo(self, index, index_type, rotation=''):

    """Converts a specific row or column of PEPS into MPO.

    Parameters
    ----------
    index (int): The row or column index to convert.
    index_type (str): The index type to convert, either 'row' or 'column'.
    rotation (str): Optional string indicating the rotation of the PEPS tensor.

    """

    if index_type == 'row':
        nx = index
        H = Mpo(N=self.Ny)
        for ny in range(self.Ny):
            site = (nx, ny)
            top = self[site]
            if top.ndim == 3:
                top = top.unfuse_legs(axes=(0, 1))
            btm = top.swap_gate(axes=(0, 1, 2, 3))
            H.A[ny] = DoublePepsTensor(top=top, btm=btm)
    elif index_type == 'column':
        ny = index
        H = Mpo(N=self.Nx)
        for nx in range(self.Nx):
            site = (nx, ny)
            top = self[site]
            if top.ndim == 3:
                top = top.unfuse_legs(axes=(0, 1))
            btm = top.swap_gate(axes=(0, 1, 2, 3))
            H.A[nx] = DoublePepsTensor(top=top, btm=btm)

    return H

def boundary_mps(self, rotation=''):

    r"""Returns a boundary MPS at the right most column.

    Parameters
    ----------
        rotation (str): Optional string indicating the rotation of the PEPS tensor.

    """
    psi = Mps(N=self.Nx)
    cfg = self._data[(0, 0)].config
    n0 = (0,) * cfg.sym.NSYM
    leg0 = tensor.Leg(cfg, s=-1, t=(n0,), D=(1,))
    for nx in range(self.Nx):
        site = (nx, self.Ny-1)
        A = self[site]
        if A.ndim == 3:
            legA = A.get_legs(axes=1)
            _, legA = tensor.leg_undo_product(legA)
        else:
            legA = A.get_legs(axes=3)
        legAAb = tensor.leg_outer_product(legA, legA.conj())
        psi[nx] = initialize.ones(config=cfg, legs=[leg0, legAAb.conj(), leg0.conj()])

    return psi 


Lattice.mpo = mpo
Lattice.boundary_mps = boundary_mps

 

