from ...tn.mps import Mps, Mpo
from ._doublePepsTensor import DoublePepsTensor
from ... import tensor, initialize


def show_leg_structure(peps):
   """ Prints the leg structure of each site tensor in a PEPS """
   for ms in peps.sites():
        xs = peps[ms].unfuse_legs((0, 1))
        print("site ", str(ms), xs.get_shape())



def transfer_mpo(self, index, index_type, one_layer=False):

    """Converts a specific row or column of PEPS into MPO.

    Parameters
    ----------
    index (int): The row or column index to convert.
    index_type (str): The index type to convert, either 'row' or 'column'.
    rotation (str): Optional string indicating the rotation of the PEPS tensor.
    """

    if index_type == 'row':
        nx = index  # is this ever used?
        H = Mpo(N=self.Ny)
        for ny in range(self.Ny):
            site = (nx, ny)
            top = self[site]
            if top.ndim == 3:
                top = top.unfuse_legs(axes=(0, 1))
            H.A[ny] = top.transpose(axes=(1, 2, 3, 0)) if one_layer else \
                      DoublePepsTensor(top=top, btm=top, transpose=(1, 2, 3, 0))
    elif index_type == 'column':
        ny = index
        H = Mpo(N=self.Nx)
        for nx in range(self.Nx):
            site = (nx, ny)
            top = self[site]
            if top.ndim == 3:
                top = top.unfuse_legs(axes=(0, 1))
            H.A[nx] = top if one_layer else \
                      DoublePepsTensor(top=top, btm=top)
    return H
