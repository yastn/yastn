""" Mps structure and its basics. """
from __future__ import annotations
from ... import initialize, YastnError
from ._mps import MpsMpo


def load_from_dict(config, in_dict) -> yastn.tn.mps.MpsMpo:
    r"""
    Create MPS/MPO from dictionary.

    Parameters
    ----------
    config : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    in_dict: dict
        dictionary containing serialized MPS/MPO, i.e.,
        a result of :meth:`yastn.tn.mps.MpsMpo.save_to_dict`.
    """
    nr_phys = in_dict['nr_phys']
    N = in_dict['N'] if 'N' in in_dict else len(in_dict['A'])  # backwards compability
    out_mps = MpsMpo(N, nr_phys=nr_phys)
    if 'factor' in in_dict:  # backwards compability
        out_mps.factor = in_dict['factor']
    for n in range(out_mps.N):
        out_mps.A[n] = initialize.load_from_dict(config=config, d=in_dict['A'][n])
    return out_mps


def load_from_hdf5(config, file, my_address) -> yastn.tn.mps.MpsMpo:
    r"""
    Create MPS/MPO from HDF5 file.

    Parameters
    ----------
    config : module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    file: File
        A `pointer` to a file opened by the user

    my_address: str
        Name of a group in the file, where the Mps is saved, e.g., './state/'
    """

    nr_phys = int(file[my_address].get('nr_phys')[()])
    N = file[my_address].get('N')
    if N is None:
        N = len(file[my_address+'/A'].keys())
    else:
        N = int(N[()])
    out_Mps = MpsMpo(N, nr_phys=nr_phys)

    factor = file[my_address].get('factor')
    if factor:
        out_Mps.factor = factor[()]
    for n in range(out_Mps.N):
        out_Mps.A[n] = initialize.load_from_hdf5(config, file, my_address+'/A/'+str(n))
    return out_Mps
