""" Mps structure and its basics. """
from ... import initialize, YastnError
from ._mps import MpsMpo


def load_from_dict(config, in_dict):
    r"""
    Create MPS/MPO from dictionary.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    in_dict: dict
        dictionary containing serialized MPS/MPO, i.e.,
        a result of :meth:`yastn.tn.mps.MpsMpo.save_to_dict`.

    Returns
    -------
    yastn.tn.mps.MpsMpo
    """
    nr_phys = in_dict['nr_phys']
    N = len(in_dict['A'])
    out_Mps = MpsMpo(N, nr_phys=nr_phys)
    for n in range(out_Mps.N):
        out_Mps.A[n] = initialize.load_from_dict(config=config, d=in_dict['A'][n])
    return out_Mps


def load_from_hdf5(config, file, my_address):
    r"""
    Create MPS/MPO from HDF5 file.

    Parameters
    ----------
    config : module, types.SimpleNamespace, or typing.NamedTuple
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    file: File
        A `pointer` to a file opened by the user

    my_address: str
        Name of a group in the file, where the Mps is saved, e.g., './state/'

    Returns
    -------
    yastn.tn.mps.MpsMpo
    """
    sym_id = file[my_address].get('sym/SYM_ID')[()]
    nsym = file[my_address].get('sym/NSYM')[()]
    if not sym_id.decode('ascii') == config.sym.SYM_ID or not nsym == config.sym.NSYM:
        raise YastnError("config doesn't match the one for saved data")
    nr_phys = int(file[my_address].get('nr_phys')[()])
    N = len(file[my_address+'/A'].keys())
    out_Mps = MpsMpo(N, nr_phys=nr_phys)
    for n in range(out_Mps.N):
        out_Mps.A[n] = initialize.load_from_hdf5(config, file, my_address+'/A/'+str(n))
    return out_Mps
