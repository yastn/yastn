import h5py
import transport_vectorization as main_fun

def test_import_h5py(tensor_type, filename):
    big_file = h5py.File(filename, 'r')
    phi = main_fun.import_psi_from_h5py(big_file, tensor_type=tensor_type)
    phi.canonize_sweep(to='first')
    big_file.close()


if __name__ == "__main__":
    # pass
    filename = 'models/transport_open/test_output.h5'
    tensor_type = ('U1', 'complex128')
    test_import_h5py(tensor_type ,filename)