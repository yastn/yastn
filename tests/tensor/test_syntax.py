""" list supported operations on a tensor (not all arguments are shown)"""
import unittest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


class TestSyntaxTensorCreation(unittest.TestCase):

    # a = yast.randR(config=config_U1, s=(-1, 1, 1, -1),
    #             t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
    #             D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    def test_syntax_tensor_creation_operations(self):
        # 
        # Initialize several rank-4 tensors, with the following signature
        #             ___
        #  (-) 0--<--| a |--<--1 (+)
        #  (+) 2-->--|___|-->--3 (-)
        #
        # The signatures can be interpreted as tensor legs being directed: ingoing for (+) 
        # or outgoing for (-).
        #  
        # The symmetry, U(1), is specified in config_U1. We specify the charge 
        # sectors on all legs by tuple t, with its first member, t[0], defining charge
        # sectors on the first leg, t[1] on second leg and so on.
        # The corresponding dimensions of each charge sector are specified by tuple D
        # with analogous structure as t.
        #
        # Then, upon creation, all blocks which respect charge conservation will be
        # initialized and filled with either random numbers, ones, or zeros in the examples
        # below.
        #
        # The dtype of the tensor elements as well as the device on which its data 
        # reside is given in config_U1.
        #
        a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        b = yast.ones(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        c = yast.zeros(config=config_U1, s=(-1, 1, 1, -1),
                       t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                       D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        
        #
        # The identity tensor behaves as rank-2 tensor with automatic signature (1,-1)
        # or (-1,1). It is enough to provide charge sectors and their dimensions 
        # for single leg, the data for other leg is inferred automatically. 
        e = yast.eye(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4))

    def test_syntax_create_empty_tensor_and_fill(self):
        # 
        # Initialize empty rank-4 tensor, with the following signature
        #             ___
        #  (-) 0--<--| a |--<--1 (+)
        #  (+) 2-->--|___|-->--3 (-)
        #
        # The symmetry, U(1), is specified in config_U1.
        #
        # Then, initialize some blocks with random values. The charges
        # of the block, ts, are given as a tuple with the length identical to
        # the rank of the tensor. Similarly, the dimensions of the block Ds.
        #
        d = yast.Tensor(config=config_U1, s=(-1, 1, 1, -1))
        d.set_block(ts=(1, -1, 2, 0), Ds=(2, 4, 9, 2), val='rand')
        d.set_block(ts=(2, 0, 2, 0), Ds=(3, 3, 9, 2), val='rand')

        #
        # Once the dimension is assigned to charge sector on a leg of the tensor
        # attempt to create block with different dimension will raise an error. 
        # In the example above sector with charge 2 on 3rd leg has dimension 9.
        #
        # Attempting to create new block with different dimension for the same
        # sector 2 on 3rd leg throws an error
        #
        with self.assertRaises(yast.YastError):
            d.set_block(ts=(2, 1, 2, 1), Ds=(3, 3, 10, 2), val='rand')


class TestSyntaxBasicAlgebra(unittest.TestCase):

    def test_syntax_basic_algebra(self):
        a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        
        # 
        # Tensor can be multiplied by scalar 
        # 
        tensor = a / 2
        tensor = 2. * a
        tensor = a * 2.

        #
        # Tensors can be added or subtracted assuming their structure is 
        # compatible.
        #
        b = yast.ones(config=config_U1, s=(-1, 1, 1, -1),
              t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
              D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        tensor = a + b
        tensor = a - b
        
        #
        # Attempting to add/subtract two tensors with different total charge,
        # or different dimension of a common charge sector raises exception
        # 
        c = yast.ones(config=config_U1, s=(-1, 1, 1, -1),
              t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
              D=((1, 2, 3), (4, 6, 6), (7, 8, 9), (10, 11, 12)))
        with self.assertRaises(Exception):
            tensor = a + c


class TestSyntaxTensorExportImport(unittest.TestCase):

    def test_syntax_tensor_export_import_operations(self):
        #
        # First, we crate a random U(1) symmetric tensor
        # Such tensor is stored as dict of non-zero blocks, indexed by charges
        #
        a= yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

        #
        # We can serialize symmetric tensors into 1-D vector, holding
        # reshaped raw-data of blocks and dictionary, meta, which holds
        # the symmetric structure of the tensors. Each entry of meta represents
        # non-zero block indexed by charges and it points to location of 1-D vector
        # where the raw data of that block is stored
        # 
        vector, meta = yast.compress_to_1d(a)
        vector, meta = a.compress_to_1d(meta=meta)
        tensor = yast.decompress_from_1d(vector, config_U1, meta)

        # 
        # Tensors can be also serialized directly into basic Python dictionary
        #
        dictionary = yast.export_to_dict(a)
        dictionary = a.export_to_dict()
        tensor = yast.import_from_dict(config=config_U1, d=dictionary)


class TestSyntaxBlockAccess(unittest.TestCase):

    def test_syntax_block_access(self):
        a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

        #
        # directly access block with charges (1, 1, 2, 2).
        #
        a[(1, 1, 2, 2)]

class TestSyntaxTensorBlocking(unittest.TestCase):

    def test_syntax_block_tensors(self):
        a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        b = yast.ones(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        c = yast.zeros(config=config_U1, s=(-1, 1, 1, -1),
                       t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                       D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

        # block tensors
        tensor = yast.block({(1, 1): a, (1, 2): b, (2, 1): c}, common_legs=(0, 1))
        # new tensor filled with ones, matching structure of selected legs -- to be used for e.g. dot
        tensor = yast.match_legs([a, a, b], legs=[1, 2, 0], conjs=[0, 0, 1], val='ones')
        # combined with ncon
        yast.ncon([tensor, a, b], [(1, 2, 3), (-1, 1, 2, -2), (3, -4, -5, -6)], conjs=(0, 0, 1))

    def test_syntax_general_operations(self):
        a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        b = yast.ones(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

        # algebra
        tensor = a.apxb(b, x=1)
        tensor = yast.apxb(a, b, x=1)

        # coping/cloning
        tensor = a.copy()
        tensor = yast.copy(a)
        tensor = a.clone()
        tensor = yast.clone(a)
        tensor = a.detach()
        tensor = yast.detach(a)
        tensor = a.to(device='cpu')

        # get info
        a.show_properties()
        a.get_ndim()
        a.get_size()
        a.get_tensor_charge()
        a.get_signature()
        a.get_leg_fusion()
        str(a)
        a.get_blocks_charges()
        a.get_blocks_shapes()
        a.get_leg_charges_and_dims()
        a.get_shape(axes=2)
        ls = a.get_leg_structure(axis=1)
        print(ls)
        

        # output dense
        array = a.to_dense()
        array = a.to_numpy()
        ls = {1: b.get_leg_structure(axis=1)}
        array = a.to_dense(leg_structures=ls)  # on selected legs, enforce to include cherges read in previous line
        tensor = a.to_nonsymmetric()

        # conj
        tensor = a.conj()
        tensor = yast.conj(a)
        tensor = a.conj_blocks()
        tensor = yast.conj_blocks(a)
        tensor = a.flip_signature()
        tensor = yast.flip_signature(a)

        # permute
        tensor = a.transpose(axes=(2, 3, 0, 1))
        tensor = yast.transpose(a, axes=(2, 3, 0, 1))

        tensor = a.moveaxis(source=2, destination=3)
        tensor = yast.moveaxis(a, source=2, destination=3)

        # elementwise operations
        tensor = a.exp(step=1)
        tensor = yast.exp(a, step=1)
        tensor = a.abs()
        tensor = a.absolute()
        tensor = yast.absolute(a)

        tensor = a.abs().sqrt()
        tensor = yast.sqrt(a.abs())
        tensor = a.abs().rsqrt(cutoff=1e-12)
        tensor = yast.rsqrt(a.abs(), cutoff=1e-12)
        tensor = a.reciprocal(cutoff=1e-12)
        tensor = yast.reciprocal(a, cutoff=1e-12)

        # contraction
        tensor = yast.tensordot(a, b, axes=((1, 2), (1, 2)), conj=(1, 0))
        tensor = a.tensordot(b, axes=((1, 2), (1, 2)), conj=(1, 0))

        tensor = a.tensordot(b, axes=((0, 1, 2, 3), (0, 1, 2, 3)), conj=(1, 0))
        number = tensor.to_number()
        python_scalar = tensor.item()

        # scalar product
        number = yast.vdot(a, b)
        number = a.vdot(b)

        # trace
        a2 = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(1, 0))
        tensor = a2.trace(axes=(0, 2))
        tensor = a2.trace(axes=((0, 1), (2, 3)))
        number = tensor.to_number()

        # linalg / split
        U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)))
        U, S, V = yast.svd(a, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation
        U, S, V = a.svd(axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation
        try:
            U, S, V = yast.svd_lowrank(a, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2, n_iter=5, k_fac=2)
            U, S, V = a.svd_lowrank(axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2, n_iter=5, k_fac=2)
        except NameError:
            pass

        Q, R = yast.linalg.qr(a, axes=((0, 1), (2, 3)))
        Q, R = yast.qr(a, axes=((0, 1), (2, 3)))
        Q, R = a.qr(axes=((0, 1), (2, 3)))

        D, U = yast.linalg.eigh(a2, axes=((0, 1), (2, 3)))
        D, U = yast.eigh(a2, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation
        D, U = a2.eigh(axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation

        # linalg
        number = a.norm()
        number = yast.norm(a, p='fro')  # p = 'for', i.e. frobenius is default
        number = yast.linalg.norm(a, p='inf')

        number = a.norm_diff(b)
        number = yast.norm_diff(a, b)
        number = yast.linalg.norm_diff(a, b)
        number = yast.norm(a - b)

        # utils
        entropy, Smin, normalization = yast.entropy(a, axes=((0, 1), (2, 3)))

        # fuse
        tensor = a.fuse_legs(axes=(0, (1, 3), 2))
        tensor = tensor.unfuse_legs(axes=1)

        tensor = yast.fuse_legs(a, axes=(0, (1, 3), 2))
        yast.unfuse_legs(tensor, axes=(0, (1, 3), 2), inplace=True)

        # tests
        a.is_consistent()
        a.are_independent(b)
>>>>>>> links to examples in tests


if __name__ == '__main__':
    unittest.main()
