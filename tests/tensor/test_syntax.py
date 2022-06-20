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

        # 
        # element-wise exponentiation, absolute value, reciprocal i.e. x -> 1/x, 
        # square root and its reciprocal x -> 1/sqrt(x)
        #
        tensor = a.exp(step=1)
        tensor = yast.exp(a, step=1)
        
        tensor = abs(a)

        tensor = a.reciprocal(cutoff=1e-12)
        tensor = yast.reciprocal(a, cutoff=1e-12)
        
        tensor = abs(a).sqrt()
        tensor = yast.sqrt(abs(a))
        
        tensor = abs(a).rsqrt(cutoff=1e-12)
        tensor = yast.rsqrt(abs(a), cutoff=1e-12)
        

        #
        # Sometimes a composite operation is faster than serial execution of individual 
        # operations. For example, multiplication by scalar and addition, a + x*b, 
        # are handled by specialized function
        # 
        tensor = a.apxb(b, x=1)
        tensor = yast.apxb(a, b, x=1)


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
        dictionary = yast.save_to_dict(a)
        dictionary = a.save_to_dict()
        tensor = yast.load_from_dict(config=config_U1, d=dictionary)


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

        tensor = yast.ones(config=a.config, legs=[a.get_legs(1).conj(), a.get_legs(2).conj(), b.get_legs(0)])
        # combined with ncon
        yast.ncon([tensor, a, b], [(1, 2, 3), (-1, 1, 2, -2), (3, -4, -5, -6)], conjs=(0, 0, 1))


class TestSyntaxContractions(unittest.TestCase):

    def test_syntax_contraction(self):
        # create a set of U(1)-symmetric tensors
        a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        b = yast.ones(config=config_U1, s=(-1, 1, 1, -1),
                      t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        c = yast.rand(config=config_U1, s=(1, 1, -1),
                      t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                      D=((10, 11, 12), (7, 8, 9), (4, 5, 6)))

        # Contract a and b by two indices. The a tensor is conjugated, which
        # reverses the signature on its indices
        #       __           _                ___
        #  0->-|a*|->-1 1->-|b|->-0 =    0->-|a*b|->-0->2
        #  3->-|__|->-2 2->-|_|->-3   1<-3->-|___|->-3
        #
        # The order of the indices on the resulting tensor is as follows:
        # First, the outgoing indices of a (the first argument to tensordot), then
        # the outgoing indices of tensor b
        tensor = yast.tensordot(a, b, axes=((1, 2), (1, 2)), conj=(1, 0))
        
        # tensordot can also be invoked also as a function of the tensor itself
        #
        tensor = a.tensordot(b, axes=((1, 2), (1, 2)), conj=(1, 0))

        # If no axes are specified, the outer product of two tensors is returned
        tensor = yast.tensordot( c,c, axes=((),()) )
        assert tensor.get_rank()==6


        # A shorthand notation for the specific contraction
        #      _           _             __
        # 0-<-|a|-<-2     |c|-<-1 = 0-<-|ac|-<-2
        # 1->-|_|->-3 0->-|_|->-2   1->-|  |-<-1->3
        #                               |__|->-2->4
        t0= yast.tensordot(a, c, axes=(a.ndim - 1, 0)) 
        # 
        # is the @ operator. For rank-2 tensor it is thus equivalent to matrix multiplication
        t1 = a @ c
        assert yast.norm(t0-t1) < tol


        # Another special case of tensor contraction is a dot product of vectorized tensors
        #  __           _
        # |a*|-<-0 0-<-|b| = scalar
        # |  |->-1 1->-| |
        # |  |->-2 2->-| | 
        # |__|-<-3 3-<-|_|
        tensor = a.tensordot(b, axes=((0, 1, 2, 3), (0, 1, 2, 3)), conj=(1, 0))
        assert isinstance(tensor,yast.Tensor)
        #
        # such single element symmetric Tensor can be converted to a single-element
        # tensor of the backend type, or even further to python scalar
        number = tensor.to_number()
        python_scalar = tensor.item()
        assert isinstance(python_scalar,float)

        # A shorthand function for computing dot products is vdot
        number = yast.vdot(a, b)
        number = a.vdot(b)

        # Trace over certain indices can be computed using identically named function.
        # In this case, a2_ijil = a2_jl
        a2 = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(1, 0))
        tensor = a2.trace(axes=(0, 2))
        assert tensor.get_rank()==2
        # 
        # More pairs of indices can be traced at once a_ijij = scalar
        tensor = a2.trace(axes=((0, 1), (2, 3)))
        number = tensor.to_number()


class TestSyntaxGeneral(unittest.TestCase):

    def test_syntax_noDocs(self):
        # initialization
        cfg_none = yast.make_config()  # backend_np; sym_none
        # not imported config
        if config_U1.backend.BACKEND_ID == 'numpy':
            cfg_U1 = yast.make_config(sym=yast.sym.sym_U1, backend=yast.backend.backend_np)
        else:
            cfg_U1 = yast.make_config(sym=yast.sym.sym_U1, backend=yast.backend.backend_torch)
        legs = [yast.Leg(cfg_U1, s=-1, t=(-1, 1, 0), D=(1, 2, 3)),
                yast.Leg(cfg_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
                yast.Leg(cfg_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
                yast.Leg(cfg_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]
        a = yast.rand(config=cfg_U1, legs=legs)
        b = yast.ones(config=config_U1, legs=legs)

        # conj - documented example in test_conj.py
        tensor = a.conj()
        tensor = yast.conj(a)
        tensor = a.conj_blocks()
        tensor = yast.conj_blocks(a)
        tensor = a.flip_signature()
        tensor = yast.flip_signature(a)

        # coping/cloning - documented example in test_autograd.py
        tensor = a.copy()
        tensor = yast.copy(a)
        tensor = a.clone()
        tensor = yast.clone(a)
        tensor = a.detach()
        tensor = yast.detach(a)

        # to
        tensor = a.to(device='cpu')
        tensor = a.to(dtype='complex128')

        # get info
        a.show_properties()
        a.get_rank()
        a.size
        a.get_tensor_charge()
        a.get_signature()
        a.get_leg_fusion()
        str(a)
        a.get_blocks_charge()
        a.get_blocks_shape()
        a.get_shape()
        a.get_shape(axis=2)
        legs = a.get_legs()
        leg = a.get_legs(axis=2)  # legs[2] = leg
        print(leg.tD) # dict od charges with dimensions spanning the leg
        print(leg)
        a.get_dtype()
        a.dtype

        # output dense
        array = a.to_dense()
        array = a.to_numpy()
        ls = {1: b.get_legs(axis=1)}
        array = a.to_dense(legs=ls)  # on selected legs, enforce to include charges read in previous line
        tensor = a.to_nonsymmetric()

        # permute - documented example in test_transpose.py
        tensor = a.transpose(axes=(2, 3, 0, 1))
        tensor = yast.transpose(a, axes=(2, 3, 0, 1))

        tensor = a.move_leg(source=2, destination=3)
        tensor = yast.move_leg(a, source=2, destination=3)

        a2 = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(1, 0))
        # linalg / split
        U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)))
        U, S, V = yast.svd(a, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation
        U, S, V = a.svd(axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation
        try:
            U, S, V = yast.svd(a, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2, n_iter=5, k_fac=2, policy='lowrank')
            U, S, V = a.svd(axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2, n_iter=5, k_fac=2, policy='lowrank')
        except NameError:
            pass

        Q, R = yast.linalg.qr(a, axes=((0, 1), (2, 3)))
        Q, R = yast.qr(a, axes=((0, 1), (2, 3)))
        Q, R = a.qr(axes=((0, 1), (2, 3)))

        D, U = yast.linalg.eigh(a2, axes=((0, 1), (2, 3)))
        D, U = yast.eigh_with_truncation(a2, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation
        D, U = a2.eigh_with_truncation(axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation

        # linalg
        number = a.norm()
        number = yast.norm(a, p='fro')  # p = 'for', i.e. frobenius is default
        number = yast.linalg.norm(a, p='inf')

        number = yast.norm(a - b)
        number = yast.norm(a - b)
        number = yast.linalg.norm(a - b)
        number = yast.norm(a - b)

        # utils
        entropy, Smin, normalization = yast.entropy(a, axes=((0, 1), (2, 3)))

        # fuse
        tensor = a.fuse_legs(axes=(0, (1, 3), 2))
        tensor = tensor.unfuse_legs(axes=1)

        tensor = yast.fuse_legs(a, axes=(0, (1, 3), 2))
        tensor = yast.unfuse_legs(tensor, axes=(0, (1, 3), 2))

        # tests
        a.is_consistent()
        a.are_independent(b)


if __name__ == '__main__':
    unittest.main()
