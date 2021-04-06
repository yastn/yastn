import yast
import config_U1_R

tol = 1e-12

def test_syntax():
    """ List of commands and syntax. Not all possible parameters of some functions are shown below."""

    # initialization:
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    # a = yast.randR(config=config_U1_R, s=(-1, 1, 1, -1),
    #             t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
    #             D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    b = yast.ones(config=config_U1_R, s=(-1, 1, 1, -1),
                t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    c = yast.zeros(config=config_U1_R, s=(-1, 1, 1, -1),
                t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    d = yast.Tensor(config=config_U1_R, s=(-1, 1, 1, -1))
    d.set_block(ts=(1, -1, 2, 0), Ds=(2, 4, 9, 2), val='rand')
    d.set_block(ts=(2, 0, 2, 0), Ds=(3, 3, 9, 2), val='rand')
    e = yast.eye(config=config_U1_R, t=(-1, 0, 1), D=(2, 3, 4))

    # import and export
    vector, meta = yast.compress_to_1d(a)
    vector, meta = a.compress_to_1d(meta=meta)
    tensor = yast.decompress_from_1d(vector, config_U1_R, meta)

    dictionary = yast.export_to_dict(a)
    dictionary = a.export_to_dict()
    tensor = yast.import_from_dict(config=config_U1_R, d=dictionary)

    # block tensors
    tensor = yast.block({(1, 1): a, (1, 2): b, (2, 1): c}, common_legs=(0, 1))
    # new tensor filled with ones, matching structure of selected legs -- to be used for e.g. dot
    tensor = yast.match_legs([a, a, b], legs=[1, 2, 0], conjs=[0, 0, 1], val='ones')
    # combined with ncon
    yast.ncon([tensor, a, b], [(1, 2, 3), (-1, 1, 2, -2), (3, -4, -5, -6)], conjs=(0, 0, 1))

    # algebra
    tensor = a + b
    tensor = a - b
    tensor = a / 2
    tensor = 2. * a
    tensor = a * 2.
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
    tensor = a.copy_empty()

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
    a.get_leg_structure(axis=1)
    a[(1, 1, 2, 2)]  # directly access block of charges (1, 1, 2, 2)

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
    tensor = a.transpose(axes=(2,3,0,1))
    tensor = yast.transpose(a, axes=(2,3,0,1))
    
    tensor = a.moveaxis(source=2, destination=3)
    tensor = yast.moveaxis(a, source=2, destination=3)

    # elementwise operations
    tensor = a.exp(step=1)
    tensor = yast.exp(a, step=1)
    tensor = a.abs()
    tensor = yast.abs(a)

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
    U, S, V = yast.svd(a, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block = 2)  # here with truncation
    try:
        U, S, V = yast.svd_lowrank(a, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block = 2, n_iter=5, k_fac=2)
    except NameError:
        pass

    Q, R = yast.linalg.qr(a, axes=((0, 1), (2, 3)))
    Q, R = yast.qr(a, axes=((0, 1), (2, 3)))

    D, U = yast.linalg.eigh(a2, axes=((0, 1), (2, 3)))
    D, U = yast.eigh(a2, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block = 2)  # here with truncation

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
    a.is_independent(b)

if __name__ == '__main__':
    test_commands()
