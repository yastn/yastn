import json
import argparse
import os
from itertools import groupby
from typing import Sequence, NamedTuple, Union
import numpy as np
import yastn
from yastn.tensor._merging import _meta_fuse_hard
from yastn.tensor.linalg import _meta_svd
import timeit
from unittest.mock import patch
import logging
LOGFILE="bench_C2x2.log"
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(LOGFILE,mode = "w+"),])
logger = logging.getLogger()


def read_tensors(config, D : int, fpath):
    """
    D : test case to read
    fpath : Union[str,Path] 
        path to JSON with test case definitions

    Read structure of on-site and environment tensors for selected D from JSON.
    These correspond to realistic example of U(1)-symmetric Heisenberg model with bond dimension
    from D=2,...,8
    
    Legs of on-site tensor are ordered as physical, up/top, left, down/bottom, right
    """
    with open(fpath) as f:
        d = json.load(f)
        test_case= d["test_cases"][f"{D}"]
    n=config.sym.zero() if not ("onsite_charge" in test_case) else test_case["onsite_charge"]
    a= yastn.rand(config, s= eval(d["onsite_signature"]), n=n, t=eval(test_case["onsite_leg_charges"]), D=eval(test_case["onsite_leg_dims"]))
    env_leg= yastn.Leg(config, -1, eval(test_case["env_leg_charges"]), eval(test_case["env_leg_dims"]))
    C_tr= yastn.rand(config, legs=(env_leg, env_leg.conj()))
    T_t= yastn.rand(config, legs=(env_leg, a.get_legs(1).conj(), a.get_legs(1), env_leg.conj()))
    T_r= yastn.rand(config, legs=(env_leg, a.get_legs(4).conj(), a.get_legs(4), env_leg.conj()))
    return a, C_tr, T_t, T_r

def generate_tensors(config, cs: Sequence[int], Ds : Sequence[int]):
    """
    cs : charge sectors for auxiliary index of on-site tensor
    Ds : bond dimensions for auxiliary index of on-site tensor

    Generate U(1)-symmetric tensors for realistic test case corresponding to environment dimension 2*(D_total^2).

    For up to D=8, the optimal sequene for AFM NN spin-1/2 Heisenberg model is observed to be

    D_total cs Ds
    2       (0,2) (1,1)
    3       (0,2) (2,1)
    4       (-2,0,2) (1,2,1)
    5       (-2,0,2) (1,2,2)
    6       (-2,0,2) (2,2,2)
    7       (-2,0,2) (2,3,2)
    8       (-2,0,2) (2,3,3)
    9*      (-2,0,2) (3,3,3)
     
    For this content on auxiliary legs and optimized tensor elements, the content of environment leg 
    is observed to behave as environment dimensions 
        
        S * exp(-(charge**2)/(2*sigma**2)), with :math:`sigma \approx 2.6`
        
    Where the overall scale S, assumming total environment dimension chi=2*(D^2) to be S = 1.5 + 0.4 * D^2
    """
    assert config.sym.SYM_ID=="U1","Expects only U(1) symmetry"
    sigma=2.6
    S= (1.52 + 0.439 * sum(Ds)**2.14)
    max_charge= int( np.sqrt( -(2.*(sigma**2)) * np.log(1./S)) )
    max_charge= max_charge if max_charge%2==0 else max_charge-1

    charges= tuple( c for c in range(-max_charge,max_charge+2,2) )
    dims= tuple( int( S*np.exp(-c**2/(2*sigma**2)) ) for c in charges )
    env_leg= yastn.Leg(config, -1, charges, dims)

    phys_leg= yastn.Leg(config, -1, (-1,1), (1,1))
    aux_leg= yastn.Leg(config, -1, cs, Ds)
    a= yastn.rand(config, n=1, legs=(phys_leg,aux_leg,aux_leg,aux_leg.conj(),aux_leg.conj()))
    C_tr= yastn.rand(config, legs=(env_leg, env_leg.conj()))
    T_t= yastn.rand(config, legs=(env_leg, a.get_legs(1).conj(), a.get_legs(1), env_leg.conj()))
    T_r= yastn.rand(config, legs=(env_leg, a.get_legs(4).conj(), a.get_legs(4), env_leg.conj()))

    return a, C_tr, T_t, T_r


def enlarged_corner(a,C,T_t,T_r):
    # 
    # Contract the network
    #
    # a(0-)---T_t---(3+)A(0-)---C
    #        / |                |
    #      (1+)(2-)            (1+)
    #       C   E               B
    #       |   |              (0-)
    #  b----a---|--------D(1+)--T_r     
    #       | G |              /|
    #  c----|---a*-------F(2-)  |
    #       |   |              (3+)
    #       e   f               d
    #
    C2x2_tr= yastn.einsum('aCEA,AB,BDFd,GCbeD,GEcfF->abcdef', T_t,C,T_r,a,a.conj(), order='ABCDEFG')
    return C2x2_tr

def enlarged_corner_with_dl(a,C,T_t,T_r):
    # 
    # Contract the network. Here, we contract physical index of on-site tensors, 
    # creating a double-layer tensor with indices CEbcefDF, 
    # before attaching it to the environment tensors.
    #
    # a(0-)---T_t---(3+)A(0-)---C
    #        / |                |
    #      (1+)(2-)            (1+)
    #       C   E               B
    #       |   |              (0-)
    #  b----a---|--------D(1+)--T_r     
    #       | G |              /|
    #  c----|---a*-------F(2-)  |
    #       |   |              (3+)
    #       e   f               d
    #
    C2x2_tr= yastn.einsum('aCEA,AB,BDFd,GCbeD,GEcfF->abcdef', T_t,C,T_r,a,a.conj(), order='ABGCDEF')
    return C2x2_tr


def _log_meta_svd(*args,**kwargs):
    res= _meta_svd(*args,**kwargs) # meta, Ustruct, Usl, Sstruct, Ssl, Vstruct, Vsl
    logger.info(json.dumps( {"_meta_svd": {
        "meta": res[0],
        "Ustruct": res[1]._asdict(), 
        "Sstruct": res[3]._asdict(), 
        "Vstruct": res[5]._asdict()
    }} ))
    return res

@patch('yastn.tensor.linalg._meta_svd', wraps=_meta_svd,
       side_effect=_log_meta_svd)
def svd_enlarged_corner(C2x2, mocked_meta_svd):
    #
    # Perform svd of block-sparse matrix
    #
    assert len(C2x2.s)==2,"Not a (block-sparse) matrix"
    res= C2x2.svd()
    return res


def _log_meta_fuse_hard(*args,**kwargs):
    res= _meta_fuse_hard(*args,**kwargs)
    logger.info(json.dumps( {"_meta_fuse_hard": {
        "struct_new": res[0]._asdict(), 
        "slices_new": tuple(x._asdict() for x in res[1]),
        "meta_mrg": res[2],  "t_in": res[3], "D_in": res[4]
    }} ))
    return res

@patch('yastn.tensor._merging._meta_fuse_hard', wraps=_meta_fuse_hard,
       side_effect=_log_meta_fuse_hard)
def fuse_enlarged_corner(C2x2_tr, mocked_meta_fuse_hard):
    #
    # From block-sparse tensor to block-sparse matrix
    #
    # (0----C2x2  -> 0--C2x2 
    #  1----|   |       |
    #  2)---|___|       1
    #       | | |     
    #      (4 5 3)
    #
    res= C2x2_tr.fuse_legs(axes=((0,1,2),(3,4,5)))
    return res


def plain_fuse_hard(config,
    data, 
    order : Sequence[int], 
    struct : tuple[Sequence[Sequence[int]], Sequence[int]], 
    slices : Sequence[slice], 
    meta_mrg : Sequence[tuple[Sequence[int], slice, Sequence[int], Sequence[Sequence[int]], Sequence[int]]]
):
    """
    Full implementation at yastn/backend/[backend_np.py,backend_torch.py]

    data : 1D backend data array (source)
    order : desired order of indices
    struct : structure of new tensor (target) - charge sectors and their dimensions on each index and the total number of elements
    slices : location of individual blocks in underlying data array of target  
    meta_mrg :  t1 -> is an effective charge of the source block after fusion. I.e. t1==tn, hence
                      this source block will belong to the destination block tn
                slo -> specifies the location of the source block in data
                Do  -> shape of the source block
                Dscl-> list of slice data which specifies the location of the "transformed"
                        source block in the destination block tn
                Drsh-> the shape of the "transformed" source block in the destination block tn
    """
    # 1. get structure of fused tensor=target (from pre-computed metadata by _meta_fuse_hard) 
    meta_new = tuple((x, y, z["slcs"][0]) for x, y, z in zip(struct["t"], struct["D"], slices))

    # 2. 
    newdata = config.backend.zeros( (struct["size"],), dtype=config.default_dtype, device=config.default_dtype)

    # 3. place data from source into target
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
            assert tn == t1
            temp = newdata[slice(*sln)].reshape(Dn)
            for (_, slo, Do, Dslc, Drsh) in gr:
                slcs = tuple(slice(*x) for x in Dslc)
                temp[slcs] = data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)
    return newdata


def test_plain_fuse_hard(caplog):
    caplog.set_level(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(LOGFILE,mode = "w+"))
    
    config= yastn.make_config(sym='U1')
    a, C_tr, T_t, T_r= generate_tensors(config, [-2,0,2], [2,2,2])
    C2x2_tr= enlarged_corner(a, C_tr, T_t, T_r)
    C2x2_f= fuse_enlarged_corner(C2x2_tr)

    # reproduce fusion
    with open(LOGFILE) as f:
        for line in f: pass
        fusion_metadata=json.loads(line.strip())["_meta_fuse_hard"]

    order= list(i for i in range(len(C2x2_tr.s))) # no permutation
    data_f= plain_fuse_hard(config, C2x2_tr.data, order, 
                    fusion_metadata["struct_new"], fusion_metadata["slices_new"], fusion_metadata["meta_mrg"])
    assert np.allclose(C2x2_f.data, data_f)


def plain_svd(config, data, 
              meta : Sequence[tuple[slice,Sequence[int],slice,Sequence[int],slice,slice,Sequence[int]]], 
              sizes : Sequence[int]):
    """
    Full implementation at yastn/backend/[backend_np.py,backend_torch.py]

    data : 1D backend data array (source)
    meta :  sl -> slice with the source block 
            D  -> shape of the source block
            slU-> slice for target block in U
            DU -> shape of the traget block in U
            slS-> slice for target block in S
            slV-> slice for target block in Vh
            DV -> shape of the traget block in Vh
    sizes : sizes of the resulting 1D data arrays holding U,S, and Vh
    """
    Udata = config.backend.empty((sizes[0],), dtype=config.default_dtype, device=config.default_device)
    Sdata = config.backend.empty((sizes[1],), dtype="float64", device=config.default_device)
    Vhdata = config.backend.empty((sizes[2],), dtype=config.default_dtype, device=config.default_device)
    for (sl, D, slU, DU, slS, slV, DV) in meta:
        U, S, Vh = np.linalg.svd(data[slice(*sl)].reshape(D), compute_uv=True)
        Udata[slice(*slU)].reshape(DU)[:] = U
        Sdata[slice(*slS)] = S
        Vhdata[slice(*slV)].reshape(DV)[:] = Vh
    return Udata, Sdata, Vhdata


def test_plain_svd(caplog):
    caplog.set_level(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(LOGFILE,mode = "w+"))
    
    config= yastn.make_config(sym='U1')
    a, C_tr, T_t, T_r= generate_tensors(config, [-2,0,2], [2,2,2])
    C2x2_tr= enlarged_corner(a, C_tr, T_t, T_r)
    C2x2_f= fuse_enlarged_corner(C2x2_tr)
    U,S,Vh= svd_enlarged_corner(C2x2_f)

    # reproduce fusion
    with open(LOGFILE) as f:
        for line in f: pass
        svd_metadata=json.loads(line.strip())["_meta_svd"]

    data_U, data_S, data_Vh= plain_svd(config, C2x2_f.data, svd_metadata["meta"], 
                      [svd_metadata[x]["size"] for x in ["Ustruct","Sstruct", "Vstruct"]])
    assert np.allclose(U.data, data_U)
    assert np.allclose(S.data, data_S)
    assert np.allclose(Vh.data, data_Vh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", type=int, default=None, help='choose pre-computed test case')
    parser.add_argument('-cs', nargs='+', type=int, help='choose U(1) sectors for auxiliary leg of on-site tensor', required=False)
    parser.add_argument('-Ds', nargs='+', type=int, help='choose U(1) sector sizes for auxiliary leg of on-site tensor', required=False)
    parser.add_argument("-sym", type=str, default='U1', choices=['U1','U1xU1'])
    parser.add_argument("-backend", type=str, default='np', choices=['np', 'torch'])
    parser.add_argument("-device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("-max_seconds", type=int, default=3600)
    args = parser.parse_args()

    if args.backend == 'np':
        import yastn.backend.backend_np as backend
    elif args.backend == 'torch':
        import yastn.backend.backend_torch as backend
    config= yastn.make_config(sym=args.sym, backend=backend, default_device=args.device)

    if args.cs or args.Ds:
        assert args.cs and args.Ds and len(args.cs)==len(args.Ds),"Both cs and Ds, of the same length, have to be provided"
        assert args.sym=="U1","Expects only U(1) symmetry"

    a, C_tr, T_t, T_r= None, None, None, None
    if args.D:
        dirname = os.path.dirname(__file__)
        if args.sym=="U1":
            fpath= os.path.join(dirname,"ipeps_U1.json")
        elif args.sym=="U1xU1":
            fpath= os.path.join(dirname,"ipeps_U1xU1.json")
        a, C_tr, T_t, T_r= read_tensors(config, args.D, fpath)
    elif args.sym=="U1" and args.Ds and args.cs:
        a, C_tr, T_t, T_r= generate_tensors(config, args.cs, args.Ds)

    # timings
    C2x2_tr= enlarged_corner(a, C_tr, T_t, T_r)
    res=timeit.repeat(stmt='enlarged_corner(a, C_tr, T_t, T_r)', setup='pass', repeat=5, number=1, globals=globals())
    print(f"enlarged_corner t[s]  {res}")

    if args.sym=="U1xU1":
        C2x2_tr= enlarged_corner_with_dl(a, C_tr, T_t, T_r)
        res=timeit.repeat(stmt='enlarged_corner_with_dl(a, C_tr, T_t, T_r)', setup='pass', repeat=5, number=1, globals=globals())
        print(f"enlarged_corner_with_dl t[s]  {res}")

    C2x2_f= fuse_enlarged_corner(C2x2_tr)
    res=timeit.repeat(stmt='fuse_enlarged_corner(C2x2_tr)', setup='pass', repeat=5, number=1, globals=globals())
    print(f"fuse_enlarged_corner t[s]  {res}")

    U,S,V= svd_enlarged_corner(C2x2_f)
    res=timeit.repeat(stmt='svd_enlarged_corner(C2x2_f)', setup='pass', repeat=5, number=1, globals=globals())
    print(f"svd t[s]  {res}")