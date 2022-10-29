import argparse
import numpy as np
import yast, yamps
from yamps._generate import Hterm, generate_H1, generate_mpo

parser= argparse.ArgumentParser(description='',allow_abbrev=False)
parser.add_argument("--init_D", type=int, default=1, help="bond dimension")
parser.add_argument("--max_D", type=int, default=1, help="bond dimension")
parser.add_argument("--opt_max_iter", type=int, default=100, help="maximal number of epochs")
parser.add_argument("--eps_conv", type=float, default=1.0e-12, help="convergence criterion for DMRG")
parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
parser.add_argument("--N", type=int, default=2, help="Number of sites")
parser.add_argument("--theta", type=float, default=0., help="bilinear-biquadratic angle [in units of pi]")
parser.add_argument("--init_n", type=int, default=0, help="U(1) charge of the initial state")
args, unknown_args= parser.parse_known_args()

# 1) define function constructing 2-site term at position i,i+1 
#    of the Hamiltonian parametrized by the angle \theta [in units of \pi]
def h_2site(N,i,theta,J=1.0):
    assert i<N-1,"Invalid position for 2-site operator"

    # Define Hamiltonian as MPO
    #
    # 0) prepare algebra of Spin-1 operators with explicit U(1)-symmetry
    #    associated to S^z quantum number
    S1= yast.operators.Spin1(sym='U1')

    # 1.0) init generator
    G= yamps.Generator(N,S1)
    
    # 1.1) get identity MPO
    I= G.I()
    
    # 1.2) build local 2-site term.
    #      First build S_i.S_j term
    #
    #      0     1    
    #      S_i . S_j
    #      2     3
    #
    #      then square it
    #
    SiSj= yast.tensordot(S1.g(), S1.vec_s(), ([1],[0]))
    SiSj= yast.tensordot(S1.vec_s(), SiSj, ([0],[0])).transpose(axes=(0,2,1,3))
    SiSj2= yast.tensordot(SiSj,SiSj,([2,3],[0,1]))
    # 
    #      sum bilinear and biquadratic interactions
    #
    h2= J*(np.cos(theta*np.pi)*SiSj + np.sin(theta*np.pi)*SiSj2)

    # 1.3) perform SVD to split h2 into 2-site MPO
    #
    #      0     1     0                1
    #      S_i . S_j = Qi--2(-1) (+1)0--Rj
    #      2     3     1                2
    Qi,Rj= yast.linalg.qr(h2, axes=((0,2), (1,3)), sQ=-1)

    # 1.4) add extra legs to Qi and Rj to conform to MPO tensor
    #
    Qi= Qi.add_leg(axis=0,s=1)
    Rj= Rj.add_leg(axis=-1,s=-1)

    # replace identity tensors in the identity MPO
    I[i]= Qi
    I[i+1]= Rj
    return I

# 2) define function generating random MPS
#
def random_mps(N, charge, D, sigma=1):
    # 0) prepare algebra of Spin-1 operators with explicit U(1)-symmetry
    #    associated to S^z quantum number
    S1= yast.operators.Spin1(sym='U1')

    # 1.0) init generator
    G= yamps.Generator(N,S1)
    return G.random_mps(n=charge,D_total=D,sigma=sigma)

# 3) define function generating dimer MPS
#    as superposition of dimers on even and odd sublattices   
#
def dimer_mps(N):
    S1= yast.operators.Spin1(sym='U1')

    _A= yast.Tensor(config=S1.I().config, s=(1,1,-1), n=0)
    _B= _A.copy()
    _A.set_block(ts=(0,0,0), Ds=(1,1,1), val=1.)
    _N= _A.copy()
    _A.set_block(ts=(0,1,1), Ds=(1,1,1), val=1.)
    _A.set_block(ts=(0,-1,-1), Ds=(1,1,1), val=1.)
    _B.set_block(ts=(0,0,0), Ds=(1,1,1), val=-1.)
    _B.set_block(ts=(-1,1,0), Ds=(1,1,1), val=1.)
    _B.set_block(ts=(1,-1,0), Ds=(1,1,1), val=1.)
    psi_even= yamps.Mps(N)
    psi_odd= yamps.Mps(N)
    for i in range(N//2):
        psi_even[2*i]=_A.copy()
        psi_even[2*i+1]=_B.copy()
    if N%2==1: psi_even[N]=_N.copy()
    for i in range((N-1)//2):
        psi_odd[1+2*i]=_A.copy()
        psi_odd[2+2*i]=_B.copy()
    if N%2==0: 
        psi_odd[0]=_N.copy()
        psi_odd[N-1]=_N.copy()

    return psi_even+psi_odd

# 3) define MPOs for observables
#
def obs_ops(N):
    # 0) prepare algebra of Spin-1 operators with explicit U(1)-symmetry
    #    associated to S^z quantum number
    S1= yast.operators.Spin1(sym='U1')

    # 1.0) init generator
    G= yamps.Generator(N,S1)
    
    # 1.1) get identity MPO
    I= G.I()

    mpos_Sz= [generate_H1(I, Hterm(positions=(i,), operators=(S1.sz(),))) \
        for i in range(N)]
    
    def _gen_SSnn(i):
        return [
            Hterm(positions=(i,i+1), operators=(S1.sz(),S1.sz())),
            Hterm(amplitude=0.5, positions=(i,i+1), operators=(S1.sp(),S1.sm())),
            Hterm(amplitude=0.5, positions=(i,i+1), operators=(S1.sm(),S1.sp()))
            ]

    svd_opts= None
    mpos_SSnn= [generate_mpo(I, _gen_SSnn(i), svd_opts) for i in range(N-1)]

    return mpos_Sz, mpos_SSnn

def main():
    assert args.N>1,"Number of sites must be larger than 1."
    
    # 0) create all 2-site terms
    #
    H_2site_terms= [h_2site(args.N,i,args.theta) for i in range(args.N-1)]

    # 1) add them up and (losslessly) compress resulting MPO
    H= yamps.add(*H_2site_terms)
    H.canonize_sweep(to='last', normalize=False)
    H.truncate_sweep(to='first', opts={"tol": 1e-14}, normalize=False)

    # 2) define initial state
    # psi0 = random_mps(args.N, charge=args.init_n, D=args.init_D)
    psi0 = dimer_mps(args.N)
    psi_opt = psi0.copy() 

    # 3) define observables
    mpos_Sz, mpos_SSnn= obs_ops(args.N)
    total_Sz_qpi= yamps.add(*mpos_Sz, amplitudes=[(-1)**i for i in range(args.N)])
    total_Sz_qpi.canonize_sweep(to='last', normalize=False)
    total_Sz_qpi.truncate_sweep(to='first', opts={"tol": 1e-14}, normalize=False)
    
    def measure(sweep,psi,env3,E,disc_w):
        if (sweep+1)%1==0:
            _total_Sz_qpi= yamps.measure_mpo(psi, total_Sz_qpi, psi)
            print(f"{sweep} {E} {disc_w} {_total_Sz_qpi}")

    project=[]
    opts_svd={'D_total': args.max_D}
    env_psiHpsi, info = yamps.dmrg(psi_opt, H, project=project, version="2site",
        converge='energy', measure=measure, atol=args.eps_conv, max_sweeps=args.opt_max_iter, 
        opts_svd=opts_svd, return_info=True)

    total_E= env_psiHpsi.measure()

    print(f"E_total {total_E}")
    print(f"conv. info {info}")

    # 2) measure observables
    # 2.1) S^z profile
    print("\n\ni Sz S.S_nn")
    for i in range(args.N):
        _sz= yamps.measure_mpo(psi_opt, mpos_Sz[i], psi_opt)
        _SSnn= yamps.measure_mpo(psi_opt, mpos_SSnn[i], psi_opt) if i<args.N-1 else None
        print(f"{i} {_sz} {_SSnn}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()