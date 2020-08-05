import numpy as np
import yamps.mps as mps
import yamps.mps.tdvp as tdvp
import transport_vectorization_full as main_full
import transport_vectorization_sym as main_sym
import transport_vectorization_general as general

# Imaginary time evolution of full Lindbladian


def test_tdvp_total(main, choice):
    # MODEL
    NL = 2
    w0 = 1
    v = .5
    wS = 0
    mu = .5
    dV = 0
    gamma = 1.  # dissipation rate
    temp = 0.  # temperature
    distribution = 1
    # how to arrange modes. 0-spatial 1d to energy modes via sine-transformation.
    # 1-uniform spacing through W=4*w0. Are not places on the very corners of W.
    ordered = False  # order basis for mps/mpo
    if choice == 'Z2':
        basis = 0
    elif choice == 'U1':
        basis = 1
        # what basis for spanning density and mpo. 0 - choose I, Z, X, Y basis
        # 1 - choose cp c, c cp, c, cp

    # MPS
    tol_svd = 1e-6
    D_total = 16
    io = [0.5]  # initial occupation on the impurity

    # algorithm
    dtype = 'complex128'
    sgn = 1j  # imaginary time evolution for sgn == 1j
    dt = .125 * sgn  # time step - time step for single tdvp
    tmax = 300. * dt * sgn  # total time
    opts_svd = {'tol': tol_svd, 'D_total': D_total}
    eigs_tol = 1e-14

    # STORAGE
    directory = 'tests/models/'
    name = directory+'testZ2_0'
    name_txt = name + '_output.txt'
    name_npy = name + '_output.npy'

    names = ['NL', 'w0', 'v', 'wS', 'mu', 'dV', 'gamma', 'temp', 'distribution',
             'ordered', 'basis', 'tol_svd', 'D_total', 'io', 'dtype', 'dt']
    values = [NL, w0, v, wS, mu, dV, gamma, temp, distribution,
              ordered, basis, tol_svd, D_total, io, dtype, dt]
    general.save_to_file(names, values, name_npy)

    # EXECUTE
    LSR, wk, temp, vk, dV, gamma = general.generate_discretization(
        NL=NL, w0=w0, wS=wS, mu=mu, v=v, dV=dV, tempL=temp, tempR=temp, method=distribution, ordered=ordered, gamma=gamma)
    psi = main.thermal_state(LSR=LSR, io=io, ww=wk,
                             temp=temp, basis=basis, dtype=dtype)
    LL, LdagL = main.Lindbladian_1AIM_mixed(
        NL=NL, LSR=LSR, wk=wk, temp=temp, vk=vk, dV=dV, gamma=gamma, basis=basis, AdagA=True, dtype=dtype)
    #H, hermitian, dmrg = LL, False, False
    H, hermitian, dmrg = LdagL, True, True

    # canonize MPS
    psi.normalize = True
    psi.canonize_sweep(to='last')
    psi.canonize_sweep(to='first')

    # compress MPO
    H.normalize = False
    H.canonize_sweep(to='last')
    H.canonize_sweep(to='first')

    # trace rho
    trace_rho = mps.env2.Env2(bra=main.identity(psi.N, basis=basis), ket=psi)

    # operator H exp val
    env = mps.env3.Env3(bra=psi, op=H, ket=psi)

    # current
    curr_LS = main.current(LSR, -4.*np.pi*vk, cut='LS', basis=basis)
    curr_SR = main.current(LSR, -4.*np.pi*vk, cut='SR', basis=basis)
    JLS = mps.env2.Env2(bra=curr_LS, ket=psi)
    JSR = mps.env2.Env2(bra=curr_SR, ket=psi)

    # Occupation
    occ_L = main.measure_sumOp(choice=-1, LSR=LSR, Op='nn', basis=basis)
    occ_S = main.measure_sumOp(choice=2, LSR=LSR, Op='nn', basis=basis)
    occ_R = main.measure_sumOp(choice=1, LSR=LSR, Op='nn', basis=basis)
    NL = mps.env2.Env2(bra=occ_L, ket=psi)
    NS = mps.env2.Env2(bra=occ_S, ket=psi)
    NR = mps.env2.Env2(bra=occ_R, ket=psi)

    OP_Nocc = [0.]*psi.N
    for n in range(psi.N):
        tmp = main.measure_Op(N=psi.N, id=n, Op='nn', basis=basis)
        OP_Nocc[n] = mps.env2.Env2(bra=tmp, ket=psi)

    # EVOLUTION
    qt = 0
    Dmax = max(psi.get_D())
    out = psi.measuring([env, NL, NS, NR, JLS, JSR], norm=trace_rho)
    E, nl, ns, nr, jls, jsr = out[0].real, out[1].real, out[2].real, out[3].real, out[4].imag, out[5].imag
    E *= trace_rho.measure().real
    print('Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, 'normalization: ', (trace_rho.measure().real), ' E = ', E, ' Tot_Occ= ',
          round(nl+ns+nr, 4), ' JLS=', jls, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5))

    Nocc = psi.measuring(OP_Nocc, norm=trace_rho)
    print([round(it.real, 5) for it in Nocc])
    with open(name_txt, 'a') as f:
        print(qt, E, jls, jsr, nl, ns, nr, file=f)

    init_steps = 5
    qt = 0
    while abs(qt) < abs(tmax):
        exp_tol = 1e-2  # tol_svd*.01
        if dmrg:
            ddt = dt
            env = mps.dmrg.dmrg_sweep_2site(
                psi=psi, H=H, env=env, eigs_tol=eigs_tol, dtype=dtype, hermitian=True,  opts_svd=opts_svd)
        else:
            for it in range(init_steps):
                exp_tol = 1e-14
                ddt = dt*2**(it-init_steps)
                if D_total > 1:
                    env = mps.tdvp.tdvp_sweep_2site(
                        psi=psi, H=H, env=env, dt=ddt, eigs_tol=eigs_tol, exp_tol=exp_tol,  dtype=dtype, hermitian=hermitian,  opts_svd=opts_svd)
                else:
                    env = mps.tdvp.tdvp_sweep_1site(
                        psi=psi, H=H, env=env, dt=ddt, eigs_tol=eigs_tol, exp_tol=exp_tol,  dtype=dtype, hermitian=hermitian,  opts_svd=opts_svd)
            else:
                ddt = dt
                env = mps.tdvp.tdvp_sweep_1site(psi=psi, H=H, env=env, dt=ddt, eigs_tol=eigs_tol,
                                                exp_tol=exp_tol,  dtype=dtype, hermitian=hermitian,  opts_svd=opts_svd)
        qt += abs(ddt)
        #
        Dmax = max(psi.get_D())
        out = psi.measuring(
            [env, NL, NS, NR, JLS, JSR], norm=trace_rho)
        E, nl, ns, nr, jls, jsr = out[0].real, out[1].real, out[2].real, out[3].real, out[4].real, out[5].real
        E *= trace_rho.measure().real
        print('Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, 'normalization: ', (trace_rho.measure().real), ' E = ', E, ' Tot_Occ= ',  round(
            nl+ns+nr, 4), ' JLS=', jls, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5))
        Nocc = psi.measuring(OP_Nocc, norm=trace_rho)
        print([round(it.real, 5) for it in Nocc])
        print(psi.get_D())
        with open(name_txt, 'a') as f:
            print(qt, E, jls, jsr, nl, ns, nr, file=f)


def test_tdvp_total_full(choice):
    test_tdvp_total(main_full, choice)


def test_tdvp_total_sym(choice):
    test_tdvp_total(main_sym, choice)


if __name__ == "__main__":
    # pass
    #test_tdvp_total_full('Z2');print()
    test_tdvp_total_full('U1');print()
    #test_tdvp_total_sym('Z2');print()
    #test_tdvp_total_sym('U1');print()
