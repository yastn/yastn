import logging
import h5py
import time
import numpy as np
import yamps.mps as mps
import transport_vectorization as main_fun
import transport_vectorization_full as main_full
import transport_vectorization_Z2 as main_Z2
import transport_vectorization_U1 as main_U1
import transport_vectorization_general as general
#
import yamps.mps.measure as measure


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("models/transport_open/transport.log"),
        logging.StreamHandler()
    ])


def transport(main, basis, tensor, filename): # PRA MODEL
    # main - module to produce hamiltonian
    # choice - basis, Majorana or Dirac
    # tensor - tensor type: full, Z2 or U1
    #
    # MODEL
    NL = 3
    NS = 2
    wj = [1]*2
    vj = [[.5, 0], #L-S1, R-S1
         [0, .5]]  #L-S2, R-S2
    wS = [0, 0]
    muj = [-.25, .25]
    dVj = [0]*2
    gamma = 1.  # dissipation rate
    tempj = [.025]*2 # temperature
    discretization = 0
    vS = (1.+np.sqrt(2))*.25
    HS = np.array([[wS[0], vS], 
                   [vS, wS[1]]])
    # MPS
    ordered = True  # order basis for mps/mpo
    tensor_type = main.get_tensor_type(basis)
    tol_svd = 1e-6
    D_total = 32
    io = [.5, .5]  # initial occupation on the impurity

    # algorithm
    algorithm = 'arnoldi'
    dt = .125  # time step - time step for single tdvp
    tmax = 100.  # total time
    opts_svd = {'tol': tol_svd, 'D_total': D_total}

    # STORAGE
    directory = 'models/generalized_AIM/results/'
    name = directory+filename+'_Nj_' + str([NL, NS, NL]) + '_algorithm_'+algorithm
    name_txt = name+'_output.txt'

    # EXECUTE
    LSR, temp, dV, gamma, corr= general.generate_discretization(Nj=[NL, NS, NL], wj=wj, HS=HS, muj=muj, dVj=dVj, tempj=tempj, vj=vj, method=discretization, ordered=ordered, gamma=gamma)
    psi = main.thermal_state(tensor_type=tensor_type, LSR=LSR, io=io, ww=corr.diagonal()+dV, temp=temp, basis=basis)
    LL, LdagL = main.vectorized_Lindbladian(tensor_type=tensor_type, LSR=LSR, temp=temp, dV=dV, gamma=gamma, corr=corr, basis=basis, AdagA=True)

    # TDVP
    H, hermitian, dmrg, version, HH, sgn = LL, False, False, None, LL, +1.
    #H, hermitian, dmrg, version, HH, sgn = LdagL, True, False, None, LdagL, -1.
    # DMRG
    #H, hermitian, dmrg, version, HH = LL, False, True, None, LL
    #H, hermitian, dmrg, version, HH = LdagL, True, True, None, LdagL
    print('LSR', LSR)
    print('temp', temp)
    print('dV', dV)
    print('corr', corr, '\n')
    # canonize MPS
    psi.canonize_sweep(to='last')
    psi.canonize_sweep(to='first')

    # compress MPO
    compress = True
    if compress:
        H.canonize_sweep(to='last', normalize=False)
        H.sweep_truncate(to='first', opts={'tol': 1e-14}, normalize=False)
        H.canonize_sweep(to='last', normalize=False)

    # trace rho
    trace_rho = main.identity(tensor_type=tensor_type, N=psi.N, basis=basis)

    # current
    JLS=[None]*NS
    JSR=[None]*NS
    for n, mS in zip(range(NS), np.nonzero(LSR==2)[0]):
        print('n, mS', n, mS)
        tmp = np.zeros(len(LSR))
        tmp[np.nonzero(LSR==-1)[0]] = corr[mS, LSR==-1]
        print(tmp)
        JLS[n] = main.current(tensor_type=tensor_type, sign_from=-1, id_to=mS, sign_list=LSR, vk=-4.*np.pi*tmp, direction=+1, basis=basis)

        tmp *= 0.
        tmp[np.nonzero(LSR==1)[0]] = corr[mS, LSR==1]
        print(tmp)
        JSR[n] = main.current(tensor_type=tensor_type, sign_from=+1, id_to=mS, sign_list=LSR, vk=-4.*np.pi*tmp, direction=-1, basis=basis)

    tmp *= 0.
    tmp[np.nonzero(LSR==2)[0]] = corr[mS, LSR==2]
    print(tmp, mS)
    JSS = main.current(tensor_type=tensor_type, sign_from=2, id_to=mS, sign_list=LSR, vk=-4.*np.pi*tmp, direction=+1, basis=basis)

    # Occupation
    NL = main.measure_sumOp(tensor_type=tensor_type,
                            choice=-1, LSR=LSR, Op='nn', basis=basis)
    NS = main.measure_sumOp(tensor_type=tensor_type,
                            choice=2, LSR=LSR, Op='nn', basis=basis)
    NR = main.measure_sumOp(tensor_type=tensor_type,
                            choice=1, LSR=LSR, Op='nn', basis=basis)
    OP_Nocc = [0.]*psi.N
    for n in range(psi.N):
        OP_Nocc[n] = main.measure_Op(
            tensor_type=tensor_type, N=psi.N, id=n, Op='nn', basis=basis)

    # EVOLUTION
    env = None
    init_steps = [5, 5, 4, 3, 2, 1]
    qt = 0
    it_step = 0
    qt = 0

    max_steps = int((abs(tmax)/abs(dt)).real)+10
    cpu_time = np.zeros(max_steps)
    Es = np.zeros(max_steps)
    occ_L = np.zeros(max_steps)
    occ_S = np.zeros(max_steps)
    occ_R = np.zeros(max_steps)
    curr_LS = np.zeros(max_steps)
    curr_SS = np.zeros(max_steps)
    curr_SR = np.zeros(max_steps)
    Ds = np.zeros((max_steps, psi.N+1))
    occ = np.zeros((max_steps, psi.N))

    # MEASURE
    Dmax = max(psi.get_D())
    E, _ = general.measure_MPOs(psi, [HH])
    E = E[0].real
    out, norm = general.measure_overlaps(psi, [NL, NS, NR], norm=trace_rho)
    nl, ns, nr = out[0].real, out[1].real, out[2].real
    Nocc, _ = general.measure_overlaps(psi, OP_Nocc, norm=trace_rho)
    Nocc = Nocc.real
    out, _ = general.measure_overlaps(psi, JLS, norm=trace_rho)
    jls = sum(out.real)
    out, _ = general.measure_overlaps(psi, JSR, norm=trace_rho)
    jsr = sum(out.real)
    out, _ = general.measure_overlaps(psi, [JSS], norm=trace_rho)
    jss = sum(out.real)

    # SAVE
    start = time.time()
    cpu_time[it_step] = time.time()-start
    Es[it_step] = E
    occ_L[it_step] = nl
    occ_S[it_step] = ns
    occ_R[it_step] = nr
    curr_LS[it_step] = jls
    curr_SS[it_step] = jss
    curr_SR[it_step] = jsr
    Ds[it_step, :] = psi.get_D()
    occ[it_step, :] = Nocc

    # PRINT
    print('Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, ' E = ', E, ' N = ', norm, ' Tot_Occ= ', round(
        nl+ns+nr, 4), ' JLS=', jls, ' JSS=', jss, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5))
    with open(name_txt, 'a') as f:
        print(qt, 1, E, jls, jss, jsr, nl, ns, nr, cpu_time[it_step], file=f)
    # EXPORT
    export = {'Es': Es, 'cpu_time': cpu_time, 'occ_L': occ_L, 'occ_S': occ_S,
                'occ_R': occ_R, 'curr_LS': curr_LS, 'curr_SS': curr_SS, 'curr_SR': curr_SR, 'Ds': Ds, 'occ': occ}
    with h5py.File(name + '_output.h5', 'w') as f:
        f.create_group('measurements')
        for inm, ival in export.items():
            f['measurements'].create_dataset(inm, data=ival)
        main_fun.save_psi_to_h5py(f, psi)
    #
    while abs(qt) < abs(tmax):
        #
        version = '2site'
        eigs_tol = 1e-13
        if dmrg:
            ddt = dt
            _, _, _ = mps.dmrg.dmrg_OBC(psi=psi, H=LdagL, version=version, cutoff_sweep=1,
                                        eigs_tol=eigs_tol, hermitian=hermitian,  opts_svd=opts_svd)
        else:
            if it_step < len(init_steps):
                ddt = dt*2**(-init_steps[it_step])
                opts_svd['tol'] = tol_svd*.001
                opts_svd['D_total'] = int(1.5*D_total)
                exp_tol = 1e-14
            else:
                ddt = dt
                opts_svd['tol'] = tol_svd
                opts_svd['D_total'] = D_total
                exp_tol = tol_svd*.01
            env, _, _ = mps.tdvp.tdvp_OBC(psi=psi, tmax=sgn*ddt, dt=sgn*ddt, H=H, env=env, version=version,
                                          eigs_tol=eigs_tol, exp_tol=exp_tol, hermitian=hermitian,  opts_svd=opts_svd, algorithm=algorithm)
        qt += abs(ddt)

        # MEASURE
        Dmax = max(psi.get_D())
        E, _ = general.measure_MPOs(psi, [HH])
        E = E[0].real
        out, norm = general.measure_overlaps(psi, [NL, NS, NR], norm=trace_rho)
        nl, ns, nr = out[0].real, out[1].real, out[2].real
        Nocc, _ = general.measure_overlaps(psi, OP_Nocc, norm=trace_rho)
        Nocc = Nocc.real
        out, _ = general.measure_overlaps(psi, JLS, norm=trace_rho)
        jls = sum(out.real)
        out, _ = general.measure_overlaps(psi, JSR, norm=trace_rho)
        jsr = sum(out.real)
        out, _ = general.measure_overlaps(psi, [JSS], norm=trace_rho)
        jss = sum(out.real)

        # SAVE
        cpu_time[it_step] = time.time()-start
        Es[it_step] = E
        occ_L[it_step] = nl
        occ_S[it_step] = ns
        occ_R[it_step] = nr
        curr_LS[it_step] = jls
        curr_SS[it_step] = jss
        curr_SR[it_step] = jsr
        Ds[it_step, :] = psi.get_D()
        occ[it_step, :] = Nocc
        # PRINT
        print('Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, ' E = ', E, ' N = ', norm, ' Tot_Occ= ', round(
            nl+ns+nr, 4), ' JLS=', jls, ' JSS=', jss, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5))
        with open(name_txt, 'a') as f:
            print(qt, Dmax, E, jls, jss, jsr, nl, ns, nr, cpu_time[it_step], file=f)
        # EXPORT
        export = {'Es': Es, 'cpu_time': cpu_time, 'occ_L': occ_L, 'occ_S': occ_S,
                  'occ_R': occ_R, 'curr_LS': curr_LS, 'curr_SS': curr_SS, 'curr_SR': curr_SR, 'Ds': Ds, 'occ': occ}
        if it_step % 100 == 0:
            with h5py.File(name + '_output.h5', 'w') as f:
                f.create_group('measurements')
                for inm, ival in export.items():
                    f['measurements'].create_dataset(inm, data=ival)
                main_fun.save_psi_to_h5py(f, psi)
        it_step += 1
    #
    with h5py.File(name + '_output.h5', 'w') as f:
        f.create_group('measurements')
        for inm, ival in export.items():
            f['measurements'].create_dataset(inm, data=ival)
        main_fun.save_psi_to_h5py(f, psi)


def transport_full(choice):
    transport(main_full, choice, 'full', 'test_full_'+choice)


def transport_Z2(choice):
    if choice == 'Majorana':
        transport(main_Z2, choice, 'Z2', 'test_Z2_'+choice)
    elif choice == 'Dirac':
        print('Z2/Dirac - Option is not implemented.')
        pass


def transport_U1(choice):
    if choice == 'Dirac':
        transport(main_U1, choice, 'U1', 'test_U1_'+choice)
    elif choice == 'Majorana':
        print('U1/Majorana - Option is not implemented.')
        pass


if __name__ == "__main__":
    # pass
    #transport_full('Majorana')
    #transport_full('Dirac')
    #transport_Z2('Majorana')
    transport_U1('Dirac')
