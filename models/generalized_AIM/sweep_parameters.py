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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transport.log"),
        logging.StreamHandler()
    ])


def transport(main, basis, tensor, filename):  # PRA MODEL
    # main - module to produce hamiltonian
    # choice - basis, Majorana or Dirac
    # tensor - tensor type: full, Z2 or U1
    #
    # MODEL
    NL = 3
    NS = 2
    discretization = 0
    wj = [1]*2
    vj = [[.5, 0],  # L-S1, R-S1
          [0, .5]]  # L-S2, R-S2
    wS = [0, 0]
    muj = [-.25, .25]
    dVj = [0]*2
    gamma_val = [1.]#10**np.arange(2, -4.0001, -.25)  # dissipation rate
    tempj = [.025]*2  # temperature
    vS = (1.+np.sqrt(2))*.25
    HS = np.array([[wS[0], vS],
                   [vS, wS[1]]])

    # MPS
    ordered = True  # order basis for mps/mpo
    tensor_type = main.get_tensor_type(basis)
    tol_svd = 1e-6
    D_total_val = [32]
    io = [.5, .5]  # initial occupation on the impurity

    # algorithm
    alpha = 1  # calculate von neumann entropy
    algorithm = 'arnoldi'
    dt0 = .125  # base dt, might be rescaled to gamma
    section_time0 = 200
    opts_svd = {'tol': 0, 'D_total': 0}
    eigs_tol = 1e-13

    # STORAGE
    directory = 'models/generalized_AIM/results/'
    name = directory+filename+'_Nj_' + str([NL, NS, NL]) + \
        '_gamma_'+str(min(gamma_val)) +\
        '_Dtot_'+str(max(D_total_val))+'_algorithm_'+algorithm
    name_txt = name+'_output.txt'

    # SAVE information about simulation
    names = ['Nj', 'wj', 'vj', 'LSR', 'corr' 'muj', 'dVj', 'gamma_val', 'tempj', 'discretization',
             'ordered', 'basis', 'tol_svd', 'D_total_val', 'io', 'sym', 'dtype', 'dt0']
    values = [[NL, NS, NL], wj, vj, HS, muj, dVj, gamma_val, tempj, discretization,
              ordered, basis, tol_svd, D_total_val, io, tensor_type[0], tensor_type[1], dt0]
    with h5py.File(name + '_output.h5', 'w') as f:
        f.create_group('measurements/')
        f.create_group('measurements/sweep/')
        f.create_group('parameters')
        for inm, ival in zip(names, values):
            f['parameters'].create_dataset(inm, data=ival)

    # Initial setup
    LSR, temp, dV, gamma, corr = general.generate_discretization(
        Nj=[NL, NS, NL], wj=wj, HS=HS, muj=muj, dVj=dVj, tempj=tempj, vj=vj, method=discretization, ordered=ordered, gamma=gamma_val[0])
    psi = main.thermal_state(tensor_type=tensor_type, LSR=LSR,
                             io=io, ww=corr.diagonal()+dV, temp=temp, basis=basis)

    # canonize MPS
    psi.canonize_sweep(to='last')
    psi.canonize_sweep(to='first')

    # compress MPO
    compress = True

    # trace rho
    trace_rho = main.identity(tensor_type=tensor_type, N=psi.N, basis=basis)

    # current
    compress_J = False
    JLS = [None]*NS
    JSR = [None]*NS
    for n, mS in zip(range(NS), np.nonzero(LSR == 2)[0]):
        tmp = np.zeros(len(LSR))
        tmp[np.nonzero(LSR == -1)[0]] = corr[mS, LSR == -1]
        JLS[n] = main.current(tensor_type=tensor_type, sign_from=-1, id_to=mS,
                              sign_list=LSR, vk=-4.*np.pi*tmp, direction=+1, basis=basis)

        tmp *= 0.
        tmp[np.nonzero(LSR == 1)[0]] = corr[mS, LSR == 1]
        JSR[n] = main.current(tensor_type=tensor_type, sign_from=+1, id_to=mS,
                              sign_list=LSR, vk=-4.*np.pi*tmp, direction=-1, basis=basis)
        if compress_J:
            JLS[n].canonize_sweep(to='last', normalize=False)
            JLS[n].sweep_truncate(to='first', opts={
                'tol': 1e-14}, normalize=False)
            JSR[n].canonize_sweep(to='last', normalize=False)
            JSR[n].sweep_truncate(to='first', opts={
                'tol': 1e-14}, normalize=False)

    tmp *= 0.
    tmp[np.nonzero(LSR == 2)[0]] = corr[mS, LSR == 2]
    JSS = main.current(tensor_type=tensor_type, sign_from=2, id_to=mS,
                       sign_list=LSR, vk=-4.*np.pi*tmp, direction=+1, basis=basis)
    if compress_J:
        JSS.canonize_sweep(to='last', normalize=False)
        JSS.sweep_truncate(to='first', opts={
            'tol': 1e-14}, normalize=False)

    # Occupation
    NL = main.measure_sumOp(tensor_type=tensor_type,
                            choice=-1, LSR=LSR, Op='nn', basis=basis)
    NS = main.measure_sumOp(tensor_type=tensor_type,
                            choice=2, LSR=LSR, Op='nn', basis=basis)
    NR = main.measure_sumOp(tensor_type=tensor_type,
                            choice=1, LSR=LSR, Op='nn', basis=basis)
    """
    OP_Nocc = [0.]*psi.N
    for n in range(psi.N):
        OP_Nocc[n] = main.measure_Op(
            tensor_type=tensor_type, N=psi.N, id=n, Op='nn', basis=basis)
    """

    # EVOLUTION
    env = None
    init_steps = [5, 5, 4, 3, 2, 1]
    qt = 0
    it_step = 0
    qt = 0

    max_steps = int(len(D_total_val)*len(gamma_val)*int(section_time0/dt0)+10)
    cpu_time = np.zeros(max_steps)
    gamma_data = np.zeros(max_steps)
    D_total_data = np.zeros(max_steps)
    Es = np.zeros(max_steps)
    occ_L = np.zeros(max_steps)
    occ_S = np.zeros(max_steps)
    occ_R = np.zeros(max_steps)
    curr_LS = np.zeros(max_steps)
    curr_SS = np.zeros(max_steps)
    curr_SR = np.zeros(max_steps)
    Ds = np.zeros((max_steps, psi.N+1))
    occ = np.zeros((max_steps, psi.N))
    entropy = np.zeros((max_steps, psi.N))
    SV_min = np.zeros((max_steps, psi.N))
    start = time.time()
    Smin = np.ones(psi.N)
    for gmm in gamma_val:
        dt = min([1./gmm, dt0])  # time step - time step for single tdvp
        section_time = section_time0*dt/dt0
        #
        LL, _ = main.vectorized_Lindbladian(
            tensor_type=tensor_type, LSR=LSR, temp=temp, dV=dV, gamma=gamma*gmm/gamma_val[0], corr=corr, basis=basis)
        H, hermitian, HH, sgn = LL, None, LL, +1.
        if compress:
            H.canonize_sweep(to='last', normalize=False)
            H.sweep_truncate(to='first', opts={
                                'tol': 1e-14}, normalize=False)
            H.canonize_sweep(to='last', normalize=False)
        for D_total in D_total_val:
            qt_section = 0
            while qt_section < section_time:
                #
                if it_step < len(init_steps):
                    opts_svd['D_total'] = int(1.5*D_total)
                    ddt = dt*2**(-init_steps[it_step])
                    opts_svd['tol'] = 1e-14
                    exp_tol = 1e-14
                    version = '2site'
                else:
                    opts_svd['D_total'] = D_total
                    ddt = dt
                    opts_svd['tol'] = tol_svd
                    exp_tol = tol_svd*.01
                    version = 'mix'
                env, _, _ = mps.tdvp.tdvp_OBC(psi=psi, tmax=sgn*ddt, dt=sgn*ddt, H=H, env=env, version=version, SV_min = Smin,
                                                eigs_tol=eigs_tol, exp_tol=exp_tol, hermitian=hermitian,  opts_svd=opts_svd, algorithm=algorithm)
                qt += abs(ddt)
                qt_section += abs(ddt)

                # MEASURE
                virt_dims, Schmidt_spectrum, Smin, Hs = psi.get_S(alpha)
                Dmax = max(virt_dims)
                E, _ = general.measure_MPOs(psi, [HH])
                E = E[0].real
                out, norm = general.measure_overlaps(
                    psi, [NL, NS, NR], norm=trace_rho)
                nl, ns, nr = out[0].real, out[1].real, out[2].real
                #Nocc, _ = general.measure_overlaps(psi, OP_Nocc, norm=trace_rho)
                #Nocc = Nocc.real
                out, _ = general.measure_overlaps(psi, JLS, norm=trace_rho)
                jls = sum(out.real)
                out, _ = general.measure_overlaps(psi, JSR, norm=trace_rho)
                jsr = sum(out.real)
                out, _ = general.measure_overlaps(
                    psi, [JSS], norm=trace_rho)
                jss = sum(out.real)

                # SAVE
                cpu_time[it_step] = time.time()-start
                gamma_data[it_step] = gmm
                D_total_data[it_step] = D_total
                Es[it_step] = E
                occ_L[it_step] = nl
                occ_S[it_step] = ns
                occ_R[it_step] = nr
                curr_LS[it_step] = jls
                curr_SS[it_step] = jss
                curr_SR[it_step] = jsr
                entropy[it_step, :] = Hs
                SV_min[it_step, :] = Smin
                Ds[it_step, :] = virt_dims
                #occ[it_step, :] = Nocc

                # PRINT
                print('Iteration:', it_step, 'Time: ', round(abs(qt), 4), ' Dmax: ', Dmax, ' E = ', E, ' N = ', norm, ' Tot_Occ= ', round(
                    nl+ns+nr, 4), ' JLS=', jls, ' JSS=', jss, ' JSR=', jsr, ' NL=', round(nl, 5), ' NS=', round(ns, 5), ' NR=', round(nr, 5), ' max_Entropy=', max(Hs))
                with open(name_txt, 'a') as f:
                    print(qt, Dmax, E, jls, jss, jsr, nl, ns,
                            nr, max(Hs), cpu_time[it_step], gmm, D_total, file=f)

                # EXPORT
                export = {'Es': Es, 'cpu_time': cpu_time, 'gamma_data': gamma_data, 'D_total_data': D_total_data, 'occ_L': occ_L, 'occ_S': occ_S,
                            'occ_R': occ_R, 'curr_LS': curr_LS, 'curr_SS': curr_SS,
                            'curr_SR': curr_SR, 'SV_min': SV_min, 'entropy': entropy, 'Ds': Ds, 'occ': occ}
                if abs(qt_section) % abs(section_time/2) < dt and abs(qt/dt) > 1:
                    print('Exporting...')
                    with h5py.File(name + '_output.h5', 'a') as f:
                        try:
                            del f['measurements/sweep/'+'SV_spectrum_Dtot_' +
                                    str(D_total)+'_gamma_'+str(gmm)]
                        except:
                            pass
                        f['measurements/sweep/'].create_dataset('SV_spectrum_Dtot_'+str(
                            D_total)+'_gamma_'+str(gmm), data=Schmidt_spectrum)
                        for inm, ival in export.items():
                            try:
                                del f['measurements/'+inm]
                            except:
                                pass
                            f.create_dataset(
                                'measurements/'+inm, data=ival)
                it_step += 1
            # save Schmidt spectrum
            with h5py.File(name + '_output.h5', 'a') as f:
                try:
                    del f['measurements/sweep/'+'SV_spectrum_Dtot_' +
                            str(D_total)+'_gamma_'+str(gmm)]
                except:
                    pass
                f['measurements/sweep/'].create_dataset('SV_spectrum_Dtot_'+str(
                    D_total)+'_gamma_'+str(gmm), data=Schmidt_spectrum)
                main_fun.save_psi_to_h5py(f, psi)
                for inm, ival in export.items():
                    try:
                        del f['measurements/'+inm]
                    except:
                        pass
                    f.create_dataset('measurements/'+inm, data=ival)


def transport_full(choice):
    transport(main_full, choice, 'full', 'sweep_param_full_'+choice)


def transport_Z2(choice):
    if choice == 'Majorana':
        transport(main_Z2, choice, 'Z2', 'sweep_param_Z2_'+choice)
    elif choice == 'Dirac':
        print('Z2/Dirac - Option is not implemented.')
        pass


def transport_U1(choice):
    if choice == 'Dirac':
        transport(main_U1, choice, 'U1', 'sweep_param_U1_'+choice)
    elif choice == 'Majorana':
        print('U1/Majorana - Option is not implemented.')
        pass


if __name__ == "__main__":
    # pass
    transport_full('Majorana')
    #transport_full('Dirac')
    #transport_Z2('Majorana')
    #transport_U1('Dirac')
