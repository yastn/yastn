import numpy as np
from scipy.integrate import quadrature
from scipy import optimize as LO


def wdk_AS(Influence, Tscale, W, u, Fcoup, Frel):
    """
    %wdk_AS returns an array of frequencies, spacings, couplings, and
    %relaxation strengths for discrete reservoirs based on the measure
    %Influence(wk)=scale
    %   %wdk_AS generates N modes by *equal influence* according to the
    %   influence function Influence. The modes are placed with particle-hole
    %   symmetry (a symmetry assumed of Influence too) and no mode is placed at
    %   zero frequency. The modes are for one reservoir only. The modes are
    %   placed at the midpoint of the bin. Note that many of these assumptions
    %   are questionable (e.g., not necessarily optimal or the "best" - most
    %   stable, robust, smallest error).
    %
    %   The input variables are
    %      Influence: The name of the function that contains the continuum influence measure (assuming relaxation=0)
    %      Tscale: Target influence scale (note that "T" might be mistaken for temperature, and that is intentional)
    %      W: Bandwidth. Modes are generated from -W/2 to W/2 assuming particle-hole symmetry [wk(wk<0) == -wk(wk>0)]
    %      v: Coupling from L-S and S-R in real space
    %      u: Bias. Applied as u/2 in L and -u/2 in R
    %      Fcoup: Flag for coupling. Fcoup==1 gives midpoint approximation Fcoup~=1 gives integrated coupling
    %      Frel: Flag for relaxation. Frel==1 gives constant relaxation (2*<dk>_BW)
    %
    %   The output variables are
    %      N: The actual number of modes generated
    %      scale: The actual scale
    %      wk: The mode frequencies
    %      dk: The mode spacings
    %      vk: The mode couplings to the impurity
    %      gk: The mode relaxation strengths
    """
    xtol = 1e-6  # %Tolerance on the optmization of Influence(wk)=scale

    # %Influence function. ALL influence functions are evaluated at frequencies w and assumed to take W (bandwidth) and u (bias) as arguments
    def Inf(w): return Influence(w, W, u)
    # %Total influence in the bias window
    Infu, _ = quadrature(Inf, 0., u/2, rtol=1e-12, tol=1e-14)
    # %Total influence outside the bias window
    Info, _ = quadrature(Inf, u/2, W/2, rtol=1e-12, tol=1e-14)

    # %Total number of bias window modes, taken as np.ceiling of the non-integer number of modes needed to have Tscale influence
    Nu = np.ceil(Infu/Tscale)
    scale = Infu/Nu  # %The actual scale

    # %Total number of bias window modes, taken as np.ceiling of the non-integer number of modes needed to have scale influence. One ignores a minor boundary effect here (at the band edge)
    No = np.ceil(Info/scale)
    
    """
    %**************************************************************************
    Placing modes according to Influence(wk)=scale ***************************
    %**************************************************************************
    """
    Nu = int(Nu)
    dku = [0]*Nu
    wku = [0]*Nu
    #%Bias window modes of postive frequency:
    wli = 0.  # %Lower boundary mode i, i=1, inside the bias window
    for cu in range(Nu):
        #%Numerical optimization of the upper boundary for mode i
        def IntBin(wui): return abs(quadrature(
            Inf, wli, wui, rtol=1e-12, tol=1e-14)[0]-scale)
        # %The last mode can't reach Influence(w_k)=scale, but fminbnd will set W/2 as the upper bound wui
        wui = LO.fminbound(IntBin, 0., u/2, xtol=xtol)

        # %Mode placed at the midpoint between boundaries. Later try placing at max Influence(w) in the bin
        wku[cu] = (wli+wui)/2
        dku[cu] = wui-wli  # %Mode spacing
        wli = wui  # %Lower boundary for next mode
    # %"Fix" numerical errors so that bias window modes exactly cover the bias window
    wku = [it*(u/2)/sum(dku) for it in wku]
    # %"Fix" numerical errors so that bias window modes exactly cover the bias window
    dku = [it*(u/2)/sum(dku) for it in dku]

    No = int(No)
    dko = [0]*No
    wko = [0]*No
    #%Outside the bias window modes of postive frequency:
    wli = u/2  # %Lower boundary mode i, i=1, outside the bias window
    for co in range(No):
        #%Numerical optimization of the upper boundary for mode i
        # %The last mode can't reach Influence(w_k)=scale, but LO.fminbound will set W/2 as the upper bound wui
        def IntBin(wui): return abs(quadrature(
            Inf, wli, wui, rtol=1e-12, tol=1e-14)[0]-scale)
        # %The last mode can't reach Influence(w_k)=scale, but fminbnd will set W/2 as the upper bound wui
        wui = LO.fminbound(IntBin, u/2, W/2, xtol=xtol)

        # %Mode placed at the midpoint between boundaries. Later try placing at max Influence(w) in the bin
        wko[co] = (wli+wui)/2
        dko[co] = wui-wli  # %Mode spacing
        wli = wui  # %Lower boundary for next mode
    # %"Fix" numerical errors so that outside modes exactly cover the remaining spectrum
    wko = [it*(W/2-u/2)/sum(dko) for it in wko]
    # %"Fix" numerical errors so that outside modes exactly cover the remaining spectrum
    dko = [it*(W/2-u/2)/sum(dko) for it in dko]

    #%Negative and positive frequency modes assuming particle-hole symmetry:
    wk = np.array([-it for it in wko[::-1]] +
                  [-it for it in wku[::-1]] + wku + wko)
    dk = np.array(dko[::-1] + dku[::-1] + dku + dko)

    N = len(wk)  # %Total number of modes

    """
    %**************************************************************************
    End placing modes according to Influence(wk)=scale ***********************
    %**************************************************************************
    """

    #%Relaxation strengths
    if Frel == 1:
        gk = np.ones(N)*2*np.mean(dk[abs(wk) < u/2])
    else:
        gk = 2*dk

    #%Coupling constants to the impurity
    if Fcoup == 1:
        #%Coupling constants with a midpoint approximation
        vk = np.sqrt(4.*dk/(np.pi*W)*np.sqrt(1.-(2.*wk/W)**2))
    else:
        #%Coupling constants (squared) integrated over each interval
        lbk = wk-dk/2  # %Lower bin boundaries
        ubk = wk+dk/2  # %Upper bin boundaries
        vk = 1./np.sqrt(np.pi)*np.sqrt(2*ubk*np.sqrt(1-4*ubk**2/W ^ 2)/W - 2 *
                                       lbk*np.sqrt(1-4*lbk**2/W ^ 2)/W
                                       + np.arcsin((2*ubk)/W) - np.arcsin((2*lbk)/W))

    print('Modes in the bias window ',Nu,' and ', No, 'outside. Total: ', len(wk), '\nwku: ',wku,'\nwko: ',wko)

    return N, scale, wk, dk, vk, gk
