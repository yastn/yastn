import numpy as np
from scipy import linalg as LA


def wdk_AS(Influence, Tscale, W, v, u, Fcoup, Frel):
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
    options = optimset(
        'TolX', 1e-6)  # %Tolerance on the optmization of Influence(wk)=scale

    # %Influence function. ALL influence functions are evaluated at frequencies w and assumed to take W (bandwidth) and u (bias) as arguments
    Int = lambda w: Influence(w, W, u)

    # %Total influence in the bias window
    Infu = LA.integrate(Int, 0, u/2, rtol=1e-12, atol=1e-14)
    # %Total influence outside the bias window
    Info = LA.integrate(Int, u/2, W/2, rtol=1e-12, atol=1e-14)

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

    #%Bias window modes of postive frequency:
    wli = 0  # %Lower boundary mode i, i=1, inside the bias window
    for cu in range(1, Nu):
        #%Numerical optimization of the upper boundary for mode i
            IntBin = lambda wui: abs(LA.integral(
                Int, wli, wui, rtol=1e-12, atol=1e-14)-scale)
            wui = fminbnd(IntBin, 0, u/2, options)

        wku[cu]=(wli+wui)/2    #%Mode placed at the midpoint between boundaries. Later try placing at max Influence(w) in the bin
        dku[cu]=wui-wli        #%Mode spacing
        wli=wui                #%Lower boundary for next mode
    wku=wku*(u/2)/sum(dku)     #%"Fix" numerical errors so that bias window modes exactly cover the bias window
    dku=dku*(u/2)/sum(dku)     #%"Fix" numerical errors so that bias window modes exactly cover the bias window

    #%Outside the bias window modes of postive frequency:
    wli=u/2  #%Lower boundary mode i, i=1, outside the bias window
    for co in range(1,No):
        #%Numerical optimization of the upper boundary for mode i
            IntBin=lambda wui: abs(LA.integral(Int,wli,wui,rtol=1e-12,atol=1e-14)-scale)
            wui=fminbnd(IntBin,u/2,W/2,options) #%The last mode can't reach Influence(w_k)=scale, but fminbnd will set W/2 as the upper bound wui
        
        wko[co]=(wli+wui)/2    #%Mode placed at the midpoint between boundaries. Later try placing at max Influence(w) in the bin
        dko[co]=wui-wli        #%Mode spacing
        wli=wui                #%Lower boundary for next mode
    wko=wko*(W/2-u/2)/sum(dko)     #%"Fix" numerical errors so that outside modes exactly cover the remaining spectrum
    dko=dko*(W/2-u/2)/sum(dko)     #%"Fix" numerical errors so that outside modes exactly cover the remaining spectrum

    #%Negative and positive frequency modes assuming particle-hole symmetry:
    wk=[-wko(No:-1:1), -wku(Nu:-1:1), wku, wko]
    dk=[dko(No:-1:1), dku(Nu:-1:1), dku, dko]

    N=length(wk)   #%Total number of modes

    """
    %**************************************************************************
    End placing modes according to Influence(wk)=scale ***********************
    %**************************************************************************
    """
    
    #%Relaxation strengths
    if Frel==1:
        gk=np.ones(1,N)*2*mean(dk(abs(wk)<u/2))
    else:
        gk=2*dk

    #%Coupling constants to the impurity
    if Fcoup==1:
        #%Coupling constants with a midpoint approximation
        vk=np.sqrt(4*v^2*dk./(np.pi*W)*np.sqrt(1-(2*wk/W)**2))
    else:
        #%Coupling constants (squared) integrated over each interval
        lbk=wk-dk/2    #%Lower bin boundaries
        ubk=wk+dk/2    #%Upper bin boundaries
        vk=v/np.sqrt(np.pi)*np.sqrt( 2*ubk*np.sqrt(1-4*ubk**2/W^2)/W - 2*lbk*np.sqrt(1-4*lbk**2/W^2)/W + acsc(W./(2*ubk)) - acsc(W./(2*lbk)) )
    return N, scale, wk, dk, vk, gk
