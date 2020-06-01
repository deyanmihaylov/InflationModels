import numpy as np
import pygsl.odeiv as odeiv
import pygsl.spline as spline

from calcpath import *

knos = 1575 # total number of k-values to evaluate
kinos = 214 # total number of k-values to use for integration
k_file = "ks_eval.dat" # file containing k-values at which to evaluate spectrum
ki_file = "ks.dat" # file containing k-values for integration
Y = 50 # Y = value of k/aH at which to initialize mode fcns
VERYSMALLNUM = 1.E-18

class params:
    def __init__(self):
        self.a_init = None # initial val of the scale factor
        self.k = None # comoving wavenumber
        self.eps = None
        self.sig = None
        self.H = None
        self.xi = None


def spectrum(y_final, y, u_s, u_t, N, derivs1, scalarsys, tensorsys):
    i = None

    h = 0.01
    h2 = 1.e-6 # init step size for mode integration

    abserr1 = 1.e-8 # absolute error tolerance - DO NOT ADJUST THESE VALUES!
    relerr1 = 1.e-8 # relative error tolerance

    abserr2 = 1e-10 # absolute error tolerance
    relerr2 = 1e-10 # relative error tolerance

    spec_params = params()

    # Read in k files
    k = None
    ks = np.empty(knos)
    kis = np.empty(kinos)

    try:
        ks = np.loadtxt(k_file)
    except IOError as e:
        print(f"Could not open file {k_file}, errno = {e}.")
        sys.exit()
        
    try:
        kis = np.loadtxt(ki_file)
    except IOError as e:
        print(f"Could not open file {ki_file}, errno = {e}.")
        sys.exit()

    realu_init = np.empty(2)
    imu_init = np.empty(2)

    realu_s = np.empty(kmax)
    realu_t = np.empty(kmax)

    imu_s = np.empty(kmax)
    imu_t = np.empty(kmax)

    P_s = np.empty(kinos)
    P_t = np.empty(kinos)

    j = None
    l = None
    m = None
    o = None
    status = None

    countback = 0
    count = 0

    ydoub = np.empty(NEQS)

    Ninit = None # N_obs from flow integration
    Nfinal = None # Smallest N value from flow integration

    spec_norm = None

    ru_init = None
    dru_init = None

    iu_init = None
    diu_init = None

    nu = None
    Yeff = None
    Phi = None

    # Buffers for interpolations
    Nefoldsback = np.empty(kmax)
    
    flowback = np.empty((5,kmax))

    Nordered = np.empty(kmax)
    uordered_s = np.empty(kmax)
    uordered_t = np.empty(kmax)

    """
    Initialize/allocate gsl stepper routines and variable
    step-size routines.  Define ode system.
    """
    s = odeiv.step_rk4(NEQS, derivs1)
    c = odeiv.control_y_new(s, abserr1, relerr1)
    e = odeiv.evolve(s, c, NEQS)

    """
    Set the initial value of the scale factor.  This is chosen
    so that k = aH (with k corresponding to the quadrupole) at the
    value N_obs from the path file.  The scale factor as a 
    function of N is a(N) = a_init*exp(-# of efolds).
    Units are hM_PL
    """
    Ninit = N
    spec_params.a_init = (1.73e-61/y[1]) * np.exp(Ninit)
    spec_params.k = k

    """
    To improve stability/efficiency, we first generate
    an interpolating function for H, epsilon, sigma and xi^2.  We then pass these values
    as parameters to the mode equation, rather than solving the mode equation along with
    the full set of flow equations each time.
    """

    """
    Integrate backwards from end of inflation to the earliest time needed in order to initialize the
    largest scale fluctuations in the BD limlt.
    """

    ydoub[:] = y_final[:NEQS].copy()
    N = y_final[NEQS]
    Nfinal = N

    while (kis[0]*5.41e-58) / (spec_params.a_init*np.exp(-N)*ydoub[1]) < Y:
        flowback[:, countback] = ydoub[:5].copy()

        Nefoldsback[countback] = N

        try:
            N, h2, ydoub = e.apply(N, 1000, h2, ydoub)
        except:
            status = 0
            return status

        countback += 1

    Nefoldsback[countback] = N

    flowback[:, countback] = ydoub[:5].copy()

    H = np.empty(countback+1)
    eps = np.empty(countback+1)
    sig = np.empty(countback+1)
    xi = np.empty(countback+1)
    # Nefolds = np.empty(kmax)
    Nefolds = np.empty(countback+1)
    phi = np.empty(countback+1)

    H[:] = flowback[1, :countback+1].copy()
    eps[:] = flowback[2, :countback+1].copy()
    sig[:] = flowback[3, :countback+1].copy()
    xi[:] = flowback[4, :countback+1].copy()
    phi[:] = flowback[0, :countback+1].copy()
    Nefolds[:] = Nefoldsback[:countback+1].copy()

    # Generate interpolating functions for H, eps, sig, xi and phi (for path gen. only)
    spline1 = spline.cspline(countback+1)
    spline1.init(Nefolds, H)

    spline2 = spline.cspline(countback+1)
    spline2.init(Nefolds, eps)

    spline3 = spline.cspline(countback+1)
    spline3.init(Nefolds, sig)

    spline4 = spline.cspline(countback+1)
    spline4.init(Nefolds, xi)

    spline0 = spline.cspline(countback+1)
    spline0.init(Nefolds, phi)
    
    h2 = -h2

    """
    Find scalar spectra first.
    """
    for m in range(kinos):
        kis[m] *= 5.41e-58 # converts to Planck from hMpc^-1
        N = Ninit
        ydoub[1] = spline1.eval(N)
        ydoub[2] = spline2.eval(N)
        count = 0

        """
        First, check to see if the given k value is in the
        Bunch-Davies limit at the start of inflation.  This limit is
        set by the #define Y=k/aH.  If the given k value yields a
        larger Y than the BD limit, then we must integrate forward
        (to smaller N) until we reach the proper value for Y.  If it is
        smaller, we must integrate backwards (to larger N).  These
        integrators are given a fixed stepsize to ensure that we don't
        inadvertently step too far beyond Y.
        """

        if k/1.73e-61 > Y: # 1.73e-61 is the present Hubble radius (~3.2e-4 hMpc^-1) in Planck units
            while k / (spec_params.a_init*np.exp(-N)*ydoub[1]*(1-ydoub[2])) > Y:
                N += -0.01
                ydoub[1] = spline1.eval(N)
                ydoub[2] = spline2.eval(N)
        else
            while k / (spec_params.a_init*np.exp(-N)*ydoub[1]*(1-ydoub[2])) < Y:
                N += 0.01
                ydoub[1] = spline1.eval(N)
                ydoub[2] = spline2.eval(N)

        spec_params.k = k
        nu = (3.-spline2.eval(N)) / (2.*(1.-spline2.eval(N)))
        Yeff = k / (spec_params.a_init*(np.exp(-N)*(spline1.eval(N)*(1.-spline2.eval(N)))))

        






































def derivs1(t, y, dydN):
    dydN = np.zeros(NEQS, dtype=float, order='C')
    
    if y[2] > VERYSMALLNUM:
        dydN[0]= - np.sqrt(y[2]/(4*np.pi))
    else:
        dydN[0] = 0.

    dydN[1] = y[1] * y[2]
    dydN[2] = y[2] * (y[3]+2.*y[2])
    dydN[3] = 2.*y[4] - 5.*y[2]*y[3] - 12.*y[2]*y[2]
    
    for i in range(4, NEQS-1):
         dydN[i] = (0.5*(i-3)*y[3]+(i-4)*y[2])*y[i] + y[i+1]

    dydN[NEQS-1] = (0.5*(NEQS-4)*y[3]+(NEQS-5)*y[2]) * y[NEQS-1]

    return dydN

def scalarsys(t, y, dydN, parameters):
    p = params()
    p = parameters

    dydN[0] = y[1]
    dydN[1] = (1-p.eps)*y[1] - (((p.k)*(p.k))/((p.a_init)*(p.a_init)*np.exp(-2.*t)*(p.H)*(p.H))-2.*(1.-2.*(p.eps)-0.75*(p.sig) - (p.eps)*(p.eps) + 0.125*(p.sig)*(p.sig) + 0.5*(p.xi)))*y[0]

def tensorsys(t, y, dydN, parameters):
    p = params()
    p = parameters

    dydN[0] = y[1]
    dydN[1] = (1-p.eps)*y[1] - (((p.k)*(p.k))/((p.a_init)*(p.a_init)*np.exp(-2.*t)*(p.H)*(p.H))-(2.-p.eps))*y[0]







    























