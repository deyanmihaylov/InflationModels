import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import *

import pygsl.odeiv as odeiv
import pygsl.spline as spline
from pygsl.testing import _ufuncs

from calcpath import *

knos = 1575 # total number of k-values to evaluate
kinos = 214 # total number of k-values to use for integration

k_file = "ks_eval.dat" # file containing k-values at which to evaluate spectrum
ki_file = "ks.dat" # file containing k-values for integration

Y = 50 # Y = value of k/aH at which to initialize mode fcns
knorm = 0.05 # normalization scale
Amp = 2.0803249e-9 # scalar amplitude at knorm

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
    # i = None

    h = 1e-2
    h2 = 1e-6 # init step size for mode integration

    abserr1 = 1e-8 # absolute error tolerance - DO NOT ADJUST THESE VALUES!
    relerr1 = 1e-8 # relative error tolerance

    abserr2 = 1e-10 # absolute error tolerance
    relerr2 = 1e-10 # relative error tolerance

    spec_params = params()

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

    countback = 0
    count = 0

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
    ydoub = y_final[:NEQS].copy()
    N = y_final[NEQS]
    Nfinal = N

    def event1(t, y):
        return (kis[0]*5.41e-58) / (spec_params.a_init*np.exp(-t)*y[1]) - Y

    event1.terminal = True
    event1.direction = 1.

    e = solve_ivp(
        derivs1,
        (N, 1000),
        ydoub,
        method='RK45',
        first_step=h2,
        rtol=abserr1,
        atol=relerr1,
        events=event1,
    )

    status = e.status
    flowback = e.y
    Nefoldsback = e.t

    phi = flowback[0, :].copy()
    H = flowback[1, :].copy()
    eps = flowback[2, :].copy()
    sig = flowback[3, :].copy()
    xi = flowback[4, :].copy()
    Nefolds = Nefoldsback.copy()

    spline0 = CubicSpline(Nefolds, phi)
    spline1 = CubicSpline(Nefolds, H)
    spline2 = CubicSpline(Nefolds, eps)
    spline3 = CubicSpline(Nefolds, sig)
    spline4 = CubicSpline(Nefolds, xi)
    
    """
    Find scalar spectra first.
    """
    for m in range(kinos):
        print(m)

        k = kis[m] * 5.41e-58 # converts to Planck from hMpc^-1
        kis[m] = k
        N = Ninit

        ydoub[1] = spline1(N)
        ydoub[2] = spline2(N)

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
                ydoub[1] = spline1(N)
                ydoub[2] = spline2(N)
        else:
            while k / (spec_params.a_init*np.exp(-N)*ydoub[1]*(1-ydoub[2])) < Y:
                N += 0.01
                ydoub[1] = spline1(N)
                ydoub[2] = spline2(N)

        spec_params.k = k
        nu = (3-spline2(N)) / (2*(1-spline2(N)))
        
        Yeff = k / (spec_params.a_init*(np.exp(-N)*(spline1(N)*(1-spline2(N)))))

        if spline2(N) < 1:
            ru_init = realu_init[0] = 0.5 * np.sqrt(np.pi/k) * np.sqrt(Yeff) * jv(nu, Yeff)
            dru_init = realu_init[1] = 0.5 * np.sqrt(np.pi/k) * (k/(spec_params.a_init*np.exp(-N)*spline1(N))) * (jv(nu, Yeff)/(2.*np.sqrt(Yeff))+(np.sqrt(Yeff)*(-jv(nu+1, Yeff)+(nu*(1-spline2(N))*jv(nu, Yeff))/(Yeff*(1-spline2(N))))))
            iu_init = imu_init[0] = 0.5 * np.sqrt(np.pi/k) * np.sqrt(Yeff) * yn(nu, Yeff)
            diu_init = imu_init[1] = 0.5 * np.sqrt(np.pi/k) * (k/(spec_params.a_init*np.exp(-N)*spline1(N))) * (yn(nu, Yeff)/(2.*np.sqrt(Yeff))+(np.sqrt(Yeff)*(-yn(nu+1, Yeff)+(nu*(1-spline2(N))*yn(nu, Yeff))/(Yeff*(1-spline2(N))))))
        else:
            ru_init = realu_init[0] = -0.5 * np.sqrt(np.pi/k) * np.sqrt(Yeff) * yn(nu, Yeff)
            dru_init = realu_init[1] = -0.5 * np.sqrt(np.pi/k) * (k/(spec_params.a_init*np.exp(-N)*spline1(N))) * (yn(nu, Yeff)/(2.*np.sqrt(Yeff))+(np.sqrt(Yeff)*(-yn(nu+1, Yeff)+(nu*(1-spline2(N))*yn(nu, Yeff))/(Yeff*(1-spline2(N))))))
            iu_init = imu_init[0] = 0.5 * np.sqrt(np.pi/k) * np.sqrt(Yeff) * jn(nu, Yeff)
            diu_init = imu_init[1] = 0.5 * np.sqrt(np.pi/k) * (k/(spec_params.a_init*np.exp(-N)*spline1(N))) * (jn(nu, Yeff)/(2.*np.sqrt(Yeff))+(np.sqrt(Yeff)*(-jn(nu+1, Yeff)+(nu*(1-spline2(N))*jn(nu, Yeff))/(Yeff*(1-spline2(N))))))

        """
        Solve for real part of u first.
        """
        def event_s(t, y):
            return t - Nfinal

        event_s.terminal = True
        event_s.direction = -1.

        e2 = solve_ivp(
            scalarsys,
            (N, 0),
            realu_init,
            args=spec_params,
            method='RK45',
            first_step=h2,
            rtol=abserr2,
            atol=relerr2,
            events=event_s,
        )

        realu_s = (e2.y)**2
        Nefolds = e2.t

        Nordered = Nefolds[::-1].copy()
        uordered_s = realu_s[::-1].copy()

        """
        Generate interpolating function for realu(N)
        """
        spline5 = CubicSpline(Nordered, uordered_s)

        """
        Imaginary part
        """
        N = Nefolds[0]

        e2 = solve_ivp(
            scalarsys,
            (N, 0),
            imu_init,
            args=spec_params,
            method='RK45',
            first_step=h2,
            rtol=abserr2,
            atol=relerr2,
            events=event_s,
        )

        imu_s = (e2.y)**2
        Nefolds = e2.t

        count = Nefolds.shape[0]-2

        P_s[m] = (k**3/(2*(np.pi**2))) * (spline5(Nefolds[count])+imu_s[count]) / ((spec_params.a_init*np.exp(-Nefolds[count])*spec_params.a_init*np.exp(-Nefolds[count])*spline2(Nefolds[count]))/(4*np.pi))

        """
        Tensor spectra
        """
        count = 0
        
        N = Nefolds[0]
        realu_init[0] = ru_init
        realu_init[1] = dru_init

        e2 = solve_ivp(
            tensorsys,
            (N, 0),
            realu_init,
            args=spec_params,
            method='RK45',
            first_step=h2,
            rtol=abserr2,
            atol=relerr2,
            events=event_s,
        )

        realu_t = (e2.y)**2
        Nefolds = e2.t

        Nordered = Nefolds[::-1].copy()
        uordered_t = realu_t[::-1].copy()

        spline7 = CubicSpline(Nordered, uordered_t)

        """
        Imaginary part
        """
        count = 0

        N = Nefolds[0]
        imu_init[0] = iu_init
        imu_init[1] = diu_init

        e2 = solve_ivp(
            tensorsys,
            (N, 0),
            imu_init,
            args=spec_params,
            method='RK45',
            first_step=h2,
            rtol=abserr2,
            atol=relerr2,
            events=event_s,
        )

        imu_s = (e2.y)**2
        Nefolds = e2.t

        count = Nefolds.shape[0]-2

        P_t[m] = 64 * np.pi * (k**3/(2*np.pi**2)) * (spline7(Nefolds[count])+imu_t[count]) / ((spec_params.a_init*np.exp(-Nefolds[count])*spec_params.a_init*np.exp(-Nefolds[count])))

        if kis[m] == knorm * 5.41e-58: # normalize here
            spec_norm = Amp / (P_s[m]+P_t[m])

            """
            This is a little different from the C code,
            because the y[1] change is outside the if statement
            """
            y[1] = np.sqrt(spec_norm) # normalize H for later recon

    """
    Now that we have finished calculating the spectra, interpolate each spectrum and evaluate at k-values of interest
    """
    spline6 = CubicSpline(kis, P_s)
    spline8 = CubicSpline(kis, P_t)

    u_s = np.array([ks, spec_norm*spline6(ks*5.41e-58)])
    u_t = np.array([ks, spec_norm*spline8(ks*5.41e-58)])

    return status

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

def scalarsys(t, y, parameters):
    dydN = np.empty(2)

    p = params()
    p = parameters

    dydN[0] = y[1]
    dydN[1] = (1-p.eps)*y[1] - (((p.k)*(p.k))/((p.a_init)*(p.a_init)*np.exp(-2.*t)*(p.H)*(p.H))-2.*(1.-2.*(p.eps)-0.75*(p.sig) - (p.eps)*(p.eps) + 0.125*(p.sig)*(p.sig) + 0.5*(p.xi)))*y[0]

    return dydN

def tensorsys(t, y, parameters):
    dydN = np.empty(2)

    p = params()
    p = parameters

    dydN[0] = y[1]
    dydN[1] = (1-p.eps)*y[1] - (((p.k)*(p.k))/((p.a_init)*(p.a_init)*np.exp(-2.*t)*(p.H)*(p.H))-(2.-p.eps))*y[0]

    return dydN
