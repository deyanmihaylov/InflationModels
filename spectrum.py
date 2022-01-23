import numpy as np
import numba
# import pygsl.odeiv as odeiv

from numba import float64, complex128, types, typed, typeof
from numba import vectorize

from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar, root
from scipy.special import jv, yv

from interpolate import CubicSpline
from interpolate import cubic_spline, calc_spline_params, piece_wise_spline

from calcpath import *

# import matplotlib.pyplot as plt

knos = 1575 # total number of k-values to evaluate
kinos = 214 # total number of k-values to use for integration
k_file = "ks_eval.dat" # file containing k-values at which to evaluate spectrum
ki_file = "ks.dat" # file containing k-values for integration
Y = 50 # Y = value of k/aH at which to initialize mode fcns
knorm = 0.05 # normalization scale
Amp = 2.0803249e-9 # scalar amplitude at knorm

class params:
    def __init__(self):
        self.a_init = None # initial val of the scale factor
        self.k = None # comoving wavenumber
        # self.eps = None
        # self.sig = None
        # self.H = None
        # self.xi = None


def ode_eq_solver(
    N_init,
    u_init,
    step,
    fun,
    event_fun,
    fun_args,
    rtol,
    atol,
):
    solution = solve_ivp(
        fun,
        (N_init, 0),
        u_init,
        method = 'DOP853',
        events=event_fun,
        first_step=step,
        args = fun_args,
        vectorized=True,
        rtol=rtol,
        atol=atol,
    )

    return solution

def spectrum(
    y_final,
    y,
    N,
    derivs1,
    scalarsys,
    tensorsys,
):
    h = 0.01
    h2 = 1e-6

    abserr1 = 1e-8
    relerr1 = 1e-8

    abserr2 = 1e-10
    relerr2 = 1e-10

    try:
        ks = np.loadtxt(k_file)
    except IOError as e:
        print("Could not open file " + k_file + ", errno = " + e + ".")
        sys.exit()
        
    try:
        kis = np.loadtxt(ki_file)
    except IOError as e:
        print("Could not open file " + ki_file + ", errno = " + e + ".")
        sys.exit()

    P_s = np.empty(kinos)
    P_t = np.empty(kinos)

    """
    Set the initial value of the scale factor.  This is chosen
    so that k = aH (with k corresponding to the quadrupole) at the
    value N_obs from the path file.  The scale factor as a 
    function of N is a(N) = a_init*exp(-# of efolds).
    Units are hM_PL
    """
    Ninit = N
    a_init = (1.73e-61/y[1]) * np.exp(Ninit)

    """
    To improve stability/efficiency, we first generate an interpolating 
    function for H, epsilon, sigma and xi^2.  We then pass these values
    as parameters to the mode equation, rather than solving the mode 
    equation along with the full set of flow equations each time.
    """

    """
    Integrate backwards from end of inflation to the earliest time 
    needed in order to initialize the largest scale fluctuations in the 
    BD limlt.
    """
    ydoub = y_final[:NEQS].copy()
    N = y_final[NEQS]
    Nfinal = N

    def BD_limit(N, y):
        res = (kis[0]*5.41e-58) / (a_init*np.exp(-N)*y[1]) - Y

        return res
    
    BD_limit.terminal = True

    eq1 = solve_ivp(
        derivs2,
        (N, 1000),
        ydoub,
        method = 'DOP853',
        events=BD_limit,
        first_step=h2,
        rtol=relerr1,
        atol=abserr1,
    )

    countback = eq1.t.size - 1
    Nefoldsback = eq1.t.copy()
    flowback = eq1.y[:5].copy()

    phi = flowback[0,:].copy()
    H = flowback[1,:].copy()
    eps = flowback[2,:].copy()
    sig = flowback[3,:].copy()
    xi = flowback[4,:].copy()

    Nefolds = Nefoldsback.copy()

    """
    Generate interpolating functions for H, eps, sig, xi and phi (for 
    path generation only)
    """

    phi_interp = CubicSpline(Nefolds, phi)
    H_interp = CubicSpline(Nefolds, H)
    eps_interp = CubicSpline(Nefolds, eps)
    sig_interp = CubicSpline(Nefolds, sig)
    xi_interp = CubicSpline(Nefolds, xi)

    k_Planck = kis * 5.41e-58
    N = Ninit * np.ones(k_Planck.size)

    """
    First, check to see if the given k value is in the Bunch-Davies 
    limit at the start of inflation. This limit is set by Y=k/aH. 
    If the given k value yields a larger Y than the BD limit, then 
    we must integrate forward (to smaller N) until we reach the 
    proper value for Y.  If it is smaller, we must integrate 
    backwards (to larger N).  These integrators are given a fixed 
    stepsize to ensure that we don't inadvertently step too far 
    beyond Y.
    """

    def BD_limit_eq(N):
        res = (
            (
                k_Planck
                / (
                    a_init * np.exp(-N)
                    * H_interp.eval(N) * (1-eps_interp.eval(N))
                )
            )
            - Y
        )

        return res

    BD_limit_sol = root(
        BD_limit_eq,
        method="lm",
        x0=N,
        tol=relerr1,
    )

    N = BD_limit_sol.x

    H = H_interp.eval(N)
    eps = eps_interp.eval(N)

    nu = (3 - eps) / (2 * (1 - eps))
    Y_eff = k_Planck / (a_init * np.exp(-N) * H * (1-eps))

    J_nu = jv(nu, Y_eff)
    J_nu1 = jv(nu+1, Y_eff)
    
    Y_nu = yv(nu, Y_eff)
    Y_nu1 = yv(nu+1, Y_eff)

    J_Y_nu = J_nu + 1j*Y_nu
    J_Y_nu1 = J_nu1 + 1j*Y_nu1

    u_init = np.array([
        0.5 * np.sqrt(np.pi / k_Planck) * np.sqrt(Y_eff) * J_Y_nu,
        (
            0.5 * np.sqrt(np.pi / k_Planck)
            * (k_Planck / (a_init * np.exp(-N) * H)) * (
                J_Y_nu / (2 * np.sqrt(Y_eff)) 
                + np.sqrt(Y_eff) * (
                    - J_Y_nu1
                    + nu * J_Y_nu / Y_eff
                )
            )
        ),
    ])

    u_init[:, np.where(eps >= 1)] *= -1j
    
    # @numba.njit(
    #     float64(
    #         float64,
    #         complex128[:,:],
    #         float64,
    #         float64,
    #         types.ListType(CubicSpline.class_type.instance_type),
    #     ),
    #     cache = True,
    #     fastmath = True,
    # )
    def Nfinal_event(N, y, a_init, k, interps):
        return N - Nfinal

    Nfinal_event.terminal = True
    Nfinal_event.direction = -1

    for i in range(N.size):
        print(i)

        """
        Scalar spectra
        """

        solution_s = ode_eq_solver(
            N[i],
            u_init[:, i],
            h2,
            scalarsys,
            Nfinal_event,
            (
                a_init,
                k_Planck[i],
                typed.List([H_interp, eps_interp, sig_interp, xi_interp])
            ),
            relerr2,
            abserr2,
        )

        k = k_Planck[i]

        N_val = solution_s.t[-1]
        u2_s = np.abs(solution_s.y[0,-1])**2

        P_s[i] = (
            (k**3/(2*(np.pi**2)))
            * u2_s
            / (
                (
                    ((a_init*np.exp(-N_val))**2)
                    * eps_interp.eval(N_val)[()]
                )
                / (4*np.pi)
            )
        )

        """
        Tensor spectra
        """

        solution_t = ode_eq_solver(
            N[i],
            u_init[:, i],
            h2,
            tensorsys,
            Nfinal_event,
            (
                a_init,
                k_Planck[i],
                typed.List([H_interp, eps_interp, sig_interp, xi_interp])
            ),
            relerr2,
            abserr2,
        )

        N_val = solution_t.t[-1]
        u2_t = np.abs(solution_t.y[0,-1])**2

        P_t[i] = (
            64 * np.pi * (k**3 / (2 * np.pi**2))
            * u2_t
            / (a_init * np.exp(-N_val))**2
        )
    
    status = 0

    norm_idx = np.argwhere(k_Planck == knorm * 5.41e-58)[0][0]
    spec_norm = Amp / (P_s[norm_idx]+P_t[norm_idx])

    y[1] = np.sqrt(spec_norm) # normalize H for later recon

    """
    Now that we have finished calculating the spectra, interpolate each 
    spectrum and evaluate at k-values of interest
    """

    P_s_interp = cubic_spline(k_Planck, P_s)
    P_t_interp = cubic_spline(k_Planck, P_t)

    u_s = np.c_[ks.copy(), spec_norm * P_s_interp(ks*5.41e-58)]
    u_t = np.c_[ks.copy(), spec_norm * P_t_interp(ks*5.41e-58)]

    # u_s[0, :knos] = ks.copy()
    # u_s[1, :knos] = spec_norm * P_s_interp(ks*5.41e-58)

    # u_t[0, :knos] = ks.copy()
    # u_t[1, :knos] = spec_norm * P_t_interp(ks*5.41e-58)

    return status, u_s, u_t

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

@numba.njit(
    cache = True,
    fastmath = True,
)
def derivs2(N, y):
    dy_dN = np.zeros(NEQS)
    
    if y[2] > VERYSMALLNUM:
        dy_dN[0]= - np.sqrt(y[2]/(4*np.pi))
    else:
        dy_dN[0] = 0

    dy_dN[1] = y[1] * y[2]
    dy_dN[2] = y[2] * (y[3]+2*y[2])
    dy_dN[3] = 2*y[4] - 5*y[2]*y[3] - 12*y[2]*y[2]
    
    for i in range(4, NEQS-1):
        dy_dN[i] = (0.5*(i-3)*y[3]+(i-4)*y[2])*y[i] + y[i+1]

    dy_dN[NEQS-1] = (0.5*(NEQS-4)*y[3]+(NEQS-5)*y[2]) * y[NEQS-1]

    return dy_dN

@numba.njit(
    complex128[:](
        float64,
        complex128[:,:],
        float64,
        float64,
        types.ListType(CubicSpline.class_type.instance_type),
    ),
    cache = True,
    fastmath = True,
)
def scalarsys(
    N,
    y,
    a_init,
    k,
    interps,
): 
    H_interp, eps_interp, sig_interp, xi_interp = interps

    H = H_interp.eval(N)
    eps = eps_interp.eval(N)
    sig = sig_interp.eval(N)
    xi = xi_interp.eval(N)

    y = y.ravel()

    dy_dN = np.zeros(2, dtype=np.complex128)

    dy_dN[0] = y[1]
    dy_dN[1] = (
        (1-eps) * y[1]
        - (
            (k**2) / ((a_init**2)*np.exp(-2*N)*(H**2))
            - 2 * (1 - 2*eps - 0.75*sig - eps**2 + 0.125*sig**2 + 0.5*xi)
        ) * y[0]
    )

    return dy_dN

@numba.njit(
    complex128[:](
        float64,
        complex128[:,:],
        float64,
        float64,
        types.ListType(CubicSpline.class_type.instance_type),
    ),
    cache = True,
    fastmath = True,
)
def tensorsys(
    N,
    y,
    a_init,
    k,
    interps,
):
    H_interp, eps_interp, sig_interp, xi_interp = interps

    H = H_interp.eval(N)
    eps = eps_interp.eval(N)

    y = y.ravel()
    
    dy_dN = np.zeros(2, dtype=np.complex128)

    dy_dN[0] = y[1]
    dy_dN[1] = (
        (1-eps)*y[1]
        - (
            (k**2) / ((a_init**2)*np.exp(-2*N)*(H**2))
            - (2 - eps)
        ) * y[0]
    )

    return dy_dN
