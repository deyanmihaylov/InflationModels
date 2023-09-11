import numpy as np
import numba

from macros import *
from int_de import *

c = 4 * (np.log(2) + np.euler_gamma) - 5

def calculate_path(
    calc,
    backend: str = "julia",
):
    result = "internal_error"
    
    """
    Check to make sure we are calculating to sufficient order
    """
    if NEQS < 6:
        raise Exception("calculate_path(): NEQS must be at least 6\n")
        sys.exit()
    
    """
    First find the end of inflation, when epsilon crosses through unity
    """
    N_start = LOTS_OF_EFOLDS
    N_end = 0
    
    inflation_ends.direction = 1
    forward_solution = int_de(
        calc.y_init,
        (N_start, N_end),
        derivs,
        event=inflation_ends,
    )
    xp = forward_solution.t
    yp = forward_solution.y
    y = yp[:,-1].copy()
    kount = forward_solution.t.size
    z = 1 if forward_solution.success is not True else 0

    if z:
        retval = "internal_error"
        z = 0
    else:
        """
        Find when epsilon passes through unity
        """
        i = find_convergence(yp)

        if i == 0:
            """
            We never found an end to inflation, so we must be at a
            late-time attractor
            """
            if y[2] > SMALL_NUM or y[3] < 0:
                """
                The system did not evolve to a known asymptote
                """
                retval = "noconverge"
            else:
                retval = "asymptote"
        else:
            """
            We found an end to inflation (find_convergence > 0):
            integrate backwards Nefolds e-folds from that point
            """
            N_start = 1e-8
            N_end = calc.N_efolds
            y = forward_solution.y_events[0].flatten()
            
            backward_solution = int_de(y, (N_start, N_end), derivs)
            xp = backward_solution.t
            yp = backward_solution.y
            y = yp[:,-1].copy()
            kount = backward_solution.t.size
            z = 1 if backward_solution.success is not True else 0

            if z:
                retval = "internal_error"
                z = 0
            elif find_convergence(yp) > 0:
                """
                Not enough inflation.
                """
                retval = "insuff"
            else:
                retval = "nontrivial"

    """
    Normalize H to give the correct CMB amplitude. If we are not
    interested in generating power spectra, normalizing H to give CMB
    amplitude of 10^-5 at horizon crossing (N = Nefolds) is sufficient.
    """
    if SPECTRUM == False:
        if retval == "nontrivial":
            Hnorm = 0.00001 * 2 * np.pi * np.sqrt(y[2]) / y[1]
            y[1] = Hnorm * y[1]

            yp[1, :] = Hnorm * yp[1, :]
            yp[0, :] = yp[0, :] - y[0]

    if retval != "internal_error" and kount > 1:
        N = xp.copy()
        path = yp.copy()

        count = kount
    else:
        count = 0

    calc.npoints = count

    return y, N, path, retval

@numba.njit(
    cache = True,
    fastmath = True,
)
def derivs(
    N,
    y,
):
    dy_dN = np.zeros(NEQS)
    
    if y[2] >= 1:
        dy_dN = np.zeros(NEQS)
    else:
        if y[2] > VERY_SMALL_NUM:
            dy_dN[0] = - np.sqrt(y[2] / (4 * np.pi))
        else:
            dy_dN[0] = 0.0
        
        dy_dN[1] = y[1] * y[2]
        dy_dN[2] = y[2] * (y[3] + 2 * y[2])
        dy_dN[3] = 2 * y[4] - 5 * y[2] * y[3] - 12 * y[2] * y[2]
        
        for i in range(4, NEQS-1):
            dy_dN[i] = (0.5 * (i-3) * y[3] + (i-4) * y[2]) * y[i] + y[i+1]
            
        dy_dN[NEQS-1] = (0.5 * (NEQS-4) * y[3] + (NEQS-5) * y[2]) * y[NEQS-1]

    return dy_dN

@numba.njit(
    cache = True,
    fastmath = True,
)
def inflation_ends(
    N,
    y,
):
    return y[2] - (1 - 1e-8)

@numba.njit(
    cache = True,
    fastmath = True,
)
def find_convergence(
    y,
):
    i = np.argmax(y[2] >= 1)
    if i > 0:
        return i
    else:
        return 0

@numba.njit(
    cache = True,
    fastmath = True,
)
def tensor_scalar_ratio(
    y,
):
    r = 16 * y[2] * (1 - c * (y[3] + 2 * y[2]))

    return r

@numba.njit(
    cache = True,
    fastmath = True,
)
def spectral_index(
    y,
):
    if SECONDORDER is True:
        n = (
            1 + y[3]
            - (5 - 3*c) * y[2] * y[2]
            - 0.25 * (3-5*c) * y[2] * y[3]
            + 0.5 * (3-c) * y[4]
        )
    else:
        n = (
            1 + y[3]
            - 4.75564 * y[2] * y[2]
            - 0.64815 * y[2] * y[3]
            + 1.45927 * y[4]
            + 7.55258 * y[2] * y[2] * y[2]
            + 12.0176 * y[2] * y[2] * y[3]
            + 3.12145 * y[2] * y[3] * y[3]
            + 0.0725242 * y[3] * y[3] * y[3]
            + 5.92913 * y[2] * y[4]
            + 0.085369 * y[3] * y[4]
            + 0.290072 * y[5]
        )

    return n

@numba.njit(
    cache = True,
    fastmath = True,
)
def d_spectral_index(
    y,
):
    dy_dN = derivs(0, y)

    if SECONDORDER is True:
        dn_dN = (
            - (1/(1 - y[2]) * (
                dy_dN[3]
                - 2 * (5-3*c) * y[2] * dy_dN[2]
                - 0.25 * (3-5*c) * (y[2] * dy_dN[3] + y[3] * dy_dN[2])
                + 0.5 * (3 - c) * dy_dN[4]
            ))
        )
    else:
        dn_dN = (
            - (1/(1 - y[2]) * (
                dy_dN[3]
                - 2 * 4.75564 * y[2] * dy_dN[2]
                - 0.64815 * (y[2] * dy_dN[3] + dy_dN[2] * y[3])
                + 1.45927 * dy_dN[4]
                + 3 * 7.55258 * y[2] * y[2] * dy_dN[2]
                + 12.0176 * (y[2] * y[2] * dy_dN[3] + 2 * y[2] * dy_dN[2] * y[3])
                + 3.12145 * (2 * y[2] * y[3] * dy_dN[3] + dy_dN[2] * y[3] * y[3])
                + 3 * 0.0725242 * y[3] * y[3] * dy_dN[3]
                + 5.92913 * (y[2] * dy_dN[4] + dy_dN[2] * y[4])
                + 0.085369 * (y[3] * dy_dN[4] + dy_dN[3] * y[4])
                + 0.290072 * dy_dN[5]
            ))
        )

    return dn_dN























