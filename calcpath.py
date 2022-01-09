import numpy as np
import numba

from MacroDefinitions import *
from int_de import *

kmax = 20000

# c = 0.0814514 # = 4 (ln(2)+\gamma)-5, \gamma = 0.5772156649
c = 4 * (np.log(2) + np.euler_gamma) - 5

def calcpath(Nefolds, y, path, N, calc):
    retval = "internal_error"
    i = None
    j = None
    k = None
    z = None
    kount = None
    Hnorm = None
    
    # Check to make sure we are calculating to sufficient order.
    if NEQS < 6:
        raise Exception("calcpath(): NEQS must be at least 6\n")
        sys.exit()
    
    # Allocate buffers for integration.
    # dydN = derivatives of flow functions wrt N
    # yp = intermediate values for y
    # xp = intermediate values for N
    yp = np.zeros((NEQS, kmax), dtype=float, order='C')
    xp = np.zeros(kmax, dtype=float, order='C')
    
    # First find the end of inflation, when epsilon crosses through unity.
    Nstart = LOTSOFEFOLDS
    Nend = 0.
    
    z, kount = int_de(y, Nstart, Nend, kount, kmax, yp, xp, NEQS, derivs)

    if z:
        retval = "internal_error"
        z = 0
    else:
        # Find when epsilon passes through unity
        i = check_convergence(yp, kount)

        if i == 0:
            # We never found an end to inflation, so we must be at a late-time attractor
            if y[2] > SMALLNUM or y[3] < 0.:
                # The system did not evolve to a known asymptote
                retval = "noconverge"
            else:
                retval = "asymptote"
        else: # if check_convergence: we have found an end to inflation
            # We found an end to inflation: integrate backwards Nefolds e-folds from that point

            Nstart = xp[i-2] - xp[i-1]
            Nend = Nefolds

            y[:] = yp[:, i-2].copy()

            yp = np.zeros((NEQS, kmax), dtype=float, order='C')
            xp = np.zeros(kmax, dtype=float, order='C')

            z, kount = int_de(y, Nstart, Nend, kount, kmax, yp, xp, NEQS, derivs)

            if z:
                retval = "internal_error"
                z = 0
            elif check_convergence(yp, kount):
                # Not enough inflation.
                retval = "insuff"
            else:
                retval = "nontrivial"

    # Normalize H to give the correct CMB amplitude.  If we are not interested in generating power
    # spectra, normalizing H to give CMB amplitude of 10^-5 at horizon crossing (N = Nefolds) is
    # sufficient

    if SPECTRUM == False:
        if retval == "nontrivial":
            Hnorm = 0.00001 * 2 * np.pi * np.sqrt(y[2]) / y[1]
            y[1] = Hnorm * y[1]

            yp[1, :] = Hnorm * yp[1, :]
            yp[0, :] = yp[0, :] - y[0]

    # Fill in return buffers with path info. Note that the calling
    # function is responsible for freeing these buffers! The
    # buffers are only filled in if non-null pointers are provided.

    if (path is not None) and (N is not None) and (retval != "internal_error") and kount > 1:
        N.resize(kount, refcheck=False)
        path.resize(NEQS, kount, refcheck=False)

        for j in range(kount):
            N[j] = xp[j]

            for i in range(NEQS):
                path[i, j] = yp[i, j]

        count = kount
    else:
        count = 0

    calc.npoints = count

    return retval

# def derivs(t, y, dydN):
#     dydN = np.zeros(NEQS, dtype=float, order='C')
    
#     if y[2] >= 1.0:
#         dydN = np.zeros(NEQS , dtype=float , order='C')
#     else:
#         if y[2] > VERYSMALLNUM:
#             dydN[0] = - np.sqrt(y[2] / (4 * np.pi))
#         else:
#             dydN[0] = 0.0
        
#         dydN[1] = y[1] * y[2]
#         dydN[2] = y[2] * (y[3] + 2.0 * y[2])
#         dydN[3] = 2. * y[4] - 5. * y[2] * y[3] - 12. * y[2] * y[2]
        
#         for i in range(4, NEQS-1):
#             dydN[i] = ( 0.5 * (i-3) * y[3] + (i-4) * y[2] ) * y[i] + y[i+1]
            
#         dydN[NEQS-1] = ( 0.5 * (NEQS-4) * y[3] + (NEQS-5) * y[2] ) * y[NEQS-1]

#     return dydN

@numba.njit(
    cache = True,
    fastmath = True,
)
def derivs(
    t,
    y,
):
    dy_dN = np.zeros(NEQS)
    
    if y[2] >= 1:
        dy_dN = np.zeros(NEQS)
    else:
        if y[2] > VERYSMALLNUM:
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

def check_convergence(
    yy,
    kount,
):
    for i in range(kount):
        if np.abs(yy[2, i]) >= 1:
            return i
        
    return 0

def tsratio(
    y,
):
    tsratio = 16 * y[2] * (1 - c * (y[3] + 2 * y[2]))

    return tsratio

def specindex(
    y,
):
    if SECONDORDER is True:
        specindex = (
            1 + y[3]
            - (5-3*c) * y[2] * y[2]
            - 0.25 * (3-5*c) * y[2] * y[3]
            + 0.5 * (3-c) * y[4]
        )
    else:
        specindex = (
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

    return specindex

def dspecindex(
    y,
):
    ydoub = y.copy()

    dydN = np.zeros(NEQS)
    # dydN = derivs(0, ydoub, dydN)
    dydN = derivs(0, ydoub)

    y = ydoub.copy()

    if SECONDORDER is True:
        dspecindex = (
            - (1/(1 - y[2]) * (
                dydN[3]
                - 2 * (5-3*c) * y[2] * dydN[2]
                - 0.25 * (3-5*c) * (y[2] * dydN[3] + y[3] * dydN[2])
                + 0.5 * (3 - c) * dydN[4]
            ))
        )
    else:
        dspecindex = (
            - (1/(1 - y[2]) * (
                dydN[3]
                - 2 * 4.75564 * y[2] * dydN[2]
                - 0.64815 * (y[2] * dydN[3] + dydN[2] * y[3])
                + 1.45927 * dydN[4]
                + 3 * 7.55258 * y[2] * y[2] * dydN[2]
                + 12.0176 * (y[2] * y[2] * dydN[3] + 2 * y[2] * dydN[2] * y[3])
                + 3.12145 * (2 * y[2] * y[3] * dydN[3] + dydN[2] * y[3] * y[3])
                + 3 * 0.0725242 * y[3] * y[3] * dydN[3]
                + 5.92913 * (y[2] * dydN[4] + dydN[2] * y[4])
                + 0.085369 * (y[3] * dydN[4] + dydN[3] * y[4])
                + 0.290072 * dydN[5]
            ))
        )

    return dspecindex























