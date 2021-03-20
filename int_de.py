import sys

from MacroDefinitions import *

from scipy.integrate import solve_ivp

def int_de(y, N, Nend, kmax, NEQS, derivs):
    h = 1e-6

    abs_tol = 1e-8
    rel_tol = 1e-8

    res = solve_ivp(
        derivs,
        (N, Nend),
        y,
        method='RK45',
        first_step=h,
        rtol=abs_tol,
        atol=rel_tol,
    )

    status = res.status

    yp = res.y
    xp = res.t

    count = xp.shape[0]

    if count > kmax:
        sys.exit("BIG PROBLEM with count")

    return status, count, yp, xp