import numpy as np
import time
from scipy.integrate import solve_ivp, odeint

from MacroDefinitions import *

def int_de(
    y,
    N_interval,
    derivs,
    event=None,
):
    N, Nend = N_interval
    h = 1e-6
    status = 0

    sol = solve_ivp(
        derivs,
        [N, Nend],
        y,
        events=event,
        method='DOP853',
        first_step=h,
    )

    # print(sol)

    # if sol.success is not True: status = 1

    # count = sol.t.size

    # ypp = sol.y.copy()
    # xpp = sol.t.copy()

    # y[:] = ypp[:,-1].copy()

    return sol
