import numpy as np
from time import process_time
from scipy.integrate import solve_ivp, odeint

from macros import *

def int_de(
    y,
    N_interval,
    derivs,
    event=None,
):
    N, Nend = N_interval
    h = 1e-6
    status = 0

    # t1 = process_time()
    sol = solve_ivp(
        derivs,
        [N, Nend],
        y,
        events=event,
        method='DOP853',
        first_step=h,
    )
    # t2 = process_time()
    # print(f"Time for int_de: {t2 - t1}")

    # print(sol)

    # if sol.success is not True: status = 1

    # count = sol.t.size

    # ypp = sol.y.copy()
    # xpp = sol.t.copy()

    # y[:] = ypp[:,-1].copy()

    return sol
