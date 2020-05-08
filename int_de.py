# from scipy.integrate import solve_ivp
import pygsl.odeiv as odeiv

from MacroDefinitions import *

def int_de(y, N, Nend, kount, kmax, ypp, xpp, NEQS, derivs):
    h = 1e-6
    i = None
    status = 0
    count = 0

    s = odeiv.step_rk4(NEQS, derivs)
    c = odeiv.control_y_new(s, 1e-8, 1e-8)
    e = odeiv.evolve(s, c, NEQS)

    ydoub = y.copy()

    if N > Nend:
        h = -h

    while N != Nend:
        try:
            N, h, ydoub = e.apply(N, Nend, h, ydoub)
        except:
            status = 1
            break
        
        for i in range(NEQS): y[i] = ydoub[i]

        ypp[:, count] = y
        xpp[count] = N

        count += 1

        if count == kmax:
            break

    kount = count

    return status, kount