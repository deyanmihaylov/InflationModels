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

        ypp[:, count] = y.copy()
        xpp[count] = N

        count += 1

        if count == kmax:
            break

    ypp_temp = ypp[:, 0:count].copy()
    xpp_temp = xpp[0:count].copy()

    ypp.resize((NEQS, count), refcheck=False)
    xpp.resize(count, refcheck=False)

    ypp[:,:] = ypp_temp.copy()
    xpp[:] = xpp_temp.copy()

    kount = count

    return status, kount