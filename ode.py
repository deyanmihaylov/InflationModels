# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from julia import Main
# jl.using("DifferentialEquations")

from diffeqpy import de
from time import process_time
from julia import Main
import numba
import numpy as np

jul_f = Main.eval("""
function f(du,u,p,t)
  x, y, z = u
  sigma, rho, beta = p
  du[1] = sigma * (y - x)
  du[2] = x * (rho - z) - y
  du[3] = x * y - beta * z
end""")
u0 = [1.0,0.0,0.0]
tspan = (0., 100.)
p = [10.0,28.0,2.66]
prob = de.ODEProblem(jul_f, u0, tspan, p)
sol = de.solve(prob)

julia_times = []

for _ in range(1000):
    t1 = process_time()
    sol_new = de.solve(prob)
    t2 = process_time()
    julia_times.append(t2 - t1)

du = np.zeros((3), dtype=np.float32)

def f2(t, u, du, sigma, rho, beta):
    x, y, z = u
    du[0] = sigma * (y - x)
    du[1] = x * (rho - z) - y
    du[2] = x * y - beta * z
    return du

numba_f = numba.jit(f2, nopython=True)

from scipy.integrate import solve_ivp

sol = solve_ivp(
    numba_f,
    tspan,
    u0,
    method='RK45',
    t_eval=None,
    dense_output=False,
    events=None,
    vectorized=False,
    args=(du,10.0,28.0,2.66),
)

scipy_times = []

for _ in range(1000):
    t3 = process_time()
    sol_new = solve_ivp(
        numba_f,
        tspan,
        u0,
        method='RK45',
        t_eval=None,
        dense_output=False,
        events=None,
        vectorized=False,
        args=(du,10.0,28.0,2.66),
    )
    t4 = process_time()

    scipy_times.append(t4 - t3)

julia_avg = np.mean(julia_times)
scipy_avg = np.mean(scipy_times)

print(f"Julia: {julia_avg}")
print(f"scipy: {scipy_avg}")
