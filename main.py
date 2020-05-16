import numpy as np
import sys
from scipy.integrate import solve_ivp

from MacroDefinitions import *
from calcpath import *
from int_de import *

import pygsl.rng

my_random = pygsl.rng.ranlxd2()
my_random.set(0)

np.random.seed(0)

NMAX = 1.5
NMIN = 0.5

OUTFILE1_NAME = "nr.dat"
OUTFILE2_NAME = "esigma.dat"
NUMPOINTS = 20000

NUMEFOLDSMAX = 60.
NUMEFOLDSMIN = 46.

class Calc:
    def __init__(self):
        self.Y = np.zeros(NEQS, dtype=float, order='C')
        self.initY = np.zeros(NEQS, dtype=float, order='C')
        self.ret = ""
        self.npoints = 0
        self.Nefolds = 0.0


def pick_init_vals ():
    init_vals = np.zeros(NEQS, dtype=float, order='C')
    
    init_vals[0] = 0.0
    init_vals[1] = 1.0
    init_vals[2] = my_random.uniform() * 0.8
    init_vals[3] = my_random.uniform() - 0.5
    init_vals[4] = my_random.uniform()*0.1 - 0.05

    prefact = 0.05
    
    for i in range (5 , NEQS):
        init_vals[i] = my_random.uniform() * prefact - (0.5*prefact)
        prefact *= 0.1
        
    init_Nefolds = my_random.uniform() * (NUMEFOLDSMAX - NUMEFOLDSMIN) + NUMEFOLDSMIN
    
    return init_vals, init_Nefolds

def main():
    calc = Calc()

    if SPECTRUM is True:
        # write code for SPECTRUM
        pass

    try:
        outfile1 = open(OUTFILE1_NAME, "w")
    except IOError as e:
        print(f"Could not open file {OUTFILE1_NAME}, errno = {e}.")
        sys.exit()
        
    try:
        outfile2 = open(OUTFILE2_NAME, "w")
    except IOError as e:
        print(f"Could not open file {OUTFILE2_NAME}, errno = {e}.")
        sys.exit()

    # allocate buffers
    y = np.zeros(NEQS, dtype=float, order='C')
    yinit = np.zeros(NEQS, dtype=float, order='C')
    N = np.array([])

    path = np.array([[]])

    # iters = total number of iterations
    iters = 0

    # points = points saved with n < NMAX
    points = 0

    errcount = 0

    outcount = 0

    # asymcount = points with 0 < n < NMAX , r = 0
    asymcount = 0

    # nontrivcount = nontrivial points
    nontrivcount = 0

    # insuffcount = points where slow roll breaks down before N-efolds
    insuffcount = 0

    # noconvcount = points that do not converge to either a late time attractor or end of inflation
    noconvcount = 0

    badncount = 0

    savedone = 0

    while nontrivcount < NUMPOINTS:
        iters += 1

        if iters % 100 == 0:
            if iters % 1000 == 0:
                print(f"\n asymcount = {asymcount}, nontrivcount = {nontrivcount}, insuffcount = {insuffcount}, noconvcount = {noconvcount}, badncount = {badncount}, errcount = {errcount}")
                print(f"\n{iters}", end="")
            else:
                print(".", end="")

        yinit, calc.Nefolds = pick_init_vals()

        y = yinit.copy()

        calc.ret = calcpath(calc.Nefolds, y, path, N, calc)

        for i in range(NEQS):
            for j in range(calc.npoints):
                print("%.18e "%path[i, j], end="")

            print("\n")

        exit(0)

        if calc.ret == "asymptote":
            # Check to see if the spectral index is within the slow roll range
            if specindex(y) >= NMIN and specindex(y) <= NMAX:
                # Output final values, outfile1 contains
                # observables r, n,  dn/dlog(k), outfile2 countains
                # epsilon, sigma, xsi.
                asymcount += 1

                outfile1.write("%.10f %.10f %.10f\n" % (tsratio(y), specindex(y), dspecindex(y)))
                outfile1.flush()

                for i in range(NEQS):
                    outfile2.write("%le " % (y[i]))

                outfile2.write("\n")
                outfile2.flush()

                points += 1
                savedone = 0
            else: # Spectral index out of range
                badncount += 1
        elif calc.ret == "nontrivial":
            outfile1.write("%.10f %.10f %.10f\n" % (tsratio(y), specindex(y), dspecindex(y)))
            outfile1.flush()

            for i in range(NEQS):
                outfile2.write("%le " % (y[i]))

            outfile2.write("%f\n" % (calc.Nefolds))
            outfile2.flush()

            points += 1
            savedone = 0
            nontrivcount += 1

            if SPECTRUM is True:
                # Add SPECTRUM part of code here
                pass
        elif calc.ret == "insuff":
            insuffcount += 1
        elif calc.ret == "noconverge":
            noconvcount += 1
        else:
            errcount += 1

        if SAVEPATHS is True:
            # Add SAVEPATHS code later
            pass

        # print("D", calc.npoints)
        # exit()

if __name__ == "__main__":
    main()











