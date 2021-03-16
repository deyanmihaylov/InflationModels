import numpy as np
import sys
from scipy.integrate import solve_ivp

from MacroDefinitions import *
from calcpath import *
from int_de import *
if SPECTRUM: from spectrum import *

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

if SAVEPATHS is True:
    PRINTEVERY = 100

class Calc:
    def __init__(self):
        self.Y = np.zeros(NEQS, dtype=float, order='C')
        self.initY = np.zeros(NEQS, dtype=float, order='C')
        self.ret = ""
        self.npoints = 0
        self.Nefolds = 0.0


def pick_init_vals():
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

def we_should_save_this_path(retval, save, pointcount):
    if ((retval == "asymptote" or retval == "nontrivial") and not(save)):
        if not(pointcount % PRINTEVERY):
            return True
        else:
            return False
    else:
        return False

def we_should_calc_spec(y):
    if (specindex(y) > 0.8 and specindex(y) < 1.2):
        return True
    else:
        return False

def save_path(y, N, kount, fname):
    # Open output file
    try:
        outfile = open(fname, "w")
    except IOError as e:
        print(f"Could not open file {fname}, errno = {e}.")
        sys.exit()

    # Output intermediate data from the integration
    for i in range(kount):
        for j in range(NEQS):
            outfile.write("%le " % (y[j, i]))

        outfile.write("%lf " % (N[i]))

        # Calculate "reconstructed" value of the potential, in Planck units.
        V = (3./(8.*np.pi)) * y[1, i] * y[1, i] * (1.-y[2, i]/3.)
        outfile.write("%le " % (V))
        outfile.write("%le\n" % ((V*y[2, i]) / (3.-y[2, i])))

    outfile.close()

def main():
    calc = Calc()

    if SPECTRUM is True:
        spectrum_status = None

        u_s = np.empty((2, kmax))
        u_t = np.empty((2, kmax))

        specnum_s = ""
        specnum_t = ""

        spec_count = 0

        y_final = np.empty(NEQS + 1)

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
        print(iters)

        if iters % 100 == 0:
            if iters % 1000 == 0:
                print(f"\n asymcount = {asymcount}, nontrivcount = {nontrivcount}, insuffcount = {insuffcount}, noconvcount = {noconvcount}, badncount = {badncount}, errcount = {errcount}")
                print(f"\n{iters}", end="")
            else:
                print(".", end="")

        yinit, calc.Nefolds = pick_init_vals()

        # remove when spectrum code is finished
        # if iters < 121:
        #     continue

        y = yinit.copy()

        calc.ret = calcpath(calc.Nefolds, y, path, N, calc)

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
            else:
                # Spectral index out of range
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
                if we_should_calc_spec(y):
                    print(f"Evaluating spectrum {spec_count}")

                    specnum_s = f"spec_s{str(spec_count).zfill(3)}.dat"
                    specnum_t = f"spec_t{str(spec_count).zfill(3)}.dat"

                    spec_count += 1

                    # try:
                    #     spec_s = open(specnum_s, "w")
                    # except IOError as e:
                    #     print(f"Could not open file {specnum_s}, errno = {e}.")
                    #     sys.exit()

                    # try:
                    #     spec_t = open(specnum_t, "w")
                    # except IOError as e:
                    #     print(f"Could not open file {specnum_t}, errno = {e}.")
                    #     sys.exit()

                    y_final[:NEQS] = path[:NEQS, 3]
                    y_final[NEQS] = N[3]

                    spectrum_status = spectrum(y_final, y, u_s, u_t, calc.Nefolds, derivs1, scalarsys, tensorsys)

                    if spectrum_status:
                        errcount += 1

                    # Here is where the spectrum files are written. Choose desired format.
                    # for i in range(knos):
                    #     spec_s.write("%.15e %.15e\n" % (u_s[0, i], u_s[1, i]))
                    #     spec_t.write("%.15e %.15e\n" % (u_t[0, i], u_t[1, i]))

                    np.savetxt(specnum_s, u_s[:,0:knos].T, fmt='%.15e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                    np.savetxt(specnum_t, u_t[:,0:knos].T, fmt='%.15e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

                    # spec_s.close()
                    # spec_t.close()

        elif calc.ret == "insuff":
            insuffcount += 1
        elif calc.ret == "noconverge":
            noconvcount += 1
        else:
            errcount += 1

        if SAVEPATHS is True:
            """
            Check to see if initial data yielded suitable results for the
            entire path to be generated and saved.  If so, the initial data
            are stored in temporary buffers.
            """
            if SPECTRUM is True:
                # if we calc spectrum, we want path
                criterion = we_should_calc_spec(y) and calc.ret == "nontrivial"

            if SPECTRUM is False:
                # if not, choose different criterion
                criterion = we_should_save_this_path(calc.ret, savedone, points)

            if criterion:
                if SPECTRUM is True:
                    """
                    If we are calculating spectra, we must normalize H here instead
                    of in calcpath.c.
                    """
                    for j in range(calc.npoints):
                        path[0, j] = path[0, j] - path[0, calc.npoints-1]
                        path[1, j] = path[1, j] * y[1]

                fname = f"path{str(outcount).zfill(3)}.dat"
                outcount += 1

                save_path(path, N, calc.npoints, fname)

                savedone = 1

            if spec_count > 0:
                exit()

    print(f"Done. points = {points}, iters = {iters}, errcount = {errcount}")
    print(f"asymcount = {asymcount}, nontrivcount = {nontrivcount}, insuffcount = {insuffcount}, noconvcount = {noconvcount}, badncount = {badncount}, errcount = {errcount}")

    outfile1.close()
    outfile2.close()

if __name__ == "__main__":
    main()
