import numpy as np
import sys

import numba
from numba import types, float64, uint64
from numba.experimental import jitclass

from macros import *
from calcpath import calculate_path
from int_de import *

if SPECTRUM: from spectrum import *
if SAVEPATHS is True: PRINTEVERY = 100

@jitclass([
    ('Y', float64[:]),
    ('initY', float64[:]),
    ('ret', types.unicode_type),
    ('npoints', uint64),
    ('y_init', float64[:]),
    ('N_efolds', float64),  
])
class Calc:
    def __init__(self):
        self.Y = np.zeros(NEQS)
        self.initY = np.zeros(NEQS)
        self.ret = ""
        self.npoints = 0
        self.y_init = np.zeros(NEQS)
        self.N_efolds = 0.0

        # self.init_vals()

    def init_vals(self):
        self.y_init, self.N_efolds = _pick_init_vals()


@numba.njit(
    cache = True,
    fastmath = True,
)
def _pick_init_vals():
    np.random.seed(0)
    init_vals = np.zeros(NEQS)

    init_vals[0] = 0.0
    init_vals[1] = 1.0
    init_vals[2] = np.random.uniform(0, 0.8)
    init_vals[3] = np.random.uniform(-0.5, 0.5)
    init_vals[4] = np.random.uniform(-0.05, 0.05)

    width = 0.05
    
    for i in range(5 , NEQS):
        init_vals[i] = np.random.uniform(-0.5*width, 0.5*width)
        width *= 0.1
        
    init_N_efolds = np.random.uniform(NUM_EFOLDS_MIN, NUM_EFOLDS_MAX)
    
    return init_vals, init_N_efolds

def we_should_save_this_path(
    retval,
    save,
    pointcount,
):
    if ((retval == "asymptote" or retval == "nontrivial") and not(save)):
        if not(pointcount % PRINTEVERY):
            return True
        else:
            return False
    else:
        return False

@numba.njit(
    cache = True,
    fastmath = True,
)
def check_calculate_spectrum(
    y,
):
    n = spectral_index(y)
    if 0.8 < n < 1.2:
        return True
    else:
        return False

def save_data(
    file_name,
    data,
    format = '%.15e',
    delimiter = ',',
    header = "",
):
    np.savetxt(
        file_name,
        data,
        fmt=format,
        delimiter=delimiter,
        newline='\n',
        header=header,
        footer='',
        comments='# ',
        encoding=None,
    )

def save_path(
    y,
    N,
    kount,
    fname,
):
    V = (3/(8*np.pi)) * y[1] * y[1] * (1 - y[2]/3)
    V1 = (V * y[2]) / (3 - y[2])

    data_for_output = np.c_[y.T, N, V, V1]
    save_data(fname, data_for_output, format='%.6e')

def main():
    try:
        outfile1 = open(OUTFILE1_NAME, "w")
    except IOError as e:
        print("Could not open file " + OUTFILE1_NAME + ", errno = " + e + ".")
        sys.exit()
        
    try:
        outfile2 = open(OUTFILE2_NAME, "w")
    except IOError as e:
        print("Could not open file " + OUTFILE2_NAME + ", errno = " + e + ".")
        sys.exit()

    """
    iters = total number of iterations
    points = points saved with n < N_MAX
    asymcount = points with 0 < n < N_MAX , r = 0
    nontrivcount = nontrivial points
    insuffcount = points where slow roll breaks down before N-efolds
    noconvcount = points that do not converge to either a late time 
        attractor or end of inflation
    badncount = points for which n is outside of the bounds
    """
    iters = 0
    points = 0
    errcount = 0
    outcount = 0
    asymcount = 0
    nontrivcount = 0
    insuffcount = 0
    noconvcount = 0
    badncount = 0
    savedone = 0

    if SPECTRUM is True:
        spec_count = 0

    initial_conds = np.genfromtxt(fname = "ics.dat", delimiter=",")
    
    while nontrivcount < NUM_POINTS:
        iters += 1
        # if iters < 120: continue
        # if iters > 200:
        #     exit()
        print(iters)

        if iters % 100 == 0:
            if iters % 1000 == 0:
                print(
                    f"asymcount = {asymcount},\n"
                    f"nontrivcount = {nontrivcount},\n"
                    f"insuffcount = {insuffcount},\n"
                    f"noconvcount = {noconvcount},\n"
                    f"badncount = {badncount},\n"
                    f"errcount = {errcount},\n"
                    f"iters = {iters}"
                )
            else:
                print(".")

        calc = Calc()
        calc.y_init = initial_conds[iters-1, 0:-1]
        calc.N_efolds = initial_conds[iters-1, -1]
        # y = yinit.copy()

        y, N, path, calc.ret = calculate_path(calc)

        match calc.ret:
            case "asymptote":
                pass
            case "nontrivial":
                pass
            case "insuff":
                pass
            case "noconverge":
                pass
            case _:
                pass

        if calc.ret == "asymptote":
            """
            Check to see if the spectral index is within the slow roll range
            """
            n = spectral_index(y)

            if N_MIN <= n <= N_MAX:
                """
                Output final values, outfile1 contains observables r, n,
                dn/dlog(k), outfile2 countains epsilon, sigma, xsi.
                """
                asymcount += 1
                
                r = tensor_scalar_ratio(y)
                dn_dN = d_spectral_index(y)

                outfile1.write("%.10f %.10f %.10f\n" % (r, n, dn_dN))
                outfile1.flush()

                for i in range(NEQS):
                    outfile2.write("%le " % (y[i]))

                outfile2.write("\n")
                outfile2.flush()

                points += 1
                savedone = 0
            else:
                """
                Spectral index out of range
                """
                badncount += 1
        elif calc.ret == "nontrivial":
            r = tensor_scalar_ratio(y)
            n = spectral_index(y)
            dn_dN = d_spectral_index(y)
            outfile1.write("%.10f %.10f %.10f\n" % (r, n, dn_dN))
            outfile1.flush()

            for i in range(NEQS):
                outfile2.write("%le " % (y[i]))

            outfile2.write("%f\n" % (calc.N_efolds))
            outfile2.flush()

            points += 1
            savedone = 0
            nontrivcount += 1

            if SPECTRUM is True:
                # print(y)
                if check_calculate_spectrum(y):
                    print(f"Evaluating spectrum {spec_count}")

                    specnum_s = f"spec_s{spec_count:03d}.dat"
                    specnum_t = f"spec_t{spec_count:03d}.dat"

                    spec_count += 1

                    y_final = np.append(path[:NEQS, 3], N[3])

                    spectrum_status, u_s, u_t = spectrum(
                        y_final,
                        y,
                        calc.N_efolds,
                        derivs1,
                        scalarsys,
                        tensorsys
                    )
                    print(spectrum_status)

                    if spectrum_status:
                        errcount += 1
                    
                    save_data(specnum_s, u_s)
                    save_data(specnum_t, u_t)

        elif calc.ret == "insuff":
            insuffcount += 1
        elif calc.ret == "noconverge":
            noconvcount += 1
        else:
            errcount += 1

        if SAVEPATHS is True:
            """
            Check to see if initial data yielded suitable results for
            the entire path to be generated and saved.  If so, the
            initial data are stored in temporary buffers.
            """
            if SPECTRUM is True:
                """
                If we calculate the spectrum, we want to save the path
                """
                criterion = check_calculate_spectrum(y) and calc.ret == "nontrivial"
            else:
                """
                If we do not, choose different criterion
                """
                criterion = we_should_save_this_path(calc.ret, savedone, points)

            if criterion:
                if SPECTRUM is True:
                    """
                    If we are calculating spectra, we must normalize H
                    here instead of in calcpath.py.
                    """
                    path[0, :] -= path[0, calc.npoints-1]
                    path[1, :] *= y[1]

                fname = f"path{outcount:03d}.dat"
                outcount += 1

                save_path(path, N, calc.npoints, fname)

                savedone = 1

            # if spec_count > 0:
            #     exit()

    print(
        "Done.\n"
        f"points = {points}\n"
        f"iters = {iters}\n"
        f"errcount = {errcount}\n"
        f"asymcount = {asymcount}\n"
        f"nontrivcount = {nontrivcount}\n"
        f"insuffcount = {insuffcount}\n"
        f"noconvcount = {noconvcount}\n"
        f"badncount = {badncount}\n"
        f"errcount = {errcount}\n"
    )

    outfile1.close()
    outfile2.close()

if __name__ == "__main__":
    main()
