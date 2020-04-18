import numpy as np
import sys
from scipy.integrate import solve_ivp

import pygsl.rng

my_random = pygsl.rng.ranlxd2()
my_random.set(0)

np.random.seed(0)

SPECTRUM = False

NEQS = 8
kmax = 20000

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

def derivs(t, y):
    dydN = np.zeros(NEQS , dtype=float , order='C')
    
    if y[2] >= 1.0:
        dydN = np.zeros(NEQS , dtype=float , order='C')
    else:
        if y[2] > VERYSMALLNUM:
            dydN[0] = - np.sqrt ( y[2] / (4 * np.pi ) )
        else:
            dydN[0] = 0.0
        
        dydN[1] = y[1] * y[2]
        dydN[2] = y[2] * ( y[3] + 2.0 * y[2] )
        dydN[3] = 2.0 * y[4] - 5.0 * y[2] * y[3] - 12.0 * y[2] * y[2]
        
        for i in range(4, NEQS-1):
            dydN[i] = ( 0.5 * (i-3) * y[3] + (i-4) * y[2] ) * y[i] + y[i+1]
            
        dydN[NEQS-1] = ( 0.5 * (NEQS-4) * y[3] + (NEQS-5) * y[2] ) * y[NEQS-1]
    
    return dydN

def int_de(y, N, Nend, derivs):
    h = 1e-6
    ydoub = y.copy()
        
    Nsol = solve_ivp(derivs, (N , Nend), ydoub, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False , first_step=h)
    
    sol_status = Nsol['status']
    sol_length = len ( Nsol['t'] )
    sol_x = Nsol['t']
    sol_Y = Nsol['y']
    
    if sol_length > kmax:
        sol_length = kmax
        sol_x = sol_x [0:kmax]
        sol_Y = sol_Y [: , 0:kmax]
        
    return sol_status , sol_length , sol_x , sol_Y

VERYSMALLNUM = 1e-18
LOTSOFEFOLDS = 1000.0

def calcpath(Nefolds, y, path, N, count):
    retval = "internal_error"
    
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
    Nend = 0.0
    
    z, kount, xp, yp = int_de(y, Nstart, Nend, derivs)

    print(z, kount, xp, yp)
    exit()
    
    y = yp [:, -1]
    
    if z < 0:
        retval = "internal_error"
        z = 0
#         goto end;
    else:
#         Find when epsilon passes through unity.
        i = check_convergence ( yp , kount )
        
        if not i:
#             We never found an end to inflation, so we must be at a late-time attractor.
            if y[2] > SMALLNUM or y[3] < 0.0:
#                 The system did not evolve to a known asymptote.
                retval = "noconverge"
            else:
                retval = "asymptote"
        else:
#             We found an end to inflation: integrate backwards Nefolds e-folds from that point.
            Nstart = xp[i-2] - xp[i-1]
            Nend = Nefolds
            
            y = yp [ : , i-2 ].copy()
            
            z , kount , xp , yp = int_de ( y , Nstart , Nend , derivs )
            
            if z < 0:
                retval = "internal_error"
                z = 0
            else:
                if check_convergence ( yp , kount ):
                    retval = "insuff"
                else:
                    retval = "nontrivial"
                    
    print (retval)

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
                print(f"asymcount = {asymcount}, nontrivcount = {nontrivcount}, insuffcount = {insuffcount}, \
                        noconvcount = {noconvcount}, badncount = {badncount}, errcount = {errcount}")
                print(iters)
            else:
                print(".")

        yinit, calc.Nefolds = pick_init_vals()

        y = yinit.copy()

        calc.ret = calcpath(calc.Nefolds, y, path, N, calc.npoints)

        break

if __name__ == "__main__":
    main()






























"""



def check_convergence (yy , kount):
    for i in range (kount):
        if numpy.absolute ( yy [ 2 , i ] ) >= 1.0:
            return i
            break
        
    return 0

OUTFILE1_NAME = "nr.dat"  
OUTFILE2_NAME = "esigma.dat"

NEQS = 8
kmax = 20000

NUMPOINTS = 20000

NUMEFOLDSMAX = 60.0
NUMEFOLDSMIN = 40.0

SMALLNUM = 0.0001

LOTSOFEFOLDS = 1000.0

calc = Calc()

path = numpy.zeros (( NEQS , kmax ))

# Open or create output files
try:
    outfile1 = open (OUTFILE1_NAME, "w")
except IOError as e:
    print ("Could not open file" , str(OUTFILE1_NAME) , ", errno =" , str(e) , ".")
    
try:
    outfile2 = open (OUTFILE2_NAME, "w")
except IOError as e:
    print ("Could not open file" , str(OUTFILE2_NAME) , ", errno =" , str(e) , ".")
    
# Allocate buffers

y = numpy.zeros(NEQS, dtype=float, order='C')
yinit = numpy.zeros(NEQS, dtype=float, order='C')

# iters = total number of iterations
# points = points saved with n < NMAX
# asymcount = points with 0 < n < NMAX , r = 0
# nontrivcount = nontrivial points
# insuffcount = points where slow roll breaks down before N efolds
# noconvcount = points that do not converge to either a late
# time attractor or end of inflation.
    
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

# Currently this loop is set to depend on the number of
# nontrivial points.  This can be changed to depend on total
# number of models, or other criteria.

deyancount = 0

while nontrivcount < NUMPOINTS:
    deyancount += 1
    
    iters += 1
    
    if iters % 100 == 0:
        if iters % 1000 == 0:
            print ("\n asymcount =" , asymcount , ",nontrivcount =" , nontrivcount , ",insuffcount =" , insuffcount , ", noconvcount =" , noconvcount , ", badncount =" , badncount , ", errcount =" , errcount , "\n")
            print ("\n", iters)
        else:
            print (".")
            
# Select a random initial condition such that the slow
# roll hierarchy converges.

    yinit , calc.Nefolds = pick_init_vals()
    
    y = yinit.copy()
    
    calc.ret = calcpath ( calc.Nefolds , y , path , calc.npoints)
    
    if deyancount > 100:
        break


"""









