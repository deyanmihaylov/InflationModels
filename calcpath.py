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

def derivs(t, y):
    dydN = np.zeros(NEQS, dtype=float, order='C')
    
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