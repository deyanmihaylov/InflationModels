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