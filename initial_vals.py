import numpy as np
import numba as nb

from MacroDefinitions import *

@nb.njit(cache = True, fastmath = True)
def _pick_init_vals():
    np.random.seed(0)
    init_vals = np.zeros(NEQS)

    init_vals[0] = 0.0
    init_vals[1] = 1.0
    init_vals[2] = np.random.uniform(0, 0.8)
    init_vals[3] = np.random.uniform(-0.5, 0.5)
    init_vals[4] = np.random.uniform(-0.05, 0.05)

    width = 0.05
    
    for i in range(5, NEQS):
        init_vals[i] = np.random.uniform(-0.5 * width, 0.5 * width)
        width *= 0.1
        
    init_N_efolds = np.random.uniform(NUM_EFOLDS_MIN, NUM_EFOLDS_MAX)
    
    return init_vals, init_N_efolds