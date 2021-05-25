import numpy as np
def fill_a( r: float, a: 'int[:]' ):

    if ( r == 0.0 ):
        return 0

    for i in range ( len(a) ):
        a[i] = 1.0 / r

    return 1

def get_sum():
    r = 4.0
    a = np.empty(10,dtype=int)
    fill_a ( r, a )
    result = 0
    for ai in a:
        result += ai
    return result
