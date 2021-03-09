@types( 'float32[:]', 'float32', 'int')
@types( 'double[:]' , 'double' , 'int')
def array_real_1d_scalar_add( x, a, x_len ):
    for i in range(x_len):
        x[i] += a

import numpy as np

x = np.ones(4)
array_real_1d_scalar_add( x, 3.0, 4 )

print(x)
