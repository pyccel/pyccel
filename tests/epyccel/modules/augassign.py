# pylint: disable=missing-function-docstring, missing-module-docstring/

from pyccel.decorators import types

# += 

@types('int[:]')
def augassign_add_1d_int(a):
    import numpy as np
    b = a
    b += 42
    return b

@types('float[:]')
def augassign_add_1d_float(a):
    import numpy as np
    b = a
    b += 4.2
    return b

@types('complex[:]')
def augassign_add_1d_complex(a):
    import numpy as np
    b = a
    b += (4.0 + 2.0j)
    return b

@types('int[:,:]')
def augassign_add_2d_int(a):
    import numpy as np
    b = a
    b += 42
    return b

@types('float[:,:]')
def augassign_add_2d_float(a):
    import numpy as np
    b = a
    b += 4.2
    return b

@types('complex[:,:]')
def augassign_add_2d_complex(a):
    import numpy as np
    b = a
    b += (4.0 + 2.0j)
    return b

# -=

@types('int[:]')
def augassign_sub_1d_int(a):
    import numpy as np
    b = a
    b -= 42
    return b

@types('float[:]')
def augassign_sub_1d_float(a):
    import numpy as np
    b = a
    b -= 4.2
    return b

@types('complex[:]')
def augassign_sub_1d_complex(a):
    import numpy as np
    b = a
    b -= (4.0 + 2.0j)
    return b

@types('int[:,:]')
def augassign_sub_2d_int(a):
    import numpy as np
    b = a
    b -= 42
    return b

@types('float[:,:]')
def augassign_sub_2d_float(a):
    import numpy as np
    b = a
    b -= 4.2
    return b

@types('complex[:,:]')
def augassign_sub_2d_complex(a):
    import numpy as np
    b = a
    b -= (4.0 + 2.0j)
    return b

# *=

@types('int[:]')
def augassign_mul_1d_int(a):
    import numpy as np
    b = a
    b *= 42
    return b

@types('float[:]')
def augassign_mul_1d_float(a):
    import numpy as np
    b = a
    b *= 4.2
    return b

@types('complex[:]')
def augassign_mul_1d_complex(a):
    import numpy as np
    b = a
    b *= (4.0 + 2.0j)
    return b

@types('int[:,:]')
def augassign_mul_2d_int(a):
    import numpy as np
    b = a
    b *= 42
    return b

@types('float[:,:]')
def augassign_mul_2d_float(a):
    import numpy as np
    b = a
    b *= 4.2
    return b

@types('complex[:,:]')
def augassign_mul_2d_complex(a):
    import numpy as np
    b = a
    b *= (4.0 + 2.0j)
    return b

# /=

@types('int[:]')
def augassign_div_1d_int(a):
    import numpy as np
    b = a
    b /= 42
    return b

@types('float[:]')
def augassign_div_1d_float(a):
    import numpy as np
    b = a
    b /= 4.2
    return b

@types('complex[:]')
def augassign_div_1d_complex(a):
    import numpy as np
    b = a
    b /= (4.0 + 2.0j)
    return b

@types('int[:,:]')
def augassign_div_2d_int(a):
    import numpy as np
    b = a
    b /= 42
    return b

@types('float[:,:]')
def augassign_div_2d_float(a):
    import numpy as np
    b = a
    b /= 4.2
    return b

@types('complex[:,:]')
def augassign_div_2d_complex(a):
    import numpy as np
    b = a
    b /= (4.0 + 2.0j)
    return b
