# pylint: disable=missing-function-docstring, missing-module-docstring/

from pyccel.decorators import types

def complex_pos():
    import numpy as np
    b = np.sign(complex(1+2j))
    return b

def complex_neg():
    import numpy as np
    b = np.sign(complex(-1-2j))
    return b

def complex64_pos():
    import numpy as np
    b = np.sign(np.complex64(64+64j))
    return b

def complex64_neg():
    import numpy as np
    b = np.sign(np.complex64(-64-64j))
    return b

def complex128_pos():
    import numpy as np
    b = np.sign(np.complex128(128+128j))
    return b

def complex128_neg():
    import numpy as np
    b = np.sign(np.complex128(-128-128j))
    return b

def int16_pos():
    import numpy as np
    b = np.sign(np.int16(16))
    return b

def int16_neg():
    import numpy as np
    b = np.sign(np.int16(-16))
    return b

def int32_pos():
    import numpy as np
    b = np.sign(np.int32(32))
    return b

def int32_neg():
    import numpy as np
    b = np.sign(np.int32(-32))
    return b

def int64_pos():
    import numpy as np
    b = np.sign(np.int64(64))
    return b

def int64_neg():
    import numpy as np
    b = np.sign(np.int64(-64))
    return b

def float_pos():
    import numpy as np
    b = np.sign(np.float(32.32))
    return b

def float_neg():
    import numpy as np
    b = np.sign(np.float(-32.32))
    return b

def float64_pos():
    import numpy as np
    b = np.sign(np.float64(64.64))
    return b

def float64_neg():
    import numpy as np
    b = np.sign(np.float64(-64.64))
    return b

def literal_complex_pos():
    import numpy as np
    b = np.sign(1+2j)
    return b

def literal_complex_neg():
    import numpy as np
    b = np.sign(-1-2j)
    return b

def literal_int_pos():
    import numpy as np
    b = np.sign(42)
    return b

def literal_int_neg():
    import numpy as np
    b = np.sign(-42)
    return b

def literal_float_pos():
    import numpy as np
    b = np.sign(42.42)
    return b

def literal_float_neg():
    import numpy as np
    b = np.sign(-42.42)
    return b

###################
# Arrays tests
###################

@types('int64[:]')
def array_int_1d(x):
    import numpy as np
    x[:] = np.sign(x)

@types('int64[:,:]')
def array_int_2d(x):
    import numpy as np
    x[:,:] = np.sign(x)

@types('float64[:]')
def array_float_1d(x):
    import numpy as np
    x[:] = np.sign(x)

@types('float64[:,:]')
def array_float_2d(x):
    import numpy as np
    x[:,:] = np.sign(x)

@types('complex64[:]')
def array_complex_1d(x):
    import numpy as np
    x[:] = np.sign(x)

@types('complex64[:,:]')
def array_complex_2d(x):
    import numpy as np
    x[:,:] = np.sign(x)
