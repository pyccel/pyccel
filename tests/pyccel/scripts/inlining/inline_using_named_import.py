from numpy import sin
from pyccel.decorators import inline

@inline
def sin_2(d : float):
    return sin(2*d)
