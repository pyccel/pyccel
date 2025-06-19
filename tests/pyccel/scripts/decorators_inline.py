# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import inline, template

pi = 3.14159

@inline
@template(name='T', types=['int', 'float'])
def add(a:'T',b:'T'):
    return a+b

@inline
def get_powers(s : int):
    return s, s*s, s*s*s

@inline
def power_4(s : 'int|float'):
    tmp = s*s
    return tmp*tmp

@inline
def f(s : 'int|float'):
    return power_4(s) / 2

@inline
def sin_base_1(d : float):
    return np.sin(2*pi*d)

@inline
def fill_pi(a : 'float[:]'):
    pi = 3.14159
    for i in range(a.shape[0]):
        a[i] = pi

def not_inline():
    return 1.602e-19

@inline
def positron_charge():
    # comment
    return -not_inline()

@inline
@template(name='T',types=['int8[:]','int8[:,:]','int8[:,:,:]','int8[:,:,:,:]','int8[:,:,:,:,:]','int8[:,:,:,:,:,:]',\
                           'int16[:]','int16[:,:]','int16[:,:,:]','int16[:,:,:,:]','int16[:,:,:,:,:]','int16[:,:,:,:,:,:]',\
                           'int32[:]','int32[:,:]','int32[:,:,:]','int32[:,:,:,:]','int32[:,:,:,:,:]','int32[:,:,:,:,:,:]',\
                           'int64[:]','int64[:,:]','int64[:,:,:]','int64[:,:,:,:]','int64[:,:,:,:,:]','int64[:,:,:,:,:,:]',\
                           'float32[:]','float32[:,:]','float32[:,:,:]','float32[:,:,:,:]','float32[:,:,:,:,:]','float32[:,:,:,:,:,:]',\
                           'float64[:]','float64[:,:]','float64[:,:,:]','float64[:,:,:,:]','float64[:,:,:,:,:]','float64[:,:,:,:,:,:]'])
def add2(a : 'T', b : 'T', c:'T'):
    c[...] = a+b

if __name__ == '__main__':
    print(get_powers(3))
    a,b,c = get_powers(4)
    print(a,b,c)
    print(power_4(5))
    print(f(3))
    print(f(3.))
    print(sin_base_1(0.5))
    print(sin_base_1(0.7))
    arr = np.empty(4)
    fill_pi(arr)
    print(arr)
    print(add(1,2))
    print(add(1.,2.))
    a1 = np.ones(10)
    b1 = np.ones(10)
    c1 = np.zeros(10)
    a2 = np.ones((10,10))
    b2 = np.ones((10,10))
    c2 = np.zeros((10,10))
    add2(a1, b1, c1)
    add2(a2, b2, c2)
    add2(a2, b2, c2)
