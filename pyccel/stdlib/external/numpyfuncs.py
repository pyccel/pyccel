
from pyccel.macros import pass_lhs, declare_passed_args, declare_dummy_args, mlen, val, ndarray_slice, cast
from pyccel.decorators import inline, template
# Add overload decorator
# Add implicite casting decorator
# Fix Fortran interface bug when it returns different types
# allow the user to create ndarray_slice using tuples to avoid using the macro

@inline
@pass_lhs(names=['a'])
@declare_passed_args({'name':'a', 'rank':mlen('shape'), 'dtype':val('dtype'), 'order':val('order')})
@declare_dummy_args({'name':'a', 'allocatable':True, 'lbound':False}, 
                    {'name':'shape', 'intent':'in'})
@template('T', types=['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'complex64', 'complex128'])
def empty(a:'T[1:15]', shape:'int[:]', dtype:'str'='float64', order:'str'='C'):
    from pyccel.fstdlib import allocate
    allocate(ndarray_slice(a, shape))

@inline
@pass_lhs(names=['a'])
@declare_passed_args({'name':'a', 'rank':mlen('shape'), 'dtype':val('dtype'), 'order':val('order')})
@declare_dummy_args({'name':'a', 'allocatable':True, 'lbound':False}, 
                    {'name':'shape', 'intent':'in'})
@template('T', types=['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'complex64', 'complex128'])
def zeros(a:'T[1:15]', shape:'int[:]', dtype:'str'='float64', order:'str'='C'):
    from pyccel.fstdlib import allocate
    allocate(ndarray_slice(a, shape))
    a[...] = cast(a,0)

@inline
@pass_lhs(names=['a'])
@declare_passed_args({'name':'a', 'rank':mlen('shape'), 'dtype':val('dtype'), 'order':val('order')})
@declare_dummy_args({'name':'a', 'allocatable':True, 'lbound':False}, 
                    {'name':'shape', 'intent':'in'})
@template('T', types=['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'complex64', 'complex128'])
def ones(a:'T[1:15]', shape:'int[:]', dtype:'str'='float64', order:'str'='C'):
    from pyccel.fstdlib import allocate
    allocate(ndarray_slice(a, shape))
    a[...] = cast(a,1)

@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_ceil(x: 'T'):
    from numpy import int64

    result = cast(x, int64(x))
    if not(x <= 0. or x == result):
        result = result + cast(x, 1)
    return result

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_sin(x: 'T'):
    from pyccel.fstdlib import sin
    return sin(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_cos(x: 'T'):
    from pyccel.fstdlib import cos
    return cos(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_tan(x: 'T'):
    from pyccel.fstdlib import tan
    return tan(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_sinh(x: 'T'):
    from pyccel.fstdlib import sinh
    return sinh(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_cosh(x: 'T'):
    from pyccel.fstdlib import cosh
    return cosh(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_tanh(x: 'T'):
    from pyccel.fstdlib import tanh
    return tanh(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_sqrt(x: 'T'):
    from pyccel.fstdlib import sqrt
    return sqrt(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_log(x: 'T'):
    from pyccel.fstdlib import log
    return log(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_log10(x: 'T'):
    from pyccel.fstdlib import log10
    return log10(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_log2(x: 'T'):
    from pyccel.fstdlib import log
    return log(x)/log(2.)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_arcsin(x: 'T'):
    from pyccel.fstdlib import asin
    return asin(x)

@inline
@pure
@elemental
@template('T', types=['float32', 'float64'])
def np_arccos(x: 'T'):
    from pyccel.fstdlib import acos
    return acos(x)

#if __name__ == '__main__':
x1 = empty((10,10), dtype='float64', order='C')
x2 = zeros((10,10), dtype='float64', order='C')
x3 = ones((10,10), dtype='complex64', order='C')

a1 = np_ceil(4.3)
a2 = np_cos(1.3)
a3 = np_sin(2.1)
a4 = np_tan(1.)
a5 = np_cosh(1.3)
a6 = np_sinh(2.1)
a7 = np_tanh(1.)
a8 = np_log(1.)
a9 = np_log2(2.)
a10 = np_log10(10.)
a11 = np_arcsin(1.)
a12 = np_arccos(-1.)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

