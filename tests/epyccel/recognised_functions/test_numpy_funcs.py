import pytest
from numpy.random import rand, randint
from numpy import isclose

from pyccel.decorators import types
from pyccel.epyccel import epyccel
from conftest import *

# Functions still to be tested:
#    full_like
#    empty_like
#    zeros_like
#    ones_like
#    array
#    # ...
#    norm
#    int
#    real
#    imag
#    float
#    double
#    mod
#    float32
#    float64
#    int32
#    int64
#    complex128
#    complex64
#    matmul
#    prod
#    product
#    linspace
#    diag
#    where
#    cross
#    # ---

def test_fabs_call():
    @types('real')
    def fabs_call(x):
        from numpy import fabs
        return fabs(x)

    f1 = epyccel(fabs_call)
    x = rand()
    assert(isclose(f1(x), fabs_call(x), rtol=1e-15, atol=1e-15))

def test_fabs_phrase():
    @types('real','real')
    def fabs_phrase(x,y):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), fabs_phrase(x,y), rtol=1e-15, atol=1e-15))

@pytest.mark.xfail(reason = "fabs should always return a float")
def test_fabs_return_type():
    @types('int')
    def fabs_return_type(x):
        from numpy import fabs
        a = fabs(x)
        return a

    f1 = epyccel(fabs_return_type)
    x = randint(100)
    assert(isclose(f1(x), fabs_return_type(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x)) == type(fabs_return_type(x))) # pylint: disable=unidiomatic-typecheck

def test_absolute_call():
    @types('real')
    def absolute_call(x):
        from numpy import absolute
        return absolute(x)

    f1 = epyccel(absolute_call)
    x = rand()
    assert(isclose(f1(x), absolute_call(x), rtol=1e-15, atol=1e-15))

def test_absolute_phrase():
    @types('real','real')
    def absolute_phrase(x,y):
        from numpy import absolute
        a = absolute(x)*absolute(y)
        return a

    f2 = epyccel(absolute_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), absolute_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_absolute_return_type():
    @types('int')
    def absolute_return_type(x):
        from numpy import absolute
        a = absolute(x)
        return a

    f1 = epyccel(absolute_return_type)
    x = randint(100)
    assert(isclose(f1(x), absolute_return_type(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x)) == type(absolute_return_type(x).item())) # pylint: disable=unidiomatic-typecheck

def test_sin_call():
    @types('real')
    def sin_call(x):
        from numpy import sin
        return sin(x)

    f1 = epyccel(sin_call)
    x = rand()
    assert(isclose(f1(x), sin_call(x), rtol=1e-15, atol=1e-15))

def test_sin_phrase():
    @types('real','real')
    def sin_phrase(x,y):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), sin_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_cos_call():
    @types('real')
    def cos_call(x):
        from numpy import cos
        return cos(x)

    f1 = epyccel(cos_call)
    x = rand()
    assert(isclose(f1(x), cos_call(x), rtol=1e-15, atol=1e-15))

def test_cos_phrase():
    @types('real','real')
    def cos_phrase(x,y):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), cos_phrase(x,y), rtol=1e-15, atol=1e-15))


def test_tan_call():
    @types('real')
    def tan_call(x):
        from numpy import tan
        return tan(x)

    f1 = epyccel(tan_call)
    x = rand()
    assert(isclose(f1(x), tan_call(x), rtol=1e-15, atol=1e-15))

def test_tan_phrase():
    @types('real','real')
    def tan_phrase(x,y):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), tan_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_exp_call():
    @types('real')
    def exp_call(x):
        from numpy import exp
        return exp(x)

    f1 = epyccel(exp_call)
    x = rand()
    assert(isclose(f1(x), exp_call(x), rtol=1e-15, atol=1e-15))

def test_exp_phrase():
    @types('real','real')
    def exp_phrase(x,y):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), exp_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_log_call():
    @types('real')
    def log_call(x):
        from numpy import log
        return log(x)

    f1 = epyccel(log_call)
    x = rand()
    assert(isclose(f1(x), log_call(x), rtol=1e-15, atol=1e-15))

def test_log_phrase():
    @types('real','real')
    def log_phrase(x,y):
        from numpy import log
        a = log(x)+log(y)
        return a

    f2 = epyccel(log_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), log_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_arcsin_call():
    @types('real')
    def arcsin_call(x):
        from numpy import arcsin
        return arcsin(x)

    f1 = epyccel(arcsin_call)
    x = rand()
    assert(isclose(f1(x), arcsin_call(x), rtol=1e-15, atol=1e-15))

def test_arcsin_phrase():
    @types('real','real')
    def arcsin_phrase(x,y):
        from numpy import arcsin
        a = arcsin(x)+arcsin(y)
        return a

    f2 = epyccel(arcsin_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), arcsin_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_arccos_call():
    @types('real')
    def arccos_call(x):
        from numpy import arccos
        return arccos(x)

    f1 = epyccel(arccos_call)
    x = rand()
    assert(isclose(f1(x), arccos_call(x), rtol=1e-15, atol=1e-15))

def test_arccos_phrase():
    @types('real','real')
    def arccos_phrase(x,y):
        from numpy import arccos
        a = arccos(x)+arccos(y)
        return a

    f2 = epyccel(arccos_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), arccos_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_arctan_call():
    @types('real')
    def arctan_call(x):
        from numpy import arctan
        return arctan(x)

    f1 = epyccel(arctan_call)
    x = rand()
    assert(isclose(f1(x), arctan_call(x), rtol=1e-15, atol=1e-15))

def test_arctan_phrase():
    @types('real','real')
    def arctan_phrase(x,y):
        from numpy import arctan
        a = arctan(x)+arctan(y)
        return a

    f2 = epyccel(arctan_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), arctan_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_sinh_call():
    @types('real')
    def sinh_call(x):
        from numpy import sinh
        return sinh(x)

    f1 = epyccel(sinh_call)
    x = rand()
    assert(isclose(f1(x), sinh_call(x), rtol=1e-15, atol=1e-15))

def test_sinh_phrase():
    @types('real','real')
    def sinh_phrase(x,y):
        from numpy import sinh
        a = sinh(x)+sinh(y)
        return a

    f2 = epyccel(sinh_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), sinh_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_cosh_call():
    @types('real')
    def cosh_call(x):
        from numpy import cosh
        return cosh(x)

    f1 = epyccel(cosh_call)
    x = rand()
    assert(isclose(f1(x), cosh_call(x), rtol=1e-15, atol=1e-15))

def test_cosh_phrase():
    @types('real','real')
    def cosh_phrase(x,y):
        from numpy import cosh
        a = cosh(x)+cosh(y)
        return a

    f2 = epyccel(cosh_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), cosh_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_tanh_call():
    @types('real')
    def tanh_call(x):
        from numpy import tanh
        return tanh(x)

    f1 = epyccel(tanh_call)
    x = rand()
    assert(isclose(f1(x), tanh_call(x), rtol=1e-15, atol=1e-15))

def test_tanh_phrase():
    @types('real','real')
    def tanh_phrase(x,y):
        from numpy import tanh
        a = tanh(x)+tanh(y)
        return a

    f2 = epyccel(tanh_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), tanh_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_arctan2_call():
    @types('real','real')
    def arctan2_call(x,y):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call)
    x = rand()
    y = rand()
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=1e-15, atol=1e-15))

def test_arctan2_phrase():
    @types('real','real','real')
    def arctan2_phrase(x,y,z):
        from numpy import arctan2
        a = arctan2(x,y)+arctan2(x,z)
        return a

    f2 = epyccel(arctan2_phrase)
    x = -rand()
    y = rand()
    z = rand()
    assert(isclose(f2(x,y,z), arctan2_phrase(x,y,z), rtol=1e-15, atol=1e-15))

def test_sqrt_call():
    @types('real')
    def sqrt_call(x):
        from numpy import sqrt
        return sqrt(x)

    f1 = epyccel(sqrt_call)
    x = rand()
    assert(isclose(f1(x), sqrt_call(x), rtol=1e-15, atol=1e-15))

def test_sqrt_phrase():
    @types('real','real')
    def sqrt_phrase(x,y):
        from numpy import sqrt
        a = sqrt(x)*sqrt(y)
        return a

    f2 = epyccel(sqrt_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), sqrt_phrase(x,y), rtol=1e-15, atol=1e-15))


def test_sqrt_return_type():
    @types('real')
    def sqrt_return_type_real(x):
        from numpy import sqrt
        a = sqrt(x)
        return a
    @types('complex')
    def sqrt_return_type_comp(x):
        from numpy import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_real)
    x = rand()
    assert(isclose(f1(x), sqrt_return_type_real(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x)) == type(sqrt_return_type_real(x).item())) # pylint: disable=unidiomatic-typecheck

    f1 = epyccel(sqrt_return_type_comp)
    x = rand() + 1j * rand()
    assert(isclose(f1(x), sqrt_return_type_comp(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x)) == type(sqrt_return_type_comp(x).item())) # pylint: disable=unidiomatic-typecheck


def test_floor_call():
    @types('real')
    def floor_call(x):
        from numpy import floor
        return floor(x)

    f1 = epyccel(floor_call)
    x = rand()
    assert(isclose(f1(x), floor_call(x), rtol=1e-15, atol=1e-15))

def test_floor_phrase():
    @types('real','real')
    def floor_phrase(x,y):
        from numpy import floor
        a = floor(x)*floor(y)
        return a

    f2 = epyccel(floor_phrase)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), floor_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_floor_return_type():
    @types('int')
    def floor_return_type_int(x):
        from numpy import floor
        a = floor(x)
        return a

    @types('real')
    def floor_return_type_real(x):
        from numpy import floor
        a = floor(x)
        return a

    f1 = epyccel(floor_return_type_int)
    x = randint(100)
    assert(isclose(f1(x), floor_return_type_int(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x)) == type(floor_return_type_int(x).item())) # pylint: disable=unidiomatic-typecheck

    f1 = epyccel(floor_return_type_real)
    x = randint(100)
    assert(isclose(f1(x), floor_return_type_real(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x)) == type(floor_return_type_real(x).item())) # pylint: disable=unidiomatic-typecheck

def test_shape_indexed():
    @types('int[:]')
    def test_shape_1d(f):
        from numpy import shape
        return shape(f)[0]

    @types('int[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d)
    f2 = epyccel(test_shape_2d)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

def test_shape_property():
    @types('int[:]')
    def test_shape_1d(f):
        return f.shape[0]

    @types('int[:,:]')
    def test_shape_2d(f):
        a = f.shape
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d)
    f2 = epyccel(test_shape_2d)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(all(isclose(f2(x2), test_shape_2d(x2))))

def test_shape_tuple_output():
    @types('int[:]')
    def test_shape_1d(f):
        from numpy import shape
        s = shape(f)
        return s[0]

    @types('int[:]')
    def test_shape_1d_tuple(f):
        from numpy import shape
        s, = shape(f)
        return s

    @types('int[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a, b = shape(f)
        return a, b

    from numpy import empty
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    f1 = epyccel(test_shape_1d)
    assert(f1(x1)   == test_shape_1d(x1))
    f1_t = epyccel(test_shape_1d_tuple)
    assert(f1_t(x1) == test_shape_1d_tuple(x1))
    f2 = epyccel(test_shape_2d)
    assert(f2(x2)   == test_shape_2d(x2))

def test_shape_real():
    @types('real[:]')
    def test_shape_1d(f):
        from numpy import shape
        b = shape(f)
        return b[0]

    @types('real[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d)
    f2 = epyccel(test_shape_2d)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = float)
    x2 = empty((n2,n3), dtype = float)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

def test_shape_int():
    @types('int[:]')
    def test_shape_1d(f):
        from numpy import shape
        b = shape(f)
        return b[0]

    @types('int[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    f1 = epyccel(test_shape_1d)
    f2 = epyccel(test_shape_2d)

    from numpy import empty
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

def test_shape_bool():
    @types('bool[:]')
    def test_shape_1d(f):
        from numpy import shape
        b = shape(f)
        return b[0]

    @types('bool[:,:]')
    def test_shape_2d(f):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d)
    f2 = epyccel(test_shape_2d)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = bool)
    x2 = empty((n2,n3), dtype = bool)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

def test_full_basic_int():
    @types('int')
    def create_full_shape_1d(n):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_full_shape_2d(n):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int')
    def create_full_val(val):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    @types('int')
    def create_full_arg_names(val):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)

    f_shape_1d  = epyccel(create_full_shape_1d)
    assert(f_shape_1d(size) == create_full_shape_1d(size))

    f_shape_2d  = epyccel(create_full_shape_2d)
    assert(f_shape_2d(size) == create_full_shape_2d(size))

    f_val       = epyccel(create_full_val)
    assert(f_val(size)      == create_full_val(size))
    assert(type(f_val(size)[0])       == type(create_full_val(size)[0].item())) # pylint: disable=unidiomatic-typecheck

    f_arg_names = epyccel(create_full_arg_names)
    assert(f_arg_names(size) == create_full_arg_names(size))
    assert(type(f_arg_names(size)[0]) == type(create_full_arg_names(size)[0].item())) # pylint: disable=unidiomatic-typecheck

def test_full_basic_real():
    @types('int')
    def create_full_shape_1d(n):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_full_shape_2d(n):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    @types('real')
    def create_full_val(val):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    @types('real')
    def create_full_arg_names(val):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = rand()*5

    f_shape_1d  = epyccel(create_full_shape_1d)
    assert(f_shape_1d(size)     == create_full_shape_1d(size))

    f_shape_2d  = epyccel(create_full_shape_2d)
    assert(f_shape_2d(size)     == create_full_shape_2d(size))

    f_val       = epyccel(create_full_val)
    assert(f_val(val)           == create_full_val(val))
    assert(type(f_val(val)[0])       == type(create_full_val(val)[0].item())) # pylint: disable=unidiomatic-typecheck

    f_arg_names = epyccel(create_full_arg_names)
    assert(f_arg_names(val)     == create_full_arg_names(val))
    assert(type(f_arg_names(val)[0]) == type(create_full_arg_names(val)[0].item())) # pylint: disable=unidiomatic-typecheck

@pytest.mark.xfail(reason = "f2py converts bools to int")
def test_full_basic_bool():
    @types('int')
    def create_full_shape_1d(n):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_full_shape_2d(n):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    @types('bool')
    def create_full_val(val):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    @types('bool')
    def create_full_arg_names(val):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = bool(randint(2))

    f_shape_1d  = epyccel(create_full_shape_1d)
    assert(f_shape_1d(size)     == create_full_shape_1d(size))

    f_shape_2d  = epyccel(create_full_shape_2d)
    assert(f_shape_2d(size)     == create_full_shape_2d(size))

    f_val       = epyccel(create_full_val)
    assert(f_val(val)           == create_full_val(val))
    assert(type(f_val(val)[0])       == type(create_full_val(val)[0])) # pylint: disable=unidiomatic-typecheck

    f_arg_names = epyccel(create_full_arg_names)
    assert(f_arg_names(val)     == create_full_arg_names(val))
    assert(type(f_arg_names(val)[0]) == type(create_full_arg_names(val)[0])) # pylint: disable=unidiomatic-typecheck

def test_full_order():
    @types('int','int')
    def create_full_shape_C(n,m):
        from numpy import full, shape
        a = full((n,m),4, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_full_shape_F(n,m):
        from numpy import full, shape
        a = full((n,m),4, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_full_shape_C)
    assert(f_shape_C(size_1,size_2) == create_full_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_full_shape_F)
    assert(f_shape_F(size_1,size_2) == create_full_shape_F(size_1,size_2))

def test_full_dtype():
    @types('int')
    def create_full_val_int_int(val):
        from numpy import full
        a = full(3,val,int)
        return a[0]
    @types('int')
    def create_full_val_int_float(val):
        from numpy import full
        a = full(3,val,float)
        return a[0]
    @types('int')
    def create_full_val_int_complex(val):
        from numpy import full
        a = full(3,val,complex)
        return a[0]
    @types('real')
    def create_full_val_real_int32(val):
        from numpy import full, int32
        a = full(3,val,int32)
        return a[0]
    @types('real')
    def create_full_val_real_float32(val):
        from numpy import full, float32
        a = full(3,val,float32)
        return a[0]
    @types('real')
    def create_full_val_real_float64(val):
        from numpy import full, float64
        a = full(3,val,float64)
        return a[0]
    @types('real')
    def create_full_val_real_complex64(val):
        from numpy import full, complex64
        a = full(3,val,complex64)
        return a[0]
    @types('real')
    def create_full_val_real_complex128(val):
        from numpy import full, complex128
        a = full(3,val,complex128)
        return a[0]

    val_int   = randint(100)
    val_float = rand()*100

    f_int_int   = epyccel(create_full_val_int_int)
    assert(     f_int_int(val_int)        ==      create_full_val_int_int(val_int))
    assert(type(f_int_int(val_int))       == type(create_full_val_int_int(val_int).item())) # pylint: disable=unidiomatic-typecheck

    f_int_float = epyccel(create_full_val_int_float)
    assert(isclose(     f_int_float(val_int)     ,      create_full_val_int_float(val_int), rtol=1e-15, atol=1e-15))
    assert(type(f_int_float(val_int))     == type(create_full_val_int_float(val_int).item())) # pylint: disable=unidiomatic-typecheck

    f_int_complex = epyccel(create_full_val_int_complex)
    assert(isclose(     f_int_complex(val_int)     ,      create_full_val_int_complex(val_int), rtol=1e-15, atol=1e-15))
    assert(type(f_int_complex(val_int))     == type(create_full_val_int_complex(val_int).item())) # pylint: disable=unidiomatic-typecheck

    f_real_int32   = epyccel(create_full_val_real_int32)
    assert(     f_real_int32(val_float)        ==      create_full_val_real_int32(val_float))
    assert(type(f_real_int32(val_float))       == type(create_full_val_real_int32(val_float).item())) # pylint: disable=unidiomatic-typecheck

    f_real_float32   = epyccel(create_full_val_real_float32)
    assert(isclose(     f_real_float32(val_float)       ,      create_full_val_real_float32(val_float), rtol=1e-15, atol=1e-15))
    assert(type(f_real_float32(val_float))       == type(create_full_val_real_float32(val_float).item())) # pylint: disable=unidiomatic-typecheck

    f_real_float64   = epyccel(create_full_val_real_float64)
    assert(isclose(     f_real_float64(val_float)       ,      create_full_val_real_float64(val_float), rtol=1e-15, atol=1e-15))
    assert(type(f_real_float64(val_float))       == type(create_full_val_real_float64(val_float).item())) # pylint: disable=unidiomatic-typecheck

    f_real_complex64   = epyccel(create_full_val_real_complex64)
    assert(isclose(     f_real_complex64(val_float)       ,      create_full_val_real_complex64(val_float), rtol=1e-15, atol=1e-15))
    assert(type(f_real_complex64(val_float))       == type(create_full_val_real_complex64(val_float).item())) # pylint: disable=unidiomatic-typecheck

    f_real_complex128   = epyccel(create_full_val_real_complex128)
    assert(isclose(     f_real_complex128(val_float)       ,      create_full_val_real_complex128(val_float), rtol=1e-15, atol=1e-15))
    assert(type(f_real_complex128(val_float))       == type(create_full_val_real_complex128(val_float).item())) # pylint: disable=unidiomatic-typecheck

def test_full_combined_args():
    def create_full_1_shape():
        from numpy import full, shape
        a = full((2,1),4.0,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_1_val():
        from numpy import full
        a = full((2,1),4.0,int,'F')
        return a[0,0]
    def create_full_2_shape():
        from numpy import full, shape
        a = full((4,2),dtype=float,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_2_val():
        from numpy import full
        a = full((4,2),dtype=float,fill_value=1)
        return a[0,0]
    def create_full_3_shape():
        from numpy import full, shape
        a = full(order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_3_val():
        from numpy import full
        a = full(order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        return a[0,0]

    f1_shape = epyccel(create_full_1_shape)
    f1_val   = epyccel(create_full_1_val)
    assert(f1_shape() == create_full_1_shape())
    assert(f1_val()   == create_full_1_val()  )
    assert(type(f1_val())  == type(create_full_1_val().item())) # pylint: disable=unidiomatic-typecheck

    f2_shape = epyccel(create_full_2_shape)
    f2_val   = epyccel(create_full_2_val)
    assert(f2_shape() == create_full_2_shape()    )
    assert(isclose(f2_val()  , create_full_2_val()      , rtol=1e-15, atol=1e-15))
    assert(type(f2_val())  == type(create_full_2_val().item())) # pylint: disable=unidiomatic-typecheck

    f3_shape = epyccel(create_full_3_shape)
    f3_val   = epyccel(create_full_3_val)
    assert(             f3_shape() ==    create_full_3_shape()      )
    assert(isclose(     f3_val()  ,      create_full_3_val()        , rtol=1e-15, atol=1e-15))
    assert(type(f3_val())  == type(create_full_3_val().item())) # pylint: disable=unidiomatic-typecheck

def test_empty_basic():
    @types('int')
    def create_empty_shape_1d(n):
        from numpy import empty, shape
        a = empty(n)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_empty_shape_2d(n):
        from numpy import empty, shape
        a = empty((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_empty_shape_1d)
    assert(     f_shape_1d(size)      ==      create_empty_shape_1d(size))

    f_shape_2d  = epyccel(create_empty_shape_2d)
    assert(     f_shape_2d(size)      ==      create_empty_shape_2d(size))

def test_empty_order():
    @types('int','int')
    def create_empty_shape_C(n,m):
        from numpy import empty, shape
        a = empty((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_empty_shape_F(n,m):
        from numpy import empty, shape
        a = empty((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_empty_shape_C)
    assert(     f_shape_C(size_1,size_2) == create_empty_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_empty_shape_F)
    assert(     f_shape_F(size_1,size_2) == create_empty_shape_F(size_1,size_2))

def test_empty_dtype():
    def create_empty_val_int():
        from numpy import empty
        a = empty(3,int)
        return a[0]
    def create_empty_val_float():
        from numpy import empty
        a = empty(3,float)
        return a[0]
    def create_empty_val_complex():
        from numpy import empty
        a = empty(3,complex)
        return a[0]
    def create_empty_val_int32():
        from numpy import empty, int32
        a = empty(3,int32)
        return a[0]
    def create_empty_val_float32():
        from numpy import empty, float32
        a = empty(3,float32)
        return a[0]
    def create_empty_val_float64():
        from numpy import empty, float64
        a = empty(3,float64)
        return a[0]
    def create_empty_val_complex64():
        from numpy import empty, complex64
        a = empty(3,complex64)
        return a[0]
    def create_empty_val_complex128():
        from numpy import empty, complex128
        a = empty(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_empty_val_int)
    assert(type(f_int_int())         == type(create_empty_val_int().item())) # pylint: disable=unidiomatic-typecheck

    f_int_float = epyccel(create_empty_val_float)
    assert(type(f_int_float())       == type(create_empty_val_float().item())) # pylint: disable=unidiomatic-typecheck

    f_int_complex = epyccel(create_empty_val_complex)
    assert(type(f_int_complex())     == type(create_empty_val_complex().item())) # pylint: disable=unidiomatic-typecheck

    f_real_int32   = epyccel(create_empty_val_int32)
    assert(type(f_real_int32())      == type(create_empty_val_int32().item())) # pylint: disable=unidiomatic-typecheck

    f_real_float32   = epyccel(create_empty_val_float32)
    assert(type(f_real_float32())    == type(create_empty_val_float32().item())) # pylint: disable=unidiomatic-typecheck

    f_real_float64   = epyccel(create_empty_val_float64)
    assert(type(f_real_float64())    == type(create_empty_val_float64().item())) # pylint: disable=unidiomatic-typecheck

    f_real_complex64   = epyccel(create_empty_val_complex64)
    assert(type(f_real_complex64())  == type(create_empty_val_complex64().item())) # pylint: disable=unidiomatic-typecheck

    f_real_complex128   = epyccel(create_empty_val_complex128)
    assert(type(f_real_complex128()) == type(create_empty_val_complex128().item())) # pylint: disable=unidiomatic-typecheck

def test_empty_combined_args():
    def create_empty_1_shape():
        from numpy import empty, shape
        a = empty((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_1_val():
        from numpy import empty
        a = empty((2,1),int,'F')
        return a[0,0]
    def create_empty_2_shape():
        from numpy import empty, shape
        a = empty((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_2_val():
        from numpy import empty
        a = empty((4,2),dtype=float)
        return a[0,0]
    def create_empty_3_shape():
        from numpy import empty, shape
        a = empty(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_3_val():
        from numpy import empty
        a = empty(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_empty_1_shape)
    f1_val   = epyccel(create_empty_1_val)
    assert(     f1_shape() ==      create_empty_1_shape()      )
    assert(type(f1_val())  == type(create_empty_1_val().item())) # pylint: disable=unidiomatic-typecheck

    f2_shape = epyccel(create_empty_2_shape)
    f2_val   = epyccel(create_empty_2_val)
    assert(all(isclose(     f2_shape(),      create_empty_2_shape()      )))
    assert(type(f2_val())  == type(create_empty_2_val().item())) # pylint: disable=unidiomatic-typecheck

    f3_shape = epyccel(create_empty_3_shape)
    f3_val   = epyccel(create_empty_3_val)
    assert(all(isclose(     f3_shape(),      create_empty_3_shape()      )))
    assert(type(f3_val())  == type(create_empty_3_val().item())) # pylint: disable=unidiomatic-typecheck

def test_ones_basic():
    @types('int')
    def create_ones_shape_1d(n):
        from numpy import ones, shape
        a = ones(n)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_ones_shape_2d(n):
        from numpy import ones, shape
        a = ones((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_ones_shape_1d)
    assert(     f_shape_1d(size)      ==      create_ones_shape_1d(size))

    f_shape_2d  = epyccel(create_ones_shape_2d)
    assert(     f_shape_2d(size)      ==      create_ones_shape_2d(size))

def test_ones_order():
    @types('int','int')
    def create_ones_shape_C(n,m):
        from numpy import ones, shape
        a = ones((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_ones_shape_F(n,m):
        from numpy import ones, shape
        a = ones((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_ones_shape_C)
    assert(     f_shape_C(size_1,size_2) == create_ones_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_ones_shape_F)
    assert(     f_shape_F(size_1,size_2) == create_ones_shape_F(size_1,size_2))

def test_ones_dtype():
    def create_ones_val_int():
        from numpy import ones
        a = ones(3,int)
        return a[0]
    def create_ones_val_float():
        from numpy import ones
        a = ones(3,float)
        return a[0]
    def create_ones_val_complex():
        from numpy import ones
        a = ones(3,complex)
        return a[0]
    def create_ones_val_int32():
        from numpy import ones, int32
        a = ones(3,int32)
        return a[0]
    def create_ones_val_float32():
        from numpy import ones, float32
        a = ones(3,float32)
        return a[0]
    def create_ones_val_float64():
        from numpy import ones, float64
        a = ones(3,float64)
        return a[0]
    def create_ones_val_complex64():
        from numpy import ones, complex64
        a = ones(3,complex64)
        return a[0]
    def create_ones_val_complex128():
        from numpy import ones, complex128
        a = ones(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_ones_val_int)
    assert(     f_int_int()          ==      create_ones_val_int())
    assert(type(f_int_int())         == type(create_ones_val_int().item())) # pylint: disable=unidiomatic-typecheck

    f_int_float = epyccel(create_ones_val_float)
    assert(isclose(     f_int_float()       ,      create_ones_val_float(), rtol=1e-15, atol=1e-15))
    assert(type(f_int_float())       == type(create_ones_val_float().item())) # pylint: disable=unidiomatic-typecheck

    f_int_complex = epyccel(create_ones_val_complex)
    assert(isclose(     f_int_complex()     ,      create_ones_val_complex(), rtol=1e-15, atol=1e-15))
    assert(type(f_int_complex())     == type(create_ones_val_complex().item())) # pylint: disable=unidiomatic-typecheck

    f_real_int32   = epyccel(create_ones_val_int32)
    assert(     f_real_int32()       ==      create_ones_val_int32())
    assert(type(f_real_int32())      == type(create_ones_val_int32().item())) # pylint: disable=unidiomatic-typecheck

    f_real_float32   = epyccel(create_ones_val_float32)
    assert(isclose(     f_real_float32()    ,      create_ones_val_float32(), rtol=1e-15, atol=1e-15))
    assert(type(f_real_float32())    == type(create_ones_val_float32().item())) # pylint: disable=unidiomatic-typecheck

    f_real_float64   = epyccel(create_ones_val_float64)
    assert(isclose(     f_real_float64()    ,      create_ones_val_float64(), rtol=1e-15, atol=1e-15))
    assert(type(f_real_float64())    == type(create_ones_val_float64().item())) # pylint: disable=unidiomatic-typecheck

    f_real_complex64   = epyccel(create_ones_val_complex64)
    assert(isclose(     f_real_complex64()  ,      create_ones_val_complex64(), rtol=1e-15, atol=1e-15))
    assert(type(f_real_complex64())  == type(create_ones_val_complex64().item())) # pylint: disable=unidiomatic-typecheck

    f_real_complex128   = epyccel(create_ones_val_complex128)
    assert(isclose(     f_real_complex128() ,      create_ones_val_complex128(), rtol=1e-15, atol=1e-15))
    assert(type(f_real_complex128()) == type(create_ones_val_complex128().item())) # pylint: disable=unidiomatic-typecheck

def test_ones_combined_args():
    def create_ones_1_shape():
        from numpy import ones, shape
        a = ones((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_1_val():
        from numpy import ones
        a = ones((2,1),int,'F')
        return a[0,0]
    def create_ones_2_shape():
        from numpy import ones, shape
        a = ones((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_2_val():
        from numpy import ones
        a = ones((4,2),dtype=float)
        return a[0,0]
    def create_ones_3_shape():
        from numpy import ones, shape
        a = ones(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_3_val():
        from numpy import ones
        a = ones(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_ones_1_shape)
    f1_val   = epyccel(create_ones_1_val)
    assert(     f1_shape() ==      create_ones_1_shape()      )
    assert(     f1_val()   ==      create_ones_1_val()        )
    assert(type(f1_val())  == type(create_ones_1_val().item())) # pylint: disable=unidiomatic-typecheck

    f2_shape = epyccel(create_ones_2_shape)
    f2_val   = epyccel(create_ones_2_val)
    assert(     f2_shape() ==      create_ones_2_shape()      )
    assert(isclose(     f2_val()  ,      create_ones_2_val()        , rtol=1e-15, atol=1e-15))
    assert(type(f2_val())  == type(create_ones_2_val().item())) # pylint: disable=unidiomatic-typecheck

    f3_shape = epyccel(create_ones_3_shape)
    f3_val   = epyccel(create_ones_3_val)
    assert(     f3_shape() ==      create_ones_3_shape()      )
    assert(isclose(     f3_val()  ,      create_ones_3_val()        , rtol=1e-15, atol=1e-15))
    assert(type(f3_val())  == type(create_ones_3_val().item())) # pylint: disable=unidiomatic-typecheck

def test_zeros_basic():
    @types('int')
    def create_zeros_shape_1d(n):
        from numpy import zeros, shape
        a = zeros(n)
        s = shape(a)
        return len(s),s[0]
    @types('int')
    def create_zeros_shape_2d(n):
        from numpy import zeros, shape
        a = zeros((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_zeros_shape_1d)
    assert(     f_shape_1d(size)      ==      create_zeros_shape_1d(size))

    f_shape_2d  = epyccel(create_zeros_shape_2d)
    assert(     f_shape_2d(size)      ==      create_zeros_shape_2d(size))

def test_zeros_order():
    @types('int','int')
    def create_zeros_shape_C(n,m):
        from numpy import zeros, shape
        a = zeros((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    @types('int','int')
    def create_zeros_shape_F(n,m):
        from numpy import zeros, shape
        a = zeros((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_zeros_shape_C)
    assert(     f_shape_C(size_1,size_2) == create_zeros_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_zeros_shape_F)
    assert(     f_shape_F(size_1,size_2) == create_zeros_shape_F(size_1,size_2))

def test_zeros_dtype():
    def create_zeros_val_int():
        from numpy import zeros
        a = zeros(3,int)
        return a[0]
    def create_zeros_val_float():
        from numpy import zeros
        a = zeros(3,float)
        return a[0]
    def create_zeros_val_complex():
        from numpy import zeros
        a = zeros(3,complex)
        return a[0]
    def create_zeros_val_int32():
        from numpy import zeros, int32
        a = zeros(3,int32)
        return a[0]
    def create_zeros_val_float32():
        from numpy import zeros, float32
        a = zeros(3,float32)
        return a[0]
    def create_zeros_val_float64():
        from numpy import zeros, float64
        a = zeros(3,float64)
        return a[0]
    def create_zeros_val_complex64():
        from numpy import zeros, complex64
        a = zeros(3,complex64)
        return a[0]
    def create_zeros_val_complex128():
        from numpy import zeros, complex128
        a = zeros(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_zeros_val_int)
    assert(     f_int_int()          ==      create_zeros_val_int())
    assert(type(f_int_int())         == type(create_zeros_val_int().item())) # pylint: disable=unidiomatic-typecheck

    f_int_float = epyccel(create_zeros_val_float)
    assert(isclose(     f_int_float()       ,      create_zeros_val_float(), rtol=1e-15, atol=1e-15))
    assert(type(f_int_float())       == type(create_zeros_val_float().item())) # pylint: disable=unidiomatic-typecheck

    f_int_complex = epyccel(create_zeros_val_complex)
    assert(isclose(     f_int_complex()     ,      create_zeros_val_complex(), rtol=1e-15, atol=1e-15))
    assert(type(f_int_complex())     == type(create_zeros_val_complex().item())) # pylint: disable=unidiomatic-typecheck

    f_real_int32   = epyccel(create_zeros_val_int32)
    assert(     f_real_int32()       ==      create_zeros_val_int32())
    assert(type(f_real_int32())      == type(create_zeros_val_int32().item())) # pylint: disable=unidiomatic-typecheck

    f_real_float32   = epyccel(create_zeros_val_float32)
    assert(isclose(     f_real_float32()    ,      create_zeros_val_float32(), rtol=1e-15, atol=1e-15))
    assert(type(f_real_float32())    == type(create_zeros_val_float32().item())) # pylint: disable=unidiomatic-typecheck

    f_real_float64   = epyccel(create_zeros_val_float64)
    assert(isclose(     f_real_float64()    ,      create_zeros_val_float64(), rtol=1e-15, atol=1e-15))
    assert(type(f_real_float64())    == type(create_zeros_val_float64().item())) # pylint: disable=unidiomatic-typecheck

    f_real_complex64   = epyccel(create_zeros_val_complex64)
    assert(isclose(     f_real_complex64()  ,      create_zeros_val_complex64(), rtol=1e-15, atol=1e-15))
    assert(type(f_real_complex64())  == type(create_zeros_val_complex64().item())) # pylint: disable=unidiomatic-typecheck

    f_real_complex128   = epyccel(create_zeros_val_complex128)
    assert(isclose(     f_real_complex128() ,      create_zeros_val_complex128(), rtol=1e-15, atol=1e-15))
    assert(type(f_real_complex128()) == type(create_zeros_val_complex128().item())) # pylint: disable=unidiomatic-typecheck

def test_zeros_combined_args():
    def create_zeros_1_shape():
        from numpy import zeros, shape
        a = zeros((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_1_val():
        from numpy import zeros
        a = zeros((2,1),int,'F')
        return a[0,0]
    def create_zeros_2_shape():
        from numpy import zeros, shape
        a = zeros((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_2_val():
        from numpy import zeros
        a = zeros((4,2),dtype=float)
        return a[0,0]
    def create_zeros_3_shape():
        from numpy import zeros, shape
        a = zeros(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_3_val():
        from numpy import zeros
        a = zeros(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_zeros_1_shape)
    f1_val   = epyccel(create_zeros_1_val)
    assert(     f1_shape() ==      create_zeros_1_shape()      )
    assert(     f1_val()   ==      create_zeros_1_val()        )
    assert(type(f1_val())  == type(create_zeros_1_val().item())) # pylint: disable=unidiomatic-typecheck

    f2_shape = epyccel(create_zeros_2_shape)
    f2_val   = epyccel(create_zeros_2_val)
    assert(     f2_shape() ==      create_zeros_2_shape()      )
    assert(isclose(     f2_val()  ,      create_zeros_2_val()        , rtol=1e-15, atol=1e-15))
    assert(type(f2_val())  == type(create_zeros_2_val().item())) # pylint: disable=unidiomatic-typecheck

    f3_shape = epyccel(create_zeros_3_shape)
    f3_val   = epyccel(create_zeros_3_val)
    assert(     f3_shape() ==      create_zeros_3_shape()      )
    assert(isclose(     f3_val()  ,      create_zeros_3_val()        , rtol=1e-15, atol=1e-15))
    assert(type(f3_val())  == type(create_zeros_3_val().item())) # pylint: disable=unidiomatic-typecheck

def test_array():
    def create_array_list_val():
        from numpy import array
        a = array([[1,2,3],[4,5,6]])
        return a[0,0]
    def create_array_list_shape():
        from numpy import array, shape
        a = array([[1,2,3],[4,5,6]])
        s = shape(a)
        return len(s), s[0], s[1]
    def create_array_tuple_val():
        from numpy import array
        a = array(((1,2,3),(4,5,6)))
        return a[0,0]
    def create_array_tuple_shape():
        from numpy import array, shape
        a = array(((1,2,3),(4,5,6)))
        s = shape(a)
        return len(s), s[0], s[1]
    f1_shape = epyccel(create_array_list_shape)
    f1_val   = epyccel(create_array_list_val)
    assert(f1_shape() == create_array_list_shape())
    assert(f1_val()   == create_array_list_val())
    assert(type(f1_val()) == type(create_array_list_val().item())) # pylint: disable=unidiomatic-typecheck
    f2_shape = epyccel(create_array_tuple_shape)
    f2_val   = epyccel(create_array_tuple_val)
    assert(f2_shape() == create_array_tuple_shape())
    assert(f2_val()   == create_array_tuple_val())
    assert(type(f2_val()) == type(create_array_tuple_val().item())) # pylint: disable=unidiomatic-typecheck

def test_rand_basic():
    def create_val():
        from numpy.random import rand # pylint: disable=reimported
        return rand()

    f1 = epyccel(create_val)
    y = [f1() for i in range(10)]
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

def test_rand_args():

    @types('int')
    def create_array_size_1d(n):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n)
        return shape(a)[0]

    @types('int','int')
    def create_array_size_2d(n,m):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n,m)
        return shape(a)[0], shape(a)[1]

    @types('int','int','int')
    def create_array_size_3d(n,m,p):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n,m,p)
        return shape(a)[0], shape(a)[1], shape(a)[2]

    def create_array_vals_1d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(4)
        return a[0], a[1], a[2], a[3]

    def create_array_vals_2d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(2,2)
        return a[0,0], a[0,1], a[1,0], a[1,1]

    n = randint(10)
    m = randint(10)
    p = randint(5)
    f_1d = epyccel(create_array_size_1d)
    assert( f_1d(n)       == create_array_size_1d(n)      )

    f_2d = epyccel(create_array_size_2d)
    assert( f_2d(n, m)    == create_array_size_2d(n, m)   )

    f_3d = epyccel(create_array_size_3d)
    assert( f_3d(n, m, p) == create_array_size_3d(n, m, p))

    g_1d = epyccel(create_array_vals_1d)
    y = g_1d()
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

    g_2d = epyccel(create_array_vals_2d)
    y = g_2d()
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

def test_rand_expr():
    def create_val():
        from numpy.random import rand # pylint: disable=reimported
        x = 2*rand()
        return x

    f1 = epyccel(create_val)
    y = [f1() for i in range(10)]
    assert(all([yi <  2 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.xfail(reason="a is not allocated")
def test_rand_expr_array():
    def create_array_vals_2d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(2,2)*0.5 + 3
        return a[0,0], a[0,1], a[1,0], a[1,1]

    f2 = epyccel(create_array_vals_2d)
    y = f2()
    assert(all([yi <  3.5 for yi in y]))
    assert(all([yi >= 3   for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

def test_randint_basic():
    @types('int')
    def create_val(high):
        from numpy.random import randint # pylint: disable=reimported
        return randint(high)

    @types('int','int')
    def create_val_low(low, high):
        from numpy.random import randint # pylint: disable=reimported
        return randint(low, high)

    f1 = epyccel(create_val)
    y = [f1(100) for i in range(10)]
    assert(all([yi <  100 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

    f2 = epyccel(create_val_low)
    y = [f2(25, 100) for i in range(10)]
    assert(all([yi <  100 for yi in y]))
    assert(all([yi >= 25 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

def test_randint_expr():
    @types('int')
    def create_val(high):
        from numpy.random import randint # pylint: disable=reimported
        x = 2*randint(high)
        return x

    @types('int','int')
    def create_val_low(low, high):
        from numpy.random import randint # pylint: disable=reimported
        x = 2*randint(low, high)
        return x

    f1 = epyccel(create_val)
    y = [f1(27) for i in range(10)]
    assert(all([yi <  54 for yi in y]))
    assert(all([yi >= 0  for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

    f2 = epyccel(create_val_low)
    y = [f2(21,46) for i in range(10)]
    assert(all([yi <  92 for yi in y]))
    assert(all([yi >= 42 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

def test_sum_int():
    @types('int[:]')
    def sum_call(x):
        from numpy import sum as np_sum
        return np_sum(x)

    f1 = epyccel(sum_call)
    x = randint(99,size=10)
    assert(f1(x) == sum_call(x))

def test_sum_real():
    @types('real[:]')
    def sum_call(x):
        from numpy import sum as np_sum
        return np_sum(x)

    f1 = epyccel(sum_call)
    x = rand(10)
    assert(isclose(f1(x), sum_call(x), rtol=1e-15, atol=1e-15))

def test_sum_phrase():
    @types('real[:]','real[:]')
    def sum_phrase(x,y):
        from numpy import sum as np_sum
        a = np_sum(x)*np_sum(y)
        return a

    f2 = epyccel(sum_phrase)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), sum_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_sum_property():
    @types('int[:]')
    def sum_call(x):
        return x.sum()

    f1 = epyccel(sum_call)
    x = randint(99,size=10)
    assert(f1(x) == sum_call(x))

def test_min_int():
    @types('int[:]')
    def min_call(x):
        from numpy import min as np_min
        return np_min(x)

    f1 = epyccel(min_call)
    x = randint(99,size=10)
    assert(f1(x) == min_call(x))

def test_min_real():
    @types('real[:]')
    def min_call(x):
        from numpy import min as np_min
        return np_min(x)

    f1 = epyccel(min_call)
    x = rand(10)
    assert(isclose(f1(x), min_call(x), rtol=1e-15, atol=1e-15))

def test_min_phrase():
    @types('real[:]','real[:]')
    def min_phrase(x,y):
        from numpy import min as np_min
        a = np_min(x)*np_min(y)
        return a

    f2 = epyccel(min_phrase)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), min_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_min_property():
    @types('int[:]')
    def min_call(x):
        return x.min()

    f1 = epyccel(min_call)
    x = randint(99,size=10)
    assert(f1(x) == min_call(x))

def test_max_int():
    @types('int[:]')
    def max_call(x):
        from numpy import max as np_max
        return np_max(x)

    f1 = epyccel(max_call)
    x = randint(99,size=10)
    assert(f1(x) == max_call(x))

def test_max_real():
    @types('real[:]')
    def max_call(x):
        from numpy import max as np_max
        return np_max(x)

    f1 = epyccel(max_call)
    x = rand(10)
    assert(isclose(f1(x), max_call(x), rtol=1e-15, atol=1e-15))

def test_max_phrase():
    @types('real[:]','real[:]')
    def max_phrase(x,y):
        from numpy import max as np_max
        a = np_max(x)*np_max(y)
        return a

    f2 = epyccel(max_phrase)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), max_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_max_property():
    @types('int[:]')
    def max_call(x):
        return x.max()

    f1 = epyccel(max_call)
    x = randint(99,size=10)
    assert(f1(x) == max_call(x))

