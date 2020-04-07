from pyccel.decorators import types
from pyccel import epyccel
import pytest
from numpy.random import rand, randint

"""
    'full'      : Full,
    'empty'     : Empty,
    'zeros'     : Zeros,
    'ones'      : Ones,
    'full_like' : FullLike,
    'empty_like': EmptyLike,
    'zeros_like': ZerosLike,
    'ones_like' : OnesLike,
    'array'     : Array,
    # ...
    'shape'     : Shape,
    'norm'      : Norm,
    'int'       : NumpyInt,
    'real'      : Real,
    'imag'      : Imag,
    'float'     : NumpyFloat,
    'double'    : Float64,
    'mod'       : Mod,
    'float32'   : Float32,
    'float64'   : Float64,
    'int32'     : Int32,
    'int64'     : Int64,
    'complex128': Complex128,
    'complex64' : Complex64,
    'matmul'    : Matmul,
    'sum'       : NumpySum,
    'prod'      : Prod,
    'product'   : Prod,
    'linspace'  : Linspace,
    'diag'      : Diag,
    'where'     : Where,
    'cross'     : Cross,
    # ---
"""
def test_fabs_call():
    @types('real')
    def fabs_call(x):
        from numpy import fabs
        return fabs(x)

    f1 = epyccel(fabs_call)
    x = rand()
    assert(f1(x) == fabs_call(x))

def test_fabs_phrase():
    @types('real','real')
    def fabs_phrase(x,y):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == fabs_phrase(x,y))

@pytest.mark.xfail
def test_fabs_return_type():
    @types('int')
    def fabs_return_type(x):
        from numpy import fabs
        a = fabs(x)
        return a

    f1 = epyccel(fabs_return_type)
    x = randint(100)
    assert(f1(x) == fabs_return_type(x))
    assert(type(f1(x)) == type(fabs_return_type(x))) # pylint: disable=unidiomatic-typecheck

def test_absolute_call():
    @types('real')
    def absolute_call(x):
        from numpy import absolute
        return absolute(x)

    f1 = epyccel(absolute_call)
    x = rand()
    assert(f1(x) == absolute_call(x))

def test_absolute_phrase():
    @types('real','real')
    def absolute_phrase(x,y):
        from numpy import absolute
        a = absolute(x)*absolute(y)
        return a

    f2 = epyccel(absolute_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == absolute_phrase(x,y))

@pytest.mark.xfail
def test_absolute_return_type():
    @types('int')
    def absolute_return_type(x):
        from numpy import absolute
        a = absolute(x)
        return a

    f1 = epyccel(absolute_return_type)
    x = randint(100)
    assert(f1(x) == absolute_return_type(x))
    assert(type(f1(x)) == type(absolute_return_type(x))) # pylint: disable=unidiomatic-typecheck

def test_sin_call():
    @types('real')
    def sin_call(x):
        from numpy import sin
        return sin(x)

    f1 = epyccel(sin_call)
    x = rand()
    assert(f1(x) == sin_call(x))

def test_sin_phrase():
    @types('real','real')
    def sin_phrase(x,y):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == sin_phrase(x,y))

def test_cos_call():
    @types('real')
    def cos_call(x):
        from numpy import cos
        return cos(x)

    f1 = epyccel(cos_call)
    x = rand()
    assert(f1(x) == cos_call(x))

def test_cos_phrase():
    @types('real','real')
    def cos_phrase(x,y):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == cos_phrase(x,y))

def test_tan_call():
    @types('real')
    def tan_call(x):
        from numpy import tan
        return tan(x)

    f1 = epyccel(tan_call)
    x = rand()
    assert(f1(x) == tan_call(x))

def test_tan_phrase():
    @types('real','real')
    def tan_phrase(x,y):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == tan_phrase(x,y))

def test_exp_call():
    @types('real')
    def exp_call(x):
        from numpy import exp
        return exp(x)

    f1 = epyccel(exp_call)
    x = rand()
    assert(f1(x) == exp_call(x))

def test_exp_phrase():
    @types('real','real')
    def exp_phrase(x,y):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == exp_phrase(x,y))

def test_log_call():
    @types('real')
    def log_call(x):
        from numpy import log
        return log(x)

    f1 = epyccel(log_call)
    x = rand()
    assert(f1(x) == log_call(x))

def test_log_phrase():
    @types('real','real')
    def log_phrase(x,y):
        from numpy import log
        a = log(x)+log(y)
        return a

    f2 = epyccel(log_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == log_phrase(x,y))

def test_arcsin_call():
    @types('real')
    def arcsin_call(x):
        from numpy import arcsin
        return arcsin(x)

    f1 = epyccel(arcsin_call)
    x = rand()
    assert(f1(x) == arcsin_call(x))

def test_arcsin_phrase():
    @types('real','real')
    def arcsin_phrase(x,y):
        from numpy import arcsin
        a = arcsin(x)+arcsin(y)
        return a

    f2 = epyccel(arcsin_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == arcsin_phrase(x,y))

def test_arccos_call():
    @types('real')
    def arccos_call(x):
        from numpy import arccos
        return arccos(x)

    f1 = epyccel(arccos_call)
    x = rand()
    assert(f1(x) == arccos_call(x))

def test_arccos_phrase():
    @types('real','real')
    def arccos_phrase(x,y):
        from numpy import arccos
        a = arccos(x)+arccos(y)
        return a

    f2 = epyccel(arccos_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == arccos_phrase(x,y))

def test_arctan_call():
    @types('real')
    def arctan_call(x):
        from numpy import arctan
        return arctan(x)

    f1 = epyccel(arctan_call)
    x = rand()
    assert(f1(x) == arctan_call(x))

def test_arctan_phrase():
    @types('real','real')
    def arctan_phrase(x,y):
        from numpy import arctan
        a = arctan(x)+arctan(y)
        return a

    f2 = epyccel(arctan_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == arctan_phrase(x,y))

def test_sinh_call():
    @types('real')
    def sinh_call(x):
        from numpy import sinh
        return sinh(x)

    f1 = epyccel(sinh_call)
    x = rand()
    assert(f1(x) == sinh_call(x))

def test_sinh_phrase():
    @types('real','real')
    def sinh_phrase(x,y):
        from numpy import sinh
        a = sinh(x)+sinh(y)
        return a

    f2 = epyccel(sinh_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == sinh_phrase(x,y))

def test_cosh_call():
    @types('real')
    def cosh_call(x):
        from numpy import cosh
        return cosh(x)

    f1 = epyccel(cosh_call)
    x = rand()
    assert(f1(x) == cosh_call(x))

def test_cosh_phrase():
    @types('real','real')
    def cosh_phrase(x,y):
        from numpy import cosh
        a = cosh(x)+cosh(y)
        return a

    f2 = epyccel(cosh_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == cosh_phrase(x,y))

def test_tanh_call():
    @types('real')
    def tanh_call(x):
        from numpy import tanh
        return tanh(x)

    f1 = epyccel(tanh_call)
    x = rand()
    assert(f1(x) == tanh_call(x))

def test_tanh_phrase():
    @types('real','real')
    def tanh_phrase(x,y):
        from numpy import tanh
        a = tanh(x)+tanh(y)
        return a

    f2 = epyccel(tanh_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == tanh_phrase(x,y))

@pytest.mark.xfail
def test_arctan2_call():
    @types('real')
    def arctan2_call(x,y):
        from numpy import arctan2
        return arctan2(x)

    f1 = epyccel(arctan2_call)
    x = rand()
    y = rand()
    assert(f1(x,y) == arctan2_call(x,y))

@pytest.mark.xfail
def test_arctan2_phrase():
    @types('real','real')
    def arctan2_phrase(x,y,z):
        from numpy import arctan2
        a = arctan2(x,y)+arctan2(x,y,z)
        return a

    f2 = epyccel(arctan2_phrase)
    x = rand()
    y = rand()
    z = rand()
    assert(f2(x,y,z) == arctan2_phrase(x,y,z))

def test_sqrt_call():
    @types('real')
    def sqrt_call(x):
        from numpy import sqrt
        return sqrt(x)

    f1 = epyccel(sqrt_call)
    x = rand()
    assert(f1(x) == sqrt_call(x))

def test_sqrt_phrase():
    @types('real','real')
    def sqrt_phrase(x,y):
        from numpy import sqrt
        a = sqrt(x)*sqrt(y)
        return a

    f2 = epyccel(sqrt_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == sqrt_phrase(x,y))


@pytest.mark.xfail
def test_sqrt_return_type():
    @types('real')
    def sqrt_return_type_real(x):
        from numpy import sqrt
        a = sqrt(x)
        return a
    #TODO we should use cmath instead of numpy
    @types('complex')
    def sqrt_return_type_comp(x):
        from numpy import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_real)
    x = rand()
    assert(f1(x) == sqrt_return_type_real(x))
    assert(type(f1(x)) == type(sqrt_return_type_real(x))) # pylint: disable=unidiomatic-typecheck

    f1 = epyccel(sqrt_return_type_comp)
    x = rand() + 1j * rand()
    assert(f1(x) == sqrt_return_type_comp(x))
    assert(type(f1(x)) == type(sqrt_return_type_comp(x))) # pylint: disable=unidiomatic-typecheck


def test_floor_call():
    @types('real')
    def floor_call(x):
        from numpy import floor
        return floor(x)

    f1 = epyccel(floor_call)
    x = rand()
    assert(f1(x) == floor_call(x))

def test_floor_phrase():
    @types('real','real')
    def floor_phrase(x,y):
        from numpy import floor
        a = floor(x)*floor(y)
        return a

    f2 = epyccel(floor_phrase)
    x = rand()
    y = rand()
    assert(f2(x,y) == floor_phrase(x,y))

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
    assert(f1(x) == floor_return_type_int(x))
    assert(type(f1(x)) == type(floor_return_type_int(x))) # pylint: disable=unidiomatic-typecheck

    f1 = epyccel(floor_return_type_real)
    x = randint(100)
    assert(f1(x) == floor_return_type_real(x))
    assert(type(f1(x)) == type(floor_return_type_real(x))) # pylint: disable=unidiomatic-typecheck

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
    assert(f1(x1)==test_shape_1d(x1))
    assert(f2(x2)==test_shape_2d(x2))

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
    assert(f1(x1)==test_shape_1d(x1))
    assert(f2(x2)==test_shape_2d(x2))

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
    assert(f1(x1)==test_shape_1d(x1))
    assert(f2(x2)==test_shape_2d(x2))

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
    assert(f1(x1)==test_shape_1d(x1))
    assert(f2(x2)==test_shape_2d(x2))

#def test_full():
#    def create_full():
#        from numpy import full
#        a = full(3,4)
#        a = full((2,3),True)
#        a = full((1,2),3,float)
#        a = full((2,1),4.0,int,'F')
#        a = full((4,2),dtype=bool)
#
#    f1 = epyccel(create_full)

