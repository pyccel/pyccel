#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import math

from pyccel.ast.variable  import Constant
from pyccel.ast.internals import PyccelInternalFunction
from pyccel.ast.datatypes import (NativeInteger, NativeBool, NativeReal,
                                  default_precision)

__all__ = (
    'math_constants',
    'math_functions',
    # ---
    'MathFunctionBase',
    'MathFunctionFloat',
    'MathFunctionInt',
    'MathFunctionBool',
    # ---
    'MathAcos',
    'MathAcosh',
    'MathAsin',
    'MathAsinh',
    'MathAtan',
    'MathAtan2',
    'MathAtanh',
    'MathCopysign',
    'MathCos',
    'MathCosh',
    'MathDegrees',
    'MathErf',
    'MathErfc',
    'MathExp',
    'MathExpm1',
    'MathFabs',
    'MathFmod',
    'MathFsum',
    'MathGamma',
    'MathHypot',
    'MathLdexp',
    'MathLgamma',
    'MathLog',
    'MathLog10',
    'MathLog1p',
    'MathLog2',
    'MathPow',
    'MathRadians',
    'MathSin',
    'MathSinh',
    'MathSqrt',
    'MathTan',
    'MathTanh',
    'MathRemainder',
    # ---
    'MathCeil',
    'MathFactorial',
    'MathFloor',
    'MathGcd',
    'MathLcm',
    'MathTrunc',
    # ---
    'MathIsclose',
    'MathIsfinite',
    'MathIsinf',
    'MathIsnan',
    # ---
    'MathFrexp', # TODO
    'MathModf',  # TODO
)

#==============================================================================
# Constants
#==============================================================================
math_constants = {
    'e'  : Constant('real', 'e'  , value=math.e  ),
    'pi' : Constant('real', 'pi' , value=math.pi ),
    'inf': Constant('real', 'inf', value=math.inf),
    'nan': Constant('real', 'nan', value=math.nan),
    'tau': Constant('real', 'tau', value=2.*math.pi),
}

#==============================================================================
# Base classes
#==============================================================================
class MathFunctionBase(PyccelInternalFunction):
    """Abstract base class for the Math Functions"""
    __slots__ = ()
    _shape = ()
    _rank  = 0
    _order = None

class MathFunctionFloat(MathFunctionBase):
    __slots__ = ()
    _dtype = NativeReal()
    _precision = default_precision['real']

class MathFunctionInt(MathFunctionBase):
    __slots__ = ()
    _dtype = NativeInteger()
    _precision = default_precision['integer']

class MathFunctionBool(MathFunctionBase):
    __slots__ = ()
    _dtype = NativeBool()
    _precision = default_precision['bool']

#==============================================================================
# Functions that return one value
#==============================================================================

# Floating-point result
class MathAcos    (MathFunctionFloat):
    """Represent a call to the acos function in the Math library"""
    __slots__ = ()
class MathAcosh   (MathFunctionFloat):
    """Represent a call to the acosh function in the Math library"""
    __slots__ = ()
class MathAsin    (MathFunctionFloat):
    """Represent a call to the asin function in the Math library"""
    __slots__ = ()
class MathAsinh   (MathFunctionFloat):
    """Represent a call to the asinh function in the Math library"""
    __slots__ = ()
class MathAtan    (MathFunctionFloat):
    """Represent a call to the atan function in the Math library"""
    __slots__ = ()
class MathAtan2   (MathFunctionFloat):
    """Represent a call to the atan2 function in the Math library"""
    __slots__ = ()
class MathAtanh   (MathFunctionFloat):
    """Represent a call to the atanh function in the Math library"""
    __slots__ = ()
class MathCopysign(MathFunctionFloat):
    """Represent a call to the copysign function in the Math library"""
    __slots__ = ()
class MathCos     (MathFunctionFloat):
    """Represent a call to the cos function in the Math library"""
    __slots__ = ()
class MathCosh    (MathFunctionFloat):
    """Represent a call to the cosh function in the Math library"""
    __slots__ = ()
class MathErf     (MathFunctionFloat):
    """Represent a call to the erf function in the Math library"""
    __slots__ = ()
class MathErfc    (MathFunctionFloat):
    """Represent a call to the erfc function in the Math library"""
    __slots__ = ()
class MathExp     (MathFunctionFloat):
    """Represent a call to the exp function in the Math library"""
    __slots__ = ()
class MathExpm1   (MathFunctionFloat):
    """Represent a call to the expm1 function in the Math library"""
    __slots__ = ()
class MathFabs    (MathFunctionFloat):
    """Represent a call to the fabs function in the Math library"""
    __slots__ = ()
class MathFmod    (MathFunctionFloat):
    """Represent a call to the fmod function in the Math library"""
    __slots__ = ()
class MathFsum    (MathFunctionFloat):
    """Represent a call to the fsum function in the Math library"""
    __slots__ = ()
class MathGamma   (MathFunctionFloat):
    """Represent a call to the gamma function in the Math library"""
    __slots__ = ()
class MathHypot   (MathFunctionFloat):
    """Represent a call to the hypot function in the Math library"""
    __slots__ = ()
class MathLdexp   (MathFunctionFloat):
    """Represent a call to the ldexp function in the Math library"""
    __slots__ = ()
class MathLgamma  (MathFunctionFloat):
    """Represent a call to the lgamma function in the Math library"""
    __slots__ = ()
class MathLog     (MathFunctionFloat):
    """Represent a call to the log function in the Math library"""
    __slots__ = ()
class MathLog10   (MathFunctionFloat):
    """Represent a call to the log10 function in the Math library"""
    __slots__ = ()
class MathLog1p   (MathFunctionFloat):
    """Represent a call to the log1p function in the Math library"""
    __slots__ = ()
class MathLog2    (MathFunctionFloat):
    """Represent a call to the log2 function in the Math library"""
    __slots__ = ()
class MathPow     (MathFunctionFloat):
    """Represent a call to the pow function in the Math library"""
    __slots__ = ()
class MathSin     (MathFunctionFloat):
    """Represent a call to the sin function in the Math library"""
    __slots__ = ()
class MathSinh    (MathFunctionFloat):
    """Represent a call to the sinh function in the Math library"""
    __slots__ = ()
class MathSqrt    (MathFunctionFloat):
    """Represent a call to the sqrt function in the Math library"""
    __slots__ = ()
class MathTan     (MathFunctionFloat):
    """Represent a call to the tan function in the Math library"""
    __slots__ = ()
class MathTanh    (MathFunctionFloat):
    """Represent a call to the tanh function in the Math library"""
    __slots__ = ()
class MathRemainder (MathFunctionFloat):
    """Represent a call to the remainder function in the Math library"""
    __slots__ = ()

class MathRadians (MathFunctionFloat):
    """Represent a call to the radians function in the Math library"""
    __slots__ = ()
class MathDegrees (MathFunctionFloat):
    """Represent a call to the degrees function in the Math library"""
    __slots__ = ()

# Integer result
class MathFactorial(MathFunctionInt):
    """Represent a call to the factorial function in the Math library"""
    __slots__ = ()
class MathGcd      (MathFunctionInt):
    """Represent a call to the gcd function in the Math library"""
    __slots__ = ()
class MathLcm      (MathFunctionInt):
    """Represent a call to the lcm function in the Math library"""
    __slots__ = ()

class MathCeil     (MathFunctionInt):
    """Represent a call to the ceil function in the Math library"""
    __slots__ = ()
class MathFloor    (MathFunctionInt):
    """Represent a call to the floor function in the Math library"""
    __slots__ = ()
class MathTrunc    (MathFunctionInt):
    """Represent a call to the trunc function in the Math library"""
    __slots__ = ()

# Boolean result
class MathIsclose (MathFunctionBool):
    """Represent a call to the isclose function in the Math library"""
    __slots__ = ()
class MathIsfinite(MathFunctionBool):
    """Represent a call to the isfinite function in the Math library"""
    __slots__ = ()
class MathIsinf   (MathFunctionBool):
    """Represent a call to the isinf function in the Math library"""
    __slots__ = ()
class MathIsnan   (MathFunctionBool):
    """Represent a call to the isnan function in the Math library"""
    __slots__ = ()

#==============================================================================
# Functions that return two values
#==============================================================================

# TODO
class MathFrexp(MathFunctionBase):
    """
    frexp(x)

    Return the mantissa and exponent of x, as pair (m, e).
    m is a float and e is an int, such that x = m * 2.**e.
    If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0.
    """
    __slots__ = ()

# TODO
class MathModf(MathFunctionBase):
    """
    modf(x)

    Return the fractional and integer parts of x.  Both results carry the sign
    of x and are floats.
    """
    __slots__ = ()

#==============================================================================
# Dictionary to map math functions to classes above
#==============================================================================

_base_classes = (
    'MathFunctionBase',
    'MathFunctionFloat',
    'MathFunctionInt',
    'MathFunctionBool'
)

math_functions = {}
for k, v in globals().copy().items():
    if k.startswith('Math') and (k not in _base_classes):
        name = k[4:].lower()
        math_functions[name] = v
