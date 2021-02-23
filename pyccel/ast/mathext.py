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
    def _set_shape(self):
        self._shape = ()

class MathFunctionFloat(MathFunctionBase):
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeReal()

class MathFunctionInt(MathFunctionBase):
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeInteger()

class MathFunctionBool(MathFunctionBase):
    __slots__ = ()
    def _set_dtype(self):
        self._dtype = NativeBool()

#==============================================================================
# Functions that return one value
#==============================================================================

# Floating-point result
class MathAcos    (MathFunctionFloat): __slots__ = ()
class MathAcosh   (MathFunctionFloat): __slots__ = ()
class MathAsin    (MathFunctionFloat): __slots__ = ()
class MathAsinh   (MathFunctionFloat): __slots__ = ()
class MathAtan    (MathFunctionFloat): __slots__ = ()
class MathAtan2   (MathFunctionFloat): __slots__ = ()
class MathAtanh   (MathFunctionFloat): __slots__ = ()
class MathCopysign(MathFunctionFloat): __slots__ = ()
class MathCos     (MathFunctionFloat): __slots__ = ()
class MathCosh    (MathFunctionFloat): __slots__ = ()
class MathErf     (MathFunctionFloat): __slots__ = ()
class MathErfc    (MathFunctionFloat): __slots__ = ()
class MathExp     (MathFunctionFloat): __slots__ = ()
class MathExpm1   (MathFunctionFloat): __slots__ = ()
class MathFabs    (MathFunctionFloat): __slots__ = ()
class MathFmod    (MathFunctionFloat): __slots__ = ()
class MathFsum    (MathFunctionFloat): __slots__ = ()
class MathGamma   (MathFunctionFloat): __slots__ = ()
class MathHypot   (MathFunctionFloat): __slots__ = ()
class MathLdexp   (MathFunctionFloat): __slots__ = ()
class MathLgamma  (MathFunctionFloat): __slots__ = ()
class MathLog     (MathFunctionFloat): __slots__ = ()
class MathLog10   (MathFunctionFloat): __slots__ = ()
class MathLog1p   (MathFunctionFloat): __slots__ = ()
class MathLog2    (MathFunctionFloat): __slots__ = ()
class MathPow     (MathFunctionFloat): __slots__ = ()
class MathSin     (MathFunctionFloat): __slots__ = ()
class MathSinh    (MathFunctionFloat): __slots__ = ()
class MathSqrt    (MathFunctionFloat): __slots__ = ()
class MathTan     (MathFunctionFloat): __slots__ = ()
class MathTanh    (MathFunctionFloat): __slots__ = ()
class MathRemainder (MathFunctionFloat): __slots__ = ()

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

class MathCeil     (MathFunctionInt): __slots__ = ()
class MathFloor    (MathFunctionInt): __slots__ = ()
class MathTrunc    (MathFunctionInt): __slots__ = ()

# Boolean result
class MathIsclose (MathFunctionBool): __slots__ = ()
class MathIsfinite(MathFunctionBool): __slots__ = ()
class MathIsinf   (MathFunctionBool): __slots__ = ()
class MathIsnan   (MathFunctionBool): __slots__ = ()

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
