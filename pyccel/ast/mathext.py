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
    _shape = ()
    _rank  = 0
    _order = None
    def __init__(self, *args):
        super().__init__(*args)

class MathFunctionFloat(MathFunctionBase):
    _dtype = NativeReal()
    _precision = default_precision['real']
    def __init__(self, *args):
        super().__init__(*args)

class MathFunctionInt(MathFunctionBase):
    _dtype = NativeInteger()
    _precision = default_precision['integer']
    def __init__(self, *args):
        super().__init__(*args)

class MathFunctionBool(MathFunctionBase):
    _dtype = NativeBool()
    _precision = default_precision['bool']
    def __init__(self, *args):
        super().__init__(*args)

#==============================================================================
# Functions that return one value
#==============================================================================

# Floating-point result
class MathAcos    (MathFunctionFloat): pass
class MathAcosh   (MathFunctionFloat): pass
class MathAsin    (MathFunctionFloat): pass
class MathAsinh   (MathFunctionFloat): pass
class MathAtan    (MathFunctionFloat): pass
class MathAtan2   (MathFunctionFloat): pass
class MathAtanh   (MathFunctionFloat): pass
class MathCopysign(MathFunctionFloat): pass
class MathCos     (MathFunctionFloat): pass
class MathCosh    (MathFunctionFloat): pass
class MathErf     (MathFunctionFloat): pass
class MathErfc    (MathFunctionFloat): pass
class MathExp     (MathFunctionFloat): pass
class MathExpm1   (MathFunctionFloat): pass
class MathFabs    (MathFunctionFloat): pass
class MathFmod    (MathFunctionFloat): pass
class MathFsum    (MathFunctionFloat): pass
class MathGamma   (MathFunctionFloat): pass
class MathHypot   (MathFunctionFloat): pass
class MathLdexp   (MathFunctionFloat): pass
class MathLgamma  (MathFunctionFloat): pass
class MathLog     (MathFunctionFloat): pass
class MathLog10   (MathFunctionFloat): pass
class MathLog1p   (MathFunctionFloat): pass
class MathLog2    (MathFunctionFloat): pass
class MathPow     (MathFunctionFloat): pass
class MathSin     (MathFunctionFloat): pass
class MathSinh    (MathFunctionFloat): pass
class MathSqrt    (MathFunctionFloat): pass
class MathTan     (MathFunctionFloat): pass
class MathTanh    (MathFunctionFloat): pass
class MathRemainder (MathFunctionFloat): pass

class MathRadians (MathFunctionFloat):
    """Represent a call to the radians function in the Math library"""
class MathDegrees (MathFunctionFloat):
    """Represent a call to the degrees function in the Math library"""

# Integer result
class MathFactorial(MathFunctionInt):
    """Represent a call to the factorial function in the Math library"""
class MathGcd      (MathFunctionInt):
    """Represent a call to the gcd function in the Math library"""
class MathLcm      (MathFunctionInt):
    """Represent a call to the lcm function in the Math library"""

class MathCeil     (MathFunctionInt): pass
class MathFloor    (MathFunctionInt): pass
class MathTrunc    (MathFunctionInt): pass

# Boolean result
class MathIsclose (MathFunctionBool): pass
class MathIsfinite(MathFunctionBool): pass
class MathIsinf   (MathFunctionBool): pass
class MathIsnan   (MathFunctionBool): pass

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

# TODO
class MathModf(MathFunctionBase):
    """
    modf(x)

    Return the fractional and integer parts of x.  Both results carry the sign
    of x and are floats.
    """

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
