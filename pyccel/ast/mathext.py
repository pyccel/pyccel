#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the math module understood by pyccel
"""

import math

from pyccel.ast.core      import PyccelFunctionDef, Module
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeFloat
from pyccel.ast.internals import PyccelInternalFunction
from pyccel.ast.variable  import Constant

__all__ = (
    'math_mod',
    'math_constants',
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
# Base classes
#==============================================================================
class MathFunctionBase(PyccelInternalFunction):
    """Abstract base class for the Math Functions"""
    __slots__ = ()
    _shape = None
    _rank  = 0
    _order = None

class MathFunctionFloat(MathFunctionBase):
    __slots__ = ()
    name = 'float'
    _dtype = NativeFloat()
    _precision = -1

class MathFunctionInt(MathFunctionBase):
    __slots__ = ()
    name = 'int'
    _dtype = NativeInteger()
    _precision = -1

class MathFunctionBool(MathFunctionBase):
    __slots__ = ()
    name = 'bool'
    _dtype = NativeBool()
    _precision = -1

#==============================================================================
# Functions that return one value
#==============================================================================

# Floating-point result
class MathAcos    (MathFunctionFloat):
    """Represent a call to the acos function in the Math library"""
    __slots__ = ()
    name = 'acos'
class MathAcosh   (MathFunctionFloat):
    """Represent a call to the acosh function in the Math library"""
    __slots__ = ()
    name = 'acosh'
class MathAsin    (MathFunctionFloat):
    """Represent a call to the asin function in the Math library"""
    __slots__ = ()
    name = 'asin'
class MathAsinh   (MathFunctionFloat):
    """Represent a call to the asinh function in the Math library"""
    __slots__ = ()
    name = 'asinh'
class MathAtan    (MathFunctionFloat):
    """Represent a call to the atan function in the Math library"""
    __slots__ = ()
    name = 'atan'
class MathAtan2   (MathFunctionFloat):
    """Represent a call to the atan2 function in the Math library"""
    __slots__ = ()
    name = 'atan2'
class MathAtanh   (MathFunctionFloat):
    """Represent a call to the atanh function in the Math library"""
    __slots__ = ()
    name = 'atanh'
class MathCopysign(MathFunctionFloat):
    """Represent a call to the copysign function in the Math library"""
    __slots__ = ()
    name = 'copysign'
class MathCos     (MathFunctionFloat):
    """Represent a call to the cos function in the Math library"""
    __slots__ = ()
    name = 'cos'
class MathCosh    (MathFunctionFloat):
    """Represent a call to the cosh function in the Math library"""
    __slots__ = ()
    name = 'cosh'
class MathErf     (MathFunctionFloat):
    """Represent a call to the erf function in the Math library"""
    __slots__ = ()
    name = 'erf'
class MathErfc    (MathFunctionFloat):
    """Represent a call to the erfc function in the Math library"""
    __slots__ = ()
    name = 'erfc'
class MathExp     (MathFunctionFloat):
    """Represent a call to the exp function in the Math library"""
    __slots__ = ()
    name = 'exp'
class MathExpm1   (MathFunctionFloat):
    """Represent a call to the expm1 function in the Math library"""
    __slots__ = ()
    name = 'expm1'
class MathFabs    (MathFunctionFloat):
    """Represent a call to the fabs function in the Math library"""
    __slots__ = ()
    name = 'fabs'
class MathFmod    (MathFunctionFloat):
    """Represent a call to the fmod function in the Math library"""
    __slots__ = ()
    name = 'fmod'
class MathFsum    (MathFunctionFloat):
    """Represent a call to the fsum function in the Math library"""
    __slots__ = ()
    name = 'fsum'
class MathGamma   (MathFunctionFloat):
    """Represent a call to the gamma function in the Math library"""
    __slots__ = ()
    name = 'gamma'
class MathHypot   (MathFunctionFloat):
    """Represent a call to the hypot function in the Math library"""
    __slots__ = ()
    name = 'hypot'
class MathLdexp   (MathFunctionFloat):
    """Represent a call to the ldexp function in the Math library"""
    __slots__ = ()
    name = 'ldexp'
class MathLgamma  (MathFunctionFloat):
    """Represent a call to the lgamma function in the Math library"""
    __slots__ = ()
    name = 'lgamma'
class MathLog     (MathFunctionFloat):
    """Represent a call to the log function in the Math library"""
    __slots__ = ()
    name = 'log'
class MathLog10   (MathFunctionFloat):
    """Represent a call to the log10 function in the Math library"""
    __slots__ = ()
    name = 'log10'
class MathLog1p   (MathFunctionFloat):
    """Represent a call to the log1p function in the Math library"""
    __slots__ = ()
    name = 'log1p'
class MathLog2    (MathFunctionFloat):
    """Represent a call to the log2 function in the Math library"""
    __slots__ = ()
    name = 'log2'
class MathPow     (MathFunctionFloat):
    """Represent a call to the pow function in the Math library"""
    __slots__ = ()
    name = 'pow'
class MathSin     (MathFunctionFloat):
    """Represent a call to the sin function in the Math library"""
    __slots__ = ()
    name = 'sin'
class MathSinh    (MathFunctionFloat):
    """Represent a call to the sinh function in the Math library"""
    __slots__ = ()
    name = 'sinh'
class MathSqrt    (MathFunctionFloat):
    """Represent a call to the sqrt function in the Math library"""
    __slots__ = ()
    name = 'sqrt'
class MathTan     (MathFunctionFloat):
    """Represent a call to the tan function in the Math library"""
    __slots__ = ()
    name = 'tan'
class MathTanh    (MathFunctionFloat):
    """Represent a call to the tanh function in the Math library"""
    __slots__ = ()
    name = 'tanh'
class MathRemainder (MathFunctionFloat):
    """Represent a call to the remainder function in the Math library"""
    __slots__ = ()
    name = 'remainder'

class MathRadians (MathFunctionFloat):
    """Represent a call to the radians function in the Math library"""
    __slots__ = ()
    name = 'radians'
class MathDegrees (MathFunctionFloat):
    """Represent a call to the degrees function in the Math library"""
    __slots__ = ()
    name = 'degrees'

# Integer result
class MathFactorial(MathFunctionInt):
    """Represent a call to the factorial function in the Math library"""
    __slots__ = ()
    name = 'factorial'
class MathGcd      (MathFunctionInt):
    """Represent a call to the gcd function in the Math library"""
    __slots__ = ()
    name = 'gcd'
class MathLcm      (MathFunctionInt):
    """Represent a call to the lcm function in the Math library"""
    __slots__ = ()
    name = 'lcm'

class MathCeil     (MathFunctionInt):
    """Represent a call to the ceil function in the Math library"""
    __slots__ = ()
    name = 'ceil'
class MathFloor    (MathFunctionInt):
    """Represent a call to the floor function in the Math library"""
    __slots__ = ()
    name = 'floor'
class MathTrunc    (MathFunctionInt):
    """Represent a call to the trunc function in the Math library"""
    __slots__ = ()
    name = 'trunc'

# Boolean result
class MathIsclose (MathFunctionBool):
    """Represent a call to the isclose function in the Math library"""
    __slots__ = ()
    name = 'isclose'
class MathIsfinite(MathFunctionBool):
    """Represent a call to the isfinite function in the Math library"""
    __slots__ = ()
    name = 'isfinite'
class MathIsinf   (MathFunctionBool):
    """Represent a call to the isinf function in the Math library"""
    __slots__ = ()
    name = 'isinf'
class MathIsnan   (MathFunctionBool):
    """Represent a call to the isnan function in the Math library"""
    __slots__ = ()
    name = 'isnan'

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
    name = 'frexp'

# TODO
class MathModf(MathFunctionBase):
    """
    modf(x)

    Return the fractional and integer parts of x.  Both results carry the sign
    of x and are floats.
    """
    __slots__ = ()
    name = 'modf'

#==============================================================================
# Dictionary to map math functions to classes above
#==============================================================================

_base_classes = (
    'MathFunctionBase',
    'MathFunctionFloat',
    'MathFunctionInt',
    'MathFunctionBool'
)

math_functions = [PyccelFunctionDef(v.name, v) for k, v in globals().copy().items() \
        if k.startswith('Math') and (k not in _base_classes)]

#==============================================================================
# Constants
#==============================================================================
math_constants = {
    'e'  : Constant('float', 'e'  , value=math.e  ),
    'pi' : Constant('float', 'pi' , value=math.pi ),
    'inf': Constant('float', 'inf', value=math.inf),
    'nan': Constant('float', 'nan', value=math.nan),
    'tau': Constant('float', 'tau', value=2.*math.pi),
}

math_mod = Module('math',
        variables = math_constants.values(),
        funcs     = math_functions)
