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
    _class_type = NativeFloat()


class MathFunctionInt(MathFunctionBase):
    __slots__ = ()
    name = 'int'
    _dtype = NativeInteger()
    _precision = -1
    _class_type = NativeInteger()


class MathFunctionBool(MathFunctionBase):
    __slots__ = ()
    name = 'bool'
    _dtype = NativeBool()
    _precision = -1
    _class_type = NativeBool()

#==============================================================================
# Functions that return one value
#==============================================================================

# Floating-point result
class MathAcos(MathFunctionFloat):
    """
    Class representing a call to the `math.acos` function.

    A class which represents a call to the `acos` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'acos'


class MathAcosh(MathFunctionFloat):
    """
    Class representing a call to the `math.acosh` function.

    A class which represents a call to the `acosh` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'acosh'


class MathAsin(MathFunctionFloat):
    """
    Class representing a call to the `math.asin` function.

    A class which represents a call to the `asin` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'asin'


class MathAsinh(MathFunctionFloat):
    """
    Class representing a call to the `math.asinh` function.

    A class which represents a call to the `asinh` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'asinh'


class MathAtan(MathFunctionFloat):
    """
    Class representing a call to the `math.atan` function.

    A class which represents a call to the `atan` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'atan'


class MathAtan2   (MathFunctionFloat):
    """
    Class representing a call to the `math.atan2` function.

    A class which represents a call to the `atan2` function from the `math` library.

    Parameters
    ----------
    y : TypedAstNode
        The first expression passed as argument to the function.
    x : TypedAstNode
        The second expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'atan2'


class MathAtanh(MathFunctionFloat):
    """
    Class representing a call to the `math.atanh` function.

    A class which represents a call to the `atanh` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'atanh'


class MathCopysign(MathFunctionFloat):
    """
    Class representing a call to the `math.copysign` function.

    A class which represents a call to the `copysign` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'copysign'


class MathCos(MathFunctionFloat):
    """
    Class representing a call to the `math.cos` function.

    A class which represents a call to the `cos` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'cos'


class MathCosh(MathFunctionFloat):
    """
    Class representing a call to the `math.cosh` function.

    A class which represents a call to the `cosh` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'cosh'


class MathErf(MathFunctionFloat):
    """
    Class representing a call to the `math.erf` function.

    A class which represents a call to the `erf` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'erf'


class MathErfc(MathFunctionFloat):
    """
    Class representing a call to the `math.erfc` function.

    A class which represents a call to the `erfc` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'erfc'


class MathExp(MathFunctionFloat):
    """
    Class representing a call to the `math.exp` function.

    A class which represents a call to the `exp` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'exp'


class MathExpm1   (MathFunctionFloat):
    """Represent a call to the expm1 function in the Math library"""
    __slots__ = ()
    name = 'expm1'


class MathFabs(MathFunctionFloat):
    """
    Class representing a call to the `math.fabs` function.

    A class which represents a call to the `fabs` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'fabs'


class MathFmod(MathFunctionFloat):
    """
    Class representing a call to the `math.fmod` function.

    A class which represents a call to the `fmod` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'fmod'


class MathFsum(MathFunctionFloat):
    """
    Class representing a call to the `math.fsum` function.

    A class which represents a call to the `fsum` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'fsum'


class MathGamma(MathFunctionFloat):
    """
    Class representing a call to the `math.gamma` function.

    A class which represents a call to the `gamma` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'gamma'


class MathHypot(MathFunctionFloat):
    """
    Class representing a call to the `math.hypot` function.

    A class which represents a call to the `hypot` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'hypot'


class MathLdexp(MathFunctionFloat):
    """
    Class representing a call to the `math.ldexp` function.

    A class which represents a call to the `ldexp` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'ldexp'


class MathLgamma(MathFunctionFloat):
    """
    Class representing a call to the `math.lgamma` function.

    A class which represents a call to the `lgamma` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'lgamma'


class MathLog(MathFunctionFloat):
    """
    Class representing a call to the `math.log` function.

    A class which represents a call to the `log` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'log'


class MathLog10   (MathFunctionFloat):
    """
    Class representing a call to the `math.log10` function.

    A class which represents a call to the `log10` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'log10'


class MathLog1p   (MathFunctionFloat):
    """
    Class representing a call to the `math.log1p` function.

    A class which represents a call to the `log1p` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'log1p'


class MathLog2    (MathFunctionFloat):
    """
    Class representing a call to the `math.log2` function.

    A class which represents a call to the `log2` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'log2'


class MathPow(MathFunctionFloat):
    """
    Class representing a call to the `math.pow` function.

    A class which represents a call to the `pow` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'pow'


class MathSin(MathFunctionFloat):
    """
    Class representing a call to the `math.sin` function.

    A class which represents a call to the `sin` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'sin'


class MathSinh(MathFunctionFloat):
    """
    Class representing a call to the `math.sinh` function.

    A class which represents a call to the `sinh` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'sinh'


class MathSqrt(MathFunctionFloat):
    """
    Class representing a call to the `math.sqrt` function.

    A class which represents a call to the `sqrt` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'sqrt'


class MathTan(MathFunctionFloat):
    """
    Class representing a call to the `math.tan` function.

    A class which represents a call to the `tan` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'tan'


class MathTanh(MathFunctionFloat):
    """
    Class representing a call to the `math.tanh` function.

    A class which represents a call to the `tanh` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'tanh'


class MathRemainder(MathFunctionFloat):
    """
    Class representing a call to the `math.remainder` function.

    A class which represents a call to the `remainder` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'remainder'

class MathRadians(MathFunctionFloat):
    """
    Class representing a call to the `math.radians` function.

    A class which represents a call to the `radians` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'radians'


class MathDegrees(MathFunctionFloat):
    """
    Class representing a call to the `math.degrees` function.

    A class which represents a call to the `degrees` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'degrees'

# Integer result
class MathFactorial(MathFunctionInt):
    """
    Class representing a call to the `math.factorial` function.

    A class which represents a call to the `factorial` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'factorial'


class MathGcd(MathFunctionInt):
    """
    Class representing a call to the `math.gcd` function.

    A class which represents a call to the `gcd` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'gcd'


class MathLcm(MathFunctionInt):
    """
    Class representing a call to the `math.lcm` function.

    A class which represents a call to the `lcm` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'lcm'

class MathCeil(MathFunctionInt):
    """
    Class representing a call to the `math.ceil` function.

    A class which represents a call to the `ceil` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'ceil'


class MathFloor(MathFunctionInt):
    """
    Class representing a call to the `math.floor` function.

    A class which represents a call to the `floor` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'floor'


class MathTrunc(MathFunctionInt):
    """
    Class representing a call to the `math.trunc` function.

    A class which represents a call to the `trunc` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'trunc'

# Boolean result
class MathIsclose(MathFunctionBool):
    """
    Class representing a call to the `math.isclose` function.

    A class which represents a call to the `isclose` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isclose'


class MathIsfinite(MathFunctionBool):
    """
    Class representing a call to the `math.isfinite` function.

    A class which represents a call to the `isfinite` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isfinite'


class MathIsinf(MathFunctionBool):
    """
    Class representing a call to the `math.isinf` function.

    A class which represents a call to the `isinf` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isinf'


class MathIsnan(MathFunctionBool):
    """
    Class representing a call to the `math.isnan` function.

    A class which represents a call to the `isnan` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
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
