#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the math module understood by pyccel
"""

import math

from pyccel.ast.core      import PyccelFunctionDef, Module
from pyccel.ast.datatypes import PythonNativeInt, PythonNativeBool, PythonNativeFloat
from pyccel.ast.internals import PyccelFunction
from pyccel.ast.literals  import Literal, LiteralInteger
from pyccel.ast.variable  import Constant

__all__ = (
    # --- Base classes ---
    'MathFunctionBase',
    'MathFunctionBool',
    'MathFunctionFloat',
    'MathFunctionInt',
    # --- Functions in Math module ---
    'MathAcos',
    'MathAcosh',
    'MathAsin',
    'MathAsinh',
    'MathAtan',
    'MathAtan2',
    'MathAtanh',
    'MathCeil',
    'MathCopysign',
    'MathCos',
    'MathCosh',
    'MathDegrees',
    'MathErf',
    'MathErfc',
    'MathExp',
    'MathExpm1',
    'MathFabs',
    'MathFactorial',
    'MathFloor',
    'MathFmod',
    'MathFrexp',
    'MathFsum',
    'MathGamma',
    'MathGcd',
    'MathHypot',
    'MathIsclose',
    'MathIsfinite',
    'MathIsinf',
    'MathIsnan',
    'MathLcm',
    'MathLdexp',
    'MathLgamma',
    'MathLog',
    'MathLog10',
    'MathLog1p',
    'MathLog2',
    'MathModf',
    'MathPow',
    'MathRadians',
    'MathRemainder',
    'MathSin',
    'MathSinh',
    'MathSqrt',
    'MathTan',
    'MathTanh',
    'MathTrunc',
    # --- Import tools ---
    'math_constants',
    'math_mod',
)

#==============================================================================
# Base classes
#==============================================================================
class MathFunctionBase(PyccelFunction):
    """
    Abstract base class for the Math Functions.

    A super-class from which all functions in the `math` library
    should inherit.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ()
    _shape = None


class MathFunctionFloat(MathFunctionBase):
    """
    Super-class from which functions returning a float inherit.

    A super-class from which functions in the `math` library which
    return a float should inherit.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ()
    name = 'float'
    _class_type = PythonNativeFloat()


class MathFunctionInt(MathFunctionBase):
    """
    Super-class from which functions returning an integer inherit.

    A super-class from which functions in the `math` library which
    return an integer should inherit.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ()
    name = 'int'
    _class_type = PythonNativeInt()


class MathFunctionBool(MathFunctionBase):
    """
    Super-class from which functions returning a boolean inherit.

    A super-class from which functions in the `math` library which
    return a boolean should inherit.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ()
    name = 'bool'
    _class_type = PythonNativeBool()

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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, y, x):
        super().__init__(y, x)

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
    def __init__(self, x):
        super().__init__(x)


class MathCopysign(MathFunctionFloat):
    """
    Class representing a call to the `math.copysign` function.

    A class which represents a call to the `copysign` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The first expression passed as argument to the function.
    y : TypedAstNode
        The second expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'copysign'
    def __init__(self, x, y):
        super().__init__(x, y)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


class MathExpm1   (MathFunctionFloat):
    """
    Class representing a call to the `math.expm1` function.

    A class which represents a call to the `expm1` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'expm1'
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


class MathHypot(MathFunctionFloat):
    """
    Class representing a call to the `math.hypot` function.

    A class which represents a call to the `hypot` function from the `math` library.

    Parameters
    ----------
    *args : TypedAstNode
        The arguments passed to the function.
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
        The first expression passed as argument to the function.
    i : TypedAstNode
        The second expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'ldexp'
    def __init__(self, x, i):
        super().__init__(x, i)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


class MathPow(MathFunctionFloat):
    """
    Class representing a call to the `math.pow` function.

    A class which represents a call to the `pow` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The first expression passed as argument to the function.
    y : TypedAstNode
        The second expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'pow'
    def __init__(self, x, y):
        super().__init__(x, y)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


class MathRemainder(MathFunctionFloat):
    """
    Class representing a call to the `math.remainder` function.

    A class which represents a call to the `remainder` function from the `math` library.

    Parameters
    ----------
    x : TypedAstNode
        The first expression passed as argument to the function.
    y : TypedAstNode
        The second expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'remainder'
    def __init__(self, x, y):
        super().__init__(x, y)

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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)

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
    def __init__(self, x):
        super().__init__(x)


class MathGcd(MathFunctionInt):
    """
    Class representing a call to the `math.gcd` function.

    A class which represents a call to the `gcd` function from the `math` library.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ()
    name = 'gcd'


class MathLcm(MathFunctionInt):
    """
    Class representing a call to the `math.lcm` function.

    A class which represents a call to the `lcm` function from the `math` library.

    Parameters
    ----------
    *args : tuple of TypedAstNode
        The arguments passed to the function.
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

    def __new__(cls, x):
        if isinstance(x, Literal):
            return LiteralInteger(math.ceil(x.python_value), dtype = cls._class_type)
        else:
            return super().__new__(cls)

    def __init__(self, x):
        super().__init__(x)


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

    def __new__(cls, x):
        if isinstance(x, Literal):
            return LiteralInteger(math.floor(x.python_value), dtype = cls._class_type)
        else:
            return super().__new__(cls)

    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)

# Boolean result
class MathIsclose(MathFunctionBool):
    """
    Class representing a call to the `math.isclose` function.

    A class which represents a call to the `isclose` function from the `math` library.

    Parameters
    ----------
    a : TypedAstNode
        The first expression passed as argument to the function.
    b : TypedAstNode
        The second expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isclose'
    def __init__(self, a, b):
        super().__init__(a, b)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)


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
    def __init__(self, x):
        super().__init__(x)

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
    'e'  : Constant(PythonNativeFloat(), 'e'  , value=math.e  ),
    'pi' : Constant(PythonNativeFloat(), 'pi' , value=math.pi ),
    'inf': Constant(PythonNativeFloat(), 'inf', value=math.inf),
    'nan': Constant(PythonNativeFloat(), 'nan', value=math.nan),
    'tau': Constant(PythonNativeFloat(), 'tau', value=2.*math.pi),
}

math_mod = Module('math',
        variables = math_constants.values(),
        funcs     = math_functions)
