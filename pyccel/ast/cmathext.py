#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the cmath module understood by pyccel
"""

import cmath

from pyccel.ast.builtins  import PythonReal, PythonImag
from pyccel.ast.core      import PyccelFunctionDef, Module
from pyccel.ast.datatypes import NativeBool, NativeFloat, NativeComplex
from pyccel.ast.internals import PyccelInternalFunction
from pyccel.ast.literals  import LiteralInteger, LiteralFloat
from pyccel.ast.operators import PyccelOr
from pyccel.ast.variable  import Constant

from .mathext import math_constants, MathAtan2, MathFunctionBase
from .mathext import MathIsfinite, MathIsinf, MathIsnan

__all__ = (
        'CmathFunctionBool',
        'CmathFunctionComplex',
        'CmathAcos',
        'CmathAcosh',
        'CmathAsin',
        'CmathAsinh',
        'CmathAtan',
        'CmathAtanh',
        'CmathCos',
        'CmathCosh',
        'CmathExp',
        'CmathSin',
        'CmathSinh',
        'CmathSqrt',
        'CmathTan',
        'CmathTanh',
        'CmathIsclose',
        'CmathIsfinite',
        'CmathIsinf',
        'CmathIsnan',
        'CmathPhase',
        'CmathPolar',
        'CmathRect',
        'cmath_mod'
    )

class CmathFunctionBool(MathFunctionBase):
    __slots__ = ()
    _dtype = NativeBool()
    _precision = -1

class CmathFunctionComplex(MathFunctionBase):
    """
    Super-class from which functions returning a complex number inherit.

    Paramters
    ---------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    _dtype = NativeComplex()
    _precision = -1
    _shape = None
    _rank  = 0
    _order = None

    def __init__(self, z : 'PyccelAstNode'):
        super().__init__(z)

#==============================================================================
# Functions that return one value
#==============================================================================

#==============================================================================
# Complex results
#==============================================================================

class CmathAcos    (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.acos` function.

    A class which represents a call to the `acos` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'acos'

#==============================================================================

class CmathAcosh   (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.acosh` function.

    A class which represents a call to the `acosh` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'acosh'

#==============================================================================

class CmathAsin    (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.asin` function.

    A class which represents a call to the `asin` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'asin'

#==============================================================================

class CmathAsinh   (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.asinh` function.

    A class which represents a call to the `asinh` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'asinh'

#==============================================================================

class CmathAtan    (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.atan` function.

    A class which represents a call to the `atan` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'atan'

#==============================================================================

class CmathAtanh    (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.atanh` function.

    A class which represents a call to the `atanh` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'atanh'

#==============================================================================

class CmathCos     (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.cos` function.

    A class which represents a call to the `cos` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'cos'

#==============================================================================

class CmathCosh    (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.cosh` function.

    A class which represents a call to the `cosh` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'cosh'

#==============================================================================

class CmathExp     (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.exp` function.

    A class which represents a call to the `exp` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'exp'

#==============================================================================

class CmathSin     (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.sin` function.

    A class which represents a call to the `sin` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'sin'

#==============================================================================

class CmathSinh    (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.sinh` function.

    A class which represents a call to the `sinh` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'sinh'

#==============================================================================

class CmathSqrt    (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.sqrt` function.

    A class which represents a call to the `sqrt` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'sqrt'

#==============================================================================

class CmathTan     (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.tan` function.

    A class which represents a call to the `tan` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'tan'

#==============================================================================

class CmathTanh    (CmathFunctionComplex):
    """
    Class representing a call to the `cmath.tanh` function.

    A class which represents a call to the `tanh` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'tanh'

#==============================================================================
# Boolean results
#==============================================================================

class CmathIsclose (CmathFunctionBool):
    """
    Class representing a call to the `cmath.isclose` function.

    A class which represents a call to the `isclose` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isclose'

#==============================================================================

class CmathIsfinite(CmathFunctionBool):
    """
    Class representing a call to the `cmath.isfinite` function.

    A class which represents a call to the `isfinite` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isfinite'
    def __new__(cls, z):
        if z.dtype is not NativeComplex():
            return MathIsfinite(z)
        else:
            return PyccelOr(MathIsfinite(PythonImag(z)), MathIsfinite(PythonReal(z)))

#==============================================================================

class CmathIsinf   (CmathFunctionBool):
    """
    Class representing a call to the `cmath.isinf` function.

    A class which represents a call to the `isinf` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isinf'
    def __new__(cls, z):
        if z.dtype is not NativeComplex():
            return MathIsinf(z)
        else:
            return PyccelOr(MathIsinf(PythonImag(z)), MathIsinf(PythonReal(z)))

#==============================================================================

class CmathIsnan   (CmathFunctionBool):
    """
    Class representing a call to the `cmath.isnan` function.

    A class which represents a call to the `isnan` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isnan'
    def __new__(cls, z):
        if z.dtype is not NativeComplex():
            return MathIsnan(z)
        else:
            return PyccelOr(MathIsnan(PythonImag(z)), MathIsnan(PythonReal(z)))

#==============================================================================
# Dictionary to map math functions to classes above
#==============================================================================

class CmathPhase(PyccelInternalFunction):
    """
    Class representing a call to the `cmath.phase` function.

    A class which represents a call to the `phase` function from the `cmath` library
    which calculates the phase angle of a complex number.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'phase'
    def __new__(cls, z):
        if z.dtype is not NativeComplex():
            return LiteralFloat(0.0)
        else:
            return MathAtan2(PythonImag(z), PythonReal(z))

class CmathPolar(PyccelInternalFunction):
    """
    Class representing a call to the `cmath.polar` function.

    A class which represents a call to the `polar` function from the `cmath` library.

    Parameters
    ----------
    z : PyccelAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'polar'
    _dtype = NativeFloat()
    _precision = -1
    _shape = (LiteralInteger(2),)
    _rank  = 1
    _order = None

    def __init__(self, z):
        super().__init__(z)

class CmathRect(PyccelInternalFunction):
    """
    Class representing a call to the `cmath.rect` function.

    A class which represents a call to the `rect` function from the `cmath` library.

    Parameters
    ----------
    r : PyccelAstNode
        The first argument to the function, representing the radius.
    phi : PyccelAstNode
        The second argument to the function, representing the polar angle.
    """
    __slots__ = ()
    name = 'rect'
    _dtype = NativeComplex()
    _precision = -1
    _shape = None
    _rank  = 0
    _order = None
    def __init__(self, r, phi):
        super().__init__(r, phi)

#==============================================================================
# Dictionary to map cmath functions to classes above
#==============================================================================

cmath_functions = [PyccelFunctionDef(v.name, v) for v in
        (CmathAcos, CmathAcosh, CmathAsin, CmathAsinh, CmathAtan, CmathAtanh, CmathCos, CmathCosh,
            CmathExp, CmathIsclose, CmathIsfinite, CmathIsinf, CmathIsnan, CmathPhase,
            CmathPolar, CmathRect, CmathSin, CmathSinh, CmathSqrt, CmathTan, CmathTanh)]

cmath_constants = { **math_constants,
    'infj': Constant('complex', 'infj', value=cmath.infj),
    'nanj': Constant('complex', 'nanj', value=cmath.nanj),
    }

cmath_mod = Module('cmath',
        variables = math_constants.values(),
        funcs     = cmath_functions)
