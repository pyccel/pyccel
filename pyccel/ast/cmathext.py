#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the cmath module understood by pyccel
"""

import cmath

from pyccel.ast.builtins  import PythonReal, PythonImag
from pyccel.ast.core      import PyccelFunctionDef, Module
from pyccel.ast.datatypes import PythonNativeBool, PythonNativeFloat, PythonNativeComplex
from pyccel.ast.datatypes import PrimitiveComplexType, HomogeneousTupleType
from pyccel.ast.internals import PyccelFunction
from pyccel.ast.literals  import LiteralInteger
from pyccel.ast.operators import PyccelAnd, PyccelOr
from pyccel.ast.variable  import Constant

from .mathext import math_constants, MathFunctionBase
from .mathext import MathIsfinite, MathIsinf, MathIsnan

__all__ = (
        'CmathAcos',
        'CmathAcosh',
        'CmathAsin',
        'CmathAsinh',
        'CmathAtan',
        'CmathAtanh',
        'CmathCos',
        'CmathCosh',
        'CmathExp',
        'CmathFunctionBool',
        'CmathFunctionComplex',
        'CmathIsclose',
        'CmathIsfinite',
        'CmathIsinf',
        'CmathIsnan',
        'CmathPhase',
        'CmathPolar',
        'CmathRect',
        'CmathSin',
        'CmathSinh',
        'CmathSqrt',
        'CmathTan',
        'CmathTanh',
        'cmath_mod',
    )

class CmathFunctionBool(MathFunctionBase):
    """
    Super-class from which functions returning a boolean inherit.

    A super-class from which functions in the `cmath` library which
    return a boolean should inherit.

    Parameters
    ----------
    *args : TypedAstNode
        The arguments passed to the function.
    """
    __slots__ = ()
    _class_type = PythonNativeBool()

class CmathFunctionComplex(MathFunctionBase):
    """
    Super-class from which functions returning a complex number inherit.

    A super-class from which functions in the `cmath` library which
    return a complex number should inherit.

    Parameters
    ----------
    z : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    _shape = None
    _class_type = PythonNativeComplex()

    def __init__(self, z : 'TypedAstNode'):
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    z : TypedAstNode
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
    a : TypedAstNode
        The first argument passed to the function.
    b : TypedAstNode
        The second argument passed to the function.
    rel_tol : TypedAstNode
        The relative tolerance.
    abs_tol : TypedAstNode
        The absolute tolerance.
    """
    __slots__ = ()
    name = 'isclose'
    def __init__(self, a, b, *, rel_tol=1e-09, abs_tol=0.0):
        super().__init__(a, b, rel_tol, abs_tol)

#==============================================================================

class CmathIsfinite(CmathFunctionBool):
    """
    Class representing a call to the `cmath.isfinite` function.

    A class which represents a call to the `isfinite` function from the `cmath` library.

    Parameters
    ----------
    z : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isfinite'
    def __new__(cls, z):
        if not isinstance(z.dtype.primitive_type, PrimitiveComplexType):
            return MathIsfinite(z)
        else:
            return PyccelAnd(MathIsfinite(PythonImag(z)), MathIsfinite(PythonReal(z)))

#==============================================================================

class CmathIsinf   (CmathFunctionBool):
    """
    Class representing a call to the `cmath.isinf` function.

    A class which represents a call to the `isinf` function from the `cmath` library.

    Parameters
    ----------
    z : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isinf'
    def __new__(cls, z):
        if not isinstance(z.dtype.primitive_type, PrimitiveComplexType):
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
    z : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'isnan'
    def __new__(cls, z):
        if not isinstance(z.dtype.primitive_type, PrimitiveComplexType):
            return MathIsnan(z)
        else:
            return PyccelOr(MathIsnan(PythonImag(z)), MathIsnan(PythonReal(z)))

#==============================================================================
# Dictionary to map math functions to classes above
#==============================================================================

class CmathPhase(PyccelFunction):
    """
    Class representing a call to the `cmath.phase` function.

    A class which represents a call to the `phase` function from the `cmath` library
    which calculates the phase angle of a complex number.

    Parameters
    ----------
    z : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'phase'
    def __init__(self, z):
        super().__init__(z)

class CmathPolar(PyccelFunction):
    """
    Class representing a call to the `cmath.polar` function.

    A class which represents a call to the `polar` function from the `cmath` library.

    Parameters
    ----------
    z : TypedAstNode
        The expression passed as argument to the function.
    """
    __slots__ = ()
    name = 'polar'
    _shape = (LiteralInteger(2),)
    _class_type = HomogeneousTupleType.get_new(PythonNativeFloat())

    def __init__(self, z):
        super().__init__(z)

class CmathRect(PyccelFunction):
    """
    Class representing a call to the `cmath.rect` function.

    A class which represents a call to the `rect` function from the `cmath` library.

    Parameters
    ----------
    r : TypedAstNode
        The first argument to the function, representing the radius.
    phi : TypedAstNode
        The second argument to the function, representing the polar angle.
    """
    __slots__ = ()
    name = 'rect'
    _shape = None
    _class_type = PythonNativeComplex()
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
    'infj': Constant(PythonNativeComplex(), 'infj', value=cmath.infj),
    'nanj': Constant(PythonNativeComplex(), 'nanj', value=cmath.nanj),
    }

cmath_mod = Module('cmath',
        variables = math_constants.values(),
        funcs     = cmath_functions)
