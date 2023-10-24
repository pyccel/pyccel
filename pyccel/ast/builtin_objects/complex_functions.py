# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This module implements built-in functions which operate on complex numbers.
"""

from ..builtins   import PythonInt, PythonComplex

from ..class_defs import IntegerClass, FloatClass, ComplexClass

from ..core       import PyccelFunctionDef

from ..datatypes  import NativeBool, NativeFloat, NativeComplex

from ..internals  import PyccelInternalFunction

from ..literals   import convert_to_literal

__all__ = (
    'PythonComplexProperty',
    'PythonReal',
    'PythonImag',
    'PythonConjugate',
    )

#==============================================================================
class PythonComplexProperty(PyccelInternalFunction):
    """Represents a call to the .real or .imag property

    e.g:
    > a = 1+2j
    > a.real
    1.0

    arg : Variable, Literal
    """
    __slots__ = ()
    _dtype = NativeFloat()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def internal_var(self):
        """Return the variable on which the function was called"""
        return self._args[0]

#==============================================================================
class PythonReal(PythonComplexProperty):
    """Represents a call to the .real property

    e.g:
    > a = 1+2j
    > a.real
    1.0

    arg : Variable, Literal
    """
    __slots__ = ()
    name = 'real'
    def __new__(cls, arg):
        if isinstance(arg.dtype, NativeBool):
            return PythonInt(arg)
        elif not isinstance(arg.dtype, NativeComplex):
            return arg
        else:
            return super().__new__(cls)

    def __str__(self):
        return f'Real({str(self.internal_var)})'

#==============================================================================
class PythonImag(PythonComplexProperty):
    """
    Represents a call to the .imag property.

    Represents a call to the .imag property of an object with a complex type.
    e.g:
    >>> a = 1+2j
    >>> a.imag
    1.0

    Parameters
    ----------
    arg : Variable, Literal
        The object on which the property is called.
    """
    __slots__ = ()
    name = 'imag'
    def __new__(cls, arg):
        if arg.dtype is not NativeComplex():
            return convert_to_literal(0, dtype = arg.dtype)
        else:
            return super().__new__(cls)

    def __str__(self):
        return f'Imag({str(self.internal_var)})'

#==============================================================================
class PythonConjugate(PyccelInternalFunction):
    """
    Represents a call to the .conjugate() function.

    Represents a call to the conjugate function which is a member of
    the builtin types int, float, complex. The conjugate function is
    called from Python as follows:

    > a = 1+2j
    > a.conjugate()
    1-2j

    Parameters
    ----------
    arg : TypedAstNode
        The variable/expression which was passed to the
        conjugate function.
    """
    __slots__ = ()
    _dtype = NativeComplex()
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    name = 'conjugate'

    def __new__(cls, arg):
        if arg.dtype is NativeBool():
            return PythonInt(arg)
        elif arg.dtype is not NativeComplex():
            return arg
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        super().__init__(arg)

    @property
    def internal_var(self):
        """Return the variable on which the function was called"""
        return self._args[0]

    def __str__(self):
        return f'Conjugate({str(self.internal_var)})'

#==============================================================================

PythonComplex._real_cast = PythonReal #pylint: disable = protected-access
PythonComplex._imag_cast = PythonImag #pylint: disable = protected-access

#==============================================================================
#                        Update class definitions
#==============================================================================
ComplexClass.add_new_method(PyccelFunctionDef('imag', func_class = PythonImag,
                decorators={'property':'property', 'numpy_wrapper':'numpy_wrapper'}))
ComplexClass.add_new_method(PyccelFunctionDef('real', func_class = PythonReal,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}))
ComplexClass.add_new_method(PyccelFunctionDef('conjugate', func_class = PythonConjugate,
                decorators={'numpy_wrapper': 'numpy_wrapper'}))

FloatClass.add_new_method(PyccelFunctionDef('imag', func_class = PythonImag,
                decorators={'property':'property', 'numpy_wrapper':'numpy_wrapper'}))
FloatClass.add_new_method(PyccelFunctionDef('real', func_class = PythonReal,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}))
FloatClass.add_new_method(PyccelFunctionDef('conjugate', func_class = PythonConjugate,
                decorators={'numpy_wrapper': 'numpy_wrapper'}))

IntegerClass.add_new_method(PyccelFunctionDef('imag', func_class = PythonImag,
                decorators={'property':'property', 'numpy_wrapper':'numpy_wrapper'}))
IntegerClass.add_new_method(PyccelFunctionDef('real', func_class = PythonReal,
                decorators={'property':'property', 'numpy_wrapper': 'numpy_wrapper'}))
IntegerClass.add_new_method(PyccelFunctionDef('conjugate', func_class = PythonConjugate,
                decorators={'numpy_wrapper': 'numpy_wrapper'}))
