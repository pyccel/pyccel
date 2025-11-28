#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" This module contains all literal types
"""
import numpy as np

from .basic     import TypedAstNode, PyccelAstNode
from .datatypes import VoidType, PythonNativeInt, PythonNativeBool
from .datatypes import PythonNativeFloat, StringType, PythonNativeComplex
from .datatypes import PrimitiveIntegerType, PrimitiveFloatingPointType, PrimitiveBooleanType
from .datatypes import PrimitiveComplexType, FixedSizeNumericType

__all__ = (
    'Literal',
    'LiteralComplex',
    'LiteralEllipsis',
    'LiteralFalse',
    'LiteralFloat',
    'LiteralImaginaryUnit',
    'LiteralInteger',
    'LiteralString',
    'LiteralTrue',
    'Nil',
    'NilArgument',
    'convert_to_literal',
)

#------------------------------------------------------------------------------
class Literal(TypedAstNode):
    """
    Class representing a literal value.

    Class representing a literal value. A literal is a value that is expressed
    as itself rather than as a variable or an expression, e.g. the number 3
    or the string "Hello".

    This class is abstract and should be implemented for each dtype
    """
    __slots__ = ()
    _attribute_nodes  = ()
    _shape     = None

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """

    def __repr__(self):
        return f"Literal({repr(self.python_value)})"

    def __str__(self):
        return str(self.python_value)

    def __eq__(self, other):
        if isinstance(other, TypedAstNode):
            return isinstance(other, type(self)) and self.python_value == other.python_value
        else:
            return self.python_value == other

    def __hash__(self):
        return hash(self.python_value)

#------------------------------------------------------------------------------
class LiteralTrue(Literal):
    """
    Class representing the Python value True.

    Class representing the Python value True.

    Parameters
    ----------
    dtype : FixedSizeType
        The exact type of the literal.
    """
    __slots__ = ('_class_type',)

    def __init__(self, dtype = PythonNativeBool()):
        self._class_type = dtype
        super().__init__()

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return True

#------------------------------------------------------------------------------
class LiteralFalse(Literal):
    """
    Class representing the Python value False.

    Class representing the Python value False.

    Parameters
    ----------
    dtype : FixedSizeType
        The exact type of the literal.
    """
    __slots__ = ('_class_type',)

    def __init__(self, dtype = PythonNativeBool()):
        self._class_type = dtype
        super().__init__()

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return False

#------------------------------------------------------------------------------
class LiteralInteger(Literal):
    """
    Class representing an integer literal in Python.

    Class representing an integer literal, such as 3, in Python.

    Parameters
    ----------
    value : int
        The Python literal.

    dtype : FixedSizeType
        The exact type of the literal.
    """
    __slots__   = ('_value', '_class_type')

    def __init__(self, value, dtype = PythonNativeInt()):
        assert value >= 0
        if not isinstance(value, (int, np.integer)):
            raise TypeError("A LiteralInteger can only be created with an integer")
        self._value = int(value)
        self._class_type = dtype
        super().__init__()

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return self._value

    def __index__(self):
        return self.python_value

#------------------------------------------------------------------------------
class LiteralFloat(Literal):
    """
    Class representing a float literal in Python.

    Class representing a float literal, such as 3.5, in Python.

    Parameters
    ----------
    value : float
        The Python literal.

    dtype : FixedSizeType
        The exact type of the literal.
    """
    __slots__   = ('_value', '_class_type')

    def __init__(self, value, dtype = PythonNativeFloat()):
        if not isinstance(value, (int, float, LiteralFloat, np.integer, np.floating)):
            raise TypeError("A LiteralFloat can only be created with an integer or a float")
        if isinstance(value, LiteralFloat):
            self._value = value.python_value
        else:
            self._value = float(value)
        self._class_type = dtype
        super().__init__()

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return self._value


#------------------------------------------------------------------------------
class LiteralComplex(Literal):
    """
    Class representing a complex literal in Python.

    Class representing a complex literal, such as 3+2j, in Python.

    Parameters
    ----------
    real : float
        The real part of the Python literal.

    imag : float
        The imaginary part of the Python literal.

    dtype : FixedSizeType
        The exact type of the literal.
    """
    __slots__   = ('_real_part','_imag_part', '_class_type')

    def __new__(cls, real, imag, dtype = PythonNativeComplex()):
        if cls is LiteralImaginaryUnit:
            return super().__new__(cls)
        real_part = cls._collect_python_val(real)
        imag_part = cls._collect_python_val(imag)
        if real_part == 0 and imag_part == 1:
            return LiteralImaginaryUnit()
        else:
            return super().__new__(cls)

    def __init__(self, real, imag, dtype = PythonNativeComplex()):
        self._real_part = LiteralFloat(self._collect_python_val(real),
                                       dtype = dtype.element_type)
        self._imag_part = LiteralFloat(self._collect_python_val(imag),
                                       dtype = dtype.element_type)
        self._class_type = dtype
        super().__init__()

    @staticmethod
    def _collect_python_val(arg):
        """
        Extract the Python value from the input argument.

        Extract the Python value from the input argument which can either
        be a literal or a Python variable. The input argument represents
        either the real or the imaginary part of the complex literal.

        Parameters
        ----------
        arg : Literal | int | float
            The Python value.

        Returns
        -------
        float
            The Python value of the argument.
        """
        if isinstance(arg, Literal):
            return float(arg.python_value)
        elif isinstance(arg, (int, float, np.integer, np.floating)):
            return float(arg)
        else:
            raise TypeError(f"LiteralComplex argument must be an int/float/LiteralInt/LiteralFloat not a {type(arg)}")

    @property
    def real(self):
        """
        Return the real part of the complex literal.

        Return the real part of the complex literal.
        """
        return self._real_part

    @property
    def imag(self):
        """
        Return the imaginary part of the complex literal.

        Return the imaginary part of the complex literal.
        """
        return self._imag_part

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return self.real.python_value + self.imag.python_value*1j

#------------------------------------------------------------------------------
class LiteralImaginaryUnit(LiteralComplex):
    """
    Class representing the Python value j.

    Class representing the imaginary unit j in Python.

    Parameters
    ----------
    real : float = 0
        The value of the real part. This argument is necessary to handle the
        inheritance but should not be provided explicitly.
    imag : float = 0
        The value of the real part. This argument is necessary to handle the
        inheritance but should not be provided explicitly.
    dtype : FixedSizeType
        The exact type of the literal.
    """
    __slots__ = ()
    def __new__(cls, real = 0, imag = 1, dtype = PythonNativeComplex()):
        return super().__new__(cls, 0, 1, dtype = dtype)

    def __init__(self, real = 0, imag = 1, dtype = PythonNativeComplex()):
        super().__init__(0, 1, dtype)

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return 1j

#------------------------------------------------------------------------------
class LiteralString(Literal):
    """
    Class representing a string literal in Python.

    Class representing a string literal, such as 'hello' in Python.

    Parameters
    ----------
    arg : str
        The Python literal.
    """
    __slots__ = ('_string',)
    _class_type = StringType()
    _shape = (None,)

    def __init__(self, arg):
        super().__init__()
        if not isinstance(arg, str):
            raise TypeError('arg must be of type str')
        self._string = arg

    def __repr__(self):
        return f"'{self.python_value}'"

    def __str__(self):
        return str(self.python_value)

    def __add__(self, o):
        if isinstance(o, LiteralString):
            return LiteralString(self._string + o._string)
        return NotImplemented

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return self._string

#------------------------------------------------------------------------------

class Nil(Literal):
    """
    Class representing a None object in the code.

    Class representing the Python value None in the code.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _class_type = VoidType()

    def __str__(self):
        return 'None'

    def __bool__(self):
        return False

    def __eq__(self, other):
        return isinstance(other, Nil)

    def __hash__(self):
        return hash('Nil')+hash(None)

#------------------------------------------------------------------------------

class NilArgument(PyccelAstNode):
    """
    Represents None when passed as an argument to an inline function.

    Represents the Python value None when passed as an argument
    to an inline function. This class is necessary as to avoid
    accidental substitution due to Singletons.
    """
    __slots__ = ()
    _attribute_nodes = ()

    def __str__(self):
        return 'Argument(None)'

    def __bool__(self):
        return False

#------------------------------------------------------------------------------

class LiteralEllipsis(Literal):
    """
    Class representing an Ellipsis object in the code.

    Class representing the Python value Ellipsis in the code.
    """
    __slots__ = ()

    def __str__(self):
        return '...'

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return ...

#------------------------------------------------------------------------------

def convert_to_literal(value, dtype = None):
    """
    Convert a Python value to a pyccel Literal.

    Convert a Python value to a pyccel Literal.

    Parameters
    ----------
    value : int/float/complex/bool/str
        The Python value.
    dtype : DataType
        The datatype of the Python value.
        Default : Matches type of 'value'.

    Returns
    -------
    Literal
        The Python value 'value' expressed as a literal
        with the specified dtype.
    """
    from .operators import PyccelUnarySub # Imported here to avoid circular import

    # Calculate the default datatype
    if dtype is None:
        if isinstance(value, bool):
            dtype = PythonNativeBool()
        elif isinstance(value, int):
            dtype = PythonNativeInt()
        elif isinstance(value, float):
            dtype = PythonNativeFloat()
        elif isinstance(value, complex):
            dtype = PythonNativeComplex()
        elif isinstance(value, str):
            dtype = StringType()
        else:
            raise TypeError(f'Unknown type of object {value}')

    # Resolve any datatypes which don't inherit from FixedSizeType
    if isinstance(dtype, StringType):
        return LiteralString(value)

    assert isinstance(dtype, FixedSizeNumericType)

    primitive_type = dtype.primitive_type
    if isinstance(primitive_type, PrimitiveIntegerType):
        if value >= 0:
            literal_val = LiteralInteger(value, dtype)
        else:
            literal_val = PyccelUnarySub(LiteralInteger(-value, dtype))
    elif isinstance(primitive_type, PrimitiveFloatingPointType):
        literal_val = LiteralFloat(value, dtype)
    elif isinstance(primitive_type, PrimitiveComplexType):
        literal_val = LiteralComplex(value.real, value.imag, dtype)
    elif isinstance(primitive_type, PrimitiveBooleanType):
        if value:
            literal_val = LiteralTrue(dtype)
        else:
            literal_val = LiteralFalse(dtype)
    else:
        raise TypeError(f'Unknown type {dtype}')

    return literal_val
