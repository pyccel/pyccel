#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" This module contains all literal types
"""
from pyccel.utilities.metaclasses import Singleton

from .basic              import TypedAstNode, PyccelAstNode
from .datatypes          import (NativeGeneric, NativeInteger, NativeBool, NativeFloat,
                                  NativeComplex, NativeString)

__all__ = (
    'convert_to_literal',
    'Literal',
    'LiteralComplex',
    'LiteralFalse',
    'LiteralFloat',
    'LiteralImaginaryUnit',
    'LiteralInteger',
    'LiteralNumeric',
    'LiteralString',
    'LiteralTrue',
    'Nil',
    'NilArgument',
)

#------------------------------------------------------------------------------
class Literal(TypedAstNode):
    """
    Class representing a literal value.

    Class representing a literal value. A literal is a value that is expressed
    as itself rather than as a variable or an expression, e.g. the number 3
    or the string "Hello".

    This class is abstract and should be implemented for each dtype.
    """
    __slots__ = ()
    _attribute_nodes  = ()
    _rank      = 0
    _shape     = None
    _order     = None

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the constant value of this literal as a Python object.
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
class LiteralNumeric(Literal):
    """
    Class representing a literal numeric type.

    Class representing a literal numeric type. A numeric type is
    a type representing a number (boolean/integer/float/complex).

    Parameters
    ----------
    precision : int
        The precision of the data type.
    """
    __slots__ = ('_precision',)

    def __init__(self, precision):
        if not isinstance(precision, int):
            raise TypeError("precision must be an integer")
        self._precision = precision
        super().__init__()

#------------------------------------------------------------------------------
class LiteralTrue(LiteralNumeric):
    """
    Class representing the Python value True.

    Class representing the Python value True.

    Parameters
    ----------
    precision : int, optional
        The precision of the boolean.
    """
    __slots__ = ()
    _dtype     = NativeBool()

    def __init__(self, precision = -1):
        super().__init__(precision)

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return True

#------------------------------------------------------------------------------
class LiteralFalse(LiteralNumeric):
    """
    Represents the Python value False.

    Class representing the Python value False.

    Parameters
    ----------
    precision : int, optional
        The precision of the boolean.
    """
    __slots__ = ()
    _dtype     = NativeBool()

    def __init__(self, precision = -1):
        super().__init__(precision)

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return False

#------------------------------------------------------------------------------
class LiteralInteger(LiteralNumeric):
    """
    Class representing an integer literal in Python.

    Class representing an integer literal, such as 3, in Python.

    Parameters
    ----------
    value : int
        The literal integer.

    precision : int, optional
        The precision of the integer. The default is Python built-in precision.
    """
    __slots__ = ('_value',)
    _dtype     = NativeInteger()

    def __init__(self, value, precision = -1):
        super().__init__(precision)
        assert(value >= 0)
        if not isinstance(value, int):
            raise TypeError("A LiteralInteger can only be created with an integer")
        self._value = value

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
class LiteralFloat(LiteralNumeric):
    """
    Class representing a float literal in Python.

    Class representing a float literal, such as 3.5, in Python.

    Parameters
    ----------
    value : float
        The literal float.

    precision : int, optional
        The precision of the float. The default is Python built-in precision.
    """
    __slots__ = ('_value',)
    _dtype     = NativeFloat()

    def __init__(self, value, *, precision = -1):
        if not isinstance(value, (int, float, LiteralFloat)):
            raise TypeError("A LiteralFloat can only be created with an integer or a float")
        super().__init__(precision)
        if isinstance(value, LiteralFloat):
            self._value = value.python_value
        else:
            self._value = float(value)

    @property
    def python_value(self):
        """
        Get the Python literal represented by this instance.

        Get the Python literal represented by this instance.
        """
        return self._value


#------------------------------------------------------------------------------
class LiteralComplex(LiteralNumeric):
    """
    Class representing a complex literal in Python.

    Class representing a complex literal, such as 3+2j, in Python.

    Parameters
    ----------
    real : float
        The real part of the literal complex.

    imag : float
        The imaginary part of the literal complex.

    precision : int, optional
        The precision of the complex. The default is Python built-in precision.
    """
    __slots__ = ('_real_part','_imag_part')
    _dtype     = NativeComplex()

    def __new__(cls, real, imag, precision = -1):
        if cls is LiteralImaginaryUnit:
            return super().__new__(cls)
        real_part = cls._collect_python_val(real)
        imag_part = cls._collect_python_val(imag)
        if real_part == 0 and imag_part == 1:
            return LiteralImaginaryUnit()
        else:
            return super().__new__(cls)

    def __init__(self, real, imag, precision = -1):
        super().__init__(precision)
        self._real_part = LiteralFloat(self._collect_python_val(real), precision = precision)
        self._imag_part = LiteralFloat(self._collect_python_val(imag), precision = precision)

    @staticmethod
    def _collect_python_val(arg):
        if isinstance(arg, Literal):
            return arg.python_value
        elif isinstance(arg, (int, float)):
            return arg
        else:
            raise TypeError("LiteralComplex argument must be an int/float/LiteralInt/LiteralFloat")

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
    precision : int, optional
        The precision of the complex. The default is Python built-in precision.
    """
    __slots__ = ()
    def __new__(cls, precision = -1):
        return super().__new__(cls, 0, 1, precision=precision)

    def __init__(self, real = 0, imag = 1, precision = -1):
        super().__init__(0, 1)

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
        The literal string.
    """
    __slots__ = ('_string',)
    _dtype     = NativeString()
    _precision = 0

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

class Nil(Literal, metaclass=Singleton):
    """
    Class representing a None object in the code.

    Class representing the Python value None in the code.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _dtype = NativeGeneric
    _precision = 0

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

def convert_to_literal(value, dtype = None, precision = None):
    """
    Convert a Python value to a Pyccel Literal.

    Convert a Python object to the equivalent Pyccel Literal
    object.

    Parameters
    ----------
    value : int/float/complex/bool/str
        The Python value.
    dtype : DataType
        The datatype of the Python value.
        Default : Matches type of 'value'.
    precision : int
        The precision of the value in the generated code.
        Default : Python precision (see default_precision).

    Returns
    -------
    Literal
        The Python value 'value' expressed as a literal
        with the specified dtype and precision.
    """
    from .operators import PyccelUnarySub # Imported here to avoid circular import

    if dtype is None:
        if isinstance(value, int):
            dtype = NativeInteger()
        elif isinstance(value, float):
            dtype = NativeFloat()
        elif isinstance(value, complex):
            dtype = NativeComplex()
        elif isinstance(value, bool):
            dtype = NativeBool()
        elif isinstance(value, str):
            dtype = NativeString()
        else:
            raise TypeError('Unknown type')

    if precision is None and dtype is not NativeString():
        precision = -1

    if isinstance(dtype, NativeInteger):
        if value >= 0:
            literal_val = LiteralInteger(value, precision)
        else:
            literal_val = PyccelUnarySub(LiteralInteger(-value, precision))
    elif isinstance(dtype, NativeFloat):
        literal_val = LiteralFloat(value, precision=precision)
    elif isinstance(dtype, NativeComplex):
        literal_val = LiteralComplex(value.real, value.imag, precision)
    elif isinstance(dtype, NativeBool):
        if value:
            literal_val = LiteralTrue(precision)
        else:
            literal_val = LiteralFalse(precision)
    elif isinstance(dtype, NativeString):
        literal_val = LiteralString(value)
    else:
        raise TypeError('Unknown type')

    return literal_val
