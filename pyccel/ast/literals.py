#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" This module contains all literal types
"""
from pyccel.utilities.metaclasses import Singleton

from .basic              import PyccelAstNode, Basic
from .datatypes          import (NativeGeneric, NativeInteger, NativeBool, NativeFloat,
                                  NativeComplex, NativeString)

__all__ = (
    'Literal',
    'LiteralTrue',
    'LiteralFalse',
    'LiteralInteger',
    'LiteralFloat',
    'LiteralComplex',
    'LiteralImaginaryUnit',
    'LiteralString',
    'Nil',
    'NilArgument',
    'get_default_literal_value'
)

#------------------------------------------------------------------------------
class Literal(PyccelAstNode):
    """
    Represents a Python literal.

    The abstract class from which the representations of the literal value of
    each dtype should derive.

    Parameters
    ----------
    precision : int
        The precision of the literal.
    """
    __slots__ = ('_precision',)
    _attribute_nodes  = ()
    _rank      = 0
    _shape     = None
    _order     = None

    def __init__(self, precision):
        if not isinstance(precision, int):
            raise TypeError("precision must be an integer")
        self._precision = precision
        super().__init__()

    @PyccelAstNode.precision.setter
    def precision(self, precision):
        """ Set precision for a literal class"""
        self._precision = precision

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
        if isinstance(other, PyccelAstNode):
            return isinstance(other, type(self)) and self.python_value == other.python_value
        else:
            return self.python_value == other

    def __hash__(self):
        return hash(self.python_value)

#------------------------------------------------------------------------------
class LiteralTrue(Literal):
    """
    Represents the Python value True.

    Represents a True object in the code.

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
        return True

#------------------------------------------------------------------------------
class LiteralFalse(Literal):
    """
    Represents the Python value False.

    Represents a False object in the code.

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
        return False

#------------------------------------------------------------------------------
class LiteralInteger(Literal):
    """
    Represents an integer literal in Python.

    Class, inheriting from Literal, representing a literal integer in the code.

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
        return self._value

    def __index__(self):
        return self.python_value

#------------------------------------------------------------------------------
class LiteralFloat(Literal):
    """
    Represents a float literal in Python.

    Class, inheriting from Literal, representing a literal float in the code.

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
        Literal.__init__(self, precision)
        if isinstance(value, LiteralFloat):
            self._value = value.python_value
        else:
            self._value = float(value)

    @property
    def python_value(self):
        return self._value


#------------------------------------------------------------------------------
class LiteralComplex(Literal):
    """
    Represents a complex literal in Python.

    Class, inheriting from Literal, representing a literal complex in the
    code.

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
        """ Return the real part of the complex literal """
        return self._real_part

    @property
    def imag(self):
        """ Return the imaginary part of the complex literal """
        return self._imag_part

    @property
    def python_value(self):
        return self.real.python_value + self.imag.python_value*1j

#------------------------------------------------------------------------------
class LiteralImaginaryUnit(LiteralComplex):
    """
    Represents the Python value j.

    Represents the imaginary unit.
    """
    __slots__ = ()
    def __new__(cls):
        return super().__new__(cls, 0, 1)

    def __init__(self):
        super().__init__(0, 1)

    @property
    def python_value(self):
        return 1j

#------------------------------------------------------------------------------
class LiteralString(Literal):
    """
    Represents a string literal in Python.

    Class, inheriting from Literal, which represents a literal string in the
    code.

    Parameters
    ----------
    arg : str
        The literal string.
    """
    __slots__ = ('_string',)
    _dtype     = NativeString()

    def __init__(self, arg):
        self._precision = 0
        super().__init__(self._precision)
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
        return self.arg

#------------------------------------------------------------------------------

class Nil(PyccelAstNode, metaclass=Singleton):
    """
    Class representing a None object in the code.

    A class representing a None object.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _dtype = NativeGeneric
    _precision = 0
    _rank = 0
    _shape = None
    _order = None

    def __str__(self):
        return 'None'

    def __bool__(self):
        return False

    def __eq__(self, other):
        return isinstance(other, Nil)

    def __hash__(self):
        return hash('Nil')+hash(None)

#------------------------------------------------------------------------------

class NilArgument(Basic):
    """
    Represents None as an argument to an inline function.

    Represents the Python value None when passed as an argument
    to an inline function. This class is necessary as to avoid
    accidental substitution due to Singletons
    """
    __slots__ = ()
    _attribute_nodes = ()

    def __str__(self):
        return 'Argument(None)'

    def __bool__(self):
        return False

#------------------------------------------------------------------------------

def get_default_literal_value(dtype):
    """Returns the default value of a native datatype."""
    if isinstance(dtype, NativeInteger):
        value = LiteralInteger(0)
    elif isinstance(dtype, NativeFloat):
        value = LiteralFloat(0.0)
    elif isinstance(dtype, NativeComplex):
        value = LiteralComplex(0.0, 0.0)
    elif isinstance(dtype, NativeBool):
        value = LiteralFalse()
    elif isinstance(dtype, NativeString):
        value = LiteralString('')
    else:
        raise TypeError('Unknown type')
    return value

#------------------------------------------------------------------------------

def convert_to_literal(value, dtype = None, precision = None):
    """
    Convert a Python value to a Pyccel Literal.

    Convert a Python object to the equivalent Pyccel Literal
    object.

    Parameters
    ----------
    value : int/float/complex/bool/str
                The Python value
    dtype : DataType
                The datatype of the Python value
                Default : Matches type of 'value'
    precision : int
                The precision of the value in the generated code
                Default : Python precision (see default_precision)

    Returns
    -------
    literal_val : Literal
                  The Python value 'value' expressed as a literal
                  with the specified dtype and precision
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
