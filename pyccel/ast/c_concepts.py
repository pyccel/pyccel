#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Module representing concepts that are only applicable to C code (e.g. ObjectAddress).
"""
from functools import cache

from pyccel.utilities.metaclasses import Singleton
from .basic     import TypedAstNode, PyccelAstNode
from .datatypes import HomogeneousContainerType, FixedSizeType, FixedSizeNumericType, PrimitiveIntegerType
from .datatypes import CharType
from .internals import PyccelFunction
from .literals  import LiteralString

__all__ = ('CMacro',
           'CNativeInt',
           'CStackArray',
           'CStrStr',
           'CStringExpression',
           'ObjectAddress',
           'PointerCast')

#------------------------------------------------------------------------------

class CNativeInt(FixedSizeNumericType):
    """
    Class representing C's native integer type.

    Class representing C's native integer type.
    """
    __slots__ = ()
    _name = 'int'
    _primitive_type = PrimitiveIntegerType()
    _precision = None

#------------------------------------------------------------------------------

class CStackArray(HomogeneousContainerType, metaclass=Singleton):
    """
    A data type representing an array allocated on the stack.

    A data type representing an array allocated on the stack.
    E.g. `float a[4];`

    Parameters
    ----------
    element_type : FixedSizeType
        The type of the elements inside the array.
    """
    __slots__ = ('_element_type',)
    _name = 'c_stackarray'
    _container_rank = 1
    _order = None

    @classmethod
    @cache
    def get_new(cls, element_type):
        def __init__(self):
            self._element_type = element_type
            HomogeneousContainerType.__init__(self)
        return type(f'CStackArray{type(element_type).__name__}', (CStackArray,),
                    {'__init__' : __init__})()

#------------------------------------------------------------------------------
class ObjectAddress(TypedAstNode):
    """
    Class representing the address of an object.

    Class representing the address of an object. In most situations it will not be
    necessary to use this object explicitly. E.g. if you assign a pointer to a
    target then the pointer will be printed using `AliasAssign`. However for the
    `_print_AliasAssign` function to print neatly, this class will be used.

    Parameters
    ----------
    obj : TypedAstNode
        The object whose address should be printed.

    Examples
    --------
    >>> CCodePrinter._print(ObjectAddress(Variable(PythonNativeInt(),'a')))
    '&a'
    >>> CCodePrinter._print(ObjectAddress(Variable(PythonNativeInt(),'a', memory_handling='alias')))
    'a'
    """

    __slots__ = ('_obj', '_shape', '_class_type')
    _attribute_nodes = ('_obj',)

    def __init__(self, obj):
        if not isinstance(obj, TypedAstNode):
            raise TypeError("object must be an instance of TypedAstNode")
        self._obj        = obj
        self._shape      = obj.shape
        self._class_type = obj.class_type
        super().__init__()

    @property
    def obj(self):
        """The object whose address is of interest
        """
        return self._obj

    @property
    def is_alias(self):
        """
        Indicate that an ObjectAddress uses alias memory handling.

        Indicate that an ObjectAddress uses alias memory handling.
        """
        return True

#------------------------------------------------------------------------------
class PointerCast(TypedAstNode):
    """
    A class which represents the casting of one pointer to another.

    A class which represents the casting of one pointer to another in C code.
    This is useful for storing addresses in a void pointer.
    Using this class is not strictly necessary to produce correct C code,
    but avoids compiler warnings about the implicit conversion of pointers.

    Parameters
    ----------
    obj : Variable
        The pointer being cast.
    cast_type : TypedAstNode
        A TypedAstNode describing the object resulting from the cast.
    """
    __slots__ = ('_obj', '_shape', '_class_type', '_cast_type')
    _attribute_nodes = ('_obj',)

    def __init__(self, obj, cast_type):
        if not isinstance(obj, TypedAstNode):
            raise TypeError("object must be an instance of TypedAstNode")
        assert getattr(obj, 'is_alias', False)
        self._obj        = obj
        self._shape      = cast_type.shape
        self._class_type = cast_type.class_type
        self._cast_type  = cast_type
        super().__init__()

    @property
    def obj(self):
        """
        The object whose address is of interest.

        The object whose address is of interest.
        """
        return self._obj

    @property
    def cast_type(self):
        """
        Get the TypedAstNode which describes the object resulting from the cast.

        Get the TypedAstNode which describes the object resulting from the cast.
        """
        return self._cast_type

    @property
    def is_argument(self):
        """
        Indicates whether the variable is an argument.

        Indicates whether the variable is an argument.
        """
        return self._obj.is_argument

#------------------------------------------------------------------------------
class CStringExpression(PyccelAstNode):
    """
    Internal class used to hold a C string that has LiteralStrings and C macros.

    Parameters
    ----------
    *args : str / LiteralString / CMacro / CStringExpression
            any number of arguments to be added to the expression
            note: they will get added in the order provided

    Example
    ------
    >>> expr = CStringExpression(
    ...     CMacro("m"),
    ...     CStringExpression(
    ...         LiteralString("the macro is: "),
    ...         CMacro("mc")
    ...     ),
    ...     LiteralString("."),
    ... )
    """
    __slots__  = ('_expression',)
    _attribute_nodes  = ('_expression',)

    def __init__(self, *args):
        self._expression = []
        super().__init__()
        for arg in args:
            self.append(arg)

    def __repr__(self):
        return ''.join(repr(e) for e in self._expression)

    def __str__(self):
        return ''.join(str(e) for e in self._expression)

    def __add__(self, o):
        """
        return new CStringExpression that has `o` at the end

        Parameter
        ----------
        o : str / LiteralString / CMacro / CStringExpression
            the expression to add
        """
        if isinstance(o, str):
            o = LiteralString(o)
        if not isinstance(o, (LiteralString, CMacro, CStringExpression)):
            raise TypeError(f"unsupported operand type(s) for +: '{self.__class__}' and '{type(o)}'")
        return CStringExpression(*self._expression, o)

    def __radd__(self, o):
        if isinstance(o, LiteralString):
            return CStringExpression(o, self)
        return NotImplemented

    def __iadd__(self, o):
        self.append(o)
        return self

    def append(self, o):
        """
        append the argument `o` to the end of the list _expression

        Parameter
        ---------
        o : str / LiteralString / CMacro / CStringExpression
            the expression to append
        """
        if isinstance(o, str):
            o = LiteralString(o)
        if not isinstance(o, (LiteralString, CMacro, CStringExpression)):
            raise TypeError(f"unsupported operand type(s) for append: '{self.__class__}' and '{type(o)}'")
        self._expression += (o,)
        o.set_current_user_node(self)

    def join(self, lst):
        """
        insert self between each element of the list `lst`

        Parameter
        ---------
        lst : list
            the list to insert self between its elements

        Example
        -------
        >>> a = [
        ...     CMacro("m"),
        ...     CStringExpression(LiteralString("the macro is: ")),
        ...     LiteralString("."),
        ... ]
        >>> b = CStringExpression("?").join(a)
        ...
        ... # is the same as:
        ...
        >>> b = CStringExpression(
        ...     CMacro("m"),
        ...     CStringExpression("?"),
        ...     CStringExpression(LiteralString("the macro is: ")),
                CStringExpression("?"),
        ...     LiteralString("."),
        ... )
        """
        result = CStringExpression()
        if not lst:
            return result
        result += lst[0]
        for elm in lst[1:]:
            result += self
            result += elm
        return result

    def get_flat_expression_list(self):
        """
        returns a list of LiteralStrings and CMacros after merging every
        consecutive LiteralString
        """
        tmp_res = []
        for e in self.expression:
            if isinstance(e, CStringExpression):
                tmp_res.extend(e.get_flat_expression_list())
            else:
                tmp_res.append(e)
        if not tmp_res:
            return []
        result = [tmp_res[0]]
        for e in tmp_res[1:]:
            if isinstance(e, LiteralString) and isinstance(result[-1], LiteralString):
                result[-1] += e
            else:
                result.append(e)
        return result

    @property
    def expression(self):
        """ The list containing the literal strings and c macros
        """
        return self._expression

#------------------------------------------------------------------------------
class CMacro(PyccelAstNode):
    """Represents a c macro"""
    __slots__ = ('_macro',)
    _attribute_nodes  = ()

    def __init__(self, arg):
        super().__init__()
        if not isinstance(arg, str):
            raise TypeError('arg must be of type str')
        self._macro = arg

    def __repr__(self):
        return str(self._macro)

    def __add__(self, o):
        if isinstance(o, (LiteralString, CStringExpression)):
            return CStringExpression(self, o)
        return NotImplemented

    def __radd__(self, o):
        if isinstance(o, LiteralString):
            return CStringExpression(o, self)
        return NotImplemented

    @property
    def macro(self):
        """ The string containing macro name
        """
        return self._macro

#-------------------------------------------------------------------
#                         String functions
#-------------------------------------------------------------------
class CStrStr(PyccelFunction):
    """
    A class which extracts a const char* from a literal string.

    A class which extracts a const char* from a literal string. This
    is useful for calling C functions which were not designed for
    STC.

    Parameters
    ----------
    arg : TypedAstNode | CMacro
        The object which should be passed as a const char*.
    """
    __slots__ = ()
    _class_type = CharType()
    _shape = (None,)

    def __new__(cls, arg):
        if isinstance(arg, CMacro):
            return arg
        else:
            return super().__new__(cls)

    def __init__(self, arg):
        super().__init__(arg)
