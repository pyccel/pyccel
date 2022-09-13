#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module representing object address.
"""

from .basic import PyccelAstNode, Basic
from .literals  import LiteralString

class ObjectAddress(PyccelAstNode):
    """Represents the address of an object.
    ObjectAddress(Variable('int','a'))                            is  &a
    ObjectAddress(Variable('int','a', memory_handling='alias'))   is   a
    """

    __slots__ = ('_obj', '_rank', '_precision', '_dtype', '_shape', '_order')
    _attribute_nodes = ('_obj',)

    def __init__(self, obj):
        if not isinstance(obj, PyccelAstNode):
            raise TypeError("object must be an instance of PyccelAstNode")
        self._obj       = obj
        self._rank      = obj.rank
        self._shape     = obj.shape
        self._precision = obj.precision
        self._dtype     = obj.dtype
        self._order     = obj.order
        super().__init__()

    @property
    def obj(self):
        """The object whose address is of interest
        """
        return self._obj

#------------------------------------------------------------------------------
class CStringExpression(Basic):
    """
    Internal class used to hold a C string that has LiteralStrings and C macros
    Parameters:
        *args: str or LiteralString or CMacro or CStringExpression
            any number of arguments to be added to the expression
            note: they will get added in the order provided
    Example:
        CStringExpression(
            CMacro("m"),
            CStringExpression(
                LiteralString("the macro is: "),
                CMacro("mc")
            ),
            LiteralString("."),
        )
    """
    __slots__  = ('_expression',)
    _attribute_nodes  = ()

    def __init__(self, *args):
        super().__init__()
        self._expression = []
        for arg in args:
            self += arg

    def __repr__(self):
        return ''.join([repr(e) for e in self._expression])

    def __str__(self):
        return ''.join([str(e) for e in self._expression])

    def __add__(self, o):
        """
        append the argument o to the end of the list _expression
        Parameter:
            o: str or LiteralString or CMacro or CStringExpression
        """
        if isinstance(o, str):
            o = LiteralString(o)
        if not isinstance(o, (LiteralString, CMacro, CStringExpression)):
            raise TypeError(f"unsupported operand type(s) for +: '{self.__class__}' and '{type(o)}'")
        self.expression.append(o)
        return self

    def __radd__(self, o):
        if isinstance(o, LiteralString):
            return CStringExpression(o, self)
        return NotImplemented

    def intersperse(self, lst):
        """
        insert self between each element of the list `lst`
        Parameter:
            lst: list of (str or LiteralString or CMacro or CStringExpression)
            the list to insert self between its elements
        Example:
            a = [
                CMacro("m"),
                CStringExpression(LiteralString("the macro is: ")),
                LiteralString("."),
            ]

            b = CStringExpression("?").intersperse(a)

            is the same as:

            b = CStringExpression(
                CMacro("m"),
                CStringExpression("?"),
                CStringExpression(LiteralString("the macro is: ")),
                CStringExpression("?"),
                LiteralString("."),
            )
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
class CMacro(Basic):
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

    def __str__(self):
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
