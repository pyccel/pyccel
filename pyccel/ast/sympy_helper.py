#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Module containing functions which allow us to treat expressions expressed as Pyccel nodes with SymPy,
by providing translations between the SymPy representation and the Pyccel nodes
"""

import sympy as sp
from sympy.core.numbers import One, NegativeOne, Zero, Half

from pyccel.utilities.strings import create_incremented_string

from .builtins  import PythonRange, PythonTuple, PythonMin, PythonMax

from .datatypes import PrimitiveIntegerType
from .internals import PyccelArrayShapeElement
from .literals  import LiteralInteger, LiteralFloat, LiteralComplex
from .literals  import LiteralTrue, LiteralFalse
from .mathext   import MathCeil, MathFabs
from .numpyext  import NumpyFloor
from .operators import PyccelAdd, PyccelMul, PyccelPow, PyccelUnarySub
from .operators import PyccelDiv, PyccelMinus, PyccelAssociativeParenthesis
from .operators import PyccelFloorDiv
from .operators import PyccelEq, PyccelNe, PyccelLt, PyccelLe, PyccelGt, PyccelGe
from .operators import PyccelAnd, PyccelOr, PyccelNot
from .variable  import Variable

__all__ = ('pyccel_to_sympy',
           'sympy_to_pyccel')

#==============================================================================
def sympy_to_pyccel(expr, symbol_map):
    """
    Convert a SymPy expression to a Pyccel expression.

    Convert a SymPy expression to a Pyccel expression replacing SymPy symbols with
    Pyccel expressions provided in a symbol_map

    Parameters
    ----------
    expr : SymPy object
        The SymPy expression to be translated.

    symbol_map : dict
        Dictionary mapping SymPy symbols to Pyccel objects.

    Returns
    -------
    TypedAstNode
        The Pyccel equivalent of the SymPy object `expr`.
    """

    #Constants
    if isinstance(expr, sp.Integer):
        if expr.p >= 0:
            return LiteralInteger(expr.p)
        else:
            return PyccelUnarySub(LiteralInteger(-expr.p))
    elif isinstance(expr, One):
        return LiteralInteger(1)
    elif isinstance(expr, NegativeOne):
        return LiteralInteger(-1)
    elif isinstance(expr, Zero):
        return LiteralInteger(0)
    elif isinstance(expr, sp.Float):
        return LiteralFloat(float(expr))
    elif isinstance(expr, Half):
        return LiteralFloat(0.5)
    elif isinstance(expr, sp.Rational):
        return LiteralFloat(float(expr))
    elif isinstance(expr, sp.Symbol) and expr in symbol_map:
        return symbol_map[expr]

    #Operators
    elif isinstance(expr, sp.Mul):
        args = [sympy_to_pyccel(e, symbol_map) for e in expr.args]

        # Handle priority
        for i,a in enumerate(args):
            if isinstance(a, (PyccelAdd, PyccelMinus)):
                args[i] = PyccelAssociativeParenthesis(a)

        return PyccelMul(*args)
    elif isinstance(expr, sp.Add):
        args = [sympy_to_pyccel(e, symbol_map) for e in expr.args]
        result = args[0]

        # Find positive and negative elements
        for a in args[1:]:
            if isinstance(a, PyccelUnarySub):
                result = PyccelMinus(result, a.args[1])
            else:
                result = PyccelAdd(result, a)
        return result
    elif isinstance(expr, sp.Pow):
        # Recognise division
        if isinstance(expr.args[1], NegativeOne):
            return PyccelDiv(LiteralInteger(1), sympy_to_pyccel(expr.args[0], symbol_map))
        else:
            return PyccelPow(*[sympy_to_pyccel(e, symbol_map) for e in expr.args])
    elif isinstance(expr, sp.ceiling):
        arg = sympy_to_pyccel(expr.args[0], symbol_map)
        # Only apply ceiling where appropriate
        if getattr(arg.dtype, 'primitive_type', None) is PrimitiveIntegerType():
            return arg
        else:
            return MathCeil(arg)

    elif isinstance(expr, sp.Abs):
        arg = sympy_to_pyccel(expr.args[0], symbol_map)
        # Only apply ceiling where appropriate
        return MathFabs(arg)

    elif isinstance(expr, sp.Min):
        args = [sympy_to_pyccel(a, symbol_map) for a in expr.args]
        return PythonMin(*args)

    elif isinstance(expr, sp.Max):
        args = [sympy_to_pyccel(a, symbol_map) for a in expr.args]
        return PythonMax(*args)

    elif isinstance(expr, sp.Tuple):
        args = [sympy_to_pyccel(a, symbol_map) for a in expr]
        return PythonTuple(*args)

    elif isinstance(expr, sp.floor):
        arg = sympy_to_pyccel(expr.args[0], symbol_map)
        return NumpyFloor(arg)

    else:
        raise TypeError(str(type(expr)))

#==============================================================================
def pyccel_to_sympy(expr, symbol_map, used_names):
    """
    Convert a Pyccel expression to a SymPy expression.

    Convert a Pyccel expression to a SymPy expression saving any Pyccel objects
    converted to SymPy symbols in a dictionary to allow the reverse conversion
    to be carried out later.

    Parameters
    ----------
    expr : TypedAstNode
        The Pyccel node to be translated.

    symbol_map : dict
        Dictionary containing any Pyccel objects converted to SymPy symbols.

    used_names : Set
        A set of all the names which already exist and therefore cannot
        be used to create new symbols.

    Returns
    -------
    SymPy Object
        The SymPy equivalent of the `expr` argument.
    """

    #Constants
    if isinstance(expr, LiteralInteger):
        return sp.Integer(expr.python_value)

    elif isinstance(expr, LiteralFloat):
        return sp.Float(expr.python_value)

    elif isinstance(expr, LiteralComplex):
        return sp.Float(expr.real) + sp.Float(expr.imag) * 1j

    #Operators
    elif isinstance(expr, PyccelDiv):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] / args[1]

    elif isinstance(expr, PyccelFloorDiv):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] // args[1]

    elif isinstance(expr, PyccelMul):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] * args[1]

    elif isinstance(expr, PyccelMinus):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] - args[1]

    elif isinstance(expr, PyccelUnarySub):
        arg = pyccel_to_sympy(expr.args[0], symbol_map, used_names)
        return -arg

    elif isinstance(expr, PyccelAdd):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] + args[1]

    elif isinstance(expr, PyccelEq):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return sp.Eq(args[0], args[1])

    elif isinstance(expr, PyccelNe):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return sp.Ne(args[0], args[1])

    elif isinstance(expr, PyccelLe):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] <= args[1]

    elif isinstance(expr, PyccelLt):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] < args[1]

    elif isinstance(expr, PyccelGe):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] >= args[1]

    elif isinstance(expr, PyccelGt):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] > args[1]

    elif isinstance(expr, PyccelAnd):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return sp.And(*args)

    elif isinstance(expr, PyccelOr):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return sp.Or(*args)

    elif isinstance(expr, PyccelNot):
        arg = pyccel_to_sympy(expr.args[0], symbol_map, used_names)
        return sp.Not(arg)

    elif isinstance(expr, PyccelPow):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] ** args[1]

    elif isinstance(expr, PyccelAssociativeParenthesis):
        return pyccel_to_sympy(expr.args[0], symbol_map, used_names)

    elif isinstance(expr, MathCeil):
        return sp.ceiling(pyccel_to_sympy(expr.args[0], symbol_map, used_names))

    elif isinstance(expr, MathFabs):
        return sp.Abs(pyccel_to_sympy(expr.args[0], symbol_map, used_names))

    elif isinstance(expr, PythonMin):
        args = [pyccel_to_sympy(ee, symbol_map, used_names) for e in expr.args for ee in e]
        return sp.Min(*args)

    elif isinstance(expr, PythonMax):
        args = [pyccel_to_sympy(ee, symbol_map, used_names) for e in expr.args for ee in e]
        return sp.Max(*args)

    elif expr in symbol_map.values():
        return list(symbol_map.keys())[list(symbol_map.values()).index(expr)]

    elif isinstance(expr, Variable):
        sym = sp.Symbol(expr.name)
        symbol_map[sym] = expr
        return sym

    elif isinstance(expr, PyccelArrayShapeElement):
        sym_name,_ = create_incremented_string(used_names, prefix = 'tmp_size')
        used_names.add(sym_name)
        sym = sp.Symbol(sym_name)
        symbol_map[sym] = expr
        return sym

    elif isinstance(expr, PythonRange):
        start = pyccel_to_sympy(expr.start, symbol_map, used_names)
        stop  = pyccel_to_sympy(expr.stop , symbol_map, used_names)
        step  = pyccel_to_sympy(expr.step , symbol_map, used_names)
        return sp.Range(start, stop, step)

    elif isinstance(expr, PythonTuple):
        args = [pyccel_to_sympy(a, symbol_map, used_names) for a in expr]
        return sp.Tuple(*args)

    elif isinstance(expr, LiteralTrue):
        return sp.logic.boolalg.BooleanTrue()

    elif isinstance(expr, LiteralFalse):
        return sp.logic.boolalg.BooleanFalse()

    elif isinstance(expr, (sp.core.basic.Atom, sp.core.operations.AssocOp, sp.Set)):
        # Already translated
        return expr

    else:
        raise TypeError(str(type(expr)))
