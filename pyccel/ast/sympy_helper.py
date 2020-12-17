#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module containing functions which allow us to treat expressions expressed as pyccel nodes with sympy,
by providing translations between the sympy representation and the pyccel nodes
"""

import sympy as sp
from sympy.core.numbers import One, NegativeOne, Zero, Half

from .operators import PyccelAdd, PyccelMul, PyccelPow, PyccelUnarySub
from .operators import PyccelDiv, PyccelMinus, PyccelAssociativeParenthesis
from .core      import Variable, create_incremented_string, PyccelArraySize
from .core      import CodeBlock, Comment, For, Assign

from .builtins  import PythonRange

from .mathext   import MathCeil

from .literals  import LiteralInteger, LiteralFloat

from .datatypes import NativeInteger

#==============================================================================
def sympy_to_pyccel(expr, symbol_map):
    """
    Convert a sympy expression to a pyccel expression replacing sympy symbols with
    pyccel expressions provided in a symbol_map

      Parameters
      ----------
      expr       : PyccelAstNode
                   The pyccel node to be translated
      symbol_map : dict
                   Dictionary mapping sympy symbols to pyccel objects

      Returns
      ----------
      expr       : pyccel Object
    """

    #Constants
    if isinstance(expr, sp.Integer):
        return LiteralInteger(expr.p)
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
        minus_args = []
        plus_args = []

        # Find positive and negative elements
        for a in args[1:]:
            if isinstance(a, PyccelMul) and a.args[0] == LiteralInteger(-1):
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
        if arg.dtype is NativeInteger():
            return arg
        else:
            return MathCeil(arg)
    else:
        raise TypeError(str(type(expr)))

#==============================================================================
def pyccel_to_sympy(expr, symbol_map, used_names):
    """
    Convert a pyccel expression to a sympy expression saving any pyccel objects
    converted to sympy symbols in a dictionary to allow the reverse conversion
    to be carried out later

      Parameters
      ----------
      expr       : PyccelAstNode
                   The pyccel node to be translated
      symbol_map : dict
                   Dictionary containing any pyccel objects converted to sympy symbols
      used_names : Set
                   A set of all the names which already exist and therefore cannot
                   be used to create new symbols

      Returns
      ----------
      expr       : sympy Object
    """

    #Constants
    if isinstance(expr, LiteralInteger):
        return sp.Integer(expr.p)

    elif isinstance(expr, LiteralFloat):
        return sp.Float(expr)

    #Operators
    elif isinstance(expr, PyccelDiv):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] / args[1]

    elif isinstance(expr, PyccelMul):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return sp.Mul(*args)

    elif isinstance(expr, PyccelMinus):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return args[0] - args[1]

    elif isinstance(expr, PyccelUnarySub):
        arg = pyccel_to_sympy(expr.args[0], symbol_map, used_names)
        return -arg

    elif isinstance(expr, PyccelAdd):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return sp.Add(*args)

    elif isinstance(expr, PyccelPow):
        args = [pyccel_to_sympy(e, symbol_map, used_names) for e in expr.args]
        return sp.Pow(*args)

    elif isinstance(expr, PyccelAssociativeParenthesis):
        return pyccel_to_sympy(expr.args[0], symbol_map, used_names)

    elif isinstance(expr, MathCeil):
        return sp.ceiling(pyccel_to_sympy(expr.args[0], symbol_map, used_names))

    elif expr in symbol_map.values():
        return list(symbol_map.keys())[list(symbol_map.values()).index(expr)]

    elif isinstance(expr, Variable):
        sym = sp.Symbol(expr.name)
        symbol_map[sym] = expr
        return sym

    elif isinstance(expr, PyccelArraySize):
        sym_name,_ = create_incremented_string(used_names, prefix = 'tmp_size')
        sym = sp.Symbol(sym_name)
        symbol_map[sym] = expr
        return sym

    elif isinstance(expr, CodeBlock):
        body = (pyccel_to_sympy(b, symbol_map, used_names) for b in expr.body)
        return CodeBlock(body)

    elif isinstance(expr, (Comment)):
        return Comment('')

    elif isinstance(expr, For):
        target = pyccel_to_sympy(expr.target, symbol_map, used_names)
        iter_obj = pyccel_to_sympy(expr.iterable, symbol_map, used_names)
        body = pyccel_to_sympy(expr.body, symbol_map, used_names)
        return For(target, iter_obj, body)

    elif isinstance(expr, PythonRange):
        start = pyccel_to_sympy(expr.start, symbol_map, used_names)
        stop  = pyccel_to_sympy(expr.stop , symbol_map, used_names)
        step  = pyccel_to_sympy(expr.step , symbol_map, used_names)
        return sp.Range(start, stop, step)

    elif isinstance(expr, Assign):
        lhs = pyccel_to_sympy(expr.lhs, symbol_map, used_names)
        rhs = pyccel_to_sympy(expr.rhs, symbol_map, used_names)
        return Assign(lhs, rhs)

    elif isinstance(expr, (sp.core.basic.Atom, sp.core.operations.AssocOp, sp.Set)):
        # Already translated
        return expr

    else:
        raise TypeError(str(type(expr)))
