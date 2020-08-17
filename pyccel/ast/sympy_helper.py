import sympy as sp
from sympy.core.numbers import One, NegativeOne, Zero, Half

from .core      import PyccelAdd, PyccelMul, PyccelPow
from .core      import PyccelDiv, PyccelMinus, PyccelAssociativeParenthesis

from .mathext   import MathCeil

from .numbers   import Integer as PyccelInteger, Float as PyccelFloat

from .datatypes import NativeInteger

#==============================================================================
def sympy_to_pyccel(expr, symbol_map):
    """
    Convert a sympy expression to a pyccel expression replacing sympy symbols with
    pyccel expressions provided in a symbol_map
    """

    #Constants
    if isinstance(expr, sp.Integer):
        return PyccelInteger(expr)
    elif isinstance(expr, One):
        return PyccelInteger(1)
    elif isinstance(expr, NegativeOne):
        return PyccelInteger(-1)
    elif isinstance(expr, Zero):
        return PyccelInteger(0)
    elif isinstance(expr, sp.Float):
        return PyccelFloat(expr)
    elif isinstance(expr, Half):
        return PyccelFloat(0.5)
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
        minus_args = []
        plus_args = []

        # Find positive and negative elements
        for a in args:
            if isinstance(a, PyccelMul) and a.args[0] == PyccelInteger(-1):
                minus_args.append(a.args[1])
            else:
                plus_args.append(a)

        #Use pyccel Add or Minus as appropriate
        if len(minus_args) == 0:
            return PyccelAdd(*plus_args)
        elif len(plus_args) == 0:
            return PyccelMul(PyccelInteger(-1), PyccelAssociativeParenthesis(PyccelAdd(*minus_args)))
        else:
            if len(plus_args)>1:
                plus_args = [PyccelAdd(*plus_args)]
            if len(minus_args)>1:
                minus_args = [PyccelAssociativeParenthesis(PyccelAdd(*minus_args))]
            return PyccelMinus(*plus_args,*minus_args)
    elif isinstance(expr, sp.Pow):
        # Recognise division
        if isinstance(expr.args[1], NegativeOne):
            return PyccelDiv(PyccelInteger(1), sympy_to_pyccel(expr.args[0], symbol_map))
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
