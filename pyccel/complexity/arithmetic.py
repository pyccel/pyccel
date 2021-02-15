# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

# TODO: - Pow
#       - If
#       - FunctionalFor

"""
This module provides us with functions and objects that allow us to compute
the arithmetic complexity a of a program.

Example
-------

"""

from collections import OrderedDict

from sympy import count_ops as sympy_count_ops
from sympy import Tuple
from sympy import summation
from sympy import Symbol
from sympy import Function

from pyccel.ast.literals import Literal
from pyccel.ast.core     import For, Assign, AugAssign, CodeBlock, Comment, EmptyNode
from pyccel.ast.core     import Allocate, Deallocate
from pyccel.ast.core     import FunctionDef, FunctionCall
from pyccel.ast.core     import Return
from pyccel.ast.core     import AddOp, SubOp, MulOp, DivOp
from pyccel.ast.numpyext import NumpyUfuncBase
from pyccel.ast.numpyext import ( NumpySin, NumpyCos, NumpyTan, NumpyArcsin,
                                  NumpyArccos, NumpyArctan, NumpyArctan2, NumpySinh, NumpyCosh, NumpyTanh,
                                  NumpyArcsinh, NumpyArccosh, NumpyArctanh )
from pyccel.ast.numpyext import ( NumpyMax, NumpyMin, NumpyFloor, NumpyAbs, NumpyFabs, NumpyExp, NumpyLog,
                                  NumpySqrt )

from pyccel.ast.internals import PyccelArraySize, Slice
from pyccel.ast.operators import PyccelAdd, PyccelMinus, PyccelDiv, PyccelMul, PyccelFloorDiv
from pyccel.ast.variable  import IndexedElement, Variable
from pyccel.ast.numpyext  import NumpyZeros, NumpyOnes
from pyccel.ast.operators import PyccelOperator, PyccelAssociativeParenthesis
from pyccel.ast.builtins  import PythonAbs
from pyccel.ast.sympy_helper import pyccel_to_sympy
from pyccel.complexity.basic import Complexity

__all__ = ["OpComplexity"]

# ...
ADD = Symbol('ADD')
SUB = Symbol('SUB')
MUL = Symbol('MUL')
DIV = Symbol('DIV')
IDIV = Symbol('IDIV')
ABS = Symbol('ABS')

op_registry = {
    AddOp(): ADD,
    SubOp(): SUB,
    MulOp(): MUL,
    DivOp(): DIV,
#    ModOp: MOD,
    }

SHAPE = Function('shape')
# ...

# ==============================================================================
def _compute_size_lhs(expr):
    ntimes = 1

    if isinstance(expr.lhs, IndexedElement):
        indices = [(e,i) for e,i in enumerate(expr.lhs.indices) if isinstance(i, Slice)]
        for e,i in indices:
            # ...
            start = 0
            if not i.start == None:
                start = i.start.python_value
            # ...

            # ...
            stop = SHAPE(expr.lhs.base, e)
            if not i.stop == None:
                stop = i.stop.python_value
            # ...

            # ...
            step = 1
            if not i.step == None:
                step = i.step.python_value
            # ...

            if not(step == 1):
                raise NotImplementedError('only step == 1 is treated')

            # TODO uncomment this
            #      this was commented because we get floor(...)
            ntimes *= (stop - start) #// step

    return ntimes

# ==============================================================================
class OpComplexity(Complexity):
    """class for Operation complexity computation."""

    def cost(self, visual=True, mode=None):
        """
        Computes the complexity of the given code.

        verbose: bool
            talk more

        mode: string
            possible values are (None, simple)
        """
        # ...
        self._visual = visual
        self._mode = mode
        # ...

        # ...
        costs = OrderedDict()

        # ... first we treat declared functions
        if self.functions:
            for fname, d in self.functions.items():
                expr =  self._cost(d)

                costs[fname] = expr

        self._costs.update(costs)
        # ...

        # ... then we compute the complexity for the main program
        expr = self._cost(self.ast)
        # ...

        return expr

    def _cost(self, expr, **settings):
        if expr is None:
            return 0

        classes = type(expr).__mro__
        for cls in classes:
            method = '_cost_' + cls.__name__
            if hasattr(self, method):
                obj = getattr(self, method)(expr, **settings)
                return obj
            else:
                raise NotImplementedError('{} not available for {}'.format(method, type(expr)))

    def _cost_CodeBlock(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.body)

    def _cost_Comment(self, expr, **settings):
        return 0

    def _cost_EmptyNode(self, expr, **settings):
        return 0

    def _cost_Variable(self, expr, **settings):
        return 0

    def _cost_LiteralInteger(self, expr, **settings):
        return 0

    def _cost_LiteralFloat(self, expr, **settings):
        return 0

    def _cost_IndexedElement(self, expr, **settings):
        return 0

    def _cost_Tuple(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr)

    def _cost_list(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr)

    def _cost_tuple(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr)

    def _cost_PyccelArraySize(self, expr, **settings):
        return 0

    def _cost_NumpyZeros(self, expr, **settings):
        return 0

    def _cost_NumpyOnes(self, expr, **settings):
        return 0

    def _cost_Allocate(self, expr, **settings):
        return 0

    def _cost_Deallocate(self, expr, **settings):
        return 0

    def _cost_NumpyFloor(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('FLOOR')

    def _cost_NumpyExp(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('EXP')

    def _cost_NumpyLog(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('LOG')

    def _cost_NumpySqrt(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('SQRT')

    def _cost_NumpySin(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('SIN')

    def _cost_NumpyCos(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('COS')

    def _cost_NumpyTan(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('TAN')

    def _cost_NumpyArcsin(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('ARCSIN')

    def _cost_NumpyArccos(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('ARCCOS')

    def _cost_NumpyArctan(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('ARCTAN')

    def _cost_NumpyArctan2(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('ARCTAN2')

    def _cost_NumpySinh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('SINH')

    def _cost_NumpyCosh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('COSH')

    def _cost_NumpyTanh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('TANH')

    def _cost_NumpyArcsinh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('ARCSINH')

    def _cost_NumpyArccosh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('ARCCOSH')

    def _cost_NumpyArctanh(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        return ops + Symbol('ARCTANH')

    def _cost_PyccelAssociativeParenthesis(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr._args)

    def _cost_Return(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in [expr.stmt, expr.expr])

    def _cost_PyccelAdd(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + ADD

    def _cost_PyccelMinus(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + SUB

    def _cost_PyccelDiv(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + DIV

    def _cost_PyccelFloorDiv(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + IDIV

    def _cost_PyccelMul(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 1
        else:
            return ops + MUL

    def _cost_PythonAbs(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.args)
        if self.mode:
            return ops + 0
        else:
            return ops + ABS

    def _cost_PyccelPow(self, expr, **settings):
        # TODO
        return 0

    def _cost_PyccelOperator(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_FunctionDef(self, expr, **settings):
        return self._cost(expr.body, **settings)

    def _cost_FunctionCall(self, expr, **settings):
        if self.costs is None:
            raise ValueError('costs dict is None')

        fname = expr.func_name

        if not fname in self.costs.keys():
            raise ValueError('Cannot find the cost of the function {}'.format(fname))

        return self.costs[fname]

    def _cost_NumpyUfuncBase(self, expr, **settings):
        try:
            f = numpy_functions_registery[type(expr)]
        except:
            raise NotImplementedError('{}'.format(type(expr)))

        ops = sum(self._cost(i, **settings) for i in expr.args)

        return Symbol(f.upper()) + ops

    def _cost_For(self, expr, **settings):
        ops = sum(self._cost(i, **settings) for i in expr.body.body)

        # ...
        i = expr.target
        i = pyccel_to_sympy(i, self._symbol_map, self._used_names)

        b = expr.iterable.start
        b = pyccel_to_sympy(b, self._symbol_map, self._used_names)

        e = expr.iterable.stop
        e = pyccel_to_sympy(e, self._symbol_map, self._used_names)
        # ...

        # TODO treat the case step /= 1
        return summation(ops, (i, b, e-1))

    def _cost_Assign(self, expr, **settings):
        if isinstance(expr.rhs, (NumpyZeros, NumpyOnes, Comment, EmptyNode)):
            return 0

        ntimes = _compute_size_lhs(expr)

        return ntimes * ( self._cost(expr.rhs, **settings) )

    def _cost_AugAssign(self, expr, **settings):
        if isinstance(expr.rhs, (NumpyZeros, NumpyOnes, Comment, EmptyNode)):
            return 0

        # ...
        if self.mode:
            op = 1
        else:
            op = op_registry[expr.op]
        # ...

        ntimes = _compute_size_lhs(expr)

        return ntimes * ( op + self._cost(expr.rhs, **settings) )
