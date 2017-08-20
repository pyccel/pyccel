"""
Types used to *describe* the underlying computation to be generated. The main
entry point to be used by most users is `routine`. For more control, the
classes may be used directly. See the docstrings for more details.

"""

from __future__ import print_function, division

from sympy.core import Symbol, Tuple, Expr, Basic, Integer, Dict
from sympy.utilities.iterables import iterable
from sympy.core.sympify import _sympify
from sympy import simplify
from sympy.core.assumptions import _assume_defined
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement

from pyccel.types.ast import (Assign, Argument, DataType, datatype,
        InOutArgument, OutArgument, InArgument, Bool, Int, Float, Double)
from pyccel.utilities.util import do_once, iterate


class RoutineResult(Basic):
    """Base class for all outgoing information from a routine."""
    pass


class RoutineReturn(RoutineResult):
    """Represents a result provided via a `Return`.

    Parameters
    ----------
    dtype : DataType
        Datatype of returned expression.
    expr : sympifyable
        Expression to be returned.

    Attributes
    ----------
    dtype : DataType
        Datatype of returned expression.
    expr : sympifyable
        Expression to be returned.

    """

    def __new__(cls, dtype, expr):
        if isinstance(dtype, str):
            dtype = datatype(dtype)
        elif not isinstance(dtype, DataType):
            raise TypeError("datatype must be an instance of DataType.")
        expr = _sympify(expr)
        if not isinstance(expr, (Expr, MatrixExpr)):
            raise TypeError("Unsupported expression type %s." % type(expr))
        return Basic.__new__(cls, dtype, expr)

    @property
    def dtype(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]


class RoutineInplace(RoutineResult):
    """Represents a result provided via an inplace manipulation.

    Parameters
    ----------
    arg : OutArgument or InOutArgument
        Argument in which the result will be returned.
    expr : sympifyable
        Expression to be returned.

    Attributes
    ----------
    dtype : DataType
        Datatype of returned expression.
    argument : OutArgument or InOutArgument
        Argument in which the result will be returned.
    expr : sympifyable
        Expression to be returned.

    """

    def __new__(cls, arg, expr):
        if not isinstance(arg, (OutArgument, InOutArgument)):
            raise TypeError("arg must be of type OutArgument or InOutArgument")
        expr = _sympify(expr)
        if not isinstance(expr, (Expr, MatrixExpr)):
            raise TypeError("Unsupported expression type %s." % type(expr))
        return Basic.__new__(cls, arg, expr)

    @property
    def dtype(self):
        return self.name.dtype

    @property
    def argument(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]


def routine_result(expr):
    """Easy creation of instances of `RoutineResult`.

    `Assign` objects are created as `RoutineInplace`. Everything else is
    created as a `RoutineReturn`. For more control, use the appropriate
    `RoutineResult` class directly.

    Parameters
    ----------
    expr : sympifyable
        Expression for which a RoutineResult should be created.

    """

    expr = _sympify(expr)
    if isinstance(expr, Assign):
        lhs = expr.lhs
        return RoutineInplace(OutArgument(datatype(lhs), lhs), expr.rhs)
    else:
        return RoutineReturn(datatype(expr), expr)


class Routine(Basic):
    """Represents a routine definition.

    Parameters
    ----------
    name : str
        The name of the routine.
    args : iterable
        The arguments to the routine, of type `Argument`.
    results : iterable
        The results of the routine, of type `RoutineResult`.

    Attributes
    ----------
    name : Symbol
        The name of the routine.
    arguments : iterable
        The arguments to the routine, of type `Argument`.
    results : iterable
        The results of the routine, of type `RoutineResult`.
    returns : tuple
        A tuple of all results of type `RoutineReturn`.
    inplace : tuple
        A tuple of all results of type `RoutineInplace`.

    Methods
    -------
    annotate

    """

    def __new__(cls, name, args, results):
        # name
        if isinstance(name, str):
            name = Symbol(name)
        elif not isinstance(name, Symbol):
            raise TypeError("name must be Symbol or string")
        # args
        if not iterable(args):
            raise TypeError("args must be an iterable")
        if not all(isinstance(a, Argument) for a in args):
            raise TypeError("All args must be of type Argument")
        args = Tuple(*args)
        # results
        if not iterable(results):
            raise TypeError("results must be an iterable")
        if not all(isinstance(i, RoutineResult) for i in results):
            raise TypeError("All results must be of type RoutineResult")
        results = Tuple(*results)
        return Basic.__new__(cls, name, args, results)

    @property
    def name(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def results(self):
        return self._args[2]

    @property
    def returns(self):
        return tuple(r for r in self.results if isinstance(r, RoutineReturn))

    @property
    def inplace(self):
        return tuple(r for r in self.results if isinstance(r, RoutineInplace))

    def annotate(self):
        """Prints out a description of the routine"""
        pass

    def __call__(self, *args):
        return RoutineCall(self, args)


def routine(name, args, expr):
    """Easy interface for creating instances of Routine.

    Parameters
    ----------
    name : str
        The name of the routine.
    args : iterable
        The arguments to the routine. Can be Symbols or MatrixSymbols.
    expr
        The expression to generate code for. Can be a single expression,
        or a tuple of expressions. Tuples will result in multiple results.

    Returns
    -------
    Routine

    """

    if isinstance(name, str):
        name = Symbol(name)
    elif not isinstance(name, Symbol):
        raise TypeError("name must be str or Symbol")
    expr = _sympify(expr)
    args = _make_arguments(args, expr)
    results = [routine_result(i) for i in iterate(expr)]
    return Routine(name, args, results)


def _make_arguments(args, expr):
    """Helper function, used for creating Argument instances automatically.

    Infers argument type based on use in the expr. All lhs symbols of
    `Assign` types are set as `OutArguments`. If a symbol is only used to
    calculate an expression, it's an `InArgument`. Symbols that satisfy both
    are `InOutArguments`.

    If arguments are missing for the expression, throws a ValueError.

    Parameters
    ----------
    args : iterable
        An iterable of arguments to the routine.
    expr
        The expression to generate code for, or an iterable of expressions.

    Returns
    -------
    arglist : list
        List of arguments for the routine, in order they were provided.

    """

    frees = expr.free_symbols
    args_set = set(args)
    missing = frees - args_set
    if not args_set == frees:
        raise ValueError("Missing arguments {0}".format(', '.join(
                str(a) for a in missing)))
    outs = set([i.lhs for i in iterate(expr) if isinstance(i, Assign)])
    getvars = lambda x: x.rhs.free_symbols if isinstance(x, Assign) else x.free_symbols
    ins = set.union(*[getvars(i) for i in iterate(expr)])
    inouts = ins.intersection(outs)
    ins = ins - inouts
    outs = outs - inouts
    arglist = []
    for i in args:
        if i in ins:
            arglist.append(InArgument(datatype(i), i))
        elif i in outs:
            arglist.append(OutArgument(datatype(i), i))
        elif i in inouts:
            arglist.append(InOutArgument(datatype(i), i))
        else:
            raise ValueError("How did you even get here????")
    return arglist


# A dictionary of acceptable type aliases. The key is the type of the
# argument defined in the Routine. The value is a tuple of types that
# are acceptable to pass to the routine for that argument.
_accepted_types = {Int: (Int,),
                   Bool: (Bool,),
                   Double: (Double, Float, Int),
                   Float: (Double, Float, Int)}


class RoutineCall(Basic):
    """Represents a call to a `Routine` in the generated code.

    Parameters
    ----------
    routine : Routine
        The `Routine` being called.
    args : iterable
        The arguments being passed to the `Routine`, of type `Argument`.

    Attributes
    ----------
    routine : Routine
        The `Routine` being called.
    arguments : iterable
        The arguments being passed to the `Routine`, of type `Argument`.
    returns : tuple
        A `tuple` of `ReturnCallResult` types. These can be used in other
        expressions, to represent the result of this expression call.
    inplace : dict
        A `dict` of `ReturnCallResult` types. These can be used in other
        expressions, to represent the result of this expression call. The keys
        are the `Symbol` or `MatrixSymbol` representing the `OutArgument` or
        `InOutArgument` that the result is returned via.

    """

    def __new__(cls, routine, args):
        if not isinstance(routine, Routine):
            raise TypeError("routine must be of type Routine")
        if len(routine.arguments) != len(args):
            raise ValueError("Incorrect number of arguments")
        for n, (a, p) in enumerate(zip(args, routine.arguments)):
            cls._validate_arg(a, p)
        args = Tuple(*args)
        return Basic.__new__(cls, routine, args)

    @staticmethod
    def _validate_arg(arg, param):
        arg_type = datatype(arg)
        if arg_type not in _accepted_types[param.dtype]:
            raise ValueError("Type mismatch on argument %s. Expected "
                            "%s, got %s." % (param, param.dtype, arg_type))

    @property
    def routine(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def returns(self):
        """Returns a tuple of return values"""
        return self._returns()

    @property
    def inplace(self):
        """Returns a dict of implicit return values"""
        return self._inplace()

    @staticmethod
    def _result_dispatch(res):
        if isinstance(res.expr, Expr):
            return ScalarRoutineCallResult
        elif isinstance(res.expr, MatrixExpr):
            return MatrixRoutineCallResult
        else:
            raise TypeError("Unknown expression type {0}".format(type(res)))

    @do_once
    def _returns(self):
        ret = self.routine.returns
        if len(ret) == 1:
            return self._result_dispatch(ret[0])(self, -1)
        else:
            return Tuple(*[self._result_dispatch(i)(self, n) for n, i in
                    enumerate(ret)])

    @do_once
    def _inplace(self):
        inp = self.routine.inplace
        d = dict((i.argument.name, self._result_dispatch(i)(self,
                i.argument.name)) for i in iterate(inp))
        return Dict(d)

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join(sstr(a) for a in self.arguments)
        name = sstr(self.routine.name)
        return "{0}({1})".format(name, args)

    def _eval_subs(self, old, new):
        """Don't perform subs inside the routine"""
        args = self.arguments.subs(old, new)
        return self.func(self.routine, args)

    def _eval_simplify(self, **kwargs):
        args = simplify(self.arguments)
        return self.func(self.routine, args)


class RoutineCallResult(Basic):
    """Base class for all objects returned from calls to `Routine`s.

    Objects of this type can be used in further expressions, representing the
    result of the `Routine`. Assumptions for this result are "aliased" to the
    symbolic expression being represented. Classes that subclass from this
    should define `_alias_type` to be the type of object they're aliasing
    (e.g. `Expr` or `MatrixExpr`).

    """

    def __new__(cls, routine_call, idx):
        if not isinstance(routine_call, RoutineCall):
            raise TypeError("routine_call must be of type RoutineCall")
        idx = _sympify(idx)
        if isinstance(idx, Integer):
            if not -1 <= idx < len(routine_call.routine.returns):
                raise ValueError("idx out of bounds")
        elif isinstance(idx, Symbol):
            names = [a.argument.name for a in routine_call.routine.inplace]
            if idx not in names:
                raise KeyError("unknown inplace result %s" % idx)
        # Get the name of the symbol
        if idx == -1:
            expr = routine_call.routine.returns[0].expr
        elif isinstance(idx, Integer):
            expr = routine_call.routine.returns[idx].expr
        else:
            inp = routine_call.routine.inplace
            expr = [i.expr for i in inp if idx == i.argument.name][0]
        # Sub in values to expression
        args = [i.name for i in routine_call.routine.arguments]
        values = [i for i in routine_call.arguments]
        expr = expr.subs(dict(zip(args, values)))
        # Create the object
        s = cls._alias_type.__new__(cls, routine_call, idx)
        s._expr = expr
        cls._alias_assumptions(s, expr)
        return s

    @staticmethod
    def _alias_assumptions(alias, obj):
        """Alias all calls to alias.is_* to obj.is_*. Note that this assumes *no*
        default assumptions"""
        def _make_func(name):
            def lookup(self):
                return getattr(self.expr, 'is_' + name)
            return lookup
        for a in _assume_defined:
            alias._prop_handler[a] = _make_func(a)
        alias._assumptions.clear()

    def _sympystr(self, printer):
        sstr = printer.doprint
        call = sstr(self.rcall)
        if self.idx == -1:
            return "{0}.returns".format(call)
        elif isinstance(self.idx, Integer):
            return "{0}.returns[{1}]".format(call, sstr(self.idx))
        else:
            return "{0}.inplace[{1}]".format(call, sstr(self.idx))

    def _eval_subs(self, old, new):
        """Don't perform subs on the idx"""
        rcall = self.rcall.subs(old, new)
        return self.func(rcall, self.idx)

    @property
    def rcall(self):
        return self._args[0]

    @property
    def idx(self):
        return self._args[1]

    @property
    def expr(self):
        return self._expr

    @property
    def free_symbols(self):
        return self.rcall.arguments.free_symbols


class ScalarRoutineCallResult(RoutineCallResult, Expr):
    """Represents a scalar result returned from a routine call.

    Parameters
    ----------
    routine_call : RoutineCall
        Call that the result is from.
    idx : int or Expr/MatrixExpr
        The index used to get this object. If an `int`, is the index for
        `routine_call.returns`. If `idx` is -1, indicates that there is only
        one `RoutineReturn` for this `Routine`. If `idx` is a
        `Symbol`/`MatrixSymbol`, it's the key used to get the result from
        `routine_call.inplace`.

    Attributes
    ----------
    rcall : Routinecall
        Call that the result is from.
    idx : int or Expr/MatrixExpr
        The index used to get this object.
    expr : sympy expression
        The expression that this result represents.

    """

    _alias_type = Expr


class MatrixRoutineCallResult(RoutineCallResult, MatrixExpr):
    """Represents a matrix result returned from a routine call.

    Parameters
    ----------
    routine_call : RoutineCall
        Call that the result is from.
    idx : int or Expr/MatrixExpr
        The index used to get this object. If an `int`, is the index for
        `routine_call.returns`. If `idx` is -1, indicates that there is only
        one `RoutineReturn` for this `Routine`. If `idx` is a
        `Symbol`/`MatrixSymbol`, it's the key used to get the result from
        `routine_call.inplace`.

    Attributes
    ----------
    rcall : Routinecall
        Call that the result is from.
    idx : int or Expr/MatrixExpr
        The index used to get this object.
    expr : sympy expression
        The expression that this result represents.

    """

    _alias_type = MatrixExpr

    @property
    def shape(self):
        return _sympify(self.expr.shape)

    def _entry(self, i, j):
        return MatrixElement(self, i, j)
