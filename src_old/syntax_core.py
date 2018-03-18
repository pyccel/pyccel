# coding: utf-8

# TODO: - for the moment we do not infer the results type

import numpy as np
from numpy import ndarray
from numpy import asarray

from sympy import Integer as sp_Integer
from sympy import Float   as sp_Float
from sympy.core.expr import Expr
from sympy.core.containers import Tuple
from sympy import Symbol, Integer, Float, Add, Mul,Pow
from sympy import true, false, pi
from sympy.logic.boolalg import And,Or
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.core.basic import Basic
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.function import Function
from sympy import preorder_traversal
from sympy import (Abs, sqrt, sin,  cos,  exp,  log, \
                   csc,  cos,  sec,  tan,  cot,  asin, \
                   acsc, acos, asec, atan, acot, atan2)
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse
from sympy import Lambda
from sympy import sympify
from sympy import symbols as sp_symbols


from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.compatibility import string_types
from sympy.core.operations import LatticeOp
from sympy.core.function import Derivative
from sympy.core.function import _coeff_isneg
from sympy.core.singleton import S
from sympy.utilities.iterables import iterable
from sympy import Integral, Symbol
from sympy.simplify.radsimp import fraction
from sympy.logic.boolalg import BooleanFunction
from sympy.core.containers import Dict

from pyccel.parser.syntax.basic   import BasicStmt

from pyccel.ast.core import allocatable_like
from pyccel.ast.core import FunctionCall,MethodCall
from pyccel.ast.core import ConstructorCall
from pyccel.ast.core import is_pyccel_datatype, is_iterable_datatype
from pyccel.ast.core import DataType, CustomDataType, DataTypeFactory
from pyccel.ast.core import NativeBool, NativeFloat, NativeComplex, NativeDouble, NativeInteger
from pyccel.ast.core import NativeBool, NativeFloat, NativeNil, NativeVector, NativeStencil
from pyccel.ast.core import NativeComplex, NativeDouble, NativeInteger
from pyccel.ast.core import NativeRange, NativeTensor, NativeSymbol
from pyccel.ast.core import Import
from pyccel.ast.core import DottedName
from pyccel.ast.core import Nil
from pyccel.ast.core import Random
from pyccel.ast.core import Eval
from pyccel.ast.core import Load
from pyccel.ast.core import EmptyLine
from pyccel.ast.core import (Tile, Range, Tensor, \
                             For, ForIterator, Assign, \
                             Declare, Vector, Stencil, Variable, ValuedVariable, \
                             FunctionHeader, ClassHeader, MethodHeader, \
                             VariableHeader, \
                             datatype, While, With, NativeFloat, \
                             EqualityStmt, NotequalStmt, \
                             AugAssign, \
                             FunctionDef, ClassDef, Del, Print, \
                             Comment, AnnotatedComment, \
                             IndexedVariable, Slice, Assert, If, \
                             Ceil, Break, Continue, Raise, \
                             Zeros, Ones, Array, ZerosLike, Shape, Len, \
                             Dot, Sign, IndexedElement,\
                             Pass, \
                             Min, Max, Mod)

from pyccel.ast.parallel.mpi import MPI


DEBUG = False
#DEBUG = True

# TODO set to None
DEFAULT_TYPE = 'double'

known_functions = {
    "abs": "Abs",
#    "asin": "asin",
#    "acsc": "acsc",
#    "acot": "acot",
#    "acos": "acos",
#    "asec": "asec",
    "atan": "atan",
    "atan2": "atan2",
    "ceil": "Ceil",
    "cos": "cos",
    "cosh": "cosh",
    "cot": "cot",
    "csc": "csc",
    "dot": "dot",
    "exp": "exp",
    "len": "Len",
    "log": "log",
    "min": "Min",
    "max": "Max",
    "pow": "pow",
    "mod": "Mod",
    "sec": "sec",
    "shape": "Shape",
    "sign": "Sign",
    "sin": "sin",
    "sinh": "sinh",
    "sqrt": "sqrt",
    "vector": "Vector",
    "stencil": "Stencil",
    "eval": "Eval",
    "load": "Load",
    "random": "Random",
    "lambdify": "lambdify",
    "tan": "tan",
    "tanh": "tanh"
}

# TODO: treat the inout case

# ...
def get_headers():
    """Returns the global variable headers."""
    global headers

    return headers
# ...

# ...
def get_namespace():
    """Returns the global variable namespace."""
    global namespace

    return namespace
# ...

# ...
def clean_namespace():
    """Cleans the global variables."""
    global namespace
    global declarations
    global _extra_stmts

    namespace = {}
    namespace['cls_constructs'] = {}

    declarations   = {}
    _extra_stmts   = []
# ...

# ...
def update_namespace(d):
    """Updates the global variables."""
    global namespace

    for k,v in d.items():
        if not(k == 'cls_constructs'):
            if k in namespace:
                raise ValueError('{0} already exists in namespace.'.format(k))

            namespace[k] = v

    if 'cls_constructs' in d:
        for k,v in d['cls_constructs'].items():
            namespace['cls_constructs'][k] = v
# ...

# ...
def get_class_construct(name):
    """Returns the class datatype for name."""
    global namespace

    return namespace['cls_constructs'][name]
# ...

# ...
def set_class_construct(name, value):
    """Sets the class datatype for name."""
    global namespace

    namespace['cls_constructs'][name] = value
# ...

# ...
def datatype_from_string(txt):
    if not isinstance(txt, str):
        raise TypeError('Expecting a string')

    if txt == 'int':
        return NativeInteger()
    elif txt == 'float':
        return NativeFloat()
    elif txt == 'double':
        return NativeDouble()
    elif txt == 'complex':
        return NativeComplex()
    elif txt == 'bool':
        return NativeBool()
# ...

# Global variable namespace
namespace    = {}
namespace['cls_constructs'] = {}

headers      = {}
declarations = {}
_extra_stmts  = []

# TODO do we keep this?
namespace["True"]  = true
namespace["False"] = false
namespace["pi"]    = pi


# ... builtin types
builtin_types      = ['int', 'float', 'double', 'complex', 'bool']
builtin_datatypes  = [datatype(i) for i in builtin_types]
# ...

# ... builtin functions
builtin_funcs_math_nores = ['print']
builtin_funcs_math_noarg = []

builtin_funcs_math_un = ['abs', \
#                         'asin', 'acsc', 'acot', \
#                         'acos', 'asec', \
                         'atan', 'atan2', \
                         'ceil', 'cos', 'cosh', 'cot', 'csc', \
                         'exp', 'log', 'max', 'min', \
                         'sec', 'sign', 'sin', 'sinh', \
                         'sqrt', 'tan', 'tanh']
builtin_funcs_math_bin = ['dot', 'pow', 'mod']

builtin_funcs_math  = builtin_funcs_math_noarg
builtin_funcs_math += builtin_funcs_math_un
builtin_funcs_math += builtin_funcs_math_bin
builtin_funcs_math += ['random']

builtin_funcs  = ['zeros', 'ones', 'array', 'zeros_like',
                  'len', 'shape', 'vector', 'stencil',
                  'eval', 'load', 'lambdify']
builtin_funcs += builtin_funcs_math

builtin_funcs_iter = ['range', 'tensor']
builtin_funcs += builtin_funcs_iter
# ...

# ...
def print_namespace():
    print("-------- namespace --------")
    for key, value in list(namespace.items()):
        if not(key in ['True', 'False', 'pi']):
            print('{0} :: {1}'.format(key, type(value)))
    print("---------------------------")
# ...

# ...
def print_declarations():
    print("-------- declarations --------")
    for key, value in list(declarations.items()):
        print('{0} :: {1}'.format(key, value))
    print("---------------------------")
# ...

def infere_type(expr):
    """
    finds attributs of the expression
    """
    d_var = {}
    d_var['datatype']    = None
    d_var['allocatable'] = None
    d_var['shape']       = None
    d_var['rank']        = None
    d_var['is_pointer']  = None

    if isinstance(expr, dict):
        d_var['datatype']    = expr['datatype']
        d_var['allocatable'] = expr['allocatable']
        d_var['shape']       = expr['shape']
        d_var['rank']        = expr['rank']
        return d_var
    elif isinstance(expr, (list, tuple)):
        if not expr:
            return d_var

        ds = [infere_type(a) for a in expr]

        a = ds[0]
        d_var['datatype']    = a['datatype']
        d_var['allocatable'] = a['allocatable']
        d_var['shape']       = a['shape']
        d_var['rank']        = a['rank']
        if len(ds) == 1:
            return d_var
        for a in ds[1:]:
            if a['datatype'] == 'double':
                d_var['datatype'] = a['datatype']
            if a['allocatable']:
                d_var['allocatable'] = a['allocatable']
            if a['shape']:
                d_var['shape'] = a['shape']
            if a['rank'] > 0:
                d_var['rank'] = a['rank']
        return d_var
    elif isinstance(expr, Tuple):
        a = infere_type(expr[0])

        d_var['datatype']    = a['datatype']
        d_var['allocatable'] = False
        d_var['shape']       = len(expr)
        d_var['rank']        = 1
        return d_var
    elif isinstance(expr, Integer):
        d_var['datatype']    = 'int'
        d_var['allocatable'] = False
        d_var['rank']        = 0
        return d_var
    elif isinstance(expr, Float):
        # TODO choose precision
        d_var['datatype']    = DEFAULT_TYPE
        d_var['allocatable'] = False
        d_var['rank']        = 0
        return d_var
    elif isinstance(expr, (BooleanTrue, BooleanFalse)):
        d_var['datatype']    = NativeBool()
        d_var['allocatable'] = False
        d_var['rank']        = 0
        return d_var
    elif isinstance(expr, str):
        d_var['datatype']    = NativeBool()
        d_var['allocatable'] = True
        d_var['rank']        = 1
        d_var['shape']       = len(expr)
        return d_var
    elif isinstance(expr, Nil):
        d_var['datatype']    = NativeNil()
        return d_var
    elif isinstance(expr, MPI):
        raise NotImplementedError('')
    elif isinstance(expr, CustomDataType):
        raise NotImplementedError('')
        d_var['datatype']    = expr
        d_var['allocatable'] = False
        d_var['shape']       = None
        d_var['rank']        = 0
        d_var['cls_base']    = None
        return d_var
    elif isinstance(expr, DottedName):
        var    = get_class_attribut(expr)
        parent = str(expr.name[0])

        d_var['datatype']    = var.dtype
        d_var['allocatable'] = var.allocatable
        d_var['shape']       = var.shape
        d_var['rank']        = var.rank

        d_var['cls_base']    = namespace[parent]
        return d_var
    elif isinstance(expr, (Ceil, Len)):
        d_var['datatype']    = 'int'
        d_var['allocatable'] = False
        d_var['rank']        = 0
        return d_var
    elif isinstance(expr, Random):
        d_var['datatype']    = 'double'
        d_var['allocatable'] = False
        d_var['rank']        = 0
        return d_var
    elif isinstance(expr, (Dot, Min, Max, Sign)):
        arg = expr.args[0]
        if isinstance(arg, Integer):
            d_var['datatype'] = 'int'
        elif isinstance(arg, Float):
            d_var['datatype'] = DEFAULT_TYPE
        elif isinstance(arg, Variable):
            d_var['datatype'] = arg.dtype
        d_var['allocatable'] = False
        d_var['rank']        = 0
        return d_var
    elif isinstance(expr, IndexedVariable):
        name = str(expr)
        if name in namespace:
            var = namespace[name]

            d_var['datatype']    = var.dtype
            d_var['allocatable'] = var.allocatable
            d_var['shape']       = var.shape
            d_var['rank']        = var.rank
            return d_var
    elif isinstance(expr, IndexedElement):
        d_var['datatype']    = expr.dtype
        name = str(expr.base)
        if name in namespace:
            var = namespace[name]
            d_var['datatype']    = var.dtype

            if iterable(var.shape):
                shape = []
                for s,i in zip(var.shape, expr.indices):
                    if isinstance(i, Slice):
                        shape.append(i)
            else:
                shape = None

            rank = var.rank - expr.rank
            if rank > 0:
                d_var['allocatable'] = var.allocatable

            d_var['shape']       = shape
            d_var['rank']        = rank
            return d_var
    elif isinstance(expr, Variable):
        d_var['datatype']    = expr.dtype
        d_var['allocatable'] = expr.allocatable
        d_var['shape']       = expr.shape
        d_var['rank']        = expr.rank
        return d_var
    elif isinstance(expr, ConstructorCall):
        this = expr.this
        # this datatype is polymorphic
        dtype = this.dtype

        if not(dtype.name in namespace):
            raise ValueError('Undeclared class type {0}'.format(dtype.name))

        # remove Pyccel from prefix
        prefix = dtype.prefix
        prefix = prefix.replace('Pyccel', '')

        dtype = DataTypeFactory(dtype.name, ("_name"), \
                                prefix=prefix, \
                                alias=dtype.alias, \
                                is_iterable=dtype.is_iterable, \
                                is_with_construct=dtype.is_with_construct, \
                                is_polymorphic=False)()

        d_var['datatype']    = dtype
        d_var['allocatable'] = this.allocatable
        d_var['shape']       = this.shape
        d_var['rank']        = this.rank
        d_var['cls_base']    = namespace[dtype.name]
        d_var['cls_parameters'] = expr.arguments[1:]
        return d_var
    elif isinstance(expr, FunctionCall):
        func = expr.func
        results = func.results
        if not(len(results) == 1):
            raise ValueError("Expecting one result, given : {}".format(results))
        result = results[0]
        d_var['datatype']    = result.dtype
        d_var['allocatable'] = result.allocatable
        d_var['shape']       = result.shape
        d_var['rank']        = result.rank
        return d_var
    elif isinstance(expr, Function):
        name = str(type(expr).__name__)
        avail_funcs = builtin_funcs
        avail_funcs = []
        for n, F in list(namespace.items()):
            if isinstance(F, (FunctionDef, Lambda)):
                avail_funcs.append(str(n))
        avail_funcs += builtin_funcs

        # this is to treat the upper/lower cases
        _known_functions = []
        for k, n in list(known_functions.items()):
            _known_functions += [k, n]
        avail_funcs += _known_functions

        if not(name in avail_funcs):
            raise Exception("Could not find function {0}".format(name))

        if name in namespace:
            F = namespace[name]

            if isinstance(F, FunctionDef):
                results = F.results

                if not(len(results) == 1):
                    raise ValueError("Expecting a function with one return.")

                var = results[0]
                d_var['datatype']    = var.dtype
                d_var['allocatable'] = var.allocatable
                d_var['rank']        = var.rank
                if not(var.shape is None):
                    d_var['shape'] = var.shape
            elif isinstance(F, Lambda):
                d_var['datatype'] = NativeSymbol()
            else:
                raise NotImplementedError('TODO')

        elif name in _known_functions:
            var = expr.args[0]
            if isinstance(var, Integer):
                d_var['datatype'] = 'int'
                d_var['allocatable'] = False
                d_var['rank']        = 0
            elif isinstance(var, Float):
                d_var['datatype'] = DEFAULT_TYPE
                d_var['allocatable'] = False
                d_var['rank']        = 0
            elif isinstance(var, Variable):
                d_var['datatype']    = var.dtype
                d_var['allocatable'] = var.allocatable
                d_var['rank']        = var.rank
                d_var['shape']       = var.shape
        else:
            raise ValueError("Undefined function {}".format(name))
        return d_var
    elif isinstance(expr, Expr):
        skipped = (Variable, IndexedVariable, IndexedElement, \
                   Function, FunctionDef)
        args = []
        for arg in expr.args:
            if not(isinstance(arg, skipped)):
                args.append(arg)
            else:
                if isinstance(arg, (Variable, IndexedVariable, FunctionDef)):
#                    print_namespace()
                    name = arg.name
                    if not isinstance(name, DottedName):
                        var = namespace[name]
                        return infere_type(var)
                    else:
                        # see remark in expr_with_trailer (TrailerDots)
                        cls_base = namespace[name.name[0]]
                        # TODO wil not work in nested classes
                        member   = name.name[1]

                        attributs = []
                        if isinstance(cls_base, ClassDef):
                            d_attributs = cls_base.attributs_as_dict
                            if not(member in d_attributs):
                                raise ValueError('{0} is not a member of '
                                                 '{1}'.format(member, expr))

                            attribut = d_attributs[member]
                            return infere_type(attribut)
                else:
                    return infere_type(arg)
        return infere_type(args)
    else:
        raise TypeError("infere_type is not available for {0}".format(type(expr)))

    return d_var

# TODO add kwargs
def builtin_function(name, args, lhs=None, op=None):
    """
    User friendly interface for builtin function calls.

    name: str
        name of the function
    args: list
        list of arguments
    lhs: str
        name of the variable to assign to
    op: str
        operation for AugAssign statement
    """
    # ...
    def get_arguments_zeros():
        # TODO appropriate default type
        dtype = DEFAULT_TYPE
        allocatable = True
        shape = []
        grid = None
        rank = None
        for i in args:
            if isinstance(i, DataType):
                dtype = i
            elif isinstance(i, Tuple):
                shape = [j for j in i]
            elif isinstance(i, Tensor):
                grid = i
                rank = len(grid.ranges)
            elif isinstance(i, Range):
                grid = i
                rank = 1
            elif isinstance(i, Variable):
                ctype = i.dtype
                if ctype in builtin_datatypes:
                    shape.append(i)
                else: # iterator
                    cls_name = ctype.name
                    obj = eval(cls_name)()
                    grid = obj.get_ranges(i)
                    # grid is now a Tensor

                    rank = grid.dim
            elif isinstance(i, Integer):
                shape.append(i)
            else:
                raise TypeError('wrong argument {0} of type {1}'.format(i, type(i)))

        if rank is None:
            rank = len(shape)

        if rank == 1:
            if not grid:
                shape = shape[0]

        if isinstance(shape, (list, tuple, Tuple)):
            if len(shape) == 0:
                shape = None

        d_var = {}
        d_var['datatype'] = dtype
        d_var['allocatable'] = allocatable
        d_var['shape'] = shape
        d_var['rank'] = rank

        return d_var, grid
    # ...

    # ...
    def get_arguments_array():
        # TODO appropriate default type
        dtype = DEFAULT_TYPE
        allocatable = True
        for i in args:
            if isinstance(i, DataType):
                dtype = i
            elif isinstance(i, Tuple):
                arr = [j for j in i]
            else:
                raise TypeError("Expecting a Tuple or DataType.")
        arr = asarray(arr)
        shape = arr.shape
        rank = len(shape)

        d_var = {}
        d_var['datatype'] = dtype
        d_var['allocatable'] = allocatable
        d_var['shape'] = shape
        d_var['rank'] = rank

        return d_var, arr
    # ...

    # ...
    def assign(l, expr, op, strict=False, status=None, like=None):
        if op is None:
            return Assign(l, expr, \
                          strict=strict, \
                          status=status, \
                          like=like)
        else:
            return AugAssign(l, op, expr, \
                          strict=strict, \
                          status=status, \
                          like=like)
    # ...

    # ...
    if name in ["zeros", "ones"]:
        if not lhs:
            raise ValueError("Expecting a lhs.")
        d_var, grid = get_arguments_zeros()
#        print_namespace()
        insert_variable(lhs, **d_var)
#        print('\n')
#        print_namespace()
#        import sys; sys.exit(0)
        lhs = namespace[lhs]
        f_name = name.capitalize()
        f_name = eval(f_name)
        return f_name(lhs, shape=d_var['shape'], grid=grid)
    elif name == "array":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        d_var, arr = get_arguments_array()
        insert_variable(lhs, **d_var)
        lhs = namespace[lhs]
        return Array(lhs, arr, d_var['shape'])
    elif name == "zeros_like":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        if not(len(args) == 1):
            raise ValueError("Expecting exactly one argument.")
        if not(args[0].name in namespace):
            raise ValueError("Undefined variable {0}".format(name))

        var = args[0]

        d_var = {}
        d_var['datatype']    = var.dtype
        d_var['allocatable'] = True
        d_var['shape']       = var.shape
        d_var['rank']        = var.rank

        insert_variable(lhs, **d_var)
        lhs = namespace[lhs]
        return ZerosLike(lhs, var)
    elif name == "dot":
        # TODO do we keep or treat inside math_bin?
        if lhs is None:
            return Dot(*args)
        else:
            d_var = {}
            d_var['datatype'] = args[0].dtype
            d_var['rank']     = 0
            insert_variable(lhs, **d_var)
            expr = Dot(*args)
            return assign(lhs, expr, op)
    elif name in ['max', 'min']:
        func = eval(known_functions[name])
        if lhs is None:
            return func(*args)
        else:
            d_var = {}
            d_var['datatype'] = args[0].dtype
            d_var['rank']     = 0
            insert_variable(lhs, **d_var)
            lhs = namespace[lhs]
            expr = func(*args)
            return assign(lhs, expr, op)
    elif name in ['mod']:
        func = eval(known_functions[name])
        if lhs is None:
            return func(*args)
        else:
            d_var = {}
            d_var['datatype'] = args[0].dtype
            d_var['rank']     = 0
            insert_variable(lhs, **d_var)
            lhs = namespace[lhs]
            expr = func(*args)
            return assign(lhs, expr, op)
    elif name in builtin_funcs_math_un + ['len', 'shape']:
        if not(len(args) == 1):
            raise ValueError("function takes exactly one argument")

        func = eval(known_functions[name])
        if lhs is None:
            return func(*args)
        else:
            d_var = {}
            # TODO get dtype from args
            if name in ['ceil', 'len']:
                d_var['datatype'] = 'int'
                d_var['rank']     = 0
            elif name in ['shape']:
                d_var['datatype'] = 'int'
                d_var['rank']     = 1
                d_var['allocatable'] = True
            else:
                d_var['datatype'] = DEFAULT_TYPE
            insert_variable(lhs, **d_var)
            lhs = namespace[lhs]
            expr = func(*args)
            return assign(lhs, expr, op)
    elif name in ['random']:
        # TODO add arguments
#        if not(len(args) == 0):
#            raise ValueError("function takes no arguments")

        func = eval(known_functions[name])
        _args = [None]
        if lhs is None:
            return func(*_args)
        else:
            d_var = {}
            d_var['datatype'] = 'double'
            d_var['rank']     = 0
            insert_variable(lhs, **d_var)
            lhs = namespace[lhs]
            expr = func(*_args)
            return assign(lhs, expr, op)
    elif name in builtin_funcs_math_bin:
        if not(len(args) == 2):
            raise ValueError("function takes exactly two arguments")

        func = eval(known_functions[name])
        if lhs is None:
            return func(*args)
        else:
            d_var = {}
            # TODO get dtype from args
            d_var['datatype'] = DEFAULT_TYPE
            insert_variable(lhs, **d_var)
            lhs = namespace[lhs]
            expr = func(*args)
            return assign(lhs, expr, op)
    elif name == "range":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        if not(len(args) in [2, 3]):
            raise ValueError("Expecting exactly two or three arguments.")

        expr = Range(*args)

        d_var = {}
        d_var['datatype']    = NativeRange()
        d_var['allocatable'] = False
        d_var['shape']       = None
        d_var['rank']        = 0
        d_var['cls_base']    = expr

        # needed when lhs is a class member
        if lhs in namespace:
            if isinstance(namespace[lhs], Symbol):
                namespace.pop(lhs)

        insert_variable(lhs, **d_var)
#        print_namespace()
        lhs = namespace[lhs]
        return assign(lhs, expr, op, strict=False)
    elif name == "tensor":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        if not(len(args) in [2, 3]):
            raise ValueError("Expecting exactly two or three arguments.")

        expr = Tensor(*args)

        d_var = {}
        d_var['datatype']    = NativeTensor()
        d_var['allocatable'] = False
        d_var['shape']       = None
        d_var['rank']        = 0
        d_var['cls_base']    = expr

        # needed when lhs is a class member
        if lhs in namespace:
            if isinstance(namespace[lhs], Symbol):
                namespace.pop(lhs)

        insert_variable(lhs, **d_var)
#        print_namespace()
        lhs = namespace[lhs]
        return assign(lhs, expr, op, strict=False)
    elif name == "vector":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        if not(len(args) in [2, 3]):
            raise ValueError("Expecting exactly two or three arguments.")

        _args = []
        for i in args:
            if not isinstance(i, (list, tuple, Tuple)):
                _args.append([i])
            else:
                _args.append(i)
        args = _args

        d_var = {}
        d_var['datatype']    = NativeVector()
        d_var['allocatable'] = True
        d_var['shape']       = None
        d_var['rank']        = len(args[0])

        # needed when lhs is a class member
        if lhs in namespace:
            if isinstance(namespace[lhs], Symbol):
                namespace.pop(lhs)

        insert_variable(lhs, **d_var)
        expr = Vector(lhs, *args)
        return expr
#        namespace[lhs] = expr
#        return assign(lhs, expr, op, strict=False)
    elif name == "stencil":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        if not(len(args) in [2, 3]):
            raise ValueError("Expecting exactly two or three arguments.")

        _args = []
        for i in args:
            if not isinstance(i, (list, tuple, Tuple)):
                _args.append([i])
            else:
                _args.append(i)
        args = _args

        d_var = {}
        d_var['datatype']    = NativeStencil()
        d_var['allocatable'] = True
        d_var['shape']       = None
        d_var['rank']        = len(args[0]) + len(args[-1])
        # because 'args[0] = starts' and  'args[-1] = pads'

        # needed when lhs is a class member
        if lhs in namespace:
            if isinstance(namespace[lhs], Symbol):
                namespace.pop(lhs)

        insert_variable(lhs, **d_var)
        expr = Stencil(lhs, *args)
        return expr
#        namespace[lhs] = expr
#        return assign(lhs, expr, op, strict=False)
    elif name == "eval":
        if not lhs:
            raise ValueError("Expecting a lhs.")
#        if not(len(args) in [2, 3]):
#            raise ValueError("Expecting exactly two or three arguments.")

        _args = []
        for i in args:
            if not isinstance(i, (list, tuple, Tuple)):
                _args.append([i])
            else:
                _args.append(i)
        args = _args

        d_var = {}
        d_var['datatype'] = NativeSymbol()

        # needed when lhs is a class member
        if lhs in namespace:
            if isinstance(namespace[lhs], Symbol):
                namespace.pop(lhs)

        insert_variable(lhs, **d_var)
        expr = Eval(lhs, *args)
        return expr
    elif name == "load":
        if not lhs:
            raise ValueError("Expecting a lhs.")

        # right now args are all of type sympy Symbol
        _args = []
        for i in args:
            a = None
            if isinstance(i, Symbol):
                a = str(i)
            elif isinstance(i, (list, tuple, Tuple)):
                a = [str(j) for j in i]
            else:
                a = i
            _args.append(a)
        args = _args

        loader = Load(*args)
        funcs  = loader.execute()
        for f_name, f in zip(loader.funcs, funcs):
            if str(f_name) in namespace:
                raise ValueError('{0} already in defined.'.format(f))

            namespace[str(f_name)] = f

        # TODO keep it like this?
        return None
    elif name == "lambdify":
        if not lhs:
            raise ValueError("Expecting a lhs.")

        func = args[0]
        if not isinstance(func, Lambda):
            raise TypeError('Expecting a Lambda function, given'
                            ' {0}'.format(type(func)))

        f_name = str(func)

        arguments = []
        for a in func.variables:
            arg = Variable('double', str(a))
            arguments.append(arg)


        # since we allow Lambda expressions to return a Tuple,
        # we have to use isinstance
        if isinstance(func.expr, (Tuple, list, tuple)):
            n = len(func.expr)
            x_out   = Variable('double', 'x_out', rank=1, shape=n)
            results = [x_out]

            expressions = []
            for i in func.expr:
                expressions += [sympify(i).evalf()]
            expr = Tuple(*expressions)
        else:
            # TODO to treat other cases
            x_out   = Variable('double', 'x_out')
            results = [x_out]

            expr = sympify(func.expr).evalf()

        body = [Assign(x_out, expr)]

        # TODO local_vars must be updated inside FunctionDef
        #      this is needed for _print_FunctionDef
        F = FunctionDef(str(lhs), arguments, results, body, local_vars=arguments)
        namespace[str(lhs)] = F
        return F
    else:
        raise ValueError("Expecting a builtin function. given : ", name)
    # ...
# ...

# ...
def get_class_attribut(name):
    """
    Returns the attribut (if exists) of a class providing a DottedName.
    In the special case of class attribut, this function will return None, if
    the attribut is not a member of the class.

    name: DottedName
        a class attribut
    """
    parent = str(name.name[0])
    if not(parent in namespace):
        raise ValueError('Undefined object {0}'.format(parent))
    if len(name.name) > 2:
        raise ValueError('Only one level access is available.')

    attr_name = str(name.name[1])
    cls = namespace[parent]
    if not isinstance(cls, Variable):
        raise TypeError("Expecting a Variable")
    if not is_pyccel_datatype(cls.dtype):
        raise TypeError("Expecting a Pyccel DataType instance")
    dtype = cls.dtype
    cls_name = dtype.name
    cls = namespace[cls_name]

    attributs = {}
    for i in cls.attributs:
        attributs[str(i.name)] = i

    if not (attr_name in attributs):
        return None

    return attributs[attr_name]
# ...

# ...
def insert_variable(var_name, \
                    datatype=None, \
                    rank=None, \
                    allocatable=None, \
                    shape=None, \
                    intent=None, \
                    var=None, \
                    cls_base=None, \
                    to_declare=True, \
                    value=None, \
                    cls_parameters=None):
    """
    Inserts a variable as a symbol into the namespace. Appends also its
    declaration and the corresponding variable.

    var_name: str
        variable name

    datatype: str, DataType
        datatype variable attribut. One among {'int', 'float', 'complex'}

    allocatable: bool
        if True then the variable needs memory allocation.

    rank: int
        if rank > 0, then the variable is an array

    shape: int or list of int
        shape of the array.

    intent: None, str
        used to specify if the variable is in, out or inout argument.

    var: pyccel.ast.core.Variable
        if attributs are not given, then var must be provided.

    cls_base: class
        class base if variable is an object or an object member

    to_declare:
        declare the variable if True.

    value: Expr
        variable value

    cls_parameters: list, tuple
        a list of parameters. These are the arguments that are passed to the
        class constructor
    """
    if type(var_name) in [int, float]:
        return

    if DEBUG:
#    if True:
        print("> inserting : {0}".format(var_name))
        txt = ('     datatype    = {0} \n'
               '     rank        = {1}, \n'
               '     allocatable = {2}, \n'
               '     shape       = {3}, \n'
               '     intent      = {4}').format(datatype, rank,
                                                allocatable, shape,
                                                intent)
        print(txt)

    if isinstance(var_name, Variable):
        var_name = var_name.name

    if not isinstance(var_name, (str, DottedName)):
        raise TypeError("Expecting a str or DottedName, "
                        "given {0}.".format(type(var_name)))

    if isinstance(var_name, str):
        if var_name in namespace:
            var = namespace[var_name]
    elif isinstance(var_name, DottedName):
        var = get_class_attribut(var_name)
        var_name = str(var_name)
        to_declare = False

    if var:
        if datatype is None:
            datatype = var.dtype
        if rank is None:
            rank = var.rank
        if allocatable is None:
            allocatable = var.allocatable
        if shape is None:
            shape = var.shape
    if not rank:
        rank=0

    # we create a variable (for annotation)
    var = Variable(datatype, var_name, \
                   rank=rank, \
                   allocatable=allocatable, \
                   shape=shape, \
                   cls_base=cls_base, \
                   cls_parameters=cls_parameters)

    # we create a declaration for code generation
    dec = Declare(datatype, var, intent=intent, value=value)

    if var_name in namespace:
        namespace.pop(var_name)
        declarations.pop(var_name, None)

    namespace[var_name] = var
    if to_declare:
        declarations[var_name] = dec

# ...
def expr_with_trailer(expr, trailer):
    # we apply the 'expr' property after checking which type is the trailer

    # we use str(.) because expr.name can be a DottedName
    if isinstance(expr, str):
        expr = namespace[expr]

    if isinstance(trailer, TrailerSubscriptList):
        args = trailer.expr
        if not hasattr(args, '__iter__'):
            args = [args]


        expr = IndexedVariable(expr.name, dtype=expr.dtype).__getitem__(*args)

    elif isinstance(trailer, TrailerDots):

        # TODO add IndexedVariable, IndexedElement
        if not(isinstance(expr, Variable)):
            raise TypeError("Expecting Variable, given "
                            "{0}".format(type(expr)))

        if not expr.cls_base:
            raise ValueError("Expecting an object")

        arg      = trailer.expr
        cls_base = expr.cls_base

        # be careful, inside '__init__' attributs are not yet known
        # for the moment, inside the definition of the class, cls_base is of
        # type str
        attributs = []
        if isinstance(cls_base, ClassDef):
            d_attributs = cls_base.attributs_as_dict
            if not(arg in d_attributs):
                raise ValueError('{0} is not a member of '
                                 '{1}'.format(arg, expr))

            attribut = d_attributs[arg]

            var_name = DottedName(expr.name, arg)
            var = attribut.clone(var_name)
            return var
        elif isinstance(cls_base, str):
            var_name = DottedName(expr.name, arg)
            if str(var_name) in namespace:
                expr = namespace[str(var_name)]
            else:
                if not(expr.name in namespace):
                    raise ValueError("Undefined variable {}".format(expr.name))

                expr = DottedName(expr, arg)

                attr = get_class_attribut(expr)
                if not(attr is None):
                    return attr
                else:
                    # now, we insert the class attribut as a sympy Symbol in the
                    # namespace. Later, this will be decorated, when processing an
                    # AssignStmt.
                    namespace[str(expr)] = Symbol(str(expr))
        else:
            raise TypeError('expecting a ClassDef or str')

    elif isinstance(trailer, TrailerArgList):
        args = trailer.expr

        ls = []
        for i in args:
            if isinstance(i, (list, tuple)):
                ls.append(Tuple(*i))
            else:
                ls.append(i)
        args = ls
        name = str(expr)
        if name in builtin_funcs_math + ['len']:
            expr = builtin_function(name, args)
        elif isinstance(expr, ClassDef):
            cls = namespace[str(expr.name)]
            methods = cls.methods
            for i in methods:
                if str(i.name)=='__init__':
                    method=i

            d_var = {}
            dtype = get_class_construct(str(expr.name))()

            d_var['datatype']    = dtype
            d_var['allocatable'] = False
            d_var['shape']       = None
            d_var['rank']        = 0
            d_var['intent']      = 'inout'
            d_var['cls_base']    = namespace[str(expr.name)]

            insert_variable('self', **d_var)
            args = [namespace['self']] + list(args)
            expr = ConstructorCall(method, args, cls_variable=namespace['self'])
        elif isinstance(expr, (FunctionDef, Lambda)):
            expr = expr(*args)
#            print('> expr = {0}'.format(expr))
        else:
            f_name = str(expr)

            # ... prepare args
            _args = []
            for a in args:
                if str(a) in namespace:
                    _args.append(namespace[str(a)])
                else:
                    # TODO may be we should raise an error here
                    _args.append(a)
            # ...

            # ... we need to remove '.' from _args for 'load'
            #     see Load in ast for more details
            if f_name == 'load':
                _args[0] = _args[0].replace('.', '__')
            # ...

            # ...
            if f_name in builtin_funcs + namespace.keys():
                if f_name in namespace:
                    F = namespace[f_name]
                    expr = F(*_args)
                else:
                    expr = Function(f_name)(*_args)
            else:
                raise TypeError('Wrong type for {0}, '
                                'given {1}'.format(f_name, type(expr)))
            # ...
    return expr
# ...

# ...
# TODO: refactoring
def do_arg(a):
    if isinstance(a, str):
        arg = Symbol(a, integer=True)
    elif isinstance(a, (Integer, Float)):
        arg = a
    elif isinstance(a, ArithmeticExpression):
        arg = a.expr
        if isinstance(arg, (Symbol, Variable)):
            arg = Symbol(arg.name, integer=True)
        else:
            arg = convert_to_integer_expression(arg)
    else:
        raise Exception('Wrong instance in do_arg')

    return arg
# ...

# ... TODO improve. this version is not working with function calls
def convert_to_integer_expression(expr):
    """
    converts an expression to an integer expression.

    expr: sympy.expression
        a sympy expression
    """
    args_old = expr.free_symbols
    args = [Symbol(str(a), integer=True) for a in args_old]
    for a,b in zip(args_old, args):
        expr = expr.subs(a,b)
    return expr
# ...

# ...
def is_Float(s):
    """
    returns True if the string s is a float number.

    s: str, int, float
        a string or a number
    """
    try:
        float(s)
        return True
    except:
        return False
# ...

# ...
def convert_numpy_type(dtype):
    """
    convert a numpy type to standard python type that are understood by the
    syntax.

    dtype: int, float, complex
        a numpy datatype
    """
    # TODO improve, numpy dtypes are int64, float64, ...
    if dtype == int:
        datatype = 'int'
    elif dtype == float:
        datatype = DEFAULT_TYPE
    elif dtype == complex:
        datatype = 'complex'
    else:
        raise TypeError('Expecting int, float or complex for numpy dtype.')
    return datatype
# ...

# ...
class Pyccel(object):
    """Class for Pyccel syntax."""

    def __init__(self, **kwargs):
        """
        Constructor for Pyccel.

        statements : list
            list of parsed statements.
        """
        self.statements = kwargs.pop('statements', [])

    @property
    def declarations(self):
        """
        Returns the list of all declarations using objects from pyccel.ast.core
        """
        d = {}
        for key,dec in list(declarations.items()):
            if dec.intent is None:
                d[key] = dec
        return d

    # TODO write as a property
    def get_namespace(self):
        """
        Returns the list of all namespace using objects from pyccel.ast.core
        """
        d = {}
        for key,dec in list(namespace.items()):
            d[key] = dec
        return d

    @property
    def extra_stmts(self):
        """
        Returns the list of all extra_stmts
        """
        return _extra_stmts

    # TODO add example
    @property
    def expr(self):
        """Converts the IR into AST that is fully compatible with sympy."""
        ast = []
        for stmt in self.statements:
            expr = stmt.expr
            if not(expr is None):
                ast.append(expr)

        return ast

class ConstructorStmt(BasicStmt):
    """
    Class representing a Constructor statement.

    Constructors are used to mimic static typing in Python.
    """
    def __init__(self, **kwargs):
        """
        Constructor for the Constructor statement class.

        lhs: str
            variable to construct
        constructor: str
            a builtin constructor
        """
        self.lhs         = kwargs.pop('lhs')
        self.constructor = kwargs.pop('constructor')

        super(ConstructorStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Constructor statement by inserting the lhs variable in the
        global dictionaries.
        """
        var_name    = str(self.lhs)
        constructor = str(self.constructor)
        # TODO improve
        rank     = 0
        datatype = constructor
        insert_variable(var_name, datatype=datatype, rank=rank)
        return EmptyLine()

# TODO: improve by creating the corresponding object in pyccel.ast.core
class DelStmt(BasicStmt):
    """Class representing a delete statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the Delete statement class.

        variables: list of str
            variables to delete
        """
        self.variables = kwargs.pop('variables')

        super(DelStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Delete statement by returning a pyccel.ast.core object
        """
        variables = [v.expr for v in self.variables]
        ls = []
        for var in variables:
            if isinstance(var, Variable):
                name = var.name
                if isinstance(name, (list, tuple)):
                    name = '{0}.{1}'.format(name[0], name[1])
                if name in namespace:
                    ls.append(namespace[name])
                else:
                    raise Exception('Unknown variable {}'.format(name))
            elif isinstance(var, Tensor):
                ls.append(var)
            else:
                raise NotImplementedError('Only Variable is trated')

        self.update()

        return Del(ls)

# TODO: improve by creating the corresponding object in pyccel.ast.core
class PassStmt(BasicStmt):
    """Class representing a Pass statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the Pass statement class.

        label: str
            label must be equal to 'pass'
        """
        self.label = kwargs.pop('label')

        super(PassStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Delete statement by returning a pyccel.ast.core object
        """
        self.update()

        return Pass()

class AssertStmt(BasicStmt):
    """Class representing an Assert statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the Assert statement class.

        test: Test
            represents the condition for the Assert statement.
        """
        self.test = kwargs.pop('test')

        super(AssertStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the If statement by returning a pyccel.ast.core object
        """
        self.update()
        test = self.test.expr

        return Assert(test)


class IfStmt(BasicStmt):
    """Class representing an If statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the If statement class.

        body_true: list
            statements tree as given by the textX, for the true block (if)
        body_false: list
            statements tree as given by the textX, for the false block (else)
        body_elif: list
            statements tree as given by the textX, for the elif blocks
        test: Test
            represents the condition for the If statement.
        """
        self.body_true  = kwargs.pop('body_true')
        self.body_false = kwargs.pop('body_false', None)
        self.body_elif  = kwargs.pop('body_elif',  None)
        self.test       = kwargs.pop('test')

        super(IfStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """Returns the statement variables."""
        ls = []
        for stmt in self.body_true.stmts:
            ls += stmt.local_vars
            ls += stmt.stmt_vars
        if not self.body_false==None:
            for stmt in self.body_false.stmts:
                ls += stmt.local_vars
                ls += stmt.stmt_vars
        if not self.body_elif==None:
            for elif_block in self.body_elif:
                for stmt in elif_block.body.stmts:
                    ls += stmt.local_vars
                    ls += stmt.stmt_vars
        return ls

    @property
    def expr(self):
        """
        Process the If statement by returning a pyccel.ast.core object
        """
        self.update()
        args = [(self.test.expr, self.body_true.expr)]

        if not self.body_elif==None:
            for elif_block in self.body_elif:
                args.append((elif_block.test.expr, elif_block.body.expr))

        if not self.body_false==None:
            args.append((True, self.body_false.expr))

        return If(*args)

class AssignStmt(BasicStmt):
    """Class representing an assign statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the Assign statement.

        lhs: str
            variable to assign to
        rhs: ArithmeticExpression
            expression to assign to the lhs
        trailer: Trailer
            a trailer is used for a function call or Array indexing.
        """
        self.lhs     = kwargs.pop('lhs')
        self.rhs     = kwargs.pop('rhs')
        self.trailer = kwargs.pop('trailer', None)

        super(AssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """Statement variables."""
        # TODO must be improved in the case of Tuples
        return [self.lhs]

    @property
    def expr(self):
        """
        Process the Assign statement by returning a pyccel.ast.core object
        """
        if not isinstance(self.rhs, (ArithmeticExpression, \
                                     ExpressionLambda, \
                                     ExpressionTuple, \
                                     ExpressionList, \
                                     ExpressionDict)):
            raise TypeError("Expecting an expression")

        rhs      = self.rhs.expr
        lhs      = self.lhs.expr
        status   = None
        like     = None

#        print('{0} := {1}'.format(lhs, rhs))
#        print_namespace()
#        print('{0} :: {1}'.format(lhs, type(lhs)))
#        print('{0} :: {1}'.format(rhs, type(rhs)))

        if isinstance(rhs, Function):
            name = str(type(rhs).__name__)
            if name.lower() in builtin_funcs:
                # here rhs.args are sympy Symbols
                args = []
                for a in rhs.args:
                    if str(a) in namespace:
                        args.append(namespace[str(a)])
                    else:
                        # TODO may be we should raise an error here
                        args.append(a)
                # we use str(lhs) to make it work for DottedName
                return builtin_function(name.lower(), args, lhs=str(lhs))

        elif isinstance(rhs, Lambda):
            lhs = Symbol(str(lhs))
            namespace[str(lhs)] = rhs
            return rhs

        # TODO results must be set as stmt_vars,
        #      so they can be deleted by the next block
        elif isinstance(rhs, FunctionCall):
            func = rhs.func
            results = func.results
            if isinstance(results, Tuple) and isinstance(lhs, Tuple):
                if not(len(results) == len(lhs)):
                    raise ValueError('Wrong number of results')

                for res,e in zip(results, lhs):
                    d_var = infere_type(res)
                    insert_variable(str(e), **d_var)

        if isinstance(lhs, str) and not(lhs in namespace):
            d_var = infere_type(rhs)

            if not isinstance(rhs, Tuple):
                # to be allocatable, the shape must not be None or empry list
                if isinstance(d_var['shape'], (list, tuple, Tuple)):
                    if len(d_var['shape']) > 0:
                        d_var['allocatable'] = True

                if d_var['allocatable']:
                    if DEBUG:
                        print(("> Found an unallocated variable: ", lhs))
                    status = 'unallocated'
                    like = allocatable_like(rhs)

            # TODO improve assignable
            assignable = (sp_Integer, sp_Float)
            if isinstance(rhs, assignable):
                d_var['value'] = rhs
            if is_pyccel_datatype(d_var['datatype']):
                d_var['cls_base']= namespace[d_var['datatype'].name]
            insert_variable(lhs, **d_var)
            lhs = namespace[lhs]

        # change lhs from Symbol to Pyccel datatype (Variable, etc)
#        print_namespace()
        if isinstance(lhs, DottedName) and (str(lhs) in namespace):
            d_var = infere_type(rhs)

            if not isinstance(rhs, Tuple):
                if isinstance(namespace[str(lhs)], Symbol):
                    # TODO improve this when handling symbolic computation
                    d_var['allocatable'] = not(d_var['shape'] is None)
                    status = 'unallocated'
                    if d_var['rank'] > 0:
                        like = allocatable_like(rhs)
                    else:
                        like = None

            # TODO improve assignable
            assignable = (sp_Integer, sp_Float)
            if isinstance(rhs, assignable):
                d_var['value'] = rhs
            if is_pyccel_datatype(d_var['datatype']):
                d_var['cls_base']= namespace[d_var['datatype'].name]

            # we remove the sympy Symbol
            namespace.pop(str(lhs))

            insert_variable(lhs, **d_var)
            lhs = namespace[str(lhs)]

#        print('{0} := {1}'.format(lhs, rhs))
#        print('{0} :: {1}'.format(lhs, type(lhs)))
#        print_namespace()
#        import sys; sys.exit(0)

        return Assign(lhs, rhs, strict=False, status=status, like=like)

class AugAssignStmt(BasicStmt):
    """Class representing an assign statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the AugAssign statement.

        lhs: str
            variable to assign to
        rhs: ArithmeticExpression
            expression to assign to the lhs
        trailer: Trailer
            a trailer is used for a function call or Array indexing.
        """
        self.lhs     = kwargs.pop('lhs')
        self.rhs     = kwargs.pop('rhs')
        self.op      = kwargs.pop('op')
        self.trailer = kwargs.pop('trailer', None)

        super(AugAssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """Statement variables."""
        return [self.lhs]

    @property
    def expr(self):
        """
        Process the AugAssign statement by returning a pyccel.ast.core object
        """
        if not isinstance(self.rhs, ArithmeticExpression):
            raise TypeError("Expecting an expression")

        rhs      = self.rhs.expr
        op       = str(self.op[0])
        status   = None
        like     = None

        var_name = self.lhs
        trailer  = None
        args     = None
        if self.trailer:
            trailer = self.trailer.args
            args    = self.trailer.expr
            if isinstance(trailer, TrailerDots):
                var_name = '{0}.{1}'.format(self.lhs, args)

        if isinstance(rhs, Function):
            name = str(type(rhs).__name__)
            if name.lower() in builtin_funcs:
                args = rhs.args
                return builtin_function(name.lower(), args, lhs=var_name, op=op)

        found_var = (var_name in namespace)
        if not(found_var):
            d_var = infere_type(rhs)

#            print ">>>> AugAssignStmt : ", var_name, d_var

            d_var['allocatable'] = not(d_var['shape'] is None)
            if d_var['shape']:
                if DEBUG:
                    print(("> Found an unallocated variable: ", var_name))
                status = 'unallocated'
                like = allocatable_like(rhs)
            insert_variable(var_name, **d_var)

        if self.trailer is None:
            l = namespace[self.lhs]
        else:
            if isinstance(trailer, TrailerSubscriptList):
                v = namespace[str(self.lhs)]
                if not hasattr(args, '__iter__'):
                    args = [args]

                l = IndexedVariable(v.name, dtype=v.dtype).__getitem__(*args)
            elif isinstance(trailer, TrailerDots):
                # class attribut
                l = namespace[var_name]
            else:
                raise TypeError("Expecting SubscriptList or Dot")

        return AugAssign(l, op, rhs, strict=False, status=status, like=like)

class RangeStmt(BasicStmt):
    """Class representing a Range statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the Range statement.

        start: str
            start index
        end: str
            end index
        step: str
            step for the iterable. if not given, 1 will be used.
        """
        self.start    = kwargs.pop('start')
        self.end      = kwargs.pop('end')
        self.step     = kwargs.pop('step', None)

        super(RangeStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Range statement by returning a pyccel.ast.core object
        """
        b = self.start.expr
        e = self.end.expr
        if self.step:
            s = self.step.expr
        else:
            s = 1

        return Range(b,e,s)

class ForStmt(BasicStmt):
    """Class representing a For statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the For statement.

        iterable: str
            the iterable variable
        range: Range
            range for indices
        body: list
            a list of statements for the body of the For statement.
        """
        self.iterable = kwargs.pop('iterable')
        self.range    = kwargs.pop('range')
        self.body     = kwargs.pop('body')

        super(ForStmt, self).__init__(**kwargs)

    @property
    def local_vars(self):
        """Local variables of the For statement."""
        if isinstance(self.iterable, list):
            return self.iterable
        else:
            return [self.iterable]

    @property
    def stmt_vars(self):
        """Statement variables."""
        ls = []
        for stmt in self.body.stmts:
            ls += stmt.local_vars
            ls += stmt.stmt_vars
        return ls

    def update(self):
        """
        Update before processing the statement
        """
        # check that start and end were declared, if they are symbols
        for i in self.local_vars:
            d_var = {}
            d_var['datatype']    = 'int'
            d_var['allocatable'] = False
            d_var['rank']        = 0
            insert_variable(str(i), **d_var)

    @property
    def expr(self):
        """
        Process the For statement by returning a pyccel.ast.core object
        """
        if isinstance(self.iterable, list):
            i = [Symbol(a, integer=True) for a in self.iterable]
        else:
            i = Symbol(self.iterable, integer=True)

        if isinstance(self.range, (RangeStmt, ArithmeticExpression)):
            r = self.range.expr
        elif isinstance(self.range, str):
            if not self.range in namespace:
                raise ValueError('Undefined range.')
            r = namespace[self.range]
        else:
            raise TypeError('Expecting an Iterable')

        if not isinstance(r, (Range, Tensor, Variable, ConstructorCall)):
            raise TypeError('Expecting an Iterable or an object, '
                            'given {0}'.format(type(r)))

        if isinstance(r, Variable):
            if not is_iterable_datatype(r.dtype):
                raise TypeError('Expecting an iterable variable, given '
                                '{0}'.format(r.dtype))

        self.update()

        body = self.body.expr

        if isinstance(r, Variable) and is_iterable_datatype(r.dtype):
            return ForIterator(i, r, body)
        if isinstance(r, ConstructorCall) and is_iterable_datatype(r.this.dtype):
            return ForIterator(i, r, body)
        else:
            return For(i, r, body)

class WithStmt(BasicStmt):
    """Class representing a With statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the With statement.

        domain: WithDomain
            domain of with statement
        body: list
            a list of statements for the body of the With statement.
        """
        self.domain = kwargs.pop('domain')
        self.body   = kwargs.pop('body')

        super(WithStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the With statement by returning a pyccel.ast.core object
        """
        self.update()

        domain = self.domain.expr

        if not (isinstance(domain, (Variable, ConstructorCall))):
            if not(domain in namespace):
                raise ValueError('undefined {0} domain'.format(domain))

        body = self.body.expr
        settings = None

        return With(domain, body, settings)

class WhileStmt(BasicStmt):
    """Class representing a While statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the While statement.

        test: Test
            a test expression
        body: list
            a list of statements for the body of the While statement.
        """
        self.test = kwargs.pop('test')
        self.body = kwargs.pop('body')

        super(WhileStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """Statement variables."""
        ls = []
        for stmt in self.body.stmts:
            ls += stmt.local_vars
            ls += stmt.stmt_vars
        return ls

    @property
    def expr(self):
        """
        Process the While statement by returning a pyccel.ast.core object
        """
        test = self.test.expr

        self.update()

        body = self.body.expr

        return While(test, body)

class ExpressionElement(object):
    """Class representing an element of an expression."""
    def __init__(self, **kwargs):
        """
        Constructor for the ExpessionElement class.

        parent: ArithmeticExpression
            parent ArithmeticExpression
        op:
            attribut in the ArithmeticExpression (see the grammar)
        """
        # textX will pass in parent attribute used for parent-child
        # relationships. We can use it if we want to.
        self.parent = kwargs.get('parent', None)

        # We have 'op' attribute in all grammar rules
        self.op = kwargs['op']

        super(ExpressionElement, self).__init__()

class FactorSigned(ExpressionElement, BasicStmt):
    """Class representing a signed factor."""

    def __init__(self, **kwargs):
        """
        Constructor for a signed factor.

        sign: str
            one among {'+', '-'}
        """
        self.sign    = kwargs.pop('sign', '+')

        super(FactorSigned, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the signed factor, by returning a sympy expression
        """
        if DEBUG:
            print("> FactorSigned ")
        expr = self.op.expr
        return -expr if self.sign == '-' else expr

class AtomExpr(ExpressionElement, BasicStmt):
    """Class representing an atomic expression."""

    def __init__(self, **kwargs):
        """
        Constructor for a atomic expression.

        trailer: Trailer
            a trailer is used for a function call or Array indexing.
        """
        self.trailers = kwargs.pop('trailers', None)

        super(AtomExpr, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the atomic expression, by returning a sympy expression
        """
        if DEBUG:
            print("> AtomExpr ")

        e = self.op.expr
        trailer = self.trailers

        # if trailer is None or empty list
        if not trailer:
            return e

        # now we loop over all trailers
        # this must be done from left to right
        for i in trailer:
            e = expr_with_trailer(e, i.args)
        return e

class Power(ExpressionElement, BasicStmt):
    """Class representing an atomic expression."""

    def __init__(self, **kwargs):
        """
        Constructor for a atomic expression.

        exponent: str
            a exponent.
        """
        self.exponent = kwargs.pop('exponent', None)

        super(Power, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the atomic expression, by returning a sympy expression
        """
        if DEBUG:
            print("> Power ")
        expr = self.op.expr
        if self.exponent is None:
            return expr
        else:
            exponent = self.exponent.expr
            return Pow(expr, exponent)

class Term(ExpressionElement):
    """Class representing a term in the grammar."""

    @property
    def expr(self):
        """
        Process the term, by returning a sympy expression
        """
        if DEBUG:
            print("> Term ")

        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            if operation == '*':
                ret = Mul(ret, operand.expr)
            else:
                a   = Pow(operand.expr, -1)
                ret = Mul(ret, a)
        return ret

class ArithmeticExpression(ExpressionElement):
    """Class representing an expression in the grammar."""

    @property
    def expr(self):
        """
        Process the expression, by returning a sympy expression
        """
        if DEBUG:
            print("> ArithmeticExpression ")

        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):

            if operation == '+':
                ret = Add(ret, operand.expr)
            else:
                a   = Mul(-1, operand.expr)
                ret = Add(ret, a)

        return ret

class Atom(ExpressionElement):
    """Class representing an atom in the grammar."""

    @property
    def expr(self):
        """
        Process the atom, by returning a sympy atom
        """
        if DEBUG:
            print("> Atom ")

        op = self.op
#        print('> {0} of type {1}'.format(op, type(op)))
#        if op in ['shape']:
#            raise ValueError('shape function can not be used in an expression.')

        if type(op) == int:
            return Integer(op)
        elif is_Float(op):
            # op is here a string that can be converted to a number
            # TODO use Default precision
            return Float(float(op))
        elif type(op) == list:
            # op is a list
            for O in op:
                if O in namespace:
                    return namespace[O]
                elif type(O) == int:
                    return Integer(O)
                elif type(O) == float:
                    # TODO use Default precision
                    return Float(O)
                else:
                    raise Exception('Unknown variable "{}" at position {}'
                                    .format(O, self._tx_position))
        elif isinstance(op, ExpressionElement):
            return op.expr
        elif isinstance(op, ExpressionList):
            e = op.expr
            if len(e) == 1:
                return e[0]
            else:
                return e
        elif isinstance(op, ExpressionTuple):
            e = op.expr
            if len(e) == 1:
                return e[0]
            else:
                return e
        elif op in namespace:
            # function arguments are not known yet.
            # they will be handled in expr_with_trailer
            return namespace[op]
        elif op in builtin_funcs:
            return Function(op)
        elif op in builtin_types:
            return datatype(op)
        elif op == 'None':
            return Nil()
        elif op == 'True':
            return true
        elif op == 'False':
            return false
        elif isinstance(op, str):
            return op
        else:
            txt = 'Undefined variable "{0}" of type {1}'.format(op, type(op))
            raise Exception(txt)

class Test(ExpressionElement):
    """Class representing a test expression as described in the grammmar."""

    @property
    def expr(self):
        """
        Process the test expression, by returning a sympy expression
        """
        if DEBUG:
            print("> DEBUG ")
        ret = self.op.expr
        return ret

# TODO improve using sympy And, Or, Not, ...
class OrTest(ExpressionElement):
    """Class representing an Or term expression as described in the grammmar."""

    @property
    def expr(self):
        """
        Process the Or term, by returning a sympy expression
        """
        if DEBUG:
            print("> DEBUG ")

        ret = self.op[0].expr
        for operand in self.op[1:]:
            ret = Or(ret,operand.expr)

        return ret

# TODO improve using sympy And, Or, Not, ...
class AndTest(ExpressionElement):
    """Class representing an And term expression as described in the grammmar."""

    @property
    def expr(self):
        """
        Process the And term, by returning a sympy expression
        """
        if DEBUG:
            print("> DEBUG ")

        ret = self.op[0].expr


        for operand in self.op[1:]:
            ret = And(ret,operand.expr)
        return ret

# TODO improve using sympy And, Or, Not, ...
class NotTest(ExpressionElement):
    """Class representing an Not term expression as described in the grammmar."""

    @property
    def expr(self):
        """
        Process the Not term, by returning a sympy expression
        """
        if DEBUG:
            print("> DEBUG ")

        ret = self.op.expr
        ret = (not ret)
        return ret

class Comparison(ExpressionElement):
    """Class representing the comparison expression as described in the grammmar."""

    @property
    def expr(self):
        """
        Process the comparison, by returning a sympy expression
        """
        if DEBUG:
            print("> Comparison ")

        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            if operation == "==":
                ret = Eq(ret, operand.expr)
            elif operation == ">":
                ret = Gt(ret, operand.expr)
            elif operation == ">=":
                ret = Ge(ret, operand.expr)
            elif operation == "<":
                ret = Lt(ret, operand.expr)
            elif operation == "<=":
                ret = Le(ret, operand.expr)
            elif (operation == "<>") or (operation == "!="):
                ret = Ne(ret, operand.expr)
            else:
                raise NotImplementedError('operation {0} not yet available'.format(operation))
        return ret

class ExpressionLambda(BasicStmt):
    """Base class representing a lambda expression in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Expression list statement

        args: list, tuple
            list of elements
        """
        self.args = kwargs.pop('args')
        self.rhs  = kwargs.pop('rhs')

        super(ExpressionLambda, self).__init__(**kwargs)

    @property
    def expr(self):

        args = sp_symbols(self.args)

        # ... we update the namespace
        ls = args
        if isinstance(args, Symbol):
            ls = [args]

        for i in ls:
            namespace[str(i)] = i
        # ...

        # ... we do it here after the namespace has been updated
        e = self.rhs.expr
        # ...

        # ... we clean the namespace
        for i in ls:
            namespace.pop(str(i))
        # ...

        return Lambda(args, e)

class ExpressionTuple(BasicStmt):
    """Base class representing a list of elements statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Expression list statement

        args: list, tuple
            list of elements
        """
        self.args = kwargs.pop('args')

        super(ExpressionTuple, self).__init__(**kwargs)

    @property
    def expr(self):
        args = [a.expr for a in self.args]
        return Tuple(*args)

class ExpressionList(BasicStmt):
    """Base class representing a list of elements statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Expression list statement

        args: list, tuple
            list of elements
        """
        self.args = kwargs.pop('args')

        super(ExpressionList, self).__init__(**kwargs)

    @property
    def expr(self):
        args = [a.expr for a in self.args]
        # TODO use List object from AST
        #return List(*args)
        return Tuple(*args)

class ExpressionDict(BasicStmt):
    """Base class representing a dictionary of elements statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Expression dictionary statement

        args: list, tuple
            list of elements
        """
        self.args = kwargs.pop('args')

        super(ExpressionDict, self).__init__(**kwargs)

    @property
    def expr(self):
        raise NotImplementedError('No fortran backend yet for dictionaries.')
        args = {}
        for a in self.args:
            key   = a.key # to treat
            value = a.value
            args[key] = value
        return Dict(**args)

class ArgValued(BasicStmt):
    """Base class representing a list element with key in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a list element with key

        key: str, None
            entry key
        value: Expression
            entry value
        """
        self.key   = kwargs.pop('key', None)
        self.value = kwargs.pop('value')

        super(ArgValued, self).__init__(**kwargs)

    @property
    def expr(self):
        key   = self.key
        value = self.value.expr
        if key:
            return {'key': key, 'value': value}
        else:
            return value


class FlowStmt(BasicStmt):
    """Base class representing a Flow statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Flow statement

        label: str
            name of the flow statement.
            One among {'break', 'continue', 'return', 'raise', 'yield'}
        """
        self.label = kwargs.pop('label')

class BreakStmt(FlowStmt):
    """Base class representing a Break statement in the grammar."""
    def __init__(self, **kwargs):
        super(BreakStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        return Break()

class ContinueStmt(FlowStmt):
    """Base class representing a Continue statement in the grammar."""
    def __init__(self, **kwargs):
        super(ContinueStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        return Continue()

# TODO improve
class ReturnStmt(FlowStmt):
    """Base class representing a Return statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a return statement flow.

        variables: list
            list of variables to return, as Expression
        results: list
            list of variables to return, as pyccel.ast.core objects
        """
        self.variables = kwargs.pop('variables')

        super(ReturnStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the return flow statement
        """
        return [e.expr for e in self.variables]

class RaiseStmt(FlowStmt):
    """Base class representing a Raise statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a return statement flow.

        exception:
            exception to raise
        """
        self.exception = kwargs.pop('exception')

        super(RaiseStmt, self).__init__(**kwargs)

    # TODO finish return Raise
    @property
    def expr(self):
        exception = self.exception.expr
        return Raise()

class YieldStmt(FlowStmt):
    """Base class representing a Yield statement in the grammar."""
    pass

class FunctionDefStmt(BasicStmt):
    """Class representing the definition of a function in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for the definition of a function.

        name: str
            name of the function
        args: list
            list of the function arguments
        body: list
            list of statements as given by the parser.
        parent: stmt
            parent statement.
        """
        self.name    = kwargs.pop('name')
        self.trailer = kwargs.pop('trailer')
        self.body    = kwargs.pop('body')
        self.parent  = kwargs.get('parent', None)

        super(FunctionDefStmt, self).__init__(**kwargs)

    @property
    def local_vars(self):
        """returns the local variables of the body."""
        return self.body.local_vars

    @property
    def stmt_vars(self):
        """returns the statement variables of the body."""
        return self.body.stmt_vars

    # TODO scope
    @property
    def expr(self):

        """
        Process the Function Definition by returning the appropriate object from
        pyccel.ast.core
        """
        name = str(self.name)
        args = self.trailer.expr

#        print (">>>>>>>>>>> FunctionDefStmt {0}: Begin".format(name))
#        print_namespace()
#        print_declarations()

        local_vars  = []
        global_vars = []

        cls_instance = None
        this = None
        if isinstance(self.parent, SuiteStmt):
            if isinstance(self.parent.parent, ClassDefStmt):
                cls_instance = self.parent.parent.name

        if cls_instance:
            name = '{0}.{1}'.format(cls_instance, name)

            # insert self to namespace
            d_var = {}
            dtype = get_class_construct(cls_instance)()
            d_var['datatype']    = dtype
            d_var['allocatable'] = False
            d_var['shape']       = None
            d_var['rank']        = 0
            d_var['intent']      = 'inout'
            d_var['cls_base']    = cls_instance
            insert_variable('self', **d_var)

            # remove self from args
            if args[0] == 'self':
                args = args[1:]
                this = namespace['self']

        # ...
        results = []
        for stmt in self.body.stmts:
            if isinstance(stmt, ReturnStmt):
                results += stmt.expr
        # ...

        # ...
        h          = None
        arg_names  = []

        scope_vars = {}
        scope_decs = {}
        # ...

        # ...
        with_header = False
        if (len(results) > 0) or (len(args) > 0):
            if not(name in headers):
                raise Exception('Function header could not be found for {0}.'
                               .format(name))

            if not(len(args) == len(headers[name].dtypes)):
                raise Exception("Wrong number of arguments in the header.")

            with_header = True

        # ...............................
        #         Treating inputs
        # ...............................
        if with_header and (len(args) > 0):
            h = headers[name]

            # old occurence of args will be stored in scope
            for a, d in zip(args, h.dtypes):
                # case of arg with key
                if isinstance(a, dict):
                    arg_name = a['key']
                else:
                    arg_name = str(a)
                arg_names.append(arg_name)

                if arg_name in namespace:
                    var = namespace.pop(arg_name)
                    dec = declarations.pop(arg_name)

                    scope_vars[arg_name] = var
                    scope_decs[arg_name] = dec

                rank = 0
                for i in d[1]:
                    if isinstance(i, Slice):
                        rank += 1
                d_var = {}
                if d[0]==cls_instance:
                    d_var['datatype'] = get_class_construct(cls_instance)()
                    d_var['cls_base'] = cls_instance
                else:
                    d_var['datatype']    = d[0]
                d_var['allocatable'] = False
    #            d_var['allocatable'] = d[2]
                d_var['shape']       = None
                d_var['rank']        = rank
                d_var['intent']      = 'in'
                insert_variable(arg_name, **d_var)
            args = [namespace[a] for a in arg_names]
        # ...............................

        # ... define functiondef kind
        kind = None
        if with_header:
            h = headers[name]
            kind = h.kind
        # ...

        # ... case of class constructor
        if self.name == '__init__':
            # first we construct the list of attributs
            attr = []
            for i in self.body.stmts:
                if isinstance(i, AssignStmt) and i.lhs == 'self':

                    c = {'lhs':i.trailer.expr, 'rhs':i.rhs}
                    Var = AssignStmt(**c).expr
                    attr += [Var.lhs]
            # we first create and append an empty class to the namespace
            namespace[cls_instance] = ClassDef(cls_instance,attr,[],[])
        # ...

        body = self.body.expr
        if this:
            arg_names += ['self']

        prelude = [declarations[a] for a in arg_names]

        # ... TODO improve
        for a in results:
            if a in namespace:
                var = namespace.pop(a)
                dec = declarations.pop(a)

        # TODO: for the moment we do not infer the results type
        if with_header and len(results) > 0:
            if not(len(h.results) == len(results)):
                raise ValueError('Incompatible results with header.')

            _results = []
            result_names = []
            for a, d in zip(results, h.results):
                result_name = a
                result_names.append(result_name)

                rank = 0
                for i in d[1]:
                    if isinstance(i, Slice):
                        rank += 1
                d_var = {}
                d_var['datatype']    = d[0]
    #            d_var['allocatable'] = False
                d_var['allocatable'] = d[2]
                d_var['shape']       = None
                d_var['rank']        = rank
                d_var['intent']      = 'out'
                insert_variable(result_name, **d_var)
                var = namespace[result_name]
                _results.append(var)
            results = _results
        # ...

        # ... replace dict by ValuedVariable
        _args = []
        for a in args:
            if isinstance(a, dict):
                var = namespace[a['key']]
                # TODO trea a['value'] correctly
                _args.append(ValuedVariable(var, a['value']))
            else:
                _args.append(a)
        args = _args
        # ...

        # ...
        if this:
            args = [this] + args
        # ...

        # ...
        body = prelude + body
        # ...

        # ... define local_vars as any lhs in Assign, if it is not global
        #     or class member 'self.member'
        # TODO: to improve
        for stmt in body:
            if isinstance(stmt, (Assign, Zeros, ZerosLike, Ones)):
                if (isinstance(stmt.lhs, Variable) and
                    not(stmt.lhs in results) and
                    not(stmt.lhs in global_vars)):
                    if isinstance(stmt.lhs.name, DottedName):
                        lhs  = stmt.lhs.name
                        this = str(lhs.name[0])
                        if not(this == 'self'):
                            local_vars += [stmt.lhs]
                    else:
                        local_vars += [stmt.lhs]
        # ...

        # ...
        local_vars += [i.expr for i in self.stmt_vars if isinstance(i, Variable)]
        local_vars = list(set(local_vars))
        # ...

        # ... remove results from local_vars
        #     and from declarations
        res_names = [str(i) for i in results]
        arg_names = [str(i) for i in args]
        local_vars = [i for i in local_vars if not(str(i) in res_names + arg_names)]
        for i in res_names + arg_names:
            if i in declarations:
                del declarations[i]
        # ...

        # rename the method in the class case
        # TODO do we keep this?
        f_name = name
        cls_name = None
        if cls_instance:
            f_name   = name.split('.')[-1]
            cls_name = name.split('.')[0]
        stmt = FunctionDef(f_name, args, results, body,
                           local_vars, global_vars,
                           cls_name=cls_name, kind=kind)
        namespace[name] = stmt

        # ...
        # we keep 'self.*' in the stack
        ls = [str(i) for i in local_vars+results if not str(i).startswith('self.')]

        for var_name in ls:
            if var_name in namespace:
                del namespace[var_name]

            if var_name in declarations:
                del declarations[var_name]
        # ...

        # ... cleaning the namespace
        for a in arg_names:
            if a in declarations:
                del declarations[a]

            if a in namespace:
                del namespace[a]

        # ...
        for arg_name, var in list(scope_vars.items()):
            var = scope_vars.pop(arg_name)
            namespace[arg_name] = var

        for arg_name, dec in list(scope_decs.items()):
            dec = scope_decs.pop(arg_name)
            declarations[arg_name] = dec
        # ...

        # ... cleaning
        #for k in local_vars:
        #    if k.name in namespace.keys():
        #        namespace.pop(k.name)
        #    if k.name in declarations.keys():
        #        declarations.pop(k.name)
        # ...
#        print_namespace()
#        print_declarations()
#
#        print "<<<<<<<<<<< FunctionDefStmt : End"
        return stmt

class ClassDefStmt(BasicStmt):
    """Class representing the definition of a class in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for the definition of a class.
        We only allow for single inheritence, to match with Fortran specs.

        name: str
            name of the class
        base: list
            base class
        body: list
            list of statements as given by the parser.
        """
        self.name = kwargs.pop('name')
        self.base = kwargs.pop('base')
        self.body = kwargs.pop('body')

        super(ClassDefStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Class Definition by returning the appropriate object from
        pyccel.ast.core
        """
        name = str(self.name)

#        print (">>>>>>>>>>> ClassDefStmt {0}: Begin".format(name))
#        print('===== BEFORE =====')
#        print_namespace()

        if not(name in headers):
            raise Exception('Class header could not be found for {0}.'
                           .format(name))

        header  = headers[name]
        options = header.options

        body    = self.body.expr

        # ... we first process the __init__ method to find class attributs
        init_method = [i for i in body if isinstance(i, FunctionDef) and
                       str(i.name) == '__init__']

        if not init_method:
            raise ValueError('Class missing __init__ method.')
        if len(init_method) > 1:
            raise ValueError('Found multiple definitions of __init__.')

        init_method = init_method[0]
        # ...

        # ... we then process __init__ method to find all attributs.
        #     the idea is that they will be defined by an Assign statement.
        attributs = []
        for stmt in init_method.body:
            if isinstance(stmt, (Assign, Zeros, Ones, ZerosLike)):
#                print(stmt.lhs, type(stmt.lhs))
                if (isinstance(stmt.lhs, (Variable, IndexedVariable)) and
                    isinstance(stmt.lhs.name, DottedName)):

                    lhs = stmt.lhs.name

                    this = str(lhs.name[0])
                    if this == 'self':
                        if len(lhs.name) > 2:
                            raise ValueError('Only one level access is available.')

                        attr_name = str(lhs.name[1])
                        if not(str(lhs) in namespace):
                            raise ValueError('Namespace must contain '
                                             '{0}'.format(lhs))

                        # then we clone 'self.member' to 'member'
                        attr = namespace[str(lhs)]
                        # TODO must check if attr can be cloned
                        attr = attr.clone(attr_name)

                        attributs.append(attr)
                        # we do not forget to remove lhs from namespace
                        namespace.pop(str(lhs))
                        if str(lhs) in declarations:
                            declarations.pop(str(lhs))
        attributs = list(set(attributs))
        # ...

        # ...
        methods = []
        for stmt in body:
            if isinstance(stmt, FunctionDef):
                methods.append(stmt)

        stmt = ClassDef(name, attributs, methods, options)
        namespace[name] = stmt
        # ...

        # ... local variables
        local_vars = []
        for m in methods:
            local_vars += m.local_vars
        local_vars = list(set(local_vars))
        # ...

        # ... cleaning
        for k in attributs + local_vars:
            if k.name in namespace.keys():
                namespace.pop(k.name)
            if k.name in declarations.keys():
                declarations.pop(k.name)

        for m in methods:
            if not(str(m.name) == '__init__'):
                namespace.pop('{0}.{1}'.format(name, str(m.name)))
        # ...

#        print('===== AFTER =====')
#        print_namespace()
#        print_declarations()

#        print "<<<<<<<<<<< ClassDefStmt : End"

        return stmt

class CommentStmt(BasicStmt):
    """Class representing a Comment in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Comment.

        text: str
            text that appears in the comment
        """
        self.text = kwargs.pop('text')

        # TODO improve
        #      to remove:  # coding: utf-8
        if ("coding:" in self.text) or ("utf-8" in self.text):
            self.text = ""

        super(CommentStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Comment statement,
        by returning the appropriate object from pyccel.ast.core
        """
        self.update()
        if self.text:
            return Comment(self.text)
        else:
            return EmptyLine()

class DocstringsCommentStmt(CommentStmt):
    """Class representing a Docstrings Comment in the grammar."""
    pass

class SuiteStmt(BasicStmt):
    """Class representing a Suite statement in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor for a Suite statement.

        stmts: list
            list of statements as given by the parser.
        parent: stmt
            parent statement.
        """
        self.stmts  = kwargs.pop('stmts')
        self.parent = kwargs.get('parent', None)

        super(SuiteStmt, self).__init__(**kwargs)

    @property
    def local_vars(self):
        """returns local variables for every statement in stmts."""
        ls = []
        for stmt in self.stmts:
            ls += stmt.local_vars
        s = set(ls)
        return list(s)

    @property
    def stmt_vars(self):
        """returns statement variables for every statement in stmts."""
        ls = []
        for stmt in self.stmts:
            ls += stmt.stmt_vars
        s = set(ls)
        return list(s)

    @property
    def expr(self):
        """
        Process the Suite statement,
        by returning a list of appropriate objects from pyccel.ast.core
        """
#        print "local_vars = ", self.local_vars
#        print "stmt_vars  = ", self.stmt_vars
        self.update()
        ls = [stmt.expr for stmt in  self.stmts]
        return ls

class BasicTrailer(BasicStmt):
    """Base class representing a Trailer in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor for a Base Trailer.

        args: list or ArgList
            arguments of the trailer
        """
        self.args = kwargs.pop('args', None)

        super(BasicTrailer, self).__init__(**kwargs)

class Trailer(BasicTrailer):
    """Class representing a Trailer in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor for a Trailer.
        """
        super(Trailer, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process a Trailer by returning the approriate objects from
        pyccel.ast.core
        """
        self.update()
        return self.args.expr

class TrailerArgList(BasicTrailer):
    """Class representing a Trailer with list of arguments in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor of the Trailer ArgList
        """
        super(TrailerArgList, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process a Trailer by returning the approriate objects from
        pyccel.ast.core
        """
        # ...
        def _do_arg(arg):
            if isinstance(arg, TrailerArgList):
                args = [_do_arg(i) for i in arg.args]
                return Tuple(*args)
            elif isinstance(arg, ArgValued):
                return arg.expr
            raise TypeError('Expecting ArgValued or TrailerArgList')
        # ...

        args = [_do_arg(i) for i in self.args]
        return args

class TrailerSubscriptList(BasicTrailer):
    """Class representing a Trailer with list of subscripts in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor of the Trailer with subscripts
        """
        super(TrailerSubscriptList, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process a Trailer by returning the approriate objects from
        pyccel.ast.core
        """
        self.update()
        args = []
        for a in self.args:
            if isinstance(a, ArithmeticExpression):
                arg = do_arg(a)

                # TODO treat n correctly
                n = Symbol('n', integer=True)
                i = Idx(arg, n)
                args.append(i)
            elif isinstance(a, BasicSlice):
                arg = a.expr
                args.append(arg)
            else:
                raise Exception('Wrong instance')
        return args

class TrailerDots(BasicTrailer):
    """Class representing a Trailer with dots in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor of the Trailer
        """
        super(TrailerDots, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process a Trailer by returning the approriate objects from
        pyccel.ast.core
        """
        self.update()
        # args is not a list
        return self.args
#        return [arg.expr for arg in  self.args]

class BasicSlice(BasicStmt):
    """Base class representing a Slice in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor for the base slice.
        The general form of slices is 'a:b'

        start: str, int, ArithmeticExpression
            Starting index of the slice.
        end: str, int, ArithmeticExpression
            Ending index of the slice.
        """
        self.start = kwargs.pop('start', None)
        self.end   = kwargs.pop('end',   None)

        super(BasicSlice, self).__init__(**kwargs)

    def extract_arg(self, name):
        """
        returns an argument as a variable, given its name

        name: str
            variable name
        """
        if name is None:
            return None

        var = None
        if isinstance(name, (Integer, Float)):
            var = Integer(name)
        elif isinstance(name, str):
            if name in namespace:
                var = namespace[name]
            else:
                raise Exception("could not find {} in namespace ".format(name))
        elif isinstance(name, ArithmeticExpression):
            var = do_arg(name)
        else:
            raise Exception("Unexpected type {0} for {1}".format(type(name), name))

        return var

    @property
    def expr(self):
        """
        Process the Slice statement, by giving its appropriate object from
        pyccel.ast.core
        """
        start = self.extract_arg(self.start)
        end   = self.extract_arg(self.end)

        return Slice(start, end)

class TrailerSlice(BasicSlice):
    """
    Class representing a Slice in the grammar.
    A Slice is of the form 'a:b'
    """
    pass

class TrailerSliceRight(BasicSlice):
    """
    Class representing a right Slice in the grammar.
    A right Slice is of the form 'a:'
    """
    pass

class TrailerSliceLeft(BasicSlice):
    """
    Class representing a left Slice in the grammar.
    A left Slice is of the form ':b'
    """
    pass

class TrailerSliceEmpty(BasicSlice):
    """
    Class representing an empty Slice in the grammar.
    An empty Slice is of the form ':'
    """
    def __init__(self, **kwargs):
        """
        """
        self.dots  = kwargs.pop('dots')
        super(TrailerSliceEmpty, self).__init__(**kwargs)

class ArgList(BasicStmt):
    """Class representing a list of arguments."""
    def __init__(self, **kwargs):
        """
        Constructor for ArgList statement.

        args: list
            list of arguments
        """
        self.args = kwargs.pop('args', None)

        super(ArgList, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the ArgList statement,
        by returning a list of appropriate objects from pyccel.ast.core
        """
        ls = []
        for arg in self.args:
            if isinstance(arg, ArgList):
                ls.append(arg.expr)
            elif type(arg) == int:
                ls.append(int(arg))
            elif is_Float(arg):
                ls.append(float(arg))
            else:
                if arg in namespace:
                    ls.append(namespace[arg])
                else:
                    ls.append(arg)
        return ls

class VariableHeaderStmt(BasicStmt):
    """Base class representing a header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a VariableHeader statement.
        In the case of builtin datatypes, we export a Variable

        name: str
            variable name
        dec: list, tuple
            list of argument types
        """
        self.name = kwargs.pop('name')
        self.dec  = kwargs.pop('dec')

        super(VariableHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtype = self.dec.dtype
        star  = (self.dec.star == '*')
        attr = ''
        if not(self.dec.trailer is None):
            attr = self.dec.trailer.expr

        h = VariableHeader(self.name, (dtype, attr, star))
        headers[self.name] = h

        # for builtin types, we will return a variable instead of its header
        if not(dtype in builtin_types):
            return h

        # ... computing attributs for Variable
        rank = 0
        for i in attr:
            if isinstance(i, Slice):
                rank += 1
        d_var = {}
        d_var['datatype']    = dtype
        d_var['allocatable'] = star
        d_var['shape']       = None
        d_var['rank']        = rank
        insert_variable(self.name, **d_var)
        # ...

        # we don't forget to remove the variable from headers
        var = namespace[self.name]
        headers.pop(self.name)

        return var

class FunctionHeaderStmt(BasicStmt):
    """Base class representing a function header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a FunctionHeader statement

        name: str
            function name
        kind: str
            function or procedure
        decs: list, tuple
            list of argument types
        results: list, tuple
            list of output types
        """
        self.name    = kwargs.pop('name')
        self.kind    = kwargs.pop('kind', 'function')
        self.decs    = kwargs.pop('decs')
        self.results = kwargs.pop('results', None)

        super(FunctionHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtypes    = [dec.dtype for dec in self.decs]
        stars     = [(dec.star == '*') for dec in self.decs]
        attributs = []
        for dec in self.decs:
            if dec.trailer is None:
                attr = ''
            else:
                attr = dec.trailer.expr
            attributs.append(attr)

        self.dtypes = list(zip(dtypes, attributs, stars))

        if not (self.results is None):
            r_dtypes    = [dec.dtype for dec in self.results.decs]
            r_stars     = [(dec.star == '*' ) for dec in self.results.decs]
            attributs = []
            for dec in self.results.decs:
                if dec.trailer is None:
                    attr = ''
                else:
                    attr = dec.trailer.expr
                attributs.append(attr)
            self.results = list(zip(r_dtypes, attributs, r_stars))

        if self.kind is None:
            kind = 'function'
        else:
            kind = str(self.kind)

        h = FunctionHeader(self.name, self.dtypes, \
                           results=self.results, kind=kind)
        headers[self.name] = h
        return h

class ClassHeaderStmt(BasicStmt):
    """Base class representing a class header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Header statement

        name: str
            class name
        options: list, tuple
            list of class options
        """
        self.name    = kwargs.pop('name')
        self.options = kwargs.pop('options')

        super(ClassHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        options = [str(i) for i in self.options]

        iterable       = ('iterable' in options)
        with_construct = ('with' in options)

        # create a new Datatype for the current class
        dtype = DataTypeFactory(self.name, ("_name"),
                                is_iterable=iterable,
                                is_with_construct=with_construct)
        set_class_construct(self.name, dtype)

        h = ClassHeader(self.name, self.options)
        headers[self.name] = h

        return h

class MethodHeaderStmt(BasicStmt):
    """Base class representing a function header statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a MethodHeader statement

        name: str
            function name
        decs: list, tuple
            list of input types
        results: list, tuple
            list of output types
        """
        self.name    = kwargs.pop('name')
        self.decs    = kwargs.pop('decs')
        self.results = kwargs.pop('results', None)

        super(MethodHeaderStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        dtypes    = [dec.dtype for dec in self.decs]
        attributs = []
        for dec in self.decs:
            if dec.trailer is None:
                attr = ''
            else:
                attr = dec.trailer.expr
            attributs.append(attr)
        self.dtypes = list(zip(dtypes, attributs))

        if not (self.results is None):
            r_dtypes    = [dec.dtype for dec in self.results.decs]
            attributs = []
            for dec in self.results.decs:
                if dec.trailer is None:
                    attr = ''
                else:
                    attr = dec.trailer.expr
                attributs.append(attr)
            self.results = list(zip(r_dtypes, attributs))

        cls_instance = self.dtypes[0]
        cls_instance = cls_instance[0] # remove the attribut
        dtypes = self.dtypes[1:]
        h = MethodHeader((cls_instance, self.name), dtypes, self.results)
        headers[h.name] = h
#        print('\n')
#        for k,v in headers.items():
#            print('"{0}": {1}'.format(k, v))

        return h

class ImportFromStmt(BasicStmt):
    """Class representing an Import statement in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor for an Import statement.

        dotted_name: list
            modules path
        import_as_names: textX object
            everything that can be imported
        """
        self.dotted_name     = kwargs.pop('dotted_name')
        self.import_as_names = kwargs.pop('import_as_names')

        super(ImportFromStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Import statement,
        by returning the appropriate object from pyccel.ast.core
        """
        self.update()

        # TODO how to handle dotted packages?
        names = self.dotted_name.names
        if isinstance(names, (list, tuple)):
            if len(names) == 1:
                fil = str(names[0])
            else:
                names = [str(n) for n in names]
                fil = DottedName(*names)
        elif isinstance(names, str):
            fil = str(names)

        funcs = self.import_as_names
        if isinstance(funcs, ImportAsNames):
            funcs = funcs.names
        if not isinstance(funcs, (tuple, list)):
            funcs = str(funcs) # cast unicode to str

        return Import(fil, funcs)

class ImportAsNames(BasicStmt):
    """class representing import as names in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor

        names: str
            list of names
        """
        self.names = kwargs.pop('names')

        super(ImportAsNames, self).__init__(**kwargs)

class CallStmt(BasicStmt):
    """Class representing the call to a function in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for the call to a function.

        name: str
            name of the function
        args: list
            list of the function arguments
        """
        self.name    = kwargs.pop('name')
        self.trailer = kwargs.pop('trailer')

        super(CallStmt, self).__init__(**kwargs)

    # TODO scope
    @property
    def expr(self):

        """
        Process the Function Definition by returning the appropriate object from
        pyccel.ast.core
        """
        f_name = str(self.name)
        if len(self.trailer)==1:
            args=self.trailer[0].expr
        else:
            args=[]
            for i in self.trailer:
                a=i.expr
                if not isinstance(a,str):
                    args+=a
                else:
                    args+=[a]

        # ... replace dict by ValuedVariable
        _args = []
        for a in args:
            if isinstance(a, dict):
                var = namespace[a['key']]
                # TODO trea a['value'] correctly
                _args.append(ValuedVariable(var, a['value']))
            else:
                _args.append(a)
        args = _args
        # ...

        # ...
        if not(f_name in namespace) and not(f_name in builtin_funcs_math_nores):
            raise ValueError("Undefined function call {}.".format(f_name))
        # ...

        # ...

        if f_name in namespace:
            F = namespace[f_name]

            if isinstance(F,Variable) and F.cls_base:

                methods=namespace[F.dtype.name].methods
                for method in methods:
                    if str(method.name)==args[0]:
                        return MethodCall(method,args[1:],cls_variable=F,kind=None)


            if not(isinstance(F, FunctionDef)):
                raise TypeError("Expecting a FunctionDef")

            if len(F.results) > 0:
                raise ValueError("Expecting no results")

            return FunctionCall(F, args, kind=None)
        # ...

        # ... now f_name should be a builtin function without results
        if f_name == 'print':
            expressions=[]

            for arg in args:
               expressions.append(arg)
            return Print(expressions)
        # ...
