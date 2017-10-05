# coding: utf-8
import numpy as np
from numpy import ndarray
from numpy import asarray

from sympy.core.expr import Expr
from sympy.core.containers import Tuple
from sympy import Symbol, Integer, Float, Add, Mul
from sympy import true, false, pi
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.core.basic import Basic
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.power import Pow
from sympy.core.function import Function
from sympy import preorder_traversal
from sympy import (Abs, sqrt, sin,  cos,  exp,  log, \
                   csc,  cos,  sec,  tan,  cot,  asin, \
                   acsc, acos, asec, atan, acot, atan2)


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

from pyccel.types.ast import allocatable_like
from pyccel.types.ast import FunctionCall
from pyccel.types.ast import DottedVariable
from pyccel.types.ast import DataType, DataTypeFactory
from pyccel.types.ast import NativeBool, NativeFloat, NativeComplex, NativeDouble, NativeInteger
from pyccel.types.ast import (For, Assign, Declare, Variable, \
                              FunctionHeader, ClassHeader, MethodHeader, \
                              datatype, While, NativeFloat, \
                              EqualityStmt, NotequalStmt, \
                              MultiAssign, \
                              FunctionDef, ClassDef, Del, Print, \
                              Comment, AnnotatedComment, \
                              IndexedVariable, Slice, If, \
                              ThreadID, ThreadsNumber, \
                              Stencil, Ceil, Break, \
                              Zeros, Ones, Array, ZerosLike, Shape, Len, \
                              Dot, Sign, IndexedElement,\
                              Min, Max)

from pyccel.core.syntax     import BasicStmt
from pyccel.imports.syntax import ImportFromStmt
from pyccel.openmp.syntax   import OpenmpStmt

from pyccel.parallel.mpi import MPI
from pyccel.parallel.mpi import MPI_ERROR, MPI_STATUS
from pyccel.parallel.mpi import MPI_Assign, MPI_Declare
from pyccel.parallel.mpi import MPI_waitall
from pyccel.parallel.mpi import MPI_INTEGER, MPI_FLOAT, MPI_DOUBLE
from pyccel.parallel.mpi import MPI_comm_world, MPI_COMM_WORLD
from pyccel.parallel.mpi import MPI_status_size, MPI_STATUS_SIZE
from pyccel.parallel.mpi import MPI_proc_null, MPI_PROC_NULL
from pyccel.parallel.mpi import MPI_comm_size, MPI_comm_rank
from pyccel.parallel.mpi import MPI_comm_recv, MPI_comm_send
from pyccel.parallel.mpi import MPI_comm_irecv, MPI_comm_isend
from pyccel.parallel.mpi import MPI_comm_sendrecv

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
    "sec": "sec",
    "sign": "Sign",
    "sin": "sin",
    "sinh": "sinh",
    "sqrt": "sqrt",
    "tan": "tan",
    "tanh": "tanh"
}

# TODO: treat the inout case
# TODO: 1. check that every stmt is well implementing
#          the local_vars and stmt_vars properties.

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
    elif txt == 'mpi_int':
        return MPI_INTEGER()
    elif txt == 'mpi_float':
        return MPI_FLOAT()
    elif txt == 'mpi_double':
        return MPI_DOUBLE()
# ...

# ...
def append_mpi(namespace, declarations):
    """Adds MPI functions and constants to the namespace

    namespace: dict
        dictorionary containing all declared variables/functions/classes.

    declarations: dict
        dictorionary containing all declarations.
    """
    # ...
    namespace['mpi_comm_world']  = MPI_COMM_WORLD
    namespace['mpi_status_size'] = MPI_STATUS_SIZE
    namespace['mpi_proc_null']   = MPI_PROC_NULL
    # ...

    # ...
    for i in [MPI_ERROR, MPI_STATUS]:
        namespace[i.name] = i

        dec = MPI_Declare(i.dtype, i)
        declarations[i.name] = dec
    # ...

    # ...
    body        = []
    local_vars  = []
    global_vars = []
    hide        = True
    kind        = 'procedure'
    # ...

    # ...
    args        = []
    datatype    = 'int'
    allocatable = False
    shape       = None
    rank        = 0

    var_name = 'result_%d' % abs(hash(datatype))

    for f_name in ['mpi_init', 'mpi_finalize']:
        var = Variable(datatype, var_name)
        results = [var]

        stmt = FunctionDef(f_name, args, results, \
                           body, local_vars, global_vars, \
                           hide=hide, kind=kind)

        namespace[f_name] = stmt
    # ...

    # ...
    i = np.random.randint(10)
    var_name = 'result_%d' % abs(hash(i))
    err_name = 'error_%d' % abs(hash(i))

    for f_name in ['mpi_comm_size', 'mpi_comm_rank', 'mpi_abort']:
        var = Variable(datatype, var_name)
        err = Variable(datatype, err_name)
        results = [var, err]

        args = [namespace['mpi_comm_world']]
        stmt = FunctionDef(f_name, args, results, \
                           body, local_vars, global_vars, \
                           hide=hide, kind=kind)

        namespace[f_name] = stmt
    # ...

    return namespace, declarations
# ...

# Global variable namespace
namespace    = {}
headers      = {}
declarations = {}

namespace["True"]  = true
namespace["False"] = false
namespace["pi"]    = pi
namespace, declarations = append_mpi(namespace, declarations)

# ... builtin types
builtin_types  = ['int', 'float', 'double', 'complex']
# ...

# ... will contain user defined types
custom_types   = {}

# TODO MPI
#custom_types['Communicator'] = Communicator
# ...

# ... builtin functions
builtin_funcs_math_un = ['abs', \
#                         'asin', 'acsc', 'acot', \
#                         'acos', 'asec', \
                         'atan', 'atan2', \
                         'ceil', 'cos', 'cosh', 'cot', 'csc', \
                         'exp', 'log', 'max', 'min', \
                         'sec', 'sign', 'sin', 'sinh', \
                         'sqrt', 'tan', 'tanh']
builtin_funcs_math_bin = ['dot', 'pow']
builtin_funcs_math = builtin_funcs_math_un + \
                     builtin_funcs_math_bin

builtin_funcs  = ['zeros', 'ones', 'array', 'zeros_like', 'len', 'shape']
builtin_funcs += builtin_funcs_math

builtin_funcs_mpi = ['mpi_waitall']
builtin_funcs += builtin_funcs_mpi
# ...

# ...
def print_namespace():
    print "-------- namespace --------"
    for key, value in namespace.items():
        if not(key in ['True', 'False', 'pi']):
            if isinstance(value, Variable):
                print key, type(value), value.rank #, id(value)
            else:
                print key, type(value)
    print "---------------------------"
# ...

def get_attributs(expr):
    """
    finds attributs of the expression
    """
    d_var = {}
    d_var['datatype']    = None
    d_var['allocatable'] = None
    d_var['shape']       = None
    d_var['rank']        = None

#    print '>>>> expr = ', expr, type(expr)
#    print '>>>> expr = ', type(expr)
#    if isinstance(expr, Variable):
#        print '>>>> expr = ', expr, " rank = ", expr.rank, " id = ", id(expr)
#    else:
#        print '>>>> expr = ', expr, type(expr), " id = ", id(expr)

    if isinstance(expr, dict):
        d_var['datatype']    = expr['datatype']
        d_var['allocatable'] = expr['allocatable']
        d_var['shape']       = expr['shape']
        d_var['rank']        = expr['rank']
        return d_var
    elif isinstance(expr, (list, tuple)):
        ds = [get_attributs(a) for a in expr]

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
    elif isinstance(expr, DottedVariable):
        comm = expr.name[0]
        attr = expr.name[-1]
        base = comm.cls_base
        if not isinstance(base, MPI):
            raise TypeError("Expecting MPI instance")

        dtype = base.dtype(attr)

        d_var['datatype']    = dtype
        d_var['allocatable'] = False
        d_var['shape']       = None
        d_var['rank']        = 0
        d_var['cls_base']    = None
        return d_var
    elif isinstance(expr, MPI):
        if expr.is_integer:
            d_var['datatype']    = 'int'
        else:
            raise ValueError("Expecting a integer")
        d_var['allocatable'] = False
        d_var['shape']       = None
        d_var['rank']        = 0
        d_var['cls_base']    = MPI_COMM_WORLD
        return d_var
    elif isinstance(expr, (Ceil, Len)):
        d_var['datatype']    = 'int'
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
    elif isinstance(expr, FunctionCall):
        func = expr.func
        results = func.results
        if not(len(results) == 1):
            raise ValueError("Expecting one result.")
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
        for n, F in namespace.items():
            if isinstance(F, FunctionDef):
                avail_funcs.append(str(n))
        avail_funcs += builtin_funcs

        # this is to treat the upper/lower cases
        _known_functions = []
        for k, n in known_functions.items():
            _known_functions += [k, n]
        avail_funcs += _known_functions

        if not(name in avail_funcs):
            raise Exception("Could not find function {0}".format(name))

        if name in namespace:
            F = namespace[name]
            results = F.results

            if not(len(results) == 1):
                raise ValueError("Expecting a function with one return.")

            var = results[0]
            d_var['datatype']    = var.dtype
            d_var['allocatable'] = var.allocatable
            d_var['rank']        = var.rank
            if not(var.shape is None):
                d_var['shape'] = var.shape
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
                    name = arg.name
                    var = namespace[name]
                    return get_attributs(var)
                else:
                    return get_attributs(arg)
        return get_attributs(args)
    else:
        raise TypeError("get_attributs is not available for {0}".format(type(expr)))

    return d_var

# TODO add kwargs
def builtin_function(name, args, lhs=None):
    """
    User friendly interface for builtin function calls.

    name: str
        name of the function
    args: list
        list of arguments
    lhs: str
        name of the variable to assign to
    """
    # ...
    def get_arguments():
        # TODO appropriate default type
        dtype = DEFAULT_TYPE
        allocatable = True
        shape = []
        for i in args:
            if isinstance(i, DataType):
                dtype = i
            elif isinstance(i, Tuple):
                shape = [j for j in i]
            else:
                # TODO further check
                shape.append(i)
        rank = len(shape)
        if rank == 1:
            shape = shape[0]

        d_var = {}
        d_var['datatype'] = dtype
        d_var['allocatable'] = allocatable
        d_var['shape'] = shape
        d_var['rank'] = rank

        return d_var
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
    if name == "zeros":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        d_var = get_arguments()
        insert_variable(lhs, **d_var)
        lhs = namespace[lhs]
        return Zeros(lhs, d_var['shape'])
    elif name == "ones":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        d_var = get_arguments()
        insert_variable(lhs, **d_var)
        lhs = namespace[lhs]
        return Ones(lhs, d_var['shape'])
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
            return Assign(Symbol(lhs), expr)
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
            return Assign(lhs, expr)
    elif name in builtin_funcs_math_un + ['len']:
        if not(len(args) == 1):
            raise ValueError("pow takes exactly one argument")

        func = eval(known_functions[name])
        if lhs is None:
            return func(*args)
        else:
            d_var = {}
            # TODO get dtype from args
            if name in ['ceil', 'len']:
                d_var['datatype'] = 'int'
                d_var['rank']     = 0
            else:
                d_var['datatype'] = DEFAULT_TYPE
            insert_variable(lhs, **d_var)
            lhs = namespace[lhs]
            expr = func(*args)
            return Assign(lhs, expr)
    elif name in builtin_funcs_math_bin:
        if not(len(args) == 2):
            raise ValueError("pow takes exactly two arguments")

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
            return Assign(lhs, expr)
    elif name == "mpi_waitall":
        if not lhs:
            raise ValueError("Expecting a lhs.")
        if not(len(args) == 2):
            raise ValueError("Expecting exactly two arguments.")
        if not(args[0].name in namespace):
            raise ValueError("Undefined variable {0}".format(name))

        d_var = {}
        d_var['datatype']    = 'int'
        d_var['allocatable'] = False
        d_var['shape']       = None
        d_var['rank']        = 0

        insert_variable(lhs, **d_var)
        lhs = namespace[lhs]
        rhs = MPI_waitall(*args)
        return MPI_Assign(lhs, rhs, strict=False)
    else:
        raise ValueError("Expecting a builtin function. given : ", name)
    # ...

def insert_variable(var_name, \
                    datatype=None, \
                    rank=None, \
                    allocatable=None, \
                    shape=None, \
                    intent=None, \
                    var=None, \
                    cls_base=None):
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

    var: pyccel.types.ast.Variable
        if attributs are not given, then var must be provided.

    cls_base: class
        class base if variable is an object or an object member
    """
    if type(var_name) in [int, float]:
        return

    if DEBUG:
#    if True:
        print ">>>> trying to insert : ", var_name
        txt = '     datatype={0}, rank={1}, allocatable={2}, shape={3}, intent={4}'\
                .format(datatype, rank, allocatable, shape, intent)
        print txt

    if not isinstance(var_name, str):
        raise TypeError("Expecting a string for var_name.")

    if var_name in namespace:
        var = namespace[var_name]
        if datatype is None:
            datatype = var.dtype
        if rank is None:
            rank = var.rank
        if allocatable is None:
            allocatable = var.allocatable
        if shape is None:
            shape = var.shape

    # we create a variable (for annotation)
    var = Variable(datatype, var_name, \
                   rank=rank, \
                   allocatable=allocatable, \
                   shape=shape, \
                   cls_base=cls_base)

    # we create a declaration for code generation
    dec = Declare(datatype, var, intent=intent)

    if var_name in namespace:
        namespace.pop(var_name)
        declarations.pop(var_name)

    namespace[var_name]    = var
    declarations[var_name] = dec

# ...
def expr_with_trailer(expr, trailer=None):
    if trailer is None:
        return expr
    if isinstance(trailer, (tuple, list)):
        if len(trailer) == 0:
            return expr
        if len(trailer) == 1:
            return expr_with_trailer(expr, trailer[0])
        if expr.cls_base:
            if isinstance(expr.cls_base, MPI):
                comm = expr
                func = trailer[0].expr
                args = trailer[1].expr
                args += [comm]
#                print [type(a) for a in args]
                expr = eval('MPI_comm_{0}'.format(func))(*args)
        else:
            raise ValueError('Unable to construct expr from trailers.')
        return expr
    if isinstance(trailer, Trailer):
        return expr_with_trailer(expr, trailer.args)

    if isinstance(trailer, TrailerArgList):
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
        else:
            if len(args) > 0:
                expr = Function(str(expr))(*args)
            else:
                func = namespace[str(expr)]
                expr = FunctionCall(func, None)
    elif isinstance(trailer, TrailerSubscriptList):
        args = trailer.expr

        v = namespace[expr.name]
        expr = IndexedVariable(v.name, dtype=v.dtype)[args]
    elif isinstance(trailer, TrailerDots):
        args = trailer.expr

        # TODO add IndexedVariable, IndexedElement
        dottables = (Variable)
        if not(isinstance(expr, dottables)):
            raise TypeError("Expecting Variable")
        var_name = '{0}.{1}'.format(expr.name, args)
        found_var = (var_name in namespace)
#            if not(found_var):
#                raise ValueError("Undefined variable {}".format(var_name))
        if var_name in namespace:
            expr = namespace[var_name]
        else:
            if not(expr.name in namespace):
                raise ValueError("Undefined variable {}".format(expr.name))
            expr = DottedVariable(expr, args)

            obj  = expr.name[0]
            attr = expr.name[-1]
            base = obj.cls_base
            if isinstance(base, MPI):
                expr = eval('MPI_comm_{0}'.format(attr))(obj)
            else:
                expr = DottedVariable(expr, args)
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

        # ... reset global variables
        namespace    = {}
        headers      = {}
        declarations = {}

        namespace["True"]  = true
        namespace["False"] = false
        namespace["pi"]    = pi

        #namespace = append_mpi(namespace)
        # ...

    @property
    def declarations(self):
        """
        Returns the list of all declarations using objects from pyccel.types.ast
        """
        d = {}
        for key,dec in declarations.items():
            if dec.intent is None:
                d[key] = dec
        return d

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
        return Comment("")

class DeclarationStmt(BasicStmt):
    """Class representing a declaration statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the declaration statement.

        variables_names: list of str
            list of variable names.
        datatype: str
            datatype of the declared variables.
        """
        self.variables_name = kwargs.pop('variables')
        self.datatype = kwargs.pop('datatype')

        raise Exception("Need to be updated! not used anymore.")

        super(DeclarationStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the declaration statement. This property will return a list of
        declarations statements.
        """
        datatype = str(self.datatype)
        decs = []
        # TODO depending on additional options from the grammar
        for var in self.variables:
            dec = Variable(datatype, var.expr)
            decs.append(Declare(datatype, dec, intent='in'))

        self.update()

        return decs

# TODO: improve by creating the corresponding object in pyccel.types.ast
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
        Process the Delete statement by returning a pyccel.types.ast object
        """
        variables = [v.expr for v in self.variables]
        ls = []
        for var in variables:
            name = var.name
            if isinstance(name, (list, tuple)):
                name = '{0}.{1}'.format(name[0], name[1])
            if name in namespace:
                ls.append(namespace[name])
            else:
                raise Exception('Unknown variable {}'.format(name))

        self.update()

        return Del(ls)

# TODO: improve by creating the corresponding object in pyccel.types.ast
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
        Process the Delete statement by returning a pyccel.types.ast object
        """
        self.update()

        return self.label

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
        Process the If statement by returning a pyccel.types.ast object
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
        return [self.lhs]

    @property
    def expr(self):
        """
        Process the Assign statement by returning a pyccel.types.ast object
        """
        if not isinstance(self.rhs, ArithmeticExpression):
            raise TypeError("Expecting an expression")

        rhs      = self.rhs.expr
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
                return builtin_function(name.lower(), args, lhs=var_name)

        found_var = (var_name in namespace)
        if not(found_var):
            d_var = get_attributs(rhs)

#            print ">>>> AssignStmt : ", var_name, d_var

            d_var['allocatable'] = not(d_var['shape'] is None)
            if d_var['shape']:
                if DEBUG:
                    print "> Found an unallocated variable: ", var_name
                status = 'unallocated'
                like = allocatable_like(rhs)
            insert_variable(var_name, **d_var)

        if self.trailer is None:
            l = namespace[self.lhs]
        else:
            if isinstance(trailer, TrailerSubscriptList):
                v = namespace[str(self.lhs)]
                l = IndexedVariable(v.name, dtype=v.dtype)[args]
            elif isinstance(trailer, TrailerDots):
                # class attribut
                l = namespace[var_name]
            else:
                raise TypeError("Expecting SubscriptList or Dot")

        if not isinstance(rhs, MPI):
            return Assign(l, rhs, strict=False, status=status, like=like)
        else:
            return MPI_Assign(l, rhs, strict=False, status=status, like=like)

class MultiAssignStmt(BasicStmt):
    """
    Class representing multiple assignments.
    In fortran, this correspondans to the call of a subroutine.
    """
    def __init__(self, **kwargs):
        """
        Constructor for the multi Assign statement.

        lhs: list of str
            variables to assign to
        rhs: ArithmeticExpression
            expression to assign to the lhs
        """
        self.lhs = kwargs.pop('lhs')
        self.rhs = kwargs.pop('rhs')

        super(MultiAssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """Statement variables."""
        return self.lhs

    @property
    def expr(self):
        """
        Process the MultiAssign statement by returning a pyccel.types.ast object
        """
        lhs = self.lhs
        rhs = self.rhs.expr

        if not(isinstance(rhs, Function)):
            raise TypeError("Expecting a Function")

        f_name = str(type(rhs).__name__)
        # TODO additional functions like : shape
        if not(f_name in namespace):
            raise ValueError("Undefined function call {}.".format(f_name))

        F = namespace[f_name]
        if not(isinstance(F, FunctionDef)):
            raise TypeError("Expecting a FunctionDef")

        if not(len(F.results) == len(self.lhs)):
            raise ValueError("Wrong number of outputs.")

        for (var_name, result) in zip(self.lhs, F.results):
            if not(var_name in namespace):
                d_var = {}
                d_var['datatype']    = result.dtype
                d_var['allocatable'] = result.allocatable
                d_var['shape']       = result.shape
                d_var['rank']        = result.rank
                d_var['intent']      = None

#                print ">>>> MultiAssignStmt : ", var_name
#                print "                     : ", d_var

                d_var['allocatable'] = not(d_var['shape'] is None)
#                if d_var['shape']:
#                    if DEBUG:
#                        print "> Found an unallocated variable: ", var_name
#                    status = 'unallocated'
#                    like   = allocatable_like(rhs)
#                    print ">>>> Found variable : ", like
                insert_variable(var_name, **d_var)
        return MultiAssign(lhs, rhs)

#        if name == 'shape':
#            if not(len(args) == 1):
#                raise ValueError('shape takes only one argument.')
#            return Shape(lhs, args[0])
#        else:
#            return MultiAssign(lhs, rhs, args)

class ForStmt(BasicStmt):
    """Class representing a For statement."""

    def __init__(self, **kwargs):
        """
        Constructor for the For statement.

        iterable: str
            the iterable variable
        start: str
            start index
        end: str
            end index
        step: str
            step for the iterable. if not given, 1 will be used.
        body: list
            a list of statements for the body of the For statement.
        """
        self.iterable = kwargs.pop('iterable')
        self.start    = kwargs.pop('start')
        self.end      = kwargs.pop('end')
        self.step     = kwargs.pop('step', None)
        self.body     = kwargs.pop('body')

        super(ForStmt, self).__init__(**kwargs)

    @property
    def local_vars(self):
        """Local variables of the For statement."""
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
        d_var = {}
        d_var['datatype']    = 'int'
        d_var['allocatable'] = False
        d_var['rank']        = 0
        insert_variable(self.iterable, **d_var)

    @property
    def expr(self):
        """
        Process the For statement by returning a pyccel.types.ast object
        """
        i = Symbol(self.iterable, integer=True)

        if self.start in namespace:
            b = namespace[self.start]
        else:
            b = do_arg(self.start)

        if self.end in namespace:
            e = namespace[self.end]
        else:
            e = do_arg(self.end)

        if self.step is None:
            s = 1
        else:
            if self.step in namespace:
                s = namespace[self.step]
            else:
                s = do_arg(self.step)

        self.update()

        body = self.body.expr

        return For(i, (b,e,s), body)

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
        Process the While statement by returning a pyccel.types.ast object
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
            print "> FactorSigned "
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
            print "> AtomExpr "
        expr = self.op.expr

        return expr_with_trailer(expr, self.trailers)

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
            print "> AtomExpr "
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
            print "> Term "

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
            print "> ArithmeticExpression "

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
#        if True:
            print "> Atom "
            print self.op, type(self.op)

        op = self.op
        if op in ['shape']:
            raise ValueError('shape function can not be used in an expression.')

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
        elif op in namespace:
            if isinstance(namespace[op], FunctionDef):
                return Function(op)
            else:
                return namespace[op]
        elif op in builtin_funcs:
            return Function(op)
        elif op in builtin_types:
            return datatype(op)
        elif op == 'None':
            raise ValueError("Atom None not yet available.")
        elif op == 'True':
            return true
        elif op == 'False':
            return false
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
            print "> DEBUG "
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
            print "> DEBUG "

        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            ret = (ret or operand.expr)
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
            print "> DEBUG "

        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            ret = (ret and operand.expr)
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
            print "> DEBUG "

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
            print "> Comparison "

        ret = self.op[0].expr
        for operation, operand in zip(self.op[1::2], self.op[2::2]):
            if operation == "==":
                ret = EqualityStmt(ret, operand.expr)
            elif operation == ">":
                ret = Gt(ret, operand.expr)
            elif operation == ">=":
                ret = Ge(ret, operand.expr)
            elif operation == "<":
                ret = Lt(ret, operand.expr)
            elif operation == "<=":
                ret = Le(ret, operand.expr)
            elif operation == "<>":
                ret = NotequalStmt(ret, operand.expr)
            else:
                raise Exception('operation not yet available at position {}'
                                .format(self._tx_position))
        return ret

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

# TODO improve
class ReturnStmt(FlowStmt):
    """Base class representing a Return statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a return statement flow.

        variables: list
            list of variables to return, as strings
        results: list
            list of variables to return, as pyccel.types.ast objects
        """
        self.variables = kwargs.pop('variables')
        self.results   = None

        super(ReturnStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the return flow statement
        """
        self.update()

        decs = []
        # TODO depending on additional options from the grammar
        # TODO check that var is in namespace
        for var_name in self.variables:
            if var_name in namespace:
                var = namespace[var_name]
#                print var_name, var
                if isinstance(var, Variable): # TODO intent must be out => result
                    res = Variable(var.dtype, var_name, \
                                   rank=var.rank, \
                                   allocatable=var.allocatable, \
                                   shape=var.shape)
                else:
                    # TODO is it correct? raise?
                    datatype = var.datatype
                    res = Variable(datatype, var_name)
            else:
                raise()

            decs.append(res)

        self.results = decs
        return decs

class RaiseStmt(FlowStmt):
    """Base class representing a Raise statement in the grammar."""

class YieldStmt(FlowStmt):
    """Base class representing a Yield statement in the grammar."""

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
        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')
        self.body = kwargs.pop('body')
        self.parent = kwargs.get('parent', None)

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
        pyccel.types.ast
        """
#        print "*********** FunctionDefStmt.expr: Begin"
        name = str(self.name)
        args = self.args

        cls_instance = None
        if isinstance(self.parent, SuiteStmt):
            if isinstance(self.parent.parent, ClassDefStmt):
                cls_instance = self.parent.parent.name

        if cls_instance:
            name = '{0}.{1}'.format(cls_instance, name)
            # remove self from args
            args = args[1:]

            # insert self to namespace
            d_var = {}
            dtype = custom_types[cls_instance]()
            d_var['datatype']    = dtype
            d_var['allocatable'] = False
            d_var['shape']       = None
            d_var['rank']        = 0
            d_var['intent']      = 'inout'
            insert_variable('self', **d_var)

        if not(name in headers):
            raise Exception('Function header could not be found for {0}.'
                           .format(name))

        if not(len(args) == len(headers[name].dtypes)):
            raise Exception("Wrong number of arguments in the header.")

        # old occurence of args will be stored in scope
        scope_vars = {}
        scope_decs = {}
        h = headers[name]
        for arg_name, d in zip(args, h.dtypes):
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
            d_var['datatype']    = d[0]
            d_var['allocatable'] = False
            d_var['shape']       = None
            d_var['rank']        = rank
            d_var['intent']      = 'in'
            insert_variable(arg_name, **d_var)
            var = namespace[arg_name]

        body = self.body.expr

        args    = [namespace[arg_name] for arg_name in self.args]
        prelude = [declarations[arg_name] for arg_name in self.args]

        # ...
        results = []
        for stmt in self.body.stmts:
            if isinstance(stmt, ReturnStmt):
                results += stmt.results
        # ...

        # ... cleaning the namespace
        for arg_name in self.args:
            del declarations[arg_name]
            del namespace[arg_name]

        ls = self.local_vars + self.stmt_vars
        for var_name in ls:
            if var_name in namespace:
                prelude.append(declarations[var_name])

                del namespace[var_name]
                del declarations[var_name]
        # ...

        # ...
        for arg_name, var in scope_vars.items():
            var = scope_vars.pop(arg_name)
            namespace[arg_name] = var

        for arg_name, dec in scope_decs.items():
            dec = scope_decs.pop(arg_name)
            declarations[arg_name] = dec
        # ...

        body = prelude + body

        local_vars  = []
        global_vars = []

        # rename the method in the class case
        f_name = name
        cls_name = None
        if cls_instance:
            f_name   = name.split('.')[-1]
            cls_name = name.split('.')[0]
        stmt = FunctionDef(f_name, args, results, body, \
                           local_vars, global_vars, \
                           cls_name=cls_name)
        namespace[name] = stmt
#        print "*********** FunctionDefStmt.expr: End"

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
        pyccel.types.ast
        """
#        print "*********** ClassDefStmt.expr: Begin"
        name = str(self.name)

        if not(name in headers):
            raise Exception('Class header could not be found for {0}.'
                           .format(name))

        header  = headers[name]
        options = header.options

        body    = self.body.expr

        # ...
        attributs = []
        d = {}
        for key,v in namespace.items():
            if key.startswith('self.'):
                d[key] = v
                n = key.split('self.')[-1]
                var = v.clone(n)
                attributs.append(var)
        # ...

        methods = []
        for stmt in body:
            if isinstance(stmt, FunctionDef):
                methods.append(stmt)

        stmt = ClassDef(name, attributs, methods, options)
        namespace[name] = stmt

        # ... cleaning
        for k,v in d.items():
            namespace.pop(k)
            declarations.pop(k)
        # ...

#        print "*********** ClassDefStmt.expr: End"

        return stmt

class PythonPrintStmt(BasicStmt):
    """Class representing a Print statement as described in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Print statement.

        name: str
            is equal to 'print'
        args: list
            list of atoms to print
        """
        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')

        super(PythonPrintStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Print statement,
        by returning the appropriate object from pyccel.types.ast
        """
        self.update()

        func_name   = self.name
        args        = self.args
        expressions=[]

        for arg in args:
            if not isinstance(arg,str):
               expressions.append(arg.expr)
            else:
                expressions.append(arg)
        return Print(expressions)

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
        by returning the appropriate object from pyccel.types.ast
        """
        self.update()
        return Comment(self.text)

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
        by returning a list of appropriate objects from pyccel.types.ast
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
        pyccel.types.ast
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
        pyccel.types.ast
        """
        self.update()
        return [arg.expr for arg in  self.args]

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
        pyccel.types.ast
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
        pyccel.types.ast
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
        pyccel.types.ast
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

class ThreadStmt(BasicStmt):
    """Class representing a Thread call function in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Thread function call.

        lhs: str
            variable name to create
        func: str
            function to call
        """
        self.lhs  = kwargs.pop('lhs')
        self.func = kwargs.pop('func')

        super(ThreadStmt, self).__init__(**kwargs)

    def update(self):
        """
        appends the variable to the namespace
        """
        var_name = str(self.lhs)
        if not(var_name in namespace):
            insert_variable(var_name, datatype='int', rank=0)
        else:
            raise Exception('Already declared variable for thread_id.')

    @property
    def expr(self):
        """
        Process the Thread function call,
        by returning the appropriate object from pyccel.types.ast
        """
        self.update()

        var_name = str(self.lhs)
        var = Symbol(var_name)

        func = str(self.func)
        if func == 'thread_id':
            return ThreadID(var)
        elif func == 'thread_number':
            return ThreadsNumber(var)
        else:
            raise Exception('Wrong value for func.')

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
        by returning a list of appropriate objects from pyccel.types.ast
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

class StencilStmt(AssignStmt):
    """Class representing a Stencil statement in the grammar."""

    def __init__(self, **kwargs):
        """
        Constructor for a Stencil statement.

        lhs: str
            variable name to create
        parameters: list
            list of parameters needed for the Stencil object.
        """
        self.lhs        = kwargs.pop('lhs')
        self.parameters = kwargs.pop('parameters')

        labels = [str(p.label) for p in self.parameters]
        values = [p.value.value for p in self.parameters]
        d = {}
        for (label, value) in zip(labels, values):
            d[label] = value
        self.parameters = d

        try:
            self.datatype = self.parameters['dtype']
        except:
            self.datatype = DEFAULT_TYPE

        try:
            self.shape = self.parameters['shape']
            # on LRZ, self.shape can be a list of ArgList
            # this is why we do the following check
            # maybe a bug in textX
            if isinstance(self.shape, list):
                if isinstance(self.shape[0], ArgList):
                    self.shape = self.shape[0].args
            elif isinstance(self.shape, ArgList):
                self.shape = self.shape.args
        except:
            raise Exception('Expecting shape at position {}'
                            .format(self._tx_position))

        try:
            self.step = self.parameters['step']
            # on LRZ, self.step can be a list of ArgList
            # this is why we do the following check
            # maybe a bug in textX
            if isinstance(self.step, list):
                if isinstance(self.step[0], ArgList):
                    self.step = self.step[0].args
            elif isinstance(self.step, ArgList):
                self.step = self.step.args
        except:
            raise Exception('Expecting step at position {}'
                            .format(self._tx_position))

        super(AssignStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """returns statement variables."""
        return [self.lhs]

    def update(self):
        """
        specific treatments before process
        """
        var_name = self.lhs
        if not(var_name in namespace):
            if DEBUG:
                print("> Found new variable " + var_name)

            datatype = self.datatype

            # ...
            def format_entry(s_in):
                rank = 0
                if isinstance(s_in, int):
                    s_out = s_in
                    rank = 1
                elif isinstance(s_in, float):
                    s_out = int(s_in)
                    rank = 1
                elif isinstance(s_in, list):
                    s_out = []
                    for s in s_in:
                        if isinstance(s, (int, float)):
                            s_out.append(int(s))
                        elif isinstance(s, str):
                            if not(s in namespace):
                                raise Exception('Could not find s_out variable.')
                            s_out.append(namespace[s])
    #                    elif isinstance(s,ArgList):
    #                        s_out.append(s.expr)
                        else:
                            print ("> given type: ", type(s))
                            raise TypeError('Expecting a int, float or string')
                    rank = len(s_out)
                else:
                    s_out = str(s_in)
                    if s_out in namespace:
                        s_out = namespace[s_out]
                        # TODO compute rank
                        rank = 1
                    else:
                        raise Exception('Wrong instance for s_out : '.format(type(s_in)))
                return s_out, rank
            # ...

            # ...
            self.shape, r_1 = format_entry(self.shape)
            self.step,  r_2 = format_entry(self.step)
            rank = r_1 + r_2
            # ...

            if datatype is None:
                if DEBUG:
                    print("> No Datatype is specified, int will be used.")
                datatype = 'int'
            elif isinstance(datatype, list):
                datatype = datatype[0] # otherwise, it's not working on LRZ
            # TODO check if var is a return value
            insert_variable(var_name, \
                            datatype=datatype, \
                            rank=rank, \
                            allocatable=True,shape = self.shape)

    @property
    def expr(self):
        """
        Process the Stencil statement,
        by returning the appropriate object from pyccel.types.ast
        """
        self.update()

        shape = self.shape
        step  = self.step

        var_name = self.lhs
        var = Symbol(var_name)

        return Stencil(var, shape, step)

class EvalStmt(BasicStmt):
    """
    Class representing an Eval statement in the grammar
    """
    def __init__(self, **kwargs):
        """
        Constructor for a eval statement.

        lhs: str
            variable name to create
        expression: str
            Expression to be evaluated
        """
        self.lhs        = kwargs.pop('lhs')
        self.expression = kwargs.pop('expression')

        super(EvalStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Eval statement,
        by returning a list of appropriate objects from pyccel.types.ast
        """
        f_name = str(self.lhs)

        if not(f_name in headers):
            raise Exception('Function header could not be found for {0}.'
                           .format(f_name))

        h = headers[f_name]

        # ... treating function/procedure arguments
        args = []
        for d in h.dtypes:
            var_name = 'arg_%d' % abs(hash(d))
            rank = 0
            for i in d[1]:
                if isinstance(i, Slice):
                    rank += 1
            datatype    = d[0]
            allocatable = False
            shape       = None
            var = Variable(datatype, var_name, \
                           rank=rank, \
                           allocatable=allocatable, \
                           shape=shape)
            args.append(var)
        # ...

        # ... treating function/procedure results
        results = []
        for d in h.results:
            var_name = 'result_%d' % abs(hash(d))
            rank = 0
            for i in d[1]:
                if isinstance(i, Slice):
                    rank += 1
            datatype    = d[0]
            allocatable = False
            shape       = None
            var = Variable(datatype, var_name, \
                           rank=rank, \
                           allocatable=allocatable, \
                           shape=shape)
            results.append(var)
        # ...

        body        = []
        local_vars  = []
        global_vars = []
        hide        = True
        stmt = FunctionDef(f_name, args, results, \
                           body, local_vars, global_vars, \
                           hide=hide, kind=h.kind)
        namespace[f_name] = stmt

        return stmt

class ExecStmt(BasicStmt):
    """
    Class representing an Exec statement in the grammar
    """
    def __init__(self, **kwargs):
        """
        Constructor for a eval statement.

        lhs: str
            variable name to create
        module: str
            module where the function lives
        function: str
            function to call from the module
        args: list
            list of arguments to feed the function call
        """
        self.lhs      = kwargs.pop('lhs')
        self.module   = kwargs.pop('module')
        self.function = kwargs.pop('function')
        self.args     = kwargs.pop('args')

        super(ExecStmt, self).__init__(**kwargs)

    @property
    def stmt_vars(self):
        """returns the statement variables."""
        return self.lhs

    def update(self):
        """
        Pre-process. We check that the lhs is not in the namespace.
        """
        for var_name in self.lhs:
            if not(var_name in namespace):
                raise Exception('Undefined variable {}.'.format(var_name))

    @property
    def expr(self):
        """
        Process the Exec statement,
        by returning a list of appropriate objects from pyccel.types.ast
        """
        # TODO must check compatibility
#        self.update()

        module_name   = self.module
        function_name = self.function

        try:
            import importlib
            module   = importlib.import_module(module_name)
        except:
            raise Exception('Could not import module {}.'.format(module_name))

        try:
            function = getattr(module, "{}".format(function_name))
        except:
            raise Exception('Could not import function {}.'.format(function_name))

        args = self.args.expr
        rs   = function(*args)

        if isinstance(rs, tuple):
            rs = list(rs)

        if not isinstance(rs, list):
            rs = [rs]

        if not(len(rs) == len(self.lhs)):
            raise Exception('Incompatible lhs with function output.')

        ls = []
        for (l,r) in zip(self.lhs, rs):
            if isinstance(r, (int, float, complex)):
                rank        = 0
                shape       = None
                allocatable = False
                # check if numpy variable
                if (type(r).__module__ == np.__name__):
                    t = r.dtype
                else:
                    t = type(r)
                datatype = convert_numpy_type(t)

            elif isinstance(r, ndarray):
                shape       = r.shape
                rank        = len(shape)
                allocatable = True
                datatype    = convert_numpy_type(r.dtype)
            else:
                raise TypeError('Expecting int, float, complex or numpy array.')

            if l in namespace:
                raise Exception('Variable {} already defined, '
                                'cannot be used in eval statement.'.format(l))

            insert_variable(l, \
                            datatype=datatype, \
                            rank=rank, \
                            allocatable=allocatable, \
                            shape=shape)

            var = namespace[l]
#            print type(r)
            if isinstance(r, ndarray):
                stmt = Array(var, r, shape)
            else:
                stmt = Assign(var, r)

            ls.append(stmt)

        return ls

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
        attributs = []
        for dec in self.decs:
            if dec.trailer is None:
                attr = ''
            else:
                attr = dec.trailer.expr
            attributs.append(attr)

        self.dtypes = zip(dtypes, attributs)

        if not (self.results is None):
            r_dtypes    = [dec.dtype for dec in self.results.decs]
            attributs = []
            for dec in self.results.decs:
                if dec.trailer is None:
                    attr = ''
                else:
                    attr = dec.trailer.expr
                attributs.append(attr)
            self.results = zip(r_dtypes, attributs)

        kind = str(self.kind)

        h = FunctionHeader(self.name, self.dtypes, self.results, kind=kind)
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
        # create a new Datatype for the current class
        custom_types[self.name] = DataTypeFactory(self.name, ("_name"))

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
        self.dtypes = zip(dtypes, attributs)

        if not (self.results is None):
            r_dtypes    = [dec.dtype for dec in self.results.decs]
            attributs = []
            for dec in self.results.decs:
                if dec.trailer is None:
                    attr = ''
                else:
                    attr = dec.trailer.expr
                attributs.append(attr)
            self.results = zip(r_dtypes, attributs)

        cls_instance = self.dtypes[0]
        cls_instance = cls_instance[0] # remove the attribut
        dtypes = self.dtypes[1:]
        h = MethodHeader((cls_instance, self.name), dtypes, self.results)
        headers[h.name] = h
        return h

