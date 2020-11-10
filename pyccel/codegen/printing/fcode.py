# coding: utf-8
# pylint: disable=R0201
# pylint: disable=missing-function-docstring

"""Print to F90 standard. Trying to follow the information provided at
www.fortran90.org as much as possible."""


import string
from itertools import chain
from collections import OrderedDict

import functools
import operator

from numpy import asarray

from sympy.core import Symbol
from sympy.core import Tuple
from sympy.core.function import Function, Application
from sympy.core.numbers import NegativeInfinity as NINF
from sympy.core.numbers import Infinity as INF

from sympy.logic.boolalg import Not

from pyccel.ast.core import get_iterable_ranges
from pyccel.ast.core import AddOp, MulOp, SubOp, DivOp
from pyccel.ast.core import Nil
from pyccel.ast.core import SeparatorComment, Comment
from pyccel.ast.core import ConstructorCall
from pyccel.ast.core import Subroutine
from pyccel.ast.core import ErrorExit
from pyccel.ast.itertoolsext import Product
from pyccel.ast.core import (Assign, AliasAssign, Variable,
                             VariableAddress,
                             TupleVariable, Declare,
                             IndexedVariable, CodeBlock,
                             IndexedElement, Slice, Dlist,
                             DottedName, AsName,
                             If, PyccelArraySize)


from pyccel.ast.core      import PyccelAdd, PyccelMul, PyccelDiv, PyccelMinus
from pyccel.ast.core      import PyccelUnarySub, PyccelMod
from pyccel.ast.core      import FunctionCall

from pyccel.ast.builtins  import (PythonEnumerate, PythonInt, PythonLen,
                                  PythonMap, PythonPrint, PythonRange,
                                  PythonZip, PythonTuple, PythonFloat)
from pyccel.ast.builtins  import PythonComplex, PythonBool
from pyccel.ast.datatypes import is_pyccel_datatype
from pyccel.ast.datatypes import is_iterable_datatype, is_with_construct_datatype
from pyccel.ast.datatypes import NativeSymbol, NativeString, str_dtype
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeReal
from pyccel.ast.datatypes import NativeRange, NativeTensor, NativeTuple
from pyccel.ast.datatypes import CustomDataType
from pyccel.ast.numbers   import Integer, Float
from pyccel.ast.numbers   import BooleanTrue

from pyccel.ast.utilities import builtin_import_registery as pyccel_builtin_import_registery

from pyccel.ast.numpyext import NumpyFull, NumpyArray, NumpyLinspace, NumpyDiag, NumpyCross
from pyccel.ast.numpyext import NumpyReal, NumpyWhere
from pyccel.ast.numpyext import NumpyComplex, NumpyMod, NumpyFloat
from pyccel.ast.numpyext import NumpyFullLike, NumpyEmptyLike, NumpyZerosLike, NumpyOnesLike
from pyccel.ast.numpyext import NumpyRand, NumpyRandint
from pyccel.ast.numpyext import NumpyNewArray
from pyccel.ast.numpyext import Shape

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *
from pyccel.codegen.printing.codeprinter import CodePrinter


# TODO: add examples
# TODO: use _get_statement when returning a string

__all__ = ["FCodePrinter", "fcode"]

known_functions = {
    "sign": "sign",       # TODO: move to numpyext
    "conjugate": "conjg"  # TODO: move to numpyext
}

numpy_ufunc_to_fortran = {
    'NumpyAbs'  : 'abs',
    'NumpyFabs'  : 'abs',
    'NumpyMin'  : 'minval',
    'NumpyMax'  : 'maxval',
    'NumpyFloor': 'floor',  # TODO: might require special treatment with casting
    # ---
    'NumpyExp' : 'exp',
    'NumpyLog' : 'Log',
    # 'NumpySqrt': 'Sqrt',  # sqrt is printed using _Print_NumpySqrt
    # ---
    'NumpySin'    : 'sin',
    'NumpyCos'    : 'cos',
    'NumpyTan'    : 'tan',
    'NumpyArcsin' : 'asin',
    'NumpyArccos' : 'acos',
    'NumpyArctan' : 'atan',
    'NumpyArctan2': 'atan2',
    'NumpySinh'   : 'sinh',
    'NumpyCosh'   : 'cosh',
    'NumpyTanh'   : 'tanh',
    'NumpyArcsinh': 'asinh',
    'NumpyArccosh': 'acosh',
    'NumpyArctanh': 'atanh',
}

math_function_to_fortran = {
    'MathAcos'   : 'acos',
    'MathAcosh'  : 'acosh',
    'MathAsin'   : 'asin',
    'MathAsinh'  : 'asinh',
    'MathAtan'   : 'atan',
    'MathAtan2'  : 'atan2',
    'MathAtanh'  : 'atanh',
    'MathCopysign': 'sign',
    'MathCos'    : 'cos',
    'MathCosh'   : 'cosh',
    # 'MathDegrees': '???',  # TODO
    'MathErf'    : 'erf',
    'MathErfc'   : 'erfc',
    'MathExp'    : 'exp',
    # 'MathExpm1'  : '???', # TODO
    'MathFabs'   : 'abs',
    # 'MathFmod'   : '???',  # TODO
    # 'MathFsum'   : '???',  # TODO
    'MathGamma'  : 'gamma',
    'MathHypot'  : 'hypot',
    # 'MathLdexp'  : '???',  # TODO
    'MathLgamma' : 'log_gamma',
    'MathLog'    : 'log',
    'MathLog10'  : 'log10',
    # 'MathLog1p'  : '???', # TODO
    # 'MathLog2'   : '???', # TODO
    # 'MathPow'    : '???', # TODO
    # 'MathRadians': '???', # TODO
    'MathSin'    : 'sin',
    'MathSinh'   : 'sinh',
    # 'MathSqrt'   : 'sqrt', # sqrt is printed using _Print_MathSqrt
    'MathTan'    : 'tan',
    'MathTanh'   : 'tanh',
    # ---
    'MathCeil'     : 'ceiling',
    # 'MathFactorial': '???', # TODO
    'MathFloor'    : 'floor',
    # 'MathGcd'      : '???', # TODO
    'MathTrunc'    : 'dint', # TODO
    # ---
    # 'MathIsclose' : '???', # TODO
    # 'MathIsfinite': '???', # TODO
    # 'MathIsinf'   : '???', # TODO
    'MathIsnan'   : 'isnan',
}

_default_methods = {
    '__init__': 'create',
    '__del__' : 'free',
}

python_builtin_datatypes = {
    'integer' : PythonInt,
    'real'    : PythonFloat,
    'bool'    : PythonBool,
    'complex' : PythonComplex
}

errors = Errors()

class FCodePrinter(CodePrinter):
    """A printer to convert sympy expressions to strings of Fortran code"""
    printmethod = "_fcode"
    language = "Fortran"

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 15,
        'user_functions': {},
        'human': True,
        'source_format': 'fixed',
        'tabwidth': 2,
        'contract': True,
        'standard': 77
    }

    _operators = {
        'and': '.and.',
        'or': '.or.',
        'xor': '.neqv.',
        'equivalent': '.eqv.',
        'not': '.not. ',
    }

    _relationals = {
        '!=': '/=',
    }


    def __init__(self, parser, settings={}):

        prefix_module = settings.pop('prefix_module', None)

        if parser.filename:
            errors.set_target(parser.filename, 'file')

        CodePrinter.__init__(self, settings)
        self.parser = parser
        self._namespace = self.parser.namespace
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._current_function = None

        self._additional_code = None

        self.prefix_module = prefix_module

    def set_current_function(self, name):

        if name:
            self._namespace = self._namespace.sons_scopes[name]
            if self._current_function:
                name = DottedName(self._current_function, name)
        else:
            self._namespace = self._namespace.parent_scope
            if isinstance(self._current_function, DottedName):

                # case of a function inside a function

                name = self._current_function.name[:-1]
                if len(name) > 1:
                    name = DottedName(*name)
                else:
                    name = name[0]
        self._current_function = name

    def get_function(self, name):
        container = self._namespace
        while container:
            if name in container.functions:
                return container.functions[name]
            container = container.parent_scope
        errors.report(UNDEFINED_FUNCTION, symbol=name,
            severity='fatal')


    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {0}".format(text)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    def _handle_fortran_specific_a_prioris(self, var_list):
        for v in var_list:
            if isinstance(v, TupleVariable):
                if v.is_pointer or v.inconsistent_shape:
                    v.is_homogeneous = False

    # ============ Elements ============ #

    def _print_Module(self, expr):
        self._handle_fortran_specific_a_prioris(self.parser.get_variables(self._namespace))

        name = self._print(expr.name)
        name = name.replace('.', '_')
        if not name.startswith('mod_') and self.prefix_module:
            name = '{prefix}_{name}'.format(prefix=self.prefix_module,
                                            name=name)

        imports = ''.join(self._print(i) for i in expr.imports)
        decs    = ''.join(self._print(i) for i in expr.declarations)
        body    = ''

        # ... TODO add other elements
        private_funcs = [f.name for f in expr.funcs if f.is_private]
        private = private_funcs
        if private:
            private = ','.join(self._print(i) for i in private)
            private = 'private :: {}\n'.format(private)
        else:
            private = ''
        # ...

        # ...
        sep = self._print(SeparatorComment(40))
        interfaces = ''
        if expr.interfaces:
            interfaces = '\n'.join(self._print(i) for i in expr.interfaces if not i.hide)
            for interface in expr.interfaces:
                if not interface.hide:
                    body += '\n'.join(''.join([sep, self._print(i), sep]) for i in interface.functions)

        if expr.funcs:
            body += '\n'.join(''.join([sep, self._print(i), sep]) for i in expr.funcs)
        # ...

        # ...
        for i in expr.classes:
            # update decs with declarations from ClassDef
            c_decs, c_funcs = self._print(i)
            decs = '{0}\n{1}'.format(decs, c_decs)
            body = '{0}\n{1}\n'.format(body, c_funcs)
        # ...

        contains = 'contains\n' if (expr.funcs or expr.classes or expr.interfaces) else ''

        parts = ['module {}\n'.format(name),
                 imports,
                 'implicit none\n',
                 private,
                 decs,
                 interfaces,
                 contains,
                 body,
                 'end module {}\n'.format(name)]

        return '\n'.join([a for a in parts if a])

    def _print_Program(self, expr):
        self._handle_fortran_specific_a_prioris(self.parser.get_variables(self._namespace))

        name    = 'prog_{0}'.format(self._print(expr.name)).replace('.', '_')
        imports = ''.join(self._print(i) for i in expr.imports)
        body    = self._print(expr.body)

        # Print the declarations of all variables in the namespace, which include:
        #  - user-defined variables (available in Program.variables)
        #  - pyccel-generated variables added to Scope when printing 'expr.body'
        variables = self.parser.get_variables(self._namespace)
        decs = ''.join(self._print_Declare(Declare(v.dtype, v)) for v in variables)

        # Detect if we are using mpi4py
        # TODO should we find a better way to do this?
        mpi = any('mpi4py' == str(getattr(i.source, 'name', i.source)) for i in expr.imports)

        # Additional code and variable declarations for MPI usage
        # TODO: check if we should really add them like this
        if mpi:
            body = 'call mpi_init(ierr)\n'+\
                   '\nallocate(status(0:-1 + mpi_status_size)) '+\
                   '\nstatus = 0\n'+\
                   body +\
                   '\ncall mpi_finalize(ierr)'

            decs += '\ninteger :: ierr = -1' +\
                    '\ninteger, allocatable :: status (:)'

        parts = ['program {}\n'.format(name),
                 imports,
                'implicit none\n',
                 decs,
                 body,
                'end program {}\n'.format(name)]

        return '\n'.join(a for a in parts if a)

    def _print_Import(self, expr):

        source = ''
        if str(expr.source) in pyccel_builtin_import_registery:
            return ''

        if isinstance(expr.source, DottedName):
            source = expr.source.name[-1]
        else:
            source = self._print(expr.source)

        # importing of pyccel extensions is not printed
        if source in pyccel_builtin_import_registery:
            return ''

        if 'mpi4py' == str(getattr(expr.source,'name',expr.source)):
            return 'use mpi\n' + 'use mpiext\n'

        if len(expr.target) == 0:
            return 'use {}\n'.format(source)

        prefix = 'use {}, only:'.format(source)

        code = ''
        for i in expr.target:
            if isinstance(i, AsName):
                target = '{target} => {name}'.format(target=self._print(i.target),
                                                     name=self._print(i.name))
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=target)

            elif isinstance(i, DottedName):
                target = '_'.join(self._print(j) for j in i.name)
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=target)

            elif isinstance(i, str):
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=str(i))

            elif isinstance(i, Symbol):
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=str(i.name))

            else:
                raise TypeError('Expecting str, Symbol, DottedName or AsName, '
                                'given {}'.format(type(i)))

            code = (code + '\n' + line) if code else line

        # in some cases, the source is given as a string (when using metavar)
        code = code.replace("'", '')
        return self._get_statement(code) + '\n'

    def _print_TupleImport(self, expr):
        code = '\n'.join(self._print(i) for i in expr.imports)
        return self._get_statement(code) + '\n'

    def _print_PythonPrint(self, expr):
        args = []
        for f in expr.expr:
            if isinstance(f, str):
                args.append("'{}'".format(f))
            elif isinstance(f, (Tuple, PythonTuple)):
                for i in f:
                    args.append("{}".format(self._print(i)))
            elif isinstance(f, TupleVariable) and not f.is_homogeneous:
                for i in f:
                    args.append("{}".format(self._print(i)))
            elif f.dtype is NativeString() and f != expr.expr[-1]:
                args.append("{} // ' ' ".format(self._print(f)))
            else:
                args.append("{}".format(self._print(f)))

        code = ', '.join(['print *', *args])
        return self._get_statement(code) + '\n'

    def _print_SymbolicPrint(self, expr):
        # for every expression we will generate a print
        code = '\n'.join("print *, 'sympy> {}'".format(a) for a in expr.expr)
        return self._get_statement(code) + '\n'

    def _print_Comment(self, expr):
        comments = self._print(expr.text)
        return '!' + comments + '\n'

    def _print_CommentBlock(self, expr):
        txts = expr.comments
        ln = max(len(i) for i in txts)
        if ln<20:
            ln = 20
        top  = '!' + '_'*int((ln-12)/2) + 'CommentBlock' + '_'*int((ln-12)/2) + '!'
        ln = len(top)
        bottom = '!' + '_'*(ln-2) + '!'

        for i,txt in enumerate(txts):
            txts[i] = '!' + txt + ' '*(ln -2 - len(txt)) + '!'


        body = '\n'.join(i for i in txts)

        return ('{0}\n'
                '{1}\n'
                '{2}\n').format(top, body, bottom)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_NewLine(self, expr):
        return '\n'

    def _print_AnnotatedComment(self, expr):
        accel = self._print(expr.accel)
        txt   = str(expr.txt)
        if len(txt)>72:
            txts = []
            while len(txt)>72:
                txts.append(txt[:72])
                txt  = txt[72:]
            if txt:
                txts.append(txt)

            txt = '&\n!${} &'.format(accel).join(txt for txt in txts)

        return '!${0} {1}\n'.format(accel, txt)

    def _print_Tuple(self, expr):
        shape = list(reversed(asarray(expr).shape))
        if len(shape)>1:
            arg = functools.reduce(operator.concat, expr)
            elements = ', '.join(self._print(i) for i in arg)
            return 'reshape(['+ elements + '], '+ self._print(Tuple(*shape)) + ')'
        fs = ', '.join(self._print(f) for f in expr)
        return '[{0}]'.format(fs)

    def _print_PythonAbs(self, expr):
        """ print the python builtin function abs
        args : variable
        """
        return "abs({})".format(self._print(expr.arg))

    def _print_PythonTuple(self, expr):
        shape = Tuple(*reversed(expr.shape))
        if len(shape)>1:
            elements = ', '.join(self._print(i) for i in expr)
            return 'reshape(['+ elements + '], '+ self._print(shape) + ')'
        fs = ', '.join(self._print(f) for f in expr)
        return '[{0}]'.format(fs)

    def _print_TupleVariable(self, expr):
        if expr.is_homogeneous:
            return self._print_Variable(expr)
        else:
            fs = ', '.join(self._print(f) for f in expr)
            return '[{0}]'.format(fs)

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_ValuedVariable(self, expr):
        if expr.is_argument:
            return self._print_Variable(expr)
        else:
            return '{} = {}'.format(self._print(expr.name), self._print(expr.value))

    def _print_VariableAddress(self, expr):
        return self._print(expr.variable)

    def _print_Constant(self, expr):
        val = Float(expr.value)
        return self._print(val)

    def _print_DottedVariable(self, expr):
        if isinstance(expr.args[1], Function):
            func = expr.args[1].func
            name = func.__name__
            # ...
            code_args = ''
            code_args = ', '.join(self._print(i) for i in expr.args[1].args)
            code = '{0}({1})'.format(name, code_args)
                # ...
                # ...
            code = '{0}%{1}'.format(self._print(expr.args[0]), code)
            if isinstance(func, Subroutine):
                code = 'call {0}'.format(code)
            return code
        return self._print(expr.args[0]) + '%' +self._print(expr.args[1])

    def _print_DottedName(self, expr):
        return ' % '.join(self._print(n) for n in expr.name)

    def _print_Concatenate(self, expr):
        code = ', '.join(self._print(a) for a in expr.args)
        return '[' + code + ']'

    def _print_Lambda(self, expr):
        return '"{args} -> {expr}"'.format(args=expr.variables, expr=expr.expr)

#    # TODO this is not used anymore since, we are calling printer inside
#    #      numpyext. must be improved!!
#    def _print_ZerosLike(self, expr):
#        lhs = self._print(expr.lhs)
#        rhs = self._print(expr.rhs)
#        if isinstance(expr.rhs, IndexedElement):
#            shape = []
#            for i in expr.rhs.indices:
#                if isinstance(i, Slice):
#                    shape.append(i)
#            rank = len(shape)
#        else:
#            rank = expr.rhs.rank
#        rs = []
#        for i in range(1, rank+1):
#            l = 'lbound({0},{1})'.format(rhs, str(i))
#            u = 'ubound({0},{1})'.format(rhs, str(i))
#            r = '{0}:{1}'.format(l, u)
#            rs.append(r)
#        shape = ', '.join(self._print(i) for i in rs)
#        init_value = self._print(expr.init_value)
#
#        code  = ('allocate({lhs}({shape}))\n'
#                 '{lhs} = {init_value}').format(lhs=lhs,
#                                                shape=shape,
#                                                init_value=init_value)
#
#        return self._get_statement(code)

    def _print_SumFunction(self, expr):
        return str(expr)

    def _print_PythonLen(self, expr):
        var = expr.arg
        idx = 1 if var.order == 'F' else var.rank
        return 'size({},{})'.format(self._print(var), self._print(idx))

    def _print_PythonSum(self, expr):
        args = [self._print(arg) for arg in expr.args]
        return "sum({})".format(", ".join(args))

    #========================== Numpy Elements ===============================#

    def _print_NumpySum(self, expr):
        return expr.fprint(self._print)

    def _print_NumpyProduct(self, expr):
        return expr.fprint(self._print)

    def _print_NumpyMatmul(self, expr):
        return expr.fprint(self._print)

    def _print_NumpyCross(self, expr):
        return expr.fprint(self._print)

    def _print_NumpyNorm(self, expr):
        return expr.fprint(self._print)

    def _print_NumpyLinspace(self, expr):
        return expr.fprint(self._print)

    def _print_NumpyArray(self, expr):
        return expr.fprint(self._print)

    def _print_NumpyFloor(self, expr):
        result_code = self._print_MathFloor(expr)
        return 'real({}, {})'.format(result_code, expr.precision)

    def _print_PythonFloat(self, expr):
        return expr.fprint(self._print)

    # ======================================================================= #
    def _print_PyccelArraySize(self, expr):
        return expr.fprint(self._print)

    def _print_PythonInt(self, expr):
        return expr.fprint(self._print)

    def _print_MathFloor(self, expr):
        arg = expr.args[0]
        arg_code = self._print(arg)

        # math.floor on integer argument is identity,
        # but we need parentheses around expressions
        if arg.dtype is NativeInteger():
            return '({})'.format(arg_code)

        prec = expr.precision
        prec_code = self._print(prec)
        return 'floor({}, kind={})'.format(arg_code, prec_code)

    def _print_NumpyFloat(self, expr):
        return expr.fprint(self._print)

    def _print_Real(self, expr):
        return expr.fprint(self._print)

    def _print_PythonComplex(self, expr):
        return expr.fprint(self._print)

    def _print_PythonBool(self, expr):
        return expr.fprint(self._print)

    def _print_NumpyRand(self, expr):
        if expr.rank != 0:
            errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')

        if (not self._additional_code):
            self._additional_code = ''
        var_name = self.parser.get_new_name()
        var = Variable(expr.dtype, var_name, is_stack_array = all([s.is_constant for s in expr.shape]),
                shape = expr.shape, precision = expr.precision,
                order = expr.order, rank = expr.rank)

        if self._current_function:
            name = self._current_function
            func = self.get_function(name)
            func.local_vars.append(var)
        else:
            self._namespace.variables[var.name] = var

        self._additional_code = self._additional_code + self._print(Assign(var,expr)) + '\n'
        return self._print(var)

    def _print_NumpyRandint(self, expr):
        if expr.rank != 0:
            errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')
        return expr.fprint(self._print)

    def _print_PythonMin(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'minval({0})'.format(self._print(arg))
        else:
            code = ','.join(self._print(arg) for arg in args)
            code = 'min('+code+')'
        return self._get_statement(code)

    def _print_PythonMax(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'maxval({0})'.format(self._print(arg))
        else:
            code = ','.join(self._print(arg) for arg in args)
            code = 'max('+code+')'
        return self._get_statement(code)

    def _print_Dot(self, expr):
        return self._get_statement('dot_product(%s,%s)'%(self._print(expr.expr_l), self._print(expr.expr_r)))

    def _print_Ceil(self, expr):
        return self._get_statement('ceiling(%s)'%(self._print(expr.rhs)))

    def _print_Mod(self, expr):
        args = ','.join(self._print(i) for i in expr.args)
        return 'modulo({})'.format(args)

    def _print_Sign(self, expr):
        # TODO use the appropriate precision from rhs
        return self._get_statement('sign(1.0d0,%s)'%(self._print(expr.rhs)))

    # ... MACROS
    def _print_MacroShape(self, expr):
        var = expr.argument
        if not isinstance(var, (Variable, IndexedElement)):
            raise TypeError('Expecting a variable, given {}'.format(type(var)))
        shape = None
        if isinstance(var, Variable):
            shape = var.shape

        if shape is None:
            rank = var.rank
            shape = []
            for i in range(0, rank):
                l = 'lbound({var},{i})'.format(var=self._print(var),
                                               i=self._print(i+1))
                u = 'ubound({var},{i})'.format(var=self._print(var),
                                               i=self._print(i+1))
                s = '{u}-{l}+1'.format(u=u, l=l)
                shape.append(s)

        if len(shape) == 1:
            shape = shape[0]


        elif not(expr.index is None):
            if expr.index < len(shape):
                shape = shape[expr.index]
            else:
                shape = '1'

        code = '{}'.format(self._print(shape))

        return self._get_statement(code)
    # ...
    def _print_MacroType(self, expr):
        dtype = self._print(expr.argument.dtype)
        prec  = expr.argument.precision

        if dtype == 'integer':
            if prec==4:
                return 'MPI_INTEGER'
            elif prec==8:
                return 'MPI_INTEGER8'
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')

        elif dtype == 'real':
            if prec==8:
                return 'MPI_DOUBLE'
            if prec==4:
                return 'MPI_FLOAT'
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')

        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

    def _print_MacroCount(self, expr):

        var = expr.argument
        #TODO calculate size when type is pointer
        # it must work according to fortran documentation
        # but it raises somehow an error when it's a pointer
        # and shape is None

        if isinstance(var, Variable):
            shape = var.shape
            if not isinstance(shape,(tuple,list,Tuple)):
                shape = [shape]
            rank = len(shape)
            if shape is None:
                return 'size({})'.format(self._print(var))


        elif isinstance(var, IndexedElement):
            _shape = var.base.shape
            if _shape is None:
                return 'size({})'.format(self._print(var))

            shape = []
            for (s, i) in zip(_shape, var.indices):
                if isinstance(i, Slice):
                    if i.start is None and i.end is None:
                        shape.append(s)
                    elif i.start is None:
                        if (isinstance(i.end, (int, Integer)) and i.end>0) or not(isinstance(i.end, (int, Integer))):
                            shape.append(i.end)
                    elif i.end is None:
                        if (isinstance(i.start, (int, Integer)) and i.start<s-1) or not(isinstance(i.start, (int, Integer))):
                            shape.append(s-i.start)
                    else:
                        shape.append(i.end-i.start+1)

            rank = len(shape)

        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        if rank == 0:
            return '1'

        return str(functools.reduce(operator.mul, shape ))

    def _print_Declare(self, expr):
        # ... ignored declarations
        # we don't print the declaration if iterable object
        if is_iterable_datatype(expr.dtype):
            return ''

        if is_with_construct_datatype(expr.dtype):
            return ''

        if isinstance(expr.dtype, NativeSymbol):
            return ''

        if isinstance(expr.dtype, (NativeRange, NativeTensor)):
            return ''

        # meta-variables
        if (isinstance(expr.variable, Variable) and
              str(expr.variable.name).startswith('__')):
            return ''
        # ...

        if isinstance(expr.variable, TupleVariable) and not expr.variable.is_homogeneous:
            return ''.join(self._print_Declare(Declare(v.dtype,v,intent=expr.intent, static=expr.static)) for v in expr.variable)

        # ... TODO improve
        # Group the variables by intent
        var = expr.variable
        rank        = var.rank
        allocatable = var.allocatable
        shape       = var.alloc_shape
        is_pointer = var.is_pointer
        is_target = var.is_target
        is_const = var.is_const
        is_stack_array = var.is_stack_array
        is_polymorphic = var.is_polymorphic
        is_optional = var.is_optional
        is_static = expr.static
        intent = expr.intent

        if isinstance(shape, (tuple,PythonTuple)) and len(shape) ==1:
            shape = shape[0]
        # ...

        # ... print datatype
        if isinstance(expr.dtype, CustomDataType):
            dtype = expr.dtype

            name   = dtype.__class__.__name__
            prefix = dtype.prefix
            alias  = dtype.alias

            if not is_polymorphic:
                sig = 'type'
            elif dtype.is_polymorphic:
                sig = 'class'
            else:
                sig = 'type'

            if alias is None:
                name = name.replace(prefix, '')
            else:
                name = alias
            dtype = '{0}({1})'.format(sig, name)
        else:
            if isinstance(expr.dtype, NativeTuple):
                # Non-homogenous NativeTuples must be stored in TupleVariable
                if not expr.variable.is_homogeneous:
                    errors.report(LIST_OF_TUPLES,
                                  symbol=expr.variable, severity='error')
                    expr_dtype = NativeInteger()
                else:
                    expr_dtype = expr.variable.homogeneous_dtype
            else:
                expr_dtype = expr.dtype
            dtype = self._print(expr_dtype)

        # ...
            if isinstance(expr_dtype, NativeString):

                if expr.intent:
                    dtype = dtype[:9] +'(len =*)'
                    #TODO improve ,this is the case of character as argument
            else:
                dtype += '(kind={0})'.format(str(expr.variable.precision))

        code_value = ''
        if expr.value:
            code_value = ' = {0}'.format(expr.value)

        vstr = self._print(expr.variable.name)

        # arrays are 0-based in pyccel, to avoid ambiguity with range
        s = '0'
        if not(is_static) and (allocatable or (var.shape is None)):
            s = ''

        # Default empty strings
        intentstr      = ''
        allocatablestr = ''
        optionalstr    = ''
        rankstr        = ''

        # Compute intent string
        if intent:
            if intent == 'in' and rank == 0 and is_static is False:
                intentstr = ', value'
                if is_const:
                    intentstr += ', intent(in)'
            else:
                intentstr = ', intent({})'.format(intent)

        # Compute allocatable string
        if not is_static:
            if is_pointer:
                allocatablestr = ', pointer'

            elif allocatable and not intent:
                allocatablestr = ', allocatable'

            # ISSUES #177: var is allocatable and target
            if is_target:
                allocatablestr = '{}, target'.format(allocatablestr)

        # Compute optional string
        if is_optional:
            optionalstr = ', optional'

        # Compute rank string
        # TODO: improve
        if ((rank == 1) and (isinstance(shape, (int, Integer, Variable, PyccelAdd))) and
            (not(allocatable or is_pointer) or is_static or is_stack_array)):
            rankstr = '({0}:{1}-1)'.format(self._print(s), self._print(shape))

        elif ((rank > 0) and (isinstance(shape, (PythonTuple, Tuple, tuple))) and
            (not(allocatable or is_pointer) or is_static or is_stack_array)):
            #TODO fix bug when we include shape of type list

            if var.order == 'C':
                rankstr =  ','.join('{0}:{1}-1'.format(self._print(s),
                                                    self._print(i)) for i in shape[::-1])
            else:
                rankstr =  ','.join('{0}:{1}-1'.format(self._print(s),
                                                     self._print(i)) for i in shape)
            rankstr = '({rank})'.format(rank=rankstr)

        elif (rank > 0) and allocatable and intent:
            rankstr = '({})'.format(','.join(['0:'] * rank))

        elif (rank > 0) and (allocatable or is_pointer):
            rankstr = '({})'.format(','.join( [':'] * rank))

#        else:
#            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
#                severity='fatal')

        # Construct declaration
        left  = dtype + intentstr + allocatablestr + optionalstr
        right = vstr + rankstr + code_value
        return '{} :: {}\n'.format(left, right)

    def _print_AliasAssign(self, expr):
        code = ''
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(rhs, VariableAddress):
            rhs = rhs.variable

        if isinstance(lhs, TupleVariable) and not lhs.is_homogeneous:
            if isinstance(rhs, (TupleVariable, PythonTuple)):
                return self._print(CodeBlock([AliasAssign(l, rhs[i]) for i,l in enumerate(lhs)]))
            else:
                return self._print(CodeBlock([AliasAssign(l, IndexedVariable(rhs)[i]) for i,l in enumerate(lhs)]))

        if isinstance(rhs, Dlist):
            pattern = 'allocate({lhs}(0:{length}-1))\n{lhs} = {init_value}\n'
            code = pattern.format(lhs=self._print(lhs),
                                  length=self._print(rhs.length),
                                  init_value=self._print(rhs.val))
            return code

        # TODO improve
        op = '=>'
        shape_code = ''
        if lhs.rank > 0:
            shape_code = ', '.join('0:' for i in range(lhs.rank))
            shape_code = '({s_c})'.format(s_c = shape_code)

        code += '{lhs}{s_c} {op} {rhs}'.format(lhs=self._print(expr.lhs),
                                          s_c = shape_code,
                                          op=op,
                                          rhs=self._print(expr.rhs))

        return self._get_statement(code) + '\n'

    def _print_CodeBlock(self, expr):
        body = []
        for b in expr.body:
            line = self._print(b)
            if (self._additional_code):
                body.append(self._additional_code)
                self._additional_code = None
            body.append(line)
        return ''.join(body)

    # TODO the ifs as they are are, is not optimal => use elif
    def _print_SymbolicAssign(self, expr):
        errors.report(FOUND_SYMBOLIC_ASSIGN,
                      symbol=expr.lhs, severity='warning')

        stmt = Comment(str(expr))
        return self._print_Comment(stmt)


    def _print_Assign(self, expr):
        if isinstance(expr.lhs, TupleVariable) and not expr.lhs.is_homogeneous \
            and isinstance(expr.rhs, (PythonTuple,TupleVariable)):
            return '\n'.join(self._print_Assign(
                        Assign(lhs,
                                rhs,
                                strict=expr.strict,
                                status=expr.status,
                                like=expr.like,
                                )
                        ) for lhs,rhs in zip(expr.lhs,expr.rhs))

        lhs_code = self._print(expr.lhs)
        is_procedure = False
        rhs = expr.rhs
        # we don't print Range, Tensor
        # TODO treat the case of iterable classes
        if isinstance(rhs, NINF):
            rhs_code = '-Huge({0})'.format(lhs_code)
            return '{0} = {1}\n'.format(lhs_code, rhs_code)

        if isinstance(rhs, INF):
            rhs_code = 'Huge({0})'.format(lhs_code)
            return '{0} = {1}\n'.format(lhs_code, rhs_code)

        if isinstance(rhs, (PythonRange, Product)):
            return ''

        if isinstance(rhs, (PythonLen, NumpyRandint)):
            rhs_code = self._print(expr.rhs)
            return '{0} = {1}\n'.format(lhs_code, rhs_code)

        if isinstance(rhs, (PythonInt, NumpyReal, NumpyComplex)):
            lhs = self._print(expr.lhs)
            rhs = expr.rhs.fprint(self._print)
            return '{0} = {1}\n'.format(lhs,rhs)

        if isinstance(rhs, (NumpyArray, NumpyLinspace, NumpyDiag, NumpyCross,\
						NumpyWhere, PyccelArraySize)):
            return rhs.fprint(self._print, expr.lhs) + '\n'

        if isinstance(rhs, (NumpyFull, NumpyFullLike, NumpyEmptyLike,\
						NumpyZerosLike, NumpyOnesLike, NumpyRand)):

            stack_array = False
            if self._current_function:
                name = self._current_function
                func = self.get_function(name)
                lhs_name = expr.lhs.name
                vars_dict = {i.name: i for i in func.local_vars}
                if lhs_name in vars_dict:
                    stack_array = vars_dict[lhs_name].is_stack_array

            return rhs.fprint(self._print, expr.lhs, stack_array)

        if isinstance(rhs, NumpyMod):
            lhs = self._print(expr.lhs)
            args = ','.join(self._print(i) for i in rhs.args)
            rhs  = 'modulo({})'.format(args)
            return '{0} = {1}\n'.format(lhs, rhs)

        # TODO [YG, 10.03.2020]: I have just commented out this block and
        # everything still seems to work; is it dead code?
#        if isinstance(rhs, FunctionDef):
#            rhs_code = self._print(rhs.name)
#            is_procedure = rhs.is_procedure

        if isinstance(rhs, ConstructorCall):
            func = rhs.func
            name = str(func.name)

            # TODO uncomment later

#            # we don't print the constructor call if iterable object
#            if this.dtype.is_iterable:
#                return ''
#
#            # we don't print the constructor call if with construct object
#            if this.dtype.is_with_construct:
#                return ''

            if name == "__init__":
                name = "create"
            rhs_code = self._print(name)
            rhs_code = '{0} % {1}'.format(lhs_code, rhs_code)
            #TODO use is_procedure property
            is_procedure = (rhs.kind == 'procedure')

            code_args = ', '.join(self._print(i) for i in rhs.arguments)
            return 'call {0}({1})\n'.format(rhs_code, code_args)

        if isinstance(rhs, FunctionCall):

            # in the case of a function that returns a list,
            # we should append them to the procedure arguments
            if isinstance(expr.lhs, (tuple, list, Tuple, PythonTuple)):

                rhs_code = self._print(rhs.func)
                args = rhs.arguments
                code_args = [self._print(i) for i in args]
                func = rhs.funcdef
                output_names = func.results
                lhs_code = [self._print(name) + ' = ' + self._print(i) for (name,i) in zip(output_names,expr.lhs)]

                call_args = ', '.join(code_args + lhs_code)

                code = 'call {0}({1})\n'.format(rhs_code, call_args)
                return self._get_statement(code)

        if (isinstance(expr.lhs, Variable) and
              expr.lhs.dtype == NativeSymbol()):
            return ''

        # Right-hand side code
        rhs_code = self._print(rhs)

        code = ''
        # if (expr.status == 'unallocated') and not (expr.like is None):
        #     stmt = ZerosLike(lhs=lhs_code, rhs=expr.like)
        #     code += self._print(stmt)
        #     code += '\n'
        if not is_procedure:
            code += '{0} = {1}'.format(lhs_code, rhs_code)
#        else:
#            code_args = ''
#            func = expr.rhs
#            # func here is of instance FunctionCall
#            cls_name = func.func.cls_name
#            keys = func.func.arguments

#            # for MPI statements, we need to add the lhs as the last argument
#            # TODO improve
#            if isinstance(func.func, MPI):
#                if not func.arguments:
#                    code_args = lhs_code
#                else:
#                    code_args = ', '.join(self._print(i) for i in func.arguments)
#                    code_args = '{0}, {1}'.format(code_args, lhs_code)
#            else:
#                _ij_print = lambda i, j: '{0}={1}'.format(self._print(i), \
#                                                         self._print(j))
#
#                code_args = ', '.join(_ij_print(i, j) \
#                                      for i, j in zip(keys, func.arguments))
#            if (not func.arguments is None) and (len(func.arguments) > 0):
#                if (not cls_name):
#                    code_args = ', '.join(self._print(i) for i in func.arguments)
#                    code_args = '{0}, {1}'.format(code_args, lhs_code)
#                else:
#            print('code_args > {0}'.format(code_args))
#            code = 'call {0}({1})'.format(rhs_code, code_args)
        return self._get_statement(code) + '\n'

#------------------------------------------------------------------------------
    def _print_Allocate(self, expr):

        # Transpose indices because of Fortran column-major ordering
        shape = expr.shape if expr.order == 'F' else expr.shape[::-1]

        var_code = self._print(expr.variable)
        size_code = ', '.join(self._print(i) for i in shape)
        shape_code = ', '.join('0:' + self._print(PyccelMinus(i, Integer(1))) for i in shape)
        code = ''

        if expr.status == 'unallocated':
            code += 'allocate({0}({1}))\n'.format(var_code, shape_code)

        elif expr.status == 'unknown':
            code += 'if (allocated({})) then\n'.format(var_code)
            code += '  if (any(size({}) /= [{}])) then\n'.format(var_code, size_code)
            code += '    deallocate({})\n'     .format(var_code)
            code += '    allocate({0}({1}))\n'.format(var_code, shape_code)
            code += '  end if\n'
            code += 'else\n'
            code += '  allocate({0}({1}))\n'.format(var_code, shape_code)
            code += 'end if\n'

        elif expr.status == 'allocated':
            code += 'if (any(size({}) /= [{}])) then\n'.format(var_code, size_code)
            code += '  deallocate({})\n'     .format(var_code)
            code += '  allocate({0}({1}))\n'.format(var_code, shape_code)
            code += 'end if\n'

        return code

#------------------------------------------------------------------------------

    def _print_NativeBool(self, expr):
        return 'logical'

    def _print_NativeInteger(self, expr):
        return 'integer'

    def _print_NativeReal(self, expr):
        return 'real'

    def _print_NativeComplex(self, expr):
        return 'complex'

    def _print_NativeString(self, expr):
        return 'character(len=280)'
        #TODO fix improve later

    def _print_DataType(self, expr):
        return self._print(expr.name)

    def _print_BooleanTrue(self, expr):
        return '.True.'

    def _print_BooleanFalse(self, expr):
        return '.False.'

    def _print_String(self, expr):
        sp_chars = ['\a', '\b', '\f', '\r', '\t', '\v', "'", '\n']
        sub_str = ''
        formatted_str = "''"
        for c in expr.arg:
            if c in sp_chars:
                if sub_str != '':
                    formatted_str += " // '{}'".format(sub_str)
                    sub_str = ''
                formatted_str += ' // ACHAR({})'.format(ord(c))
            else:
                sub_str += c
        if sub_str != '':
            formatted_str += " // '{}'".format(sub_str)
        return formatted_str

    def _print_Interface(self, expr):
        # ... we don't print 'hidden' functions
        name = self._print(expr.name)
        if expr.is_argument:
            funcs_sigs = []
            for f in expr.functions:
                self._handle_fortran_specific_a_prioris(list(f.arguments) + list(f.results))
                parts = self.function_signature(f, f.name)
                parts = ["{}({}) {}\n".format(parts['sig'], parts['arg_code'], parts['func_end']),
                parts['arg_decs'],
                'end {} {}\n'.format(parts['func_type'], f.name)]
                funcs_sigs.append(''.join(a for a in parts))
            interface = 'interface\n' + '\n'.join(a for a in funcs_sigs) + 'end interface\n'
            return interface

        if expr.functions[0].cls_name:
            for k, m in list(_default_methods.items()):
                name = name.replace(k, m)
            cls_name = expr.cls_name
            if not (cls_name == '__UNDEFINED__'):
                name = '{0}_{1}'.format(cls_name, name)
        else:
            for i in _default_methods:
                # because we may have a class Point with init: Point___init__
                if i in name:
                    name = name.replace(i, _default_methods[i])
        interface = 'interface ' + name +'\n'
        for f in expr.functions:
            interface += 'module procedure ' + str(f.name)+'\n'
        interface += 'end interface\n'
        return interface



   # def _print_With(self, expr):
   #     test = 'call '+self._print(expr.test) + '%__enter__()'
   #     body = self._print(expr.body)
   #     end = 'call '+self._print(expr.test) + '%__exit__()'
   #     code = ('{test}\n'
   #            '{body}\n'
   #            '{end}').format(test=test, body=body, end=end)
        #TODO return code later
  #      expr.block
  #      return ''

    def _print_Block(self, expr):

        decs=[]
        for i in expr.variables:
            dec = Declare(i.dtype, i)
            decs += [dec]
        body = expr.body

        body_code = self._print(body)
        prelude   = ''.join(self._print(i) for i in decs)


        #case of no local variables
        if len(decs) == 0:
            return body_code

        return ('{name} : Block\n'
                '{prelude}\n'
                 '{body}\n'
                'end Block {name}\n').format(name=expr.name, prelude=prelude, body=body_code)

    def _print_F2PYFunctionDef(self, expr):
        name = self._print(expr.name)
        results   = list(expr.results)
        arguments = list(expr.arguments)
        arguments_inout = expr.arguments_inout
        args_decs = OrderedDict()
        for i,arg in enumerate(arguments):
            if arguments_inout[i]:
                intent='inout'
            else:
                intent='in'

            if arg in results:
                results.remove(i)

            dec = Declare(arg.dtype, arg, intent=intent , static=True)
            args_decs[str(arg.name)] = dec

        for result in results:
            dec = Declare(result.dtype, result, intent='out', static=True)
            args_decs[str(result)] = dec

        if expr.is_procedure:
            func_type = 'subroutine'
            func_end  = ''
        else:
            func_type = 'function'
            result = results.pop()
            func_end = 'result({0})'.format(result.name)
            dec = Declare(result.dtype, result, static=True)
            args_decs[str(result.name)] = dec
        # ...

        interfaces = '\n'.join(self._print(i) for i in expr.interfaces)
        arg_code  = ', '.join(self._print(i) for i in chain( arguments, results ))
        imports   = ''.join(self._print(i) for i in expr.imports)
        prelude   = ''.join(self._print(i) for i in args_decs.values())
        body_code = self._print(expr.body)

        parts = ['{0} {1}({2}) {3}\n'.format(func_type, name, arg_code, func_end),
                 imports,
                'implicit none\n',
                 prelude,
                 interfaces,
                 body_code,
                 'end {} {}\n'.format(func_type, name)]

        return '\n'.join(parts)

    def _print_FunctionAddress(self, expr):
        return expr.name

    def function_signature(self, expr, name):
        is_pure      = expr.is_pure
        is_elemental = expr.is_elemental
        out_args = []
        args_decs = OrderedDict()

        for j, i in enumerate(expr.results):
            if not i.name:
                i.rename('out_{}'.format(j))
        for j, i in enumerate(expr.arguments):
            if not i.name:
                i.rename('in_{}'.format(j))

        is_procedure = expr.is_procedure
        func_end  = ''
        rec = 'recursive' if expr.is_recursive else ''
        if is_procedure:
            func_type = 'subroutine'
            out_args = list(expr.results)
            for result in out_args:
                if result in expr.arguments:
                    dec = Declare(result.dtype, result, intent='inout')
                else:
                    dec = Declare(result.dtype, result, intent='out')
                args_decs[str(result)] = dec

            functions = expr.functions

        else:
           #todo: if return is a function
            func_type = 'function'
            result = expr.results[0]
            functions = expr.functions

            func_end = 'result({0})'.format(result.name)

            dec = Declare(result.dtype, result)
            args_decs[str(result)] = dec
        # ...

        for i,arg in enumerate(expr.arguments):
            if isinstance(arg, Variable):
                if expr.arguments_inout[i]:
                    dec = Declare(arg.dtype, arg, intent='inout')
                elif str(arg) == 'self':
                    dec = Declare(arg.dtype, arg, intent='inout')
                else:
                    dec = Declare(arg.dtype, arg, intent='in')
                args_decs[str(arg)] = dec

        #remove parametres intent(inout) from out_args to prevent repetition
        for i in expr.arguments:
            if i in out_args:
                out_args.remove(i)

        # treate case of pure function
        sig = '{0} {1} {2}'.format(rec, func_type, name)
        if is_pure:
            sig = 'pure {}'.format(sig)

        # treate case of elemental function
        if is_elemental:
            sig = 'elemental {}'.format(sig)

        arg_code  = ', '.join(self._print(i) for i in chain( expr.arguments, out_args ))

        arg_decs = ''.join(self._print(i) for i in args_decs.values())

        parts = {
                'sig' : sig,
                'arg_code' : arg_code,
                'func_end' : func_end,
                'arg_decs' : arg_decs,
                'func_type' : func_type
        }
        return parts

    def _print_FunctionDef(self, expr):
        self._handle_fortran_specific_a_prioris(list(expr.local_vars) +
                                                list(expr.arguments)  +
                                                list(expr.results))
        # ... we don't print 'hidden' functions
        if expr.hide:
            return ''

        name = self._print(expr.name)
        self.set_current_function(name)

        if expr.cls_name:
            for k, m in list(_default_methods.items()):
                name = name.replace(k, m)

            cls_name = expr.cls_name
            if not (cls_name == '__UNDEFINED__'):
                name = '{0}_{1}'.format(cls_name, name)
        else:
            for i in _default_methods:
                # because we may have a class Point with init: Point___init__
                if i in name:
                    name = name.replace(i, _default_methods[i])

        sig_parts = self.function_signature(expr, name)
        prelude = sig_parts.pop('arg_decs')
        decs = OrderedDict()
        functions = expr.functions
        func_interfaces = '\n'.join(self._print(i) for i in expr.interfaces)
        body_code = self._print(expr.body)

        for i in expr.local_vars:
            dec = Declare(i.dtype, i)
            decs[str(i)] = dec

        vars_to_print = self.parser.get_variables(self._namespace)
        for v in vars_to_print:
            if (v not in expr.local_vars) and (v not in expr.results) and (v not in expr.arguments):
                decs[str(v)] = Declare(v.dtype,v)
        prelude += ''.join(self._print(i) for i in decs.values())
        if len(functions)>0:
            functions_code = '\n'.join(self._print(i) for  i in functions)
            body_code = body_code +'\ncontains\n' + functions_code

        imports = ''.join(self._print(i) for i in expr.imports)

        self.set_current_function(None)

        parts = parts = ["{}({}) {}\n".format(sig_parts['sig'], sig_parts['arg_code'], sig_parts['func_end']),
                imports,
                'implicit none\n',
                prelude,
                func_interfaces,
                body_code,
                'end {} {}\n'.format(sig_parts['func_type'], name)]

        return '\n'.join(a for a in parts if a)

    def _print_Pass(self, expr):
        return ''

    def _print_Nil(self, expr):
        return ''

    def _print_Return(self, expr):
        code = ''
        if expr.stmt:
            code += self._print(expr.stmt)
        code +='return\n'
        return code

    def _print_Del(self, expr):
        # TODO: treate class case
        code = ''
        for var in expr.variables:
            if isinstance(var, Variable):
                dtype = var.dtype
                if is_pyccel_datatype(dtype):
                    code = 'call {0} % free()'.format(self._print(var))
                else:
                    code = 'deallocate({0}){1}'.format(self._print(var), code)
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')
        return code + '\n'

    def _print_ClassDef(self, expr):
        # ... we don't print 'hidden' classes
        if expr.hide:
            return '', ''
        # ...

        name = self._print(expr.name)
        base = None # TODO: add base in ClassDef

        decs = ''.join(self._print(Declare(i.dtype, i)) for i in expr.attributes)

        aliases = []
        names   = []
        ls = [self._print(i.name) for i in expr.methods]
        for i in ls:
            j = _default_methods.get(i,i)
            aliases.append(j)
            names.append('{0}_{1}'.format(name, self._print(j)))
        methods = ''.join('procedure :: {0} => {1}\n'.format(i, j) for i, j in zip(aliases, names))
        for i in expr.interfaces:
            names = ','.join('{0}_{1}'.format(name, self._print(j.name)) for j in i.functions)
            methods += 'generic, public :: {0} => {1}\n'.format(self._print(i.name), names)
            methods += 'procedure :: {0}\n'.format(names)



        options = ', '.join(i for i in expr.options)

        sig = 'type, {0}'.format(options)
        if not(base is None):
            sig = '{0}, extends({1})'.format(sig, base)

        code = ('{0} :: {1}').format(sig, name)
        if len(decs) > 0:
            code = ('{0}\n'
                    '{1}').format(code, decs)
        if len(methods) > 0:
            code = ('{0}\n'
                    'contains\n'
                    '{1}').format(code, methods)
        decs = ('{0}\n'
                'end type {1}').format(code, name)

        sep = self._print(SeparatorComment(40))
        # we rename all methods because of the aliasing
        cls_methods = [i.clone('{0}'.format(i.name)) for i in expr.methods]
        for i in expr.interfaces:
            cls_methods +=  [j.clone('{0}'.format(j.name)) for j in i.functions]


        methods = ''
        for i in cls_methods:
            methods = ('{methods}\n'
                     '{sep}\n'
                     '{f}\n'
                     '{sep}\n').format(methods=methods, sep=sep, f=self._print(i))

        return decs, methods

    def _print_Break(self, expr):
        return 'exit\n'

    def _print_Continue(self, expr):
        return 'cycle\n'

    def _print_AugAssign(self, expr):
        lhs    = expr.lhs
        op     = expr.op
        rhs    = expr.rhs
        strict = expr.strict
        status = expr.status
        like   = expr.like

        if isinstance(op, AddOp):
            rhs = PyccelAdd(lhs, rhs)
        elif isinstance(op, MulOp):
            rhs = PyccelMul(lhs, rhs)
        elif isinstance(op, SubOp):
            rhs = PyccelMinus(lhs, rhs)
        # TODO fix bug with division of integers
        elif isinstance(op, DivOp):
            rhs = PyccelDiv(lhs, rhs)
        else:
            raise ValueError('Unrecognized operation', op)

        stmt = Assign(lhs, rhs, strict=strict, status=status, like=like)
        return self._print_Assign(stmt)

    def _print_PythonRange(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop) + '-' + self._print(Integer(1))
        step  = self._print(expr.step)
        return '{0}, {1}, {2}'.format(start, stop, step)

    def _print_Tile(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        return '{0}, {1}'.format(start, stop)


    def _print_ForAll(self, expr):

        start = self._print(expr.iter.start)
        end   = self._print(expr.iter.stop)
        body  = ''.join(self._print(i) for i in expr.body)
        mask  = self._print(expr.mask)
        ind   = self._print(expr.target)

        code = 'forall({ind} = {start}:{end}, {mask})\n'
        code = code.format(ind=ind,start=start,end=end,mask=mask)
        code = code + body + 'end forall\n'
        return code

    def _print_FunctionalFor(self, expr):
        loops = ''.join(self._print(i) for i in expr.loops)
        return loops

    def _print_For(self, expr):
        prolog = ''
        epilog = ''

        # ...

        def _do_range(target, iterable, prolog, epilog):
            if not isinstance(iterable, PythonRange):
                # Only iterable currently supported is PythonRange
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')

            tar        = self._print(target)
            range_code = self._print(iterable)

            prolog += 'do {0} = {1}\n'.format(tar, range_code)
            epilog = 'end do\n' + epilog

            return prolog, epilog
        # ...

        if not isinstance(expr.iterable, (PythonRange, Product , PythonZip,
                            PythonEnumerate, PythonMap)):
            # Only iterable currently supported are PythonRange or Product
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        if isinstance(expr.iterable, PythonRange):
            prolog, epilog = _do_range(expr.target, expr.iterable, \
                                       prolog, epilog)

        elif isinstance(expr.iterable, Product):
            for i, a in zip(expr.target, expr.iterable.args):
                if isinstance(a, PythonRange):
                    itr_ = a
                else:
                    itr_ = PythonRange(a.shape[0])
                prolog, epilog = _do_range(i, itr_, \
                                           prolog, epilog)

        elif isinstance(expr.iterable, PythonZip):
            itr_ = PythonRange(expr.iterable.element.shape[0])
            prolog, epilog = _do_range(expr.target, itr_, \
                                       prolog, epilog)

        elif isinstance(expr.iterable, PythonEnumerate):
            itr_ = PythonRange(PythonLen(expr.iterable.element))
            prolog, epilog = _do_range(expr.target, itr_, \
                                       prolog, epilog)

        elif isinstance(expr.iterable, PythonMap):
            itr_ = PythonRange(PythonLen(expr.iterable.args[1]))
            prolog, epilog = _do_range(expr.target, itr_, \
                                       prolog, epilog)

        body = self._print(expr.body)

        return ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)

    # .....................................................
    #                   OpenMP statements
    # .....................................................
    def _print_OMP_Parallel_Construct(self, expr):
        omp_expr   = str(expr.txt)
        ompexpr = '!$omp {}\n'.format(omp_expr)
        return ompexpr

    def _print_Omp_End_Clause(self, expr):
        omp_expr = str(expr.txt)
        ompexpr = '!$omp {}\n'.format(omp_expr)
        return ompexpr

    def _print_OMP_Single_Construct(self, expr):
        omp_expr   = str(expr.txt)
        ompexpr = '!$omp {}\n'.format(omp_expr)
        return ompexpr

    def _print_OMP_For_Loop(self, expr):
        omp_expr   = str(expr.txt)
        return '!$omp do{}\n'.format(omp_expr)

    # .....................................................
    def _print_OMP_Parallel(self, expr):
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        body    = ''.join(self._print(i) for i in expr.body)

        # ... TODO adapt get_statement to have continuation with OpenMP
        prolog = '!$omp parallel {clauses}\n'.format(clauses=clauses)
        epilog = '!$omp end parallel\n'
        # ...

        # ...
        code = ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_OMP_For(self, expr):
        # ...
        loop    = self._print(expr.loop)
        clauses = ' '.join(self._print(i)  for i in expr.clauses)

        nowait  = ''
        if not(expr.nowait is None):
            nowait = 'nowait'
        # ...

        # ... TODO adapt get_statement to have continuation with OpenMP
        prolog = '!$omp do {clauses}\n'.format(clauses=clauses)
        epilog = '!$omp end do {0}\n'.format(nowait)
        # ...

        # ...
        code = ('{prolog}'
                '{loop}'
                '{epilog}').format(prolog=prolog, loop=loop, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_OMP_NumThread(self, expr):
        return 'num_threads({})'.format(self._print(expr.num_threads))

    def _print_OMP_Default(self, expr):
        status = expr.status
        if status:
            status = self._print(expr.status)
        else:
            status = ''
        return 'default({})'.format(status)

    def _print_OMP_ProcBind(self, expr):
        status = expr.status
        if status:
            status = self._print(expr.status)
        else:
            status = ''
        return 'proc_bind({})'.format(status)

    def _print_OMP_Private(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'private({})'.format(args)

    def _print_OMP_Shared(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'shared({})'.format(args)

    def _print_OMP_FirstPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'firstprivate({})'.format(args)

    def _print_OMP_LastPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'lastprivate({})'.format(args)

    def _print_OMP_Copyin(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyin({})'.format(args)

    def _print_OMP_Reduction(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        op   = self._print(expr.operation)
        return "reduction({0}: {1})".format(op, args)

    def _print_OMP_Schedule(self, expr):
        kind = self._print(expr.kind)

        chunk_size = ''
        if expr.chunk_size:
            chunk_size = ', {0}'.format(self._print(expr.chunk_size))

        return 'schedule({0}{1})'.format(kind, chunk_size)

    def _print_OMP_Ordered(self, expr):
        n_loops = ''
        if expr.n_loops:
            n_loops = '({0})'.format(self._print(expr.n_loops))

        return 'ordered{0}'.format(n_loops)

    def _print_OMP_Collapse(self, expr):
        n_loops = '{0}'.format(self._print(expr.n_loops))

        return 'collapse({0})'.format(n_loops)

    def _print_OMP_Linear(self, expr):
        variables= ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        step = self._print(expr.step)
        return "linear({0}: {1})".format(variables, step)

    def _print_OMP_If(self, expr):
        return 'if({})'.format(self._print(expr.test))
    # .....................................................

    # .....................................................
    #                   OpenACC statements
    # .....................................................
    def _print_ACC_Parallel(self, expr):
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        body    = ''.join(self._print(i) for i in expr.body)

        # ... TODO adapt get_statement to have continuation with OpenACC
        prolog = '!$acc parallel {clauses}\n'.format(clauses=clauses)
        epilog = '!$acc end parallel\n'
        # ...

        # ...
        code = ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_ACC_For(self, expr):
        # ...
        loop    = self._print(expr.loop)
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        # ...

        # ... TODO adapt get_statement to have continuation with OpenACC
        prolog = '!$acc loop {clauses}\n'.format(clauses=clauses)
        epilog = '!$acc end loop\n'
        # ...

        # ...
        code = ('{prolog}'
                '{loop}'
                '{epilog}').format(prolog=prolog, loop=loop, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_ACC_Async(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'async({})'.format(args)

    def _print_ACC_Auto(self, expr):
        return 'auto'

    def _print_ACC_Bind(self, expr):
        return 'bind({})'.format(self._print(expr.variable))

    def _print_ACC_Collapse(self, expr):
        return 'collapse({0})'.format(self._print(expr.n_loops))

    def _print_ACC_Copy(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copy({})'.format(args)

    def _print_ACC_Copyin(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyin({})'.format(args)

    def _print_ACC_Copyout(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyout({})'.format(args)

    def _print_ACC_Create(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'create({})'.format(args)

    def _print_ACC_Default(self, expr):
        return 'default({})'.format(self._print(expr.status))

    def _print_ACC_DefaultAsync(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'default_async({})'.format(args)

    def _print_ACC_Delete(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'delete({})'.format(args)

    def _print_ACC_Device(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device({})'.format(args)

    def _print_ACC_DeviceNum(self, expr):
        return 'collapse({0})'.format(self._print(expr.n_device))

    def _print_ACC_DevicePtr(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'deviceptr({})'.format(args)

    def _print_ACC_DeviceResident(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device_resident({})'.format(args)

    def _print_ACC_DeviceType(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device_type({})'.format(args)

    def _print_ACC_Finalize(self, expr):
        return 'finalize'

    def _print_ACC_FirstPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'firstprivate({})'.format(args)

    def _print_ACC_Gang(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'gang({})'.format(args)

    def _print_ACC_Host(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'host({})'.format(args)

    def _print_ACC_If(self, expr):
        return 'if({})'.format(self._print(expr.test))

    def _print_ACC_Independent(self, expr):
        return 'independent'

    def _print_ACC_Link(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'link({})'.format(args)

    def _print_ACC_NoHost(self, expr):
        return 'nohost'

    def _print_ACC_NumGangs(self, expr):
        return 'num_gangs({0})'.format(self._print(expr.n_gang))

    def _print_ACC_NumWorkers(self, expr):
        return 'num_workers({0})'.format(self._print(expr.n_worker))

    def _print_ACC_Present(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'present({})'.format(args)

    def _print_ACC_Private(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'private({})'.format(args)

    def _print_ACC_Reduction(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        op   = self._print(expr.operation)
        return "reduction({0}: {1})".format(op, args)

    def _print_ACC_Self(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'self({})'.format(args)

    def _print_ACC_Seq(self, expr):
        return 'seq'

    def _print_ACC_Tile(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'tile({})'.format(args)

    def _print_ACC_UseDevice(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'use_device({})'.format(args)

    def _print_ACC_Vector(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'vector({})'.format(args)

    def _print_ACC_VectorLength(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'vector_length({})'.format(self._print(expr.n))

    def _print_ACC_Wait(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'wait({})'.format(args)

    def _print_ACC_Worker(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'worker({})'.format(args)
    # .....................................................

    def _print_ForIterator(self, expr):
        return self._print_For(expr)

        prolog = ''
        epilog = ''

        # ...
        def _do_range(target, iterable, prolog, epilog):
            tar        = self._print(target)
            range_code = self._print(iterable)

            prolog += 'do {0} = {1}\n'.format(tar, range_code)
            epilog = 'end do\n' + epilog

            return prolog, epilog
        # ...

        # ...
        if not isinstance(expr.iterable, (Variable, ConstructorCall)):
            raise TypeError('iterable must be Variable or ConstructorCall.')
        # ...

        # ...
        targets = expr.target
        if isinstance(expr.iterable, Variable):
            iters = expr.ranges
        elif isinstance(expr.iterable, ConstructorCall):
            iters = get_iterable_ranges(expr.iterable)
        # ...

        # ...
        for i,a in zip(targets, iters):
            prolog, epilog = _do_range(i, a, \
                                       prolog, epilog)

        body = ''.join(self._print(i) for i in expr.body)
        # ...

        return ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)


    #def _print_Block(self, expr):
    #    body    = '\n'.join(self._print(i) for i in expr.body)
    #    prelude = '\n'.join(self._print(i) for i in expr.declarations)
    #    return prelude, body

    def _print_While(self,expr):
        body = self._print(expr.body)
        return ('do while ({test})\n'
                '{body}'
                'end do\n').format(test=self._print(expr.test), body=body)

    def _print_ErrorExit(self, expr):
        # TODO treat the case of MPI
        return 'STOP'

    def _print_Assert(self, expr):
        # we first create an If statement
        # TODO: depending on a debug flag we should print 'PASSED' or not.
        DEBUG = True

        err = ErrorExit()
        args = [(Not(expr.test), [PythonPrint(["'Assert Failed'"]), err])]

        if DEBUG:
            args.append((True, PythonPrint(["'PASSED'"])))

        stmt = If(*args)
        code = self._print(stmt)
        return self._get_statement(code)

    def _print_Is(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        a = expr.args[0]
        b = expr.args[1]

        if isinstance(expr.rhs, Nil):
            return '.not. present({})'.format(lhs)

        if (a.dtype is NativeBool() and b.dtype is NativeBool()):
            return '{} .eqv. {}'.format(lhs, rhs)

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                      symbol=expr, severity='fatal')

    def _print_IsNot(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        a = expr.args[0]
        b = expr.args[1]

        if isinstance(expr.rhs, Nil):
            return 'present({})'.format(lhs)

        if a.dtype is NativeBool() and b.dtype is NativeBool():
            return '{} .neqv. {}'.format(lhs, rhs)

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                      symbol=expr, severity='fatal')

    def _print_If(self, expr):
        # ...

        lines = []

        for i, (c, e) in enumerate(expr.args):

            if (not e) or (isinstance(e, CodeBlock) and not e.body):
                continue

            if i == 0:
                lines.append("if (%s) then\n" % self._print(c))
            elif i == len(expr.args) - 1 and c is BooleanTrue():
                lines.append("else\n")
            else:
                lines.append("else if (%s) then\n" % self._print(c))

            if isinstance(e, (list, tuple, Tuple, PythonTuple)):
                lines.extend(self._print(ee) for ee in e)
            else:
                lines.append(self._print(e))

        lines.append("end if\n")

        return ''.join(lines)

    def _print_IfTernaryOperator(self, expr):
        cond = PythonBool(expr.cond) if not isinstance(expr.cond.dtype, NativeBool) else expr.cond
        value_true = expr.value_true
        value_false = expr.value_false

        if value_true.dtype != value_false.dtype :
            try :
                cast_func = python_builtin_datatypes[str_dtype(expr.dtype)]
            except KeyError:
                errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
            value_true = cast_func(value_true) if value_true.dtype != expr.dtype else value_true
            value_false = cast_func(value_false) if value_false.dtype != expr.dtype else value_false

        cond = self._print(cond)
        value_true = self._print(value_true)
        value_false = self._print(value_false)
        return 'merge({true}, {false}, {cond})'.format(cond = cond, true = value_true, false = value_false)

    def _print_MatrixElement(self, expr):
        return "{0}({1}, {2})".format(expr.parent, expr.i + 1, expr.j + 1)

    def _print_PyccelPow(self, expr):
        base = expr.args[0]
        e    = expr.args[1]

        base_c = self._print(base)
        e_c    = self._print(e)
        return '{} ** {}'.format(base_c, e_c)

    def _print_PyccelAdd(self, expr):
        if expr.dtype is NativeString():
            return '//'.join('trim('+self._print(a)+')' for a in expr.args)
        else:
            return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        args = [self._print(a) for a in expr.args]

        if len(args) == 1:
            return '-{}'.format(args[0])
        return ' - '.join(args)

    def _print_PyccelMul(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' * '.join(a for a in args)

    def _print_PyccelDiv(self, expr):
        if all(a.dtype is NativeInteger() for a in expr.args):
            args = [PythonFloat(a) for a in expr.args]
        else:
            args = expr.args
        return ' / '.join(self._print(a) for a in args)

    def _print_PyccelMod(self, expr):
        is_real  = expr.dtype is NativeReal()

        def correct_type_arg(a):
            if is_real and a.dtype is NativeInteger():
                return PythonFloat(a)
            else:
                return a

        args = [self._print(correct_type_arg(a)) for a in expr.args]

        code = args[0]
        for c in args[1:]:
            code = 'MODULO({},{})'.format(code, c)
        return code

    def _print_PyccelFloorDiv(self, expr):

        code   = self._print(expr.args[0])
        adtype = expr.args[0].dtype
        is_real  = expr.dtype is NativeReal()
        for b in expr.args[1:]:
            bdtype    = b.dtype
            if adtype is NativeInteger() and bdtype is NativeInteger():
                b = PythonFloat(b)
            c = self._print(b)
            adtype = bdtype
            code = 'FLOOR({}/{},{})'.format(code, c, expr.precision)
            if is_real:
                code = 'real({}, {})'.format(code, expr.precision)
        return code

    def _print_PyccelRShift(self, expr):
        return 'RSHIFT({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelLShift(self, expr):
        return 'LSHIFT({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitXor(self, expr):
        if expr.dtype is NativeBool():
            return ' .neqv. '.join(self._print(a) for a in expr.args)
        return 'IEOR({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitOr(self, expr):
        if expr.dtype is NativeBool():
            return ' .or. '.join(self._print(a) for a in expr.args)
        return 'IOR({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitAnd(self, expr):
        if expr.dtype is NativeBool():
            return ' .and. '.join(self._print(a) for a in expr.args)
        return 'IAND({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelInvert(self, expr):
        return 'NOT({})'.format(self._print(expr.args[0]))

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

    def _print_PyccelAnd(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' .and. '.join(a for a in args)

    def _print_PyccelOr(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' .or. '.join(a for a in args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        a = expr.args[0].dtype
        b = expr.args[1].dtype

        if a is NativeBool() and b is NativeBool():
            return '{} .eqv. {}'.format(lhs, rhs)
        return '{0} == {1}'.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        a = expr.args[0].dtype
        b = expr.args[1].dtype

        if a is NativeBool() and b is NativeBool():
            return '{} .neqv. {}'.format(lhs, rhs)
        return '{0} /= {1}'.format(lhs, rhs)

    def _print_PyccelLt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} < {1}'.format(lhs, rhs)

    def _print_PyccelLe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} <= {1}'.format(lhs, rhs)

    def _print_PyccelGt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} > {1}'.format(lhs, rhs)

    def _print_PyccelGe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} >= {1}'.format(lhs, rhs)

    def _print_PyccelNot(self, expr):
        a = self._print(expr.args[0])
        return '.not. {}'.format(a)

    def _print_Header(self, expr):
        return ''

    def _print_ConstructorCall(self, expr):
        func = expr.func
        name = func.name
        if name == "__init__":
            name = "create"
        name = self._print(name)

        code_args = ''
        if not(expr.arguments) is None:
            code_args = ', '.join(self._print(i) for i in expr.arguments)
        code = '{0}({1})'.format(name, code_args)
        return self._get_statement(code)

    def _print_NumpyUfuncBase(self, expr):
        type_name = type(expr).__name__
        try:
            func_name = numpy_ufunc_to_fortran[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        args = [self._print(NumpyFloat(a) if a.dtype is NativeInteger() else a)\
				for a in expr.args]
        code_args = ', '.join(args)
        code = '{0}({1})'.format(func_name, code_args)
        return self._get_statement(code)

    def _print_MathFunctionBase(self, expr):
        type_name = type(expr).__name__
        try:
            func_name = math_function_to_fortran[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        args = []
        for arg in expr.args:
            if arg.dtype is not NativeReal():
                args.append(self._print(PythonFloat(arg)))
            else :
                args.append(self._print(arg))
        code_args = ', '.join(args)
        code = '{0}({1})'.format(func_name, code_args)
        return self._get_statement(code)

    def _print_MathPow(self, expr):
        base = expr.args[0]
        e    = expr.args[1]

        base_c = self._print(base)
        e_c    = self._print(e)
        return '{} ** {}'.format(base_c, e_c)

    def _print_NumpySqrt(self, expr):
        arg = expr.args[0]
        if arg.dtype is NativeInteger() or arg.dtype is NativeBool():
            arg = PythonFloat(arg)
        code_args = self._print(arg)
        code = 'sqrt({})'.format(code_args)
        return self._get_statement(code)

    def _print_MathSqrt(self, expr):
        arg = expr.args[0]
        if arg.dtype is NativeInteger() or arg.dtype is NativeBool():
            arg = PythonFloat(arg)
        code_args = self._print(arg)
        code = 'sqrt({})'.format(code_args)
        return self._get_statement(code)

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        return "cmplx(0,1)"

    def _print_int(self, expr):
        return str(expr)

    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])
        return "%sd0" % printed

    def _print_Complex(self, expr):
        real_str = self._print_Float(expr.real)
        imag_str = self._print_Float(expr.imag)
        return "({}, {})".format(real_str, imag_str)

    def _print_Integer(self, expr):
        return "{0}_{1}".format(str(expr.p), expr.precision)

    def _print_IndexedElement(self, expr):
        if isinstance(expr.base, IndexedVariable):
            base = expr.base.internal_variable
        else:
            base = expr.base
        if isinstance(base, Application) and not isinstance(base, PythonTuple):
            indexed_type = base.dtype
            if isinstance(indexed_type, PythonTuple):
                base = self._print_Function(expr.base.base)
            else:
                if (not self._additional_code):
                    self._additional_code = ''
                var_name = self.parser.get_new_name()
                var = Variable(base.dtype, var_name, is_stack_array = True,
                        shape=base.shape,precision=base.precision,
                        order=base.order,rank=base.rank)

                if self._current_function:
                    name = self._current_function
                    func = self.get_function(name)
                    func.local_vars.append(var)
                else:
                    self._namespace.variables[var.name] = var

                self._additional_code = self._additional_code + self._print(Assign(var,base)) + '\n'
                return self._print(IndexedVariable(var, dtype=base.dtype,
                   shape=base.shape,prec=base.precision,
                   order=base.order,rank=base.rank)[expr.indices])
        elif isinstance(base, TupleVariable) and not base.is_homogeneous:
            if len(expr.indices)==1:
                return self._print(base[expr.indices[0]])
            else:
                var = base[expr.indices[-1]]
                return self._print(IndexedVariable(var, dtype = var.dtype,
                    shape = var.shape, prec = var.precision,
                    order = var.order, rank = var.rank)[expr.indices[:-1]])
        else:
            base = self._print(expr.base.label)

        inds = list(expr.indices)
        base_shape = Shape(expr.base)
        allow_negative_indexes = (isinstance(expr.base, IndexedVariable) and \
                expr.base.internal_variable.allows_negative_indexes)

        for i, ind in enumerate(inds):
            if isinstance(ind, PyccelUnarySub) and isinstance(ind.args[0], Integer):
                inds[i] = PyccelMinus(base_shape[i], ind.args[0])
            else:
                #indices of indexedElement of len==1 shouldn't be a Tuple
                if isinstance(ind, Tuple) and len(ind) == 1:
                    inds[i] = ind[0]
                if allow_negative_indexes and not isinstance(ind, Integer):
                    inds[i] = PyccelMod(ind, base_shape[i])

        inds = [self._print(i) for i in inds]

        return "%s(%s)" % (base, ", ".join(inds))


    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Slice(self, expr):
        if expr.start is None or  isinstance(expr.start, Nil):
            start = ''
        else:
            start = self._print(expr.start)
        if (expr.end is None) or isinstance(expr.end, Nil):
            end = ''
        else:
            end = PyccelMinus(expr.end, Integer(1))
            end = self._print(end)
        return '{0}:{1}'.format(start, end)

#=======================================================================================

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
        args = [a for a in expr.arguments if not isinstance(a, Nil)]
        results = func.results

        if len(results) == 1:
            args = ['{}'.format(self._print(a)) for a in args]

            args = ', '.join(args)
            code = '{name}({args})'.format( name = str(func.name),
                                            args = args)

        elif len(results)>1:
            if (not self._additional_code):
                self._additional_code = ''
            out_vars = []
            for r in func.results:
                var_name = self.parser.get_new_name()
                var =  r.clone(name = var_name)

                if self._current_function:
                    name = self._current_function
                    func = self.get_function(name)
                    func.local_vars.append(var)
                else:
                    self._namespace.variables[var.name] = var

                out_vars.append(var)

            self._additional_code = self._additional_code + self._print(Assign(Tuple(*out_vars),expr)) + '\n'
            return self._print(Tuple(*out_vars))
        else:
            args    = ['{}'.format(self._print(a)) for a in args]
            if not func.is_header:
                results = ['{0}={0}'.format(self._print(a)) for a in results]
            else:
                results = ['{}'.format(self._print(a)) for a in results]

            newargs = ', '.join(args+results)

            code = 'call {name}({args})\n'.format( name = str(func.name),
                                                 args = newargs )
        return code

#=======================================================================================

    def _print_Application(self, expr):
        if isinstance(expr, NumpyNewArray):
            errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')
        else:
            return self._print_not_supported(expr)

#=======================================================================================

    def _pad_leading_columns(self, lines):
        result = []
        for line in lines:
            if line.startswith('!'):
                result.append("! " + line[1:].lstrip())
            else:
                result.append(line)
        return result

    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (ending with a \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # routine to find split point in a code line
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos

        # split line by line and add the splitted lines to result
        result = []
        trailing = ' &'
        for line in lines:
            if len(line)>72 and('"' in line or "'" in line or '!' in line):
                result.append(line)

            elif len(line)>72:
                # code line

                pos = split_pos_code(line, 72)
                hunk = line[:pos].rstrip()
                line = line[pos:].lstrip()
                if line:
                    hunk += trailing
                result.append(hunk)
                while len(line) > 0:
                    pos = split_pos_code(line, 65)
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                    if line:
                        hunk += trailing
                    result.append("%s%s"%("      " , hunk))
            else:
                result.append(line)

        # make sure that all lines end with a carriage return
        return [l if l.endswith('\n') else l+'\n' for l in result]

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        code = [line.lstrip(' \t') for line in code]

        inc_keyword = ('do ', 'if(', 'if ', 'do\n',
                       'else', 'type', 'subroutine', 'function',
                       'interface')
        dec_keyword = ('end do', 'enddo', 'end if', 'endif',
                       'else', 'endtype', 'end type',
                       'endfunction', 'end function',
                       'endsubroutine', 'end subroutine',
                       'endinterface', 'end interface')

        increase = [int(any(map(line.startswith, inc_keyword)))
                     for line in code]
        decrease = [int(any(map(line.startswith, dec_keyword)))
                     for line in code]
        continuation = [int(any(map(line.endswith, ['&', '&\n'])))
                         for line in code]

        level = 0
        cont_padding = 0
        tabwidth = self._default_settings['tabwidth']
        new_code = []
        for i, line in enumerate(code):
            if line in('','\n'):
                new_code.append(line)
                continue
            level -= decrease[i]

            padding = " "*(level*tabwidth + cont_padding)

            line = "%s%s" % (padding, line)

            new_code.append(line)

            if continuation[i]:
                cont_padding = 2*tabwidth
            else:
                cont_padding = 0
            level += increase[i]

        return new_code


def fcode(expr, parser, assign_to=None, **settings):
    """Converts an expr to a string of Fortran code

    expr : Expr
        A pyccel expression to be converted.
    parser : Parser
        The parser used to collect the expression
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    """

    return FCodePrinter(parser, settings).doprint(expr, assign_to)
