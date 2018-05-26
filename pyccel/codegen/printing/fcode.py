# coding: utf-8

"""Print to F90 standard. Trying to follow the information provided at
www.fortran90.org as much as possible."""


import string
from itertools import groupby

import numpy as np

from sympy import Lambda
from sympy.core import Symbol
from sympy.core import Float, Integer
from sympy.core import S, Add, N
from sympy.core import Tuple
from sympy.core.function import Function
from sympy.core.compatibility import string_types
from sympy.printing.precedence import precedence
from sympy import Eq, Ne, true, false
from sympy import Atom, Indexed
from sympy import preorder_traversal

from sympy.utilities.iterables import iterable
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse
from sympy.logic.boolalg import And, Not, Or, true, false

from pyccel.ast import Zeros, Array, Int, Shape, Sum, Rand

from pyccel.ast.core import get_initial_value
from pyccel.ast.core import get_iterable_ranges
from pyccel.ast.core import AddOp, MulOp, SubOp, DivOp
from pyccel.ast.core import String
from pyccel.ast.core import ClassDef
from pyccel.ast.core import Nil
from pyccel.ast.core import Module
from pyccel.ast.core import SeparatorComment
from pyccel.ast.core import ConstructorCall
from pyccel.ast.core import FunctionDef, Interface
from pyccel.ast.core import FunctionCall, MethodCall , Subroutine
from pyccel.ast.core import ZerosLike
from pyccel.ast.core import Return
from pyccel.ast.core import ValuedArgument
from pyccel.ast.core import ErrorExit, Exit
from pyccel.ast.core import Range, Product, Block , Zip, Enumerate
from pyccel.ast.core import get_assigned_symbols
from pyccel.ast.core import (Assign, AugAssign, Variable, Assigns,
                             Declare, ValuedVariable,
                             Len, FunctionalFor,
                             IndexedElement, Slice, List, Dlist,
                             DottedName, AsName, DottedVariable,
                             Print, If, Nil)
from pyccel.ast.datatypes import DataType, is_pyccel_datatype
from pyccel.ast.datatypes import is_iterable_datatype, is_with_construct_datatype
from pyccel.ast.datatypes import NativeBool, NativeFloat, NativeSymbol
from pyccel.ast.datatypes import NativeComplex, NativeDouble, NativeInteger, NativeString, NativeList
from pyccel.ast.datatypes import NativeRange, NativeTensor
from pyccel.ast.datatypes import CustomDataType

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.ast.parallel.mpi     import MPI
from pyccel.ast.parallel.openmp  import OMP_For
from pyccel.ast.parallel.openacc import ACC_For

import functools
import operator


# TODO: add examples
# TODO: use _get_statement when returning a string

__all__ = ["FCodePrinter", "fcode"]

known_functions = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "log": "log",
    "exp": "exp",
    "erf": "erf",
    "Abs": "abs",
    "sign": "sign",
    "conjugate": "conjg"
}

_default_methods = {
    '__init__': 'create',
    '__del__' : 'free',
}

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


    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {0}".format(text)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    # ============ Elements ============ #


    def _print_Module(self, expr):

        name = self._print(expr.name)
        name = name.replace('.', '_')
        if not name.startswith('mod_'):
            name = 'mod_{0}'.format(name)


        imports = ''.join(self._print(i) for i in expr.imports)
        decs    = '\n'.join(self._print(i) for i in expr.declarations)
        body    = ''

        # ...
        sep = self._print(SeparatorComment(40))
        interfaces = ''
        if expr.interfaces:
            interfaces = '\n'.join(self._print(i) for i in expr.interfaces if not i.hide)
            for interface in expr.interfaces:
                if not interface.hide:
                    for i in interface.functions:
                        body = ('{body}\n'
                                '{sep}\n'
                                '{f}\n'
                                '{sep}\n').format(body=body, sep=sep, f=self._print(i))

        if expr.funcs:
            for i in expr.funcs:
                body = ('{body}\n'
                         '{sep}\n'
                         '{f}\n'
                         '{sep}\n').format(body=body, sep=sep, f=self._print(i))
        # ...

        # ...
        classes = ''
        for i in expr.classes:
            # update decs with declarations from ClassDef
            c_decs, c_funcs = self._print(i)
            decs = '{0}\n{1}'.format(decs, c_decs)
            body = '{0}\n{1}\n'.format(body, c_funcs)
        # ...


        if expr.funcs or expr.classes or expr.interfaces:
            body = '\n contains\n{0}'.format(body)

        return ('module {name}\n'
                '{imports}\n'
                'implicit none\n'
                '{decs}\n'
                '{interfaces}\n'
                '{body}\n'
                'end module\n').format(name=name,
                                       imports=imports,
                                       decs=decs,
                                       interfaces=interfaces,
                                       body=body)

    def _print_Program(self, expr):

        name = 'prog_{0}'.format(self._print(expr.name))
        name = name.replace('.', '_')
        modules = ''
        mpi = False
        #we use this to detect of we are using so that we can add
        # mpi_init and mpi_finalize in the code instruction
        # TODO should we find a better way to do this? 
        for i in expr.imports:
            if i.source=='mpi4py':
                mpi = True
            
        imports = '\n'.join(self._print(i) for i in expr.imports)
        funcs   = ''
        body    = '\n'.join(self._print(i) for i in expr.body)
        
        decs    = expr.declarations
        func_in_func = False
        for func in expr.funcs:
            for i in func.body:
                if isinstance(i, FunctionDef):
                    func_in_func = True
                    break
        if expr.classes or expr.interfaces or func_in_func:
            # TODO shall we use expr.variables? or have a more involved algo
            #      we will need to walk through the expression and see what are
            #      the variables that are needed in the definitions of classes
            variables = []
            for i in expr.interfaces:
                variables += i.functions[0].global_vars
            for i in expr.funcs:
                variables += i.global_vars
            variables =list(set(variables))
            for i in range(len(decs)):
                #remove variables that are declared in the modules
                if decs[i].variable in variables:
                    decs[i] = None
            decs = [i for i in decs if i]

            module_utils = Module(expr.name, list(variables),
                                  expr.funcs, expr.interfaces, expr.classes,
                                  imports=expr.imports)
            modules = self._print(module_utils)

            imports = ('{imports}\n'
                       'use mod_{name}\n').format(imports=imports, name=expr.name)

        else:
            # ... uncomment this later and remove it from the top
#            decs    = '\n'.join(self._print(i) for i in expr.declarations)
            # ...

            # ...
            sep = self._print(SeparatorComment(40))
            funcs = ''
            if expr.funcs:
                for i in expr.funcs:
                    funcs = ('{funcs}\n'
                             '{sep}\n'
                             '{f}\n'
                             '{sep}\n').format(funcs=funcs, sep=sep, f=self._print(i))

                funcs = 'contains\n{0}'.format(funcs)
            # ...
        decs = '\n'.join(self._print(i) for i in decs)
        if mpi:
            #TODO shuold we add them in this place or do a search to put them in the right place
            body = 'call mpi_init(ierr)\n'+'\nallocate(status(0:-1 + mpi_status_size)) \n status = 0\n'+body+'\ncall mpi_finalize(ierr)'
            
            decs = decs +'\ninteger :: ierr = -1' + '\n integer, allocatable :: status (:)'

        return ('{modules}\n'
                'program {name}\n'
                '{imports}\n'
                'implicit none\n'
                '{decs}\n'
                '{body}\n'
                '{funcs}\n'
                'end program {name}\n').format(name=name,
                                               imports=imports,
                                               decs=decs,
                                               body=body,
                                               funcs=funcs,
                                               modules=modules)

    def _print_Import(self, expr):

        prefix_as = ''
        source = None
        if expr.source is None:
            prefix = 'use'
        else:
            if isinstance(expr.source, DottedName):
                source = '_'.join(self._print(j) for j in expr.source.name)
            else:
                source = self._print(expr.source)
            prefix = 'use {}, only:'.format(source)
            prefix_as = 'use {},'.format(source)

        # TODO - improve
        # importing of pyccel extensions is not printed
        if source in ['numpy', 'scipy', 'itertools']:
            return ''
        if source == 'mpi4py':
            return 'use mpi'

        code = ''
        for i in expr.target:
            if isinstance(i, AsName):
                target = '{name} => {target}'.format(name=self._print(i.name),
                                                     target=self._print(i.target))
                line = '{prefix} {target}'.format(prefix=prefix_as,
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

            # TODO keep `\n` ?
#            code = '{code}{line}'.format(code=code, line=line)
            code = '{code}\n{line}'.format(code=code, line=line)

        # in some cases, the source is given as a string (when using metavar)
        code = code.replace("'", '')
        return self._get_statement(code)

    def _print_TupleImport(self, expr):
        code = '\n'.join(self._print(i) for i in expr.imports)
        return self._get_statement(code)


    # TODO
    def _print_FromImport(self, expr):
        fil = self._print(expr.fil)
        if isinstance(expr.fil, DottedName):
            # pyccel-extension case
            if expr.fil.name[0] == 'pyccelext':
                fil = '_'.join(self._print(i) for i in expr.fil.name)
                fil = 'mod_{0}'.format(fil)
            else:
                fil = '_'.join(self._print(i) for i in expr.fil.name)
                fil = 'mod_{0}'.format(fil)

        if not expr.funcs:
            return 'use {0}'.format(fil)
        elif isinstance(expr.funcs, str):
            funcs = self._print(expr.funcs)
            return 'use {0}, only: {1}'.format(fil, funcs)
        elif isinstance(expr.funcs, (tuple, list, Tuple)):
            funcs = ', '.join(self._print(f) for f in expr.funcs)
            return 'use {0}, only: {1}'.format(fil, funcs)
        else:
            raise TypeError('Wrong type for funcs')

    def _print_Print(self, expr):
        args = []
        for f in expr.expr:
            if isinstance(f, str):
                args.append("'{}'".format(f))
            elif isinstance(f, Tuple):
                for i in f:
                    args.append("{}".format(self._print(i)))
            else:
                args.append("{}".format(self._print(f)))

        fs = ', '.join(i for i in args)

        code = 'print *, {0}'.format(fs)
        return self._get_statement(code)

    def _print_SymbolicPrint(self, expr):
        # for every expression we will generate a print
        _iprint = lambda e: "print *, 'sympy> {}'".format(e)
        code = ''
        for a in expr.expr:
            code = '{code}{p}'.format(code=code, p=_iprint(a))
        return self._get_statement(code)


    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '! {0} '.format(txt)

    def _print_EmptyLine(self, expr):
        return ''

    def _print_NewLine(self, expr):
        return '\n'

    def _print_AnnotatedComment(self, expr):
        accel = self._print(expr.accel)
        txt   = str(expr.txt)
        return '!${0} {1}'.format(accel, txt)

    def _print_Tuple(self, expr):
        import numpy
        shape = numpy.shape(expr)
        if len(shape)>1:
            arg = functools.reduce(operator.concat, expr)
            elements = ','.join(self._print(i) for i in arg)
            return 'reshape((/ '+ elements + ' /), '+ self._print(Tuple(*shape)) + ')'
        fs = ', '.join(self._print(f) for f in expr)
        return '(/ {0} /)'.format(fs)

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_Constant(self, expr):
        return self._print(expr.value)

    def _print_ValuedArgument(self, expr):
        name = self._print(expr.name)
        value = self._print(expr.value)

        code = '{0}={1}'.format(name, value)
        return code

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

    def _print_Concatinate(self, expr):
         is_list = isinstance(expr.right, Variable) and isinstance(expr.right.dtype, NativeList)
         is_list = is_list or isinstance(expr.left, Variable) and isinstance(expr.left.dtype, NativeList)
         is_list = is_list or isinstance(expr.right, List) or isinstance(expr.left, List)
         if is_list:
             return '[' + self._print(expr.left) + ', ' + self._print(expr.right) + ']'
         else:
             #TODO imporve add more checks
             return 'trim('+ self._print(expr.left) +') // trim(' + self._print(expr.right) + ')'

    def _print_Lambda(self, expr):
        return '"{args} -> {expr}"'.format(args=expr.variables, expr=expr.expr)

    def _print_ZerosLike(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        if isinstance(expr.rhs, IndexedElement):
            shape = []
            for i in expr.rhs.indices:
                if isinstance(i, Slice):
                    shape.append(i)
            rank = len(shape)
        else:
            rank = expr.rhs.rank
        rs = []
        for i in range(1, rank+1):
            l = 'lbound({0},{1})'.format(rhs, str(i))
            u = 'ubound({0},{1})'.format(rhs, str(i))
            r = '{0}:{1}'.format(l, u)
            rs.append(r)
        shape = ', '.join(self._print(i) for i in rs)
        init_value = self._print(expr.init_value)

        code  = ('allocate({lhs}({shape}))\n'
                 '{lhs} = {init_value}').format(lhs=lhs,
                                                shape=shape,
                                                init_value=init_value)

        return self._get_statement(code)

    def _print_SumFunction(self, expr):
        return str(expr)

    def _print_Len(self, expr):
        return self._get_statement('size(%s,1)'%(self._print(expr.arg)))

    def _print_Sum(self, expr):
        return expr.fprint(self._print)

    def _print_Shape(self, expr):
        return expr.fprint(self._print)

    def _print_Zeros(self, expr):
        return expr.fprint(self._print)

    def _print_Array(self, expr):
        return expr.fprint(self._print)

    def _print_Int(self, expr):
        return expr.fprint(self._print)

    def _print_Rand(self, expr):
        return expr.fprint(self._print)

    def _print_Min(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'minval({0})'.format(self._print(arg))
        else:
            raise ValueError("Expecting one argument for the moment.")
        return self._get_statement(code)

    def _print_Max(self, expr):
        args = expr.args
        args = ','.join(self._print(arg) for arg in args)
        code = 'max({0})'.format(args)
        return self._get_statement(code)

    def _print_Dot(self, expr):
        return self._get_statement('dot_product(%s,%s)'%(self._print(expr.expr_l), self._print(expr.expr_r)))

    def _print_Ceil(self, expr):
        return self._get_statement('ceiling(%s)'%(self._print(expr.rhs)))

    def _print_Sign(self, expr):
        # TODO use the appropriate precision from rhs
        return self._get_statement('sign(1.0d0,%s)'%(self._print(expr.rhs)))

    # ... MACROS
    def _print_MacroShape(self, expr):
        var = expr.argument
        if not isinstance(var, Variable):
            raise TypeError('Expecting a variable, given {}'.format(type(var)))
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

        if not(expr.index is None):
            shape = shape[expr.index]

        code = '{}'.format(self._print(shape))
        return self._get_statement(code)
    # ...
    def _print_MacroType(self, expr):
        dtype = self._print(expr.argument.dtype)
        if dtype == 'integer':
            return 'MPI_INT'
        elif dtype == 'real(kind=8)':
            return 'MPI_DOUBLE'
        elif dtype == 'real(kind=4)':
            return 'MPI_FLOAT'
        else:
            raise NotImplementedError('TODO')

    def _print_MacroCount(self, expr):
        var = expr.argument
        if isinstance(var, Variable):
            shape = var.shape
        elif isinstance(var, IndexedElement):
            shape = []
            for (s, i) in zip(var.base.shape, var.indices):
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
            raise NotImplementedError('TODO')
        if shape is None or len(shape)==0:
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

        # ... TODO improve
        # Group the variables by intent
        var = expr.variable
        arg_types        = type(var)
        rank        = var.rank
        allocatable = var.allocatable
        shape       = var.shape
        is_pointer = var.is_pointer
        is_target = var.is_target
        is_polymorphic = var.is_polymorphic
        is_optional = var.is_optional
        is_static = expr.static

        if isinstance(shape, tuple) and len(shape) ==1:
            shape = shape[0]
        # ...

        # ... print datatype
        if isinstance(expr.dtype, CustomDataType):
            dtype = expr.dtype

            name   = dtype.__class__.__name__
            prefix = dtype.prefix
            alias  = dtype.alias

            if not var.is_polymorphic:
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
            dtype = self._print(expr.dtype)
        # ...
        if isinstance(expr.dtype, NativeString):
            if expr.intent:
                dtype = dtype[:9] +'(len =*)'
                #TODO improve ,this is the case of character as argument

        code_value = ''
        if expr.value:
            code_value = ' = {0}'.format(expr.value)

        decs = []
        intent = expr.intent
        vstr = self._print(expr.variable.name)

        # arrays are 0-based in pyccel, to avoid ambiguity with range
        s = '0'
        e = ''
        enable_alloc = True
        if not(is_static) and (allocatable or (var.shape is None)):
            s = ''

        rankstr =  ''
        # TODO improve
        if ((rank == 1) and (isinstance(shape, (int, Integer, Variable))) and
            (not(allocatable or is_pointer) or is_static)):
            rankstr =  '({0}:{1})'.format(self._print(s), self._print(shape-1))
            enable_alloc = False

        elif ((rank > 0) and (isinstance(shape, (Tuple, tuple))) and
            (not(allocatable or is_pointer) or is_static)):
            #TODO fix bug when we inclue shape of type list
            rankstr =  ','.join('{0}:{1}'.format(self._print(s),
                                                 self._print(i-1)) for i in shape)
            rankstr = '({rank})'.format(rank=rankstr)
            enable_alloc = False

        elif (rank > 0) and (allocatable or is_pointer or is_target):
            rankstr = ','.join(':' for f in range(0, rank))
            rankstr = '(' + rankstr + ')'
#        else:
#            raise NotImplementedError('Not treated yet')

        allocatablestr = ''
        if not is_static:
            if is_pointer:
                allocatablestr = ', pointer'
            elif is_target:
                allocatablestr = ', target'
            elif allocatable:
                allocatablestr = ', allocatable'

        optionalstr = ''
        if is_optional:
            optionalstr = ', optional'

        allocatablestr = allocatablestr + optionalstr

        if intent:
            decs.append('{0}, intent({1}) {2} :: {3} {4}'.
                        format(dtype, intent, allocatablestr, vstr, rankstr))
        else:
            args = [dtype, allocatablestr, vstr, rankstr, code_value]
            decs.append('{0}{1} :: {2} {3} {4}'.
                        format(*args))

        return '\n'.join(decs)

    def _print_AliasAssign(self, expr):
        code = ''
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(rhs, Dlist):
            return 'allocate({lhs}(0:{length}-1))\n {lhs} = {init_value}'.format(
            lhs = self._print(lhs),
            length=self._print(rhs.length),
            init_value=self._print(rhs.val))
        # TODO improve
        op = '=>'
        if isinstance(lhs, Variable) and (lhs.rank > 0)  and (not lhs.is_pointer or not isinstance(rhs, Atom)):
            if not isinstance(rhs, Atom) and not isinstance(rhs, Indexed):
                # case of rhs an expression and lhs is pointer we then allocate the memory for it
                for i in list(preorder_traversal(rhs)):
                    if isinstance(i, (Variable, DottedVariable)) and i.rank>0:
                        rhs = i
                        break
            #TODO improve we only need to allocate the variable without setting it to zero
            stmt = ZerosLike(lhs, rhs)
            code += self._print(stmt)
            code += '\n'
            op = '='

        code += '{lhs} {op} {rhs}'.format(lhs=self._print(expr.lhs),
                                          op=op,
                                          rhs=self._print(expr.rhs))

        return self._get_statement(code)

    def _print_Assigns(self, expr):
        return '\n'.join(self._print(i) for i in expr.stmts)

    def _print_Assign(self, expr):
        lhs_code = self._print(expr.lhs)
        is_procedure = False

        # we don't print Range, Tensor
        # TODO treat the case of iterable classes
        if isinstance(expr.rhs, (Range, Product)):
            return ''

        if isinstance(expr.rhs, (Zeros, Array, Int, Shape, Sum, Rand)):
            return expr.rhs.fprint(self._print, expr.lhs)

        elif isinstance(expr.rhs, Shape):
            # expr.rhs = Shape(a) then expr.rhs.rhs is a
            a = expr.rhs.rhs

            lhs = self._print(expr.lhs)
            rhs = self._print(a)
            if isinstance(a, IndexedElement):
                shape = []
                for i in a.indices:
                    if isinstance(i, Slice):
                        shape.append(i)
                rank = len(shape)
            else:
                rank = a.rank

            code  = 'allocate({0}(0:{1}-1)) ; {0} = 0'.format(lhs, rank)

            rs = []
            for i in range(0, rank):
                l = 'lbound({0},{1})'.format(rhs, str(i+1))
                u = 'ubound({0},{1})'.format(rhs, str(i+1))
                r = '{3}({2}) = {1}-{0}'.format(l, u, str(i), lhs)
                rs.append(r)
            sizes = '\n'.join(self._print(i) for i in rs)

            code  = '{0}\n{1}'.format(code, sizes)

            return self._get_statement(code)

        elif isinstance(expr.rhs, FunctionDef):
            rhs_code = self._print(expr.rhs.name)
            is_procedure = expr.rhs.is_procedure

        elif isinstance(expr.rhs, ConstructorCall):
            func = expr.rhs.func
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
            is_procedure = (expr.rhs.kind == 'procedure')

            code_args = ', '.join(self._print(i) for i in expr.rhs.arguments)
            return 'call {0}({1})'.format(rhs_code, code_args)
        elif isinstance(expr.rhs, Function):
            # in the case of a function that returns a list,
            # we should append them to the procedure arguments
            name = type(expr.rhs).__name__
            rhs_code = self._print(name)
            args = expr.rhs.args
            code_args = ', '.join(self._print(i) for i in args)

            if isinstance(expr.lhs, (tuple, list, Tuple)):
                lhs_code = ', '.join(self._print(i) for i in expr.lhs)
                code = 'call {0}({1}, {2})'.format(rhs_code, code_args, lhs_code)
                return self._get_statement(code)
           
            rhs_code = '{0}({1})'.format(rhs_code, code_args)
            code = '{0} = {1}'.format(lhs_code, rhs_code)

            return self._get_statement(code)

        elif (isinstance(expr.lhs, Variable) and
              expr.lhs.dtype == NativeSymbol()):
            return ''
        else:
            rhs_code = self._print(expr.rhs)
#            print("ASSIGN = ", rhs_code)

        code = ''
        if (expr.status == 'unallocated') and not (expr.like is None):
            stmt = ZerosLike(lhs_code, expr.like)
            code += self._print(stmt)
            code += '\n'
        if not is_procedure:
            code += '{0} = {1}'.format(lhs_code, rhs_code)
        else:
            code_args = ''
            func = expr.rhs
            # func here is of instance FunctionCall
            cls_name = func.func.cls_name
            keys = func.func.arguments

            # for MPI statements, we need to add the lhs as the last argument
            # TODO improve
            if isinstance(func.func, MPI):
                if not func.arguments:
                    code_args = lhs_code
                else:
                    code_args = ', '.join(self._print(i) for i in func.arguments)
                    code_args = '{0}, {1}'.format(code_args, lhs_code)
            else:
                _ij_print = lambda i, j: '{0}={1}'.format(self._print(i), \
                                                         self._print(j))

                code_args = ', '.join(_ij_print(i, j) \
                                      for i, j in zip(keys, func.arguments))
#            if (not func.arguments is None) and (len(func.arguments) > 0):
#                if (not cls_name):
#                    code_args = ', '.join(self._print(i) for i in func.arguments)
#                    code_args = '{0}, {1}'.format(code_args, lhs_code)
#                else:
            print('code_args > {0}'.format(code_args))
            code = 'call {0}({1})'.format(rhs_code, code_args)
        return self._get_statement(code)

    def _print_NativeBool(self, expr):
        return 'logical'

    def _print_NativeInteger(self, expr):
        return 'integer'

    def _print_NativeFloat(self, expr):
        return 'real'

    def _print_NativeDouble(self, expr):
        return 'real(kind=8)'

    def _print_NativeComplex(self, expr):
        # TODO add precision
        return 'complex(kind=8)'

    def _print_BooleanTrue(self, expr):
        return '.true.'

    def _print_BooleanFalse(self, expr):
        return '.false.'

    def _print_NativeString(self, expr):
        return 'character(len=280)'
        #TODO fix improve later

    def _print_DataType(self, expr):
        return self._print(expr.name)

    def _print_Equality(self, expr):
        return '{0} == {1} '.format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_Unequality(self, expr):
        return '{0} /= {1} '.format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_BooleanTrue(self, expr):
        return '.True.'

    def _print_BooleanFalse(self, expr):
        return '.False.'

    def _print_String(self, expr):
        return expr.arg

    def _print_Interface(self, expr):
        # ... we don't print 'hidden' functions
        name = self._print(expr.name)
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
        functions = []
        for f in expr.functions:
            interface += 'module procedure ' + str(f.name)+'\n'
        interface += 'end interface\n'
        return interface



   # def _print_With(self, expr):
   #     test = 'call '+self._print(expr.test) + '%__enter__()'
   #     body = '\n'.join(self._print(i) for i in expr.body)
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

        body_code = '\n'.join(self._print(i) for i in body)
        prelude   = '\n'.join(self._print(i) for i in decs)

        return ('{name} : Block\n'
                '{prelude}\n'
                 '{body}\n'
                'end Block {name}').format(name=expr.name, prelude=prelude, body=body_code)



    def _print_FunctionDef(self, expr):
        # ... we don't print 'hidden' functions
        if expr.hide:
            return ''
        # ...

        name = self._print(expr.name)
        is_static = expr.is_static

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
        out_args = []
        decs = {}

        # ... local variables declarations
        for i in expr.local_vars:
            dec = Declare(i.dtype, i)
            decs[str(i)] = dec
        # ...

        # ...
        body = expr.body
        func_end  = ''
        if not expr.is_procedure:
            result = expr.results[0]
            # TODO uncomment and validate this
#            expr = subs(expr, result, str(expr.name))

            body = []
            functions = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    pass
                elif isinstance(stmt, FunctionDef):
                    functions += [stmt]
                elif not isinstance(stmt, list): # for list of Results
                    body.append(stmt)

            ret_type = self._print(result.dtype)

            func_type = 'function'

            if result.allocatable or (result.rank > 0):
                sig = 'function {0}'.format(name)
                var = Variable(result.dtype, result.name, \
                             rank=result.rank, \
                             allocatable=True, \
                             shape=result.shape)

                dec = Declare(result.dtype, var)
                decs[str(var)] = dec
            else:
                sig = '{0} function {1}'.format(ret_type, name)
                func_end  = ' result({0})'.format(result.name)
        else:
            # TODO compute intent
            # a static function is always treated as a procedure
            #TODO improve for functions without return

            out_args = [result for result in expr.results]
            for result in expr.results:
                if result in expr.arguments:
                    dec = Declare(result.dtype, result, intent='inout', static=is_static)
                else:
                    dec = Declare(result.dtype, result, intent='out', static=is_static)
                decs[str(result)] = dec

            sig = 'subroutine ' + name
            func_type = 'subroutine'

            names = [str(res.name) for res in expr.results]
            body = []
            functions = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    pass
                elif isinstance(stmt, FunctionDef):
                    functions += [stmt]
                elif not isinstance(stmt, Return):
                    body.append(stmt)
                elif isinstance(stmt,Return):
                    body += [stmt]

        # ... TODO improve to treat variables that are assigned within blocks: if, etc
        symbols = get_assigned_symbols(expr)
        assigned_names = [str(i) for i in symbols]
        # ...

        results_names = [str(i) for i in expr.results]
        for arg in expr.arguments:
            if str(arg) in results_names + assigned_names:
                dec = Declare(arg.dtype, arg, intent='inout', static=is_static)
            elif str(arg) == 'self':
                dec = Declare(arg.dtype, arg, intent='inout', static=is_static)
            else:
                dec = Declare(arg.dtype, arg, intent='in', static=is_static)
            decs[str(arg)] = dec

        decs = [v for k,v in decs.items()]


        #remove parametres intent(inout) from out_args to prevent repetition
        for i in expr.arguments:
            if i in out_args:
                out_args.remove(i)

        out_code  = ', '.join(self._print(i) for i in out_args)

        arg_code  = ', '.join(self._print(i) for i in expr.arguments)
        if len(out_code) > 0:
            arg_code  = ', '.join(i for i in [arg_code, out_code])

        body_code = '\n'.join(self._print(i) for i in body)
        prelude   = '\n'.join(self._print(i) for i in decs)
        if len(functions)>0:
            functions_code = '\n'.join(self._print(i) for  i in functions)
            body_code = body_code +'\ncontains \n' +functions_code
        body_code = prelude + '\n\n' + body_code

        return ('{0}({1}) {2}\n'
                'implicit none\n'
#                'integer, parameter:: dp=kind(0.d0)\n'
                '{3}\n'
                'end {4}').format(sig, arg_code, func_end, body_code, func_type)

    def _print_Pass(self, expr):
        return ''
   
    def _print_Nil(self, expr):
        return ''

    def _print_Return(self, expr):
        code = ''
        if expr.stmt:
            code += self._print(expr.stmt)+'\n'
        code +='return'
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
                msg  = 'Only Variable is treated.'
                msg += ' Given {0}'.format(type(var))
                raise NotImplementedError(msg)
        return code
    def _print_ClassDef(self, expr):
        # ... we don't print 'hidden' classes
        if expr.hide:
            return '', ''
        # ...

        name = self._print(expr.name)
        base = None # TODO: add base in ClassDef

        decs = '\n'.join(self._print(Declare(i.dtype, i)) for i in expr.attributes)

        aliases = []
        names   = []
        ls = [self._print(i.name) for i in expr.methods]
        for i in ls:
            j = i
            if i in _default_methods:
                j = _default_methods[i]
            aliases.append(j)
            names.append('{0}_{1}'.format(name, self._print(j)))
        methods = '\n'.join('procedure :: {0} => {1}'.format(i, j) for i, j in zip(aliases, names))
        for i in expr.interfaces:
            names = ','.join('{0}_{1}'.format(name, self._print(j.name)) for j in i.functions)
            methods += '\ngeneric, public :: {0} => {1}'.format(self._print(i.name), names)
            methods += '\nprocedure :: {0}'.format(names)



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
        cls_methods = [i.rename('{0}'.format(i.name)) for i in expr.methods]
        for i in expr.interfaces:
            cls_methods +=  [j.rename('{0}'.format(j.name)) for j in i.functions]


        methods = ''
        for i in cls_methods:
            methods = ('{methods}\n'
                     '{sep}\n'
                     '{f}\n'
                     '{sep}\n').format(methods=methods, sep=sep, f=self._print(i))

        return decs, methods

    def _print_Break(self, expr):
        return 'exit'

    def _print_Continue(self, expr):
        return 'continue'

    def _print_AugAssign(self, expr):
        lhs    = expr.lhs
        op     = expr.op
        rhs    = expr.rhs
        strict = expr.strict
        status = expr.status
        like   = expr.like

        if isinstance(op, AddOp):
            rhs = lhs + rhs
        elif isinstance(op, MulOp):
            rhs = lhs * rhs
        elif isinstance(op, SubOp):
            rhs = lhs - rhs
        # TODO fix bug with division of integers
        elif isinstance(op, DivOp):
            rhs = lhs / rhs
        else:
            raise ValueError('Unrecongnized operation', op)

        stmt = Assign(lhs, rhs, strict=strict, status=status, like=like)
        return self._print(stmt)

    def _print_Range(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop-1)
        step  = self._print(expr.step)
        return '{0}, {1}, {2}'.format(start, stop, step)

    def _print_Tile(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        return '{0}, {1}'.format(start, stop)

    def _print_FunctionalFor(self, expr):
        allocate = ','.join('0:{0}'.format(str(i)) for i in expr.target.shape)
        allocate ='allocate({0}({1}))'.format(expr.target.name, allocate)
        loops = '\n'.join(self._print(i) for i in expr.loops)
        return allocate + '\n' + loops

    def _print_For(self, expr):
        prolog = ''
        epilog = ''

        # ...
        def _do_range(target, iter, prolog, epilog):
            if not isinstance(iter, Range):
                msg = "Only iterable currently supported is Range"
                raise NotImplementedError(msg)

            tar        = self._print(target)
            range_code = self._print(iter)

            prolog += 'do {0} = {1}\n'.format(tar, range_code)
            epilog = 'end do\n' + epilog

            return prolog, epilog
        # ...

        # ...
        def _iprint(i):
            if isinstance(i, Block):
                _prelude, _body = self._print_Block(i)
                return '{0}'.format(_body)
            else:
                return '{0}'.format(self._print(i))
        # ...

        if not isinstance(expr.iterable, (Range, Product , Zip, Enumerate)):
            msg  = "Only iterable currently supported are Range, "
            msg += "Product"
            raise NotImplementedError(msg)

        if isinstance(expr.iterable, Range):
            prolog, epilog = _do_range(expr.target, expr.iterable, \
                                       prolog, epilog)
        elif isinstance(expr.iterable, Product):
            for i, a in zip(expr.target, expr.iterable.args):
                itr_=Range(a.shape[0])
                prolog, epilog = _do_range(i, itr_, \
                                           prolog, epilog)
        elif isinstance(expr.iterable, (Zip, Enumerate)):
            itr_=Range(expr.iterable.element.shape[0])
            prolog, epilog = _do_range(expr.target, itr_, \
                                       prolog, epilog)

        body = '\n'.join(_iprint(i) for i in expr.body)

        return ('{prolog}'
                '{body}\n'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)

    # .....................................................
    #                   OpenMP statements
    # .....................................................
    def _print_OMP_Parallel(self, expr):
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        body    = '\n'.join(self._print(i) for i in expr.body)

        # ... TODO adapt get_statement to have continuation with OpenMP
        prolog = '!$omp parallel {clauses}\n'.format(clauses=clauses)
        epilog = '!$omp end parallel\n'
        # ...

        # ...
        code = ('{prolog}'
                '{body}\n'
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
                '{loop}\n'
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
        body    = '\n'.join(self._print(i) for i in expr.body)

        # ... TODO adapt get_statement to have continuation with OpenACC
        prolog = '!$acc parallel {clauses}\n'.format(clauses=clauses)
        epilog = '!$acc end parallel\n'
        # ...

        # ...
        code = ('{prolog}'
                '{body}\n'
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
                '{loop}\n'
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
        depth = expr.depth

        prolog = ''
        epilog = ''
        code   = ''

        # ...
        def _do_range(target, iter, prolog, epilog):
            tar        = self._print(target)
            range_code = self._print(iter)

            prolog += 'do {0} = {1}\n'.format(tar, range_code)
            epilog = 'end do\n' + epilog

            return prolog, epilog
        # ...

        # ...
        def _iprint(i):
            if isinstance(i, Block):
                _prelude, _body = self._print_Block(i)
                return '{0}'.format(_body)
            else:
                return '{0}'.format(self._print(i))
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

        body = '\n'.join(_iprint(i) for i in expr.body)
        # ...

        return ('{prolog}'
                '{body}\n'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)


    #def _print_Block(self, expr):
    #    body    = '\n'.join(self._print(i) for i in expr.body)
    #    prelude = '\n'.join(self._print(i) for i in expr.declarations)
    #    return prelude, body

    def _print_While(self,expr):
        body = '\n'.join(self._print(i) for i in expr.body)
        return ('do while ({test}) \n'
                '{body}\n'
                'end do').format(test=self._print(expr.test), body=body)

    def _print_ErrorExit(self, expr):
        # TODO treat the case of MPI
        return 'STOP'

    def _print_Assert(self, expr):
        # we first create an If statement
        # TODO: depending on a debug flag we should print 'PASSED' or not.
        DEBUG = True

        err = ErrorExit()
        args = [(Not(expr.test), [Print(["'Assert Failed'"]), err])]

        if DEBUG:
            args.append((True, Print(["'PASSED'"])))

        stmt = If(*args)
        code = self._print(stmt)
        return self._get_statement(code)

    def _print_Is(self, expr):
        if not isinstance(expr.rhs, Nil):
            raise NotImplementedError('Only None rhs is allowed in Is statement')

        lhs = self._print(expr.lhs)
        return 'present({})'.format(lhs)

    def _print_If(self, expr):
        # ...
        def _iprint(i):
            if isinstance(i, Block):
                _prelude, _body = self._print_Block(i)
                return '{0}'.format(_body)
            else:
                return '{0}'.format(self._print(i))
        # ...

        lines = []
        for i, (c, e) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s) then" % _iprint(c))
            elif i == len(expr.args) - 1 and c == True:
                lines.append("else")
            else:
                lines.append("else if (%s) then" % _iprint(c))
            if isinstance(e, (list, tuple, Tuple)):
                for ee in e:
                    lines.append(_iprint(ee))
            else:
                lines.append(_iprint(e))
        lines.append("end if")
        return "\n".join(lines)

    def _print_MatrixElement(self, expr):
        return "{0}({1}, {2})".format(expr.parent, expr.i + 1, expr.j + 1)

    def _print_Add(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # collect the purely real and purely imaginary parts:
        pure_real = []
        pure_imaginary = []
        mixed = []
        for arg in expr.args:
            if arg.is_number and arg.is_real:
                pure_real.append(arg)
            elif arg.is_number and arg.is_imaginary:
                pure_imaginary.append(arg)
            else:
                mixed.append(arg)
        if len(pure_imaginary) > 0:
            if len(mixed) > 0:
                PREC = precedence(expr)
                term = Add(*mixed)
                t = self._print(term)
                if t.startswith('-'):
                    sign = "-"
                    t = t[1:]
                else:
                    sign = "+"
                if precedence(term) < PREC:
                    t = "(%s)" % t

                return "cmplx(%s,%s) %s %s" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                    sign, t,
                )
            else:
                return "cmplx(%s,%s)" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                )
        else:
            return CodePrinter._print_Add(self, expr)

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



    def _print_Function(self, expr):
        # for the moment, this is only used if the function has not arguments
        args = expr.args
        name = type(expr).__name__

        code_args = ', '.join(self._print(i) for i in args)

        code = '{0}({1})'.format(name, code_args)
        if isinstance(expr.func, Subroutine):
            code = 'call ' + code
       
        return self._get_statement(code)


    def _print_MethodCall(self, expr):
        func = expr.func
        name = func.name
        name = self._print(name)

        code_args = ''
        if not(expr.arguments) is None:
            code_args = ', '.join(self._print(i) for i in expr.arguments)

        this = self._print(expr.cls_variable)
        code = '{0} % {1}({2})'.format(this, name, code_args)
        if func.is_procedure:
            code = 'call {0}'.format(code)
        return self._get_statement(code)

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        return "cmplx(0,1)"

    def _print_int(self, expr):
        return str(expr)

    def _print_Mul(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        if expr.is_number and expr.is_imaginary:
            return "cmplx(0,%s)" % (
                self._print(-S.ImaginaryUnit*expr)
            )
        else:
            return CodePrinter._print_Mul(self, expr)

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp == -1:
            one = Float(1.0)
            code = '{0}/{1}'.format(self._print(one), \
                                    self.parenthesize(expr.base, PREC))
            return code
        elif expr.exp == 0.5:
            if expr.base.is_integer:
                # Fortan intrinsic sqrt() does not accept integer argument
                if expr.base.is_Number:
                    return 'sqrt(%s.0d0)' % self._print(expr.base)
                else:
                    return 'sqrt(dble(%s))' % self._print(expr.base)
            else:
                return 'sqrt(%s)' % self._print(expr.base)
        else:
            return CodePrinter._print_Pow(self, expr)

    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])
        return "%sd0" % printed

    def _print_IndexedVariable(self, expr):
        return self._print(expr.name)

    def _print_IndexedElement(self, expr):
        inds = [i for i in expr.indices]
        #indices of indexedElement of len==1 shouldn't be a Tuple
        for i, ind in enumerate(inds):
            if isinstance(ind, Tuple) and len(ind) == 1:
                inds[i] = ind[0]

        inds = [self._print(i) for i in inds]

        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))

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
            end = expr.end - 1
            end = self._print(end)
        return '{0}:{1}'.format(start, end)

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
             lines  --  a list of lines (without \\n character)

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
            if line.startswith("! "):
                # comment line
                if len(line) > 72:
                    pos = line.rfind(" ", 6, 72)
                    if pos == -1:
                        pos = 72
                    hunk = line[:pos]
                    line = line[pos:].lstrip()
                    result.append(hunk)
                    while len(line) > 0:
                        pos = line.rfind(" ", 0, 66)
                        if pos == -1 or len(line) < 66:
                            pos = 66
                        hunk = line[:pos]
                        line = line[pos:].lstrip()
                        result.append("%s%s" % ("! ", hunk))
                else:
                    result.append(line)
            elif not ("'" in line or '"' in line):
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
                #Case of a line with a sting in it we dont want to split
                #TODO improve
                result.append(line)
        return result

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, string_types):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        code = [line.lstrip(' \t') for line in code]

        inc_keyword = ('do ', 'if(', 'if ', 'do\n',
                       'else', 'type', 'subroutine', 'function')
        dec_keyword = ('end do', 'enddo', 'end if', 'endif',
                       'else', 'endtype', 'end type',
                       'endfunction', 'end function',
                       'endsubroutine', 'end subroutine')

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
            if line == '' or line == '\n':
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


def fcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of c code

    expr : Expr
        A sympy expression to be converted.
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

    return FCodePrinter(settings).doprint(expr, assign_to)
