# -*- coding: utf-8 -*-

from collections import OrderedDict
import traceback
import importlib
import pickle
import os
import sys
import re

#==============================================================================

from pyccel.ast import NativeInteger, NativeReal
from pyccel.ast import NativeBool, NativeComplex
from pyccel.ast import NativeRange
from pyccel.ast import NativeIntegerList
from pyccel.ast import NativeRealList
from pyccel.ast import NativeComplexList
from pyccel.ast import NativeList
from pyccel.ast import NativeSymbol
from pyccel.ast import String
from pyccel.ast import DataTypeFactory
from pyccel.ast import Nil, Void
from pyccel.ast import Variable
from pyccel.ast import DottedName, DottedVariable
from pyccel.ast import Assign, AliasAssign, SymbolicAssign
from pyccel.ast import AugAssign, CodeBlock
from pyccel.ast import Return
from pyccel.ast import Pass
from pyccel.ast import ConstructorCall
from pyccel.ast import FunctionDef, Interface
from pyccel.ast import PythonFunction, SympyFunction
from pyccel.ast import ClassDef
from pyccel.ast import GetDefaultFunctionArg
from pyccel.ast import For, FunctionalFor, ForIterator
from pyccel.ast import GeneratorComprehension as GC
from pyccel.ast import FunctionalSum, FunctionalMax, FunctionalMin
from pyccel.ast import If, IfTernaryOperator
from pyccel.ast import While
from pyccel.ast import Print
from pyccel.ast import SymbolicPrint
from pyccel.ast import Del
from pyccel.ast import Assert
from pyccel.ast import Comment, EmptyLine, NewLine
from pyccel.ast import Break, Continue
from pyccel.ast import Slice, IndexedVariable, IndexedElement
from pyccel.ast import FunctionHeader, ClassHeader, MethodHeader
from pyccel.ast import VariableHeader, InterfaceHeader
from pyccel.ast import MetaVariable
from pyccel.ast import MacroFunction, MacroVariable
from pyccel.ast import Concatinate
from pyccel.ast import ValuedVariable
from pyccel.ast import Argument, ValuedArgument
from pyccel.ast import Is
from pyccel.ast import Import, TupleImport
from pyccel.ast import AsName
from pyccel.ast import AnnotatedComment, CommentBlock
from pyccel.ast import With, Block
from pyccel.ast import Range, Zip, Enumerate, Product, Map
from pyccel.ast import List, Dlist, Len
from pyccel.ast import builtin_function as pyccel_builtin_function
from pyccel.ast import builtin_import as pyccel_builtin_import
from pyccel.ast import builtin_import_registery as pyccel_builtin_import_registery
from pyccel.ast import Macro
from pyccel.ast import MacroShape
from pyccel.ast import construct_macro
from pyccel.ast import SumFunction, Subroutine
from pyccel.ast import Zeros, Where, Linspace, Diag, Complex
from pyccel.ast import inline, subs, create_variable, extract_subexpressions

from pyccel.ast.core      import local_sympify, int2float, Pow, _atomic
from pyccel.ast.datatypes import sp_dtype, str_dtype


from pyccel.parser.utilities import omp_statement, acc_statement
from pyccel.parser.utilities import fst_move_directives
from pyccel.parser.utilities import reconstruct_pragma_multilines
from pyccel.parser.utilities import is_valid_filename_pyh, is_valid_filename_py
from pyccel.parser.utilities import read_file
from pyccel.parser.utilities import get_default_path

from pyccel.parser.syntax.headers import parse as hdr_parse
from pyccel.parser.syntax.openmp  import parse as omp_parse
from pyccel.parser.syntax.openacc import parse as acc_parse

from pyccel.parser.errors import Errors, PyccelSyntaxError
from pyccel.parser.errors import PyccelSemanticError

# TODO - remove import * and only import what we need
#      - use OrderedDict whenever it is possible

from pyccel.parser.messages import *

#==============================================================================

from sympy.core.function       import Function, FunctionClass, Application
from sympy.core.numbers        import ImaginaryUnit
from sympy.logic.boolalg       import Boolean, BooleanTrue, BooleanFalse
from sympy.utilities.iterables import iterable as sympy_iterable
from sympy.core.assumptions    import StdFactKB

from sympy import Sum as Summation
from sympy import KroneckerDelta, Heaviside
from sympy import Symbol, sympify, symbols
from sympy import Eq, Ne, Lt, Le, Gt, Ge
from sympy import NumberSymbol, Number
from sympy import Indexed, IndexedBase
from sympy import Add, Mul, And, Or
from sympy import FunctionClass
from sympy import ceiling, floor, Mod
from sympy import Min, Max

from sympy import oo  as INF
from sympy import Pow as sp_Pow
from sympy import Integer, Float
from sympy import true, false
from sympy import Tuple
from sympy import Lambda
from sympy import Atom
from sympy import Expr
from sympy import Dict
from sympy import Not
from sympy import cache

errors = Errors()

from pyccel.parser.base      import BasicParser
from pyccel.parser.syntactic import SyntaxParser

#==============================================================================

class SemanticParser(BasicParser):

    """ Class for a Semantic Parser."""

    def __init__(self, *args, **kwargs):
        BasicParser.__init__(self, *args, **kwargs)

    def _infere_type(self, expr, **settings):
        """
        type inference for expressions
        """

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        errors = Errors()

        verbose = settings.pop('verbose', False)
        if verbose:
            print ('*** type inference for : ', type(expr))

        d_var = {}

        d_var['datatype'      ] = NativeSymbol()
        d_var['allocatable'   ] = None
        d_var['shape'         ] = ()
        d_var['rank'          ] = 0
        d_var['is_pointer'    ] = None
        d_var['is_target'     ] = None
        d_var['is_polymorphic'] = None
        d_var['is_optional'   ] = None
        d_var['cls_base'      ] = None
        d_var['cls_parameters'] = None
        d_var['precision'     ] = 0

        # TODO improve => put settings as attribut of Parser

        DEFAULT_FLOAT = settings.pop('default_float', 'real')

        if isinstance(expr, type(None)):

            return d_var
        elif isinstance(expr, (Integer, int)):

            d_var['datatype'   ] = 'int'
            d_var['allocatable'] = False
            d_var['rank'       ] = 0
            d_var['precision'  ] = 4
            return d_var
        elif isinstance(expr, (Float, float)):

            d_var['datatype'   ] = DEFAULT_FLOAT
            d_var['allocatable'] = False
            d_var['rank'       ] = 0
            d_var['precision'  ] = 8
            return d_var
        elif isinstance(expr, String):

            d_var['datatype'   ] = 'str'
            d_var['allocatable'] = False
            d_var['rank'       ] = 0
            return d_var
        elif isinstance(expr, ImaginaryUnit):

            d_var['datatype'   ] = 'complex'
            d_var['allocatable'] = False
            d_var['rank'       ] = 0
            d_var['precision'  ] = 8
            return d_var
        elif isinstance(expr, Variable):

            d_var['datatype'      ] = expr.dtype
            d_var['allocatable'   ] = expr.allocatable
            d_var['shape'         ] = expr.shape
            d_var['rank'          ] = expr.rank
            d_var['cls_base'      ] = expr.cls_base
            d_var['is_pointer'    ] = expr.is_pointer
            d_var['is_polymorphic'] = expr.is_polymorphic
            d_var['is_optional'   ] = expr.is_optional
            d_var['is_target'     ] = expr.is_target
            d_var['order'         ] = expr.order
            d_var['precision'     ] = expr.precision
            return d_var
        elif isinstance(expr, (BooleanTrue, BooleanFalse)):

            d_var['datatype'   ] = NativeBool()
            d_var['allocatable'] = False
            d_var['is_pointer' ] = False
            d_var['rank'       ] = 0
            return d_var
        elif isinstance(expr, IndexedElement):

            d_var['datatype'] = expr.dtype
            name = _get_name(expr)
            var = self.get_variable(name)
            if var is None:
                raise ValueError('Undefined variable {name}'.format(name=name))

            d_var['datatype'] = str_dtype(var.dtype)

            if sympy_iterable(var.shape):
                shape = []
                for (s, i) in zip(var.shape, expr.indices):
                    if isinstance(i, Slice):
                        shape.append(i)
            else:
                shape = ()

            rank = max(0, var.rank - expr.rank)
            if rank > 0:
                d_var['allocatable'] = var.allocatable
                d_var['is_pointer' ] = var.is_pointer

            d_var['shape'    ] = shape
            d_var['rank'     ] = rank
            d_var['precision'] = var.precision

            return d_var
        elif isinstance(expr, IndexedVariable):

            name = _get_name(expr)
            var = self.get_variable(name)
            if var is None:
                raise ValueError('Undefined variable {name}'.format(name=name))
            d_var['datatype'   ] = var.dtype
            d_var['allocatable'] = var.allocatable
            d_var['shape'      ] = var.shape
            d_var['rank'       ] = var.rank
            d_var['precision'  ] = var.precision
            return d_var
        elif isinstance(expr, Range):

            d_var['datatype'   ] = NativeRange()
            d_var['allocatable'] = False
            d_var['shape'      ] = ()
            d_var['rank'       ] = 0
            d_var['cls_base'   ] = expr  # TODO: shall we keep it?
            return d_var
        elif isinstance(expr, Is):

            d_var['datatype'   ] = NativeBool()
            d_var['allocatable'] = False
            d_var['is_pointer' ] = False
            d_var['rank'       ] = 0
            return d_var
        elif isinstance(expr, DottedVariable):

            if isinstance(expr.lhs, DottedVariable):
                self._current_class = expr.lhs.rhs.cls_base
            else:
                self._current_class = expr.lhs.cls_base
            d_var = self._infere_type(expr.rhs)
            self._current_class = None
            return d_var


        elif isinstance(expr, Lambda):

            d_var['datatype'   ] = NativeSymbol()
            d_var['allocatable'] = False
            d_var['is_pointer' ] = False
            d_var['rank'       ] = 0
            return d_var
        elif isinstance(expr, ConstructorCall):
            cls_name = expr.func.cls_name
            cls = self.get_class(cls_name)

            dtype = self.get_class_construct(cls_name)()

            d_var['datatype'   ] = dtype
            d_var['allocatable'] = False
            d_var['shape'      ] = ()
            d_var['rank'       ] = 0
            d_var['is_target'  ] = True

            # set target  to True if we want the class objects to be pointers

            d_var['is_polymorphic'] = False
            d_var['cls_base'      ] = cls
            d_var['is_pointer'    ] = False
            return d_var
        elif isinstance(expr, Application):
            name = _get_name(expr)
            func = self.get_function(name)
            if isinstance(func, FunctionDef):
                d_var = self._infere_type(func.results[0], **settings)

            elif name in ['Zeros', 'Ones', 'Empty', 'Diag',
                          'Shape', 'Cross', 'Linspace','Where']:
                d_var['datatype'   ] = expr.dtype
                d_var['allocatable'] = True
                d_var['shape'      ] = expr.shape
                d_var['rank'       ] = expr.rank
                d_var['is_pointer' ] = False
                d_var['order'      ] = expr.order

            elif name in ['Array']:

                dvar = self._infere_type(expr.arg, **settings)

                if expr.dtype:
                    dvar['datatype' ] = expr.dtype
                    dvar['precision'] = expr.precision

                dvar['datatype'] = str_dtype(dvar['datatype'])

                d_var = {}
                d_var['allocatable'] = True
                d_var['shape'      ] = dvar['shape']
                d_var['rank'       ] = dvar['rank']
                d_var['is_pointer' ] = False
                d_var['datatype'   ] = 'ndarray' + dvar['datatype']
                d_var['precision'  ] = dvar['precision']

                d_var['is_target'] = True # ISSUE 177: TODO this should be done using update_variable

            elif name in ['Len', 'Sum', 'Rand', 'Min', 'Max']:
                d_var['datatype'   ] = sp_dtype(expr)
                d_var['rank'       ] = 0
                d_var['allocatable'] = False
                d_var['is_pointer' ] = False

            elif name in ['Int','Int32','Int64','Real','Imag',
                          'Float32','Float64','Complex',
                          'Complex128','Complex64']:

                d_var['datatype'   ] = sp_dtype(expr)
                d_var['rank'       ] = 0
                d_var['allocatable'] = False
                d_var['is_pointer' ] = False
                d_var['precision'  ] = expr.precision

            elif name in ['Mod']:

                # Determine output type/rank/shape
                # TODO [YG, 10.10.2018]: use Numpy broadcasting rules
                d_vars = [self._infere_type(arg,**settings) for arg in expr.args]
                i = 0 if (d_vars[0]['rank'] >= d_vars[1]['rank']) else 1

                d_var['datatype'   ] = d_vars[i]['datatype']
                d_var['rank'       ] = d_vars[i]['rank']
                d_var['shape'      ] = d_vars[i]['shape']
                d_var['allocatable'] = d_vars[i]['allocatable']
                d_var['is_pointer' ] = False
                d_var['precision'  ] = d_vars[i].pop('precision',4)
            elif name in ['Norm']:
                d_var = self._infere_type(expr.arg,**settings)

                d_var['shape'] = expr.shape(d_var['shape'])
                d_var['rank' ] = len(d_var['shape'])
                d_var['allocatable'] = d_var['rank']>0
                d_var['is_pointer' ] = False

            elif name in [
                    'Abs',
                    'sin',
                    'cos',
                    'exp',
                    'log',
                    'csc',
                    'cos',
                    'sec',
                    'tan',
                    'cot',
                    'asin',
                    'acsc',
                    'acos',
                    'asec',
                    'atan',
                    'acot',
                    'atan2',
                    ]:
                d_var = self._infere_type(expr.args[0], **settings)
                d_var['datatype'] = sp_dtype(expr)

            elif name in ['ZerosLike', 'FullLike']:
                d_var = self._infere_type(expr.rhs, **settings)

            elif name in ['floor']:
                d_var = self._infere_type(expr.args[0], **settings)
                d_var['datatype'] = 'int'

                if expr.args[0].is_complex and not expr.args[0].is_integer:
                    d_var['precision'] = d_var['precision']//2
                else:
                    d_var['precision'] = 4

            else:
                raise NotImplementedError('TODO')

            return d_var

        elif isinstance(expr, GC):
            return self._infere_type(expr.lhs, **settings)
        elif isinstance(expr, Expr):

            cls = (Application, DottedVariable, Variable,
                   IndexedVariable,IndexedElement)
            atoms = _atomic(expr,cls)
            ds = [self._infere_type(i, **settings) for i in
                  atoms]
            #TODO we should also look for functions call
            #to collect info about precision and shapes later when we allow
            # vectorised operations
            # we only look for atomic expression of type Variable
            # because we don't allow functions that returns an array in an expression

            allocatables = [d['allocatable'] for d in ds]
            pointers = [d['is_pointer'] or d['is_target'] for d in ds]
            ranks = [d['rank'] for d in ds]
            shapes = [d['shape'] for d in ds]
            precisions = [d['precision'] for d in ds]


            if all(i.is_integer for i in atoms):
                if expr.is_complex and not expr.is_integer:
                    precisions.append(8)

            # TODO improve
            # ... only scalars and variables of rank 0 can be handled

            if any(ranks):
                r_min = min(ranks)
                r_max = max(ranks)
                if not r_min == r_max:
                    if not r_min == 0:
                        msg = 'cannot process arrays of different ranks.'
                        raise ValueError(msg)
                rank = r_max
            else:
                rank = 0

            shape = ()
            for s in shapes:
                if s:
                    shape = s

            # ...
            d_var['datatype'   ] = sp_dtype(expr)
            d_var['allocatable'] = any(allocatables)
            d_var['is_pointer' ] = any(pointers)
            d_var['shape'      ] = shape
            d_var['rank'       ] = rank
            if len(precisions)>0:
                d_var['precision'] = max(precisions)
            else:
                if d_var['datatype']=='int':
                    d_var['precision'] = 4
                else:
                    d_var['precision'] = 8
            return d_var
        elif isinstance(expr, (tuple, list, List, Tuple)):

            import numpy
            d = self._infere_type(expr[0], **settings)

            # TODO must check that it is consistent with pyccel's rules

            d_var['datatype'] = d['datatype']
            d_var['rank'] = d['rank'] + 1
            d_var['shape'] = numpy.shape(expr)  # TODO improve
            d_var['allocatable'] = d['allocatable']
            if isinstance(expr, List):
                d_var['is_target'] = True
                dtype = str_dtype(d['datatype'])
                if dtype == 'integer':
                    d_var['datatype'] = NativeIntegerList()
                elif dtype == 'real':
                    d_var['datatype'] = NativeRealList()
                elif dtype == 'complex':
                    d_var['datatype'] = NativeComplexList()
                else:
                    raise NotImplementedError('TODO')
            return d_var
        elif isinstance(expr, Concatinate):
            import operator
            d_vars = [self._infere_type(a, **settings) for a in expr.args]
            ls = any(d['is_pointer'] or d['is_target'] for d in d_vars)

            if ls:
                shapes = [d['shape'] for d in d_vars if d['shape']]
                shapes = zip(*shapes)
                shape = tuple(sum(s) for s in shapes)
                if not shape:
                    shape = (sum(map(Len,expr.args)),)
                d_vars[0]['shape'     ] = shape
                d_vars[0]['rank'      ] = 1
                d_vars[0]['is_target' ] = True
                d_vars[0]['is_pointer'] = False

            else:
                d_vars[0]['datatype'] = 'str'
            return d_vars[0]


            if not (d_var_left['datatype'] == 'str'
                    or d_var_right['datatype'] == 'str'):
                d_var_left['shape'] = tuple(map(operator.add,
                        d_var_right['shape'], d_var_left['shape']))
            return d_var_left
        elif isinstance(expr, ValuedArgument):
            return self._infere_type(expr.value)

        elif isinstance(expr, IfTernaryOperator):
            return self._infere_type(expr.args[0][1][0])
        elif isinstance(expr, Dlist):

            import numpy
            d = self._infere_type(expr.val, **settings)

            # TODO must check that it is consistent with pyccel's rules
            # TODO improve
            d_var['datatype'   ] = d['datatype']
            d_var['rank'       ] = d['rank'] + 1
            d_var['shape'      ] = (expr.length, )
            d_var['allocatable'] = False
            d_var['is_pointer' ] = True
            return d_var

        else:
            msg = '{expr} not yet available'.format(expr=type(expr))
            raise NotImplementedError(msg)


#==============================================================================
#==============================================================================
#==============================================================================



    def _visit(self, expr, **settings):
        """Annotates the AST.

        IndexedVariable atoms are only used to manipulate expressions, we then,
        always have a Variable in the namespace."""

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        classes = type(expr).__mro__
        import time
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(expr, **settings)
        # Unknown object, we raise an error.

        raise PyccelSemanticError('{expr} not yet available'.format(expr=type(expr)))

    def _visit_list(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return Tuple(*ls, sympify=False)

    def _visit_tuple(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return Tuple(*ls, sympify=False)

    def _visit_Tuple(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return Tuple(*ls, sympify=False)

    def _visit_List(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return List(*ls, sympify=False)

    def _visit_CodeBlock(self, expr, **settings):
        return expr
    def _visit_Nil(self, expr, **settings):
        return expr
    def _visit_ValuedArgument(self, expr, **settings):
        return expr
    def _visit_EmptyLine(self, expr, **settings):
        return expr
    def _visit_NewLine(self, expr, **settings):
        return expr
    def _visit_Break(self, expr, **settings):
        return expr
    def _visit_Continue(self, expr, **settings):
        return expr
    def _visit_Comment(self, expr, **settings):
        return expr
    def _visit_CommentBlock(self, expr, **settings):
        return expr
    def _visit_AnnotatedComment(self, expr, **settings):
        return expr
    def _visit_Integer(self, expr, **settings):
        return expr
    def _visit_Float(self, expr, **settings):
        return expr
    def _visit_String(self, expr, **settings):
        return expr
    def _visit_ImaginaryUnit(self, expr, **settings):
        return expr
    def _visit_BooleanTrue(self, expr, **settings):
        return expr
    def _visit_BooleanFalse(self, expr, **settings):
        return expr
    def _visit_Pass(self, expr, **settings):
        return expr

    def _visit_NumberSymbol(self, expr, **settings):
        return expr.n()

    def _visit_Number(self, expr, **settings):
        return expr.n()

    def _visit_Variable(self, expr, **settings):
        name = expr.name
        var = self.get_variable(name)
        if var is None:
            #TODO error not yet tested
            errors.report(UNDEFINED_VARIABLE, symbol=name,
                          bounding_box=self.bounding_box,
                          severity='error', blocker=self.blocking)
        return var


    def _visit_str(self, expr, **settings):
        return repr(expr)

    def _visit_Slice(self, expr, **settings):
        args = list(expr.args)
        if args[0] is not None:
            args[0] = self._visit(args[0], **settings)

        if args[1] is not None:
            args[1] = self._visit(args[1], **settings)
        return Slice(*args)


    def _visit_Indexed(self, expr, **settings):
        name = _get_name(expr)
        var = self.get_variable(name)
        if var is None:
            errors.report(UNDEFINED_INDEXED_VARIABLE, symbol=name,
                          bounding_box=self.bounding_box,
                          severity='error', blocker=self.blocking)

         # TODO check consistency of indices with shape/rank

        args = list(expr.indices)

        if var.rank>len(args):
            # add missing dimensions

            args = args + [Slice(None, None)]*(var.rank-len(args))

        args = [self._visit(arg, **settings) for arg in args]

        if var.order == 'C':
            args.reverse()
        args = tuple(args)

         # case of Pyccel ast Variable, IndexedVariable
         # if not possible we use symbolic objects

        if hasattr(var, 'dtype'):
            dtype = var.dtype
            shape = var.shape
            prec  = var.precision
            order = var.order
            rank  = var.rank
            return IndexedVariable(name, dtype=dtype,
                   shape=shape,prec=prec,order=order,rank=rank).__getitem__(*args)
        else:
            return IndexedBase(name).__getitem__(args)


    def _visit_Symbol(self, expr, **settings):
        name = expr.name
        var = self.get_variable(name)

        if var is None:
            var = self.get_function(name)
        if var is None:
            var = self.get_symbolic_function(name)

        if var is None:

            errors.report(UNDEFINED_VARIABLE, symbol=name,
            bounding_box=self.bounding_box,
            severity='error', blocker=self.blocking)
        return var


    def _visit_DottedVariable(self, expr, **settings):

        first = self._visit(expr.lhs)
        rhs_name = _get_name(expr.rhs)
        attr_name = []
        if first.cls_base:
            attr_name = [i.name for i in first.cls_base.attributes]
        name = None

        if isinstance(expr.rhs, Symbol) and not expr.rhs.name \
            in attr_name:
            for i in first.cls_base.methods:
                if str(i.name) == expr.rhs.name and 'property' \
                    in i.decorators.keys():

                    d_var = self._infere_type(i.results[0], **settings)
                    dtype = d_var['datatype']
                    assumptions = {str_dtype(dtype): True}
                    second = Function(expr.rhs.name, **assumptions)(Nil())

                    return DottedVariable(first, second)

        if not isinstance(expr.rhs, Application):
            macro = self.get_macro(rhs_name)
            if macro:
                return macro.master

            self._current_class = first.cls_base
            second = self._visit(expr.rhs, **settings)
            self._current_class = None
        else:

            macro = self.get_macro(rhs_name)
            if not macro is None:
                master = macro.master
                name = macro.name
                master_args = macro.master_arguments
                args = expr.rhs.args
                args = [expr.lhs] + list(args)
                args = [self._visit(i, **settings) for i in args]
                args = macro.apply(args)
                if isinstance(master, FunctionDef):
                    return Subroutine(str(master.name))(*args)
                else:
                    msg = 'TODO case of interface'
                    raise NotImplementedError(msg)
            args = [self._visit(arg, **settings) for arg in
                    expr.rhs.args]
            for i in first.cls_base.methods:
                if str(i.name.name) == rhs_name:
                    if len(i.results) == 1:
                        d_var = self._infere_type(i.results[0], **settings)
                        dtype = d_var['datatype']
                        assumptions = {str_dtype(dtype): True}
                        second = Function(i.name.name, **assumptions)(*args)

                    elif len(i.results) == 0:
                        second = Subroutine(i.name.name)(*args)
                    elif len(i.results) > 1:
                        msg = 'TODO case multiple return variables'
                        raise NotImplementedError(msg)

                    expr = DottedVariable(first, second)
                    return expr
        return DottedVariable(first, second)

    def _visit_Add(self, expr, **settings):

        stmts, expr = extract_subexpressions(expr)
        if stmts:
            stmts = [self._visit(i, **settings) for i in stmts]

        atoms_str = _atomic(expr, String)
        atoms_ls  = _atomic(expr, List)

        cls       = (Symbol, Indexed, DottedVariable, List)

        atoms = _atomic(expr, cls, ignore=(Function))
        atoms = [self._visit(a, **settings) for a in atoms]
        atoms = [a.rhs if isinstance(a, DottedVariable) else a for a in atoms]

        atoms = [self._infere_type(a , **settings) for a in atoms]

        atoms = [a['is_pointer'] or a['is_target'] for a in atoms if a['rank']>0]
        args  = [self._visit(a, **settings) for a in expr.args]


        if any(atoms) or atoms_ls:
            return Concatinate(args, True)
        elif atoms_str:
            return Concatinate(args, False)


        expr_new = Add(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)
        if stmts:
            expr_new = CodeBlock(stmts + [expr_new])
        return expr_new


    def _visit_Mul(self, expr, **settings):
        stmts, expr = extract_subexpressions(expr)
        if stmts:
            stmts = [self._visit(i, **settings) for i in stmts]
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Mul(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)
        if stmts:
            expr_new = CodeBlock(stmts + [expr_new])
        return expr_new


    def _visit_Pow(self, expr, **settings):

        stmts, expr = extract_subexpressions(expr)
        if stmts:
            stmts = [self._visit(i, **settings) for i in stmts]
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Pow(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)
        if stmts:
            expr_new = CodeBlock(stmts + [expr_new])
        return expr_new

    def _visit_And(self, expr, **settings):

        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = And(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)

        return expr_new

    def _visit_Or(self, expr, **settings):

        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Or(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)

        return expr_new

    def _visit_Equality(self, expr, **settings):

        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Eq(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)
        return expr_new

    def _visit_Unequality(self, expr, **settings):


        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Ne(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)

        return expr_new

    def _visit_StrictLessThan(self, expr, **settings):

        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Lt(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)

        return expr_new

    def _visit_GreaterThan(self, expr, **settings):

        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Ge(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)

        return expr_new

    def _visit_LessThan(self, expr, **settings):

        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Le(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)

        return expr_new

    def _visit_StrictGreaterThan(self, expr, **settings):

        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = Gt(*args, evaluate=False)
        expr_new = expr_new.doit(deep=False)
        return expr_new



    def _visit_Lambda(self, expr, **settings):


        expr_names = set(map(str, expr.expr.atoms(Symbol)))
        var_names = map(str, expr.variables)
        if len(expr_names.difference(var_names)) > 0:
            msg = 'Unknown variables in lambda definition'
            raise ValueError(msg)
        funcs = expr.expr.atoms(Function)
        for func in funcs:
            name = _get_name(func)
            f = self.get_symbolic_function(name)
            if f is None:
                msg = 'Unknown function in lambda definition'
                raise ValueError(msg)
            else:

                f = f(*func.args)
                expr_new = expr.expr.subs(func, f)
                expr = Lambda(expr.variables, expr_new)
        return expr

    def _visit_Application(self, expr, **settings):
        name     = type(expr).__name__
        func     = self.get_function(name)

        stmts, new_args = extract_subexpressions(expr.args)

        stmts = [self._visit(stmt, **settings) for stmt in stmts]
        args  = [self._visit(arg, **settings) for arg in new_args]

        if name == 'lambdify':
            args = self.get_symbolic_function(str(expr.args[0]))
        F = pyccel_builtin_function(expr, args)

        if F is not None:
            if len(stmts) > 0:
                stmts.append(F)
                return CodeBlock(stmts)
            return F

        elif name in self._namespace['cls_constructs'].keys():

            # TODO improve the test
            # we must not invoke the namespace like this

            cls = self.get_class(name)
            d_methods = cls.methods_as_dict
            method = d_methods.pop('__init__', None)

            if method is None:

                # TODO improve case of class with the no __init__

                errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                       bounding_box=self.bounding_box,
                       severity='error', blocker=True)
            args = expr.args
            m_args = method.arguments[1:]  # we delete the self arg

            # TODO check compatibility
            # TODO treat parametrized arguments.

            expr = ConstructorCall(method, args, cls_variable=None)
            if len(stmts) > 0:
                stmts.append(expr)
                return CodeBlock(stmts)
            return expr
        else:

            # first we check if it is a macro, in this case, we will create
            # an appropriate FunctionCall

            macro = self.get_macro(name)
            if not macro is None:
                args = [self._visit(i, **settings) for i in args]
                func = macro.master
                name = _get_name(func.name)
                args = macro.apply(args)
            else:
                func = self.get_function(name)

            if func is None:
                errors.report(UNDEFINED_FUNCTION, symbol=name,
                       bounding_box=self.bounding_box,
                       severity='error', blocker=self.blocking)
            else:
                if not isinstance(func, (FunctionDef, Interface)):

                    expr = func(*args)

                    if isinstance(expr, (Where, Diag, Linspace)):
                        self.insert_variable(expr.index)

                    if len(stmts) > 0:
                        stmts.append(expr)
                        return CodeBlock(stmts)
                    return expr
                else:

                    if 'inline' in func.decorators.keys():
                        return inline(func,args)

                    if isinstance(func, FunctionDef):
                        results = func.results
                        f_args = func.arguments
                    elif isinstance(func, Interface):
                        arg_dvar = [self._infere_type(i, **settings) for i in args]
                        f_dvar = [[self._infere_type(j, **settings)
                                  for j in i.arguments] for i in
                                  func.functions]
                        j = -1
                        for i in f_dvar:
                            j += 1
                            found = True
                            for (idx, dt) in enumerate(arg_dvar):
                                dtype1 = str_dtype(dt['datatype'])
                                dtype2 = str_dtype(i[idx]['datatype'])
                                found = found and (dtype1 in dtype2
                                              or dtype2 in dtype1)
                                found = found and dt['rank'] \
                                              == i[idx]['rank']
                            if found:
                                break

                        if found:
                            results = func.functions[j].results
                            f_args = func.functions[j].arguments
                        else:
                            msg = 'function not found in the interface'
                            raise SystemExit(msg)

                        if func.hide:

                            # hide means that we print the real function's name
                            # and not the interface's name

                            name = str(func.functions[j].name)

                    # add the messing argument in the case of optional arguments

                    if not len(args) == len(f_args):
                        n = len(args)
                        for i in f_args[n:]:
                            if not isinstance(i, ValuedVariable):
                                msg = 'Expecting a valued variable'
                                raise TypeError(msg)
                            if not isinstance(i.value, Nil):
                                args.append(ValuedArgument(i.name, i.value))

                    if len(results) == 1:

                        d_var = self._infere_type(results[0], **settings)
                        dtype = d_var['datatype']
                        dtype = {str_dtype(dtype): True}
                        expr = Function(name,**dtype)(*args)

                    elif len(results) == 0:
                        expr = Subroutine(name)(*args)
                        if len(stmts) > 0:
                            stmts.append(expr)
                            return CodeBlock(stmts)
                        return expr
                    elif len(results) > 1:
                        expr = Function(name)(*args)
                        if len(stmts) > 0:
                            stmts.append(expr)
                            return CodeBlock(stmts)
                        return expr

                    if len(stmts) > 0:
                        stmts.append(expr)
                        return CodeBlock(stmts)
                    return expr

    def _visit_Expr(self, expr, **settings):
        msg = '{expr} not yet available'
        msg = msg.format(expr=type(expr))
        raise NotImplementedError(msg)

    def _visit_Min(self, expr, **settings):
        raise
        args = self._visit(expr.args, **settings)
        return Min(*args)

    def _visit_Max(self, expr, **settings):
        args = self._visit(expr.args, **settings)
        return Max(*args)


    def _visit_Assign(self, expr, **settings):

        # TODO unset position at the end of this part
        fst = expr.fst
        if fst:
            self._bounding_box = fst.absolute_bounding_box
        else:
            msg = 'Found a node without fst member ({})'
            msg = msg.format(type(expr))
            raise PyccelSemanticError(msg)

        rhs = expr.rhs
        lhs = expr.lhs
        assigns = None

        if isinstance(rhs, Application):
            name = _get_name(rhs)
            macro = self.get_macro(name)
            if not macro is None:

                # TODO check types from FunctionDef

                master = macro.master
                name = _get_name(master.name)

                # all terms in lhs must be already declared and available
                # the namespace
                # TODO improve

                if not sympy_iterable(lhs):
                    lhs = [lhs]

                results = []
                for a in lhs:
                    _name = _get_name(a)
                    var = self.get_variable(_name)
                    if var is None:
                        errors.report(UNDEFINED_VARIABLE,
                             symbol=_name, bounding_box=self.bounding_box,
                             severity='error', blocker=self.blocking)
                    results.append(var)

                # ...

                args = [self._visit(i, **settings) for i in
                            rhs.args]
                args = macro.apply(args, results=results)
                if isinstance(master, FunctionDef):
                    return Subroutine(name)(*args)
                else:
                    msg = 'TODO treate interface case'
                    raise NotImplementedError(msg)

        if isinstance(rhs, DottedVariable):
            var = rhs.rhs
            name = _get_name(var)
            macro = self.get_macro(name)
            if not macro is None:
                master = macro.master
                if isinstance(macro, MacroVariable):
                    self.insert_variable(master)
                    rhs = master
                else:
                    name = macro.name
                    master_args = macro.master_arguments
                    if not sympy_iterable(lhs):
                        lhs = [lhs]
                    results = []
                    for a in lhs:
                        _name = _get_name(a)
                        var = self.get_variable(_name)
                        if var is None:
                            errors.report(UNDEFINED_VARIABLE,
                            symbol=_name, bounding_box=self.bounding_box,
                            severity='error', blocker=self.blocking)
                        results.append(var)

                    # ...

                    args = rhs.rhs.args
                    args = [rhs.lhs] + list(args)
                    args = [self._visit(i, **settings) for i in args]

                    args = macro.apply(args, results=results)

                    # TODO treate interface case

                    if isinstance(master, FunctionDef):
                        return Subroutine(str(master.name))(*args)
                    else:
                        raise NotImplementedError('TODO')


        rhs = self._visit(rhs, **settings)

        if isinstance(rhs, IfTernaryOperator):
            args = rhs.args
            new_args = []
            for arg in args:
                if len(arg[1]) != 1:
                    msg = 'IfTernary body must be of length 1'
                    raise ValueError(msg)
                result = arg[1][0]
                if isinstance(expr, Assign):
                    body = Assign(lhs, result)
                else:
                    body = AugAssign(lhs, expr.op, result)
                body.set_fst(fst)
                new_args.append([arg[0], [body]])
            expr = IfTernaryOperator(*new_args)
            return self._visit_If(expr, **settings)

        elif isinstance(rhs, FunctionDef):

            # case of lambdify

            rhs = rhs.rename(_get_name(expr.lhs))
            for i in rhs.body:
                i.set_fst(fst)
            rhs = self._visit_FunctionDef(rhs, **settings)
            return rhs

        elif isinstance(rhs, Block):
            #case of inline
            results = _atomic(rhs.body,Return)
            sub = list(zip(results,[EmptyLine()]*len(results)))
            body = rhs.body
            body = subs(body,sub)
            results = [i.expr for i in results]
            lhs = expr.lhs
            if isinstance(lhs ,(list, tuple, Tuple)):
                sub = [list(zip(i,lhs)) for i in results]
            else:
                sub = [(i[0],lhs) for i in results]
            body = subs(body,sub)
            expr = Block(rhs.name, rhs.variables, body)
            return expr

        elif isinstance(rhs, GC):
            if str(rhs.lhs) != str(lhs):

                if isinstance(lhs, Symbol):
                    name = lhs.name
                    if self.get_variable(name) is None:
                        d_var = self._infere_type(rhs.lhs, **settings)
                        dtype = d_var.pop('datatype')
                        lhs = Variable(dtype, name , **d_var)
                        self.insert_variable(lhs)

                if isinstance(rhs, FunctionalSum):
                    stmt = AugAssign(lhs,'+',rhs.lhs)
                elif isinstance(rhs, FunctionalMin):
                    stmt = Assign(lhs, Min(lhs,rhs.lhs))
                elif isinstance(rhs, FunctionalMax):
                    stmt = Assign(lhs, Max(lhs, rhs.lhs))

                return CodeBlock([rhs, stmt])
            return rhs

        elif isinstance(rhs, FunctionalFor):
            return rhs


        elif isinstance(rhs, CodeBlock):
            # case of complex stmt
            # that needs to be splitted
            # into a list of stmts
            stmts = rhs.body
            stmt  = stmts[-1]
            lhs   = expr.lhs
            if isinstance(lhs, Symbol):
                name = lhs.name
                if self.get_variable(name) is None:
                    d_var = self._infere_type(stmt, **settings)
                    dtype = d_var.pop('datatype')
                    lhs = Variable(dtype, name , **d_var)
                    self.insert_variable(lhs)

            if isinstance(expr, Assign):
                stmt = Assign(lhs, stmt)
            elif isinstance(expr, AugAssign):
                stmt = AugAssign(lhs, expr.op, stmt)
            stmt.set_fst(fst)
            stmts[-1] = stmt
            return CodeBlock(stmts)

        elif isinstance(rhs, Application):

            # ARA: needed for functions defined only with a header

            name = _get_name(rhs)
            func = self.get_function(name)

            if isinstance(func, FunctionDef):
                results = func.results
                if results:
                    d_var = [self._infere_type(i, **settings)
                                 for i in results]

                # case of elemental function
                # if the input and args of func do not have the same shape,
                # then the lhs must be already declared
                if func.is_elemental:
                    # we first compare the funcdef args with the func call
                    # args
#                   d_var = None
                    func_args = func.arguments
                    call_args = rhs.args
                    f_ranks = [x.rank for x in func_args]
                    c_ranks = [x.rank for x in call_args]
                    same_ranks = [x==y for (x,y) in zip(f_ranks, c_ranks)]
                    if not all(same_ranks):
                        _name = _get_name(lhs)
                        var = self.get_variable(_name)
                        if var is None:
                            # TODO have a specific error message
                            errors.report(UNDEFINED_VARIABLE,
                                 symbol=_name, bounding_box=self.bounding_box,
                                 severity='error', blocker=self.blocking)

            elif isinstance(func, Interface):
                d_var = [self._infere_type(i, **settings) for i in
                         func.functions[0].results]

                # TODO imporve this will not work for
                # the case of different results types
                d_var[0]['datatype'] = sp_dtype(rhs)

            else:
                d_var = self._infere_type(rhs, **settings)

        elif isinstance(rhs, Map):

            name = str(rhs.args[0])
            func = self.get_function(name)

            if func is None:
                errors.report(UNDEFINED_FUNCTION, symbol=name,
                bounding_box=self.bounding_box, severity='error',
                blocker=self.blocking)

            dvar  = self._infere_type(rhs.args[1], **settings)
            d_var = [self._infere_type(result, **settings) for result in func.results]
            for i in range(len(d_var)):
                d_var[i]['shape'] = dvar['shape']
                d_var[i]['rank' ]  = dvar['rank']


        else:

            d_var = self._infere_type(rhs, **settings)
            __name__ = d_var['datatype'].__class__.__name__

            if __name__.startswith('Pyccel'):
                __name__ = __name__[6:]
                d_var['cls_base'] = self.get_class(__name__)
                d_var['is_pointer'] = d_var['is_target'] or d_var['is_pointer']

                # TODO if we want to use pointers then we set target to true
                # in the ConsturcterCall

                d_var['is_polymorphic'] = False

            if d_var['is_target']:
                # case of rhs is a target variable the lhs must be a pointer
                if isinstance(rhs, Symbol):
                    d_var['is_target' ] = False
                    d_var['is_pointer'] = True

        lhs = expr.lhs
        if isinstance(lhs, Symbol):
            if isinstance(d_var, list):
                if len(d_var) > 1:
                    msg = 'can not assign multiple object into one variable'
                    raise ValueError(msg)
                elif len(d_var) == 1:
                    d_var = d_var[0]

            name = lhs.name
            dtype = d_var.pop('datatype')

            d_lhs = d_var.copy()
            # ISSUES #177: lhs must be a pointer when rhs is allocatable array
            if d_lhs['allocatable'] and isinstance(rhs, Variable):
                d_lhs['allocatable'] = False
                d_lhs['is_pointer' ] = True

                # TODO uncomment this line, to make rhs target for
                #      lists/tuples.
                #rhs = self.update_variable(rhs, is_target=True)


            lhs = Variable(dtype, name, **d_lhs)
            var = self.get_variable(name)
            if var is None:
                self.insert_variable(lhs, name=lhs.name)

            else:

                # TODO improve check type compatibility
                if str(lhs.dtype) != str(var.dtype):
                    txt = '|{name}| {old} <-> {new}'
                    txt = txt.format(name=name, old=var.dtype, new=lhs.dtype)

                    # case where the rhs is of native type
                    # TODO add other native types
                    if isinstance(rhs, (Integer, Float)):
                        errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT,
                        symbol=txt,bounding_box=self.bounding_box,
                        severity='error', blocker=False)

                    else:
                        errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT,
                        symbol=txt, bounding_box=self.bounding_box,
                        severity='internal', blocker=False)

                # in the case of elemental, lhs is not of the same dtype as
                # var.
                # TODO d_lhs must be consistent with var!
                # the following is a small fix, since lhs must be already
                # declared
                lhs = var


        elif isinstance(lhs, DottedVariable):

            dtype = d_var.pop('datatype')
            name = lhs.lhs.name
            if self._current == '__init__':
                cls      = self.get_variable('self')
                cls_name = str(cls.cls_base.name)
                cls      = self.get_class(cls_name)

                attributes = cls.attributes
                parent     = cls.parent
                attributes = list(attributes)
                n_name     = str(lhs.rhs.name)

                # update the self variable with the new attributes

                dt       = self.get_class_construct(cls_name)()
                cls_base = self.get_class(cls_name)
                var      = Variable(dt, 'self', cls_base=cls_base)
                d_lhs    = d_var.copy()
                self.insert_variable(var, 'self')


                # ISSUES #177: lhs must be a pointer when rhs is allocatable array
                if d_lhs['allocatable'] and isinstance(rhs, Variable):
                    d_lhs['allocatable'] = False
                    d_lhs['is_pointer' ] = True

                    rhs = self.update_variable(rhs, is_target=True)

                member = Variable(dtype, n_name, **d_lhs)
                lhs    = DottedVariable(var, member)

                # update the attributes of the class and push it to the namespace
                attributes += [member]
                new_cls = ClassDef(cls_name, attributes, [], parent=parent)
                self.insert_class(new_cls)
            else:
                lhs = self._visit_DottedVariable(lhs, **settings)

        else:
            lhs = self._visit(lhs, **settings)

        if isinstance(rhs, (Map, Zip)):
            func  = _get_name(rhs.args[0])
            func  = Function(func)
            alloc = Assign(lhs, Zeros(lhs.shape, lhs.dtype))
            alloc.set_fst(fst)
            index = create_variable(expr)
            index = Variable('int',index.name)
            range_ = Function('range')(Function('len')(lhs))
            name  = _get_name(lhs)
            var   = IndexedBase(name)[index]
            args  = rhs.args[1:]
            args  = [_get_name(arg) for arg in args]
            args  = [IndexedBase(arg)[index] for arg in args]
            body  = [Assign(var, func(*args))]
            body[0].set_fst(fst)
            body  = For(index, range_, body, strict=False)
            body  = self._visit_For(body, **settings)
            body  = [alloc , body]
            return CodeBlock(body)


        if not isinstance(lhs, (list, Tuple, tuple)):
            lhs = [lhs]
            if isinstance(d_var,dict):
                d_var = [d_var]

        for (i, dic) in enumerate(d_var):

            allocatable = False
            is_pointer  = False
            if dic['allocatable']:
                allocatable = True

            if dic['is_pointer']:
                is_pointer = True

            if ('is_target' in dic.keys() and dic['is_target']  and
                isinstance(rhs, Variable)):
                is_pointer = True

            if (isinstance(rhs, IndexedElement) and
                lhs[i].rank > 0):
                allocatable = True

            elif (isinstance(rhs, Variable) and
                isinstance(rhs.dtype, NativeList)):
                is_pointer = True

            if (isinstance(lhs, Variable) and (allocatable or is_pointer)):
                lhs[i] = self.update_variable(lhs[i],
                         allocatable=allocatable,
                         is_pointer=is_pointer)

        if len(lhs) == 1:
            lhs = lhs[0]


        is_pointer = is_pointer and isinstance(rhs, (Variable, Dlist, DottedVariable))
        is_pointer = is_pointer or isinstance(lhs, (Variable, DottedVariable)) and lhs.is_pointer
        # ISSUES #177: lhs must be a pointer when rhs is allocatable array
        new_expr = Assign(lhs, rhs)
        if is_pointer:
            new_expr = AliasAssign(lhs, rhs)

        elif isinstance(expr, AugAssign):
            new_expr = AugAssign(lhs, expr.op, rhs)


        elif new_expr.is_symbolic_alias:
            new_expr = SymbolicAssign(lhs, rhs)

            # in a symbolic assign, the rhs can be a lambda expression
            # it is then treated as a def node

            F = self.get_symbolic_function(lhs)
            if F is None:
                self.insert_symbolic_function(new_expr)
            else:
                raise NotImplementedError('TODO')

        new_expr.set_fst(fst)

        return new_expr

    def _visit_For(self, expr, **settings):

        # treatment of the index/indices

        iterable = self._visit(expr.iterable, **settings)
        body     = list(expr.body)
        iterator = expr.target

        if isinstance(iterable, Variable):
            indx   = create_variable(iterable)
            assign = Assign(iterator, IndexedBase(iterable)[indx])
            assign.set_fst(expr.fst)
            iterator = indx
            body     = [assign] + body

        elif isinstance(iterable, Map):
            indx   = create_variable(iterable)
            func   = iterable.args[0]
            args   = [IndexedBase(arg)[indx] for arg in iterable.args[1:]]
            assing = assign = Assign(iterator, func(*args))
            assign.set_fst(expr.fst)
            iterator = indx
            body     = [assign] + body

        elif isinstance(iterable, Zip):
            args = iterable.args
            indx = create_variable(args)
            for i in range(len(args)):
                assign = Assign(iterator[i], IndexedBase(args[i])[indx])
                assign.set_fst(expr.fst)
                body = [assign] + body
            iterator = indx

        elif isinstance(iterable, Enumerate):
            indx   = iterator.args[0]
            var    = iterator.args[1]
            assign = Assign(var, IndexedBase(iterable.args[0])[indx])
            assign.set_fst(expr.fst)
            iterator = indx
            body     = [assign] + body

        elif isinstance(iterable, Product):
            args     = iterable.args
            iterator = list(iterator)
            for i in range(len(args)):
                indx   = create_variable(i)
                assign = Assign(iterator[i], IndexedBase(args[i])[indx])

                assign.set_fst(expr.fst)
                body        = [assign] + body
                iterator[i] = indx

        if isinstance(iterator, Symbol):
            name   = iterator.name
            var    = self.get_variable(name)
            target = var
            if var is None:
                target = Variable('int', name, rank=0)
                self.insert_variable(target)
        elif isinstance(iterator, list):
            target = []
            for i in iterator:
                name = str(i.name)
                var  = Variable('int', name, rank=0)
                self.insert_variable(var)
                target.append(var)
        else:
            dtype = type(iterator)

            # TODO ERROR not tested yet

            errors.report(INVALID_FOR_ITERABLE, symbol=expr.target,
                   bounding_box=self.bounding_box,
                   severity='error', blocker=self.blocking)

        body = [self._visit(i, **settings) for i in body]

        if isinstance(iterable, Variable):
            return ForIterator(target, iterable, body)

        return For(target, iterable, body)


    def _visit_GeneratorComprehension(self, expr, **settings):

        result   = expr.expr
        lhs_name = _get_name(expr.lhs)
        lhs      = self.get_variable(lhs_name)

        if lhs is None:
            lhs  = Variable('int', lhs_name)
            self.insert_variable(lhs)

        loops  = [self._visit(i, **settings) for i in expr.loops]
        result = self._visit(result, **settings)
        if isinstance(result, CodeBlock):
            result = result.body[-1]


        d_var = self._infere_type(result, **settings)
        dtype = d_var.pop('datatype')

        lhs = None
        if isinstance(expr.lhs, Symbol):
            lhs = Variable(dtype, lhs_name, **d_var)
            self.insert_variable(lhs)


        if isinstance(expr, FunctionalSum):
            val = Integer(0)
            if str_dtype(dtype) in ['real', 'complex']:
                val = Float(0.0)
        elif isinstance(expr, FunctionalMin):
            val = INF
        elif isinstance(expr, FunctionalMax):
            val = -INF

        stmt = Assign(expr.lhs, val)
        stmt.set_fst(expr.fst)
        loops.insert(0, stmt)

        if isinstance(expr, FunctionalSum):
            expr_new = FunctionalSum(loops, lhs=lhs)
        elif isinstance(expr, FunctionalMin):
            expr_new = FunctionalMin(loops, lhs=lhs)
        elif isinstance(expr, FunctionalMax):
            expr_new = FunctionalMax(loops, lhs=lhs)
        expr_new.set_fst(expr.fst)
        return expr_new

    def _visit_FunctionalFor(self, expr, **settings):

        target  = expr.expr
        index   = expr.index
        indices = expr.indices
        dims    = []
        body    = expr.loops[1]

        while isinstance(body, For):

            stop  = None
            start = 0
            step  = 1
            var   = body.target
            a     = self._visit(body.iterable, **settings)
            if isinstance(a, Range):
                var   = Variable('int', var.name)
                stop  = a.stop
                start = a.start
                step  = a.step
            elif isinstance(a, (Zip, Enumerate)):
                dvar  = self._infere_type(a.element, **settings)
                dtype = dvar.pop('datatype')
                if dvar['rank'] > 0:
                    dvar['rank' ] -= 1
                    dvar['shape'] = (dvar['shape'])[1:]
                if dvar['rank'] == 0:
                    dvar['allocatable'] = dvar['is_pointer'] = False
                var  = Variable(dtype, var.name, **dvar)
                stop = a.element.shape[0]
            elif isinstance(a, Variable):
                dvar  = self._infere_type(a, **settings)
                dtype = dvar.pop('datatype')
                if dvar['rank'] > 0:
                    dvar['rank'] -= 1
                    dvar['shape'] = (dvar['shape'])[1:]
                if dvar['rank'] == 0:
                    dvar['allocatable'] = dvar['is_pointer'] = False

                var  = Variable(dtype, var.name, **dvar)
                stop = a.shape[0]
            else:
                raise NotImplementedError('TODO')
            self.insert_variable(var)

            size = (stop - start) / step
            body = body.body[0]
            dims.append((size, step, start, stop))


        # we now calculate the size of the array which will be allocated

        for i in range(len(indices)):
            var = self.get_variable(indices[i].name)
            if var is None:
                raise ValueError('variable not found')
            indices[i] = var

        dim = dims[-1][0]
        for i in range(len(dims) - 1, 0, -1):
            size  = dims[i - 1][0]
            step  = dims[i - 1][1]
            start = dims[i - 1][2]
            size  = ceiling(size)
            dim   = ceiling(dim)
            dim   = dim.subs(indices[i-1], start+step*indices[i-1])
            dim   = Summation(dim, (indices[i-1], 0, size-1))
            dim   = dim.doit()
        if isinstance(dim, Summation):
            raise NotImplementedError('TODO')

        # TODO find a faster way to calculate dim
        # when step>1 and not isinstance(dim, Sum)
        # maybe use the c++ library of sympy

        # we annotate the target to infere the type of the list created

        target = self._visit(target, **settings)
        d_var = self._infere_type(target, **settings)

        dtype = d_var.pop('datatype')
        d_var['rank'] += 1
        shape = list(d_var['shape'])
        d_var['is_pointer'] = True
        shape.append(dim)
        d_var['shape'] = Tuple(*shape, sympify=False)

        lhs_name = _get_name(expr.lhs)
        lhs      = Variable(dtype, lhs_name, **d_var)
        self.insert_variable(lhs)

        loops = [self._visit(i, **settings) for i in expr.loops]
        index = self._visit(index, **settings)

        return FunctionalFor(loops, lhs=lhs, indices=indices, index=index)

    def _visit_While(self, expr, **settings):
        test = self._visit(expr.test, **settings)
        body = [self._visit(i, **settings) for i in expr.body]
        return While(test, body)

    def _visit_If(self, expr, **settings):
        args = [self._visit(i, **settings) for i in expr.args]
        return expr.func(*args)

    def _visit_VariableHeader(self, expr, **settings):

        # TODO improve
        #      move it to the ast like create_definition for FunctionHeader?

        name  = expr.name
        d_var = expr.dtypes
        dtype = d_var.pop('datatype')

        var = Variable(dtype, name, **d_var)
        self.insert_variable(var)
        return expr

    def _visit_FunctionHeader(self, expr, **settings):
        # TODO should we return it and keep it in the AST?
        self.insert_header(expr)
        return expr

    def _visit_ClassHeader(self, expr, **settings):
        # TODO should we return it and keep it in the AST?
        self.insert_header(expr)
        return expr

    def _visit_InterfaceHeader(self, expr, **settings):

        container = self.namespace['functions']

        # TODO improve test all possible containers

        if set(expr.funcs).issubset(container.keys()):
            name  = expr.name
            funcs = []
            for i in expr.funcs:
                funcs += [container[i]]

        expr            = Interface(name, funcs, hide=True)
        container[name] = expr
        return expr

    def _visit_Return(self, expr, **settings):

        results  = expr.expr
        new_vars = []
        assigns  = []

        if not isinstance(results, (list, Tuple, List)):
            results = [results]

        for result in results:
            if not isinstance(result, Symbol):
                new_vars += [create_variable(result)]
                stmt      = Assign(new_vars[-1], result)
                stmt.set_fst(expr.fst)
                assigns  += [stmt]
                assigns[-1].set_fst(expr.fst)

        if len(assigns) == 0:
            results = [self._visit_Symbol(result, **settings)
                       for result in results]
            return Return(results)
        else:
            assigns  = [self._visit_Assign(assign, **settings)
                       for assign in assigns]
            new_vars = [self._visit_Symbol(i, **settings) for i in
                        new_vars]
            assigns  = CodeBlock(assigns)
            return Return(new_vars, assigns)

    def _visit_FunctionDef(self, expr, **settings):

        name         = str(expr.name)
        name         = name.replace("'", '')
        cls_name     = expr.cls_name
        hide         = False
        kind         = 'function'
        decorators   = expr.decorators
        funcs        = []
        is_static    = False
        is_pure      = expr.is_pure
        is_elemental = expr.is_elemental
        is_private   = expr.is_private

        header = expr.header
        if header is None:
            if cls_name:
                header = self.get_header(cls_name +'.'+ name)
            else:
                header = self.get_header(name)

        if expr.arguments and not header:

            # TODO ERROR wrong position

            errors.report(FUNCTION_TYPE_EXPECTED, symbol=name,
                   bounding_box=self.bounding_box,
                   severity='error', blocker=self.blocking)

        # we construct a FunctionDef from its header

        if header:
            interfaces = header.create_definition()

            # is_static will be used for f2py

            is_static  = header.is_static

            # get function kind from the header

            kind = header.kind
        else:

            # this for the case of a function without arguments => no header

            interfaces = [FunctionDef(name, [], [], [])]

        vec_func = None
        if 'vectorize' in decorators:
            #TODO move to another place
            vec_name  = 'vec_' + name
            arg       = decorators['vectorize'][0]
            arg       = str(arg.name)
            args      = [str(i.name) for i in expr.arguments]
            index_arg = args.index(arg)
            arg       = Symbol(arg)
            vec_arg   = IndexedBase(arg)
            index     = create_variable(expr.body)
            range_    = Function('range')(Function('len')(arg))
            args      = symbols(args)
            args[index_arg] = vec_arg[index]
            body_vec        = Assign(args[index_arg], Function(name)(*args))
            body_vec.set_fst(expr.fst)
            body_vec   = [For(index, range_, [body_vec], strict=False)]
            header_vec = header.vectorize(index_arg)
            vec_func   = expr.vectorize(body_vec, header_vec)


        for m in interfaces:
            args        = []
            results     = []
            local_vars  = []
            global_vars = []
            imports     = []
            arg         = None

            self.set_current_fun(name)
            arguments = expr.arguments
            if cls_name and str(arguments[0].name) == 'self':
                arg       = arguments[0]
                arguments = arguments[1:]
                dt        = self.get_class_construct(cls_name)()
                cls_base  = self.get_class(cls_name)
                var       = Variable(dt, 'self', cls_base=cls_base)
                self.insert_variable(var, 'self')

            if arguments:
                for (a, ah) in zip(arguments, m.arguments):
                    d_var = self._infere_type(ah, **settings)
                    dtype = d_var.pop('datatype')

                    # this is needed for the static case

                    additional_args = []
                    if isinstance(a, ValuedArgument):

                        # optional argument only if the value is None

                        if isinstance(a.value, Nil):
                            d_var['is_optional'] = True
                        a_new = ValuedVariable(dtype, str(a.name),
                                    value=a.value, **d_var)
                    else:

                        # add shape as arguments if is_static and arg is array

                        rank = d_var['rank']
                        if is_static and rank > 0:
                            for i in range(0, rank):
                                n_name = 'n{i}_{name}'.format(name=str(a.name), i=i)
                                n_arg  = Variable('int', n_name)

                                # TODO clean namespace later

                                var = self.get_variable(n_name)
                                if not var is None:

                                    # TODO ERROR not tested yet

                                    errors.report(REDEFINING_VARIABLE,
                                    symbol=n_name, severity='error',
                                    blocker=self.blocking)

                                self.insert_variable(n_arg)
                                additional_args += [n_arg]

                            # update shape
                            # TODO can this be improved? add some check

                            d_var['shape'] = Tuple(*additional_args, sympify=False)
                        a_new = Variable(dtype, _get_name(a), **d_var)

                    if additional_args:
                        args += additional_args

                    args.append(a_new)
                    self.insert_variable(a_new, name=str(a_new.name))

            if len(interfaces) == 1 and len(interfaces[0].results) == 1:

                # case of recursive function
                # TODO improve

                self.insert_function(interfaces[0])

            # we annotate the body
            body = [self._visit(i, **settings) for i in
                    expr.body]

            # ISSUE 177: must update arguments to get is_target
            args = [self.get_variable(a.name) for a in args]

            # find return stmt and results

            returns = self._collect_returns_stmt(body)
            results = []

            for stmt in returns:
                results += [set(stmt.expr)]

            if not all(i == results[0] for i in results):
                #case of multiple return
                # with diffrent variable name
                msg = 'TODO not available yet'
                raise PyccelSemanticError(msg)

            if len(results) > 0:
                results = list(results[0])

            if arg and cls_name:
                dt       = self.get_class_construct(cls_name)()
                cls_base = self.get_class(cls_name)
                var      = Variable(dt, 'self', cls_base=cls_base)
                args     = [var] + args

            for var in self.get_variables():
                if not var in args + results:
                    local_vars += [var]

            # TODO should we add all the variables or only the ones used in the function

            for var in self.get_variables('parent'):
                if not var in args + results + local_vars:
                    global_vars += [var]

            is_recursive = False

            # get the imports
            imports = self._scope[self._current]['imports']
            imports = list(set(imports))
            self.set_current_fun(None)
            func_   = self.get_function(name)
            if not func_ is None and func_.is_recursive:
                is_recursive = True

            func = FunctionDef(
                name,
                args,
                results,
                body,
                local_vars=local_vars,
                global_vars=global_vars,
                cls_name=cls_name,
                hide=hide,
                kind=kind,
                is_static=is_static,
                is_pure=is_pure,
                is_elemental=is_elemental,
                is_private=is_private,
                imports=imports,
                decorators=decorators,
                is_recursive=is_recursive)
            if cls_name:
                cls = self.get_class(cls_name)
                methods = list(cls.methods) + [func]

                # update the class  methods

                self.insert_class(ClassDef(cls_name,
                        cls.attributes, methods, parent=cls.parent))

            funcs += [func]
            #clear the sympy cache
            #TODO move that inside FunctionDef
            #and clear all variable except the global one
            cache.clear_cache()

        if len(funcs) == 1:
            funcs = funcs[0]
            self.insert_function(funcs)

        else:
            funcs = [f.rename(name + '_' + str(i)) for (i,
                     f) in enumerate(funcs)]

            funcs = Interface(name, funcs)
            self.insert_function(funcs)

        if vec_func:
           vec_func  = self._visit(vec_func, **settings)
           if isinstance(funcs, Interface):
               funcs = list(funcs.funcs)+[vec_func]
           else:
               funcs = funcs.rename('sc_'+ name)
               funcs = [funcs,vec_func]

           funcs = Interface(name, funcs)
           self.insert_function(funcs)


        return funcs

    def _visit_Print(self, expr, **settings):
        args = [self._visit(i, **settings) for i in expr.expr]
        if len(args) == 0:
            raise ValueError('no arguments given to print function')

        is_symbolic = lambda var: isinstance(var, Variable) \
            and isinstance(var.dtype, NativeSymbol)
        test = all(is_symbolic(i) for i in args)

        # TODO fix: not yet working because of mpi examples
#        if not test:
#            raise ValueError('all arguments must be either symbolic or none of them')

        if is_symbolic(args[0]):
            _args = []
            for a in args:
                f = self.get_symbolic_function(a.name)
                if f is None:
                    _args.append(a)
                else:

                    # TODO improve: how can we print SymbolicAssign as  lhs = rhs

                    _args.append(f)
            return SymbolicPrint(_args)
        else:
            return Print(args)

    def _visit_ClassDef(self, expr, **settings):

        # TODO - improve the use and def of interfaces
        #      - wouldn't be better if it is done inside ClassDef?

        name = str(expr.name)
        name = name.replace("'", '')
        methods = list(expr.methods)
        parent = expr.parent
        interfaces = []

        # remove quotes for str representation
        cls = ClassDef(name, [], [], parent=parent)
        self.insert_class(cls)
        const = None

        for (i, method) in enumerate(methods):
            m_name = str(method.name).replace("'", '')

            if m_name == '__init__':
                const = self._visit_FunctionDef(method, **settings)
                methods.pop(i)
                break



        if not const:
            errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                   bounding_box=self.bounding_box,
                   severity='error', blocker=True)

        methods = [self._visit_FunctionDef(i, **settings) for i in methods]
        methods = [const] + methods
        header = self.get_header(name)

        if not header:
            msg = 'Expecting a header class for {classe} but could not find it.'
            raise ValueError(msg.format(classe=name))
        options    = header.options
        attributes = self.get_class(name).attributes
        for i in methods:
            if isinstance(i, Interface):
                methods.remove(i)
                interfaces += [i]
        return ClassDef(name, attributes, methods,
                       interfaces=interfaces, parent=parent)

    def _visit_Del(self, expr, **settings):

        ls = [self._visit(i, **settings) for i in expr.variables]
        return Del(ls)

    def _visit_Is(self, expr, **settings):

        # TODO ERROR wrong position

        if not isinstance(expr.rhs, Nil):
           errors.report(PYCCEL_RESTRICTION_IS_RHS,
                         bounding_box=self.bounding_box,
                         severity='error', blocker=self.blocking)

        name = expr.lhs
        var = self.get_variable(str(name))
        if var is None:

            errors.report(UNDEFINED_VARIABLE, symbol=name,
                   bounding_box=self.bounding_box,
                   severity='error', blocker=self.blocking)

        return Is(var, expr.rhs)

    def _visit_Import(self, expr, **settings):

        # TODO - must have a dict where to store things that have been
        #        imported
        #      - should not use namespace

        if expr.source:
            container = self._imports
            if self._current:
                container = container[self._current]

            if str(expr.source) in pyccel_builtin_import_registery:

                imports = pyccel_builtin_import(expr)
                for (name, atom) in imports:
                    if not name is None:
                        F = self.get_variable(name)

                        if F is None:
                            container[name] = atom
                        elif name in container:
                            errors.report(FOUND_DUPLICATED_IMPORT,
                                        symbol=name, severity='warning')
                        else:
                            raise NotImplementedError('must report error')
            else:

                # in some cases (blas, lapack, openmp and openacc level-0)
                # the import should not appear in the final file
                # all metavars here, will have a prefix and suffix = __

                __ignore_at_import__ = False
                __module_name__      = None
                __import_all__       = False

                # we need to use str here since source has been defined
                # using repr.
                # TODO shall we improve it?
                targets = [_get_name(i) for i in expr.target]
                p       = self.d_parsers[str(expr.source)]
                for entry in ['variables', 'classes',
                              'functions', 'cls_constructs']:

                    d_self = self._namespace[entry]
                    d_son = p.namespace[entry]
                    for (k, v) in list(d_son.items()):

                        # TODO test if it is not already in the namespace

                        if k in targets:
                            d_self[k] = v

                # we add all the macros from the son

                self._namespace['macros'
                                    ].update(p.namespace['macros'])

                # ... meta variables

                if 'ignore_at_import' in list(p.metavars.keys()):
                    __ignore_at_import__ = \
                        p.metavars['ignore_at_import']

                if 'import_all' in list(p.metavars.keys()):
                    __import_all__ = p.metavars['import_all']

                if 'module_name' in list(p.metavars.keys()):
                    __module_name__ = p.metavars['module_name']
                    expr = Import(expr.target, __module_name__)

                # ...
                if 'print' in p.metavars.keys():
                    source = str(expr.source).split('.')[-1]
                    source = 'mod_' + source
                    expr   = Import(expr.target,source=source)


                if not __ignore_at_import__:
                    return expr
                else:
                    if __import_all__:
                        expr = Import(__module_name__)
                        self.insert_import(expr)

                        # we return the expr when we are in
                        # program

                        if self._current is None:
                            return expr
                    return EmptyLine()
        return expr


    def _visit_With(self, expr, **settings):

        domaine = self._visit(expr.test, **settings)
        parent  = domaine.cls_base
        if not parent.is_with_construct:
            msg = '__enter__ or __exit__ methods not found'
            raise ValueError(msg)

        body = [self._visit(i, **settings) for i in expr.body]
        return With(domaine, body, None).block



    def _visit_MacroFunction(self, expr, **settings):

        # we change here the master name to its FunctionDef

        f_name = expr.master
        header = self.get_header(f_name)
        if header is None:
            func = self.get_function(f_name)
            if func is None:
                errors.report(MACRO_MISSING_HEADER_OR_FUNC,
                symbol=f_name, bounding_box=self.bounding_box,
                severity='error', blocker=self.blocking)
        else:
            interfaces = header.create_definition()

            # TODO -> Said: must handle interface

            func = interfaces[0]

        name = expr.name
        args = expr.arguments
        master_args = expr.master_arguments
        results = expr.results
        macro   = MacroFunction(name, args, func, master_args,
                                  results=results)
        self.insert_macro(macro)

        return macro

    def _visit_MacroVariable(self, expr, **settings):

        master = expr.master
        if isinstance(master, DottedName):
            raise NotImplementedError('TODO')
        header = self.get_header(master)
        if header is None:
            var = self.get_variable(master)
            if var is None:
                errors.report(MACRO_MISSING_HEADER_OR_FUNC,
                symbol=master, bounding_box=self.bounding_box,
                severity='error', blocker=self.blocking)
        else:
            var = Variable(header.dtype, header.name)

                # TODO -> Said: must handle interface

        expr = MacroVariable(expr.name, var)
        self.insert_macro(expr)
        return expr

    def _visit_Dlist(self, expr, **settings):

        val = self._visit(expr.val, **settings)
        if isinstance(val, (Tuple, list, tuple)):
            #TODO list of dimesion > 1 '

            msg = 'TODO not yet supported'
            raise PyccelSemanticError(msg)
        shape = self._visit(expr.length, **settings)
        return Dlist(val, shape)

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except:
        raise ValueError('Expecting an argument for filename')

    parser = SyntaxParser(filename)
    parser = SemanticParser(filename)
