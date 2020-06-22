# -*- coding: utf-8 -*-
# pylint: disable=R0201

from collections import OrderedDict
import traceback

#==============================================================================

from pyccel.ast import NativeRange
from pyccel.ast import NativeSymbol
from pyccel.ast import String
from pyccel.ast import DataTypeFactory
from pyccel.ast import Nil
from pyccel.ast import Variable
from pyccel.ast import TupleVariable
from pyccel.ast import DottedName, DottedVariable
from pyccel.ast import Assign, AliasAssign, SymbolicAssign
from pyccel.ast import AugAssign, CodeBlock
from pyccel.ast import Return
from pyccel.ast import ConstructorCall
from pyccel.ast import FunctionDef, Interface
from pyccel.ast import ClassDef
from pyccel.ast import For, FunctionalFor, ForIterator
from pyccel.ast import GeneratorComprehension as GC
from pyccel.ast import FunctionalSum, FunctionalMax, FunctionalMin
from pyccel.ast import If, IfTernaryOperator
from pyccel.ast import While
from pyccel.ast import Print
from pyccel.ast import SymbolicPrint
from pyccel.ast import Del
from pyccel.ast import EmptyLine
from pyccel.ast import Slice, IndexedVariable, IndexedElement
from pyccel.ast import FunctionHeader, ClassHeader, MethodHeader
from pyccel.ast import MacroFunction, MacroVariable
from pyccel.ast import Concatenate
from pyccel.ast import ValuedVariable
from pyccel.ast import ValuedArgument
from pyccel.ast import Is, IsNot
from pyccel.ast import Import
from pyccel.ast import AsName
from pyccel.ast import With, Block
from pyccel.ast import List, Dlist, Len
from pyccel.ast import builtin_function as pyccel_builtin_function
from pyccel.ast import builtin_import as pyccel_builtin_import
from pyccel.ast import builtin_import_registery as pyccel_builtin_import_registery
from pyccel.ast import Zeros, Where, Linspace, Diag, EmptyLike
from pyccel.ast import StarredArguments
from pyccel.ast import inline, subs, create_variable, extract_subexpressions
from pyccel.ast.core import get_assigned_symbols

from pyccel.ast.core      import _atomic

from pyccel.ast.core import PyccelPow, PyccelAdd, PyccelMul, PyccelDiv, PyccelMod, PyccelFloorDiv
from pyccel.ast.core import PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe
from pyccel.ast.core import PyccelAnd, PyccelOr,  PyccelNot, PyccelAssociativeParenthesis
from pyccel.ast.core import PyccelUnary

from pyccel.ast.core      import Product, FunctionCall
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeReal, NativeString
from pyccel.ast.datatypes import default_precision
from pyccel.ast.builtins  import python_builtin_datatype
from pyccel.ast.builtins  import Range, Zip, Enumerate, Map, PythonTuple
from pyccel.ast.numbers   import BooleanTrue, BooleanFalse
from pyccel.ast.numbers   import Integer, Float, Complex
from pyccel.ast.numpyext  import PyccelArraySize
from pyccel.ast.numpyext  import NumpyUfuncBase
from pyccel.ast.utilities import split_positional_keyword_arguments
from pyccel.ast.type_inference  import str_dtype
from pyccel.parser.errors import Errors
from pyccel.parser.errors import PyccelSemanticError
from pyccel.ast.numpyext  import NumpyInt, Int32, Int64, NumpyComplex, Complex64
from pyccel.ast.numpyext  import Complex128, NumpyFloat, Float64, Float32
from pyccel.ast.builtins  import Int as PythonInt, PythonFloat, PythonComplex, Bool as PythonBool
from pyccel.ast.mathext   import MathFunctionBase

# TODO - remove import * and only import what we need
#      - use OrderedDict whenever it is possible
# TODO move or delet extract_subexpressions when we introduce 
#   Functional programming
from pyccel.parser.messages import *

#==============================================================================

from sympy.core.function       import Application, UndefinedFunction
from sympy.core.numbers        import ImaginaryUnit
from sympy.utilities.iterables import iterable as sympy_iterable

from sympy import Sum as Summation
from sympy import Symbol
from sympy import Integer as sp_Integer
from sympy import Float as sp_Float
from sympy import Indexed, IndexedBase
from sympy import ceiling

from sympy import oo  as INF
from sympy import Tuple
from sympy import Lambda
from sympy import Expr
from sympy.core import cache

errors = Errors()

from pyccel.parser.base      import BasicParser, Scope
from pyccel.parser.base      import get_filename_from_import
from pyccel.parser.syntactic import SyntaxParser

from pyccel.ast.basic import PyccelAstNode

#==============================================================================

def _get_name(var):
    """."""

    if isinstance(var, (Symbol, IndexedVariable, IndexedBase)):
        return str(var)
    if isinstance(var, (IndexedElement, Indexed)):
        return str(var.base)
    if isinstance(var, Application):
        return type(var).__name__
    if isinstance(var, AsName):
        return var.name
    msg = 'Uncovered type {dtype}'.format(dtype=type(var))
    raise NotImplementedError(msg)

#==============================================================================

class SemanticParser(BasicParser):

    """ Class for a Semantic Parser.
    It takes a syntactic parser as input for the moment"""

    def __init__(self, inputs, **kwargs):

        # a Parser can have parents, who are importing it.
        # imports are then its sons.
        self._parents = kwargs.pop('parents', [])
        self._d_parsers = kwargs.pop('d_parsers', OrderedDict())

        # ...
        if not isinstance(inputs, SyntaxParser):
            raise TypeError('> Expecting a syntactic parser as input')

        parser = inputs
        # ...

        # ...
        BasicParser.__init__(self, **kwargs)
        # ...

        # ...
        self._fst = parser._fst
        self._ast = parser._ast

        self._possible_names = set([str(a.name) for a in self._fst.find_all('name')])

        self._filename  = parser._filename
        self._metavars  = parser._metavars
        self._namespace = parser._namespace
        self._namespace.imports['imports'] = OrderedDict()

        # we use it to detect the current method or function

        #
        self._code = parser._code
        # ...

        # ... TOD add settings
        settings = {}
        self.annotate()
        # ...

    @property
    def parents(self):
        """Returns the parents parser."""
        return self._parents

    @property
    def d_parsers(self):
        """Returns the d_parsers parser."""

        return self._d_parsers

    def annotate(self, **settings):
        """."""

        if self.semantic_done:
            print ('> semantic analysis already done')
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('semantic')

        # then we treat the current file

        ast = self.ast

        # we add the try/except to allow the parser to find all possible errors
        PyccelAstNode.stage = 'semantic'
        ast = self._visit(ast, **settings)

        self._ast = ast

        # in the case of a header file, we need to convert all headers to
        # FunctionDef etc ...

        if self.is_header_file:
            target = []

            for parent in self.parents:
                for (key, item) in parent.imports.items():
                    if get_filename_from_import(key) == self.filename:
                        target += item

            target = set(target)
            target_headers = target.intersection(self.namespace.headers.keys())

            for name in list(target_headers):
                v = self.namespace.headers[name]
                if isinstance(v, FunctionHeader) and not isinstance(v,
                        MethodHeader):
                    F = self.get_function(name)
                    if F is None:
                        interfaces = v.create_definition()
                        for F in interfaces:
                            self.insert_function(F)
                    else:

                        errors.report(IMPORTING_EXISTING_IDENTIFIED,
                                symbol=name, blocker=True,
                                severity='fatal')

        errors.check()
        self._semantic_done = True

        return ast

    def _get_new_variable(self, obj):
        var = create_variable(obj)
        name = var.name
        while name in self._possible_names:
            var = create_variable(obj)
            name = var.name
        self._possible_names.add(name)
        return var

    def _get_new_variable_name(self, obj, start_name = None):
        name = start_name if start_name is not None else create_variable(obj).name
        while name in self._possible_names:
            name = create_variable(obj).name
        self._possible_names.add(name)
        return name

    def get_variable_from_scope(self, name):
        """."""
        container = self.namespace
        while container.is_loop:
            container = container.parent_scope

        var = self._get_variable_from_scope(name, container)

        return var

    def _get_variable_from_scope(self, name, container):

        if name in container.variables:
            return container.variables[name]

        for container in container.loops:
            var = self._get_variable_from_scope(name, container)
            if var:
                return var

        return None

    def check_for_variable(self, name):
        """."""

        if self.current_class:
            for i in self._current_class.attributes:
                if str(i.name) == name:
                    var = i
                    return var

        container = self.namespace
        while container.is_loop:
            container = container.parent_scope


        imports   = container.imports
        while container:
            var = self._get_variable_from_scope(name, container)
            if var is not None:
                return var
            elif name in container.imports['variables']:
                return container.imports['variables'][name]
            container = container.parent_scope

        return None

    def get_variable(self, name):
        """."""
        var = self.check_for_variable(name)
        if var is None:
            errors.report(UNDEFINED_VARIABLE, symbol=name,
            bounding_box=self._current_fst_node.absolute_bounding_box,
            severity='fatal', blocker=True)
        else:
            return var

    def get_variables(self, container):
        # this only works one called the function scope
        # TODO needs more tests when we have nested functions
        variables = []
        variables.extend(container.variables.values())
        for container in container.loops:
            variables.extend(self.get_variables(container))
        return variables


    def get_parent_functions(self):
        container = self.namespace
        funcs = container.functions.copy()
        container = container.parent_scope
        while container:
            for i in container.functions:
                if not i in funcs:
                    funcs[i] = container.functions[i]
            container = container.parent_scope
        return funcs


    def get_class(self, name):
        """."""

        container = self.namespace

        while container:
            if name in container.classes:
                return container.classes[name]
            elif name in container.imports['classes']:
                return container.imports['classes'][name]

            container = container.parent_scope
        return None

    def insert_variable(self, var, name=None):
        """."""

        # TODO add some checks before
        if not isinstance(var, Variable):
            raise TypeError('variable must be of type Variable')

        if name is None:
            name = str(var.name)

        self.namespace.variables[name] = var


    def insert_class(self, cls, parent=False):
        """."""

        if isinstance(cls, ClassDef):
            name = str(cls.name)
            container = self.namespace
            if parent:
                container = container.parent_scope
            container.classes[name] = cls
        else:
            raise TypeError('Expected A class definition ')

    def insert_header(self, expr):
        """."""
        if isinstance(expr, MethodHeader):
            self.namespace.headers[expr.name] = expr
        elif isinstance(expr, FunctionHeader):
            self.namespace.headers[expr.func] = expr
        elif isinstance(expr, ClassHeader):
            self.namespace.headers[expr.name] = expr

            #  create a new Datatype for the current class

            iterable = 'iterable' in expr.options
            with_construct = 'with' in expr.options
            dtype = DataTypeFactory(str(expr.name), '_name',
                                    is_iterable=iterable,
                                    is_with_construct=with_construct)
            self.set_class_construct(str(expr.name), dtype)
        else:
            msg = 'header of type{0} is not supported'
            msg = msg.format(str(type(expr)))
            raise TypeError(msg)

    def get_function(self, name):
        """."""

        # TODO shall we keep the elif in _imports?

        func = None

        container = self.namespace
        while container:
            if name in container.functions:
                func = container.functions[name]
                break

            if name in container.imports['functions']:
                func =  container.imports['functions'][name]
                break
            container = container.parent_scope


        if func and self._current_function == name and not func.is_recursive:
            func = func.set_recursive()
            container.functions[name] = func

        return func



    def get_symbolic_function(self, name):
        """."""

        # TODO shall we keep the elif in _imports?
        container = self.namespace
        while container:
            if name in container.symbolic_functions:
                return container.symbolic_functions[name]

            if name in container.imports['symbolic_functions']:
                return container.imports['symbolic_functions'][name]
            container = container.parent_scope

        return None

    def get_python_function(self, name):
        """."""

        # TODO shall we keep the elif in _imports?
        container = self.namespace
        while container:
            if name in container.python_functions:
                return container.python_functions[name]

            if name in container.imports['python_functions']:
                return container.imports['python_functions'][name]

            container = container.parent_scope

        return None

    def get_macro(self, name):
        """."""

        # TODO shall we keep the elif in _imports?

        container = self.namespace
        while container:
            if name in container.macros:
                return container.macros[name]
            container = container.parent_scope

        return None

    def insert_macro(self, macro):
        """."""

        container = self.namespace.macros

        if isinstance(macro, (MacroFunction, MacroVariable)):
            name = macro.name
            if isinstance(macro.name, DottedName):
                name = name.name[-1]
            container[str(name)] = macro
        else:
            raise TypeError('Expected a macro')

    def remove_variable(self, name):
        """."""

        container = self.namespace
        while container:
            if name in container.variables:
                container.pop(name)
                break
            container = container.parent_scope

    def get_header(self, name):
        """."""
        container = self.namespace
        while container:
            if name in container.headers:
                return container.headers[name]
            container = container.parent_scope
        return None

    def get_class_construct(self, name):
        """Returns the class datatype for name."""
        container = self.namespace
        while container:
            if name in container.cls_constructs:
                return container.cls_constructs[name]
            container = container.parent_scope

        raise PyccelSemanticError('class construct {} not found'.format(name))


    def set_class_construct(self, name, value):
        """Sets the class datatype for name."""

        self.namespace.cls_constructs[name] = value

    def create_new_function_scope(self, name):
        """."""

        self.namespace._sons_scopes[name] = Scope()
        self.namespace._sons_scopes[name].parent_scope = self.namespace
        self._namespace = self._namespace._sons_scopes[name]
        if self._current_function:
            name = DottedName(self._current_function, name)
        self._current_function = name

    def exit_function_scope(self):

        self._namespace = self._namespace.parent_scope
        if isinstance(self._current_function, DottedName):

            name = self._current_function.name[:-1]
            if len(name)>1:
                name = DottedName(*name)
            else:
                name = name[0]
        else:
            name = None
        self._current_function = name

    def create_new_loop_scope(self):
        new_scope = Scope()
        new_scope._is_loop = True
        new_scope.parent_scope = self._namespace
        self._namespace._loops.append(new_scope)
        self._namespace = new_scope

    def exit_loop_scope(self):
        self._namespace = self._namespace.parent_scope

    def _collect_returns_stmt(self, ast):

        if isinstance(ast, CodeBlock):
            return self._collect_returns_stmt(ast.body)
        vars_ = []
        for stmt in ast:
            if isinstance(stmt, (For, While, CodeBlock)):
                vars_ += self._collect_returns_stmt(stmt.body)
            elif isinstance(stmt, If):
                vars_ += self._collect_returns_stmt(stmt.bodies)
            elif isinstance(stmt, Return):
                vars_ += [stmt]
        return vars_

######################################"

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
        d_var['precision'     ] = 0
        d_var['shape'         ] = ()
        d_var['rank'          ] = 0

        # TODO improve => put settings as attribut of Parser

        if isinstance(expr, type(None)):

            return d_var

        elif isinstance(expr, (Integer, Float, Complex, String,
                               BooleanTrue, BooleanFalse,
                               PyccelArraySize, Is, IndexedElement)):

            d_var['datatype'   ] = expr.dtype
            d_var['allocatable'] = expr.rank>0
            d_var['rank'       ] = expr.rank
            d_var['precision'  ] = expr.precision
            d_var['shape']       = expr.shape
            return d_var
        elif expr in (PythonInt, PythonFloat, PythonComplex, PythonBool, NumpyInt, 
                      Int32, Int64, NumpyComplex, Complex64, Complex128, NumpyFloat,
                      Float64, Float32):

            d_var['datatype'   ] = '*'
            d_var['rank'       ] = 0
            d_var['precision'  ] = 0
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

        elif isinstance(expr, PythonTuple):
            d_var['datatype'      ] = expr.dtype
            d_var['precision']      = expr.precision
            d_var['is_stack_array'] = expr.is_homogeneous
            d_var['shape'         ] = expr.shape
            d_var['rank'          ] = expr.rank
            d_var['is_pointer']     = False

            return d_var

        elif isinstance(expr, IndexedVariable):

            return self._infere_type(expr.internal_variable)

        elif isinstance(expr, Range):

            d_var['datatype'   ] = NativeRange()
            d_var['allocatable'] = False
            d_var['shape'      ] = ()
            d_var['rank'       ] = 0
            d_var['cls_base'   ] = expr  # TODO: shall we keep it?
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

            name = type(expr).__name__
            func = self.get_function(name)
            if isinstance(func, FunctionDef):
                d_var = self._infere_type(func.results[0], **settings)

            elif name in ['Full', 'Empty', 'Zeros', 'Ones', 'Diag',
                          'Cross', 'Linspace', 'Where']:
                d_var['datatype'   ] = expr.dtype
                d_var['allocatable'] = True
                d_var['shape'      ] = expr.shape
                d_var['rank'       ] = expr.rank
                d_var['is_pointer' ] = False
                d_var['order'      ] = expr.order
                d_var['precision'  ] = expr.precision

            elif name in ['Array']:
                d_var['datatype'      ] = expr.dtype
                d_var['precision'     ] = expr.precision
                d_var['rank'          ] = expr.rank
                d_var['shape'         ] = expr.shape
                d_var['order'         ] = expr.order
                d_var['allocatable'   ] = expr.rank>0
                d_var['is_stack_array'] = False
                d_var['is_pointer'    ] = False
                d_var['is_target'     ] = True # ISSUE 177: TODO this should be done using setter

            elif name in ['Len']:
                d_var['datatype'   ] = expr.dtype
                d_var['rank'       ] = 0
                d_var['allocatable'] = False
                d_var['is_pointer' ] = False

            elif name in ['Rand']:
                d_var['datatype'   ] = expr.dtype
                d_var['rank'       ] = expr.rank
                d_var['is_pointer' ] = False
                d_var['precision'  ] = expr.precision

                if expr.rank == 0:
                    d_var['allocatable'] = False
                else:
                    d_var['shape'      ] = expr.shape
                    d_var['allocatable'] = True
                    d_var['order'      ] = expr.order

            elif name in ['NumpySum', 'Product', 'Min', 'Max']:
                d_var['datatype'   ] = expr.args[0].dtype
                d_var['rank'       ] = 0
                d_var['allocatable'] = False
                d_var['is_pointer' ] = False

            elif name in ['Matmul']:

                d_vars = [self._infere_type(arg,**settings) for arg in expr.args]

                var0_is_vector = d_vars[0]['rank'] < 2
                var1_is_vector = d_vars[1]['rank'] < 2

                if(d_vars[0]['shape'] is None or d_vars[1]['shape'] is None):
                    d_var['shape'] = None
                else:

                    m = 1 if var0_is_vector else d_vars[0]['shape'][0]
                    n = 1 if var1_is_vector else d_vars[1]['shape'][1]
                    d_var['shape'] = [m, n]

                d_var['datatype'   ] = d_vars[0]['datatype']
                if var0_is_vector or var1_is_vector:
                    d_var['rank'   ] = 1
                else:
                    d_var['rank'   ] = 2
                d_var['allocatable'] = False
                d_var['is_pointer' ] = False
                d_var['precision'  ] = max(d_vars[0]['precision'],
                                           d_vars[1]['precision'])

            elif name in ['Int',
                          'PythonFloat','PythonComplex',
                          'NumpyInt','Int32','Int64',
                          'NumpyFloat','Float32','Float64',
                          'NumpyComplex', 'Complex64','Complex128',
                          'Real','Imag','Bool']:

                d_var['datatype'   ] = expr.dtype
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

            elif isinstance(expr, NumpyUfuncBase):
                d_var = self._infere_type(expr.args[0], **settings)
                d_var['datatype'   ] = expr.dtype
                d_var['is_pointer' ] = False
                d_var['precision'  ] = expr.precision

            elif isinstance(expr, MathFunctionBase):
                d_var['datatype' ] = expr.dtype
                d_var['precision'] = expr.precision
                d_var['rank'     ] = expr.rank
                d_var['shape'    ] = expr.shape

            elif name in ['EmptyLike', 'ZerosLike', 'OnesLike', 'FullLike']:
                d_var = self._infere_type(expr.rhs, **settings)
            else:
                raise NotImplementedError('Type of Application : '+type(expr).__name__+' cannot be infered')

            return d_var

        elif isinstance(expr, GC):
            return self._infere_type(expr.lhs, **settings)
        elif isinstance(expr, (Expr, FunctionCall)):
            # ...
            d_var['datatype'   ] = expr.dtype
            d_var['allocatable'] = expr.rank>0 if expr.rank else None
            d_var['is_pointer' ] = False
            d_var['shape'      ] = expr.shape
            d_var['rank'       ] = expr.rank
            d_var['precision'  ] = expr.precision

            return d_var
        elif isinstance(expr, (list, List)):

            import numpy
            d = self._infere_type(expr[0], **settings)

            # TODO must check that it is consistent with pyccel's rules

            d_var['datatype'] = d['datatype']
            d_var['rank'] = d['rank'] + 1
            d_var['shape'] = numpy.asarray(expr).shape  # TODO improve
            d_var['allocatable'] = d['allocatable']
            if isinstance(expr, List):
                d_var['is_target'] = True
                dtype              = str_dtype(d['datatype'])
                d_var['datatype']  = dtype

            return d_var
        elif isinstance(expr, Concatenate):
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
            return self._infere_type(expr.args[0][1].body[0])
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
        current_fst = self._current_fst_node

        if hasattr(expr,'fst') and expr.fst is not None:
            self._current_fst_node = expr.fst

        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                obj = getattr(self, annotation_method)(expr, **settings)
                self._current_fst_node = current_fst
                return obj

        # Unknown object, we raise an error.

        raise PyccelSemanticError('{expr} not yet available'.format(expr=type(expr)))

    def _visit_list(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return Tuple(*ls, sympify=False)

    def _visit_tuple(self, expr, **settings):
        ls = tuple(self._visit(i, **settings) for i in expr)
        return ls

    def _visit_PythonTuple(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return PythonTuple(*ls)

    def _visit_Tuple(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return Tuple(*ls, sympify=False)

    def _visit_List(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr]
        return List(*ls, sympify=False)

    def _visit_ValuedArgument(self, expr, **settings):
        value = self._visit(expr.value, **settings)
        d_var      = self._infere_type(value, **settings)
        dtype      = d_var.pop('datatype')
        return ValuedVariable(dtype, expr.name,
                               value=value, **d_var)

    def _visit_CodeBlock(self, expr, **settings):
        ls = [self._visit(i, **settings) for i in expr.body]
        return CodeBlock(ls)

    def _visit_Nil(self, expr, **settings):
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
        if isinstance(expr, Integer):
            return expr
        elif isinstance(expr, sp_Integer):
            return Integer(expr)
        else:
            raise TypeError("Integer type is not sympy Integer or pyccel Integer")
    def _visit_Float(self, expr, **settings):
        if isinstance(expr, Float):
            return expr
        elif isinstance(expr, sp_Float):
            return Float(expr)
        else:
            raise TypeError("Float type is not sympy Float or pyccel Float")
    def _visit_Complex(self, expr, **settings):
        return expr
    def _visit_String(self, expr, **settings):
        return expr
    def _visit_PythonComplex(self, expr, **settings):
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
        return self.get_variable(name)


    def _visit_str(self, expr, **settings):
        return repr(expr)

    def _visit_Slice(self, expr, **settings):
        args = list(expr.args)
        if args[0] is not None:
            args[0] = self._visit(args[0], **settings)

        if args[1] is not None:
            args[1] = self._visit(args[1], **settings)
        return Slice(*args)

    def _extract_indexed_from_var(self, var, args, name):

        # case of Pyccel ast Variable, IndexedVariable
        # if not possible we use symbolic objects

        if not isinstance(var, Variable):
            assert(hasattr(var,'__getitem__'))
            if len(args)==1:
                return var[args[0]]
            else:
                return self._visit(Indexed(var[args[0]],args[1:]))

        if var.order == 'C':
            args = args[::-1]
        args = tuple(args)

        if isinstance(var, TupleVariable) and not var.is_homogeneous:

            arg = args[-1]

            if isinstance(arg, Slice):
                if ((arg.start is not None and not isinstance(arg.start, Integer)) or
                        (arg.end is not None and not isinstance(arg.end, Integer))):
                    errors.report(INDEXED_TUPLE, symbol=var,
                        bounding_box=self._current_fst_node.absolute_bounding_box,
                        severity='fatal', blocker=self.blocking)

                idx = slice(arg.start, arg.end)
                selected_vars = var.get_var(idx)
                if len(selected_vars)==1:
                    if len(args) == 1:
                        return selected_vars[0]
                    else:
                        var = selected_vars[0]
                        return self._extract_indexed_from_var(var, args[:-1], name)
                elif len(selected_vars)<1:
                    return None
                elif len(args)==1:
                    return PythonTuple(*selected_vars)
                else:
                    return PythonTuple(*[self._extract_indexed_from_var(var, args[:-1], name) for var in selected_vars])

            elif isinstance(arg, Integer):

                if len(args)==1:
                    return var[arg]

                var = var[arg]
                return self._extract_indexed_from_var(var, args[:-1], name)

            else:
                errors.report(INDEXED_TUPLE, symbol=var,
                    bounding_box=self._current_fst_node.absolute_bounding_box,
                    severity='fatal', blocker=self.blocking)

        if hasattr(var, 'dtype'):
            dtype = var.dtype
            shape = var.shape
            prec  = var.precision
            order = var.order
            rank  = var.rank

            if isinstance(var, PythonTuple):
                if not var.is_homogeneous:
                    errors.report(LIST_OF_TUPLES, symbol=var,
                        bounding_box=self._current_fst_node.absolute_bounding_box,
                        severity='error', blocker=self.blocking)
                    dtype = 'int'
                else:
                    dtype = var.dtype

            return IndexedVariable(var, dtype=dtype,
                   shape=shape,prec=prec,order=order,rank=rank).__getitem__(*args)
        else:
            return IndexedVariable(name, dtype=dtype).__getitem__(args)

    def _visit_IndexedBase(self, expr, **settings):
        return self._visit(expr.label)

    def _visit_Indexed(self, expr, **settings):
        name = str(expr.base)
        var = self._visit(expr.base)

         # TODO check consistency of indices with shape/rank

        args = list(expr.indices)

        new_args = [self._visit(arg, **settings) for arg in args]

        if (len(new_args)==1 and isinstance(new_args[0],(TupleVariable, PythonTuple))):
            len_args = len(new_args[0])
            args = [self._visit(Indexed(args[0],i)) for i in range(len_args)]
        elif any(isinstance(arg,(TupleVariable, PythonTuple)) for arg in new_args):
            n_exprs = None
            for a in new_args:
                if hasattr(a,'__len__'):
                    if n_exprs:
                        assert(n_exprs)==len(a)
                    else:
                        n_exprs = len(a)
            new_expr_args = []
            for i in range(n_exprs):
                ls = []
                for j,a in enumerate(new_args):
                    if isinstance(a,TupleVariable):
                        ls.append(Indexed(args[j],i))
                    elif hasattr(a,'__getitem__'):
                        ls.append(args[j][i])
                    else:
                        ls.append(args[j])
                new_expr_args.append(ls)

            return Tuple(*[self._visit(Indexed(name,*a)) for a in new_expr_args])
        else:
            args = new_args
            len_args = len(args)

        if var.rank>len_args:
            # add missing dimensions

            args = args + [self._visit(Slice(None, None),**settings)]*(var.rank-len(args))

        return self._extract_indexed_from_var(var, args, name)

    def _visit_Symbol(self, expr, **settings):
        name = expr.name

        var = self.check_for_variable(name)

        if var is None:
            var = self.get_function(name)
        if var is None:
            var = self.get_symbolic_function(name)
        if var is None:
            var = python_builtin_datatype(name)

        if var is None:

            errors.report(UNDEFINED_VARIABLE, symbol=name,
            bounding_box=self._current_fst_node.absolute_bounding_box,
            severity='fatal', blocker=True)
        return var


    def _visit_DottedVariable(self, expr, **settings):

        first = self._visit(expr.lhs)
        rhs_name = _get_name(expr.rhs)
        attr_name = []
        if first.cls_base:
            attr_name = [i.name for i in first.cls_base.attributes]
        name = None

        # look for a class method
        if isinstance(expr.rhs, Application):

            macro = self.get_macro(rhs_name)
            if macro is not None:
                master = macro.master
                name = macro.name
                args = expr.rhs.args
                args = [expr.lhs] + list(args)
                args = [self._visit(i, **settings) for i in args]
                args = macro.apply(args)
                return FunctionCall(master, args)

            args = [self._visit(arg, **settings) for arg in
                    expr.rhs.args]
            methods = list(first.cls_base.methods) + list(first.cls_base.interfaces)
            for i in methods:
                if str(i.name) == rhs_name:
                    second = FunctionCall(i, args)
                    return DottedVariable(first, second)

        # look for a class attribute
        else:

            macro = self.get_macro(rhs_name)

            # Macro
            if isinstance(macro, MacroVariable):
                return macro.master
            elif isinstance(macro, MacroFunction):
                args = macro.apply([first])
                return FunctionCall(macro.master, args)

            # Attribute / property
            if isinstance(expr.rhs, Symbol) and first.cls_base:

                # standard class attribute
                if expr.rhs.name in attr_name:
                    self._current_class = first.cls_base
                    second = self._visit(expr.rhs, **settings)
                    self._current_class = None
                    return DottedVariable(first, second)

                # class property?
                else:
                    methods = list(first.cls_base.methods) + list(first.cls_base.interfaces)
                    for i in methods:
                        if str(i.name) == expr.rhs.name and 'property' \
                            in i.decorators.keys():
                            second = FunctionCall(i, [])
                            return DottedVariable(first, second)

        # did something go wrong?
        raise ValueError('attribute {} not found'.format(rhs_name))

    def _visit_PyccelAdd(self, expr, **settings):
        #stmts, expr = extract_subexpressions(expr)
        #stmts = []
        #if stmts:
        #    stmts = [self._visit(i, **settings) for i in stmts]
        args     = [self._visit(a, **settings) for a in expr.args]
        expr_new = expr.func(*args)
        #if stmts:
        #    expr_new = CodeBlock(stmts + [expr_new])
        return expr_new

    def _visit_PyccelMul(self, expr, **settings):
        #stmts, expr = extract_subexpressions(expr)
        #if stmts:
        #    stmts = [self._visit(i, **settings) for i in stmts]
        args = [self._visit(a, **settings) for a in expr.args]
        if isinstance(args[0], (TupleVariable, PythonTuple, Tuple, List)):
            expr_new = self._visit(Dlist(expr.args[0], expr.args[1]))
        else:
            expr_new = PyccelMul(*args)
        #if stmts:
        #    expr_new = CodeBlock(stmts + [expr_new])
        return expr_new

    def _visit_PyccelDiv(self, expr, **settings):
        #stmts, expr = extract_subexpressions(expr)
        #if stmts:
        #    stmts = [self._visit(i, **settings) for i in stmts]
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelDiv(*args)
        #if stmts:
        #    expr_new = CodeBlock(stmts + [expr_new])
        return expr_new

    def _visit_PyccelMod(self, expr, **settings):
        #stmts, expr = extract_subexpressions(expr)
        #if stmts:
        #    stmts = [self._visit(i, **settings) for i in stmts]
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelMod(*args)
        #if stmts:
        #    expr_new = CodeBlock(stmts + [expr_new])
        return expr_new

    def _visit_PyccelFloorDiv(self, expr, **settings):
        #stmts, expr = extract_subexpressions(expr)
        #if stmts:
        #    stmts = [self._visit(i, **settings) for i in stmts]
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelFloorDiv(*args)
        #if stmts:
        #    expr_new = CodeBlock(stmts + [expr_new])
        return expr_new

    def _visit_PyccelPow(self, expr, **settings):
        #stmts, expr = extract_subexpressions(expr)
        #if stmts:
        #    stmts = [self._visit(i, **settings) for i in stmts]
        args     = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelPow(*args)
        #if stmts:
        #    expr_new = CodeBlock(stmts + [expr_new])
        return expr_new

    def _visit_PyccelAssociativeParenthesis(self, expr, **settings):
        return PyccelAssociativeParenthesis(self._visit(expr.args[0]))

    def _visit_PyccelUnary(self, expr, **settings):
        return PyccelUnary(self._visit(expr.args[0]))

    def _visit_PyccelAnd(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelAnd(*args)

        return expr_new

    def _visit_PyccelOr(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelOr(*args)

        return expr_new

    def _visit_PyccelEq(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelEq(*args)
        return expr_new

    def _visit_PyccelNe(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelNe(*args)
        return expr_new

    def _visit_PyccelLt(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelLt(*args)
        return expr_new

    def _visit_PyccelGe(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelGe(*args)
        return expr_new

    def _visit_PyccelLe(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelLe(*args)
        return expr_new

    def _visit_PyccelGt(self, expr, **settings):
        args = [self._visit(a, **settings) for a in expr.args]
        expr_new = PyccelGt(*args)
        return expr_new

    def _visit_PyccelNot(self, expr, **settings):
        a = self._visit(expr.args[0], **settings)
        return PyccelNot(a)

    def _visit_Lambda(self, expr, **settings):


        expr_names = set(map(str, expr.expr.atoms(Symbol)))
        var_names = map(str, expr.variables)
        if len(expr_names.difference(var_names)) > 0:
            msg = 'Unknown variables in lambda definition'
            raise ValueError(msg)
        funcs = expr.expr.atoms(Application)
        for func in funcs:
            name = _get_name(func)
            f = self.get_symbolic_function(name)
            if f is None:
                msg = 'Unknown function in lambda definition'
                raise ValueError(msg)
            else:

                f = f(*func.args)
                expr_new = expr.expr.subs(func, f)
                expr = Lambda(tuple(expr.variables), expr_new)
        return expr

    def _visit_Application(self, expr, **settings):
        name     = type(expr).__name__
        func     = self.get_function(name)

        #stmts, new_args = extract_subexpressions(expr.args)
        #stmts = [self._visit(stmt, **settings) for stmt in stmts]

        args  = []
        for arg in expr.args:
            a = self._visit(arg, **settings)
            if isinstance(a, StarredArguments):
                args.extend(a.args_var)
            else:
                args.append(a)

        if name == 'lambdify':
            args = self.get_symbolic_function(str(expr.args[0]))
        F = pyccel_builtin_function(expr, args)

        if F is not None:
            #if len(stmts) > 0:
            #    stmts.append(F)
            #    return CodeBlock(stmts)
            return F

        elif name in self._namespace.cls_constructs.keys():

            # TODO improve the test
            # we must not invoke the namespace like this

            cls = self.get_class(name)
            d_methods = cls.methods_as_dict
            method = d_methods.pop('__init__', None)

            if method is None:

                # TODO improve case of class with the no __init__

                errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                bounding_box=self._current_fst_node.absolute_bounding_box,
                severity='error', blocker=True)
            args = expr.args
            m_args = method.arguments[1:]  # we delete the self arg

            # TODO check compatibility
            # TODO treat parametrized arguments.

            expr = ConstructorCall(method, args, cls_variable=None)
            #if len(stmts) > 0:
            #    stmts.append(expr)
            #    return CodeBlock(stmts)
            return expr
        else:

            # first we check if it is a macro, in this case, we will create
            # an appropriate FunctionCall

            macro = self.get_macro(name)
            if macro is not None:
                func = macro.master
                name = _get_name(func.name)
                args = macro.apply(args)
            else:
                func = self.get_function(name)

            if func is None:
                # TODO [SH, 25.02.2020] Report error
                errors.report(UNDEFINED_FUNCTION, symbol=name,
                bounding_box=self._current_fst_node.absolute_bounding_box,
                severity='error', blocker=self.blocking)
            else:
                if not isinstance(func, (FunctionDef, Interface)):

                    args, kwargs = split_positional_keyword_arguments(*args)
                    for a in args:
                        if getattr(a,'dtype',None) == 'tuple':
                            self._infere_type(a, **settings)
                    for a in kwargs.values():
                        if getattr(a,'dtype',None) == 'tuple':
                            self._infere_type(a, **settings)
                    expr = func(*args, **kwargs)

                    if isinstance(expr, (Where, Diag, Linspace)):
                        self.insert_variable(expr.index)

                    #if len(stmts) > 0:
                    #    stmts.append(expr)
                    #    return CodeBlock(stmts)
                    return expr
                else:
                    #if isinstance(func, Interface):
                    #    arg_dvar = [self._infere_type(i, **settings) for i in args]
                    #    f_dvar = [[self._infere_type(j, **settings)
                    #              for j in i.arguments] for i in
                    #              func.functions]
                    #    j = -1
                    #    for i in f_dvar:
                    #        j += 1
                    #        found = True
                    #        for (idx, dt) in enumerate(arg_dvar):
                    #            dtype1 = str_dtype(dt['datatype'])
                    #            dtype2 = str_dtype(i[idx]['datatype'])
                    #            found = found and (dtype1 in dtype2
                    #                          or dtype2 in dtype1)
                    #            found = found and dt['rank'] \
                    #                          == i[idx]['rank']
                    #        if found:
                    #            break
                    #
                    #    if found:
                    #        f_args = func.functions[j].arguments
                    #    else:
                    #        msg = 'function not found in the interface'
                    #        raise SystemExit(msg)

                    expr = FunctionCall(func, args)

                    #if len(stmts) > 0:
                    #    stmts.append(expr)
                    #    return CodeBlock(stmts)
                    return expr

    def _visit_Expr(self, expr, **settings):
        msg = '{expr} not yet available'
        msg = msg.format(expr=type(expr))
        raise NotImplementedError(msg)

    def _create_variable(self, name, dtype, rhs, d_lhs):

        if isinstance(rhs, (TupleVariable, PythonTuple)):
            elem_vars = []
            for i,r in enumerate(rhs):
                elem_name = self._get_new_variable_name( r, name + '_' + str(i) )
                elem_d_lhs = self._infere_type( r )

                self._ensure_target( r, elem_d_lhs )

                elem_dtype = elem_d_lhs.pop('datatype')

                var = self._create_variable(elem_name, elem_dtype, r, elem_d_lhs)
                elem_vars.append(var)

            d_lhs['is_pointer'] = any(v.is_pointer for v in elem_vars)
            lhs = TupleVariable(elem_vars, dtype, name, **d_lhs)

        else:
            lhs = Variable(dtype, name, **d_lhs)

        return lhs

    def _ensure_target(self, rhs, d_lhs):
        if isinstance(rhs, (Variable, DottedVariable)) and rhs.allocatable:
            d_lhs['allocatable'] = False
            d_lhs['is_pointer' ] = True

            # TODO uncomment this line, to make rhs target for
            #      lists/tuples.
            rhs.is_target = True
        if isinstance(rhs, IndexedElement) and rhs.rank > 0 and rhs.base.internal_variable.allocatable:
            d_lhs['allocatable'] = False
            d_lhs['is_pointer' ] = True

            # TODO uncomment this line, to make rhs target for
            #      lists/tuples.
            rhs.base.internal_variable.is_target = True

    def _assign_lhs_variable(self, lhs, d_var, rhs, **settings):

        if isinstance(lhs, Symbol):

            name = lhs.name
            dtype = d_var.pop('datatype')

            d_lhs = d_var.copy()
            # ISSUES #177: lhs must be a pointer when rhs is allocatable array
            self._ensure_target(rhs, d_lhs)

            var = self.get_variable_from_scope(name)

            # Variable not yet declared (hence array not yet allocated)
            if var is None:

                lhs = self._create_variable(name, dtype, rhs, d_lhs)

                # Add variable to scope
                self.insert_variable(lhs, name=lhs.name)

                # Not yet supported for arrays: x=y+z, x=b[:]
                # Because we cannot infer shape of right-hand side yet
                know_lhs_shape = lhs.shape or (lhs.rank == 0) \
                        or isinstance(rhs, (Variable, EmptyLike, DottedVariable))
                if not know_lhs_shape:
                    msg = "Cannot infer shape of right-hand side for expression {} = {}".format(lhs, rhs)
                    raise NotImplementedError(msg)

            else:

                # TODO improve check type compatibility
                if str(dtype) != str(var.dtype):
                    txt = '|{name}| {old} <-> {new}'
                    txt = txt.format(name=name, old=var.dtype, new=dtype)

                    errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT,
                    symbol=txt,bounding_box=self._current_fst_node.absolute_bounding_box,
                    severity='error', blocker=False)

                # in the case of elemental, lhs is not of the same dtype as
                # var.
                # TODO d_lhs must be consistent with var!
                # the following is a small fix, since lhs must be already
                # declared
                lhs = var


        elif isinstance(lhs, DottedVariable):

            dtype = d_var.pop('datatype')
            name = lhs.lhs.name
            if self._current_function == '__init__':

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
                if isinstance(rhs, Variable) and rhs.allocatable:
                    d_lhs['allocatable'] = False
                    d_lhs['is_pointer' ] = True

                    rhs.is_target = True

                member = self._create_variable(n_name, dtype, rhs, d_lhs)
                lhs    = DottedVariable(var, member)

                # update the attributes of the class and push it to the namespace
                attributes += [member]
                new_cls = ClassDef(cls_name, attributes, [], parent=parent)
                self.insert_class(new_cls, parent=True)
            else:
                lhs = self._visit_DottedVariable(lhs, **settings)
        return lhs


    def _visit_Assign(self, expr, **settings):
        # TODO unset position at the end of this part
        fst = expr.fst
        if fst:
            self._current_fst_node = fst
        else:
            msg = 'Found a node without fst member ({})'
            msg = msg.format(type(expr))
            raise PyccelSemanticError(msg)

        rhs = expr.rhs
        lhs = expr.lhs
        assigns = None

        if isinstance(rhs, Application):
            name = type(rhs).__name__
            macro = self.get_macro(name)
            if macro is None:
                rhs = self._visit(rhs, **settings)
            else:

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
                    results.append(var)

                # ...

                args = [self._visit(i, **settings) for i in
                            rhs.args]
                args = macro.apply(args, results=results)
                if isinstance(master, FunctionDef):
                    return FunctionCall(master, args)
                else:
                    msg = 'TODO treate interface case'
                    raise NotImplementedError(msg)

        elif isinstance(rhs, DottedVariable):
            var = rhs.rhs
            name = _get_name(var)
            macro = self.get_macro(name)
            if macro is None:
                rhs = self._visit_DottedVariable(rhs, **settings)
            else:
                master = macro.master
                if isinstance(macro, MacroVariable):
                    rhs = master
                    annotated_rhs = True
                else:

                    # If macro is function, create left-hand side variable
                    if isinstance(master, FunctionDef) and master.results:
                        d_var = self._infere_type(master.results[0], **settings)
                        dtype = d_var.pop('datatype')
                        lhs = Variable(dtype, lhs.name, **d_var)
                        var = self.get_variable_from_scope(lhs.name)
                        if var is None:
                            self.insert_variable(lhs)

                    name = macro.name
                    if not sympy_iterable(lhs):
                        lhs = [lhs]
                    results = []
                    for a in lhs:
                        _name = _get_name(a)
                        var = self.get_variable(_name)
                        results.append(var)

                    args = rhs.rhs.args
                    args = [rhs.lhs] + list(args)
                    args = [self._visit(i, **settings) for i in args]

                    args = macro.apply(args, results=results)

                    # Distinguish between function
                    if master.results:
                        return Assign(lhs[0], FunctionCall(master, args))
                    else:
                        return FunctionCall(master, args)
   
        else:
            rhs = self._visit(rhs, **settings)

        if isinstance(rhs, IfTernaryOperator):
            args = rhs.args
            new_args = []
            for arg in args:
                if len(arg[1].body) != 1:
                    msg = 'IfTernary body must be of length 1'
                    raise ValueError(msg)
                result = arg[1].body[0]
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

            rhs = rhs.rename(expr.lhs.name)
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
            if isinstance(lhs ,(list, tuple, PythonTuple)):
                sub = [list(zip(i,lhs)) for i in results]
            else:
                sub = [(i[0],lhs) for i in results]
            body = subs(body,sub)
            expr = Block(rhs.name, rhs.variables, body)
            return expr

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
                if self.check_for_variable(name) is None:
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

        elif isinstance(rhs, FunctionCall):

            func = rhs.funcdef
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
                    call_args = rhs.arguments
                    f_ranks = [x.rank for x in func_args]
                    c_ranks = [x.rank for x in call_args]
                    same_ranks = [x==y for (x,y) in zip(f_ranks, c_ranks)]
                    if not all(same_ranks):
                        _name = _get_name(lhs)
                        var = self.get_variable(_name)

            elif isinstance(func, Interface):
                d_var = [self._infere_type(i, **settings) for i in
                         func.functions[0].results]

                # TODO imporve this will not work for
                # the case of different results types
                d_var[0]['datatype'] = rhs.dtype

            else:
                d_var = self._infere_type(rhs, **settings)

        elif isinstance(rhs, Map):

            name = str(rhs.args[0])
            func = self.get_function(name)

            if func is None:
                errors.report(UNDEFINED_FUNCTION, symbol=name,
                bounding_box=self._current_fst_node.absolute_bounding_box,
                severity='error',blocker=self.blocking)

            dvar  = self._infere_type(rhs.args[1], **settings)
            d_var = [self._infere_type(result, **settings) for result in func.results]
            for i in range(len(d_var)):
                d_var[i]['shape'] = dvar['shape']
                d_var[i]['rank' ]  = dvar['rank']

        else:
            d_var  = self._infere_type(rhs, **settings)
            d_list = d_var if isinstance(d_var, list) else [d_var]

            for d in d_list:
                __name__ = d['datatype'].__class__.__name__

                if __name__.startswith('Pyccel'):
                    __name__ = __name__[6:]
                    d['cls_base'] = self.get_class(__name__)
                    #TODO: Avoid writing the default variables here
                    d['is_pointer'] = d_var.get('is_target',False) or d_var.get('is_pointer',False)

                    # TODO if we want to use pointers then we set target to true
                    # in the ConsturcterCall

                    d['is_polymorphic'] = False

                if isinstance(rhs, Variable) and rhs.is_target:
                    # case of rhs is a target variable the lhs must be a pointer
                    d['is_target' ] = False
                    d['is_pointer'] = True

        lhs = expr.lhs
        if isinstance(lhs, (Symbol, DottedVariable)):
            if isinstance(d_var, list):
                if len(d_var) == 1:
                    d_var = d_var[0]
                else:
                    errors.report(WRONG_NUMBER_OUTPUT_ARGS, symbol=expr,
                        bounding_box=self._current_fst_node.absolute_bounding_box,
                        severity='error', blocker=self.blocking)
                    return None

            lhs = self._assign_lhs_variable(lhs, d_var, rhs, **settings)
        elif isinstance(lhs, PythonTuple):
            n = len(lhs)
            if isinstance(rhs, PythonTuple):
                new_lhs = []
                for i,(l,r) in enumerate(zip(lhs,rhs)):
                    d = self._infere_type(r, **settings)
                    new_lhs.append( self._assign_lhs_variable(l, d, r, **settings) )
                lhs = PythonTuple(*new_lhs)

            elif isinstance(rhs, TupleVariable):
                new_lhs = []

                if rhs.is_homogeneous:
                    d_var = self._infere_type(rhs[0])
                    indexed_rhs = IndexedVariable(rhs, dtype=rhs.dtype,
                            shape=rhs.shape,prec=rhs.precision,order=rhs.order,rank=rhs.rank)
                    new_rhs = []
                    for i,l in enumerate(lhs):
                        new_lhs.append( self._assign_lhs_variable(l, d_var.copy(),
                            indexed_rhs.__getitem__(i), **settings) )
                        new_rhs.append(indexed_rhs.__getitem__(i))
                    rhs = PythonTuple(*new_rhs)
                    d_var = [d_var]
                else:
                    d_var = [self._infere_type(v) for v in rhs]
                    for i,(l,r) in enumerate(zip(lhs,rhs)):
                        new_lhs.append( self._assign_lhs_variable(l, d_var[i].copy(), r, **settings) )

                lhs = PythonTuple(*new_lhs)


            elif isinstance(d_var, list) and len(d_var)== n:
                new_lhs = []
                if hasattr(rhs,'__getitem__'):
                    for i,l in enumerate(lhs):
                        new_lhs.append( self._assign_lhs_variable(l, d_var[i].copy(), rhs[i], **settings) )
                else:
                    for i,l in enumerate(lhs):
                        new_lhs.append( self._assign_lhs_variable(l, d_var[i].copy(), rhs, **settings) )
                lhs = PythonTuple(*new_lhs)

            elif d_var['shape'][0]==n:
                new_lhs = []
                new_rhs = []

                for i,l in enumerate(lhs):
                    rhs_i = self._visit(Indexed(rhs,i))
                    new_lhs.append( self._assign_lhs_variable(l, self._infere_type(rhs_i), rhs_i, **settings) )
                    new_rhs.append(rhs_i)

                lhs = PythonTuple(*new_lhs)
                rhs = new_rhs
            else:
                errors.report(WRONG_NUMBER_OUTPUT_ARGS, symbol=expr,
                    bounding_box=self._current_fst_node.absolute_bounding_box,
                    severity='error', blocker=self.blocking)
                return None
        else:
            lhs = self._visit(lhs, **settings)

        if isinstance(rhs, (Map, Zip)):
            func  = _get_name(rhs.args[0])
            func  = UndefinedFunction(func)
            alloc = Assign(lhs, Zeros(lhs.shape, lhs.dtype))
            alloc.set_fst(fst)
            index = self._get_new_variable(expr)
            index = Variable('int',index.name)
            range_ = UndefinedFunction('range')(UndefinedFunction('len')(lhs))
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

        elif not isinstance(lhs, (list, tuple)):
            lhs = [lhs]
            if isinstance(d_var,dict):
                d_var = [d_var]

        if len(lhs) == 1:
            lhs = lhs[0]

        if isinstance(lhs, (Variable, DottedVariable)):
            is_pointer = lhs.is_pointer
        elif isinstance(lhs, IndexedElement):
            is_pointer = False
        elif isinstance(lhs, (PythonTuple, List)):
            is_pointer = any(l.is_pointer for l in lhs)

        # TODO: does is_pointer refer to any/all or last variable in list (currently last)
        is_pointer = is_pointer and isinstance(rhs, (Variable, Dlist, DottedVariable))
        is_pointer = is_pointer or isinstance(lhs, (Variable, DottedVariable)) and lhs.is_pointer

        # ISSUES #177: lhs must be a pointer when rhs is allocatable array
        if not ((isinstance(lhs, PythonTuple) or (isinstance(lhs, TupleVariable) and not lhs.is_homogeneous)) \
                and isinstance(rhs,(PythonTuple, TupleVariable, list))):
            lhs = [lhs]
            rhs = [rhs]

        new_expressions = []
        for l, r in zip(lhs,rhs):
            is_pointer_i = l.is_pointer if isinstance(l, (Variable, DottedVariable)) else is_pointer

            new_expr = Assign(l, r)

            if is_pointer_i:
                new_expr = AliasAssign(l, r)

            elif isinstance(expr, AugAssign):
                new_expr = AugAssign(l, expr.op, r)


            elif new_expr.is_symbolic_alias:
                new_expr = SymbolicAssign(l, r)

                # in a symbolic assign, the rhs can be a lambda expression
                # it is then treated as a def node

                F = self.get_symbolic_function(l)
                if F is None:
                    self.insert_symbolic_function(new_expr)
                else:
                    raise NotImplementedError('TODO')
            new_expressions.append(new_expr)
        if (len(new_expressions)==1):
            new_expressions = new_expressions[0]
            new_expressions.set_fst(fst)

            return new_expressions
        else:
            result = CodeBlock(new_expressions)
            result.set_fst(fst)
            return result

    def _visit_For(self, expr, **settings):


        self.create_new_loop_scope()

        # treatment of the index/indices
        iterable = self._visit(expr.iterable, **settings)
        body     = list(expr.body)
        iterator = expr.target

        if isinstance(iterable, Variable):
            indx   = self._get_new_variable(iterable)
            assign = Assign(iterator, IndexedBase(iterable)[indx])
            assign.set_fst(expr.fst)
            iterator = indx
            body     = [assign] + body

        elif isinstance(iterable, Map):
            indx   = self._get_new_variable(iterable)
            func   = iterable.args[0]
            args   = [IndexedBase(arg)[indx] for arg in iterable.args[1:]]
            assing = assign = Assign(iterator, func(*args))
            assign.set_fst(expr.fst)
            iterator = indx
            body     = [assign] + body

        elif isinstance(iterable, Zip):
            args = iterable.args
            indx = self._get_new_variable(args)
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
            args     = iterable.elements
            iterator = list(iterator)
            for i in range(len(args)):
                if not isinstance(args[i], Range):
                    indx   = self._get_new_variable(i)
                    assign = Assign(iterator[i], IndexedBase(args[i])[indx])

                    assign.set_fst(expr.fst)
                    body        = [assign] + body
                    iterator[i] = indx

        if isinstance(iterator, Symbol):
            name   = iterator.name
            var    = self.check_for_variable(name)
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
                   bounding_box=self._current_fst_node.absolute_bounding_box,
                   severity='error', blocker=self.blocking)

        body = [self._visit(i, **settings) for i in body]

        local_vars = list(self.namespace.variables.values())
        self.exit_loop_scope()

        if isinstance(iterable, Variable):
            return ForIterator(target, iterable, body)

        return For(target, iterable, body, local_vars=local_vars)


    def _visit_GeneratorComprehension(self, expr, **settings):

        result   = expr.expr
        lhs_name = _get_name(expr.lhs)
        lhs  = self.check_for_variable(lhs_name)

        if lhs is None:
            tmp_lhs  = Variable('int', lhs_name)
            self.insert_variable(tmp_lhs)
        else:
            tmp_lhs = None

        loops  = [self._visit(i, **settings) for i in expr.loops]
        result = self._visit(result, **settings)
        if isinstance(result, CodeBlock):
            result = result.body[-1]


        d_var = self._infere_type(result, **settings)
        dtype = d_var.pop('datatype')

        if tmp_lhs is not None:
            self.remove_variable(tmp_lhs)
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
        d_var['shape'] = PythonTuple(*shape)

        lhs_name = _get_name(expr.lhs)

        if isinstance(target, PythonTuple) and not target.is_homogeneous:
            errors.report(LIST_OF_TUPLES, symbol=expr,
                bounding_box=self._current_fst_node.absolute_bounding_box,
                severity='error', blocker=self.blocking)
            lhs      = self._create_variable(lhs_name, target[0].dtype, target, d_var)
        else:
            lhs      = self._create_variable(lhs_name, dtype, target, d_var)
        self.insert_variable(lhs)

        loops = [self._visit(i, **settings) for i in expr.loops]
        index = self._visit(index, **settings)

        return FunctionalFor(loops, lhs=lhs, indices=indices, index=index)

    def _visit_While(self, expr, **settings):

        self.create_new_loop_scope()

        test = self._visit(expr.test, **settings)
        body = self._visit(expr.body, **settings)
        local_vars = list(self.namespace.variables.values())
        self.exit_loop_scope()

        return While(test, body, local_vars)

    def _visit_If(self, expr, **settings):
        args = [self._visit(i, **settings) for i in expr.args]
        return expr.func(*args)

    def _visit_VariableHeader(self, expr, **settings):

        # TODO improve
        #      move it to the ast like create_definition for FunctionHeader?

        name  = expr.name
        d_var = expr.dtypes.copy()
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

        containers = [self.namespace.functions ,
        self.namespace.imports['functions']]
        # TODO improve test all possible containers
        name = None
        for container in containers:
            if set(expr.funcs).issubset(container.keys()):
                name  = expr.name
                funcs = []
                for i in expr.funcs:
                    funcs += [container[i]]

        if name is None:
            raise ValueError('inteface functions {} not found'.format(expr.funcs))
        expr            = Interface(name, funcs, hide=True)
        container[name] = expr
        return expr

    def _visit_Return(self, expr, **settings):

        results  = expr.expr
        assigns  = expr.stmt
        if assigns:
            assigns  = [self._visit_Assign(assign, **settings) for assign in assigns.body]
            results  = [self._visit_Symbol(i, **settings) for i in results]
            expr     = Return(results, CodeBlock(assigns))
        else:
            results = [self._visit_Symbol(i, **settings) for i in results]
            expr    = Return(results)

        return expr

    def _visit_FunctionDef(self, expr, **settings):

        name         = str(expr.name)
        name         = name.replace("'", '')
        cls_name     = expr.cls_name
        hide         = False
        kind         = 'function'
        decorators   = expr.decorators
        funcs        = []
        sub_funcs    = []
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
                   bounding_box=self._current_fst_node.absolute_bounding_box,
                   severity='error', blocker=self.blocking)

        # we construct a FunctionDef from its header
        if header:
            interfaces = header.create_definition()

            # get function kind from the header

            kind = header.kind
        else:

            # this for the case of a function without arguments => no header

            interfaces = [FunctionDef(name, [], [], [])]
#        TODO move this to codegen
#        vec_func = None
#        if 'vectorize' in decorators:
#            #TODO move to another place
#            vec_name  = 'vec_' + name
#            arg       = decorators['vectorize'][0]
#            arg       = str(arg.name)
#            args      = [str(i.name) for i in expr.arguments]
#            index_arg = args.index(arg)
#            arg       = Symbol(arg)
#            vec_arg   = IndexedBase(arg)
#            index     = self._get_new_variable(expr.body)
#            range_    = Function('range')(Function('len')(arg))
#            args      = symbols(args)
#            args[index_arg] = vec_arg[index]
#            body_vec        = Assign(args[index_arg], Function(name)(*args))
#            body_vec.set_fst(expr.fst)
#            body_vec   = [For(index, range_, [body_vec], strict=False)]
#            header_vec = header.vectorize(index_arg)
#            vec_func   = expr.vectorize(body_vec, header_vec)


        for m in interfaces:
            args        = []
            results     = []
            local_vars  = []
            global_vars = []
            imports     = []
            arg         = None
            arguments     = expr.arguments
            header_results = m.results

            self.create_new_function_scope(name)

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
                        a_new = Variable(dtype, a.name, **d_var)

                    if additional_args:
                        args += additional_args

                    args.append(a_new)
                    self.insert_variable(a_new, name=str(a_new.name))

            # find return stmt and results

            returns = self._collect_returns_stmt(expr.body)

            # Remove duplicated return expressions, because we cannot have
            # duplicated intent(out) arguments in Fortran.
            # TODO [YG, 12.03.2020]: find workaround using temporary variables

            for stmt in returns:
                results += [list(OrderedDict.fromkeys(stmt.expr))]

            if not all(i == results[0] for i in results):
                #case of multiple return
                # with different variable name
                msg = 'TODO not available yet'
                raise PyccelSemanticError(msg)

            if len(results) > 0:
                results = list(results[0])

            if header_results:
                new_results = []

                for a, ah in zip(results, header_results):
                    d_var = self._infere_type(ah, **settings)
                    dtype = d_var.pop('datatype')
                    a_new = Variable(dtype, a.name, **d_var)
                    self.insert_variable(a_new, name=str(a_new.name))
                    new_results.append(a_new)

                results = new_results

            if len(interfaces) == 1:
                # case of recursive function
                # TODO improve
                func = FunctionDef(name, args, results, [])
                self.insert_function(func)

            # we annotate the body
            body = self._visit(expr.body)

            # ISSUE 177: must update arguments to get is_target
            args    = [self.get_variable(a.name) for a in args]
            results = [self.get_variable(a.name) for a in results]

            if arg and cls_name:
                dt       = self.get_class_construct(cls_name)()
                cls_base = self.get_class(cls_name)
                var      = Variable(dt, 'self', cls_base=cls_base)
                args     = [var] + args

            for var in self.get_variables(self._namespace):
                if not var in args + results:
                    local_vars += [var]

            if 'stack_array' in decorators:

                for i in range(len(local_vars)):
                    var = local_vars[i]
                    var_name = var.name
                    if var_name in decorators['stack_array']:
                        d_var = self._infere_type(var, **settings)
                        d_var['is_stack_array'] = True
                        d_var['allocatable'] = False
                        d_var['is_pointer']  = False
                        d_var['is_target']   = False
                        dtype = d_var.pop('datatype')
                        var   = Variable(dtype, var_name, **d_var)
                        local_vars[i] = var

            # TODO should we add all the variables or only the ones used in the function
            container = self._namespace.parent_scope
            for var in self.get_variables(container):
                if not var in args + results + local_vars:
                    global_vars += [var]

            is_recursive = False

            # get the imports
            imports   = self.namespace.imports['imports'].values()
            imports   = list(set(imports))

            func_   = self.namespace.functions.pop(name, None)

            if not func_ is None and func_.is_recursive:
                is_recursive = True

            sub_funcs = [i for i in self.namespace.functions.values() if not i.is_header]

            self.exit_function_scope()
            # ... computing inout arguments
            args_inout = []
            for a in args:
                args_inout.append(False)

            results_names = [str(i) for i in results]

            assigned = get_assigned_symbols(body)
            assigned = [str(i) for i in assigned]

            apps = list(Tuple(*body.body).atoms(Application))
            apps = [i for i in apps if (i.__class__.__name__
                    in self.get_parent_functions())]

            d_apps = OrderedDict()
            for a in args:
                d_apps[a] = []

            for f in apps:
                a_args = set(f.args) & set(args)
                for a in a_args:
                    d_apps[a].append(f)

            for i,a in enumerate(args):
                if str(a) in results_names:
                    args_inout[i] = True

                elif str(a) in assigned:
                    args_inout[i] = True

                elif str(a) == 'self':
                    args_inout[i] = True

                if d_apps[a] and not( args_inout[i] ):
                    intent = False
                    n_fa = len(d_apps[a])
                    i_fa = 0
                    while not(intent) and i_fa < n_fa:
                        fa = d_apps[a][i_fa]
                        f_name = fa.__class__.__name__
                        func = self.get_function(f_name)

                        j = list(fa.args).index(a)
                        intent = func.arguments_inout[j]
                        if intent:
                            args_inout[i] = True

                        i_fa += 1
            # ...

            # Raise an error if one of the return arguments is either:
            #   a) a pointer
            #   b) array which is not among arguments, hence intent(out)
            for r in results:
                if r.is_pointer:
                    errors.report(UNSUPPORTED_ARRAY_RETURN_VALUE,
                    symbol=r,bounding_box=self._current_fst_node.absolute_bounding_box,
                    severity='fatal')
                elif (r not in args) and r.rank > 0:
                    errors.report(UNSUPPORTED_ARRAY_RETURN_VALUE,
                    symbol=r,bounding_box=self._current_fst_node.absolute_bounding_box,
                    severity='fatal')


            func = FunctionDef(name,
                    args,
                    results,
                    body,
                    local_vars=local_vars,
                    global_vars=global_vars,
                    cls_name=cls_name,
                    hide=hide,
                    kind=kind,
                    is_pure=is_pure,
                    is_elemental=is_elemental,
                    is_private=is_private,
                    imports=imports,
                    decorators=decorators,
                    is_recursive=is_recursive,
                    arguments_inout=args_inout,
                    functions = sub_funcs)

            if cls_name:
                cls = self.get_class(cls_name)
                methods = list(cls.methods) + [func]

                # update the class methods

                self.insert_class(ClassDef(cls_name, cls.attributes,
                methods, parent=cls.parent))

            funcs += [func]

            #clear the sympy cache
            #TODO clear all variable except the global ones
            cache.clear_cache()

        if len(funcs) == 1:
            funcs = funcs[0]
            self.insert_function(funcs)

        else:
            new_funcs = []
            for i,f in enumerate(funcs):
                #TODO add new scope for the interface
                self.namespace.sons_scopes[name+'_'+ str(i)] = self.namespace.sons_scopes[name]
                new_funcs.append(f.rename(name+'_'+ str(i)))

            funcs = Interface(name, new_funcs)
            self.insert_function(funcs)
#        TODO move this to codegen
#        if vec_func:
#           self._visit_FunctionDef(vec_func, **settings)
#           vec_func = self.namespace.functions.pop(vec_name)
#           if isinstance(funcs, Interface):
#               funcs = list(funcs.funcs)+[vec_func]
#           else:
#               self.namespace.sons_scopes['sc_'+ name] = self.namespace.sons_scopes[name]
#               funcs = funcs.rename('sc_'+ name)
#               funcs = [funcs, vec_func]
#           funcs = Interface(name, funcs)
#           self.insert_function(funcs)
        return EmptyLine()

    def _visit_Print(self, expr, **settings):
        args = [self._visit(i, **settings) for i in expr.expr]
        if len(args) == 0:
            return Print(args)

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
                self._visit_FunctionDef(method, **settings)
                methods.pop(i)
                const = self.namespace.functions.pop(m_name)
                break



        if not const:
            errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                   bounding_box=self._current_fst_node.absolute_bounding_box,
                   severity='error', blocker=True)

        ms = []
        for i in methods:
            self._visit_FunctionDef(i, **settings)
            m_name = str(i.name).replace("'", '')
            m = self.namespace.functions.pop(m_name)
            ms.append(m)

        methods = [const] + ms
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

        cls = ClassDef(name, attributes, methods,
              interfaces=interfaces, parent=parent)
        self.insert_class(cls)

        return EmptyLine()

    def _visit_Del(self, expr, **settings):

        ls = [self._visit(i, **settings) for i in expr.variables]
        return Del(ls)

    def _visit_Is(self, expr, **settings):

        # TODO ERROR wrong position ??

        name = expr.lhs
        var1 = self.get_variable(str(expr.lhs))

        var2 = self.check_for_variable(str(expr.rhs))
        if var2 is None:
            if (isinstance(expr.rhs, Nil) and not var1.is_optional):
                errors.report(PYCCEL_RESTRICTION_OPTIONAL_NONE,
                        bounding_box=self._current_fst_node.absolute_bounding_box,
                        severity='error', blocker=self.blocking)
            return Is(var1, expr.rhs)

        if ((var1.is_Boolean or isinstance(var1.dtype, NativeBool)) and
            (var2.is_Boolean or isinstance(var2.dtype, NativeBool))):
            return Is(var1, var2)

        errors.report(PYCCEL_RESTRICTION_IS_RHS,
            bounding_box=self._current_fst_node.absolute_bounding_box,
            severity='error', blocker=self.blocking)
        return Is(var1, expr.rhs)

    def _visit_IsNot(self, expr, **settings):

        # TODO ERROR wrong position ??

        name = expr.lhs
        var1 = self.get_variable(str(expr.lhs))

        var2 = self.check_for_variable(str(expr.rhs))
        if var2 is None:
            if (isinstance(expr.rhs, Nil) and not var1.is_optional):
                errors.report(PYCCEL_RESTRICTION_OPTIONAL_NONE,
                        bounding_box=self._current_fst_node.absolute_bounding_box,
                        severity='error', blocker=self.blocking)
            return IsNot(var1, expr.rhs)

        if ((var1.is_Boolean or isinstance(var1.dtype, NativeBool)) and
            (var2.is_Boolean or isinstance(var2.dtype, NativeBool))):
            return IsNot(var1, var2)

        errors.report(PYCCEL_RESTRICTION_IS_RHS,
        bounding_box=self._current_fst_node.absolute_bounding_box,
        severity='error', blocker=self.blocking)
        return IsNot(var1, expr.rhs)

    def _visit_Import(self, expr, **settings):

        # TODO - must have a dict where to store things that have been
        #        imported
        #      - should not use namespace

        if expr.source:
            container = self.namespace.imports

            if str(expr.source) in pyccel_builtin_import_registery:

                imports = pyccel_builtin_import(expr)
                for (name, atom) in imports:
                    if not name is None:
                        F = self.check_for_variable(name)

                        if F is None:
                            container['functions'][name] = atom
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
                __print__            = False

                # we need to use str here since source has been defined
                # using repr.
                # TODO shall we improve it?
                source = str(expr.source)

                targets = [i.target if isinstance(i,AsName) else i.name for i in expr.target]
                names = [i.name for i in expr.target]
                p       = self.d_parsers[source]
                for entry in ['variables', 'classes', 'functions']:
                    d_son = getattr(p.namespace, entry)
                    for t,n in zip(targets,names):
                        if t in d_son:
                            e = d_son[t]
                            if t == n:
                                container[entry][n] = e
                            else:
                                container[entry][n] = e.rename(n)

                self.namespace.cls_constructs.update(p.namespace.cls_constructs)
                self.namespace.macros.update(p.namespace.macros)

                # ... meta variables

                if 'ignore_at_import' in list(p.metavars.keys()):
                    __ignore_at_import__ = p.metavars['ignore_at_import']

                if 'import_all' in list(p.metavars.keys()):
                    __import_all__ = p.metavars['import_all']

                if 'module_name' in list(p.metavars.keys()):
                    __module_name__ = p.metavars['module_name']

                if 'print' in list(p.metavars.keys()):
                    __print__ = True

                if __import_all__:
                    expr = Import(__module_name__)
                    container['imports'][__module_name__] = expr

                elif __module_name__:
                    expr = Import(expr.target, __module_name__)
                    container['imports'][__module_name__] = expr

                # ...
                elif __print__ in p.metavars.keys():
                    source = str(expr.source).split('.')[-1]
                    source = 'mod_' + source
                    expr   = Import(expr.target, source=source)
                    container['imports'][source] = expr
                elif not __ignore_at_import__:
                    container['imports'][source] = expr

        return EmptyLine()



    def _visit_With(self, expr, **settings):

        domaine = self._visit(expr.test, **settings)
        parent  = domaine.cls_base
        if not parent.is_with_construct:
            msg = '__enter__ or __exit__ methods not found'
            raise ValueError(msg)

        body = self._visit(expr.body, **settings)
        return With(domaine, body, None).block



    def _visit_MacroFunction(self, expr, **settings):

        # we change here the master name to its FunctionDef

        f_name = expr.master
        header = self.get_header(f_name)
        if header is None:
            func = self.get_function(f_name)
            if func is None:
                errors.report(MACRO_MISSING_HEADER_OR_FUNC,
                symbol=f_name,severity='error', blocker=self.blocking,
                bounding_box=self._current_fst_node.absolute_bounding_box)
        else:
            interfaces = header.create_definition()

            # TODO -> Said: must handle interface

            func = interfaces[0]

        name = expr.name
        args = [self._visit(a, **settings) if isinstance(a, ValuedArgument)
                else a for a in expr.arguments]
        master_args = [self._visit(a, **settings) if isinstance(a, ValuedArgument)
                else a for a in expr.master_arguments]
        results = expr.results
        macro   = MacroFunction(name, args, func, master_args,
                                  results=results)
        self.insert_macro(macro)

        return macro

    def _visit_MacroShape(self, expr, **settings):
        return expr

    def _visit_MacroVariable(self, expr, **settings):

        master = expr.master
        if isinstance(master, DottedName):
            raise NotImplementedError('TODO')
        header = self.get_header(master)
        if header is None:
            var = self.get_variable(master)
        else:
            var = Variable(header.dtype, header.name)

                # TODO -> Said: must handle interface

        expr = MacroVariable(expr.name, var)
        self.insert_macro(expr)
        return expr

    def _visit_Dlist(self, expr, **settings):

        val = self._visit(expr.val, **settings)
        shape = self._visit(expr.length, **settings)
        if isinstance(val, (TupleVariable, PythonTuple)):
            if isinstance(val, TupleVariable):
                return PythonTuple(*(val.get_vars()*shape))
            else:
                return PythonTuple(*(val.args*shape))
        return Dlist(val[0], shape)

    def _visit_StarredArguments(self, expr, **settings):
        name = expr.args_var
        var = self._visit(name)
        assert(var.rank==1)
        return StarredArguments([self._visit(Indexed(name,i)) for i in range(var.shape[0])])

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except:
        raise ValueError('Expecting an argument for filename')

    parser = SyntaxParser(filename)
#    print(parser.namespace)
    parser = SemanticParser(parser)
#    print(parser.ast)
#    parser.view_namespace('variables')
