#!/usr/bin/python
# -*- coding: utf-8 -*-

from redbaron import RedBaron
from redbaron import StringNode, IntNode, FloatNode, ComplexNode, FloatExponantNode ,StarNode
from redbaron import NameNode
from redbaron import AssignmentNode
from redbaron import CommentNode, EndlNode
from redbaron import ComparisonNode
from redbaron import ComparisonOperatorNode
from redbaron import UnitaryOperatorNode
from redbaron import BinaryOperatorNode, BooleanOperatorNode
from redbaron import AssociativeParenthesisNode
from redbaron import DefNode
from redbaron import ClassNode
from redbaron import TupleNode, ListNode
from redbaron import CommaProxyList
from redbaron import LineProxyList
from redbaron import NodeList
from redbaron import DotProxyList
from redbaron import ReturnNode
from redbaron import PassNode
from redbaron import DefArgumentNode
from redbaron import ForNode
from redbaron import PrintNode
from redbaron import DelNode
from redbaron import DictNode, DictitemNode
from redbaron import WhileNode
from redbaron import IfelseblockNode, IfNode, ElseNode, ElifNode
from redbaron import DotNode
from redbaron import CallNode
from redbaron import CallArgumentNode
from redbaron import AssertNode
from redbaron import ExceptNode
from redbaron import FinallyNode
from redbaron import RaiseNode
from redbaron import TryNode
from redbaron import YieldNode
from redbaron import YieldAtomNode
from redbaron import BreakNode
from redbaron import GetitemNode, SliceNode
from redbaron import ImportNode, FromImportNode
from redbaron import DottedAsNameNode
from redbaron import NameAsNameNode
from redbaron import LambdaNode
from redbaron import WithNode
from redbaron import AtomtrailersNode

from pyccel.ast import NativeInteger, NativeFloat, NativeDouble, NativeComplex
from pyccel.ast import NativeBool
from pyccel.ast import NativeRange
from pyccel.ast import NativeIntegerList
from pyccel.ast import NativeFloatList
from pyccel.ast import NativeDoubleList
from pyccel.ast import NativeComplexList
from pyccel.ast import NativeList
from pyccel.ast import NativeSymbol
from pyccel.ast import String
from pyccel.ast import datatype, DataTypeFactory
from pyccel.ast import Nil
from pyccel.ast import Variable
from pyccel.ast import DottedName, DottedVariable
from pyccel.ast import Assign, AliasAssign, SymbolicAssign, AugAssign ,Assigns
from pyccel.ast import Return
from pyccel.ast import Pass
from pyccel.ast import FunctionCall, MethodCall, ConstructorCall
from pyccel.ast import FunctionDef, Interface, PythonFunction, SympyFunction
from pyccel.ast import ClassDef
from pyccel.ast import GetDefaultFunctionArg
from pyccel.ast import For
from pyccel.ast import If
from pyccel.ast import While
from pyccel.ast import Print
from pyccel.ast import SymbolicPrint
from pyccel.ast import Del
from pyccel.ast import Assert
from pyccel.ast import Comment, EmptyLine, NewLine
from pyccel.ast import Break
from pyccel.ast import Slice, IndexedVariable, IndexedElement
from pyccel.ast import FunctionHeader, ClassHeader, MethodHeader
from pyccel.ast import VariableHeader, InterfaceHeader
from pyccel.ast import MetaVariable
from pyccel.ast import MacroFunction
from pyccel.ast import Concatinate
from pyccel.ast import ValuedVariable
from pyccel.ast import Argument, ValuedArgument
from pyccel.ast import Is
from pyccel.ast import Import, TupleImport
from pyccel.ast import AsName
from pyccel.ast import AnnotatedComment
from pyccel.ast import With
from pyccel.ast import Range
from pyccel.ast import List
from pyccel.ast import builtin_function as pyccel_builtin_function
from pyccel.ast import builtin_import as pyccel_builtin_import
from pyccel.ast import builtin_import_registery as pyccel_builtin_import_registery
from pyccel.ast import Macro
from pyccel.ast import MacroShape
from pyccel.ast import construct_macro
from pyccel.ast import SumFunction

from pyccel.parser.utilities import omp_statement, acc_statement
from pyccel.parser.utilities import fst_move_directives
from pyccel.parser.utilities import reconstruct_pragma_multilines
from pyccel.parser.utilities import is_valid_filename_pyh, is_valid_filename_py
from pyccel.parser.utilities import read_file

from pyccel.parser.syntax.headers import parse as hdr_parse
from pyccel.parser.syntax.openmp import parse as omp_parse
from pyccel.parser.syntax.openacc import parse as acc_parse

from pyccel.parser.errors import (Errors, PyccelSyntaxError,
                                  PyccelSemanticError)

# TODO - remove import * and only import what we need
#      - use OrderedDict whenever it is possible
from pyccel.parser.messages import *


from sympy import Symbol, sympify
from sympy import Tuple
from sympy import NumberSymbol, Number
from sympy import Integer, Float
from sympy import Add, Mul, Pow, floor, Mod
from sympy import FunctionClass
from sympy import Lambda
from sympy.core.expr import Expr
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.containers import Dict
from sympy.core.function import Function, FunctionClass, Application
from sympy.logic.boolalg import And, Or
from sympy.logic.boolalg import true, false
from sympy.logic.boolalg import Not
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.utilities.iterables import iterable
from sympy import Sum as Summation, Heaviside, KroneckerDelta

from collections import OrderedDict

import traceback
import importlib
import pickle
import os
import sys


# Useful for very coarse version differentiation.
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3


#  ... useful functions for imports
# TODO installed modules. must ask python (working version) where the module is
#      installed
def get_filename_from_import(module):
    """Returns a valid filename with absolute path, that corresponds to the
    definition of module.
    The priority order is:
        - header files (extension == pyh)
        - python files (extension == py)
    """
    filename_pyh = '{}.pyh'.format(module)
    filename_py  = '{}.py'.format(module)

    if is_valid_filename_pyh(filename_pyh): return os.path.abspath(filename_pyh)
    if is_valid_filename_py(filename_py): return os.path.abspath(filename_py)

    source = module
    if len(module.split('.')) > 1:
        # we remove the last entry, since it can be a pyh file
        source = '.'.join(i for i in module.split('.')[:-1])
        _module = module.split('.')[-1]
        filename_pyh = '{}.pyh'.format(_module)
        filename_py  = '{}.py'.format(_module)

    try:
        package = importlib.import_module(source)
        package_dir = str(package.__path__[0])
    except:
        errors = Errors()
        errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE,
                      symbol=source,
                      severity='fatal')

    filename_pyh = os.path.join(package_dir, filename_pyh)
    filename_py  = os.path.join(package_dir, filename_py)

    if os.path.isfile(filename_pyh):
        return filename_pyh
    elif os.path.isfile(filename_py):
        return filename_py

    errors = Errors()
    errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE,
                  symbol=module,
                  severity='fatal')
#  ...

#  ...
def _get_variable_name(var):
    """."""

    if isinstance(var, (Variable, IndexedVariable)):
        return str(var)
    elif isinstance(var, IndexedElement):
        return str(var.base)

    raise NotImplementedError('Uncovered type {dtype}'.format(dtype=type(var)))
#  ...


class Parser(object):

    """ Class for a Parser."""

    def __init__(self, inputs, debug=False, headers=None, show_traceback=True):
        """Parser constructor.

        inputs: str
            filename or code to parse as a string

        debug: bool
            True if in debug mode.

        headers: list, tuple
            list of headers to append to the namespace

        show_traceback: bool
            prints Tracebacke exception if True

        """
        self._fst = None
        self._ast = None
        self._filename = None
        self._metavars = {}
        self._namespace = {}
        self._namespace['imports'] = OrderedDict()
        self._namespace['variables'] = {}
        self._namespace['classes'] = {}
        self._namespace['functions'] = {}
        self._namespace['macros'] = {}
        self._namespace['cls_constructs'] = {}
        self._namespace['symbolic_functions'] = {}
        self._namespace['python_functions'] = {}
        self._scope = {}  # represent the namespace of a function
        self._current_class = None
        self._current = None  # we use it to detect the current method or function
        # we use it to store the imports
        self._imports = {}
        # a Parser can have parents, who are importing it. imports are then its sons.
        self._parents = []
        self._sons = []
        self._d_parsers = OrderedDict()
        # the following flags give us a status on the parsing stage
        self._syntax_done = False
        self._semantic_done = False
        # current position for errors
        self._bounding_box = None
        # flag for blocking errors. if True, an error with this flag will cause
        # Pyccel to stop
        # TODO ERROR must be passed to the Parser __init__ as argument
        self._blocking = False
        # printing exception
        self._show_traceback = show_traceback

        # TODO use another name for headers
        #      => reserved keyword, or use __

        self._namespace['headers'] = {}
        if headers:
            if not isinstance(headers, dict):
                raise TypeError('Expecting a dict of headers')

            for (key, value) in list(headers.items()):
                self._namespace['headers'][key] = value

        # check if inputs is a file
        code = inputs
        if os.path.isfile(inputs):
            # we don't use is_valid_filename_py since it uses absolute path
            # file extension
            ext = inputs.split('.')[-1]
            if not(ext in ['py', 'pyh']):
                errors = Errors()
                errors.report(INVALID_FILE_EXTENSION,
                              symbol=ext,
                              severity='fatal')
                errors.check()
                raise SystemExit(0)

            code = read_file(inputs)
            self._filename = inputs

        self._code = code

        try:
            red = RedBaron(code)
        except Exception as e:
            errors = Errors()
            errors.report(INVALID_PYTHON_SYNTAX,
                          symbol='\n'+str(e),
                          severity='fatal')
            errors.check()
            raise SystemExit(0)

        red = fst_move_directives(red)
        self._fst = red

    @property
    def namespace(self):
        return self._namespace

    @property
    def headers(self):
        return self.namespace['headers']

    @property
    def imports(self):
        return self.namespace['imports']

    @property
    def functions(self):
        return self.namespace['functions']

    @property
    def variables(self):
        return self.namespace['variables']

    @property
    def classes(self):
        return self.namespace['classes']

    @property
    def python_functions(self):
        return self.namespace['python_functions']

    @property
    def symbolic_functions(self):
        return self.namespace['symbolic_functions']

    @property
    def macros(self):
        return self.namespace['macros']

    @property
    def filename(self):
        return self._filename

    @property
    def code(self):
        return self._code

    @property
    def fst(self):
        return self._fst

    @property
    def ast(self):
        if self._ast is None:
            self._ast = self.parse()
        return self._ast

    @property
    def metavars(self):
        return self._metavars

    @property
    def current_class(self):
        return self._current_class

    @property
    def syntax_done(self):
        return self._syntax_done

    @property
    def semantic_done(self):
        return self._semantic_done

    @property
    def parents(self):
        """Returns the parents parser."""
        return self._parents

    @property
    def sons(self):
        """Returns the sons parser."""
        return self._sons

    @property
    def d_parsers(self):
        """Returns the d_parsers parser."""
        return self._d_parsers

    @property
    def is_header_file(self):
        """Returns True if we are treating a header file."""
        if self.filename:
            return self.filename.split('.')[-1] == 'pyh'
        else:
            return False

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def blocking(self):
        return self._blocking

    @property
    def show_traceback(self):
        return self._show_traceback

    # TODO shall we need to export the Parser too?
    def dump(self, filename=None):
        """Dump the current ast using Pickle.

        filename: str
            output file name. if not given `name.pyccel` will be used and placed
            in the Pyccel directory ($HOME/.pyccel)
        """
        # ...
        use_home_dir = False
        if not filename:
            if not self.filename:
                raise ValueError('Expecting a filename to load the ast')

            use_home_dir = True
            name = os.path.basename(self.filename)
            filename = '{}.pyccel'.format(name)

        # check extension
        if not (filename.split('.')[-1] == 'pyccel'):
            raise ValueError('Expecting a .pyccel extension')

#        print('>>> home = ', os.environ['HOME'])
        # ...

        # we are only exporting the AST.
        f = open(filename, 'wb')
        pickle.dump(self.ast, f, protocol=2)
        f.close()
        print('> exported :', self.ast)

    # TODO shall we need to load the Parser too?
    def load(self, filename=None):
        """Load the current ast using Pickle.

        filename: str
            output file name. if not given `name.pyccel` will be used and placed
            in the Pyccel directory ($HOME/.pyccel)
        """
        # ...
        use_home_dir = False
        if not filename:
            if not self.filename:
                raise ValueError('Expecting a filename to load the ast')

            use_home_dir = True
            name = os.path.basename(self.filename)
            filename = '{}.pyccel'.format(name)

        # check extension
        if not (filename.split('.')[-1] == 'pyccel'):
            raise ValueError('Expecting a .pyccel extension')

#        print('>>> home = ', os.environ['HOME'])
        # ...

        f = open(filename, 'rb')
        self._ast = pickle.load(f)
        f.close()
        print('> loaded   :', self.ast)

    def append_parent(self, parent):
        """."""
        # TODO check parent is not in parents
        self._parents.append(parent)

    def append_son(self, son):
        """."""
        # TODO check son is not in sons
        self._sons.append(son)

    def parse(self, d_parsers=None, verbose=False):
        """converts redbaron fst to sympy ast."""

        if self.syntax_done:
            print('> syntax analysis already done')
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('syntax')

        # we add the try/except to allow the parser to find all possible errors
        try:
            ast = self._fst_to_ast(self.fst)
        except Exception as e:
            errors.check()
            if self.show_traceback:
                traceback.print_exc()
            raise SystemExit(0)

        self._ast = ast

        errors.check()
        self._syntax_done = True

        if d_parsers is None:
            d_parsers = OrderedDict()
        self._d_parsers = self._parse_sons(d_parsers, verbose=verbose)

        return ast

    def annotate(self, **settings):
        """."""
        if self.semantic_done:
            print('> semantic analysis already done')
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('semantic')

        # we first treat all sons to get imports
        verbose = settings.pop('verbose', False)
        self._annotate_parents(verbose=verbose)

        # then we treat the current file
        ast = self.ast
        # we add the try/except to allow the parser to find all possible errors
        try:
            ast = self._annotate(ast, **settings)
        except Exception as e:
            errors.check()
            if self.show_traceback:
                traceback.print_exc()
            raise SystemExit(0)

        self._ast = ast

        # in the case of a header file, we need to convert all headers to
        # FunctionDef etc ...

        if self.is_header_file:
            target = []
            for parent in self.parents:
                for  key,item in parent.imports.items():
                      if get_filename_from_import(key)==self.filename:
                          target += item
            target = set(target)
            target = target.intersection(self.headers.keys())

            for name in list(target):
                v = self.headers[name]
                if isinstance(v, FunctionHeader) and not isinstance(v, MethodHeader):
                    F = self.get_function(name)
                    if F is None:
                        interfaces = v.create_definition()
                        for F in interfaces:
                            self.insert_function(F)

                    else:
                        errors.report(IMPORTING_EXISTING_IDENTIFIED,
                                      symbol=name,
                                      blocker=True,
                                      severity='fatal')
        errors.check()
        self._semantic_done = True

        return ast

    def _parse_sons(self, d_parsers, verbose=False):
        """Recursive algorithm for syntax analysis on a given file and its
        dependencies.
        This function always terminates with an OrderedDict that contains parsers
        for all involved files.
        """
        treated = set(d_parsers.keys())
        imports = set(self.imports.keys())
        imports = imports.difference(treated)
        if not imports:
            return d_parsers

        for source in imports:
            if verbose:
                print('>>> treating :: {}'.format(source))

            # get the absolute path corresponding to source
            filename = get_filename_from_import(source)

            q = Parser(filename)
            q.parse(d_parsers=d_parsers)
            d_parsers[source] = q

        # link self to its sons
        imports = list(self.imports.keys())
        for source in imports:
            d_parsers[source].append_parent(self)
            self.append_son(d_parsers[source])

        return d_parsers

    def _annotate_parents(self, **settings):

        verbose = settings.pop('verbose', False)

        # ...
        def _update_from_son(p):
            # TODO - only import what is needed
            #      - use insert_variable etc
            for entry in ['variables', 'classes', 'functions',
                          'cls_constructs']:
                d_self = self._namespace[entry]
                d_son  = p.namespace[entry]
                for k,v in list(d_son.items()):
                    d_self[k] = v
        # ...

        # we first treat sons that have no imports
        for p in self.sons:
            if not p.sons:
                if verbose:
                    print('>>> treating :: {}'.format(p.filename))
                p.annotate(**settings)

        # finally we treat the remaining sons recursively
        for p in self.sons:
            if p.sons:
                if verbose:
                    print('>>> treating :: {}'.format(p.filename))
                p.annotate(**settings)


    def print_namespace(self):
        # TODO improve spacing
        print('------- namespace -------')
        for (k, v) in self.namespace.items():
            print('{var} \t :: \t {dtype}'.format(var=k, dtype=type(v)))
        print('-------------------------')

    def view_namespace(self, entry):
        # TODO improve
        try:
            from tabulate import tabulate

            table = []
            for (k, v) in self.namespace[entry].items():
                k_str = '{}'.format(k)
                if entry == 'imports':
                    if v is None:
                        v_str = '*'
                    else:
                        v_str = '{}'.format(v)
                elif entry == 'variables':
                    v_str = '{}'.format(type(v))
                else:
                    raise NotImplementedError('TODO')

                line = [k_str, v_str]
                table.append(line)
            headers = ['module', 'target']
#            txt = tabulate(table, headers, tablefmt="rst")
            txt = tabulate(table, tablefmt="rst")
            print(txt)

        except:
            print('------- namespace.{} -------'.format(entry))
            for (k, v) in self.namespace[entry].items():
                print('{var} \t :: \t {value}'.format(var=k, value=v))
            print('-------------------------')


    def dot(self, filename):
        """Exports sympy AST using graphviz then convert it to an image."""

        expr = self.ast
        graph_str = dotprint(expr)

        f = open(filename, 'w')
        f.write(graph_str)
        f.close()

        # name without path

        name = os.path.basename(filename)

        # name without extension

        name = os.path.splitext(name)[0]
        cmd = 'dot -Tps {name}.gv -o {name}.ps'.format(name=name)
        os.system(cmd)

    def insert_import(self, expr):
        """."""
        # TODO improve
        if not isinstance(expr, Import):
            raise TypeError('Expecting Import expression')

        # if source is not specified, imported things are treated as sources
        # TODO test if builtin import
        source = expr.source
        if source is None:
            for t in expr.target:
                name = str(t)
                self._namespace['imports'][name] = None
        else:
            source = str(source)
            if not(source in pyccel_builtin_import_registery):
                for t in expr.target:
                    name = [str(t)]
                    if not source in self._namespace['imports'].keys():
                        self._namespace['imports'][source] = []
                    self._namespace['imports'][source] += name



    def get_variable(self, name):
        """."""

        var = None
        if self.current_class:
            for i in self._current_class.attributes:
                if str(i.name) == name:
                    var = i
                    return var
        if self._current:
            if name in self._scope[self._current]['variables']:
                var = self._scope[self._current]['variables'][name]
        if isinstance(self._current, DottedName):
            if name in self._scope[self._current.name[0]]['variables']:
                var = self._scope[self._current.name[0]]['variables'
                        ][name]
        if name in self._namespace['variables']:
            var = self._namespace['variables'][name]
        return var

    def get_variables(self, source=None):
        if source == 'parent':
            if isinstance(self._current, DottedName):
                name = self._current.name[0]
            elif not self._current is None:
                return self._namespace['variables'].values()
            else:
                raise TypeError('there is no parent to extract variables from '
                                )
            return self._scope[name]['variables'].values()
        else:
            return self._scope[self._current]['variables'].values()

    def insert_variable(self, expr, name=None):
        """."""

        # TODO add some checks before

        if name is None:
            name = str(expr)
        if not isinstance(expr, Variable):
            raise TypeError('variable must be of type Variable')

        if self._current:
            self._scope[self._current]['variables'][name] = expr
        else:
            self._namespace['variables'][name] = expr

    def create_variable(self, expr):
        """."""
        import numpy as np
        name = 'result_'+str(abs(hash(expr) + np.random.randint(500)))[-4:]

        return Symbol(name)

    def get_class(self, name):
        """."""

        cls = None
        if name in self._namespace['classes']:
            cls = self._namespace['classes'][name]
        return cls

    def insert_class(self, cls):
        """."""

        if isinstance(cls, ClassDef):
            name = str(cls.name)
            self._namespace['classes'][name] = cls
        else:
            raise TypeError('Expected A class definition ')

    def get_function(self, name):
        """."""
        # TODO shall we keep the elif in _imports?

        func = None
        if name in self._namespace['functions']:
            func = self._namespace['functions'][name]
        elif name in self._imports:
            func = self._imports[name]
        return func

    def insert_function(self, func):
        """."""

        if isinstance(func , SympyFunction):
            self.insert_symbolic_function(func)
        elif isinstance(func, PythonFunction):
            self.insert_python_function(func)
        elif isinstance(func, (FunctionDef, Interface)):
            container = self._namespace['functions']
            if isinstance(self._current, DottedName):
                name = self._current.name[0]
                container = self._scope[name]['functions']
            container[str(func.name)] = func
        else:
            raise TypeError('Expected a Function definition')

    def get_symbolic_function(self, name):
        """."""
        # TODO shall we keep the elif in _imports?
        func = None
        if name in self._namespace['symbolic_functions']:
            func = self._namespace['symbolic_functions'][name]
        elif name in self._imports:
            func = self._imports[name]
        return func

    def insert_symbolic_function(self, func):
        """."""
        container = self._namespace['symbolic_functions']
        if isinstance(self._current, DottedName):
            name = self._current.name[0]
            container = self._scope[name]['symbolic_functions']
        if isinstance(func, SympyFunction):
            container[str(func.name)] = func
        elif isinstance(func, SymbolicAssign) and isinstance(func.rhs, Lambda):
            container[str(func.lhs)] = func.rhs
        else:
            raise TypeError('Expected a symbolic_function')


    def get_python_function(self, name):
        """."""
        # TODO shall we keep the elif in _imports?
        func = None
        if name in self._namespace['python_functions']:
            func = self._namespace['python_functions'][name]
        elif name in self._imports:
            func = self._imports[name]
        return func

    def insert_python_function(self, func):
        """."""
        container = self._namespace['python_functions']
        if isinstance(self._current, DottedName):
            name = self._current.name[0]
            container = self._scope[name]['python_functions']

        if isinstance(func, PythonFunction):
            container[str(func.name)] = func
        else:
            raise TypeError('Expected a python_function')


    def get_macro(self, name):
        """."""
        # TODO shall we keep the elif in _imports?
        macro = None
        if name in self._namespace['macros']:
            macro = self._namespace['macros'][name]
        # TODO uncomment
#        elif name in self._imports:
#            macro = self._imports[name]
        return macro

    def insert_macro(self, macro):
        """."""
        container = self._namespace['macros']
        if isinstance(self._current, DottedName):
            name = self._current.name[0]
            container = self._scope[name]['macros']

        if isinstance(macro, MacroFunction):
            container[str(macro.name)] = macro
        else:
            raise TypeError('Expected a macro')

    def remove(self, name):
        """."""
        #TODO improve to checkt each level of scoping
        if self._current:
            self._scope[self._current]['variables'].pop(name, None)
        else:
            self._namespace['variables'].pop(name, None)

    def update_variable(self, var, **options):
        """."""

        name = _get_variable_name(var).split(""".""")
        var = self.get_variable(name[0])
        if len(name) > 1:
            name_ = _get_variable_name(var)
            for i in var.cls_base.attributes:
                if str(i.name) == name[1]:
                    var = i
            name = name_
        else:
            name = name[0]
        if var is None:
            raise ValueError('Undefined variable {name}'.format(name=name))

        # TODO implement a method inside Variable

        d_var = self._infere_type(var)
        for (key, value) in options.items():
            d_var[key] = value
        dtype = d_var.pop('datatype')
        var = Variable(dtype, name, **d_var)
        self.insert_variable(var, name)  # TODO improve to insert in the right namespace
        return var

    def get_header(self, name):
        """."""

        if name in self.headers:
            return self.headers[name]
        return None

    def get_class_construct(self, name):
        """Returns the class datatype for name."""

        return self._namespace['cls_constructs'][name]

    def set_class_construct(self, name, value):
        """Sets the class datatype for name."""

        self._namespace['cls_constructs'][name] = value

    def set_current_fun(self, name):
        """."""

        if name:
            if self._current:
                name = DottedName(self._current, name)
                self._current = name
            self._scope[name] = {}
            self._scope[name]['variables'] = {}
            self._scope[name]['functions'] = {}
            self._scope[name]['symbolic_functions'] = {}
            self._scope[name]['python_functions'] = {}
            self._scope[name]['macros'] = {}
        else:
            self._scope.pop(self._current)
            if isinstance(self._current, DottedName):

                # case of a functiondef in a another function

                name = self._current.name[0]
        self._current = name

    def insert_header(self, expr):
        """."""

        if isinstance(expr, MethodHeader):
            self._namespace['headers'][str(expr.name)] = expr
        if isinstance(expr, FunctionHeader):
            self._namespace['headers'][str(expr.func)] = expr
        elif isinstance(expr, ClassHeader):
            self._namespace['headers'][str(expr.name)] = expr

            #  create a new Datatype for the current class

            iterable = 'iterable' in expr.options
            with_construct = 'with' in expr.options
            dtype = DataTypeFactory(str(expr.name), '_name',
                                    is_iterable=iterable,
                                    is_with_construct=with_construct)
            self.set_class_construct(str(expr.name), dtype)
        else:
            raise TypeError('header of type{0} is not supported'.format(str(type(expr))))

    def _collect_returns_stmt(self,ast):
        vars_ = []
        for stmt in ast:
            if isinstance(stmt, (For,While)):
                vars_ += self._collect_returns_stmt(stmt.body)
            elif isinstance(stmt, If):
                vars_ += self._collect_returns_stmt(stmt.bodies)
            elif isinstance(stmt, Return):
                vars_ +=[stmt]

        return vars_

    def _fst_to_ast(self, stmt):
        """Creates AST from FST."""

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors
        errors = Errors()

        # ...
        def _treat_iterable(stmt):
            """
            since redbaron puts the first comments after a block statement
            inside the block, we need to remove them. this is in particular the
            case when using openmp/openacc pragmas like #$ omp end loop
            """
            ls = [self._fst_to_ast(i) for i in stmt]

            if isinstance(stmt, (list, ListNode)):
                return List(*ls)

            else:
                return Tuple(*ls)
        # ...

        if isinstance(stmt, (RedBaron,
                             LineProxyList,
                             CommaProxyList,
                             NodeList,
                             TupleNode,
                             ListNode,
                             tuple,
                             list)):
            return _treat_iterable(stmt)

        elif isinstance(stmt, DottedAsNameNode):
            names = []
            for a in stmt.value:
                names.append(str(a.value))

            if len(names) == 1:
                return names[0]

            else:
                return DottedName(*names)

        elif isinstance(stmt, NameAsNameNode):
            if not isinstance(stmt.value, str):
                raise TypeError('Expecting a string')

            value = str(stmt.value)
            if not stmt.target:
                return value

            old = value
            new = self._fst_to_ast(stmt.target)

            # TODO improve

            if isinstance(old, str):
                old = old.replace("'", '')
            if isinstance(new, str):
                new = new.replace("'", '')
            return AsName(new, old)

        elif isinstance(stmt, DictNode):

            d = {}
            for i in stmt.value:
                if not isinstance(i, DictitemNode):
                    raise PyccelSyntaxError('Expecting a DictitemNode')

                key = self._fst_to_ast(i.key)
                value = self._fst_to_ast(i.value)

                # sympy does not allow keys to be strings
                if isinstance(key, str):
                    errors.report(SYMPY_RESTRICTION_DICT_KEYS,
                                  severity='error')

                d[key] = value
            return Dict(d)

        elif stmt is None:
            return Nil()

        elif isinstance(stmt, str):
            return repr(stmt)

        elif isinstance(stmt, StringNode):
            return String(stmt.value)

        elif isinstance(stmt, IntNode):
            return Integer(stmt.value)

        elif isinstance(stmt, FloatNode):
            return Float(stmt.value)

        elif isinstance(stmt, FloatExponantNode):
            return Float(stmt.value)

        elif isinstance(stmt, ComplexNode):
            raise NotImplementedError('ComplexNode not yet available')

        elif isinstance(stmt, AssignmentNode):
            lhs = self._fst_to_ast(stmt.target)
            rhs = self._fst_to_ast(stmt.value)
            if stmt.operator in ['+', '-', '*', '/']:
                expr = AugAssign(lhs, stmt.operator, rhs)
            else:
                expr = Assign(lhs, rhs)
                # we set the fst to keep track of needed information for errors
            expr.set_fst(stmt)
            return expr

        elif isinstance(stmt, NameNode):
            if isinstance(stmt.previous, DotNode):
                return self._fst_to_ast(stmt.previous)

            if isinstance(stmt.next, GetitemNode):
                return self._fst_to_ast(stmt.next)

            if stmt.value == 'None':
                return Nil()

            elif stmt.value == 'True':
                return true

            elif stmt.value == 'False':
                return false

            else:
                return Symbol(str(stmt.value))

        elif isinstance(stmt, ImportNode):
            if not(isinstance(stmt.parent, (RedBaron, DefNode))):
                errors.report(PYCCEL_RESTRICTION_IMPORT,
                              bounding_box=stmt.absolute_bounding_box,
                              severity='error')

            if isinstance(stmt.parent, DefNode):
                errors.report(PYCCEL_RESTRICTION_IMPORT_IN_DEF,
                              bounding_box=stmt.absolute_bounding_box,
                              severity='error')

            # in an import statement, we can have seperate target by commas
            ls = self._fst_to_ast(stmt.value)
            expr = Import(ls)
            self.insert_import(expr)
            expr.set_fst(stmt)
            return expr

        elif isinstance(stmt, FromImportNode):
            if not(isinstance(stmt.parent, (RedBaron, DefNode))):
                errors.report(PYCCEL_RESTRICTION_IMPORT,
                              bounding_box=stmt.absolute_bounding_box,
                              severity='error')

            source = self._fst_to_ast(stmt.value)
            if isinstance(source, DottedVariable):
                source = DottedName(*source.names)

            targets = []
            for i in stmt.targets:
                s = self._fst_to_ast(i)
                if s == '*':
                    errors.report(PYCCEL_RESTRICTION_IMPORT_STAR,
                                  bounding_box=stmt.absolute_bounding_box,
                                  severity='error')

                targets.append(s)

            expr = Import(targets, source=source)
            self.insert_import(expr)
            expr.set_fst(stmt)
            return expr

        elif isinstance(stmt, DelNode):
            arg = self._fst_to_ast(stmt.value)
            return Del(arg)

        elif isinstance(stmt, UnitaryOperatorNode):
            target = self._fst_to_ast(stmt.target)
            if stmt.value == 'not':
                return Not(target)

            elif stmt.value == '+':
                return target

            elif stmt.value == '-':
                return -target

            elif stmt.value == '~':
                errors.report(PYCCEL_RESTRICTION_UNARY_OPERATOR,
                              bounding_box=stmt.absolute_bounding_box,
                              severity='error')
            else:
                raise PyccelSyntaxError('unknown/unavailable unary operator {node}'.format(node=type(stmt.value)))

        elif isinstance(stmt, (BinaryOperatorNode, BooleanOperatorNode)):

            first = self._fst_to_ast(stmt.first)
            second = self._fst_to_ast(stmt.second)
            if stmt.value == '+':
                if isinstance(first, (String, List)) or isinstance(second,
                        (String, List)):
                    return Concatinate(first, second)

                return Add(first, second)

            elif stmt.value == '*':
                return Mul(first, second)

            elif stmt.value == '-':
                second = Mul(-1, second)
                return Add(first, second)

            elif stmt.value == '/':
                second = Pow(second, -1)
                return Mul(first, second)

            elif stmt.value == 'and':
                return And(first, second)

            elif stmt.value == 'or':
                return Or(first, second)

            elif stmt.value == '**':
                return Pow(first, second)

            elif stmt.value == '//':
                second = Pow(second, -1)
                return floor(Mul(first, second))

            elif stmt.value == '%':
                return Mod(first, second)

            else:
                raise PyccelSyntaxError('unknown/unavailable binary operator {node}'.format(node=type(stmt.value)))

        elif isinstance(stmt, ComparisonOperatorNode):
            if stmt.first == '==':
                return '=='

            elif stmt.first == '!=':
                return '!='

            elif stmt.first == '<':
                return '<'

            elif stmt.first == '>':
                return '>'

            elif stmt.first == '<=':
                return '<='

            elif stmt.first == '>=':
                return '>='

            elif stmt.first == 'is':
                return 'is'

            else:
                raise PyccelSyntaxError('unknown comparison operator {}'.format(stmt.first))

        elif isinstance(stmt, ComparisonNode):

            first = self._fst_to_ast(stmt.first)
            second = self._fst_to_ast(stmt.second)
            op = self._fst_to_ast(stmt.value)
            if op == '==':
                return Eq(first, second)

            elif op == '!=':
                return Ne(first, second)

            elif op == '<':
                return Lt(first, second)

            elif op == '>':
                return Gt(first, second)

            elif op == '<=':
                return Le(first, second)

            elif op == '>=':
                return Ge(first, second)

            elif op == 'is':
                return Is(first, second)

            else:
                raise PyccelSyntaxError('unknown/unavailable binary operator {node}'.format(node=type(op)))

        elif isinstance(stmt, PrintNode):
            expr = self._fst_to_ast(stmt.value)
            return Print(expr)

        elif isinstance(stmt, AssociativeParenthesisNode):
            return self._fst_to_ast(stmt.value)

        elif isinstance(stmt, DefArgumentNode):
            name = self._fst_to_ast(stmt.target)
            arg = Argument(str(name))
            if stmt.value is None:
                return arg

            else:
                value = self._fst_to_ast(stmt.value)
                return ValuedArgument(arg, value)

        elif isinstance(stmt, ReturnNode):
            return Return(self._fst_to_ast(stmt.value))

        elif isinstance(stmt, PassNode):
            return Pass()

        elif isinstance(stmt, DefNode):
            #  TODO check all inputs and which ones should be treated in stage 1 or 2
            if isinstance(stmt.parent, ClassNode):
                cls_name = stmt.parent.name
            else:
                cls_name = None

            name = self._fst_to_ast(stmt.name)
            name = name.replace("'", '')
            arguments = self._fst_to_ast(stmt.arguments)
            results = []
            local_vars = []
            global_vars = []
            hide = False
            kind = 'function'
            imports = []
            decorators = [i.value.value[0].value for i in stmt.decorators]  # TODO improve later
            body = stmt.value

            if 'sympy' in decorators:
                # TODO maybe we should run pylint here
                stmt.decorators.pop()
                func = SympyFunction(name, arguments, [], [stmt.__str__()])
                self.insert_function(func)
                return EmptyLine()
            elif 'python' in decorators:
                # TODO maybe we should run pylint here
                stmt.decorators.pop()
                func = PythonFunction(name, arguments, [], [stmt.__str__()])
                self.insert_function(func)
                return EmptyLine()
            else:
                body = self._fst_to_ast(body)

            return FunctionDef(name,
                               arguments,
                               results,
                               body,
                               local_vars=local_vars,
                               global_vars=global_vars,
                               cls_name=cls_name,
                               hide=hide,
                               kind=kind,
                               imports=imports,
                               decorators=decorators)

        elif isinstance(stmt, ClassNode):
            name = self._fst_to_ast(stmt.name)
            methods = [i for i in stmt.value if isinstance(i, DefNode)]
            methods = self._fst_to_ast(methods)
            attributes = methods[0].arguments
            parent = [i.value for i in stmt.inherit_from]
            expr = ClassDef(name=name,
                            attributes=attributes,
                            methods=methods,
                            parent=parent)
            # we set the fst to keep track of needed information for errors
            expr.set_fst(stmt)
            return expr

        elif isinstance(stmt, AtomtrailersNode):
            return self._fst_to_ast(stmt.value)

        elif isinstance(stmt, GetitemNode):
            parent = stmt.parent
            args = self._fst_to_ast(stmt.value)
            if isinstance(stmt.previous.previous, DotNode):
                return self._fst_to_ast(stmt.previous.previous)

             # elif isinstance(stmt.previous,GetitemNode):
             #    return self._fst_to_ast(stmt.previous.previous)

            name = Symbol(str(stmt.previous.value))
            stmt.parent.remove(stmt.previous)
            stmt.parent.remove(stmt)
            if not hasattr(args, '__iter__'):
                args = [args]
            args = tuple(args)
            return IndexedBase(name)[args]

        elif isinstance(stmt, SliceNode):
            upper = self._fst_to_ast(stmt.upper)
            lower = self._fst_to_ast(stmt.lower)
            if upper and lower:
                return Slice(lower, upper)

            elif lower:
                return Slice(lower, None)

            elif upper:
                return Slice(None, upper)

        elif isinstance(stmt, DotProxyList):
            return self._fst_to_ast(stmt[-1])

        elif isinstance(stmt, DotNode):

            suf = stmt.next
            pre = self._fst_to_ast(stmt.previous)
            if stmt.previous:
                stmt.parent.value.remove(stmt.previous)
            suf = self._fst_to_ast(suf)
            return DottedVariable(pre, suf)

        elif isinstance(stmt, CallNode):
            args = self._fst_to_ast(stmt.value)
            # TODO we must use self._fst_to_ast(stmt.previous.value)
            #      but it is not working for the moment
            f_name = str(stmt.previous.value)
            if len(args) == 0:
                args = ((), )
            func = Function(f_name)(*args)
            parent = stmt.parent
            if stmt.previous.previous \
                and isinstance(stmt.previous.previous, DotNode):
                parent.value.remove(stmt.previous)
                parent.value.remove(stmt)
                pre = self._fst_to_ast(stmt.parent)
                return DottedVariable(pre, func)

            else:
                return func

        elif isinstance(stmt, CallArgumentNode):
            return self._fst_to_ast(stmt.value)

        elif isinstance(stmt, ForNode):
            iterator = self._fst_to_ast(stmt.iterator)
            iterable = self._fst_to_ast(stmt.target)
            body = self._fst_to_ast(stmt.value)
            expr = For(iterator, iterable, body, strict=False)
            return expr

        elif isinstance(stmt, IfelseblockNode):
            args = self._fst_to_ast(stmt.value)
            return If(*args)

        elif isinstance(stmt, (IfNode, ElifNode)):
            test = self._fst_to_ast(stmt.test)
            body = self._fst_to_ast(stmt.value)
            return Tuple(test, body)

        elif isinstance(stmt, ElseNode):
            test = True
            body = self._fst_to_ast(stmt.value)
            return Tuple(test, body)

        elif isinstance(stmt, WhileNode):
            test = self._fst_to_ast(stmt.test)
            body = self._fst_to_ast(stmt.value)
            return While(test, body)

        elif isinstance(stmt, AssertNode):
            expr = self._fst_to_ast(stmt.value)
            return Assert(expr)

        elif isinstance(stmt, EndlNode):
            return NewLine()

        elif isinstance(stmt, CommentNode):
            # if annotated comment
            if stmt.value.startswith('#$'):
                env = stmt.value[2:].lstrip()
                if env.startswith('omp'):
                    txt = reconstruct_pragma_multilines(stmt)
                    return omp_parse(stmts=txt)

                elif env.startswith('acc'):
                    txt = reconstruct_pragma_multilines(stmt)
                    return acc_parse(stmts=txt)

                elif env.startswith('header'):
                    txt = reconstruct_pragma_multilines(stmt)
                    expr = hdr_parse(stmts=txt)
                    if isinstance(expr, MetaVariable):
                        # a metavar will not appear in the semantic stage.
                        # but can be used to modify the ast
                        self._metavars[str(expr.name)] = str(expr.value)
                        #return NewLine()
                        expr = EmptyLine()
                    else:
                        expr.set_fst(stmt)

                    return expr

                else:
                    # TODO an info should be reported saying that either we
                    # found a multiline pragma or an invalid pragma statement
                    return NewLine()
#                    errors.report(PYCCEL_INVALID_HEADER,
#                                  bounding_box=stmt.absolute_bounding_box,
#                                  severity='error')

            else:
                # TODO improve
                txt = stmt.value[1:].lstrip()
                return Comment(txt)

        elif isinstance(stmt, BreakNode):
            return Break()

        elif isinstance(stmt, StarNode):
            return '*'

        elif isinstance(stmt, LambdaNode):
            expr = self._fst_to_ast(stmt.value)
            args = []

            for i in stmt.arguments:
                var = self._fst_to_ast(i.name)
                args += [var]

            return Lambda(args, expr)
        elif isinstance(stmt, WithNode):
            domain = self._fst_to_ast(stmt.contexts[0].value)
            body   = self._fst_to_ast(stmt.value)
            settings = None
            return With(domain, body, settings)

        elif isinstance(stmt, (ExceptNode, FinallyNode, TryNode)):
            # this is a blocking error, since we don't want to convert the try body
            errors.report(PYCCEL_RESTRICTION_TRY_EXCEPT_FINALLY,
                          bounding_box=stmt.absolute_bounding_box,
                          severity='error')

        elif isinstance(stmt, RaiseNode):
            errors.report(PYCCEL_RESTRICTION_RAISE,
                          bounding_box=stmt.absolute_bounding_box,
                          severity='error')

        elif isinstance(stmt, (YieldNode, YieldAtomNode)):
            errors.report(PYCCEL_RESTRICTION_YIELD,
                          bounding_box=stmt.absolute_bounding_box,
                          severity='error')

        else:
            raise PyccelSyntaxError('{node} not yet available'.format(node=type(stmt)))


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
        #d_var['datatype'] = None
        d_var['datatype'] = NativeSymbol()
        d_var['allocatable'] = None
        d_var['shape'] = None
        d_var['rank'] = 0
        d_var['is_pointer'] = None
        d_var['is_target'] = None
        d_var['is_polymorphic'] = None
        d_var['is_optional'] = None
        d_var['cls_base'] = None
        d_var['cls_parameters'] = None

        # TODO improve => put settings as attribut of Parser

        DEFAULT_FLOAT = settings.pop('default_float', 'double')

        if isinstance(expr, type(None)):
            return d_var

        elif isinstance(expr, Integer):
            d_var['datatype'] = 'int'
            d_var['allocatable'] = False
            d_var['rank'] = 0
            return d_var

        elif isinstance(expr, Float):
            d_var['datatype'] = DEFAULT_FLOAT
            d_var['allocatable'] = False
            d_var['rank'] = 0
            return d_var

        elif isinstance(expr, String):
            d_var['datatype'] = 'str'
            d_var['allocatable'] = False
            d_var['rank'] = 0
            return d_var

        elif isinstance(expr, Variable):
            name = expr.name
            var = self.get_variable(name)
            if var is None:
                var = expr

            d_var['datatype'] = var.dtype
            d_var['allocatable'] = var.allocatable
            d_var['shape'] = var.shape
            d_var['rank'] = var.rank
            d_var['cls_base'] = var.cls_base
            d_var['is_pointer'] = var.is_pointer
            d_var['is_polymorphic'] = var.is_polymorphic
            d_var['is_optional'] = var.is_optional
            d_var['is_target'] = var.is_target
            return d_var

        elif isinstance(expr, (BooleanTrue, BooleanFalse)):
            d_var['datatype'] = NativeBool()
            d_var['allocatable'] = False
            d_var['is_pointer'] = False
            d_var['rank'] = 0
            return d_var

        elif isinstance(expr, IndexedElement):
            d_var['datatype'] = expr.dtype
            name = str(expr.base)
            var = self.get_variable(name)
            if var is None:
                raise ValueError('Undefined variable {name}'.format(name=name))

            d_var['datatype'] = var.dtype

            if iterable(var.shape):
                shape = []
                for (s, i) in zip(var.shape, expr.indices):
                    if isinstance(i, Slice):
                        shape.append(i)
            else:
                shape = None

            rank = max(0, var.rank - expr.rank)
            if rank > 0:
                d_var['allocatable'] = var.allocatable

            d_var['shape'] = shape
            d_var['rank'] = rank

#            # TODO pointer or allocatable case
#            d_var['is_pointer'] = var.is_pointer
#            d_var['allocatable'] = var.allocatable

            return d_var

        elif isinstance(expr, IndexedVariable):
            name = str(expr)
            var = self.get_variable(name)
            if var is None:
                raise ValueError('Undefined variable {name}'.format(name=name))

            d_var['datatype'] = var.dtype
            d_var['allocatable'] = var.allocatable
            d_var['shape'] = var.shape
            d_var['rank'] = var.rank
            return d_var

        elif isinstance(expr, Range):
            d_var['datatype'] = NativeRange()
            d_var['allocatable'] = False
            d_var['shape'] = None
            d_var['rank'] = 0
            d_var['cls_base'] = expr  # TODO: shall we keep it?
            return d_var

        elif isinstance(expr, Is):
            d_var['datatype'] = NativeBool()
            d_var['allocatable'] = False
            d_var['is_pointer'] = False
            d_var['rank'] = 0
            return d_var

        elif isinstance(expr, DottedVariable):
            if isinstance(expr.args[0], DottedVariable):
                self._current_class = expr.args[0].args[1].cls_base
            else:
                self._current_class = expr.args[0].cls_base
            d_var = self._infere_type(expr.args[1])
            self._current_class = None
            return d_var

        elif isinstance(expr, Lambda):
            d_var['datatype'] = NativeSymbol()
            d_var['allocatable'] = False
            d_var['is_pointer'] = False
            d_var['rank'] = 0
            return d_var

        elif isinstance(expr, Expr):
            #ds = [self._infere_type(i, **settings) for i in expr.atoms(Symbol)]
            #TODO should we keep it or use the other
            ds = [self._infere_type(i, **settings) for i in expr.args]
            dtypes = [d['datatype'] for d in ds]
            allocatables = [d['allocatable'] for d in ds]
            pointers = [d['is_pointer'] or d['is_target'] for d in ds]
            ranks = [d['rank'] for d in ds]
            shapes = [d['shape'] for d in ds]

            # TODO improve
            # ... only scalars and variables of rank 0 can be handled
            if ranks:
                r_min = min(ranks)
                r_max = max(ranks)
                if not r_min == r_max:
                    if not r_min == 0:
                        raise ValueError('cannot process arrays of different ranks.'
                                )
                rank = r_max
            else:
                rank = 0
            # ...

            # ...
            shape = None
            for s in shapes:
                if s:
                    shape = s
            # ...

            d_var['datatype'] = 'int'

            # TODO imporve to hadle all possible types (complex ,ndarray , ...)
            for i in dtypes:
                if isinstance(i, str):
                    if i == 'float' or i == 'double':
                        d_var['datatype'] = i
                        break
                elif isinstance(i, (NativeDouble, NativeFloat)):
                    d_var['datatype'] = i
                    break

            d_var['allocatable'] = any(allocatables)
            d_var['is_pointer'] = any(pointers)
            d_var['shape'] = shape
            d_var['rank'] = rank
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
                dtype = datatype(d['datatype'])
                if isinstance(dtype, NativeInteger):
                    d_var['datatype'] = NativeIntegerList()
                elif isinstance(dtype, NativeFloat):
                    d_var['datatype'] = NativeFloatList()
                elif isinstance(dtype, NativeDouble):
                    d_var['datatype'] = NativeDoubleList()
                elif isinstance(dtype, NativeComplex):
                    d_var['datatype'] = NativeComplexList()
                else:
                    raise NotImplementedError('TODO')
            return d_var
        elif isinstance(expr, Concatinate):
            d_var_left = self._infere_type(expr.left, **settings)
            d_var_right = self._infere_type(expr.right, **settings)
            import operator
            if not (d_var_left['datatype'] == 'str'
                    or d_var_right['datatype'] == 'str'):
                d_var_left['shape'] = tuple(map(operator.add,
                        d_var_right['shape'], d_var_left['shape']))
            return d_var_left
        else:
            raise NotImplementedError('{expr} not yet available'.format(expr=type(expr)))

    def _annotate(self, expr, **settings):
        """Annotates the AST.

        IndexedVariable atoms are only used to manipulate expressions, we then,
        always have a Variable in the namespace."""

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        errors = Errors()

        if isinstance(expr, (list, tuple, Tuple)):
            ls = []
            for i in expr:
                a = self._annotate(i, **settings)
                ls.append(a)
            if isinstance(expr, List):
                return List(*ls)

            else:
                return Tuple(*ls)

        elif isinstance(expr, (Integer, Float, String)):
            return expr

        elif isinstance(expr,NumberSymbol) or isinstance(expr,Number):
            return sympify(float(expr))

        elif isinstance(expr, (BooleanTrue, BooleanFalse)):
            return expr

        elif isinstance(expr, Variable):
            name = expr.name
            var = self.get_variable(name)
            if var is None:
                # TODO ERROR not tested yet
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                              bounding_box=self.bounding_box,
                              severity='error', blocker=self.blocking)
            return var

        elif isinstance(expr, str):
            return repr(expr)

        elif isinstance(expr, (IndexedVariable, IndexedBase)):
            # an indexed variable is only defined if the associated variable is in
            # the namespace
            name = str(expr.name)
            var = self.get_variable(name)
            if var is None:
                # TODO ERROR not tested yet
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                              bounding_box=self.bounding_box,
                              severity='error', blocker=self.blocking)
            dtype = var.dtype

            # TODO add shape
            return IndexedVariable(name, dtype=dtype)

        elif isinstance(expr, (IndexedElement, Indexed)):
            name = str(expr.base)
            var = self.get_variable(name)
            if var is None:
                errors.report(UNDEFINED_INDEXED_VARIABLE, symbol=name,
                              bounding_box=self.bounding_box,
                              severity='error', blocker=self.blocking)

            # TODO check consistency of indices with shape/rank
            args = tuple(expr.indices)
            # case of Pyccel ast Variable, IndexedVariable
            # if not possible we use symbolic objects
            if hasattr(var, 'dtype'):
                dtype = var.dtype
                return IndexedVariable(name, dtype=dtype).__getitem__(*args)
            else:
                return IndexedBase(name).__getitem__(args)


        elif isinstance(expr, Symbol):
            name = str(expr.name)
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

        elif isinstance(expr, DottedVariable):
            first = self._annotate(expr.args[0])
            attr_name = [i.name for i in first.cls_base.attributes]
            if isinstance(expr.args[1], Symbol) \
                and not expr.args[1].name in attr_name:
                for i in first.cls_base.methods:
                    if str(i.name) == expr.args[1].name and 'property' \
                        in i.decorators:
                        second = FunctionCall(i, [], kind=i.kind)
                        return DottedVariable(first, second)

            if not isinstance(expr.args[1], Function):
                self._current_class = first.cls_base
                second = self._annotate(expr.args[1])
                self._current_class = None
            else:
                for i in first.cls_base.methods:
                    if str(i.name) == str(type(expr.args[1]).__name__):
                        args = [self._annotate(arg) for arg in
                                expr.args[1].args]
                        if args == ((),) or args ==[()]:
                            args = []
                        second = FunctionCall(i, args, kind=i.kind)
            return DottedVariable(first, second)

        elif isinstance(expr, (Add, Mul, Pow, And, Or,
                                Eq, Ne, Lt, Gt, Le, Ge)):
            # we reconstruct the arithmetic expressions using the annotated
            # arguments
            args = expr.args

            # we treat the first element
            a = args[0]
            a_new = self._annotate(a, **settings)
            expr_new = a_new

            # then we treat the rest
            for a in args[1:]:
                a_new = self._annotate(a, **settings)
                if isinstance(expr, (Add, Mul)):
                    expr_new = expr._new_rawargs(expr_new, a_new)
                elif isinstance(expr, Pow):
                    expr_new = Pow(expr_new, a_new)
                elif isinstance(expr, And):
                    expr_new = And(expr_new, a_new)
                elif isinstance(expr, Or):
                    expr_new = Or(expr_new, a_new)
                elif isinstance(expr, Eq):
                    expr_new = Eq(expr_new, a_new)
                elif isinstance(expr, Ne):
                    expr_new = Ne(expr_new, a_new)
                elif isinstance(expr, Lt):
                    expr_new = Lt(expr_new, a_new)
                elif isinstance(expr, Le):
                    expr_new = Le(expr_new, a_new)
                elif isinstance(expr, Gt):
                    expr_new = Gt(expr_new, a_new)
                elif isinstance(expr, Ge):
                    expr_new = Ge(expr_new, a_new)
            return expr_new

        elif isinstance(expr, Lambda):
            expr_names =set(map(str,expr.expr.atoms(Symbol)))
            var_names = map(str, expr.variables)
            if len(expr_names.difference(var_names))>0:
                raise ValueError('Unknown variables in lambda definition ')
            funcs = expr.expr.atoms(Function)
            for func in funcs:
                name = str(type(func).__name__)
                f = self.get_symbolic_function(name)
                if f is None:
                    raise ValueError('Unknown function in lambda definition')
                else:
                    if isinstance(f, SympyFunction):
                        f = FunctionCall(f,func.args)
                    else:
                        f =  f(*func.args)
                    expr_new = expr.expr.subs(func, f)
                    expr = Lambda(expr.variables, expr_new)
            return expr

        elif isinstance(expr, (Application)):
            # ... DEBUG
            name = str(type(expr).__name__)

            func = self.get_function(name)

            # ...
            if not isinstance(func, Lambda):
                args = [self._annotate(i, **settings) for i in expr.args]
            else:
                # here args are sympy symbol
                args = expr.args
            # ...
            if name== 'lambdify':
                args = self.get_symbolic_function(str(expr.args[0]))
            F = pyccel_builtin_function(expr, args)
            if F:
                return F

            elif name in self._namespace['cls_constructs'].keys():
                # TODO improve the test
                #      we must not invoke the namespace like this, only through
                #      appropriate methods like get_variable ...
                cls = self.get_class(name)
                d_methods = cls.methods_as_dict
                method = d_methods.pop('__init__', None)

                if method is None:
                    # TODO improve
                    #      we should not stop here, there will be cases where we
                    #      want to instantiate a class that has no __init__
                    #      construct
                    errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                                  bounding_box=self.bounding_box,
                                  severity='error', blocker=True)
                args = expr.args
                m_args = method.arguments[1:]  # we delete the self arg

                # TODO check compatibility
                # TODO treat parametrized arguments.
                #      this will be done later, once it is validated for FunctionCall

                # the only thing we can do here, is to return a MethodCall,
                # without the class Variable, then we will treat it during the
                # Assign annotation
#                return MethodCall(method, args, cls_variable=None, kind=None)
                if args == ((),) or args ==[()]:
                    args = []
                return ConstructorCall(method, args, cls_variable=None)

            else:
                # if it is a user-defined function, we return a FunctionCall
                # TODO shall we keep it, or do this only in the Assign?

                # first we check if it is a macro, in this case, we will create
                # an appropriate FunctionCall
                macro = self.get_macro(name)
                if not macro is None:
                    func = macro.master

                    # ... create the appropriate arguments
                    args = macro.apply(args)
                    # ...
                else:
                    func = self.get_function(name)

                if not func is None:
                    if isinstance(func, (FunctionDef, Interface)):
                        # case of a function that takes no argument
                        if args == ((),) or args ==[()]:
                            args = []

                        if 'inline' in func.decorators:
                            return self._annotate(FunctionCall(func, args), **settings).inline

                        return self._annotate(FunctionCall(func, args), **settings)

                    else:
                        # return Function(name)(*args)
                        return func(*args)

                errors.report(UNDEFINED_FUNCTION, symbol=name,
                              bounding_box=self.bounding_box,
                              severity='error', blocker=self.blocking)

        elif isinstance(expr, FunctionCall):
            func = expr.func
            arg_dvar = [self._infere_type(i, **settings) for i in
                            expr.arguments]
            if isinstance(func, Interface):
                f_dvar = [[self._infere_type(j, **settings)
                           for j in i.arguments] for i in
                           func.functions]
                j = -1
                for i in f_dvar:
                    j += 1
                    found = True
                    for (idx, dt) in enumerate(arg_dvar):
                        # TODO imporve add the other verification shape,rank,pointer,...

                        dtype1 = dt['datatype'].__str__()
                        dtype2 = i[idx]['datatype'].__str__()
                        found  = found and (dtype1 in dtype2
                              or dtype2 in dtype1)
                        found = found and dt['rank'] \
                              == i[idx]['rank']
                       # found = found and dt['shape'] \
                       #       == i[idx]['shape']
                    if found:
                        break
                if found:
                    func = func.functions[j]
                else:
                    raise SystemExit('function not found in the interface')

            hide = expr.func.hide
            if hide:
                new_func = func
            else:
                new_func = func.rename(expr.func.name)
            expr = FunctionCall(new_func, expr.arguments, kind=expr.func.kind)


            return expr
        elif isinstance(expr, Summation):
            # treatment of the index/indices
            if isinstance(expr.args[1][0], Symbol):
                name = str(expr.args[1][0].name)
                var = self.get_variable(name)
                target = var
                if var is None:
                    target = Variable('int', name, rank=0)
                    self.insert_variable(target)
            itr = self._annotate(expr.args[1], **settings)
            body = self._annotate(expr.args[0], **settings)
            return SumFunction(body,itr)

        elif isinstance(expr, Expr):
            raise NotImplementedError('{expr} not yet available'.format(expr=type(expr)))

        elif isinstance(expr, (Assign, AugAssign)):

            # TODO unset position at the end of this part
            if expr.fst:
                self._bounding_box = expr.fst.absolute_bounding_box
            else:
                msg = 'Found a node without fst member ({})'.format(type(expr))
                raise PyccelSemanticError(msg)

            rhs = expr.rhs
            exprs = []
            from sympy import preorder_traversal
            ls = list(preorder_traversal(rhs))
            ls = ls[1:]
            for i in ls:
                if isinstance(i, (Application, Summation)):
                    exprs += [i]
            assigns = None
            if len(exprs)>0 and not isinstance(rhs, Lambda):
                #case of a function call in the rhs
                assigns = []
                exprs = exprs[::-1]
                for i in range(len(exprs)):
                    var = self.create_variable(exprs[i])
                    rhs = rhs.replace(exprs[i], var)
                    for j in range(i+1,len(exprs)):
                        exprs[j] = exprs[j].replace(exprs[i], var)
                    expr_new = Assign(var,exprs[i])
                    

          
                    # we set the fst to keep track of needed information for errors
                    expr_new.set_fst(expr.fst)
                    assigns.append(expr_new)

              
                assigns = [self._annotate(i, **settings) for i in assigns]
            # check if rhs is a call to a macro
            if isinstance(rhs, Application):
                name = str(type(rhs).__name__)
                macro = self.get_macro(name)
                if not macro is None:
                    # TODO check types from FunctionDef
                    func = macro.master

                    # ...
                    # all terms in lhs must be already declared and available in
                    # the namespace
                    lhs = expr.lhs
                    if not iterable(expr.lhs):
                        lhs = [expr.lhs]

                    results = []
                    for a in lhs:
                        _name = None
                        if isinstance(a, Symbol):
                            _name = a.name
                        else:
                            raise NotImplementedError('TODO')

                        var = self.get_variable(_name)
                        if var is None:
                            errors.report(UNDEFINED_VARIABLE, symbol=_name,
                                          bounding_box=self.bounding_box,
                                          severity='error', blocker=self.blocking)
                        results.append(var)
                    # ...

                    args = [self._annotate(i, **settings) for i in rhs.args]
                    args = macro.apply(args, results=results)

                    # TODO treate interface case
                    if isinstance(func, FunctionDef):
                        return self._annotate(FunctionCall(func, args), **settings)
                    else:
                        raise NotImplementedError('TODO')
            
            rhs = self._annotate(rhs, **settings)
                
            if isinstance(rhs, FunctionDef):
                #case of lambdify
                rhs = rhs.rename(str(expr.lhs))
                for i in rhs.body:
                    i.set_fst(expr.fst)
                rhs = self._annotate(rhs, **settings)
                return rhs
            
            # d_var can be a list of dictionaries
            elif isinstance(rhs, FunctionCall):
                # ARA: needed for functions defined only with a header
                results = rhs.func.results
                if results:
                    d_var = [self._infere_type(i, **settings) for i in results]

            elif isinstance(rhs, ConstructorCall):
                cls_name = rhs.func.cls_name  #  create a new Datatype for the current class
                cls = self.get_class(cls_name)

                dtype = self.get_class_construct(cls_name)()

                # to be moved to infere_type?

                d_var = {}
                d_var['datatype'] = dtype
                d_var['allocatable'] = False
                d_var['shape'] = None
                d_var['rank'] = 0
                d_var['is_target'] = True

                # set target  to True if we want the class objects to be pointers

                d_var['is_polymorphic'] = False
                d_var['cls_base'] = cls
                d_var['is_pointer'] = False

            elif isinstance(rhs, Function):

                name = str(type(rhs).__name__)
                if name in ['Zeros', 'Ones', 'Shape', 'Int']:

                    # TODO improve

                    d_var = {}
                    d_var['datatype'] = rhs.dtype
                    d_var['allocatable'] = not (name == 'Shape' or name
                            == 'Int')
                    d_var['shape'] = rhs.shape
                    d_var['rank'] = rhs.rank
                    d_var['is_pointer'] = False

                elif name in ['Array']:

                    dvar = self._infere_type(rhs.arg, **settings)
                    dtype = dvar['datatype']
                    d_var = {}
                    d_var['allocatable'] = True
                    d_var['shape'] = dvar['shape']
                    d_var['rank'] = dvar['rank']
                    d_var['is_pointer'] = False
                    if isinstance(dtype, NativeInteger):
                        d_var['datatype'] = 'ndarrayint'
                    elif isinstance(dtype, NativeFloat):
                        d_var['datatype'] = 'ndarrayfloat'
                    elif isinstance(dtype, NativeDouble):
                        d_var['datatype'] = 'ndarraydouble'
                    elif isinstance(dtype, NativeComplex):
                        d_var['datatype'] = 'ndarraycomplex'
                    elif isinstance(dtype, str):
                        d_var['datatype'] = 'ndarray' + dtype
                    else:
                        raise TypeError('list of type {0} not supported'.format(str(dtype)))

                elif name in ['Len', 'Sum', 'Rand']:
                    d_var = {}
                    d_var['datatype'] = rhs.dtype
                    if name in ['Sum']:
                        dvar = self._infere_type(rhs.arg, **settings)
                        d_var['datatype'] = dvar['datatype']
                    d_var['rank'] = 0
                    d_var['allocatable'] = False
                    d_var['is_pointer'] = False

                elif name in ['Mod']: # functions that return an int
                    d_var = {}
                    d_var['datatype'] = 'int'
                    d_var['rank'] = 0
                    d_var['allocatable'] = False
                    d_var['is_pointer'] = False

                elif name in [
                    'Abs',
                    'sqrt',
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
                    'Mod',
                    ]:
                    d_var = self._infere_type(rhs.args[0], **settings)
                    d_var['datatype'] = 'double' #TODO improve what datatype shoud we give here
                
                else:
                     raise NotImplementedError('TODO')

            elif isinstance(rhs , SumFunction):
                d_var = self._infere_type(rhs.body, **settings)

            else:
                d_var = self._infere_type(rhs, **settings)
                if d_var['datatype'
                         ].__class__.__name__.startswith('Pyccel'):
                    d_var['cls_base'] = self.get_class(d_var['datatype'
                            ].__class__.__name__[6:])
                    d_var['is_pointer'] = d_var['is_target'] \
                        or d_var['is_pointer']

                    # TODO if we want to use pointers then we set target to true
                    # in the ConsturcterCall

                    d_var['is_polymorphic'] = False
                if d_var['is_target']:
                    if isinstance(rhs, Symbol):
                        d_var['is_target'] = False
                        d_var['is_pointer'] = True
                    #case of rhs is a target variable the lhs must be a pointer


            lhs = expr.lhs
            if isinstance(lhs, Symbol):
                if isinstance(d_var, (list)):
                    if len(d_var)>1:
                        raise ValueError('can not assign multiple object into one variable')
                    elif len(d_var)==1:
                        d_var = d_var[0]
                name = lhs.name
                dtype = d_var.pop('datatype')
                lhs = Variable(dtype, name, **d_var)
                var = self.get_variable(name)
                if var is None:
                    self.insert_variable(lhs, name=lhs.name)
                else:
                    # TODO improve check type compatibility
                    if (str(lhs.dtype) != str(var.dtype)):
                        txt = '|{name}| {old} <-> {new}'.format(name=name,
                                                                old=var.dtype,
                                                                new=lhs.dtype)

                        # case where the rhs is of native type
                        # TODO add other native types
                        if isinstance(rhs, (Integer, Float)):
                            errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT,
                                          symbol=txt,
                                          bounding_box=self.bounding_box,
                                          severity='error', blocker=False)

                        else:
                            errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT,
                                          symbol=txt,
                                          bounding_box=self.bounding_box,
                                          severity='internal', blocker=False)

            elif isinstance(lhs, (IndexedVariable, IndexedBase)):
                # TODO check consistency of indices with shape/rank
                name = str(lhs.name)
                var = self.get_variable(name)
                if var is None:
                    # TODO ERROR not tested yet
                    errors.report(UNDEFINED_VARIABLE, symbol=name,
                                  bounding_box=self.bounding_box,
                                  severity='error', blocker=self.blocking)

                dtype = var.dtype
                lhs = IndexedVariable(name, dtype=dtype)

            elif isinstance(lhs, (IndexedElement, Indexed)):
                # TODO check consistency of indices with shape/rank
                name = str(lhs.base)
                var = self.get_variable(name)
                if var is None:
                    errors.report(UNDEFINED_INDEXED_VARIABLE, symbol=name,
                                  bounding_box=self.bounding_box,
                                  severity='error', blocker=self.blocking)

                args = tuple(lhs.indices)
                dtype = var.dtype
                lhs = IndexedVariable(name, dtype=dtype).__getitem__(*args)

            elif isinstance(lhs, DottedVariable):
                dtype = d_var.pop('datatype')
                name = lhs.args[0].name
                if self._current == '__init__':
                    cls_name = str(self.get_variable('self'
                                   ).cls_base.name)
                    cls = self.get_class(cls_name)
                    attributes = cls.attributes
                    parent = cls.parent
                    attributes = list(attributes)
                    n_name = str(lhs.args[1].name)
                    attributes += [Variable(dtype, n_name, **d_var)]

                    # update the attributes of the class and push it to the namespace
                    self.insert_class(ClassDef(cls_name, attributes,
                            [], parent=parent))

                    # update the self variable with the new attributes
                    dt = self.get_class_construct(cls_name)()
                    var = Variable(dt, 'self',
                                   cls_base=self.get_class(cls_name))
                    self.insert_variable(var, 'self')
                    lhs = DottedVariable(var, Variable(dtype, n_name,
                            **d_var))
                else:
                    lhs = self._annotate(lhs, **settings)

            # TODO ERROR must pass fst
            expr_new = Assign(lhs, rhs, strict=False)
            if not isinstance(lhs, (list, Tuple, tuple)):
                d_var = [d_var]

            for (i, dic) in enumerate(d_var):
                if not isinstance(lhs, (list, Tuple, tuple)):
                    lhs = [lhs]
                allocatable = False
                is_pointer = False
                if dic['allocatable']:
                    allocatable = True
                if dic['is_pointer']:
                    is_pointer = True
                if 'is_target' in dic.keys() and dic['is_target'] and isinstance(rhs, Variable):
                    is_pointer = True
                if isinstance(expr_new.rhs, IndexedElement) \
                    and expr_new.lhs.rank > 0:
                    allocatable = True
                elif isinstance(expr_new.rhs, Variable) \
                    and isinstance(expr_new.rhs.dtype, NativeList):
                    is_pointer = True
                if isinstance(lhs, Variable) and (allocatable
                        or is_pointer):
                    lhs[i] = self.update_variable(expr_new.lhs[i],
                            allocatable=allocatable,
                            is_pointer=is_pointer)
                if len(lhs) == 1:
                    lhs = lhs[0]
                if is_pointer:
                    expr_new = AliasAssign(lhs, rhs)
                elif expr_new.is_symbolic_alias:
                    expr_new = SymbolicAssign(lhs, rhs)
                    # in a symbolic assign, the rhs can be a lambda expression
                    # it is then treated as a def node
                    F = self.get_symbolic_function(lhs)
                    if F is None:
                        self.insert_symbolic_function(expr_new)
                    else:
                        raise NotImplementedError('TODO')

            if isinstance(expr, AugAssign):
                expr_new = AugAssign(expr_new.lhs, expr.op,
                        expr_new.rhs)
            if assigns and len(assigns)>0:
                #remove the Assignments that have rhs a function and not a subroutine
                assigns_ = assigns[:]
                for i in assigns_:
                    target = isinstance(i.rhs.func, FunctionDef) and not(i.rhs.func.is_procedure)
                    if  target or isinstance(i.rhs.func, Application):
                        expr_new = expr_new.subs(i.lhs,i.rhs)
                        assigns.remove(i)
                        self.remove(i.lhs.name)


            if assigns and len(assigns)>0:
                assigns += [expr_new]
                return Assigns(assigns)
            return expr_new

        elif isinstance(expr, For):
            # treatment of the index/indices
            if isinstance(expr.target, Symbol):
                name = str(expr.target.name)
                var = self.get_variable(name)
                target = var
                if var is None:
                    target = Variable('int', name, rank=0)
                    self.insert_variable(target)
            else:
                dtype = type(expr.target)
                # TODO ERROR not tested yet
                errors.report(INVALID_FOR_ITERABLE, symbol=expr.target,
                              bounding_box=self.bounding_box,
                              severity='error', blocker=self.blocking)

            itr = self._annotate(expr.iterable, **settings)
            body = self._annotate(expr.body, **settings)
            return For(target, itr, body)

        elif isinstance(expr, While):
            test = self._annotate(expr.test, **settings)
            body = self._annotate(expr.body, **settings)

            return While(test, body)

        elif isinstance(expr, If):
            args = self._annotate(expr.args, **settings)
            return If(*args)

        elif isinstance(expr, VariableHeader):
            # TODO improve
            #      move it to the ast like create_definition for FunctionHeader?
            name = expr.name
            d_var = expr.dtypes
            dtype = d_var.pop('datatype')

            var = Variable(dtype, name, **d_var)
            self.insert_variable(var)
            return expr

        elif isinstance(expr, FunctionHeader):

            # TODO should we return it and keep it in the AST?

            self.insert_header(expr)
            return expr
        elif isinstance(expr, ClassHeader):

            # TODO should we return it and keep it in the AST?

            self.insert_header(expr)
            return expr
        elif isinstance(expr, InterfaceHeader):
            container = self.namespace['functions']
            #TODO improve test all possible containers
            if set(expr.funcs).issubset(container.keys()):
                name = expr.name
                funcs = []
                for i in expr.funcs:
                    funcs += [container[i]]
            expr = Interface(name, funcs, hide = True)
            container[name] = expr
            return expr
        elif isinstance(expr, Return):

            results = expr.expr
            new_vars = []
            assigns = []
            

            if not isinstance(results, (list,Tuple,List)):
                results = [results]

            for result in results:
                if isinstance(result, Expr) and not isinstance(result, Symbol):
                    new_vars += [self.create_variable(result)]
                    assigns += [Assign(new_vars[-1], result)]
                    assigns[-1].set_fst(expr.fst)
            
            if len(assigns)==0:
                results = [self._annotate(result, **settings) for result in results]
                return Return(results)
            else:
                assigns = [self._annotate(assign, **settings) for assign in assigns]
                new_vars = [self._annotate(i, **settings) for i in new_vars]
                assigns = Assigns(assigns)
                return Return(new_vars,assigns)

        elif isinstance(expr, FunctionDef):
            name = str(expr.name)
            name = name.replace("'", '')  # remove quotes for str representation
            cls_name = expr.cls_name
            hide = False
            kind = 'function'
            decorators = expr.decorators
            funcs = []
            is_static = False 
            

            if cls_name:
                header = self.get_header(cls_name + """.""" + name)
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
                is_static = header.is_static

                # get function kind from the header
                kind = header.kind
            else:
                # this for the case of a function without arguments => no header
                interfaces = [FunctionDef(name, [], [], [])]
            for m in interfaces:
                args = []
                results = []
                local_vars = []
                global_vars = []
                imports = []
                arg = None
                self.set_current_fun(name)
                arguments = expr.arguments
                if cls_name and str(arguments[0].name) == 'self':
                    arg = arguments[0]
                    arguments = arguments[1:]
                    dt = self.get_class_construct(cls_name)()
                    var = Variable(dt, 'self',
                                   cls_base=self.get_class(cls_name))
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
                                    n_arg = Variable('int', n_name)
                                    # TODO clean namespace later
                                    var = self.get_variable(n_name)
                                    if not var is None:
                                        # TODO ERROR not tested yet
                                        errors.report(REDEFINING_VARIABLE,
                                                      symbol=n_name,
                                                      severity='error',
                                                      blocker=self.blocking)

                                    self.insert_variable(n_arg)
                                    additional_args += [n_arg]
                                # update shape
                                # TODO can this be improved? add some check
                                d_var['shape'] = Tuple(*additional_args)
                            a_new = Variable(dtype, str(a.name),**d_var)

                        if additional_args:
                            args += additional_args

                        args.append(a_new)

                        # TODO add scope and store already declared variables there,
                        #      then retrieve them

                        self.insert_variable(a_new,
                                name=str(a_new.name))

                # we annotate the body
                body = [self._annotate(i, **settings) for i in expr.body]
                # find return stmt and results
                returns = self._collect_returns_stmt(body)
                results = []


                for stmt in returns:
                    results += [set(stmt.expr)]

                if not all(i==results[0] for i in results):
                    raise PyccelSemanticError('multiple returns with different variables not available yet')

                if len(results)>0:
                    results = list(results[0])

                if arg and cls_name:
                    dt = self.get_class_construct(cls_name)()
                    var = Variable(dt, 'self',
                                   cls_base=self.get_class(cls_name))
                    args = [var] + args

                for var in self.get_variables():
                    if not var in args + results:
                        local_vars += [var]

                # TODO should we add all the variables or only the ones used in the function
                for var in self.get_variables('parent'):
                    if not var in args + results + local_vars:
                        global_vars += [var]
                func = FunctionDef(name, args, results, body,
                                   local_vars=local_vars,
                                   global_vars=global_vars,
                                   cls_name=cls_name,
                                   hide=hide,
                                   kind=kind,
                                   is_static=is_static,
                                   imports=imports,
                                   decorators=decorators,)

                if cls_name:
                    cls = self.get_class(cls_name)
                    methods = list(cls.methods) + [func]

                    # update the class  methods

                    self.insert_class(ClassDef(cls_name,
                            cls.attributes, methods, parent=cls.parent))

                self.set_current_fun(None)
                funcs += [func]

            if len(funcs) == 1:  # insert function def into namespace

                # TODO checking

                F = self.get_function(name)
                if F is None and not cls_name:
                    self.insert_function(funcs[0])

#                    # TODO uncomment and improve this part later.
#                    #      it will allow for handling parameters of different dtypes
#                    # for every parameterized argument, we need to create the
#                    # get_default associated function
#                    kw_args = [a for a in func.arguments if isinstance(a, ValuedVariable)]
#                    for a in kw_args:
#                        get_func = GetDefaultFunctionArg(a, func)
#                        # TODO shall we check first that it is not in the namespace?
#                        self.insert_variable(get_func, name=get_func.namer
                return funcs[0]
            else:
                funcs = [f.rename(str(f.name) + '_' + str(i)) for (i,
                         f) in enumerate(funcs)]

                # TODO checking

                funcs = Interface(name, funcs)
                F = self.get_function(name)
                if F is None and not cls_name:
                    self.insert_function(funcs)

#                    # TODO uncomment and improve this part later.
#                    #      it will allow for handling parameters of different dtypes
#                    # for every parameterized argument, we need to create the
#                    # get_default associated function
#                    kw_args = [a for a in func.arguments if isinstance(a, ValuedVariable)]
#                    for a in kw_args:
#                        get_func = GetDefaultFunctionArg(a, func)
#                        # TODO shall we check first that it is not in the namespace?
#                        self.insert_variable(get_func, name=get_func.namer

                return funcs
        elif isinstance(expr, (EmptyLine, NewLine)):
            return expr

        elif isinstance(expr, Print):
            args = self._annotate(expr.expr, **settings)
            if len(args) == 0:
                raise ValueError('no arguments given to print function')

            is_symbolic = lambda var: (isinstance(var, Variable) and
                                       isinstance(var.dtype, NativeSymbol))
            test = all(is_symbolic(i) for i in args)
            # TODO fix: not yet working because of mpi examples
#            if not test:
#                raise ValueError('all arguments must be either symbolic or none of them')

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
        elif isinstance(expr, Comment):

            return expr
        elif isinstance(expr, ClassDef):

            # TODO - improve the use and def of interfaces
            #      - wouldn't be better if it is done inside ClassDef?

            name = str(expr.name)
            name = name.replace("'", '')
            methods = list(expr.methods)
            parent = expr.parent
            interfaces = []

            # remove quotes for str representation

            self.insert_class(ClassDef(name, [], [], parent=parent))
            const = None
            for (i, method) in enumerate(methods):
                m_name = str(method.name).replace("'", '')

                # remove quotes for str representation

                if m_name == '__init__':
                    const = self._annotate(method)
                    methods.pop(i)

            methods = [self._annotate(i) for i in methods]

            if not const:
                errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                              bounding_box=self.bounding_box,
                              severity='error', blocker=True)

            methods = [const] + methods
            header = self.get_header(name)

            if not header:
                raise ValueError('Expecting a header class for {classe} but could not find it.'.format(classe=name))
            options = header.options
            attributes = self.get_class(name).attributes
            for i in methods:
                if isinstance(i, Interface):
                    methods.remove(i)
                    interfaces += [i]
            return ClassDef(name, attributes, methods,
                            interfaces=interfaces, parent=parent)

        elif isinstance(expr, Pass):
            return Pass()

        elif isinstance(expr, Del):
            ls = self._annotate(expr.variables)
            return Del(ls)

        elif isinstance(expr, Is):
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

        elif isinstance(expr, Import):

            # TODO - must have a dict where to store things that have been
            #        imported
            #      - should not use namespace

            if expr.source:
                if expr.source in pyccel_builtin_import_registery:
                    (name, atom) = pyccel_builtin_import(expr)
                    if not name is None:
                        F = self.get_variable(name)
                        if F is None:
                            # TODO remove: not up to date with Said devs on
                            # scoping
                            self._imports[name] = atom
                        else:
                            raise NotImplementedError('must report error')
                else:
                    # in some cases (blas, lapack, openmp and openacc level-0)
                    # the import should not appear in the final file
                    # all metavars here, will have a prefix and suffix = __
                    __ignore_at_import__ = False
                    __module_name__ = None
                    __import_all__ = False

                    # we need to use str here since source has been defined
                    # using repr.
                    # TODO shall we improve it?
                    p = self.d_parsers[str(expr.source)]

                    for entry in ['variables', 'classes', 'functions',
                                  'cls_constructs']:
                        d_self = self._namespace[entry]
                        d_son  = p.namespace[entry]
                        for k,v in list(d_son.items()):
                            # TODO test if it is not already in the namespace
                            if k in expr.target:
                                d_self[k] = v

                    # ... meta variables
                    if 'ignore_at_import' in list(p.metavars.keys()):
                        __ignore_at_import__ = p.metavars['ignore_at_import']

                    if 'import_all' in list(p.metavars.keys()):
                        __import_all__ = p.metavars['import_all']

                    if 'module_name' in list(p.metavars.keys()):
                        __module_name__ = p.metavars['module_name']
                        expr = Import(expr.target, __module_name__)
                    # ...

                    if not __ignore_at_import__:
                        return expr
                    else:
                        if __import_all__:
                            return Import(__module_name__)
                        else:
                            return EmptyLine()
            return expr

        elif isinstance(expr, Concatinate):
            left = self._annotate(expr.left)
            right = self._annotate(expr.right)
            return Concatinate(left, right)

        elif isinstance(expr, AnnotatedComment):
            return expr

        elif isinstance(expr, With):
            domaine = self._annotate(expr.test)
            parent = domaine.cls_base
            if not parent.is_with_construct:
                raise ValueError('with construct can only applied to '
                                    'classes with __enter__ and __exit__ methods')
            body = self._annotate(expr.body)
            return With(domaine, body, None).block

        elif isinstance(expr, MacroFunction):
            # we change here the master name to its FunctionDef
            f_name = expr.master
            header = self.get_header(f_name)
            if header is None:
                func = self.get_function(f_name)
                if func is None:
                    errors.report(MACRO_MISSING_HEADER_OR_FUNC,
                                  symbol=f_name,
                                  bounding_box=self.bounding_box,
                                  severity='error', blocker=self.blocking)
            else:
                interfaces = header.create_definition()
                # TODO -> Said: must handle interface
                func = interfaces[0]

            name = expr.name
            args = expr.arguments
            master_args = expr.master_arguments
            results = expr.results
            macro = MacroFunction(name, args, func, master_args, results=results)
            self.insert_macro(macro)

            return macro

        else:
            raise PyccelSemanticError('{expr} not yet available'.format(expr=type(expr)))


class PyccelParser(Parser):

    pass


######################################################

if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except:
        raise ValueError('Expecting an argument for filename')

    pyccel = Parser(filename)
    pyccel.parse(verbose=True)

    settings = {}
    pyccel.annotate(**settings)

#    for s in pyccel.ast:
#        print(type(s))

#    # ... using Pickle
#    # export the ast
#    pyccel.dump()
#
#    # load the ast
#    pyccel.load()
#    # ...

#    pyccel.view_namespace('variables')
#    pyccel.print_namespace()

#    pyccel.dot('ast.gv')
