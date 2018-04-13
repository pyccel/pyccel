#coding: utf-8
from redbaron import RedBaron
from redbaron import StringNode, IntNode, FloatNode, ComplexNode
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
from redbaron import ForNode, WhileNode
from redbaron import IfelseblockNode, IfNode, ElseNode, ElifNode
from redbaron import DotNode, AtomtrailersNode
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
from redbaron import GetitemNode,SliceNode
from redbaron import ImportNode, FromImportNode
from redbaron import DottedAsNameNode
from redbaron import NameAsNameNode


from pyccel.ast import NativeInteger, NativeFloat, NativeDouble, NativeComplex
from pyccel.ast import NativeBool
from pyccel.ast import NativeRange
from pyccel.ast import NativeIntegerList
from pyccel.ast import NativeFloatList
from pyccel.ast import NativeDoubleList
from pyccel.ast import NativeComplexList
from pyccel.ast import NativeList
from pyccel.ast import String
from pyccel.ast import datatype,DataTypeFactory
from pyccel.ast import Nil
from pyccel.ast import Variable
from pyccel.ast import DottedName,DottedVariable
from pyccel.ast import Assign, AliasAssign, SymbolicAssign
from pyccel.ast import Return
from pyccel.ast import Pass
from pyccel.ast import FunctionCall, MethodCall, ConstructorCall
from pyccel.ast import FunctionDef, Interface
from pyccel.ast import ClassDef
from pyccel.ast import GetDefaultFunctionArg
from pyccel.ast import For
from pyccel.ast import If
from pyccel.ast import While
from pyccel.ast import Print
from pyccel.ast import Del
from pyccel.ast import Assert
from pyccel.ast import Comment, EmptyLine
from pyccel.ast import Break
from pyccel.ast import Slice, IndexedVariable, IndexedElement
from pyccel.ast import FunctionHeader, ClassHeader, MethodHeader
from pyccel.ast import Concatinate
from pyccel.ast import ValuedVariable
from pyccel.ast import Argument, ValuedArgument
from pyccel.ast import Is
from pyccel.ast import Import, TupleImport
from pyccel.ast import AsName

from pyccel.parser.errors import Errors, PyccelSyntaxError, PyccelSemanticError
# TODO remove import * and only import what we need
from pyccel.parser.messages import *


########################
# ... TODO should be moved to pyccel.ast
from sympy.core.basic import Basic

from pyccel.ast import Range
from pyccel.ast import List
from pyccel.ast import builtin_function as pyccel_builtin_function
from pyccel.ast import builtin_import as pyccel_builtin_import

from pyccel.parser.syntax.headers import parse as hdr_parse
from pyccel.parser.syntax.openmp  import parse as omp_parse
from pyccel.parser.syntax.openacc import parse as acc_parse



from sympy import Symbol
from sympy import Tuple
from sympy import Add, Mul, Pow,floor,Mod
from sympy.core.expr import Expr
from sympy.logic.boolalg import And, Or
from sympy.logic.boolalg import true, false
from sympy.logic.boolalg import Not
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy import Integer, Float
from sympy.core.containers import Dict
from sympy.core.function import Function
from sympy.utilities.iterables import iterable
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy import FunctionClass

import os
# ...

# ... utilities
from sympy import srepr,sympify
from sympy.printing.dot import dotprint

import os

def view_tree(expr):
    """Views a sympy expression tree."""

    print(srepr(expr))
# ...


# ...
def _get_variable_name(var):
    """."""
    if isinstance(var, (Variable, IndexedVariable)):
        return str(var)
    elif isinstance(var, IndexedElement):
        return str(var.base)

    raise NotImplementedError('Uncovered type {dtype}'.format(dtype=type(var)))
# ...

# TODO use Double instead of Float? or add precision
def datatype_from_redbaron(node):
    """Returns the pyccel datatype of a RedBaron Node."""
    if isinstance(node, IntNode):
        return NativeInteger()
    elif isinstance(node, FloatNode):
        return NativeFloat()
    elif isinstance(node, ComplexNode):
        return NativeComplex()
    else:
        raise NotImplementedError('TODO')

def fst_to_ast(stmt):
    """Creates AST from FST."""
    # TODO - add settings to Errors
    #      - line and column
    #      - blocking errors
    errors = Errors()

    # ...
    def _treat_iterable(stmt):
        ls = [fst_to_ast(i) for i in stmt]
        if isinstance(stmt, (list, ListNode)):
            return List(*ls)
        else:
            return Tuple(*ls)
    # ...

    if isinstance(stmt, (RedBaron, LineProxyList, CommaProxyList,
                         NodeList, TupleNode, ListNode,
                         tuple, list)):
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
        new = fst_to_ast(stmt.target)
        # TODO improve
        if isinstance(old, str):
            old = old.replace("\'", "")
        if isinstance(new, str):
            new = new.replace("\'", "")
        return AsName(new, old)

    elif isinstance(stmt, DictNode):
        d = {}
        for i in stmt.value:
            if not isinstance(i, DictitemNode):
                raise PyccelSyntaxError('Expecting a DictitemNode')

            key   = fst_to_ast(i.key)
            value = fst_to_ast(i.value)

            # sympy does not allow keys to be strings
            if isinstance(key, str): errors.report(SYMPY_RESTRICTION_DICT_KEYS, severity='error')

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

    elif isinstance(stmt, ComplexNode):
        raise NotImplementedError('ComplexNode not yet available')

    elif isinstance(stmt, AssignmentNode):
        lhs = fst_to_ast(stmt.target)
        rhs = fst_to_ast(stmt.value)
        return Assign(lhs, rhs)

    elif isinstance(stmt, NameNode):
        if isinstance(stmt.previous,DotNode):
            return fst_to_ast(stmt.previous)
        if isinstance(stmt.next, GetitemNode):
            return fst_to_ast(stmt.next)
        if stmt.value == 'None':
            return Nil()
        elif stmt.value == 'True':
            return true
        elif stmt.value == 'False':
            return false
        else:
            return Symbol(str(stmt.value))

    elif isinstance(stmt, ImportNode):
        # in an import statement, we can have seperate target by commas
        ls = fst_to_ast(stmt.value)
        return Import(ls)

    elif isinstance(stmt, FromImportNode):
        source  = fst_to_ast(stmt.value)
        if isinstance(source, DottedVariable):
            source = DottedName(*source.names)

        targets = []
        for i in stmt.targets:
            s = fst_to_ast(i)
            targets.append(s)

        if len(targets) == 1:
            return Import(targets, source=source)
        else:
            return TupleImport(*targets)

    elif isinstance(stmt, DelNode):
        arg = fst_to_ast(stmt.value)
        return Del(arg)

    elif isinstance(stmt, UnitaryOperatorNode):
        target = fst_to_ast(stmt.target)
        if stmt.value == 'not':
            return Not(target)
        elif stmt.value == '+':
            return target
        elif stmt.value == '-':
            return -target
        elif stmt.value == '~':
            errors.report(PYCCEL_RESTRICTION_UNARY_OPERATOR, severity='error')
        else:
            raise PyccelSyntaxError('unknown/unavailable unary operator '
                                    '{node}'.format(node=type(stmt.value)))

    elif isinstance(stmt, (BinaryOperatorNode, BooleanOperatorNode)):
        first  = fst_to_ast(stmt.first)
        second = fst_to_ast(stmt.second)
        if stmt.value == '+':
            if isinstance(first,(String, List)) or isinstance(second,(String, List)):
                return Concatinate(first,second)
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
        elif stmt.value== '**':
            return Pow(first,second)
        elif stmt.value== '//':
            second = Pow(second, -1)
            return floor(Mul(first, second))
        elif stmt.value== '%':
            return Mod(first,second)
        else:
            raise PyccelSyntaxError('unknown/unavailable binary operator '
                                    '{node}'.format(node=type(stmt.value)))

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
        first  = fst_to_ast(stmt.first)
        second = fst_to_ast(stmt.second)
        op     = fst_to_ast(stmt.value)
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
            raise PyccelSyntaxError('unknown/unavailable binary operator '
                                    '{node}'.format(node=type(op)))

    elif isinstance(stmt, PrintNode):
        expr = fst_to_ast(stmt.value)
        return Print(expr)

    elif isinstance(stmt, AssociativeParenthesisNode):
        return fst_to_ast(stmt.value)

    elif isinstance(stmt, DefArgumentNode):
        name =  fst_to_ast(stmt.target)
        arg = Argument(str(name))
        if stmt.value is None:
            return arg
        else:
            value = fst_to_ast(stmt.value)
            return ValuedArgument(arg, value)

    elif isinstance(stmt, ReturnNode):
        return Return(fst_to_ast(stmt.value))

    elif isinstance(stmt, PassNode):
        return Pass()

    elif isinstance(stmt, DefNode):
        # TODO check all inputs and which ones should be treated in stage 1 or 2
        if isinstance(stmt.parent,ClassNode):
            cls_name    = stmt.parent.name
        else:
            cls_name    = None

        name        = fst_to_ast(stmt.name)
        arguments   = fst_to_ast(stmt.arguments)
        results     = []
        body        = fst_to_ast(stmt.value)
        local_vars  = []
        global_vars = []
        hide        = False
        kind        = 'function'
        imports     = []
        decorators  = [i.value.value[0].value for i in stmt.decorators] #TODO improve later
        return FunctionDef(name, arguments, results, body,
                           local_vars=local_vars, global_vars=global_vars,
                           cls_name=cls_name, hide=hide,
                           kind=kind, imports=imports ,decorators=decorators)

    elif isinstance(stmt, ClassNode):
        name = fst_to_ast(stmt.name)
        methods = [i for i in stmt.value if isinstance(i, DefNode)]
        methods = fst_to_ast(methods)
        attributes = methods[0].arguments
        parent = [i.value for i in stmt.inherit_from]
        return ClassDef(name=name, attributes=attributes, methods=methods, parent=parent)

    elif isinstance(stmt, AtomtrailersNode):
         return fst_to_ast(stmt.value)

    elif isinstance(stmt, GetitemNode):
         parent = stmt.parent
         args = fst_to_ast(stmt.value)
         if isinstance(stmt.previous.previous,DotNode):
             return fst_to_ast(stmt.previous.previous)
         #elif isinstance(stmt.previous,GetitemNode):
         #    return fst_to_ast(stmt.previous.previous)
         name = Symbol(str(stmt.previous.value))
         stmt.parent.remove(stmt.previous)
         stmt.parent.remove(stmt)
         if not hasattr(args, '__iter__'):
             args = [args]
         args = tuple(args)
         return IndexedBase(name)[args]

    elif isinstance(stmt, SliceNode):
         upper = fst_to_ast(stmt.upper)
         lower = fst_to_ast(stmt.lower)
         if upper and lower:
             return Slice(lower, upper)
         elif lower:
             return Slice(lower, None)
         elif upper:
             return Slice(None, upper)

    elif isinstance(stmt, DotProxyList):
        return fst_to_ast(stmt[-1])

    elif isinstance(stmt, DotNode):
        suf = stmt.next
        pre = fst_to_ast(stmt.previous)
        if stmt.previous:
            stmt.parent.value.remove(stmt.previous)
        suf = fst_to_ast(suf)
        return DottedVariable(pre, suf)

    elif isinstance(stmt, CallNode):
        args = fst_to_ast(stmt.value)
        f_name = str(stmt.previous)
        if len(args)==0:
            args=((),)
        func = Function(f_name)(*args)
        parent = stmt.parent
        if stmt.previous.previous and isinstance(stmt.previous.previous, DotNode):
            parent.value.remove(stmt.previous)
            parent.value.remove(stmt)
            pre = fst_to_ast(stmt.parent)
            return DottedVariable(pre, func)
        else:
            return func

    elif isinstance(stmt, CallArgumentNode):
        return fst_to_ast(stmt.value)

    elif isinstance(stmt, ForNode):
        target = fst_to_ast(stmt.iterator)
        iter   = fst_to_ast(stmt.target)
        body   = fst_to_ast(stmt.value)
        return For(target, iter, body, strict=False)

    elif isinstance(stmt, IfelseblockNode):
        args = fst_to_ast(stmt.value)
        return If(*args)

    elif isinstance(stmt,(IfNode, ElifNode)):
        test = fst_to_ast(stmt.test)
        body = fst_to_ast(stmt.value)
        return Tuple(test, body)

    elif isinstance(stmt, ElseNode):
        test = True
        body = fst_to_ast(stmt.value)
        return Tuple(test, body)

    elif isinstance(stmt, WhileNode):
        test = fst_to_ast(stmt.test)
        body = fst_to_ast(stmt.value)
        return While(test, body)

    elif isinstance(stmt, AssertNode):
        expr = fst_to_ast(stmt.value)
        return Assert(expr)

    elif isinstance(stmt, EndlNode):
        return EmptyLine()

    elif isinstance(stmt, CommentNode):
        # if annotated comment
        if stmt.value.startswith('#$'):
            env = stmt.value[2:].lstrip()
            if env.startswith('omp'):
                return omp_parse(stmts=stmt.value)
            elif env.startswith('acc'):
                return acc_parse(stmts=stmt.value)
            elif env.startswith('header'):
                return hdr_parse(stmts=stmt.value)
            else:
                errors.report(PYCCEL_INVALID_HEADER, severity='error')
        else:
            # TODO improve
            txt = stmt.value[1:].lstrip()
            return Comment(txt)

    elif isinstance(stmt, BreakNode):
        return Break()

    elif isinstance(stmt, (ExceptNode, FinallyNode, TryNode)):
        # this is a blocking error, since we don't want to convert the try body
        errors.report(PYCCEL_RESTRICTION_TRY_EXCEPT_FINALLY, severity='critical', blocker=True)

    elif isinstance(stmt, RaiseNode):
        errors.report(PYCCEL_RESTRICTION_RAISE, severity='error')

    elif isinstance(stmt, (YieldNode, YieldAtomNode)):
        errors.report(PYCCEL_RESTRICTION_YIELD, severity='error')

    else:
        raise PyccelSyntaxError('{node} not yet available'.format(node=type(stmt)))


def _read_file(filename):
    """Returns the source code from a filename."""
    f = open(filename)
    code = f.read()
    f.close()
    return code



class Parser(object):
    """ Class for a Parser."""
    def __init__(self, inputs, debug=False, headers=None):
        """Parser constructor.

        inputs: str
            filename or code to parse as a string

        debug: bool
            True if in debug mode.

        headers: list, tuple
            list of headers to append to the namespace
        """
        self._fst = None
        self._ast = None
        self._filename = None
        self._namespace = {}
        self._namespace['variables'] = {}
        self._namespace['classes'] = {}
        self._namespace['functions'] = {}
        self._namespace['cls_constructs'] = {}
        self._scope = {} #represent the namespace of a function
        self._parent = None
        self._current = None # we use it to detect the current method or function
        self._imports = {} #we use it to store the imports
        # TODO use another name for headers
        #      => reserved keyword, or use __
        self._namespace['headers'] = {}
        if headers:
            if not isinstance(headers, dict):
                raise TypeError('Expecting a dict of headers')

            for key,value in headers.items():
                self._namespace['headers'][key] = value

        # check if inputs is a file
        code = inputs
        if os.path.isfile(inputs):
            code = _read_file(inputs)
            self._filename = inputs

        self._code = code

        red = RedBaron(code)
        self._fst = red

    @property
    def namespace(self):
        return self._namespace

    @property
    def headers(self):
        return self.namespace['headers']

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

    def parse(self):
        """converts redbaron fst to sympy ast."""
        # TODO - add settings to Errors
        #      - filename
        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('syntax')

        ast = fst_to_ast(self.fst)
        self._ast = ast

        errors.check()

        return ast

    def annotate(self, **settings):
        """."""
        # TODO - add settings to Errors
        #      - filename
        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('semantic')

        ast = self.ast
        ast = self._annotate(ast, **settings)
        self._ast = ast

        errors.check()

        return ast

    def print_namespace(self):
        # TODO improve spacing
        print('------- Namespace -------')
        for k,v in self.namespace.items():
            print('{var} \t :: \t {dtype}'.format(var=k, dtype=type(v)))
        print('-------------------------')

    def dot(self, filename):
        """Exports sympy AST using graphviz then convert it to an image."""
        expr = self.ast
        graph_str = dotprint(expr)

        f = file(filename, 'w')
        f.write(graph_str)
        f.close()

        # name without path
        name = os.path.basename(filename)
        # name without extension
        name = os.path.splitext(name)[0]
        cmd = "dot -Tps {name}.gv -o {name}.ps".format(name=name)
        os.system(cmd)

    def get_variable(self, name):
        """."""
        var = None
        if self._parent:
            for i in self._parent.attributes:
                if str(i.name) == name:
                    var = i
                    return var
        if self._current:
            if name in self._scope[self._current]['variables']:
                var = self._scope[self._current]['variables'][name]
        if isinstance(self._current, DottedName):
            if name in self._scope[self._current.name[0]]['variables']:
                var = self._scope[self._current.name[0]]['variables'][name]
        if name in self._namespace['variables']:
            var =  self._namespace['variables'][name]
        return var

    def get_variables(self,source = None):
        if source=='parent':
            if isinstance(self._current, DottedName):
                name = self._current.name[0]
            elif not (self._current is None):
                return self._namespace['variables'].values()
            else:
                raise TypeError('there is no parent to extract variables from ')
            return self._scope[name]['variables'].values()
        else:
            return self._scope[self._current]['variables'].values()


    def insert_variable(self, expr, name=None):
        """."""
        # TODO add some checks before
        if name is None:
            name = str(expr)
        if not isinstance(expr,Variable):
            raise TypeError('variable must be of type Variable')

        if self._current:
            self._scope[self._current]['variables'][name] = expr
        else:
            self._namespace['variables'][name] = expr

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
        func = None
        if name in self._namespace['functions']:
            func = self._namespace['functions'][name]
        elif name in self._imports:
            func = self._imports[name]
        return func

    def insert_function(self, func):
        """."""
        if not isinstance(func, (FunctionDef,Interface)):
            raise TypeError('Expected a Function definition')
        else:
            if isinstance(self._current, DottedName):
                self._scope[self._current.name[0]]['functions'][str(func.name)] = func
            else:
                self._namespace['functions'][str(func.name)] = func

    def remove(self, name):
        """."""
        if self_current:
            self._scope[self._current]['variables'].pop(name, None)
        self._namespace['variables'].pop(name,None)

    def update_variable(self, var, **options):
        """."""
        name = _get_variable_name(var).split('.')
        var = self.get_variable(name[0])
        if len(name)>1:
            name_ = _get_variable_name(var)
            for i in var.cls_base.attributes:
                if str(i.name)==name[1]:
                    var=i
            name = name_
        else:
            name = name[0]
        if var is None:
            raise ValueError('Undefined variable {name}'.format(name=name))

        # TODO implement a method inside Variable
        d_var = self._infere_type(var)
        for key, value in options.items():
            d_var[key] = value
        dtype = d_var.pop('datatype')
        var = Variable(dtype, name, **d_var)
        self.insert_variable(var,name) #TODO improve to insert in the right namespace
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
        else:
            self._scope.pop(self._current)
            if isinstance(self._current, DottedName):
                #case of a functiondef in a another function
                name = self._current.name[0]
        self._current = name


    def insert_header(self, expr):
        """."""
        if isinstance(expr,MethodHeader):
            self._namespace['headers'][str(expr.name)] = expr
        if isinstance(expr, FunctionHeader):
            self._namespace['headers'][str(expr.func)] = expr
        elif isinstance(expr, ClassHeader):
            self._namespace['headers'][str(expr.name)] = expr
            # create a new Datatype for the current class
            iterable       = ('iterable' in expr.options)
            with_construct = ('with' in expr.options)
            dtype = DataTypeFactory(str(expr.name), ("_name"), is_iterable=iterable,\
                                    is_with_construct=with_construct)
            self.set_class_construct(str(expr.name), dtype)
        else:
            raise TypeError('header of type{0} is not supported'.format(str(type(expr))))

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
        d_var['datatype'] = None
        d_var['allocatable'] = None
        d_var['shape'] = None
        d_var['rank'] = None
        d_var['is_pointer'] = None
        d_var['is_target'] = None
        d_var['is_polymorphic'] = None
        d_var['is_optional'] = None

        # TODO improve => put settings as attribut of Parser
        DEFAULT_FLOAT = settings.pop('default_float', 'double')

        if isinstance(expr, Integer):
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
                for s,i in zip(var.shape, expr.indices):
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

            d_var['datatype']    = var.dtype
            d_var['allocatable'] = var.allocatable
            d_var['shape']       = var.shape
            d_var['rank']        = var.rank
            return d_var

        elif isinstance(expr, Range):
            d_var['datatype']    = NativeRange()
            d_var['allocatable'] = False
            d_var['shape']       = None
            d_var['rank']        = 0
            d_var['cls_base']    = expr # TODO: shall we keep it?
            return d_var

        elif isinstance(expr, Is):
            d_var['datatype'] = NativeBool()
            d_var['allocatable'] = False
            d_var['is_pointer'] = False
            d_var['rank'] = 0
            return d_var

        elif isinstance(expr, DottedVariable):
            if isinstance(expr.args[0],DottedVariable):
                self._parent = expr.args[0].args[1].cls_base
            else:
                self._parent = expr.args[0].cls_base
            d_var = self._infere_type(expr.args[1])
            self._parent = None
            return d_var

        elif isinstance(expr, Expr):
            ds = [self._infere_type(i, **settings) for i in expr.args]
            dtypes = [d['datatype'] for d in ds]
            allocatables = [d['allocatable'] for d in ds]
            pointers = [d['is_pointer'] or d['is_target'] for  d in ds]
            ranks = [d['rank'] for d in ds]
            shapes = [d['shape'] for d in ds]

            # TODO improve

            # ... only scalars and variables of rank 0 can be handled
            r_min = min(ranks)
            r_max = max(ranks)
            if not(r_min == r_max):
                if not(r_min == 0):
                    raise ValueError('cannot process arrays of different ranks.')
            rank = r_max
            # ...

            # ...
            shape = None
            for s in shapes:
                if s:
                    shape = s
            # ...
            d_var['datatype'] = 'int'
            #TODO imporve to hadle all possible types (complex ,ndarray , ...)
            for i in dtypes:
                if isinstance(i, str):
                    if i == 'float' or i == 'double':
                        d_var['datatype'] = i
                        break
                elif isinstance(i, (NativeDouble, NativeFloat)):
                    d_var['datatype'] = i
                    break

            d_var['datatype'] = dtypes[0]
            d_var['allocatable'] = any(allocatables)
            d_var['is_pointer']  = any(pointers)
            d_var['shape'] = shape
            d_var['rank'] = rank
            return d_var

        elif isinstance(expr, (tuple, list, List, Tuple)):
            import numpy
            d = self._infere_type(expr[0], **settings)
            # TODO must check that it is consistent with pyccel's rules
            d_var['datatype']    = d['datatype']
            d_var['rank']        = d['rank'] + 1
            d_var['shape']       = numpy.shape(expr) # TODO improve
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
            if not(d_var_left['datatype']=='str' or d_var_right['datatype']=='str'):
                d_var_left['shape'] = tuple(map(operator.add,d_var_right['shape'],d_var_left['shape']))
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

        elif isinstance(expr, (BooleanTrue, BooleanFalse)):
            return expr

        elif isinstance(expr, Variable):
            name = expr.name
            var = self.get_variable(name)
            if var is None:
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                              severity='error', blocker=True)
            return var

        elif isinstance(expr, str):
            return repr(expr)

        elif isinstance(expr, (IndexedVariable, IndexedBase)):
            # an indexed variable is only defined if the associated variable is in
            # the namespace
            name = str(expr.name)
            var = self.get_variable(name)
            if var is None:
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                              severity='error', blocker=True)
            dtype = var.dtype
            # TODO add shape
            return IndexedVariable(name, dtype=dtype)

        elif isinstance(expr, (IndexedElement, Indexed)):
            name = str(expr.base)
            var = self.get_variable(name)
            if var is None:
                errors.report(UNDEFINED_INDEXED_VARIABLE, symbol=name,
                              severity='error', blocker=True)
            # TODO check consistency of indices with shape/rank
            args = tuple(expr.indices)
            dtype = var.dtype
            return IndexedVariable(name, dtype=dtype).__getitem__(*args)

        elif isinstance(expr, Symbol):
            name = str(expr.name)
            var = self.get_variable(name)
            if var is None:
                raise PyccelSemanticError('Symbolic {name} variable '
                                          'is not allowed'.format(name=name))
            return var

        elif isinstance(expr, DottedVariable):
            first = self._annotate(expr.args[0])
            attr_name = [i.name for i in first.cls_base.attributes]
            if isinstance(expr.args[1], Symbol) and not expr.args[1].name in attr_name:
                for i in first.cls_base.methods:
                    if str(i.name)==expr.args[1].name and 'property' in i.decorators:
                        second = FunctionCall(i,(),kind = i.kind)
                        return DottedVariable(first, second)
            if not isinstance(expr.args[1],Function):
                self._parent =first.cls_base
                second = self._annotate(expr.args[1])
                self._parent = None
            else:
                for i in first.cls_base.methods:
                    if str(i.name) == str(type(expr.args[1]).__name__):
                        args = [self._annotate(arg) for arg in expr.args[1].args]
                        if len(args)==1 and args[0]==():
                            args=[]
                        second = FunctionCall(i,args,kind =i.kind)
            return DottedVariable(first, second)

        elif isinstance(expr, (Add, Mul, And, Or, Eq, Ne, Lt, Gt, Le, Ge)):
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

        elif isinstance(expr, Function):
            args = [self._annotate(i, **settings) for i in expr.args]
            name = str(type(expr).__name__)
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
                                  severity='warning', blocker=True)
                args = expr.args
                m_args = method.arguments[1:] #we delete the self arg
                # TODO check compatibility
                # TODO treat parametrized arguments.
                #      this will be done later, once it is validated for FunctionCall

                # the only thing we can do here, is to return a MethodCall,
                # without the class Variable, then we will treat it during the
                # Assign annotation
#                return MethodCall(method, args, cls_variable=None, kind=None)
                return ConstructorCall(method, args, cls_variable=None)
            else:
                # if it is a user-defined function, we return a FunctionCall
                # TODO shall we keep it, or do this only in the Assign?
                func = self.get_function(name)
                if not(func is None):
                    if isinstance(func, (FunctionDef, Interface)):
                        if args==[()]:
                            args = []
                            #case of a function that takes no argument
                        if 'inline' in func.decorators:
                            return FunctionCall(func,args).inline
                        return FunctionCall(func, args)
                    else:
                        return func(*args)
                        #return Function(name)(*args)
                errors.report(UNDEFINED_FUNCTION, symbol=name,
                              severity='error', blocker=True)

        elif isinstance(expr, Expr):
            raise NotImplementedError('{expr} not yet available'.format(expr=type(expr)))

        elif isinstance(expr, Assign):
            rhs = self._annotate(expr.rhs, **settings)
            # d_var can be a list of dictionaries
            if isinstance(rhs, FunctionCall):
                func = rhs.func

                # treating results
                arg_dvar = [self._infere_type(i, **settings) for i in rhs.arguments]

                if isinstance(func, Interface):
                    f_dvar = [[self._infere_type(j, **settings) for j in i.arguments] for i in func.functions]
                    j = -1
                    for i in f_dvar:
                        j += 1
                        found = True
                        for idx,dt in enumerate(arg_dvar):
                            #TODO imporve add the other verification shape,rank,pointer,...
                            dtype1 = dt['datatype'].__str__()
                            dtype2 = i[idx]['datatype'].__str__()
                            found = found and (dtype1 in dtype2 or dtype2 in dtype1)
                            found = found and (dt['rank']==i[idx]['rank'])
                            found = found and (dt['shape']==i[idx]['shape'])
                        if found:
                            break
                    if found:
                        func = func.functions[j]
                    else:
                        raise SystemExit('function not found in the interface')

                results = func.results
                d_var = [self._infere_type(i, **settings) for i in results]
                # if there is only one result, we don't consider d_var as a list
                if len(d_var) == 1:
                    d_var = d_var[0]

                rhs = FunctionCall(func.rename(rhs.func.name), rhs.arguments, kind=rhs.func.kind)

            elif isinstance(rhs, ConstructorCall):
                cls_name = rhs.func.cls_name # create a new Datatype for the current class
                cls = self.get_class(cls_name)

                dtype = self.get_class_construct(cls_name)()
                # to be moved to infere_type?
                d_var = {}
                d_var['datatype']    = dtype
                d_var['allocatable'] = False
                d_var['shape']       = None
                d_var['rank']        = 0
                d_var['is_target']   = True
                #set target  to True if we want the class objects to be pointers
                d_var['is_polymorphic'] = False
                d_var['cls_base']    = cls
                d_var['is_pointer'] = False

            elif isinstance(rhs, Function):
                name = str(type(rhs).__name__)
                if name in ['Zeros', 'Ones', 'Shape','Int']:
                    # TODO improve
                    d_var = {}
                    d_var['datatype']    = rhs.dtype
                    d_var['allocatable'] = not(name=='Shape' or name =='Int')
                    d_var['shape']       = rhs.shape
                    d_var['rank']        = rhs.rank
                    d_var['is_pointer'] = False

                elif name in ['Array']:
                    dvar = self._infere_type(rhs.arg, **settings)
                    dtype =dvar['datatype']
                    d_var = {}
                    d_var['allocatable'] = True
                    d_var['shape']       = dvar['shape']
                    d_var['rank']        = dvar['rank']
                    d_var['is_pointer'] = False
                    if isinstance(dtype, NativeInteger):
                        d_var['datatype'] = 'ndarrayint'
                    elif isinstance(dtype, NativeFloat ):
                        d_var['datatype'] = 'ndarrayfloat'
                    elif isinstance(dtype, NativeDouble):
                        d_var['datatype'] = 'ndarraydouble'
                    elif isinstance(dtype, NativeComplex):
                        d_var['datatype'] = 'ndarraycomplex'
                    else:
                        raise TypeError('list of type {0} not supported'.format(str(dtype)))
                elif name in ['Len','Sum']:
                     d_var = {}
                     d_var['datatype']='int'
                     if name == 'Sum':
                         dvar = self._infere_type(rhs.arg, **settings)
                         d_var['datatype'] = dvar['datatype']
                     d_var['rank'] = 0
                     d_var['allocatable'] = False
                     d_var['is_pointer'] = False
                else:
                    raise NotImplementedError('TODO')

            elif isinstance(rhs, MethodCall):
                raise NotImplementedError('TODO')

            else:
                d_var = self._infere_type(rhs, **settings)
                if d_var['datatype'].__class__.__name__.startswith('Pyccel'):
                    d_var['cls_base'] = self.get_class(d_var['datatype'].__class__.__name__[6:])
                    d_var['is_pointer'] = d_var['is_target'] or d_var['is_pointer']
                    #TODO if we want to use pointers then we set target to true
                    #in the ConsturcterCall

                    d_var['is_polymorphic'] = False

            lhs = expr.lhs
            if isinstance(lhs, Symbol):
                name = lhs.name
                dtype = d_var.pop('datatype')
                lhs = Variable(dtype, name, **d_var)
                var = self.get_variable(name)
                if var is None:
                    self.insert_variable(lhs, name=lhs.name)

            elif isinstance(lhs, (IndexedVariable, IndexedBase)):
                # TODO check consistency of indices with shape/rank
                name = str(lhs.name)
                var = self.get_variable(name)
                if var is None:
                    errors.report(UNDEFINED_VARIABLE, symbol=name,
                                  severity='error', blocker=True)

                dtype = var.dtype
                lhs = IndexedVariable(name, dtype=dtype)

            elif isinstance(lhs, (IndexedElement, Indexed)):
                # TODO check consistency of indices with shape/rank
                name = str(lhs.base)
                var = self.get_variable(name)
                if var is None:
                    errors.report(UNDEFINED_INDEXED_VARIABLE, symbol=name,
                                  severity='error', blocker=True)
                args = tuple(lhs.indices)
                dtype = var.dtype
                lhs = IndexedVariable(name, dtype=dtype).__getitem__(*args)

            elif isinstance(lhs, DottedVariable):
                dtype = d_var.pop('datatype')
                name = lhs.args[0].name
                if self._current == '__init__':
                     cls_name = str(self.get_variable('self').cls_base.name)
                     cls = self.get_class(cls_name)
                     attributes = cls.attributes
                     parent = cls.parent
                     attributes = list(attributes)
                     n_name = str(lhs.args[1].name)
                     attributes += [Variable(dtype, n_name, **d_var)]
                     #update the attributes of the class and push it to the namespace
                     self.insert_class(ClassDef(cls_name,attributes,[],parent = parent))
                     #update the self variable with the new attributes
                     dt=self.get_class_construct(cls_name)()
                     var = Variable(dt,'self',cls_base = self.get_class(cls_name))
                     self.insert_variable(var,'self')
                     lhs = DottedVariable(var, Variable(dtype, n_name, **d_var))
                else :
                    lhs =  self._annotate(lhs, **settings)

            expr = Assign(lhs, rhs, strict=False)
            # we check here, if it is an alias assignment
#            lhs.inspect()
#            if isinstance(expr.rhs, Variable):
#                expr.rhs.inspect()
            #if expr.is_alias:
                # here we need to know if lhs is allocatable or a pointer
                # TODO improve
            allocatable = False
            is_pointer = False
            if d_var['allocatable']:
                allocatable = True
            if d_var['is_pointer']:
                is_pointer = True
            if isinstance(expr.rhs, IndexedElement) and (expr.lhs.rank > 0):
                allocatable = True
            elif (isinstance(expr.rhs, Variable) and isinstance(expr.rhs.dtype, NativeList)):
                is_pointer = True
            if isinstance(lhs, Variable) and (allocatable or is_pointer):
                lhs = self.update_variable(expr.lhs,allocatable=allocatable,is_pointer=is_pointer)
            if is_pointer:
                return AliasAssign(lhs, rhs)
            elif expr.is_symbolic_alias:
                return SymbolicAssign(lhs, rhs)
            else:
                return expr

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
                errors.report(INVALID_FOR_ITERABLE, symbol=expr.target,
                              severity='error')

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

        elif isinstance(expr, FunctionHeader):
            # TODO should we return it and keep it in the AST?
            self.insert_header(expr)
            return expr

        elif isinstance(expr,ClassHeader):
            # TODO should we return it and keep it in the AST?
            self.insert_header(expr)
            return expr

        elif isinstance(expr, Return):
            results = expr.expr
            if isinstance(results, Symbol):
                name = results.name
                var = self.get_variable(name)
                if var is None:
                    errors.report(UNDEFINED_VARIABLE, symbol=name,
                                  severity='error', blocker=True)
                return Return([var])
            elif isinstance(results, (list, tuple, Tuple)):
                ls = []
                for i in results:
                    if not isinstance(i, Symbol):
                        raise NotImplementedError('only symbol or iterable are allowed for returns')
                    name = i.name
                    var = self.get_variable(name)
                    if var is None:
                        errors.report(RETURN_VALUE_EXPECTED, symbol=name,
                                      severity='error', blocker=True)
                    ls += [var]
                return Return(ls)
            else:
                raise NotImplementedError('only symbol or iterable are allowed for returns')

        elif isinstance(expr, FunctionDef):
            name = str(expr.name)
            name = name.replace('\'', '') # remove quotes for str representation
            cls_name    = expr.cls_name
            hide        = False
            kind        = 'function'
            decorators  = expr.decorators
            funcs = []

            is_static = False
            if cls_name:
                header = self.get_header(cls_name+'.'+name)
            else:
                header = self.get_header(name)
            if expr.arguments and not header:
                    errors.report(FUNCTION_TYPE_EXPECTED, symbol=name,
                                  severity='error', blocker=True)

                # we construct a FunctionDef from its header
            if header:
                interfaces = header.create_definition()
                # is_static will be used for f2py
                is_static = header.is_static

                # get function kind from the header
                kind = header.kind
            else:
                # TODO why are we doing this?
                interfaces = [FunctionDef(name,[],[],[])]

            for m in interfaces:
                args = []
                results = []
                local_vars  = []
                global_vars = []
                imports     = []
                arg = None
                self.set_current_fun(name)
                arguments = expr.arguments
                if cls_name and str(arguments[0].name) == 'self':
                    arg = arguments[0]
                    arguments = arguments[1:]
                    dt = self.get_class_construct(cls_name)()
                    var = Variable(dt, 'self', cls_base = self.get_class(cls_name))
                    self.insert_variable(var, 'self')

                if arguments:
                    for a, ah in zip(arguments, m.arguments):
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
                            if is_static and (rank > 0):
                                for i in range(0, rank):
                                    n_name = 'n{i}_{name}'.format(name=str(a.name), i=i)
                                    n_arg = Variable('int', n_name)
                                    # TODO clean namespace later
                                    var = self.get_variable(n_name)
                                    if not(var is None):
                                        # TODO report appropriate message
                                        errors.report('variable already defined',
                                                      symbol=n_name,
                                                      severity='error', blocker=True)

                                    self.insert_variable(n_arg)

                                    additional_args += [n_arg]

                                # update shape
                                # TODO can this be improved? add some check
                                d_var['shape'] = Tuple(*additional_args)

                            a_new = Variable(dtype, str(a.name), **d_var)

                        if additional_args:
                            args += additional_args

                        args.append(a_new)

                        # TODO add scope and store already declared variables there,
                        #      then retrieve them
                        self.insert_variable(a_new, name=str(a_new.name))

                # we annotate the body
                body = self._annotate(expr.body, **settings)
                # find return stmt and results
                # we keep the return stmt, in case of handling multi returns later
                for stmt in body:
                    # TODO case of multiple calls to return
                    if isinstance(stmt, Return):
                        results = stmt.expr
                        if isinstance(results, Variable):
                            results = [results]
                            kind = 'function'

                if arg and cls_name:
                    dt = self.get_class_construct(cls_name)()
                    var = Variable(dt, 'self', cls_base = self.get_class(cls_name))
                    args = [var] + args
                for var in self.get_variables():
                    if not var in args+results:
                        local_vars += [var]

                for var in self.get_variables('parent'):
                    if not var in args+results+local_vars:
                        global_vars += [var]
                        #TODO should we add all the variables or only the ones used in the function
                func = FunctionDef(name, args, results, body,
                                   local_vars=local_vars,
                                   global_vars=global_vars,
                                   cls_name=cls_name,
                                   hide=hide,
                                   kind=kind,
                                   is_static=is_static,
                                   imports=imports,
                                   decorators=decorators)
                if cls_name:
                    cls = self.get_class(cls_name)
                    methods = list(cls.methods) + [func]
                    #update the class  methods
                    self.insert_class(ClassDef(cls_name,cls.attributes,methods,parent=cls.parent))

                self.set_current_fun(None)
                funcs += [func]

            if len(funcs)==1:# insert function def into namespace
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
                funcs = [f.rename(str(f.name)+'_'+str(i)) for i,f in enumerate(funcs)]
                # TODO checking
                funcs = Interface(name,funcs)
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

        elif isinstance(expr, EmptyLine):
            return expr

        elif isinstance(expr, Print):
            args = self._annotate(expr.expr, **settings)
            return Print(args)

        elif isinstance(expr, Comment):
            return expr

        elif isinstance(expr, ClassDef):
            # TODO - improve the use and def of interfaces
            #      - wouldn't be better if it is done inside ClassDef?
            name = str(expr.name)
            name = name.replace('\'', '')
            methods = list(expr.methods)
            parent  = expr.parent
            interfaces = []
            # remove quotes for str representation
            self.insert_class(ClassDef(name,[],[],parent=parent))
            const = None
            for i,method in enumerate(methods):
                m_name = str(method.name).replace('\'', '')
                # remove quotes for str representation
                if m_name == '__init__':
                    const = self._annotate(method)
                    methods.pop(i)

            methods = [self._annotate(i) for i in methods]
            #remove the self object

            if not const:
                raise SystemExit('missing contuctor in the class {0}'.format(name))

            methods = [const]+methods
            header = self.get_header(name)

            if not header:
                raise ValueError('Expecting a header class for {classe} '
                                     'but could not find it.'.format(classe=name))
            options = header.options
            attributes = self.get_class(name).attributes
            for i in methods:
                if isinstance(i,Interface):
                    methods.remove(i)
                    interfaces += [i]
            return ClassDef(name, attributes, methods,interfaces=interfaces,parent=parent)

        elif isinstance(expr, Pass):
            return Pass()

        elif isinstance(expr, Del):
            ls =  self._annotate(expr.variables)
            return Del(ls)

        elif isinstance(expr, Is):
            if not isinstance(expr.rhs, Nil):
                errors.report(PYCCEL_RESTRICTION_IS_RHS, severity='error', blocker=True)

            name = expr.lhs
            var = self.get_variable(str(name))
            if var is None:
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                              severity='error', blocker=True)

            return Is(var, expr.rhs)

        elif isinstance(expr, Import):
            # TODO - must have a dict where to store things that have been
            #        imported
            #      - should not use namespace
            name, atom = pyccel_builtin_import(expr)
            if not(name is None):
                F = self.get_variable(name)
                if F is None:
                    self._imports[name] = atom
                else:
                    raise NotImplementedError('must report error')

            return expr
        elif isinstance(expr, Concatinate):
            left = self._annotate(expr.left)
            right = self._annotate(expr.right)
            return Concatinate(left, right)
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

    pyccel.parse()


    settings = {}
    pyccel.annotate(**settings)
    pyccel.print_namespace()

    pyccel.dot('ast.gv')
