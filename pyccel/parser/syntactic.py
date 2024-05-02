# -*- coding: utf-8 -*-

from collections import OrderedDict
import redbaron
import traceback
import importlib
import pickle
import os
import sys
import re

#==============================================================================

from redbaron import RedBaron
from redbaron import StringNode, IntNode, FloatNode, ComplexNode
from redbaron import FloatExponantNode, StarNode
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
from redbaron import ListComprehensionNode
from redbaron import ComprehensionLoopNode
from redbaron import ArgumentGeneratorComprehensionNode
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
from redbaron import TernaryOperatorNode
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
from redbaron import BreakNode, ContinueNode
from redbaron import GetitemNode, SliceNode
from redbaron import FromImportNode
from redbaron import DottedAsNameNode, DecoratorNode
from redbaron import NameAsNameNode
from redbaron import LambdaNode
from redbaron import WithNode
from redbaron import AtomtrailersNode

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
from pyccel.ast import Concatenate
from pyccel.ast import ValuedVariable
from pyccel.ast import Argument, ValuedArgument
from pyccel.ast import Is, IsNot
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
from pyccel.parser.utilities import fst_move_directives, preprocess_imports, preprocess_default_args
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
#==============================================================================

strip_ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]|[\n\t\r]')

redbaron.ipython_behavior = False

# use this to delete ansi_escape characters from a string
# Useful for very coarse version differentiation.

#==============================================================================

from pyccel.parser.base import BasicParser
from pyccel.parser.base import is_ignored_module

class SyntaxParser(BasicParser):

    """ Class for a Syntax Parser.

        inputs: str
            filename or code to parse as a string
    """

    def __init__(self, inputs, **kwargs):
        BasicParser.__init__(self, **kwargs)

        # check if inputs is a file
        code = inputs
        if os.path.isfile(inputs):

            # we don't use is_valid_filename_py since it uses absolute path
            # file extension

            ext = inputs.split(""".""")[-1]
            if not ext in ['py', 'pyh']:
                errors = Errors()
                errors.report(INVALID_FILE_EXTENSION, symbol=ext,
                              severity='fatal')
                errors.check()
                raise SystemExit(0)

            code = read_file(inputs)
            self._filename = inputs

        self._code = code

        try:
            code = self.code
            red = RedBaron(code)
        except Exception as e:
            errors = Errors()
            errors.report(INVALID_PYTHON_SYNTAX, symbol='\n' + str(e),
                          severity='fatal')
            errors.check()
            raise e

        preprocess_imports(red)
        preprocess_default_args(red)

        red = fst_move_directives(red)
        self._fst = red

        self.parse(verbose=True)

    def parse(self, verbose=False):
        """converts redbaron fst to sympy ast."""

        if self.syntax_done:
            print ('> syntax analysis already done')
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('syntax')

        # we add the try/except to allow the parser to find all possible errors
        try:
            ast = self._visit(self.fst)
        except Exception as e:
            errors.check()
            traceback.print_exc()
            raise e


        self._ast = ast

        errors.check()
        self._visit_done = True

        return ast

    def _treat_iterable(self, stmt):

        """
        since redbaron puts the first comments after a block statement
        inside the block, we need to remove them. this is in particular the
        case when using openmp/openacc pragmas like #$ omp end loop
        """

        ls = [self._visit(i) for i in stmt]

        if isinstance(stmt, (list, ListNode)):

            return List(*ls, sympify=False)
        else:
            return Tuple(*ls, sympify=False)

    def _visit(self, stmt):
        """Creates AST from FST."""

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        cls = type(stmt)
        syntax_method = '_visit_' + cls.__name__
        if hasattr(self, syntax_method):
            return getattr(self, syntax_method)(stmt)

        # Unknown object, we raise an error.
        raise PyccelSyntaxError('{node} not yet available'.format(node=type(stmt)))


    def _visit_RedBaron(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_LineProxyList(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_CommaProxyList(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_NodeList(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_TupleNode(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_ListNode(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_tuple(self, stmt):
        return self._treat_iterable(stmt)

    def _visit_list(self, stmt):
        return self._treat_iterable(stmt)


    def _visit_DottedAsNameNode(self, stmt):

        names = []
        for a in stmt.value:
            names.append(strip_ansi_escape.sub('', a.value))

        if len(names) == 1:
            return names[0]
        else:

            return DottedName(*names)

    def _visit_NameAsNameNode(self, stmt):

        if not isinstance(stmt.value, str):
            raise TypeError('Expecting a string')

        value = strip_ansi_escape.sub('', stmt.value)
        if not stmt.target:
            return value

        old = value
        new = self._visit(stmt.target)

        # TODO improve

        if isinstance(old, str):
            old = old.replace("'", '')
        if isinstance(new, str):
            new = new.replace("'", '')
        return AsName(new, old)

    def _visit_DictNode(self, stmt):

        d = {}
        for i in stmt.value:
            if not isinstance(i, DictitemNode):
                raise PyccelSyntaxError('Expecting a DictitemNode')

            key = self._visit(i.key)
            value = self._visit(i.value)

            # sympy does not allow keys to be strings

            if isinstance(key, str):
                errors.report(SYMPY_RESTRICTION_DICT_KEYS,
                              severity='error')

            d[key] = value
        return Dict(d)

    def _visit_NoneType(self, stmt):
        return Nil()

    def _visit_str(self, stmt):

        return repr(stmt)

    def _visit_StringNode(self, stmt):
        val =  stmt.value
        if isinstance(stmt.parent,(RedBaron, DefNode)):
            return CommentBlock(val)
        return String(val)

    def _visit_IntNode(self, stmt):

        val = strip_ansi_escape.sub('', stmt.value)
        return Integer(val)

    def _visit_FloatNode(self, stmt):

        val = strip_ansi_escape.sub('', stmt.value)

        val = val[:20] if len(val)>20 else val
        return Float(val)

    def _visit_FloatExponantNode(self, stmt):

        val = strip_ansi_escape.sub('', stmt.value)
        val = val[:20] if len(val)>20 else val
        return Float(val)

    def _visit_ComplexNode(self, stmt):

        val = strip_ansi_escape.sub('', stmt.value)
        return sympify(val, locals=local_sympify)

    def _visit_AssignmentNode(self, stmt):

        lhs = self._visit(stmt.target)
        rhs = self._visit(stmt.value)
        if stmt.operator in ['+', '-', '*', '/']:
            expr = AugAssign(lhs, stmt.operator, rhs)
        else:
            expr = Assign(lhs, rhs)

            # we set the fst to keep track of needed information for errors

        expr.set_fst(stmt)
        return expr

    def _visit_NameNode(self, stmt):
        if stmt.value == 'None':
            return Nil()

        elif stmt.value == 'True':
            return true

        elif stmt.value == 'False':
            return false

        else:
            val = strip_ansi_escape.sub('', stmt.value)
            return Symbol(val)

    def _visit_ImportNode(self, stmt):
        errors.report(PYCCEL_UNEXPECTED_IMPORT,
                      bounding_box=stmt.absolute_bounding_box,
                      severity='error')

        ls = self._visit(stmt.value)
        ls = get_default_path(ls)
        expr = Import(ls)
        expr.set_fst(stmt)
        self.insert_import(expr)
        return expr

    def _visit_FromImportNode(self, stmt):

        if not isinstance(stmt.parent, (RedBaron, DefNode)):
            errors.report(PYCCEL_RESTRICTION_IMPORT,
                          bounding_box=stmt.absolute_bounding_box,
                          severity='error')

        source = self._visit(stmt.value)

        st     = stmt.value[0]
        dots   = ''
        while isinstance(st.previous, DotNode):
            dots  = dots + '.'
            st    = st.previous

        if isinstance(source, DottedVariable):
            source = DottedName(*source.names)

        if len(dots)>1:
            if isinstance(source, DottedName):
                source = DottedName(dots[:-1], *source.name)
            else:
                source = Symbol(dots + str(source.name))

        source = get_default_path(source)
        targets = []
        for i in stmt.targets:
            s = self._visit(i)
            if s == '*':
                errors.report(PYCCEL_RESTRICTION_IMPORT_STAR,
                              bounding_box=stmt.absolute_bounding_box,
                              severity='error')

            targets.append(s)

        if is_ignored_module(source):
            return EmptyLine()

        expr = Import(targets, source=source)
        expr.set_fst(stmt)
        self.insert_import(expr)
        return expr

    def _visit_DelNode(self, stmt):
        arg = self._visit(stmt.value)
        return Del(arg)

    def _visit_UnitaryOperatorNode(self, stmt):

        target = self._visit(stmt.target)
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
            msg = 'unknown/unavailable unary operator {node}'
            msg = msg.format(node=type(stmt.value))
            raise PyccelSyntaxError(msg)

    def _visit_BinaryOperatorNode(self, stmt):

        first = self._visit(stmt.first)
        second = self._visit(stmt.second)
        if stmt.value == '+':
            return Add(first, second, evaluate=False)
        elif stmt.value == '*':

            if isinstance(first, (Tuple, List)):
                return Dlist(first[0], second)
            return Mul(first, second, evaluate=False)
        elif stmt.value == '-':

            if isinstance(stmt.second, BinaryOperatorNode) \
                and isinstance(second, (Add, Mul)):
                args = second.args
                second = second._new_rawargs(-args[0], args[1])
            else:
                second = Mul(-1, second)
            return Add(first, second, evaluate=False)
        elif stmt.value == '/':
            if isinstance(second, Mul) and isinstance(stmt.second,
                                           BinaryOperatorNode):
                args = list(second.args)
                second = Pow(args[0], -1, evaluate=False)
                second = Mul(second, args[1], evaluate=False)
            else:
                second = Pow(second, -1, evaluate=False)
            return Mul(first, second, evaluate=False)

        elif stmt.value == '**':

            return Pow(first, second, evaluate=False)
        elif stmt.value == '//':

            if isinstance(second, Mul) and isinstance(stmt.second,
                                           BinaryOperatorNode):
                args = second.args
                second = Pow(args[0], -1, evaluate=False)
                first =  floor(Mul(first, second, evaluate=False))
                return Mul(first, args[1], evaluate=False)
            else:
                second = Pow(second, -1, evaluate=False)
                return floor(Mul(first, second, evaluate=False))

        elif stmt.value == '%':
            return Mod(first, second)
        else:
            msg = 'unknown/unavailable BinaryOperatorNode {node}'
            msg = msg.format(node=type(stmt.value))
            raise PyccelSyntaxError(msg)


    def _visit_BooleanOperatorNode(self, stmt):

        first = self._visit(stmt.first)
        second = self._visit(stmt.second)
        if stmt.value == 'and':
            return And(first, second, evaluate=False)
        if stmt.value == 'or':
            return Or(first, second, evaluate=False)

        msg = 'unknown/unavailable BooleanOperatorNode {node}'
        msg = msg.format(node=type(stmt.value))
        raise PyccelSyntaxError(msg)

    def _visit_ComparisonNode(self, stmt):

        first = self._visit(stmt.first)
        second = self._visit(stmt.second)
        op = stmt.value.first
        if(stmt.value.second):
            op=op+' '+stmt.value.second

        if op == '==':
            return Eq(first, second, evaluate=False)
        if op == '!=':
            return Ne(first, second, evaluate=False)
        if op == '<':
            return Lt(first, second, evaluate=False)
        if op == '>':
            return Gt(first, second, evaluate=False)
        if op == '<=':
            return Le(first, second, evaluate=False)
        if op == '>=':
            return Ge(first, second, evaluate=False)
        if op == 'is':
            return Is(first, second)
        if op == 'is not':
            return IsNot(first, second)

        msg = 'unknown/unavailable binary operator {node}'
        msg = msg.format(node=op)
        raise PyccelSyntaxError(msg)

    def _visit_PrintNode(self, stmt):

        expr = self._visit(stmt.value)
        return Print(expr)

    def _visit_AssociativeParenthesisNode(self, stmt):

        return self._visit(stmt.value)

    def _visit_DefArgumentNode(self, stmt):

        name = str(self._visit(stmt.target))
        name = strip_ansi_escape.sub('', name)
        arg = Argument(name)
        if stmt.value is None:
            return arg
        else:

            value = self._visit(stmt.value)
            return ValuedArgument(arg, value)

    def _visit_ReturnNode(self, stmt):

        expr = Return(self._visit(stmt.value))
        expr.set_fst(stmt)
        return expr

    def _visit_PassNode(self, stmt):

        return Pass()

    def _visit_DefNode(self, stmt):

        #  TODO check all inputs and which ones should be treated in stage 1 or 2

        if isinstance(stmt.parent, ClassNode):
            cls_name = stmt.parent.name
        else:
            cls_name = None

        name = self._visit(stmt.name)
        name = name.replace("'", '')
        name = strip_ansi_escape.sub('', name)

        arguments    = self._visit(stmt.arguments)
        results      = []
        local_vars   = []
        global_vars  = []
        header       = None
        hide         = False
        kind         = 'function'
        is_pure      = False
        is_elemental = False
        is_private   = False
        is_external  = False
        is_external_call  = False
        imports      = []

        # TODO improve later
        decorators = {}
        for i in stmt.decorators:
            decorators.update(self._visit(i))

        if 'bypass' in decorators:
            return EmptyLine()

        if 'stack_array' in decorators:
            args = decorators['stack_array']
            for i in range(len(args)):
                args[i] = str(args[i]).replace("'", '')
            decorators['stack_array'] = args
        # extract the types to construct a header
        if 'types' in decorators:
            types = []
            results = []
            container = types
            i = 0
            n = len(decorators['types'])
            ls = decorators['types']
            while i<len(ls) :
                arg = ls[i]

                if isinstance(arg, Symbol):
                    arg = arg.name
                    container.append(arg)
                elif isinstance(arg, String):
                    arg = str(arg)
                    arg = arg.strip("'").strip('"')
                    container.append(arg)
                elif isinstance(arg, ValuedArgument):
                    arg_name = arg.name
                    arg  = arg.value
                    container = results
                    if not arg_name == 'results':
                        msg = '> Wrong argument, given {}'.format(arg_name)
                        raise NotImplementedError(msg)
                    ls = arg if isinstance(arg, Tuple) else [arg]
                    i = -1
                else:
                    msg = '> Wrong type, given {}'.format(type(arg))
                    raise NotImplementedError(msg)

                i = i+1

            txt  = '#$ header ' + name
            txt += '(' + ','.join(types) + ')'
            if results:
                txt += ' results(' + ','.join(results) + ')'
            header = hdr_parse(stmts=txt)
            if name in self.namespace.static_functions:
                header = header.to_static()

        body = stmt.value

        if 'sympy' in decorators.keys():
            # TODO maybe we should run pylint here
            stmt.decorators.pop()
            func = SympyFunction(name, arguments, [],
                    [stmt.__str__()])
            func.set_fst(stmt)
            self.insert_function(func)
            return EmptyLine()

        elif 'python' in decorators.keys():

            # TODO maybe we should run pylint here

            stmt.decorators.pop()
            func = PythonFunction(name, arguments, [],
                    [stmt.__str__()])
            func.set_fst(stmt)
            self.insert_function(func)
            return EmptyLine()

        else:
            body = self._visit(body)

        if 'pure' in decorators.keys():
            is_pure = True

        if 'elemental' in decorators.keys():
            is_elemental = True

        if 'private' in decorators.keys():
            is_private = True

        if 'external' in decorators.keys():
            is_external = True

        if 'external_call' in decorators.keys():
            is_external_call = True

        func = FunctionDef(
               name,
               arguments,
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
               is_external=is_external,
               is_external_call=is_external_call,
               imports=imports,
               decorators=decorators,
               header=header)

        func.set_fst(stmt)
        return func

    def _visit_ClassNode(self, stmt):

        name = self._visit(stmt.name)
        methods = [i for i in stmt.value if isinstance(i, DefNode)]
        methods = self._visit(methods)
        attributes = methods[0].arguments
        parent = [i.value for i in stmt.inherit_from]
        expr = ClassDef(name=name, attributes=attributes,
                        methods=methods, parent=parent)

        # we set the fst to keep track of needed information for errors

        expr.set_fst(stmt)
        return expr

    def _visit_AtomtrailersNode(self, stmt):

        return self._visit(stmt.value)

    def _visit_GetitemNode(self, stmt):

        ch = stmt
        args = []
        while isinstance(ch, GetitemNode):
            val = self._visit(ch.value)
            if isinstance(val, Tuple):
                args += val
            else:
                args.insert(0, val)
            ch = ch.previous
        args = tuple(args)
        return args

    def _visit_SliceNode(self, stmt):

        upper = self._visit(stmt.upper)
        lower = self._visit(stmt.lower)
        if not isinstance(upper, Nil) and not isinstance(lower, Nil):

            return Slice(lower, upper)
        elif not isinstance(lower, Nil):

            return Slice(lower, None)
        elif not isinstance(upper, Nil):

            return Slice(None, upper)
        else:

            return Slice(None, None)

    def _visit_DotProxyList(self, stmt):

        n = 0
        ls = []
        while n < len(stmt):
            var = self._visit(stmt[n])
            while n < len(stmt) and not isinstance(stmt[n].next,
                    DotNode):
                n = n + 1
            if n == len(stmt):
                n = n - 1
            if isinstance(stmt[n], GetitemNode):
                args = self._visit(stmt[n])
                var = IndexedBase(var)[args]
            elif isinstance(stmt[n], CallNode):
                var = self._visit(stmt[n])
            ls.append(var)
            n = n + 1

        if len(ls) == 1:
            expr = ls[0]
        else:
            n = 0
            var = DottedVariable(ls[0], ls[1])
            n = 2
            while n < len(ls):
                var = DottedVariable(var, ls[n])
                n = n + 1

            expr = var
        return expr

    def _visit_CallNode(self, stmt):

        if len(stmt.value) > 0 and isinstance(stmt.value[0],
                ArgumentGeneratorComprehensionNode):
            return self._visit(stmt.value[0])

        args = self._visit(stmt.value)
        f_name = str(stmt.previous.value)
        f_name = strip_ansi_escape.sub('', f_name)
        if len(args) == 0:
            args = ( )
        func = Function(f_name)(*args)
        return func

    def _visit_CallArgumentNode(self, stmt):

        target = stmt.target
        val = self._visit(stmt.value)
        if target:
            target = self._visit(target)
            return ValuedArgument(target, val)

        return val

    def _visit_DecoratorNode(self, stmt):

        name = strip_ansi_escape.sub('', stmt.value.dumps())
        args = []
        if stmt.call:
            args = [self._visit(i) for i in stmt.call.value]
        return {name: args}

    def _visit_ForNode(self, stmt):

        iterator = self._visit(stmt.iterator)
        iterable = self._visit(stmt.target)
        body = list(self._visit(stmt.value))
        expr = For(iterator, iterable, body, strict=False)
        expr.set_fst(stmt)
        return expr

    def _visit_ComprehensionLoopNode(self, stmt):

        iterator = self._visit(stmt.iterator)
        iterable = self._visit(stmt.target)
        ifs = stmt.ifs
        expr = For(iterator, iterable, [], strict=False)
        expr.set_fst(stmt)
        return expr

    def _visit_ArgumentGeneratorComprehensionNode(self, stmt):

        result = self._visit(stmt.result)

        generators = self._visit(stmt.generators)
        parent = stmt.parent.parent.parent

        if isinstance(parent, AssignmentNode):
            lhs = self._visit(parent.target)
            name = strip_ansi_escape.sub('', parent.value[0].value)
            cond = False
        else:
            lhs = create_variable(result)
            name = stmt.parent.parent
            name = strip_ansi_escape.sub('', name.value[0].value)
            cond = True
        body = result
        if name == 'sum':
            body = AugAssign(lhs, '+', body)
        else:
            body = Function(name)(lhs, body)
            body = Assign(lhs, body)

        body.set_fst(parent)
        indices = []
        generators = list(generators)
        while len(generators) > 0:
            indices.append(generators[-1].target)
            generators[-1].insert2body(body)
            body = generators.pop()
        indices = indices[::-1]
        body = [body]
        if name == 'sum':
            expr = FunctionalSum(body, result, lhs, indices, None)
        elif name == 'min':
            expr = FunctionalMin(body, result, lhs, indices, None)
        elif name == 'max':
            expr = FunctionalMax(body, result, lhs, indices, None)
        else:
            raise NotImplementedError('TODO')

        expr.set_fst(stmt)
        return expr

    def _visit_IfelseblockNode(self, stmt):

        args = self._visit(stmt.value)
        return If(*args)

    def _visit_IfNode(self, stmt):

        test = self._visit(stmt.test)
        body = self._visit(stmt.value)
        return Tuple(test, body, sympify=False)

    def _visit_ElifNode(self, stmt):
        test = self._visit(stmt.test)
        body = self._visit(stmt.value)
        return Tuple(test, body, sympify=False)

    def _visit_ElseNode(self, stmt):

        test = true
        body = self._visit(stmt.value)
        return Tuple(test, body, sympify=False)

    def _visit_TernaryOperatorNode(self, stmt):

        test1 = self._visit(stmt.value)
        first = self._visit(stmt.first)
        second = self._visit(stmt.second)
        args = [Tuple(test1, [first], sympify=False),
                Tuple(true, [second], sympify=False)]
        expr = IfTernaryOperator(*args)
        expr.set_fst(stmt)
        return expr

    def _visit_WhileNode(self, stmt):

        test = self._visit(stmt.test)
        body = self._visit(stmt.value)
        return While(test, body)

    def _visit_AssertNode(self, stmt):

        expr = self._visit(stmt.value)
        return Assert(expr)

    def _visit_EndlNode(self, stmt):

        return NewLine()

    def _visit_CommentNode(self, stmt):

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

                    # return NewLine()

                    expr = EmptyLine()
                else:
                    expr.set_fst(stmt)

                return expr
            else:

                # TODO an info should be reported saying that either we
                # found a multiline pragma or an invalid pragma statement

                return NewLine()
        else:

#                    errors.report(PYCCEL_INVALID_HEADER,
#                                  bounding_box=stmt.absolute_bounding_box,
#                                  severity='error')

            # TODO improve

            txt = stmt.value[1:].lstrip()
            return Comment(txt)

    def _visit_BreakNode(self, stmt):
        return Break()

    def _visit_ContinueNode(self, stmt):
        return Continue()

    def _visit_StarNode(self, stmt):
        return '*'

    def _visit_LambdaNode(self, stmt):

        expr = self._visit(stmt.value)
        args = []

        for i in stmt.arguments:
            var = self._visit(i.name)
            args += [var]

        return Lambda(args, expr)

    def _visit_WithNode(self, stmt):
        domain = self._visit(stmt.contexts[0].value)
        body = self._visit(stmt.value)
        settings = None
        return With(domain, body, settings)

    def _visit_ListComprehensionNode(self, stmt):

        import numpy as np
        result = self._visit(stmt.result)
        generators = list(self._visit(stmt.generators))
        lhs = self._visit(stmt.parent.target)
        index = create_variable(lhs)
        if isinstance(result, (Tuple, list, tuple)):
            rank = len(np.shape(result))
        else:
            rank = 0
        args = [Slice(None, None)] * rank
        args.append(index)
        target = IndexedBase(lhs)[args]
        target = Assign(target, result)
        assign1 = Assign(index, Integer(0))
        assign1.set_fst(stmt)
        target.set_fst(stmt)
        generators[-1].insert2body(target)
        assign2 = Assign(index, index + 1)
        assign2.set_fst(stmt)
        generators[-1].insert2body(assign2)

        indices = [generators[-1].target]
        while len(generators) > 1:
            F = generators.pop()
            generators[-1].insert2body(F)
            indices.append(generators[-1].target)
        indices = indices[::-1]
        return FunctionalFor([assign1, generators[-1]],target.rhs, target.lhs,
                             indices, index)

    def _visit_TryNode(self, stmt):
        # this is a blocking error, since we don't want to convert the try body
        errors.report(PYCCEL_RESTRICTION_TRY_EXCEPT_FINALLY,
                      bounding_box=stmt.absolute_bounding_box,
                      severity='error')

    def _visit_RaiseNode(self, stmt):
        errors.report(PYCCEL_RESTRICTION_RAISE,
                      bounding_box=stmt.absolute_bounding_box,
                      severity='error')

    def _visit_YieldAtomNode(self, stmt):
        errors.report(PYCCEL_RESTRICTION_YIELD,
                      bounding_box=stmt.absolute_bounding_box,
                      severity='error')

    def _visit_YieldNode(self, stmt):
        errors.report(PYCCEL_RESTRICTION_YIELD,
                      bounding_box=stmt.absolute_bounding_box,
                      severity='error')

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except:
        raise ValueError('Expecting an argument for filename')

    parser = SyntaxParser(filename)
    print(parser.ast)

