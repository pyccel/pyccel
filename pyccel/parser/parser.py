# coding: utf-8
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




from pyccel.ast import NativeInteger, NativeFloat, NativeDouble, NativeComplex
from pyccel.ast import Nil
from pyccel.ast import Variable
from pyccel.ast import DottedName
from pyccel.ast import Assign, AliasAssign
from pyccel.ast import Return
from pyccel.ast import Pass
from pyccel.ast import FunctionDef
from pyccel.ast import ClassDef
from pyccel.ast import For
from pyccel.ast import If
from pyccel.ast import While
from pyccel.ast import Print
from pyccel.ast import Del
from pyccel.ast import Assert
from pyccel.ast import Comment, EmptyLine
from pyccel.ast import Break
from pyccel.ast import Slice, IndexedVariable, IndexedElement
from pyccel.ast import FunctionHeader
from pyccel.ast import Concatinate

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


import os



# ... TODO should be moved to pyccel.ast
from sympy.core.basic import Basic

class Argument(Symbol):
    """An abstract Argument data structure."""
    pass

class ValuedArgument(Basic):
    """Represents a valued argument in the code."""

    def __new__(cls, expr, value):
        if not isinstance(expr, Argument):
            raise TypeError('Expecting an argument')
        return Basic.__new__(cls, expr, value)

    @property
    def argument(self):
        return self._args[0]

    @property
    def value(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint

        argument = sstr(self.argument)
        value    = sstr(self.value)
        return '{0}={1}'.format(argument, value)
# ...

# ... utilities
from sympy import srepr
from sympy.printing.dot import dotprint

import os

def view_tree(expr):
    """Views a sympy expression tree."""

    print(srepr(expr))
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
    if isinstance(stmt, (RedBaron,
                         CommaProxyList, LineProxyList, NodeList,
                         TupleNode, ListNode,
                         list, tuple)):
        ls = [fst_to_ast(i) for i in stmt]
        return Tuple(*ls)
    elif isinstance(stmt, DictNode):
        d = {}
        for i in stmt.value:
            if not isinstance(i, DictitemNode):
                raise TypeError('Expecting a DictitemNode')
            key   = fst_to_ast(i.key)
            value = fst_to_ast(i.value)
            # sympy does not allow keys to be strings
            if isinstance(key, str):
                raise TypeError('sympy does not allow keys to be strings')
            d[key] = value
        return Dict(d)
    elif stmt is None:
        return Nil()
    elif isinstance(stmt, str):
        return repr(stmt)
    elif isinstance(stmt, StringNode):
        return stmt.value
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
        if stmt.value == 'None':
            return Nil()
        elif stmt.value == 'True':
            return true
        elif stmt.value == 'False':
            return false
        else:
            return Symbol(stmt.value)
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
            raise ValueError('Invert unary operator is not covered by Pyccel.')
        else:
            raise ValueError('unknown/unavailable unary operator '
                             '{node}'.format(node=type(stmt.value)))
    elif isinstance(stmt, (BinaryOperatorNode, BooleanOperatorNode)):
        first  = fst_to_ast(stmt.first)
        second = fst_to_ast(stmt.second)
        if stmt.value == '+':
            if isinstance(first,str) or isinstance(second,str):
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
            raise ValueError('unknown/unavailable binary operator '
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
        else:
            raise ValueError('unknown comparison operator {}'.format(stmt.first))
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
        else:
            raise ValueError('unknown/unavailable binary operator '
                             '{node}'.format(node=type(op)))
    elif isinstance(stmt, PrintNode):
        expr = fst_to_ast(stmt.value)
        return Print(expr)
    elif isinstance(stmt, AssociativeParenthesisNode):
        return fst_to_ast(stmt.value)
    elif isinstance(stmt, DefArgumentNode):
        arg = Argument(str(stmt.target))
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
        name        = fst_to_ast(stmt.name)
        arguments   = fst_to_ast(stmt.arguments)
        results     = []
        body        = fst_to_ast(stmt.value)
        local_vars  = []
        global_vars = []
        cls_name    = None
        hide        = False
        kind        = 'function'
        imports     = []
        return FunctionDef(name, arguments, results, body,
                           local_vars=local_vars, global_vars=global_vars,
                           cls_name=cls_name, hide=hide,
                           kind=kind, imports=imports)
    elif isinstance(stmt, ClassNode):
        name = fst_to_ast(stmt.name)
        methods = [i for i in stmt.value if isinstance(i, DefNode)]
        methods = fst_to_ast(methods)
        attributes = methods[0].arguments
        return ClassDef(name, attributes, methods)

    elif isinstance(stmt, AtomtrailersNode):
         return fst_to_ast(stmt.value)
    elif isinstance(stmt, GetitemNode):
         return fst_to_ast(stmt.value)
    elif isinstance(stmt,SliceNode):
         upper = fst_to_ast(stmt.upper)
         lower = fst_to_ast(stmt.lower)
         if upper and lower:
             return Slice(lower,upper)
         elif lower:
             return Slice(lower,None)
         elif upper:
             return Slice(None,upper)
    elif isinstance(stmt, DotProxyList):
        # TODO handle dot trailers
        # we first get the index of the call node
        s=[isinstance(i.next, DotNode) for i in stmt]
        if not any(s):
            # case of no DotNode
            # TODO fix bug of case IndexedVariable before DottedNode
            ls = [fst_to_ast(i) for i in stmt]
            name = str(ls[0])
            args = ls[1:]
            args = args[0]
            if not hasattr(args, '__iter__'):
                args = [args]

            args = tuple(args)
            return IndexedBase(name)[args]
        call = None
        ls = []
        for i, s in enumerate(stmt):
            if isinstance(s, CallNode):
                call = stmt[i]
                break
            # we only take the string representation
            # since we will use DottedName later
            #ls.append(fst_to_ast(s))
            ls.append(repr(s))
            i += 1

        # the function name (without trailer) is the previous node to the call
        if len(ls) == 1:
            name = ls[0]
        else:
            name = DottedName(*ls)

        # dots may lead to a dotted name or a call to a method
        if call is None:
            return name
        elif isinstance(call, CallNode):
            # in this case, name must be a string
            args   = fst_to_ast(call)
            f_name = str(name)
            return Function(f_name)(*args)
        else:
            raise ValueError('Expecting a method call or a dotted name')
    elif isinstance(stmt, CallNode):
        args = fst_to_ast(stmt.value)
        return args
    elif isinstance(stmt, CallArgumentNode):
        return fst_to_ast(stmt.value)
    elif isinstance(stmt, ForNode):
        target = fst_to_ast(stmt.iterator)
        iter   = fst_to_ast(stmt.target)
        body   = fst_to_ast(stmt.value)
        strict = True
        return For(target, iter, body, strict=strict)
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
                raise ValueError('Annotated comments must start with omp, acc or header.')
        else:
            return Comment(stmt.value)
    elif isinstance(stmt, BreakNode):
        return Break()
    elif isinstance(stmt, (ExceptNode, FinallyNode, RaiseNode, TryNode, YieldNode, YieldAtomNode)):
        # TODO add appropriate message errors and refeer to Pyccel rules
        raise NotImplementedError('{node} is not covered by pyccel'.format(node=type(stmt)))
    else:
        raise NotImplementedError('{node} not yet available'.format(node=type(stmt)))


namespace = {}
# TODO use another name for headers
#      => reserved keyword, or use __
namespace['headers'] = {}

def infere_type(expr, **settings):
    """
    type inference for expressions
    """
    d_var = {}
    d_var['datatype'] = None
    d_var['allocatable'] = None
    d_var['shape'] = None
    d_var['rank'] = None
    d_var['is_pointer'] = None
    # TODO - IndexedVariable
    #      - IndexedElement

    if isinstance(expr, Integer):
        d_var['datatype'] = 'int'
        d_var['allocatable'] = False
        d_var['rank'] = 0
        return d_var
    elif isinstance(expr, Variable):
        d_var['datatype'] = expr.dtype
        d_var['allocatable'] = expr.allocatable
        d_var['shape'] = expr.shape
        d_var['rank'] = expr.rank
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
        if name in namespace:
            var = namespace[name]
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
#            # TODO pointer case
#            d_var['is_pointer'] = var.is_pointer
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
    elif isinstance(expr, Expr):
        ds = [infere_type(i, **settings) for i in expr.args]

        dtypes = [d['datatype'] for d in ds]
        allocatables = [d['allocatable'] for d in ds]
        ranks = [d['rank'] for d in ds]

        # ... only scalars and variables of rank 0 can be handled
        r_min = min(ranks)
        r_max = max(ranks)
        if not(r_min == r_max):
            if not(r_min == 0):
                raise ValueError('cannot process arrays of different ranks.')
        rank = r_max
        # ...

        # TODO improve
        d_var['datatype'] = dtypes[0]
        d_var['allocatable'] = allocatables[0]
        d_var['rank'] = rank
        return d_var
    elif isinstance(expr, (tuple, list, Tuple)):
        d = infere_type(expr[0], **settings)
        # TODO must check that it is consistent with pyccel's rules
        d_var['datatype']    = d['datatype']
        d_var['allocatable'] = d['allocatable']
        d_var['rank']        = d['rank'] + 1
        d_var['shape']       = len(expr) # TODO improve
        return d_var
    else:
        raise NotImplementedError('{expr} not yet available'.format(expr=type(expr)))


def _annotate(expr, **settings):
    """Annotates the AST.

    IndexedVariable atoms are only used to manipulate expressions, we then,
    always have a Variable in the namespace."""
    if isinstance(expr, (list, tuple, Tuple)):
        ls = []
        for i in expr:
            a = _annotate(i, **settings)
            ls.append(a)
        return Tuple(*ls)
    elif isinstance(expr, (Integer, Float)):
        return expr
    elif isinstance(expr, Variable):
        return expr
    elif isinstance(expr, (IndexedVariable, IndexedBase)):
        # an indexed variable is only defined if the associated variable is in
        # the namespace
        name = str(expr.name)
        if not name in namespace:
            raise ValueError('Undefined indexed variable {}'.format(name))
        var = namespace[name]
        dtype = var.dtype
        # TODO add shape
        return IndexedVariable(name, dtype=dtype)
    elif isinstance(expr, (IndexedElement, Indexed)):
        name = str(expr.base)
        if not name in namespace:
            raise ValueError('Undefined indexed variable {}'.format(name))
        # TODO check consistency of indices with shape/rank
        args = tuple(expr.indices)
        var = namespace[name]
        dtype = var.dtype
        return IndexedVariable(name, dtype=dtype).__getitem__(*args)
    elif isinstance(expr, Symbol):
        name = str(expr.name)
        if name in namespace:
            return namespace[name]
        else:
            # TODO improve this when using symbolic extension
            raise NotImplementedError('No Symbolic variable are allowed')
    elif isinstance(expr, Expr):
        args = expr.args
        symbols = expr.free_symbols
        names = [a.name for a in symbols]

        args = list(symbols)
        for a in args:
            name = a.name
            a_new = _annotate(a, **settings)
            expr = expr.subs(a, a_new)
        return expr
    elif isinstance(expr, Assign):
        rhs = _annotate(expr.rhs, **settings)
        d_var = infere_type(rhs, **settings)

        lhs = expr.lhs
        if isinstance(lhs, Symbol):
            name = lhs.name
            dtype = d_var.pop('datatype')
            lhs = Variable(dtype, name, **d_var)
            if not name in namespace:
                namespace[lhs.name] = lhs
        elif isinstance(lhs, (IndexedVariable, IndexedBase)):
            name = str(lhs.name)
            if not name in namespace:
                raise ValueError('Undefined indexed variable {}'.format(name))
            # TODO check consistency of indices with shape/rank
            var = namespace[name]
            dtype = var.dtype
            lhs = IndexedVariable(name, dtype=dtype)
        elif isinstance(lhs, (IndexedElement, Indexed)):
            name = str(lhs.base)
            if not name in namespace:
                raise ValueError('Undefined indexed variable {}'.format(name))
            # TODO check consistency of indices with shape/rank
            args = tuple(lhs.indices)
            var = namespace[name]
            dtype = var.dtype
            lhs = IndexedVariable(name, dtype=dtype).__getitem__(*args)
        else:
            raise NotImplementedError('Uncovered type {dtype}'.format(dtype=type(lhs)))

        expr = Assign(lhs, rhs, strict=False)
        # we check here, if it is an alias assignment
        if expr.is_alias:
#            print '------'
#            print expr
#            print expr.is_alias
#            print '------'
            return AliasAssign(expr.lhs, expr.rhs)
        else:
            return expr
    elif isinstance(expr, FunctionHeader):
        # TODO should we return it and keep it in the AST?
        namespace['headers'][str(expr.func)] = expr
        return expr
    elif isinstance(expr, Return):
        results = expr.expr
        if isinstance(results, Symbol):
            name = results.name
            if not name in namespace:
                raise ValueError('Undefined returned variable {name}'.format(name=name))
            var = namespace[name]
            return Return([var])
        elif isinstance(results, (list, tuple, Tuple)):
            ls = []
            for i in results:
                if not isinstance(i, Symbol):
                    raise NotImplementedError('only symbol or iterable are allowed for returns')
                name = i.name
                if not name in namespace:
                    raise ValueError('Undefined returned variable {name}'.format(name=name))
                var = namespace[name]
                ls += [var]
            return Return(ls)
        else:
            raise NotImplementedError('only symbol or iterable are allowed for returns')
    elif isinstance(expr, FunctionDef):
        name = str(expr.name)
        name = name.replace('\'', '') # remove quotes for str representation
        args = []
        results = []
        local_vars  = []
        global_vars = []
        cls_name    = None
        hide        = False
        kind        = 'function'
        imports     = []

        if expr.arguments or results:
            header = namespace['headers'].pop(name, None)
            if not header:
                raise ValueError('Expecting a header function for {func} '
                                 'but could not find it.'.format(func=name))

            # we construct a FunctionDef from its header
            interface = header.create_definition()

        # then use it to decorate our arguments
        if expr.arguments:
            for a, ah in zip(expr.arguments, interface.arguments):
                d_var = infere_type(ah, **settings)
                dtype = d_var.pop('datatype')
                a_new = Variable(dtype, a.name, **d_var)
                args.append(a_new)

                # TODO add scope and store already declared variables there,
                #      then retrieve them
                namespace[str(a_new.name)] = a_new

        # we annotate the body
        body = _annotate(expr.body, **settings)

        # find return stmt and results
        # we keep the return stmt, in case of handling multi returns later
        for stmt in body:
            # TODO case of multiple calls to return
            if isinstance(stmt, Return):
                results = stmt.expr
                if isinstance(results, (Symbol, Variable)):
                    results = [results]

        if results:
            _results = []
            for a, ah in zip(results, interface.results):
#                a.inspect()
#                ah.inspect()
                d_var = infere_type(ah, **settings)
                dtype = d_var.pop('datatype')
                a_new = Variable(dtype, a.name, **d_var)
                _results.append(a_new)

                # results must be variable that were already declared
                if not(str(a_new.name) in namespace):
                    raise ValueError('can not return undeclared variables')
            results = _results

        return FunctionDef(name, args, results, body,
                           local_vars=local_vars, global_vars=global_vars,
                           cls_name=cls_name, hide=hide,
                           kind=kind, imports=imports)
    elif isinstance(expr, EmptyLine):
        return expr
    elif isinstance(expr, Print):
        # TODO improve
        return expr
    elif isinstance(expr, Comment):
        return expr
    else:
        raise NotImplementedError('{expr} not yet available'.format(expr=type(expr)))


def print_namespace():
    print('>>>> Namespace ')
    for k,v in namespace.items():
        print(k, ' :: ', type(v))
    print('<<<<')


def _read_file(filename):
    """Returns the source code from a filename."""
    f = open(filename)
    code = f.read()
    f.close()
    return code



class Parser(object):
    """ Class for a Parser."""
    def __init__(self, inputs, debug=False):
        """Parser constructor.

        inputs: str
            filename or code to parse as a string

        debug: bool
            True if in debug mode.
        """
        self._fst = None
        self._ast = None
        namespace = {}

        # check if inputs is a file
        code = inputs
        if os.path.isfile(inputs):
            code = _read_file(inputs)

        self._code = code

        red = RedBaron(code)
        self._fst = red

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
        ast = fst_to_ast(self.fst)
        self._ast = ast
        return ast

    def annotate(self, **settings):
        """."""
        ast = self.ast
        ast = _annotate(ast, **settings)
        self._ast = ast
        return ast

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
    print_namespace()
