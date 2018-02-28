# coding: utf-8
from redbaron import RedBaron
from redbaron import IntNode, FloatNode, ComplexNode
from redbaron import NameNode
from redbaron import AssignmentNode
from redbaron import CommentNode, EndlNode
from redbaron import BinaryOperatorNode
from redbaron import AssociativeParenthesisNode
from redbaron import DefNode
from redbaron import ListNode
from redbaron import CommaProxyList
from redbaron import LineProxyList
from redbaron import ReturnNode
from redbaron import DefArgumentNode
from redbaron import ForNode


from pyccel.ast import NativeInteger, NativeFloat, NativeDouble, NativeComplex
from pyccel.ast import Nil
from pyccel.ast import Variable
from pyccel.ast import Assign
from pyccel.ast import Return
from pyccel.ast import FunctionDef
from pyccel.ast import For
from pyccel.ast import Comment, EmptyLine


from sympy import Symbol
from sympy import Tuple
from sympy import Add, Mul, Pow
from sympy import Integer, Float

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

    print srepr(expr)

def export_ast(expr, filename):
    """Exports sympy AST using graphviz then convert it to an image."""

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
    if isinstance(stmt, (RedBaron, CommaProxyList, LineProxyList, ListNode)):
        ls = [fst_to_ast(i) for i in stmt]
        return Tuple(*ls)
    elif stmt is None:
        return Nil()
    elif isinstance(stmt, str):
        return stmt
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
        else:
            return Symbol(stmt.value)
    elif isinstance(stmt, BinaryOperatorNode):
        first  = fst_to_ast(stmt.first)
        second = fst_to_ast(stmt.second)
        if stmt.value == '+':
            return Add(first, second)
        elif stmt.value == '*':
            return Mul(first, second)
        elif stmt.value == '-':
            second = Mul(-1, second)
            return Add(first, second)
        elif stmt.value == '/':
            second = Pow(second, -1)
            return Mul(first, second)
        else:
            raise ValueError('unknown/unavailable operator '
                             '{node}'.format(node=type(stmt.value)))
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
    elif isinstance(stmt, DefNode):
        # TODO results must be computed at the decoration stage
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
    elif isinstance(stmt, ForNode):
        target = fst_to_ast(stmt.iterator)
        iter   = fst_to_ast(stmt.target)
        body   = fst_to_ast(stmt.value)
        strict = True
        return For(target, iter, body, strict=strict)
    elif isinstance(stmt, EndlNode):
        return EmptyLine()
    elif isinstance(stmt, CommentNode):
        # TODO must check if it is a header or not
        return Comment(stmt.value)
    else:
        raise NotImplementedError('{node} not yet available'.format(node=type(stmt)))





def read_file(filename):
    """Returns the source code from a filename."""
    f = open(filename)
    code = f.read()
    f.close()
    return code

#code = read_file('ex_redbaron.py')
#code = read_file('ex_function.py')
code = read_file('ex_for.py')
red  = RedBaron(code)

print('----- FST -----')
for stmt in red:
    print stmt
#    print type(stmt)
print('---------------')

# converts redbaron fst to sympy ast
ast = fst_to_ast(red)

print('----- AST -----')
for expr in ast:
    print expr
#    print '\t', type(expr.rhs)
print('---------------')

#view_tree(ast)

export_ast(ast, filename='ast.gv')
