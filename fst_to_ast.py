# coding: utf-8
from redbaron import RedBaron
from redbaron import IntNode, FloatNode, ComplexNode
from redbaron import NameNode
from redbaron import AssignmentNode
from redbaron import CommentNode, EndlNode
from redbaron import BinaryOperatorNode
from redbaron import AssociativeParenthesisNode


from pyccel.ast import NativeInteger, NativeFloat, NativeDouble, NativeComplex
from pyccel.ast import Variable
from pyccel.ast import Assign
from pyccel.ast import Comment, EmptyLine


from sympy import Symbol
from sympy import Tuple
from sympy import Add, Mul, Pow
from sympy import Integer, Float

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
    if isinstance(stmt, RedBaron):
        ls = [fst_to_ast(i) for i in stmt]
        return Tuple(*ls)
    elif isinstance(stmt, AssignmentNode):
        lhs = fst_to_ast(stmt.target)
        rhs = fst_to_ast(stmt.value)
        return Assign(lhs, rhs)
    elif isinstance(stmt, NameNode):
        return Symbol(str(stmt))
    elif isinstance(stmt, IntNode):
        return Integer(stmt.value)
    elif isinstance(stmt, FloatNode):
        return Float(stmt.value)
    elif isinstance(stmt, ComplexNode):
        raise NotImplementedError('ComplexNode not yet available')
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
            second = Mul(-1, second)
            return Mul(first, second)
        else:
            raise ValueError('unknown/unavailable operator '
                             '{node}'.format(node=type(stmt.value)))
    elif isinstance(stmt, AssociativeParenthesisNode):
        raise NotImplementedError('TODO')
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

code = read_file('ex_redbaron.py')
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
