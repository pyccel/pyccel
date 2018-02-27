# coding: utf-8
from redbaron import RedBaron
from redbaron import IntNode, FloatNode, ComplexNode
from redbaron import NameNode
from redbaron import AssignmentNode
from redbaron import CommentNode
from redbaron import BinaryOperatorNode


from pyccel.ast import NativeInteger, NativeFloat, NativeDouble, NativeComplex
from pyccel.ast import Variable
from pyccel.ast import Assign
from pyccel.ast import Comment

from sympy import Symbol
from sympy import Add
from sympy import Integer, Float

def view_tree(expr):
    """Views a sympy expression tree."""
    from sympy import srepr
    print srepr(expr)


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
        return [fst_to_ast(i) for i in stmt]
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
        else:
            raise ValueError('unknown/unavailable operator '
                             '{node}'.format(node=type(stmt.value)))
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

view_tree(ast)



#stmt = red[0]

#type(stmt)
#stmt.target
#type(stmt.target)
#stmt.value
#type(stmt.value)

#
#
#dtype = datatype_from_redbaron(stmt.value)
#var   = Variable(dtype, str(stmt.target))
#
#expr = Assign(var, stmt.value)
