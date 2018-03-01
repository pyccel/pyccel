# coding: utf-8
from redbaron import RedBaron
from redbaron import IntNode, FloatNode, ComplexNode,StringNode
from redbaron import NameNode
from redbaron import AssignmentNode
from redbaron import CommentNode, EndlNode
from redbaron import BinaryOperatorNode,BooleanOperatorNode
from redbaron import AssociativeParenthesisNode
from redbaron import DefNode
from redbaron import ListNode,TupleNode
from redbaron import CommaProxyList,CommaNode
from redbaron import LineProxyList,DotProxyList
from redbaron import ReturnNode
from redbaron import DefArgumentNode,CallNode,CallArgumentNode
from redbaron import ForNode
from redbaron import DotNode,AtomtrailersNode,PrintNode


from pyccel.ast import NativeInteger, NativeFloat, NativeDouble, NativeComplex
from pyccel.ast import Nil
from pyccel.ast import Variable,DottedName
from pyccel.ast import Assign
from pyccel.ast import FunctionDef,FunctionCall
from pyccel.ast import For,Range
from pyccel.ast import Comment, EmptyLine,Print
from pyccel import fcode



from sympy import Symbol
from sympy import Tuple
from sympy import Add, Mul, Pow
from sympy import Integer, Float
from sympy import sympify
# ... TODO should be moved to pyccel.ast
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
# ...

# ... utilities
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
# ...


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
    if isinstance(stmt, (RedBaron, CommaProxyList, LineProxyList, ListNode,TupleNode)):
        ls = [fst_to_ast(i) for i in stmt]
        return Tuple(*ls)
    elif stmt is None:
        return Nil()
    elif isinstance(stmt,StringNode):
        return stmt.value
    elif isinstance(stmt,str):
        return stmt
    elif isinstance(stmt, AssignmentNode):
        lhs = fst_to_ast(stmt.target)
        rhs = fst_to_ast(stmt.value)
        return Assign(lhs, rhs)
    elif isinstance(stmt, NameNode):
        if stmt.previous and not isinstance(stmt.previous,CommaNode):
            return fst_to_ast(stmt.previous)
        if stmt.value == 'None':
            return Nil()
        else:
            return Symbol(stmt.value)
    elif isinstance(stmt,PrintNode):
         ls=fst_to_ast(stmt.value)
         return Print(ls)
    elif isinstance(stmt,AtomtrailersNode):
         return fst_to_ast(stmt.value)
    elif isinstance(stmt,(IntNode, FloatNode, ComplexNode)):
        return sympify(stmt.value)
    elif isinstance(stmt, AssociativeParenthesisNode):
        return fst_to_ast(stmt.value)
    elif isinstance(stmt,(BooleanOperatorNode,BinaryOperatorNode)) :
            return op(fst_to_ast(stmt.first),fst_to_ast(stmt.second),stmt.value)
    elif isinstance(stmt, DefArgumentNode):
        arg = Argument(str(stmt.target))
        if stmt.value is None:
            return arg
        else:
            value = fst_to_ast(stmt.value)
            return ValuedArgument(arg, value)
    elif isinstance(stmt,DotProxyList):
        node=fst_to_ast(stmt[-1])
        if len(stmt)>1:
            stmt.pop()
            stmt.pop()
        return node
    elif isinstance(stmt,DotNode):
        pre=fst_to_ast(stmt.previous)
        suf=stmt.next
        stmt.parent.value.remove(stmt.previous)
        return DottedName(str(pre),str(fst_to_ast(suf)))
    elif isinstance(stmt,CallNode) and stmt.previous.name.value=='range':
        return Range(*fst_to_ast(stmt.value))
    elif isinstance(stmt,CallNode) and not stmt.previous.name.value=='range':
        return FunctionCall(str(fst_to_ast(stmt.previous.name)),fst_to_ast(stmt.value)) 
    elif isinstance(stmt,CallArgumentNode):
        return fst_to_ast(stmt.value)
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


def op(x,y,operator):
    if operator=='*':
        return x*y
    elif operator=='+':
        return x+y
    elif operator=='and':
        return x and y
    elif operator=='or':
        return x or y
    elif operator=='**':
        return Pow(x,y)
    else:
        raise ValueError('unknown/unavailable operator {node}'.format())

def read_file(filename):
    """Returns the source code from a filename."""
    f = open(filename)
    code = f.read()
    f.close()
    return code

code = read_file('example.py')
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

