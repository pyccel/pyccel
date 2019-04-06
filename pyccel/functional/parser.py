# coding: utf-8

import os
from os.path import join, dirname

from sympy import Symbol, Lambda, Function, Dummy
from sympy import sympify

from textx.metamodel import metamodel_from_str

namespace = {}

#==============================================================================
# any argument
class AnyArgument(Symbol):
    pass

_ = AnyArgument('_')

#==============================================================================
class NamedAbstraction(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.abstraction = kwargs.pop('abstraction')

class Abstraction(object):
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')
        self.expr = kwargs.pop('expr')

class Application(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')

#==============================================================================
def to_sympy(stmt):

    if isinstance(stmt, NamedAbstraction):
        name = stmt.name
        expr = to_sympy(stmt.abstraction)
        # TODO add it to the namespace
#        namespace[name] = expr
        return expr

    elif isinstance(stmt, Abstraction):
        args = [to_sympy(i) for i in stmt.args]
        expr = to_sympy(stmt.expr)

        return Lambda(args, expr)

    elif isinstance(stmt, Application):
        args = [to_sympy(i) for i in stmt.args]
        name = stmt.name

        return Function(name)(*args)

    elif isinstance(stmt, (int, float)):
        return stmt

    elif isinstance(stmt, str):
        if stmt == '_':
            return _

        else:
            return sympify(stmt)

    else:
        print(stmt)
        raise TypeError('Not implemented for {}'.format(type(stmt)))

#==============================================================================
def parse(inputs, debug=False):
    this_folder = dirname(__file__)

    classes = [NamedAbstraction, Abstraction, Application]

    # Get meta-model from language description
    grammar = join(this_folder, 'grammar.tx')

    from textx.metamodel import metamodel_from_file
    meta = metamodel_from_file(grammar, debug=debug, classes=classes)

    # Instantiate model
    if os.path.isfile(inputs):
        ast = meta.model_from_file(inputs)

    else:
        ast = meta.model_from_str(inputs)

    expr = to_sympy(ast)
    print(expr)

    return expr
