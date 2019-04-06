# coding: utf-8

import os
from os.path import join, dirname

from sympy import Symbol, Lambda, Function, Dummy
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import Integer, Float
from sympy import sympify

from textx.metamodel import metamodel_from_str


from pyccel.codegen.utilities import random_string
from pyccel.ast.core import Variable, FunctionDef
from .ast import Reduce
from .ast import SeqMap, ParMap, BasicMap
from .ast import SeqTensorMap, ParTensorMap, BasicTensorMap
from .ast import SeqZip, SeqProduct
from .ast import ParZip, ParProduct
from .ast import assign_type, BasicTypeVariable

_known_functions = {'map':      SeqMap,
                    'pmap':     ParMap,
                    'tmap':     SeqTensorMap,
                    'ptmap':    ParTensorMap,
                    'zip':      SeqZip,
                    'pzip':     ParZip,
                    'product':  SeqProduct,
                    'pproduct': ParProduct,
                    'reduce':   Reduce,
                   }

_functors_registery = ['map', 'pmap', 'tmap', 'ptmap', 'reduce']

#==============================================================================
# TODO to be moved in a class
# utilities for semantic analysis
namespace  = {}

# keys = arguments            ||  values = ?
signatures = {}
# keys = global arguments and functions    ||  values = dictionary for d_var
d_types    = {}
# keys = arguments     ||  values = FunctionDef or Lambda
signatures_parent = {}

main_expr = None
#==============================================================================

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
        raise TypeError('Not implemented for {}'.format(type(stmt)))

#==============================================================================
def sanitize(expr):
    if isinstance(expr, Lambda):
        args = expr.variables
        expr = sanitize(expr.expr)

        return Lambda(args, expr)

    elif isinstance(expr, AppliedUndef):
        name = expr.__class__.__name__

        args = [sanitize(i) for i in expr.args]
        # first argument of Map & Reduce are functions
        if name in _functors_registery:
            first = args[0]
            if isinstance(first, Symbol):
                args[0] = Function(first.name)

        if name in _known_functions.keys():
            return _known_functions[name](*args)

        else:
            return Function(name)(*args)

    elif isinstance(expr, (int, float, Integer, Float, Symbol)):
        return expr

    else:
        raise TypeError('Not implemented for {}'.format(type(expr)))

#==============================================================================
def parse(inputs, debug=False, verbose=False):
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

    # ...
    expr = to_sympy(ast)
    if verbose:
        print('>>> stage 0 = ', expr)
    # ...

#    # ...
#    expr = sanitize(expr)
#    if verbose:
#        print('>>> stage 1 = ', expr)
#    # ...

    # ...
    if verbose:
        print('')
    # ...

    return expr


#==============================================================================
def annotate(L, typed_functions):
    compute_types(L, typed_functions)
    compute_shapes(L, typed_functions)

#==============================================================================
def compute_types(L, typed_functions):
    # ... add types for arguments and results
    for f in typed_functions.values():
        d_types[_get_key(f)] = assign_type(f.arguments)
        d_types[str(f.name)] = assign_type(f.results)
    # ...

    global main_expr

    i_count = 0
    max_count = 2
    main_expr = L
    while(i_count < max_count and not isinstance(main_expr, Variable)):
        print('>>> before ', main_expr)
        main_expr = _compute_types(main_expr)
        print('>>> after', main_expr)
        i_count += 1

    print(main_expr.view())
    import sys; sys.exit(0)

def _get_key(expr):
    # TODO to be replaced by domain
    if isinstance(expr, FunctionDef):
        return str(expr.name) + '_args'

    elif isinstance(expr, UndefinedFunction):
        return str(expr)

    elif isinstance(expr, Symbol):
        return expr.name

    else:
        raise NotImplementedError('for {}'.format(type(expr)))

#==============================================================================
def _compute_types(expr):
    if isinstance(expr, Lambda):
        args = expr.variables
        return _compute_types(expr.expr)

    elif isinstance(expr, BasicTypeVariable):
        return expr

    elif isinstance(expr, AppliedUndef):
        name = expr.__class__.__name__
        arguments = expr.args
        if name == 'map':
            assert( len(arguments) == 2 )
            func   = arguments[0]
            target = arguments[1]

            key = _get_key(func)
            if key in d_types.keys():
                if isinstance(target, Symbol):
#                    print('> target = ', target)

                    # increment rank
                    d_types[_get_key(target)] = assign_type(d_types[key], rank=1)

                else:
                    raise NotImplementedError('')

            else:
                print('--------')
                print(key)
                print(d_types)
                print('--------')
                print('> Unable to compute type for {} '.format(expr))

        untyped = [i for i in arguments if not(_get_key(i) in d_types.keys())]
#        print('> untyped = ', untyped)
        if not untyped:
            # TODO increment rank
            return main_expr.xreplace({expr: d_types[key]})

        else:
            return main_expr

    else:
        raise TypeError('Not implemented for {}'.format(type(expr)))

#==============================================================================
def compute_shapes(expr, namespace):
    pass
