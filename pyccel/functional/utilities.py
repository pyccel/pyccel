# -*- coding: UTF-8 -*-

# TODO use OrderedDict when possible
#      right now namespace used only globals, => needs to look in locals too

import os
import sys
import importlib
import ast
import inspect
import numpy as np
from types import FunctionType

from sympy import Indexed, IndexedBase, Tuple, Lambda
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import sympify
from sympy import Dummy

from pyccel.codegen.utilities import get_source_function
from pyccel.ast.datatypes import dtype_and_precsision_registry as dtype_registry
from pyccel.ast.core import Slice, String
from pyccel.ast.datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool
from pyccel.ast.datatypes import get_default_value
from pyccel.parser import Parser

#==============================================================================
def get_decorators(cls):
    target = cls
    decorators = {}

    def visit_FunctionDef(node):
        decorators[node.name] = []
        for n in node.decorator_list:
            name = ''
            if isinstance(n, ast.Call):
                name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            else:
                name = n.attr if isinstance(n, ast.Attribute) else n.id

            decorators[node.name].append(name)

    node_iter = ast.NodeVisitor()
    node_iter.visit_FunctionDef = visit_FunctionDef
    node_iter.visit(ast.parse(inspect.getsource(target)))
    return decorators

#==============================================================================
def get_pyccel_imports_code():
    code = ''
    code += '\nfrom pyccel.decorators import types'
    code += '\nfrom pyccel.decorators import pure'
    code += '\nfrom pyccel.decorators import external, external_call'
#    code += '\nfrom pyccel.decorators import shapes'
#    code += '\nfrom pyccel.decorators import workplace'
#    code += '\nfrom pyccel.decorators import stack_array'

    # TODO improve
    code += get_numpy_imports_code()

    return code

#==============================================================================
def get_numpy_imports_code():
    code = ''
    code += '\nfrom numpy import zeros'
    code += '\nfrom numpy import float64'

    return code

#==============================================================================
def get_dependencies_code(user_functions):
    code = ''
    for f in user_functions:
        code = '{code}\n\n{new}'.format( code = code,
                                         new  = get_source_function(f._imp_) )

    return code


#==============================================================================
def parse_where_stmt(where_stmt):
    """syntactic parsing of the where statement."""

    L = [l for l in where_stmt.values() if isinstance(l, FunctionType)]
    # we take only one of the lambda function
    # when we get the source code, we will have the whole call to lambdify, with
    # the where statement where all the lambdas are defined
    # then we parse the where statement to get the lambdas
    if len(L) > 0:
        L = L[0]

        code = get_source_function(L)
        pyccel = Parser(code)
        ast = pyccel.parse()
        calls = ast.atoms(AppliedUndef)
        where = [call for call in calls if call.__class__.__name__ == 'where']
        if not( len(where) == 1 ):
            raise ValueError('')

        where = where[0]

        # ...
        d = {}
        for arg in where.args:
            name = arg.name
            value = arg.value
            d[name] = value
        # ...

        return d

    else:
        # there is no lambda
        return where_stmt


#==============================================================================
# TODO move as method of FunctionDef
def get_results_shape(func):
    """returns a dictionary that contains for each result, its shape. When using
    the decorator @shapes, the shape value may be computed"""

    # ...
    arguments       = list(func.arguments)
    arguments_inout = list(func.arguments_inout)
    results         = list(func.results)

    inouts = [x for x,flag in zip(arguments, arguments_inout) if flag]
    # ...

    # ...
    d_args = {}
    for a in arguments:
        d_args[a.name] = a
    # ...

#    print('results = ', results)
#    print('inouts   = ', inouts)

    d_shapes = {}
    if 'shapes' in func.decorators.keys():
        d = func.decorators['shapes']
        for valued in d:
            # ...
            r = [r for r in results + inouts if r.name == valued.name]
            if not r:
                raise ValueError('Could not find {}'.format(r))

            assert(len(r) == 1)
            r = r[0]
            # ...

            # ...
            rhs = valued.value
            if isinstance(rhs, String):
                rhs = rhs.arg.replace("'",'')

            else:
                raise NotImplementedError('')
            # ...

            # ...
            rhs = sympify(rhs, locals=d_args)
            # ...

            # TODO improve
            # this must always be a list of slices
            d_shapes[r.name] = [Slice(None, rhs)]

    # TODO treate the case when shapes is not given => add some checks
#    else:
#        raise NotImplementedError('')

    return d_shapes


#==============================================================================
def _get_default_value(var, op=None):
    """Returns the default value of a variable depending on its datatype and the
    used operation."""
    dtype = var.dtype
    if op is None:
        return get_default_value(dtype)

    if isinstance(dtype, NativeInteger):
        if op == '*':
            return 1

        else:
            return 0

    elif isinstance(dtype, NativeReal):
        if op == '*':
            return 1.0

        else:
            return 0.0

    elif isinstance(dtype, NativeComplex):
        # TODO check if this fine with pyccel
        if op == '*':
            return 1.0

        else:
            return 0.0

#    elif isinstance(dtype, NativeBool):

    raise NotImplementedError('TODO')
