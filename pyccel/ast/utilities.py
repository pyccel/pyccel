#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import inspect

from sympy.core.function import Application
from sympy import Not, Function
from numpy import pi

import pyccel.decorators as pyccel_decorators
from pyccel.symbolic import lambdify
from pyccel.errors.errors import Errors

from .core     import (AsName, Import, FunctionDef, Constant,
                       Variable, IndexedVariable, ValuedVariable,
                       Assign, FunctionCall, IndexedElement,
                       Slice, For, AugAssign)

from .builtins      import (builtin_functions_dict, PythonMap,
                            PythonRange)
from .itertoolsext  import Product
from .mathext       import math_functions, math_constants
from .literals      import LiteralString, LiteralInteger

from .numpyext      import (numpy_functions, numpy_linalg_functions,
                            numpy_random_functions, numpy_constants,
                            NumpyNewArray)
from .operators     import PyccelOperator, PyccelMul, PyccelAdd

__all__ = (
    'build_types_decorator',
    'builtin_function',
    'builtin_import',
    'builtin_import_registery',
    'split_positional_keyword_arguments',
)

scipy_constants = {
    'pi': Constant('real', 'pi', value=pi),
                  }

#==============================================================================
def builtin_function(expr, args=None):
    """Returns a builtin-function call applied to given arguments."""

    if isinstance(expr, Application):
        name = str(type(expr).__name__)
    elif isinstance(expr, str):
        name = expr
    else:
        raise TypeError('expr must be of type str or Function')

    dic = builtin_functions_dict

    if name in dic.keys() :
        return dic[name](*args)

    if name == 'Not':
        return Not(*args)

    if name == 'map':
        func = Function(str(expr.args[0].name))
        args = [func]+list(args[1:])
        return PythonMap(*args)

    if name == 'lambdify':
        return lambdify(expr, args)

    return None


# TODO add documentation
builtin_import_registery = {'numpy': {
                                      **numpy_functions,
                                      **numpy_constants,
                                      'linalg':numpy_linalg_functions,
                                      'random':numpy_random_functions
                                      },
                            'numpy.linalg': numpy_linalg_functions,
                            'numpy.random': numpy_random_functions,
                            'scipy.constants': scipy_constants,
                            'itertools': {'product': Product},
                            'math': {**math_functions, ** math_constants},
                            'pyccel.decorators': None}

#==============================================================================
def collect_relevant_imports(func_dictionary, targets):
    if len(targets) == 0:
        return func_dictionary

    imports = []
    for target in targets:
        if isinstance(target, AsName):
            import_name = target.name
            code_name = target.target
        else:
            import_name = str(target)
            code_name = import_name

        if import_name in func_dictionary.keys():
            imports.append((code_name, func_dictionary[import_name]))
    return imports

def builtin_import(expr):
    """Returns a builtin pyccel-extension function/object from an import."""

    if not isinstance(expr, Import):
        raise TypeError('Expecting an Import expression')

    if isinstance(expr.source, AsName):
        source = str(expr.source.name)
    else:
        source = str(expr.source)

    if source == 'pyccel.decorators':
        funcs = [f[0] for f in inspect.getmembers(pyccel_decorators, inspect.isfunction)]
        for target in expr.target:
            search_target = target.name if isinstance(target, AsName) else str(target)
            if search_target not in funcs:
                errors = Errors()
                errors.report("{} does not exist in pyccel.decorators".format(target),
                        symbol = expr, severity='error')

    elif source in builtin_import_registery:
        return collect_relevant_imports(builtin_import_registery[source], expr.target)

    return []

#==============================================================================
def get_function_from_ast(ast, func_name):
    node = None
    for stmt in ast:
        if isinstance(stmt, FunctionDef) and str(stmt.name) == func_name:
            node = stmt
            break

    if node is None:
        print('> could not find {}'.format(func_name))

    return node

#==============================================================================
# TODO: must add a Node Decorator in core
def build_types_decorator(args, order=None):
    """
    builds a types decorator from a list of arguments (of FunctionDef)
    """
    types = []
    for a in args:
        if isinstance(a, Variable):
            dtype = a.dtype.name.lower()

        elif isinstance(a, IndexedVariable):
            dtype = a.dtype.name.lower()

        else:
            raise TypeError('unepected type for {}'.format(a))

        if a.rank > 0:
            shape = [':' for i in range(0, a.rank)]
            shape = ','.join(i for i in shape)
            dtype = '{dtype}[{shape}]'.format(dtype=dtype, shape=shape)
            if order and a.rank > 1:
                dtype = "{dtype}(order={ordering})".format(dtype=dtype, ordering=order)

        dtype = LiteralString(dtype)
        types.append(dtype)

    return types

#==============================================================================
def split_positional_keyword_arguments(*args):
    """ Create a list of positional arguments and a dictionary of keyword arguments
    """

    # Distinguish between positional and keyword arguments
    val_args = ()
    for i, a in enumerate(args):
        if isinstance(a, ValuedVariable):
            args, val_args = args[:i], args[i:]
            break

    # Convert list of keyword arguments into dictionary
    kwargs = {}
    for v in val_args:
        key   = str(v.name)
        value = v.value
        kwargs[key] = value

    return args, kwargs

#==============================================================================
def insert_index(expr, pos, index_var, language_has_vectors):
    """
    Function to insert an index into an expression at a given position

    Parameters
    ==========
    expr        : Ast Node
                The expression to be modified
    pos         : int
                The index at which the expression is modified
                (If negative then there is no index to insert)
    index_var   : Variable
                The variable which will be used for indexing
    language_has_vectors : bool
                Indicates if the language has support for vector
                operations of the same shape

    Returns
    =======
    expr        : Ast Node
                Either a modified version of expr or expr itself

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.operators import PyccelAdd
    >>> from pyccel.ast.utilities import insert_index
    >>> a = Variable('int', 'a', shape=(4,), rank=1)
    >>> b = Variable('int', 'b', shape=(4,), rank=1)
    >>> c = Variable('int', 'c', shape=(4,), rank=1)
    >>> i = Variable('int', 'i', shape=())
    >>> d = PyccelAdd(a,b)
    >>> expr = Assign(c,d)
    >>> insert_index(expr, 0, i, language_has_vectors = False)
    IndexedElement(c, i) := IndexedElement(a, i) + IndexedElement(b, i)
    >>> insert_index(expr, 0, i, language_has_vectors = True)
    c := a + b
    """
    if pos < 0:
        return expr
    if isinstance(expr, Variable):
        if expr.rank==0:
            return expr
        if expr.shape[pos]==1:
            index_var = LiteralInteger(0)
        pos = expr.rank - pos - 1 # Fortran ordering
        var = IndexedVariable(expr)
        indexes = [Slice(None,None)]*pos + [index_var]+[Slice(None,None)]*(expr.rank-1-pos)
        tmp = var[indexes]
        return var[indexes]
    elif isinstance(expr, IndexedElement):
        pos = expr.base.rank - pos - 1
        base = expr.base
        indices = list(expr.indices)
        shape = expr.base.shape[::-1]
        if expr.base.order == 'c':
            indices = indices[::-1]
        assert(isinstance(indices[pos], Slice))
        if shape[pos]==1:
            assert(indices[pos].start is None)
            index_var = LiteralInteger(0)
        else:
            if indices[pos].step is not None:
                index_var = PyccelMul(index_var, indices[pos].step)
            if indices[pos].start is not None:
                index_var = PyccelAdd(index_var, indices[pos].start)
        indices[pos] = index_var
        return base[indices]
    elif isinstance(expr, AugAssign):
        cls = type(expr)
        lhs = insert_index(expr.lhs, pos, index_var, language_has_vectors)
        rhs = insert_index(expr.rhs, pos, index_var, language_has_vectors)

        if rhs is not expr.rhs or not language_has_vectors:
            return cls(lhs, expr.op, rhs, expr.status, expr.like)
        else:
            return expr
    elif isinstance(expr, Assign):
        cls = type(expr)
        lhs = insert_index(expr.lhs, pos, index_var, language_has_vectors)
        rhs = insert_index(expr.rhs, pos, index_var, language_has_vectors)

        if rhs is not expr.rhs or not language_has_vectors:
            return cls(lhs, rhs, expr.status, expr.like)
        else:
            return expr
    elif isinstance(expr, PyccelOperator):
        cls = type(expr)
        shapes = set([a.base.shape if isinstance(a, IndexedElement) else a.shape for a in expr.args])
        if len(shapes)!=1 or not language_has_vectors:
            args = [insert_index(a, pos - expr.rank + a.rank, index_var, False) for a in expr.args]
            return cls(*args)
        else:
            return expr
    elif isinstance(expr, For):
        body = [insert_index(l,pos,index_var, language_has_vectors) for l in expr.body.body]
        return For(expr.target, expr.iterable, body, expr.local_vars)
    else:
        raise NotImplementedError("Expansion not implemented for type : {}".format(type(expr)))


#==============================================================================
def expand_to_loops(block, language_has_vectors = False, index = 0):
    """
    Re-write a list of expressions to include explicit loops where necessary

    Parameters
    ==========
    block      : list of Ast Nodes
                The expressions to be modified
    language_has_vectors : bool
                Indicates if the language has support for vector
                operations of the same shape
    index       : int
                The index from which the expression is modified

    Returns
    =======
    expr        : list of Ast Nodes
                The expressions with For loops inserted where necessary

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.operators import PyccelAdd
    >>> from pyccel.ast.utilities import expand_to_loops
    >>> a = Variable('int', 'a', shape=(4,), rank=1)
    >>> b = Variable('int', 'b', shape=(4,), rank=1)
    >>> c = Variable('int', 'c', shape=(4,), rank=1)
    >>> i = Variable('int', 'i', shape=())
    >>> d = PyccelAdd(a,b)
    >>> expr = [Assign(c,d)]
    >>> expand_to_loops(expr, language_has_vectors = False)
    [For(i_0, PythonRange(0, LiteralInteger(4), LiteralInteger(1)), CodeBlock([IndexedElement(c, i_0) := PyccelAdd(IndexedElement(a, i_0), IndexedElement(b, i_0))]), [])]
    """
    max_rank = -1
    current_block_length = -1
    started_block = False
    before_loop = []
    loop_stmts  = []
    after_loop  = []
    for i, line in enumerate(block):
        if isinstance(line, Assign) and \
                not isinstance(line.rhs, (NumpyNewArray, FunctionCall,)):
            lhs = line.lhs
            rhs = line.rhs
            if lhs.rank == max_rank and lhs.shape[index] == current_block_length:
                loop_stmts.append(line)
                continue
            elif max_rank==-1 and lhs.rank > 0 and index < lhs.rank:
                current_block_length = lhs.shape[index]
                max_rank = lhs.rank
                started_block = True
                loop_stmts.append(line)
                continue
        if started_block:
            after_loop = block[i:]
            break
        else:
            before_loop.append(line)
    if loop_stmts:
        for_loop_body, new_vars = expand_to_loops(loop_stmts, language_has_vectors, index+1)
        after_loop, new_vars2 = expand_to_loops(after_loop, language_has_vectors, index)
        new_vars += new_vars2
        index_var = Variable('int','i_{}'.format(index))

        unpacked_for_loop_body = [insert_index(l,index,index_var, language_has_vectors) for l in for_loop_body]

        if any([u is not p for u,p in zip(unpacked_for_loop_body, for_loop_body)]):
            for_block = [For(index_var, PythonRange(0,current_block_length), unpacked_for_loop_body)]
            new_vars += [index_var]
        else:
            for_block = for_loop_body
        return before_loop + for_block + after_loop, new_vars
    else:

        return block, []
