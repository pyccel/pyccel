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

from .basic    import PyccelAstNode
from .core     import (AsName, Import, FunctionDef, Constant,
                       Variable, IndexedVariable, ValuedVariable,
                       Assign, FunctionCall, IndexedElement,
                       Slice, For, AugAssign, IfTernaryOperator,
                       Nil, Dlist)

from .builtins      import (builtin_functions_dict, PythonMap,
                            PythonRange, PythonList, PythonTuple)
from .itertoolsext  import Product
from .mathext       import math_functions, math_constants, MathFunctionBase
from .literals      import LiteralString, LiteralInteger, Literal

from .numpyext      import (numpy_functions, numpy_linalg_functions,
                            numpy_random_functions, numpy_constants,
                            NumpyNewArray, NumpyFunctionBase)
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
def reduce_slice_to_index(expr, pos):
    """
    Function to insert a 0 index into an expression at a given
    position if the size is 1 in that dimension

    Parameters
    ==========
    expr        : Ast Node
                The expression to be modified
    pos         : int
                The index at which the expression is modified
                (If negative then there is no index to insert)

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
    if isinstance(expr, Variable):
        if expr.rank==0 or -pos>expr.rank:
            return expr
        if expr.shape[pos]==1:
            indexes = [Slice(None,None)]*(expr.rank+pos) + [LiteralInteger(0)]+[Slice(None,None)]*(-1-pos)
            var = IndexedVariable(expr)
            return var[indexes]
    elif isinstance(expr, IndexedElement):
        base = expr.base
        indices = list(expr.indices)
        if -pos>expr.base.rank or not isinstance(indices[pos], Slice):
            return expr
        if expr.base.shape[pos]==1:
            assert(indices[pos].start is None)
            indices[pos] = LiteralInteger(0)
            return base[indices]
        return expr
    elif isinstance(expr, PyccelAstNode) and expr.rank==0:
        return expr
    elif isinstance(expr, AugAssign):
        cls = type(expr)
        rhs = reduce_slice_to_index(expr.rhs, pos)
        lhs = reduce_slice_to_index(expr.lhs, pos)
        return cls(lhs, expr.op, rhs, expr.status, expr.like)
    elif isinstance(expr, Assign):
        cls = type(expr)
        rhs = reduce_slice_to_index(expr.rhs, pos)
        lhs = reduce_slice_to_index(expr.lhs, pos)

        return cls(lhs, rhs, expr.status, expr.like)
    elif isinstance(expr, PyccelOperator):
        cls = type(expr)
        args = [reduce_slice_to_index(a, pos) for a in expr.args]
        return cls(*args)
    elif isinstance(expr, IfTernaryOperator):
        cond        = reduce_slice_to_index(expr.cond       , pos)
        value_true  = reduce_slice_to_index(expr.value_true , pos)
        value_false = reduce_slice_to_index(expr.value_false, pos)
        return IfTernaryOperator(cond, value_true, value_false)
    return expr

#==============================================================================
def compatible_operation(*args):
    """
    Indicates whether an operation requires an index to be
    correctly understood

    Parameters
    ==========
    args : list of PyccelAstNode
           The operator arguments
    Results
    =======
    compatible : bool
                 A boolean indicating if the operation is compatible
    """
    # If the shapes don't match then an index must be required
    shapes = [a.shape for a in args if a.shape != ()]
    shapes = set(tuple(d if isinstance(d, Literal) else -1 for d in s) for s in shapes)
    order  = set(a.order for a in args if a.order is not None)
    return len(shapes) == 1 and len(order) == 1

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
    if isinstance(expr, Variable):
        if expr.rank==0 or -pos>expr.rank:
            return expr
        if expr.shape[pos]==1:
            index_var = LiteralInteger(0)
        var = IndexedVariable(expr)
        indexes = [Slice(None,None)]*(expr.rank+pos) + [index_var]+[Slice(None,None)]*(-1-pos)
        return var[indexes]

    elif isinstance(expr, IndexedElement):
        base = expr.base
        indices = list(expr.indices)
        if -pos>expr.base.rank or not isinstance(indices[pos], Slice):
            return expr
        if expr.base.shape[pos]==1:
            assert(indices[pos].start is None)
            index_var = LiteralInteger(0)
        else:
            if indices[pos].step is not None:
                index_var = PyccelMul(index_var, indices[pos].step)
            if indices[pos].start is not None:
                index_var = PyccelAdd(index_var, indices[pos].start)
        indices[pos] = index_var
        return base[indices]

    elif isinstance(expr, PyccelAstNode) and expr.rank==0:
        return expr

    elif isinstance(expr, AugAssign):
        cls = type(expr)

        compatible = compatible_operation(expr.lhs, expr.rhs)

        lhs = insert_index(expr.lhs, pos, index_var, language_has_vectors)
        rhs = insert_index(expr.rhs, pos, index_var, language_has_vectors)

        # Indicates whether the rhs needs an index inserting
        changed = not isinstance(rhs, (Variable, IndexedElement)) and rhs is not rhs

        if changed or not compatible or not language_has_vectors:
            return cls(lhs, expr.op, rhs, expr.status, expr.like)
        else:
            return expr

    elif isinstance(expr, Assign):
        cls = type(expr)
        lhs = insert_index(expr.lhs, pos, index_var, language_has_vectors)
        rhs = insert_index(expr.rhs, pos, index_var, language_has_vectors)

        compatible = compatible_operation(expr.lhs, expr.rhs)

        # a = b does not need indexing inserting if a and b have the same shape
        var_assign = isinstance(rhs, (Variable, IndexedElement)) and compatible

        if  rhs is not expr.rhs and (not var_assign or not language_has_vectors):
            return cls(lhs, rhs, expr.status, expr.like)
        else:
            return expr

    elif isinstance(expr, PyccelOperator):
        cls = type(expr)

        args = [insert_index(a, pos, index_var, False) for a in expr.args]

        compatible = compatible_operation(*expr.args)
        changed = any(a is not na for a,na in zip(expr.args, args) if not isinstance(a, (Variable, IndexedElement)))

        if changed or not compatible or not language_has_vectors:
            return cls(*args)
        else:
            return expr

    elif isinstance(expr, IfTernaryOperator):
        cond        = insert_index(expr.cond       , pos, index_var, language_has_vectors)
        value_true  = insert_index(expr.value_true , pos, index_var, language_has_vectors)
        value_false = insert_index(expr.value_false, pos, index_var, language_has_vectors)

        changed = not ( cond is expr.cond and
                        value_true is expr.value_true and
                        value_false is expr.value_false )

        if changed or not language_has_vectors:
            return IfTernaryOperator(cond, value_true, value_false)
        else:
            return expr

    else:
        raise NotImplementedError("Expansion not implemented for type : {}".format(type(expr)))

#==============================================================================
def collect_loops(block, indices, language_has_vectors = False):
    """
    Run through a code block and split it into lists of tuples of lists where
    each inner list represents a code block and the tuples contain the lists
    and the size of the code block.
    So the following:
    a = a+b
    for a: int[:,:] and b: int[:]
    Would be returned as:
    [
      ([
        ([a[i,j]=a[i,j]+b[j]],a.shape[1])
       ]
       , a.shape[0]
      )
    ]

    Parameters
    ==========
    block                   : list of Ast Nodes
                            The expressions to be modified
    indices                 : list
                            An empty list to be filled with the temporary variables created
    language_has_vectors    : bool
                            Indicates if the language has support for vector
                            operations of the same shape
    Results
    =======
    block : list of tuples of lists
            The modified expression
    """
    result = []
    current_level = 0
    array_creator_types = (NumpyNewArray, FunctionCall,
                           NumpyFunctionBase, MathFunctionBase,
                           PythonList, PythonTuple, Nil, Dlist)
    for i, line in enumerate(block):
        if isinstance(line, Assign) and \
                not isinstance(line.rhs, array_creator_types) and \
                not ( not isinstance(line, AugAssign) and isinstance(line.rhs, Variable)) and \
                not ( isinstance(line.rhs, IfTernaryOperator) and \
                (isinstance(line.rhs.value_true, array_creator_types) or \
                isinstance(line.rhs.value_false, array_creator_types)) ):
            lhs = line.lhs

            # Loop over indexes, inserting until the expression can be evaluated
            # in the desired language
            new_level = 0
            for kk, index in enumerate(range(-lhs.rank,0)):
                if lhs.rank+index >= len(indices):
                    indices.append(Variable('int','i_{}'.format(kk)))
                index_var = indices[lhs.rank+index]
                line = reduce_slice_to_index(line, index)
                new_stmt = insert_index(line, index, index_var, language_has_vectors)
                if new_stmt is line:
                    break
                else:
                    new_level += 1
                    line = new_stmt

            # Remve all slices of size 1 in remaining dimensions
            index = new_level
            while line.lhs.rank != line.rhs.rank and index < lhs.rank:
                line = reduce_slice_to_index(line, index)
                index += 1

            # Recurse through result tree to save line with lines which need
            # the same set of for loops
            save_spot = result
            j = 0
            shape = lhs.shape
            for _ in range(min(new_level,current_level)):
                # Select the existing loop if the shape matches the
                # shape of the expression
                if save_spot[-1][1] == shape[j]:
                    save_spot = save_spot[-1][0]
                    j+=1
                else:
                    break
            for k in range(j,new_level):
                # Create new loops until we have the neccesary depth
                save_spot.append(([], shape[k]))
                save_spot = save_spot[-1][0]

            # Save results
            save_spot.append(line)
            current_level = new_level
        else:
            # Save line in top level (no for loop)
            result.append(line)
            current_level = 0
    return result

def insert_fors(blocks, indices, level = 0):
    """
    Run through the output of collect_loops and create For loops of the
    requested sizes

    Parameters
    ==========
    block   : list of tuples of lists
            The result of a call to collect_loops
    indices : list
            The index variables
    level   : int
            The index of the index variable used in the outermost loop
    Results
    =======
    block : list of PyccelAstNodes
            The modified expression
    """
    if all(not isinstance(b, tuple) for b in blocks[0]):
        body = blocks[0]
    else:
        body = [insert_fors(b, indices, level+1) if isinstance(b, tuple) else [b] \
                for b in blocks[0]]
        body = [bi for b in body for bi in b]
    if blocks[1] == 1:
        return body
    else:
        return [For(indices[level], PythonRange(0,blocks[1]), body)]

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
    indices = []
    res = collect_loops(block, indices, language_has_vectors)

    body = [insert_fors(b, indices) if isinstance(b, tuple) else [b] for b in res]
    body = [bi for b in body for bi in b]
    return body, indices
