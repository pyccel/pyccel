#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import sys
from itertools import chain
from collections import namedtuple

import pyccel.decorators as pyccel_decorators
from pyccel.errors.errors import Errors, PyccelError

from .core          import (AsName, Import, FunctionDef, FunctionCall,
                            Allocate, Duplicate, Assign, For, CodeBlock,
                            Concatenate, Module, PyccelFunctionDef)

from .builtins      import (builtin_functions_dict,
                            PythonRange, PythonList, PythonTuple)
from .internals     import PyccelInternalFunction, Slice
from .itertoolsext  import itertools_mod
from .literals      import LiteralInteger, Nil
from .mathext       import math_mod
from .sysext        import sys_mod

from .numpyext      import (NumpyEmpty, NumpyArray, numpy_mod,
                            NumpyTranspose, NumpyLinspace)
from .operators     import PyccelAdd, PyccelMul, PyccelIs, PyccelArithmeticOperator
from .scipyext      import scipy_mod
from .variable      import (Variable, IndexedElement, InhomogeneousTupleVariable,
                            HomogeneousTupleVariable )

from .c_concepts import ObjectAddress

errors = Errors()

__all__ = (
    'LoopCollection',
    'builtin_function',
    'builtin_import',
    'builtin_import_registry',
    'split_positional_keyword_arguments',
)

#==============================================================================
def builtin_function(expr, args=None):
    """Returns a builtin-function call applied to given arguments."""

    if isinstance(expr, FunctionCall):
        name = str(expr.funcdef)
    elif isinstance(expr, str):
        name = expr
    else:
        raise TypeError('expr must be of type str or FunctionCall')

    dic = builtin_functions_dict

    # Unpack FunctionCallArguments
    args = [a.value for a in args]

    if name in dic.keys() :
        try:
            return dic[name](*args)
        except PyccelError as e:
            errors.report(e,
                    symbol=expr,
                    severity='fatal')

    return None

#==============================================================================
decorators_mod = Module('decorators',(),
        funcs = [PyccelFunctionDef(d, PyccelInternalFunction) for d in pyccel_decorators.__all__])
pyccel_mod = Module('pyccel',(),(),
        imports = [Import('decorators', decorators_mod)])

# TODO add documentation
builtin_import_registry = Module('__main__',
        (),(),
        imports = [
            Import('numpy', numpy_mod),
            Import('scipy', scipy_mod),
            Import('itertools', itertools_mod),
            Import('math', math_mod),
            Import('pyccel', pyccel_mod),
            Import('sys', sys_mod),
            ])
if sys.version_info < (3, 10):
    from .builtin_imports import python_builtin_libs
else:
    python_builtin_libs = set(sys.stdlib_module_names) # pylint: disable=no-member

recognised_libs = python_builtin_libs | builtin_import_registry.keys()

def recognised_source(source_name):
    """
    Determine whether the imported source is recognised by pyccel.

    Determine whether the imported source is recognised by pyccel.
    If it is not recognised then it will need to be imported and translated.

    Parameters
    ----------
    source_name : str
        The name of the imported module.

    Returns
    -------
    bool
        True if the source is recognised, False otherwise.
    """
    source = str(source_name).split('.')
    if source[0] in python_builtin_libs and source[0] not in builtin_import_registry.keys():
        return True
    else:
        return source_name in builtin_import_registry

#==============================================================================
def collect_relevant_imports(module, targets):
    """
    Extract all objects necessary to create imports from a module given a list of targets

    Parameters
    ----------
    module  : Module
              The module from which we want to collect the targets
    targets : list of str/AsName
              The names of the objects which we would like to import from the module

    Results
    -------
    imports : list of tuples
              A list where each element is a tuple containing the name which
              will be used to refer to the object in the code, and the object
    """
    imports = []
    for target in targets:
        if isinstance(target, AsName):
            import_name = target.name
            code_name = target.target
        else:
            import_name = target
            code_name = import_name

        if import_name in module.keys():
            imports.append((code_name, module[import_name]))
    return imports

def builtin_import(expr):
    """
    Return a Pyccel-extension function/object from an import of a recognised module.

    Examine an Import object which imports something which is recognised by
    Pyccel internally. The object(s) imported are then returned for use in the
    code.

    Parameters
    ----------
    expr : Import
        The expression which imports the module.

    Returns
    -------
    list
        A list of 2-tuples. The first element is the name of the imported object,
        the second element is the object itself.
    """

    if not isinstance(expr, Import):
        raise TypeError('Expecting an Import expression')

    if isinstance(expr.source, AsName):
        source = expr.source.name
    else:
        source = str(expr.source)

    if source in builtin_import_registry:
        if expr.target:
            return collect_relevant_imports(builtin_import_registry[source], expr.target)
        elif isinstance(expr.source, AsName):
            return [(expr.source.target, builtin_import_registry[source])]
        else:
            return [(expr.source, builtin_import_registry[source])]

    return []

#==============================================================================
def get_function_from_ast(ast, func_name):
    node = None
    for stmt in ast:
        if isinstance(stmt, FunctionDef) and str(stmt.name) == func_name:
            node = stmt
            break

    if node is None:
        print(f'> could not find {func_name}')

    return node

#==============================================================================
def split_positional_keyword_arguments(*args):
    """ Create a list of positional arguments and a dictionary of keyword arguments
    """

    # Distinguish between positional and keyword arguments
    val_args = ()
    for i, a in enumerate(args):
        if a.has_keyword:
            args, val_args = args[:i], args[i:]
            break

    # Collect values from args
    args = [a.value for a in args]
    # Convert list of keyword arguments into dictionary
    kwargs = {a.keyword: a.value for a in val_args}

    return args, kwargs

#==============================================================================
def compatible_operation(*args, language_has_vectors = True):
    """
    Indicates whether an operation requires an index to be
    correctly understood

    Parameters
    ==========
    args      : list of PyccelAstNode
                The operator arguments
    language_has_vectors : bool
                Indicates if the language has support for vector
                operations of the same shape
    Results
    =======
    compatible : bool
                 A boolean indicating if the operation is compatible
    """
    if language_has_vectors:
        # If the shapes don't match then an index must be required
        shapes = [a.shape[::-1] if a.order == 'F' else a.shape for a in args if a.rank != 0]
        shapes = set(tuple(d if d == LiteralInteger(1) else -1 for d in s) for s in shapes)
        order  = set(a.order for a in args if a.order is not None)
        return len(shapes) <= 1 and len(order) <= 1
    else:
        return all(a.rank == 0 for a in args)

#==============================================================================
def insert_index(expr, pos, index_var):
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
    >>> i = Variable('int', 'i')
    >>> d = PyccelAdd(a,b)
    >>> expr = Assign(c,d)
    >>> insert_index(expr, 0, i, language_has_vectors = False)
    IndexedElement(c, i) := IndexedElement(a, i) + IndexedElement(b, i)
    >>> insert_index(expr, 0, i, language_has_vectors = True)
    c := a + b
    """
    if expr.rank==0:
        return expr
    elif isinstance(expr, (Variable, ObjectAddress)):
        if expr.rank==0 or -pos>expr.rank:
            return expr
        if expr.shape[pos]==1:
            # If there is no dimension in this axis, reduce the rank
            index_var = LiteralInteger(0)

        # Add index at the required position
        indexes = [Slice(None,None)]*(expr.rank+pos) + [index_var]+[Slice(None,None)]*(-1-pos)
        return IndexedElement(expr, *indexes)

    elif isinstance(expr, NumpyTranspose):
        if expr.rank==0 or -pos>expr.rank:
            return expr
        if expr.shape[pos]==1:
            # If there is no dimension in this axis, reduce the rank
            index_var = LiteralInteger(0)

        # Add index at the required position
        if expr.rank<2:
            return insert_index(expr.internal_var, expr.rank-1+pos, index_var)
        else:
            return NumpyTranspose(insert_index(expr.internal_var, expr.rank-1+pos, index_var))

    elif isinstance(expr, IndexedElement):
        base = expr.base
        indices = list(expr.indices)
        i = -1
        while i>=pos and -i<=expr.base.rank:
            if not isinstance(indices[i], Slice):
                pos -= 1
            i -= 1
        if -pos>expr.base.rank:
            return expr

        # Add index at the required position
        if expr.base.shape[pos]==1:
            # If there is no dimension in this axis, reduce the rank
            assert(indices[pos].start is None)
            index_var = LiteralInteger(0)

        else:
            # Calculate new index to preserve slice behaviour
            if indices[pos].step is not None:
                index_var = PyccelMul(index_var, indices[pos].step, simplify=True)
            if indices[pos].start is not None:
                index_var = PyccelAdd(index_var, indices[pos].start, simplify=True)

        indices[pos] = index_var
        return IndexedElement(base, *indices)

    elif isinstance(expr, PyccelArithmeticOperator):
        return type(expr)(insert_index(expr.args[0], pos, index_var),
                          insert_index(expr.args[1], pos, index_var))

    elif hasattr(expr, '__getitem__'):
        return expr[index_var]

    else:
        raise NotImplementedError(f"Expansion not implemented for type : {type(expr)}")

#==============================================================================

LoopCollection = namedtuple('LoopCollection', ['body', 'length', 'modified_vars'])

#==============================================================================
def collect_loops(block, indices, new_index, language_has_vectors = False, result = None):
    """
    Collect blocks of code into loops.

    Run through a code block and split it into lists of tuples of lists where
    each inner list represents a code block and the tuples contain the lists
    and the size of the code block.
    So the following:
    `a = a+b`
    for a: int[:,:] and b: int[:]
    Would be returned as:
    ```
    [
      ([
        ([a[i,j]=a[i,j]+b[j]],a.shape[1])
       ]
       , a.shape[0]
      )
    ]
    ```

    Parameters
    ----------
    block : list of Ast Nodes
        The expressions to be modified.
    indices : list
        An empty list to be filled with the temporary variables created.
    new_index : function (class method of a Scope)
        A function which provides a new variable from a base name,
        avoiding name collisions.
    language_has_vectors : bool
        Indicates if the language has support for vector
        operations of the same shape.
    result : list, default: None
        The list which will be returned. If none is provided, a new list
        is created.

    Returns
    -------
    list of tuples of lists
        The modified expression.
    """
    if result is None:
        result = []
    current_level = 0
    array_creator_types = (Allocate, PythonList, PythonTuple, Concatenate, Duplicate)
    is_function_call = lambda f: ((isinstance(f, FunctionCall) and not f.funcdef.is_elemental)
                                or (isinstance(f, PyccelInternalFunction) and not f.is_elemental and not hasattr(f, '__getitem__')
                                    and not isinstance(f, (NumpyTranspose))))
    for line in block:

        if (isinstance(line, Assign) and
                not isinstance(line.rhs, (array_creator_types, Nil)) and # not creating array
                not line.rhs.get_attribute_nodes(array_creator_types) and # not creating array
                not is_function_call(line.rhs)): # not a basic function call

            # Collect lhs variable
            # This is needed to know what has already been modified in the loop
            if isinstance(line.lhs, Variable):
                lhs_vars = [line.lhs]
            elif isinstance(line.lhs, IndexedElement):
                lhs_vars = [line.lhs.base]
            else:
                lhs_vars = set(line.lhs.get_attribute_nodes((Variable, IndexedElement)))
                lhs_vars = [v.base if isinstance(v, IndexedElement) else v for v in lhs_vars]

            # Get all objects which affect where indices are inserted
            notable_nodes = line.get_attribute_nodes((Variable,
                                                       IndexedElement,
                                                       ObjectAddress,
                                                       NumpyTranspose,
                                                       FunctionCall,
                                                       PyccelInternalFunction,
                                                       PyccelIs))

            # Find all elemental function calls. Normally function call arguments are not indexed
            # However elemental functions are an exception
            elemental_func_calls  = [f for f in notable_nodes if (isinstance(f, FunctionCall) \
                                                                and f.funcdef.is_elemental)]
            elemental_func_calls += [f for f in notable_nodes if (isinstance(f, PyccelInternalFunction) \
                                                                and f.is_elemental)]

            # Collect all objects into which indices may be inserted
            variables       = [v for v in notable_nodes if isinstance(v, (Variable,
                                                                          IndexedElement,
                                                                          ObjectAddress))]
            variables      += [v for f in elemental_func_calls \
                                 for v in f.get_attribute_nodes((Variable, IndexedElement, ObjectAddress))]
            transposed_vars = [v for v in notable_nodes if isinstance(v, NumpyTranspose)] \
                                + [v for f in elemental_func_calls \
                                     for v in f.get_attribute_nodes(NumpyTranspose)]
            indexed_funcs = [v for v in notable_nodes if isinstance(v, PyccelInternalFunction) and hasattr(v, '__getitem__')]

            is_checks = [n for n in notable_nodes if isinstance(n, PyccelIs)]

            variables = list(set(variables))

            # Check if the expression is already satisfactory
            if compatible_operation(*variables, *transposed_vars, *is_checks,
                                    language_has_vectors = language_has_vectors):
                result.append(line)
                current_level = 0
                continue

            # Find function calls in this line
            funcs           = [f for f in notable_nodes+transposed_vars if (isinstance(f, FunctionCall) \
                                                            and not f.funcdef.is_elemental)]
            internal_funcs  = [f for f in notable_nodes+transposed_vars if (isinstance(f, PyccelInternalFunction) \
                                                            and not f.is_elemental and not hasattr(f, '__getitem__')) \
                                                            and not isinstance(f, NumpyTranspose)]

            # Collect all variables for which values other than the value indexed in the loop are important
            # E.g. x = np.sum(a) has a dependence on a
            dependencies = set(v for f in chain(funcs, internal_funcs) \
                                 for v in f.get_attribute_nodes((Variable, IndexedElement, ObjectAddress)))

            # Replace function calls with temporary variables
            # This ensures that the function is only called once and stops problems
            # for expressions such as:
            # c += b*np.sum(c)
            func_vars1 = [new_index(f.dtype, 'tmp') for f in internal_funcs]
            _          = [v.copy_attributes(f) for v,f in zip(func_vars1, internal_funcs)]
            assigns    = [Assign(v, f) for v,f in zip(func_vars1, internal_funcs)]


            if any(len(f.funcdef.results)!=1 for f in funcs):
                errors.report("Loop unravelling cannot handle function calls "\
                        "which return tuples or None",
                        symbol=line, severity='fatal')

            func_results = [f.funcdef.results[0].var for f in funcs]
            func_vars2 = [new_index(r.dtype, r.name) for r in func_results]
            assigns   += [Assign(v, f) for v,f in zip(func_vars2, funcs)]

            if assigns:
                # For now we do not handle memory allocation in loop unravelling
                if any(v.rank > 0 for v in func_vars1) or any(v.rank > 0 for v in func_results):
                    errors.report("Loop unravelling cannot handle extraction of function calls "\
                            "which return arrays as this requires allocation. Please place the function "\
                            "call on its own line",
                            symbol=line, severity='fatal')
                line.substitute(internal_funcs, func_vars1, excluded_nodes=(FunctionCall))
                line.substitute(funcs, func_vars2)
                result.extend(assigns)
                current_level = 0

            rank = line.lhs.rank
            shape = line.lhs.shape
            new_vars = variables
            handled_funcs = transposed_vars + indexed_funcs
            # Loop over indexes, inserting until the expression can be evaluated
            # in the desired language
            new_level = 0
            for index in range(-rank,0):
                new_level += 1
                # If an index exists at the same depth, reuse it if not create one
                if rank+index >= len(indices):
                    indices.append(new_index('int','i'))
                index_var = indices[rank+index]
                new_vars = [insert_index(v, index, index_var) for v in new_vars]
                handled_funcs = [insert_index(v, index, index_var) for v in handled_funcs]
                if compatible_operation(*new_vars, *handled_funcs, language_has_vectors = language_has_vectors):
                    break

            # TODO [NH]: get all indices when adding axis argument to linspace function
            if isinstance(line.rhs, NumpyLinspace):
                line.rhs.ind = indices[0]

            # Replace variable expressions with Indexed versions
            line.substitute(variables, new_vars,
                    excluded_nodes = (FunctionCall, PyccelInternalFunction))
            line.substitute(transposed_vars + indexed_funcs, handled_funcs,
                    excluded_nodes = (FunctionCall))
            _ = [f.substitute(variables, new_vars) for f in elemental_func_calls]
            _ = [f.substitute(transposed_vars + indexed_funcs, handled_funcs) for f in elemental_func_calls]

            # Recurse through result tree to save line with lines which need
            # the same set of for loops
            save_spot = result
            j = 0
            for _ in range(min(new_level,current_level)):
                # Select the existing loop if the shape matches the shape of the expression
                # and the loop is not used to modify one of the variable dependencies
                if save_spot[-1].length == shape[j] and not any(u in save_spot[-1].modified_vars for u in dependencies):
                    save_spot[-1].modified_vars.update(lhs_vars)
                    save_spot = save_spot[-1].body
                    j+=1
                else:
                    break

            for k in range(j,new_level):
                # Create new loops until we have the neccesary depth
                save_spot.append(LoopCollection([], shape[k], set(lhs_vars)))
                save_spot = save_spot[-1].body

            # Save results
            save_spot.append(line)
            current_level = new_level

        elif isinstance(line, Assign) and isinstance(line.lhs, IndexedElement) \
                and isinstance(line.rhs, (PythonTuple, NumpyArray)) and not language_has_vectors:

            lhs = line.lhs
            rhs = line.rhs
            if isinstance(rhs, NumpyArray):
                rhs = rhs.arg

            lhs_rank = lhs.rank

            new_assigns = [Assign(
                            insert_index(expr=lhs,
                                pos       = -lhs_rank,
                                index_var = LiteralInteger(j)),
                            rj) # lhs[j] = rhs[j]
                          for j, rj in enumerate(rhs)]
            collect_loops(new_assigns, indices, new_index, language_has_vectors, result = result)

        elif isinstance(line, Assign) and isinstance(line.rhs, Concatenate):
            lhs = line.lhs
            rhs = line.rhs
            arg1, arg2 = rhs.args
            assign1 = Assign(lhs[Slice(LiteralInteger(0), arg1.shape[0])], arg1)
            assign2 = Assign(lhs[Slice(arg1.shape[0], PyccelAdd(arg1.shape[0], arg2.shape[0], simplify=True))], arg2)
            collect_loops([assign1, assign2], indices, new_index, language_has_vectors, result = result)

        elif isinstance(line, Assign) and isinstance(line.rhs, Duplicate):
            lhs = line.lhs
            rhs = line.rhs

            if not isinstance(rhs.length, LiteralInteger):
                if len(indices) == 0:
                    indices.append(new_index('int', 'i'))
                idx = indices[0]

                assign = Assign(lhs[Slice(PyccelMul(rhs.val.shape[0], idx, simplify=True),
                                          PyccelMul(rhs.val.shape[0],
                                                    PyccelAdd(idx, LiteralInteger(1), simplify=True),
                                                    simplify=True))],
                                rhs.val)

                tmp_indices = indices[1:]

                block = collect_loops([assign], tmp_indices, new_index, language_has_vectors)
                if len(tmp_indices)>len(indices)-1:
                    indices.extend(tmp_indices[len(indices)-1:])

                result.append(LoopCollection(block, rhs.length, set([lhs])))

            else:
                assigns = [Assign(lhs[Slice(PyccelMul(rhs.val.shape[0], LiteralInteger(idx), simplify=True),
                                          PyccelMul(rhs.val.shape[0],
                                              PyccelAdd(LiteralInteger(idx), LiteralInteger(1), simplify=True),
                                              simplify=True))],
                                rhs.val) for idx in range(rhs.length)]
                collect_loops(assigns, indices, new_index, language_has_vectors, result = result)

        else:
            # Save line in top level (no for loop)
            result.append(line)
            current_level = 0

    return result

#==============================================================================

def insert_fors(blocks, indices, scope, level = 0):
    """
    Run through the output of collect_loops and create For loops of the
    requested sizes

    Parameters
    ==========
    block   : list of LoopCollection
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
    if all(not isinstance(b, LoopCollection) for b in blocks.body):
        body = blocks.body
    else:
        loop_scope = scope.create_new_loop_scope()
        body = [insert_fors(b, indices, loop_scope, level+1) if isinstance(b, LoopCollection) else [b] \
                for b in blocks.body]
        body = [bi for b in body for bi in b]

    if blocks.length == 1:
        return body
    else:
        body = CodeBlock(body, unravelled = True)
        loop_scope = scope.create_new_loop_scope()
        return [For(indices[level],
                    PythonRange(0,blocks.length),
                    body,
                    scope = loop_scope)]

#==============================================================================
def expand_inhomog_tuple_assignments(block, language_has_vectors = False):
    """
    Simplify expressions in a CodeBlock by unravelling tuple assignments into multiple lines

    Parameters
    ==========
    block      : CodeBlock
                The expression to be modified

    Examples
    --------
    >>> from pyccel.ast.builtins  import PythonTuple
    >>> from pyccel.ast.core      import Assign, CodeBlock
    >>> from pyccel.ast.literals  import LiteralInteger
    >>> from pyccel.ast.utilities import expand_to_loops
    >>> from pyccel.ast.variable  import Variable
    >>> a = Variable('int', 'a', shape=(,), rank=0)
    >>> b = Variable('int', 'b', shape=(,), rank=0)
    >>> c = Variable('int', 'c', shape=(,), rank=0)
    >>> expr = [Assign(PythonTuple(a,b,c),PythonTuple(LiteralInteger(0),LiteralInteger(1),LiteralInteger(2))]
    >>> expand_inhomog_tuple_assignments(CodeBlock(expr))
    [Assign(a, LiteralInteger(0)), Assign(b, LiteralInteger(1)), Assign(c, LiteralInteger(2))]
    """
    if not language_has_vectors:
        allocs_to_unravel = [a for a in block.get_attribute_nodes(Assign) \
                    if isinstance(a.lhs, HomogeneousTupleVariable) \
                    and isinstance(a.rhs, (HomogeneousTupleVariable, Duplicate, Concatenate))]
        new_allocs = [(Assign(a.lhs, NumpyEmpty(a.lhs.shape,
                                     dtype=a.lhs.dtype,
                                     order=a.lhs.order)
                    ), a) if a.lhs.on_stack
                    else (a) if a.lhs.on_heap
                    else (Allocate(a.lhs,
                            shape=a.lhs.shape,
                            order = a.lhs.order,
                            status="unknown"), a)
                    for a in allocs_to_unravel]
        block.substitute(allocs_to_unravel, new_allocs)

    assigns = [a for a in block.get_attribute_nodes(Assign) \
                if isinstance(a.lhs, InhomogeneousTupleVariable) \
                and isinstance(a.rhs, (PythonTuple, InhomogeneousTupleVariable))]
    if len(assigns) != 0:
        new_assigns = [[Assign(l,r) for l,r in zip(a.lhs, a.rhs)] for a in assigns]
        block.substitute(assigns, new_assigns)
        expand_inhomog_tuple_assignments(block)

#==============================================================================
def expand_to_loops(block, new_index, scope, language_has_vectors = False):
    """
    Re-write a list of expressions to include explicit loops where necessary

    Parameters
    ==========
    block          : CodeBlock
                     The expressions to be modified
    new_index      : function
                     A function which provides a new variable from a base name,
                     avoiding name collisions
    language_has_vectors : bool
                     Indicates if the language has support for vector
                     operations of the same shape

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
    >>> i = Variable('int', 'i')
    >>> d = PyccelAdd(a,b)
    >>> expr = [Assign(c,d)]
    >>> expand_to_loops(expr, language_has_vectors = False)
    [For(i_0, PythonRange(0, LiteralInteger(4), LiteralInteger(1)), CodeBlock([IndexedElement(c, i_0) := PyccelAdd(IndexedElement(a, i_0), IndexedElement(b, i_0))]), [])]
    """
    expand_inhomog_tuple_assignments(block)

    indices = []
    res = collect_loops(block.body, indices, new_index, language_has_vectors)

    body = [insert_fors(b, indices, scope) if isinstance(b, tuple) else [b] for b in res]
    body = [bi for b in body for bi in b]

    return body
