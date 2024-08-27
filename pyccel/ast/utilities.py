#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

import sys
from itertools import chain
from collections import namedtuple

import pyccel.decorators as pyccel_decorators
from pyccel.errors.errors import Errors, PyccelError

from .builtins      import PythonLen, PythonAbs, PythonRange, PythonList, PythonTuple, PythonSet
from .builtins      import builtin_functions_dict
from .core          import (AsName, Import, FunctionDef, FunctionCall,
                            Allocate, Duplicate, Assign, For, CodeBlock,
                            Concatenate, Module, PyccelFunctionDef)
from .cmathext      import cmath_mod
from .datatypes     import HomogeneousTupleType, PythonNativeInt, PrimitiveIntegerType
from .internals     import PyccelFunction, Slice, PyccelArrayShapeElement
from .itertoolsext  import itertools_mod
from .literals      import LiteralInteger, LiteralEllipsis, Nil
from .mathext       import math_mod
from .numpyext      import (NumpyEmpty, NumpyArray, numpy_mod, NumpyAbs,
                            NumpyTranspose, NumpyLinspace)
from .operators     import PyccelAdd, PyccelMul, PyccelIs, PyccelArithmeticOperator
from .operators     import PyccelUnarySub, PyccelMinus, IfTernaryOperator, PyccelLt
from .operators     import PyccelGt
from .scipyext      import scipy_mod
from .sysext        import sys_mod
from .typingext     import typing_mod
from .variable      import Variable, IndexedElement, InhomogeneousTupleVariable, Constant

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
decorators_mod = Module('decorators',(),
        funcs = [PyccelFunctionDef(d, PyccelFunction) for d in pyccel_decorators.__all__])
pyccel_mod = Module('pyccel',(),(),
        imports = [Import('decorators', decorators_mod)])

# TODO add documentation
builtin_import_registry = Module('__main__',
        (),(),
        imports = [
            Import('numpy', numpy_mod),
            Import('scipy', scipy_mod),
            Import('itertools', itertools_mod),
            Import('cmath', cmath_mod),
            Import('math', math_mod),
            Import('pyccel', pyccel_mod),
            Import('sys', sys_mod),
            Import('typing', typing_mod)
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
    args      : list of TypedAstNode
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
def get_deep_indexed_element(expr, indices):
    """
    Get the scalar element obtained by indexing the expression with all the indices.

    Get the scalar element obtained by indexed the expression with all the provided
    indices. This element is constructed by calling IndexedElement multiple times
    to create a recursive object with one IndexedElement for each container type.
    This function is used by the functions which unravel vector expressions.

    Parameters
    ----------
    expr : TypedAstNode
        The base object being indexed.
    indices : list[TypedAstNode]
        A list of the indices used to obtain the scalar element.

    Returns
    -------
    IndexedElement
        The scalar indexed element.
    """
    assert len(indices) == expr.rank
    result = expr
    while indices:
        depth = result.class_type.container_rank
        result = IndexedElement(result, *indices[:depth])
        indices = indices[depth:]
    return result

#==============================================================================
def insert_index(expr, pos, index_var):
    """
    Function to insert an index into an expression at a given position.

    Function to insert an index into an expression at a given position.

    Parameters
    ----------
    expr : PyccelAstNode
       The expression to be modified.
    pos : int
       The index at which the expression is modified
       (If negative then there is no index to insert).
    index_var : Variable
       The variable which will be used for indexing.

    Returns
    -------
    PyccelAstNode
       The modified version of expr.

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.datatypes import PythonNativeInt
    >>> from pyccel.ast.operators import PyccelAdd
    >>> from pyccel.ast.utilities import insert_index
    >>> a = Variable(PythonNativeInt(), 'a', shape=(4,))
    >>> b = Variable(PythonNativeInt(), 'b', shape=(4,))
    >>> c = Variable(PythonNativeInt(), 'c', shape=(4,))
    >>> i = Variable(PythonNativeInt(), 'i')
    >>> d = PyccelAdd(a,b)
    >>> expr = Assign(c,d)
    >>> insert_index(expr, 0, i)
    IndexedElement(c, i) := IndexedElement(a, i) + IndexedElement(b, i)
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
        return get_deep_indexed_element(expr, indexes)

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
        rank = base.rank

        # If pos indexes base then recurse
        base_container_rank = base.class_type.container_rank
        if -pos < rank-base_container_rank:
            return insert_index(base, pos+base_container_rank, index_var)

        # Ensure current indices are fully defined
        indices = list(expr.indices)
        if len(indices) == 1 and isinstance(indices[0], LiteralEllipsis):
            indices = [Slice(None,None)]*base_container_rank

        if len(indices)<rank:
            indices += [Slice(None,None)]*(rank-base_container_rank)

        # Start from last index in this indexed element
        i = base_container_rank-rank-1
        while i>=pos and -i<=base_container_rank:
            if not isinstance(indices[i], Slice):
                pos -= 1
            i -= 1

        # if no slices were found then the object is already correctly indexed
        if -pos > rank:
            return expr

        # Add index at the required position
        if base.shape[pos]==1:
            # If there is no dimension in this axis, reduce the rank
            assert indices[pos].start is None
            index_var = LiteralInteger(0)

        else:
            # Calculate new index to preserve slice behaviour
            if indices[pos].step is not None:
                index_var = PyccelMul(index_var, indices[pos].step, simplify=True)
            if indices[pos].start is not None:
                index_var = PyccelAdd(index_var, indices[pos].start, simplify=True)

        # Update index
        indices[pos] = index_var

        # Get new indexed object
        return get_deep_indexed_element(base, indices)

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
    array_creator_types = (Allocate, PythonList, PythonTuple, Concatenate, Duplicate, PythonSet)
    is_function_call = lambda f: ((isinstance(f, FunctionCall) and not f.funcdef.is_elemental)
                                or (isinstance(f, PyccelFunction) and not f.is_elemental and not hasattr(f, '__getitem__')
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
                                                       PyccelFunction,
                                                       PyccelIs))

            # Find all elemental function calls. Normally function call arguments are not indexed
            # However elemental functions are an exception
            elemental_func_calls  = [f for f in notable_nodes if (isinstance(f, FunctionCall) \
                                                                and f.funcdef.is_elemental)]
            elemental_func_calls += [f for f in notable_nodes if (isinstance(f, PyccelFunction) \
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
            indexed_funcs = [v for v in notable_nodes if isinstance(v, PyccelFunction) and hasattr(v, '__getitem__')]

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
            internal_funcs  = [f for f in notable_nodes+transposed_vars if (isinstance(f, PyccelFunction) \
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
            for index_depth in range(-rank, 0):
                new_level += 1
                # If an index exists at the same depth, reuse it if not create one
                if rank+index_depth >= len(indices):
                    indices.append(new_index(PythonNativeInt(), 'i'))
                index = indices[rank+index_depth]
                new_vars = [insert_index(v, index_depth, index) for v in new_vars]
                handled_funcs = [insert_index(v, index_depth, index) for v in handled_funcs]
                if compatible_operation(*new_vars, *handled_funcs, language_has_vectors = language_has_vectors):
                    break

            # TODO [NH]: get all indices when adding axis argument to linspace function
            if isinstance(line.rhs, NumpyLinspace):
                line.rhs.ind = indices[0]

            # Replace variable expressions with Indexed versions
            line.substitute(variables, new_vars,
                    excluded_nodes = (FunctionCall, PyccelFunction))
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
                and isinstance(line.rhs, (PythonTuple, NumpyArray)):
            lhs = line.lhs
            rhs = line.rhs
            if lhs.rank > rhs.rank:
                for index_depth in range(lhs.rank-rhs.rank):
                    # If an index exists at the same depth, reuse it if not create one
                    if index_depth >= len(indices):
                        indices.append(new_index(PythonNativeInt(), 'i'))
                    index = indices[index_depth]
                    lhs = insert_index(lhs, index_depth, index)
                collect_loops([Assign(lhs, rhs)], indices, new_index, language_has_vectors, result = result)

            elif not language_has_vectors:
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

            else:
                result.append(line)

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
                    indices.append(new_index(PythonNativeInt(), 'i'))
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
    Create For loops as requested by the output of collect_loops.

    Run through the output of collect_loops and create For loops of the
    requested sizes.

    Parameters
    ----------
    blocks : list of LoopCollection
        The result of a call to collect_loops.
    indices : list
        The index variables.
    scope : Scope
        The scope on which the loop is defined. This is where the scope for
        the new For loop will be created.
    level : int, default=0
        The index of the index variable used in the outermost loop.

    Returns
    -------
    list[TypedAstNode]
        The modified expression.
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
    Simplify expressions in a CodeBlock by unravelling tuple assignments into multiple lines.

    Simplify expressions in a CodeBlock by unravelling tuple assignments into multiple lines.
    These changes are carried out in-place.

    Parameters
    ----------
    block : CodeBlock
        The expression to be modified.

    language_has_vectors : bool, default=False
        Indicates whether the target language has built-in support for vector operations.

    Examples
    --------
    >>> from pyccel.ast.builtins  import PythonTuple
    >>> from pyccel.ast.core      import Assign, CodeBlock
    >>> from pyccel.ast.datatypes import PythonNativeInt
    >>> from pyccel.ast.literals  import LiteralInteger
    >>> from pyccel.ast.utilities import expand_to_loops
    >>> from pyccel.ast.variable  import Variable
    >>> a = Variable(PythonNativeInt(), 'a')
    >>> b = Variable(PythonNativeInt(), 'b')
    >>> c = Variable(PythonNativeInt(), 'c')
    >>> expr = [Assign(PythonTuple(a,b,c),PythonTuple(LiteralInteger(0),LiteralInteger(1),LiteralInteger(2))]
    >>> expand_inhomog_tuple_assignments(CodeBlock(expr))
    [Assign(a, LiteralInteger(0)), Assign(b, LiteralInteger(1)), Assign(c, LiteralInteger(2))]
    """
    if not language_has_vectors:
        allocs_to_unravel = [a for a in block.get_attribute_nodes(Assign) \
                    if isinstance(a.lhs, Variable) \
                    and isinstance(a.lhs.class_type, HomogeneousTupleType) \
                    and isinstance(a.rhs.class_type, HomogeneousTupleType)]
        new_allocs = [(Assign(a.lhs, NumpyEmpty(a.lhs.shape,
                                     dtype=a.lhs.dtype,
                                     order=a.lhs.order)
                    ), a) if getattr(a.lhs, 'on_stack', False)
                    else (a) if getattr(a.lhs, 'on_heap', False)
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
    Re-write a list of expressions to include explicit loops where necessary.

    Re-write a list of expressions to include explicit loops where necessary.
    The provided expression is the Pyccel representation of the user code. It
    is the output of the semantic stage. The result of this function is the
    equivalent code where any vector expressions are unrolled into explicit
    loops. The unrolling is done completely for languages such as C which have
    no support for vector operations and partially for languages such as
    Fortran which have support for vector operations on objects of the same
    shape.

    Parameters
    ----------
    block : CodeBlock
        The expressions to be modified.
    new_index : function
        A function which provides a new variable from a base name, avoiding
        name collisions.
    scope : Scope
        The scope on which the loop is defined. This is where the scope for
        the new For loop will be created.
    language_has_vectors : bool
        Indicates if the language has support for vector operations of the
        same shape.

    Returns
    -------
    list[PyccelAstNode]
        The expressions with `For` loops inserted where necessary.

    Examples
    --------
    >>> from pyccel.ast.core import Variable, Assign
    >>> from pyccel.ast.datatypes import PythonNativeInt
    >>> from pyccel.ast.operators import PyccelAdd
    >>> from pyccel.ast.utilities import expand_to_loops
    >>> a = Variable(PythonNativeInt(), 'a', shape=(4,))
    >>> b = Variable(PythonNativeInt(), 'b', shape=(4,))
    >>> c = Variable(PythonNativeInt(), 'c', shape=(4,))
    >>> i = Variable(PythonNativeInt(), 'i')
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

#==============================================================================
def is_literal_integer(expr):
    """
    Determine whether the expression is a literal integer.

    Determine whether the expression is a literal integer. A literal integer
    can be described by a LiteralInteger, a PyccelUnarySub(LiteralInteger) or
    a Constant.

    Parameters
    ----------
    expr : object
        Any Python object which should be analysed to determine whether it is an integer.

    Returns
    -------
    bool
        True if the object represents a literal integer, false otherwise.
    """
    return isinstance(expr, (int, LiteralInteger)) or \
        (isinstance(expr, PyccelUnarySub) and isinstance(expr.args[0], (int, LiteralInteger))) or \
        (isinstance(expr, Constant) and isinstance(expr.dtype.primitive_type, PrimitiveIntegerType))

#==============================================================================
def get_expression_sign(expr):
    """
    Get an expression indicating the sign of the expression passed as an argument.

    Get an expression indicating the sign of the expression passed as an argument.
    If the expression represents a known literal integer then the value of the integer
    is returned. If the expression is known to be positive then the return value
    is 1. If the expression is known to be negative then the return value is -1.
    If nothing is known about the sign of the expression the return value is None.
    """
    positive_types = (PyccelArrayShapeElement, PythonLen, NumpyAbs, PythonAbs)
    try:
        sign = int(expr)
    except TypeError:
        if isinstance(expr, positive_types):
            sign = 1
        elif isinstance(expr, PyccelUnarySub) and isinstance(expr.args[0], positive_types):
            sign = -1
        else:
            sign = None

    return sign

#==============================================================================
def get_new_slice_with_processed_arguments(_slice, array_size, allow_negative_index):
    """
    Create new slice with information collected from old slice and decorators.

    Create a new slice where the original `start`, `stop`, and `step` have
    been processed using basic simplifications, as well as additional rules
    identified by the function decorators.

    Parameters
    ----------
    _slice : Slice
        Slice needed to collect (start, stop, step).

    array_size : PyccelArrayShapeElement
        Call to function size().

    allow_negative_index : bool
        True when the decorator allow_negative_index is present.

    Returns
    -------
    Slice
        The new slice with processed arguments (start, stop, step).
    """
    start = _slice.start
    stop = _slice.stop
    step = _slice.step

    start_value = get_expression_sign(start)
    stop_value = get_expression_sign(stop)
    step_value = get_expression_sign(step)

    # negative start and end in slice
    if start_value and start_value < 0:
        start = PyccelAdd(array_size, start, simplify = True)
    elif start is not None and allow_negative_index and start_value is None:
        start = IfTernaryOperator(PyccelLt(start, LiteralInteger(0)),
                    PyccelAdd(array_size, start, simplify = True), start)

    if stop_value and stop_value < 0:
        stop = PyccelAdd(array_size, stop, simplify = True)
    elif stop is not None and allow_negative_index and stop_value is None:
        stop = IfTernaryOperator(PyccelLt(stop, LiteralInteger(0)),
                    PyccelAdd(array_size, stop, simplify = True), stop)

    # negative step in slice
    if step_value and step_value < 0:
        stop = PyccelAdd(stop, LiteralInteger(1), simplify = True) if stop is not None else LiteralInteger(0)
        start = start if start is not None else PyccelMinus(array_size, LiteralInteger(1), simplify = True)

    # variable step in slice
    elif step and allow_negative_index and step_value is None:
        if start is None:
            start = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                LiteralInteger(0), PyccelMinus(array_size , LiteralInteger(1), simplify = True))

        if stop is None:
            stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                PyccelMinus(array_size, LiteralInteger(1), simplify = True), LiteralInteger(0))
        else:
            stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                stop, PyccelAdd(stop, LiteralInteger(1), simplify = True))

    elif stop is not None:
        stop = PyccelMinus(stop, LiteralInteger(1), simplify = True)

    return Slice(start, stop, step)
