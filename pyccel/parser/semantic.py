# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" File containing SemanticParser. This class handles the semantic stage of the translation.
See the developer docs for more details
"""

from itertools import chain, product
import warnings

from sympy.utilities.iterables import iterable as sympy_iterable

from sympy import Sum as Summation
from sympy import Symbol as sp_Symbol
from sympy import Integer as sp_Integer
from sympy import ceiling

#==============================================================================
from pyccel.utilities.strings import random_string
from pyccel.ast.basic         import PyccelAstNode, TypedAstNode, ScopedAstNode

from pyccel.ast.builtins import PythonPrint, PythonTupleFunction, PythonSetFunction
from pyccel.ast.builtins import PythonComplex, PythonDict, PythonDictFunction, PythonListFunction
from pyccel.ast.builtins import builtin_functions_dict, PythonImag, PythonReal
from pyccel.ast.builtins import PythonList, PythonConjugate , PythonSet
from pyccel.ast.builtins import (PythonRange, PythonZip, PythonEnumerate,
                                 PythonTuple, Lambda, PythonMap)

from pyccel.ast.builtin_methods.list_methods import ListMethod, ListAppend
from pyccel.ast.builtin_methods.set_methods  import SetMethod, SetAdd

from pyccel.ast.core import Comment, CommentBlock, Pass
from pyccel.ast.core import If, IfSection
from pyccel.ast.core import Allocate, Deallocate
from pyccel.ast.core import Assign, AliasAssign, SymbolicAssign
from pyccel.ast.core import AugAssign, CodeBlock
from pyccel.ast.core import Return, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core import ConstructorCall, InlineFunctionDef
from pyccel.ast.core import FunctionDef, Interface, FunctionAddress, FunctionCall, FunctionCallArgument
from pyccel.ast.core import ClassDef
from pyccel.ast.core import For
from pyccel.ast.core import Module
from pyccel.ast.core import While
from pyccel.ast.core import SymbolicPrint
from pyccel.ast.core import Del
from pyccel.ast.core import Program
from pyccel.ast.core import EmptyNode
from pyccel.ast.core import Concatenate
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import With
from pyccel.ast.core import Duplicate
from pyccel.ast.core import StarredArguments
from pyccel.ast.core import Iterable
from pyccel.ast.core import Decorator
from pyccel.ast.core import PyccelFunctionDef
from pyccel.ast.core import Assert

from pyccel.ast.class_defs import get_cls_base

from pyccel.ast.datatypes import CustomDataType, PyccelType, TupleType, VoidType, GenericType
from pyccel.ast.datatypes import PrimitiveIntegerType, StringType, SymbolicType
from pyccel.ast.datatypes import PythonNativeBool, PythonNativeInt, PythonNativeFloat
from pyccel.ast.datatypes import DataTypeFactory, HomogeneousContainerType
from pyccel.ast.datatypes import InhomogeneousTupleType, HomogeneousTupleType, HomogeneousSetType, HomogeneousListType
from pyccel.ast.datatypes import PrimitiveComplexType, FixedSizeNumericType, DictType, TypeAlias

from pyccel.ast.functionalexpr import FunctionalSum, FunctionalMax, FunctionalMin, GeneratorComprehension, FunctionalFor

from pyccel.ast.headers import FunctionHeader, MethodHeader, Header
from pyccel.ast.headers import MacroFunction, MacroVariable

from pyccel.ast.internals import PyccelFunction, Slice, PyccelSymbol
from pyccel.ast.itertoolsext import Product

from pyccel.ast.literals import LiteralTrue, LiteralFalse
from pyccel.ast.literals import LiteralInteger, LiteralFloat
from pyccel.ast.literals import Nil, LiteralString, LiteralImaginaryUnit
from pyccel.ast.literals import Literal, convert_to_literal, LiteralEllipsis

from pyccel.ast.mathext  import math_constants, MathSqrt, MathAtan2, MathSin, MathCos

from pyccel.ast.numpyext import NumpyMatmul, numpy_funcs
from pyccel.ast.numpyext import NumpyWhere, NumpyArray
from pyccel.ast.numpyext import NumpyTranspose, NumpyConjugate
from pyccel.ast.numpyext import NumpyNewArray, NumpyResultType
from pyccel.ast.numpyext import process_dtype as numpy_process_dtype

from pyccel.ast.numpytypes import NumpyNDArrayType

from pyccel.ast.omp import (OMP_For_Loop, OMP_Simd_Construct, OMP_Distribute_Construct,
                            OMP_TaskLoop_Construct, OMP_Sections_Construct, Omp_End_Clause,
                            OMP_Single_Construct)

from pyccel.ast.operators import PyccelArithmeticOperator, PyccelIs, PyccelIsNot, IfTernaryOperator, PyccelUnarySub
from pyccel.ast.operators import PyccelNot, PyccelAdd, PyccelMul, PyccelPow
from pyccel.ast.operators import PyccelAssociativeParenthesis, PyccelDiv

from pyccel.ast.sympy_helper import sympy_to_pyccel, pyccel_to_sympy

from pyccel.ast.type_annotations import VariableTypeAnnotation, UnionTypeAnnotation, SyntacticTypeAnnotation
from pyccel.ast.type_annotations import FunctionTypeAnnotation, typenames_to_dtypes

from pyccel.ast.typingext import TypingFinal

from pyccel.ast.utilities import builtin_import as pyccel_builtin_import
from pyccel.ast.utilities import builtin_import_registry as pyccel_builtin_import_registry
from pyccel.ast.utilities import split_positional_keyword_arguments
from pyccel.ast.utilities import recognised_source, is_literal_integer

from pyccel.ast.variable import Constant
from pyccel.ast.variable import Variable
from pyccel.ast.variable import IndexedElement, AnnotatedPyccelSymbol
from pyccel.ast.variable import DottedName, DottedVariable

from pyccel.errors.errors import Errors
from pyccel.errors.errors import PyccelSemanticError

from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, UNDERSCORE_NOT_A_THROWAWAY,
        UNDEFINED_VARIABLE, IMPORTING_EXISTING_IDENTIFIED, INDEXED_TUPLE, LIST_OF_TUPLES,
        INVALID_INDICES, INCOMPATIBLE_ARGUMENT,
        UNRECOGNISED_FUNCTION_CALL, STACK_ARRAY_SHAPE_UNPURE_FUNC, STACK_ARRAY_UNKNOWN_SHAPE,
        ARRAY_DEFINITION_IN_LOOP, STACK_ARRAY_DEFINITION_IN_LOOP, MISSING_TYPE_ANNOTATIONS,
        INCOMPATIBLE_TYPES_IN_ASSIGNMENT, ARRAY_ALREADY_IN_USE, ASSIGN_ARRAYS_ONE_ANOTHER,
        INVALID_POINTER_REASSIGN, ARRAY_IS_ARG,
        INCOMPATIBLE_REDEFINITION_STACK_ARRAY, ARRAY_REALLOCATION, RECURSIVE_RESULTS_REQUIRED,
        PYCCEL_RESTRICTION_INHOMOG_LIST, UNDEFINED_IMPORT_OBJECT, UNDEFINED_LAMBDA_VARIABLE,
        UNDEFINED_LAMBDA_FUNCTION, UNDEFINED_INIT_METHOD, UNDEFINED_FUNCTION,
        INVALID_MACRO_COMPOSITION, WRONG_NUMBER_OUTPUT_ARGS, INVALID_FOR_ITERABLE,
        PYCCEL_RESTRICTION_LIST_COMPREHENSION_LIMITS, PYCCEL_RESTRICTION_LIST_COMPREHENSION_SIZE,
        UNUSED_DECORATORS, UNSUPPORTED_POINTER_RETURN_VALUE, PYCCEL_RESTRICTION_OPTIONAL_NONE,
        PYCCEL_RESTRICTION_PRIMITIVE_IMMUTABLE, PYCCEL_RESTRICTION_IS_ISNOT,
        FOUND_DUPLICATED_IMPORT, UNDEFINED_WITH_ACCESS, MACRO_MISSING_HEADER_OR_FUNC)

from pyccel.parser.base      import BasicParser
from pyccel.parser.syntactic import SyntaxParser

from pyccel.utilities.stage import PyccelStage

import pyccel.decorators as def_decorators
#==============================================================================

errors = Errors()
pyccel_stage = PyccelStage()

type_container = {
                   PythonTupleFunction : HomogeneousTupleType,
                   PythonListFunction : HomogeneousListType,
                   PythonSetFunction : HomogeneousSetType,
                  }

#==============================================================================

def _get_name(var):
    """."""

    if isinstance(var, str):
        return var
    if isinstance(var, (PyccelSymbol, DottedName)):
        return str(var)
    if isinstance(var, (IndexedElement)):
        return str(var.base)
    if isinstance(var, FunctionCall):
        return var.funcdef
    name = type(var).__name__
    msg = f'Name of Object : {name} cannot be determined'
    return errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg, symbol=var,
                severity='fatal')

#==============================================================================

class SemanticParser(BasicParser):
    """
    Class which handles the semantic stage as described in the developer docs.

    This class is described in detail in developer_docs/semantic_stage.md.
    It determines all semantic information which must be deduced in order to
    print a representation of the AST resulting from the syntactic stage in one
    of the target languages.

    Parameters
    ----------
    inputs : SyntaxParser
        A syntactic parser which has been used to generate a representation of
        the input code using Pyccel nodes.

    parents : list
        A list of parsers describing the files which import this file.

    d_parsers : list
        A list of parsers describing files imported by this file.

    **kwargs : dict
        Additional keyword arguments for BasicParser.
    """

    def __init__(self, inputs, *, parents = (), d_parsers = (), **kwargs):

        # a Parser can have parents, who are importing it.
        # imports are then its sons.
        self._parents = list(parents)
        self._d_parsers = dict(d_parsers)

        # ...
        if not isinstance(inputs, SyntaxParser):
            raise TypeError('> Expecting a syntactic parser as input')

        parser = inputs
        # ...

        # ...
        BasicParser.__init__(self, **kwargs)
        # ...

        # ...
        self._fst = parser._fst
        self._ast = parser._ast

        self._filename  = parser._filename
        self._mod_name  = ''
        self._metavars  = parser._metavars
        self.scope = parser.scope
        self.scope.imports['imports'] = {}
        self._module_namespace  = self.scope

        self._in_annotation = False

        # used to store the local variables of a code block needed for garbage collecting
        self._allocs = []

        # used to store code split into multiple lines to be reinserted in the CodeBlock
        self._additional_exprs = []

        # used to store variables if optional parameters are changed
        self._optional_params = {}

        # used to link pointers to their targets. This is important for classes which may
        # contain persistent pointers
        self._pointer_targets = []

        #
        self._code = parser._code
        # ...

        self.annotate()
        # ...

    #================================================================
    #                  Property accessors
    #================================================================

    @property
    def parents(self):
        """Returns the parents parser."""
        return self._parents

    @property
    def d_parsers(self):
        """Returns the d_parsers parser."""

        return self._d_parsers

    #================================================================
    #                     Public functions
    #================================================================

    def annotate(self):
        """
        Add type information to the AST.

        This function is the entry point for this class. It annotates the
        AST object created by the syntactic stage which was collected
        in the constructor. The annotation adds all necessary information
        about the type etc to describe the object sufficiently well for
        printing. See the developer docs for more details.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            An annotated object which can be printed.
        """

        if self.semantic_done:
            print ('> semantic analysis already done')
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        errors = Errors()
        if self.filename:
            errors.set_target(self.filename)

        # then we treat the current file

        ast = self.ast

        self._allocs.append(set())
        self._pointer_targets.append({})
        # we add the try/except to allow the parser to find all possible errors
        pyccel_stage.set_stage('semantic')
        ast = self._visit(ast)

        self._ast = ast

        self._semantic_done = True

        return ast

    #================================================================
    #              Utility functions for scope handling
    #================================================================

    def get_class_prefix(self, name):
        """
        Search for the class prefix of a dotted name in the current scope.

        Search for a Variable object with the class prefix found in the given
        name inside the current scope, defined by the local and global Python
        scopes. Return None if not found.

        Parameters
        ----------
        name : DottedName
            The dotted name which begins with a class definition.

        Returns
        -------
        Variable
            Returns the class definition if found or None otherwise.
        """
        prefix_parts = name.name[:-1]
        syntactic_prefix = prefix_parts[0] if len(prefix_parts) == 1 else DottedName(*prefix_parts)
        return self._visit(syntactic_prefix)

    def check_for_variable(self, name):
        """
        Search for a Variable object with the given name in the current scope.

        Search for a Variable object with the given name in the current scope,
        defined by the local and global Python scopes. Return None if not found.

        Parameters
        ----------
        name : str | DottedName
            The object describing the variable.

        Returns
        -------
        Variable
            Returns the variable if found or None.

        See Also
        --------
        get_variable
            A similar function which raises an error if the Variable is not found
            instead of returning None.
        """

        if isinstance(name, DottedName):
            prefix = self.get_class_prefix(name)
            try:
                class_def = prefix.cls_base
            except AttributeError:
                class_def = get_cls_base(prefix.class_type) or \
                            self.scope.find(str(prefix.class_type), 'classes')

            attr_name = name.name[-1]
            class_scope = class_def.scope
            if class_scope is None:
                # Pyccel defined classes have no variables
                return None

            attribute = class_scope.find(attr_name, 'variables') if class_def else None
            if attribute:
                return attribute.clone(attribute.name, new_class = DottedVariable, lhs = prefix)
            else:
                return None
        return self.scope.find(name, 'variables')

    def get_variable(self, name):
        """
        Get a Variable object with the given name from the current scope.

        Search for a Variable object with the given name in the current scope,
        defined by the local and global Python scopes. Raise an error if not found.

        Parameters
        ----------
        name : str
            The object describing the variable.

        Returns
        -------
        Variable
            Returns the variable found in the scope.

        Raises
        ------
        PyccelSemanticError
            Error raised if variable is not found.

        See Also
        --------
        check_for_variable
            A similar function which returns None if the Variable is not found
            instead of raising an error.
        """
        var = self.check_for_variable(name)
        if var is None:
            if name == '_':
                errors.report(UNDERSCORE_NOT_A_THROWAWAY,
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='fatal')
            else:
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='fatal')

        return var

    def get_variables(self, container):
        """
        Get all variables in the scope of interest.

        Get a list of all variables which are

        Parameters
        ----------
        container : Scope
            The object describing the relevant scope.

        Returns
        -------
        list
            A list of variables.
        """
        # this only works if called on a function scope
        # TODO needs more tests when we have nested functions
        variables = []
        variables.extend(container.variables.values())
        for sub_container in container.loops:
            variables.extend(self.get_variables(sub_container))
        return variables

    def get_class_construct(self, name):
        """
        Return the class datatype associated with name.

        Return the class datatype for name if it exists.
        Raise an error otherwise.

        Parameters
        ----------
        name : str
            The name of the class.

        Returns
        -------
        PyccelType
            The datatype for the class.

        Raises
        ------
        PyccelSemanticError
            Raised if the datatype cannot be found.
        """
        result = self.scope.find(name, 'cls_constructs')

        if result is None:
            msg = f'class construct {name} not found'
            return errors.report(msg,
                bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                severity='fatal')
        else:
            return result

    def insert_import(self, name, target, storage_name = None):
        """
        Insert a new import into the scope.

        Create and insert a new import in scope if it's not defined
        otherwise append target to existing import.

        Parameters
        ----------
        name : str-like
               The source from which the object is imported.
        target : AsName
               The imported object.
        storage_name : str-like
                The name which will be used to identify the Import in the
                container.
        """
        source = _get_name(name)
        if storage_name is None:
            storage_name = source
        imp = self.scope.find(source, 'imports')
        if imp is None:
            imp = self.scope.find(storage_name, 'imports')

        if imp is not None:
            imp_source = imp.source
            if imp_source == source:
                imp.define_target(target)
            else:
                errors.report(IMPORTING_EXISTING_IDENTIFIED,
                              symbol=name,
                              bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                              severity='fatal')
        else:
            container = self.scope.imports
            container['imports'][storage_name] = Import(source, target, True)


    def get_headers(self, name):
        """ Get all headers in the scope which reference the
        requested name
        """
        container = self.scope
        headers = []
        while container:
            if name in container.headers:
                if isinstance(container.headers[name], list):
                    headers += container.headers[name]
                else:
                    headers.append(container.headers[name])
            container = container.parent_scope
        return headers

    def create_tuple_of_inhomogeneous_elements(self, tuple_var):
        """
        Create a tuple of variables from a variable representing an inhomogeneous object.

        Create a tuple of variables that can be printed in a low-level language. An
        inhomogeneous object cannot be represented as is in a low-level language so
        it must be unpacked into a PythonTuple. This function is recursive so that
        variables with a type such as `tuple[tuple[int,bool],float]` generate
        `PythonTuple(PythonTuple(var_0_0, var_0_1), var_1)`.

        Parameters
        ----------
        tuple_var : Variable
            A variable which may or may not be an inhomogeneous tuple.

        Returns
        -------
        Variable | PythonTuple
            An object containing only variables that can be printed in a low-level language.
        """
        if isinstance(tuple_var.class_type, InhomogeneousTupleType):
            return PythonTuple(*[self.create_tuple_of_inhomogeneous_elements(self.scope.collect_tuple_element(v)) for v in tuple_var])
        else:
            return tuple_var

    #=======================================================
    #              Utility functions
    #=======================================================

    def _garbage_collector(self, expr):
        """
        Search in a CodeBlock if no trailing Return Node is present add the needed frees.

        The primary purpose of _garbage_collector is to search within a CodeBlock
        instance for cases where no trailing Return node is present, and when such
        situations occur, it adds the necessary deallocate operations to free up resources.

        Parameters
        ----------
        expr : CodeBlock
            The body where the method searches for the absence of trailing `Return` nodes.

        Returns
        -------
        List
            A list of instances of the `Deallocate` type.
        """

        deallocs = []
        if len(expr.body)>0 and not isinstance(expr.body[-1], Return):
            for i in self._allocs[-1]:
                if isinstance(i, DottedVariable):
                    if isinstance(i.lhs.class_type, CustomDataType) and self._current_function != '__del__':
                        continue
                if isinstance(i.class_type, CustomDataType) and i.is_alias:
                    continue
                deallocs.append(Deallocate(i))
        self._allocs.pop()
        return deallocs

    def _check_pointer_targets(self, exceptions = ()):
        """
        Check that all pointer targets to be deallocated are not needed beyond this scope.

        At the end of a scope (function/module/class) the objects contained within it are
        deallocated. However some objects may persist beyond the scope. For example a
        class instance persists after a call to a class method, and the arguments of a
        function persist after a call to that function. If one of these persistent objects
        contains a pointer then it is important that the target of that pointer has the
        same lifetime. The target must not be deallocated at the end of the function if
        the pointer persists.

        This function checks through self._pointer_targets[-1] which is the dictionary
        describing the association of pointers to targets in this scope. First it removes
        all pointers which are deallocated at the end of this context. Next it checks if
        any of the objects which will be deallocated are present amongst the targets for
        the scope. If this is the case then an error is raised. Finally it loops through
        the remaining pointer/target pairs and ensures that any arguments which are targets
        are marked as such.

        Parameters
        ----------
        exceptions : tuple of Variables
            A list of objects in `_allocs` which are to be ignored (variables appearing
            in a return statement).
        """
        for i in self._allocs[-1]:
            if isinstance(i, DottedVariable):
                if isinstance(i.lhs.class_type, CustomDataType) and self._current_function != '__del__':
                    continue
            if i in exceptions:
                continue
            self._pointer_targets[-1].pop(i, None)
        targets = {t[0]:t[1] for target_list in self._pointer_targets[-1].values() for t in target_list}
        for i in self._allocs[-1]:
            if isinstance(i, DottedVariable):
                if isinstance(i.lhs.class_type, CustomDataType) and self._current_function != '__del__':
                    continue
            if i in exceptions:
                continue
            if i in targets:
                errors.report(f"Variable {i} goes out of scope but may be the target of a pointer which is still required",
                        severity='error', symbol=targets[i])

        if self._current_function:
            func_name = self._current_function.name[-1] if isinstance(self._current_function, DottedName) else self._current_function
            current_func = self.scope.find(func_name, 'functions')
            arg_vars = {a.var:a for a in current_func.arguments}

            for p, t_list in self._pointer_targets[-1].items():
                if p in arg_vars and arg_vars[p].bound_argument:
                    for t,_ in t_list:
                        if t.is_argument:
                            argument_objects = t.get_direct_user_nodes(lambda x: isinstance(x, FunctionDefArgument))
                            assert len(argument_objects) == 1
                            argument_objects[0].persistent_target = True

    def _indicate_pointer_target(self, pointer, target, expr):
        """
        Indicate that a pointer is targetting a specific target.

        Indicate that a pointer is targetting a specific target by adding the pair
        to a dictionary in self._pointer_targets (the last dictionary in the list
        should be used as this is the one for the current scope).

        Parameters
        ----------
        pointer : Variable
            The variable which is pointing at something.

        target : Variable | IndexedElement
            The object being pointed at by the pointer.

        expr : PyccelAstNode
            The expression where the pointer was created (used for clear error
            messages).
        """
        if isinstance(pointer, DottedVariable):
            self._indicate_pointer_target(pointer.lhs, target, expr)
        elif isinstance(target, DottedVariable):
            self._indicate_pointer_target(pointer, target.lhs, expr)
        elif isinstance(target, IndexedElement):
            self._indicate_pointer_target(pointer, target.base, expr)
        elif isinstance(target, Variable):
            if target.is_alias:
                try:
                    sub_targets = self._pointer_targets[-1][target]
                except KeyError:
                    errors.report("Pointer cannot point at a non-local pointer\n"+PYCCEL_RESTRICTION_TODO,
                        severity='error', symbol=expr)
                self._pointer_targets[-1].setdefault(pointer, []).extend((t[0], expr) for t in sub_targets)
            else:
                target.is_target = True
                self._pointer_targets[-1].setdefault(pointer, []).append((target, expr))
        else:
            errors.report("Pointer cannot point at a temporary object",
                severity='error', symbol=expr)

    def _infer_type(self, expr):
        """
        Infer all relevant type information for the expression.

        Create a dictionary describing all the type information that can be
        inferred about the expression `expr`. This includes information about:
        - `class_type`
        - `shape`
        - `cls_base`
        - `memory_handling`

        Parameters
        ----------
        expr : pyccel.ast.basic.PyccelAstNode
                An AST object representing an object in the code whose type
                must be determined.

        Returns
        -------
        dict
            Dictionary containing all the type information which was inferred.
        """
        if not isinstance(expr, TypedAstNode):
            return {'class_type' : SymbolicType()}

        d_var = {
                'class_type' : expr.class_type,
                'shape'      : expr.shape,
                'cls_base'   : self.scope.find(str(expr.class_type), 'classes') or get_cls_base(expr.class_type),
                'memory_handling' : 'heap' if expr.rank > 0 else 'stack'
            }

        if isinstance(expr, Variable):
            d_var['memory_handling'] = expr.memory_handling
            if expr.cls_base:
                d_var['cls_base'   ] = expr.cls_base
            return d_var

        elif isinstance(expr, Concatenate):
            if any(getattr(a, 'on_heap', False) for a in expr.args):
                d_var['memory_handling'] = 'heap'
            else:
                d_var['memory_handling'] = 'stack'
            return d_var

        elif isinstance(expr, Duplicate):
            d = self._infer_type(expr.val)
            if d.get('on_stack', False) and isinstance(expr.length, LiteralInteger):
                d_var['memory_handling'] = 'stack'
            else:
                d_var['memory_handling'] = 'heap'
            return d_var

        elif isinstance(expr, NumpyTranspose):

            var = expr.internal_var

            d_var['memory_handling'] = 'alias' if isinstance(var, Variable) else 'heap'
            return d_var

        elif isinstance(expr, TypedAstNode):

            d_var['memory_handling'] = 'heap' if expr.rank > 0 else 'stack'
            d_var['cls_base'   ] = get_cls_base(expr.class_type)
            return d_var

        else:
            type_name = type(expr).__name__
            msg = f'Type of Object : {type_name} cannot be infered'
            return errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg, symbol=expr,
                bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                severity='fatal')

    def _extract_indexed_from_var(self, var, indices, expr):
        """
        Use indices to extract appropriate element from object 'var'.

        Use indices to extract appropriate element from object 'var'.
        This contains most of the contents of _visit_IndexedElement
        but is a separate function in order to be recursive.

        Parameters
        ----------
        var : Variable
            The variable being indexed.

        indices : iterable
            The indexes used to access the variable.

        expr : PyccelAstNode
            The node being parsed. This is useful for raising errors.

        Returns
        -------
        TypedAstNode
            The visited object.
        """

        # case of Pyccel ast Variable
        # if not possible we use symbolic objects

        if isinstance(var, PythonTuple):
            def is_literal_index(a):
                def is_int(a):
                    return isinstance(a, (int, LiteralInteger)) or \
                        (isinstance(a, PyccelUnarySub) and \
                         isinstance(a.args[0], (int, LiteralInteger)))
                if isinstance(a, Slice):
                    return all(is_int(s) or s is None for s in (a.start, a.step, a.stop))
                else:
                    return is_int(a)
            if all(is_literal_index(a) for a in indices):
                if len(indices)==1:
                    return var[indices[0]]
                else:
                    return self._visit(var[indices[0]][indices[1:]])
            else:
                pyccel_stage.set_stage('syntactic')
                tmp_var = PyccelSymbol(self.scope.get_new_name())
                assign = Assign(tmp_var, var)
                assign.set_current_ast(expr.python_ast)
                pyccel_stage.set_stage('semantic')
                self._additional_exprs[-1].append(self._visit(assign))
                var = self._visit(tmp_var)

        elif isinstance(var, Variable):
            # Nothing to do but excludes this case from the subsequent ifs
            pass

        elif hasattr(var,'__getitem__'):
            if len(indices)==1:
                return var[indices[0]]
            else:
                return self._visit(var[indices[0]][indices[1:]])

        elif isinstance(var, PyccelFunction):
            pyccel_stage.set_stage('syntactic')
            tmp_var = PyccelSymbol(self.scope.get_new_name())
            assign = Assign(tmp_var, var)
            assign.set_current_ast(expr.python_ast)
            pyccel_stage.set_stage('semantic')
            self._additional_exprs[-1].append(self._visit(assign))
            var = self._visit(tmp_var)

        else:
            errors.report(f"Can't index {type(var)}", symbol=expr,
                severity='fatal')

        indices = tuple(indices)

        if isinstance(var.class_type, InhomogeneousTupleType):

            arg = indices[0]

            if isinstance(arg, Slice):
                if ((arg.start is not None and not is_literal_integer(arg.start)) or
                        (arg.stop is not None and not is_literal_integer(arg.stop))):
                    errors.report(INDEXED_TUPLE, symbol=var,
                        bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                        severity='fatal')

                idx = slice(arg.start, arg.stop)
                orig_vars = [self.scope.collect_tuple_element(v) for v in var]
                selected_vars = orig_vars[idx]
                if len(indices)==1:
                    return PythonTuple(*selected_vars)
                else:
                    return PythonTuple(*[self._extract_indexed_from_var(var, indices[1:], expr) for var in selected_vars])

            elif isinstance(arg, LiteralInteger):

                if len(indices)==1:
                    return self.scope.collect_tuple_element(var[arg])

                var = var[arg]
                return self._extract_indexed_from_var(var, indices[1:], expr)

            else:
                errors.report(INDEXED_TUPLE, symbol=var,
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='fatal')

        if isinstance(var, PythonTuple) and not var.is_homogeneous:
            errors.report(LIST_OF_TUPLES, symbol=var,
                bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                severity='error')

        for arg in var[indices].indices:
            if not isinstance(arg, (Slice, LiteralEllipsis)) and not (hasattr(arg, 'dtype') and
                    isinstance(getattr(arg.dtype, 'primitive_type', None), PrimitiveIntegerType)):
                errors.report(INVALID_INDICES, symbol=var[indices],
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='error')
        return var[indices]

    def _create_PyccelOperator(self, expr, visited_args):
        """
        Create a PyccelOperator.

        Create a PyccelOperator by passing the visited arguments
        to the class.
        Called by _visit_PyccelOperator and other classes
        inheriting from PyccelOperator.

        Parameters
        ----------
        expr : PyccelOperator
            The expression being visited.

        visited_args : tuple of TypedAstNode
            The arguments passed to the operator.

        Returns
        -------
        PyccelOperator
            The new operator.
        """
        try:
            expr_new = type(expr)(*visited_args)
        except PyccelSemanticError as err:
            msg = str(err)
            errors.report(msg, symbol=expr, severity='fatal')
        return expr_new

    def _create_Duplicate(self, val, length):
        """
        Create a node which duplicates a tuple.

        Create a node which duplicates a tuple.
        Called by _visit_PyccelMul when a Duplicate is identified.

        Parameters
        ----------
        val : PyccelAstNode
            The tuple object. This object should have a class type which inherits from
            TupleType.

        length : LiteralInteger | TypedAstNode
            The number of times the tuple is duplicated.

        Returns
        -------
        Duplicate | PythonTuple
            The duplicated tuple.
        """
        # Arguments have been visited in PyccelMul

        if not isinstance(val.class_type, (TupleType, HomogeneousListType)):
            errors.report("Unexpected Duplicate", symbol=Duplicate(val, length),
                bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                severity='fatal')

        if isinstance(val.class_type, (HomogeneousTupleType, HomogeneousListType)):
            return Duplicate(val, length)
        else:
            if isinstance(length, LiteralInteger):
                length = length.python_value
            else:
                symbol_map = {}
                used_symbols = set()
                sympy_length = pyccel_to_sympy(length, symbol_map, used_symbols)
                if isinstance(sympy_length, sp_Integer):
                    length = int(sympy_length)
                else:
                    errors.report("Cannot create inhomogeneous tuple of unknown size",
                        symbol=Duplicate(val, length),
                        bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                        severity='fatal')

            return PythonTuple(*([self.scope.collect_tuple_element(v) for v in val]*length))

    def _handle_function_args(self, arguments):
        """
        Get a list of all function arguments.

        Get a list of all the function arguments which are passed
        to a function. This is done by visiting the syntactic
        FunctionCallArguments. If this argument contains a
        starred arguments object then the contents of this object
        are extracted into the final list.

        Parameters
        ----------
        arguments : list of FunctionCallArgument
            The arguments which were passed to the function.

        Returns
        -------
        list of FunctionCallArgument
            The arguments passed to the function.
        """
        args  = []
        for arg in arguments:
            a = self._visit(arg)
            val = a.value
            if isinstance(val, FunctionDef) and not isinstance(val, PyccelFunctionDef) and not val.is_semantic:
                semantic_func = self._annotate_the_called_function_def(val)
                a = FunctionCallArgument(semantic_func, keyword = a.keyword, python_ast = a.python_ast)

            if isinstance(val, StarredArguments):
                args.extend([FunctionCallArgument(av) for av in val.args_var])
            else:
                args.append(a)
        return args

    def _check_argument_compatibility(self, input_args, func_args, func, elemental, raise_error=True, error_type='error'):
        """
        Check that the provided arguments match the expected types.

        Check that the provided arguments match the expected types.

        Parameters
        ----------
        input_args : list
           The arguments provided to the function.
        func_args : list
           The arguments expected by the function.
        func : FunctionDef
           The called function (used for error output).
        elemental : bool
           Indicates if the function is elemental.
        raise_error : bool, default : True
           Raise the error if the arguments are incompatible.
        error_type : str, default : error
           The error type if errors are raised from the function.

        Returns
        -------
        bool
            Return True if the arguments are compatible, False otherwise.
        """
        if elemental:
            def incompatible(i_arg, f_arg):
                return i_arg.class_type.datatype != f_arg.class_type.datatype
        else:
            def incompatible(i_arg, f_arg):
                return i_arg.class_type != f_arg.class_type

        err_msgs = []
        # Compare each set of arguments
        for idx, (i_arg, f_arg) in enumerate(zip(input_args, func_args)):
            i_arg = i_arg.value
            f_arg = f_arg.var
            # Ignore types which cannot be compared
            if (i_arg is Nil()
                    or isinstance(f_arg, FunctionAddress)
                    or f_arg.class_type is GenericType()):
                continue

            # Check for compatibility
            if incompatible(i_arg, f_arg):
                expected  = str(f_arg.class_type)
                type_name = str(i_arg.class_type)
                received  = f'{i_arg} ({type_name})'
                err_msgs += [INCOMPATIBLE_ARGUMENT.format(idx+1, received, func, expected)]

        if err_msgs:
            if raise_error:
                bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset)
                errors.report('\n\n'.join(err_msgs), symbol = func, bounding_box=bounding_box, severity=error_type)
            else:
                return False
        return True

    def _handle_function(self, expr, func, args, *, is_method = False, use_build_functions = True):
        """
        Create the node representing the function call.

        Create a FunctionCall or an instance of a PyccelFunction
        from the function information and arguments.

        Parameters
        ----------
        expr : TypedAstNode
               The expression where this call is found (used for error output).

        func : FunctionDef | Interface
               The function being called.

        args : iterable
               The arguments passed to the function.

        is_method : bool, default = False
                Indicates if the function is a class method.

        use_build_functions : bool, default = True
                In `func` is a PyccelFunctionDef, indicates that the `_build_X` methods should
                be used. This is almost always true but may be false if this function is called
                from a `_build_X` method.

        Returns
        -------
        FunctionCall/PyccelFunction
            The semantic representation of the call.
        """
        if isinstance(func, PyccelFunctionDef):
            if use_build_functions:
                annotation_method = '_build_' + func.cls_name.__name__
                if hasattr(self, annotation_method):
                    if isinstance(expr, DottedName):
                        pyccel_stage.set_stage('syntactic')
                        if is_method:
                            new_expr = DottedName(args[0].value, FunctionCall(func, args[1:]))
                        else:
                            new_expr = FunctionCall(func, args)
                        new_expr.set_current_ast(expr.python_ast)
                        pyccel_stage.set_stage('semantic')
                        expr = new_expr
                    return getattr(self, annotation_method)(expr)

            argument_description = func.argument_description
            func = func.cls_name
            args, kwargs = split_positional_keyword_arguments(*args)

            # Ignore values passed by position but add any unspecified keywords
            # with the correct default value
            for kw, val in list(argument_description.items())[len(args):]:
                if kw not in kwargs:
                    kwargs[kw] = val

            try:
                new_expr = func(*args, **kwargs)
            except TypeError as e:
                message = str(e)
                if not message:
                    message = UNRECOGNISED_FUNCTION_CALL
                errors.report(message,
                              symbol = expr,
                              traceback = e.__traceback__,
                              severity = 'fatal')

            return new_expr
        else:
            if self._current_function == func.name:
                if len(func.results)>0 and not isinstance(func.results[0].var, TypedAstNode):
                    errors.report(RECURSIVE_RESULTS_REQUIRED, symbol=func, severity="fatal")

            parent_assign = expr.get_direct_user_nodes(lambda x: isinstance(x, Assign) and not isinstance(x, AugAssign))

            func_results = func.results if isinstance(func, FunctionDef) else func.functions[0].results
            if not parent_assign and len(func_results) == 1 and func_results[0].var.rank > 0:
                pyccel_stage.set_stage('syntactic')
                tmp_var = PyccelSymbol(self.scope.get_new_name())
                assign = Assign(tmp_var, expr)
                assign.set_current_ast(expr.python_ast)
                pyccel_stage.set_stage('semantic')
                self._additional_exprs[-1].append(self._visit(assign))
                return self._visit(tmp_var)

            func_args = func.arguments if isinstance(func,FunctionDef) else func.functions[0].arguments
            if len(args) > len(func_args):
                errors.report("Too many arguments passed in function call",
                        symbol = expr,
                        severity='fatal')

            new_expr = FunctionCall(func, args, self._current_function)
            for a, f_a in zip(new_expr.args, func_args):
                if f_a.persistent_target:
                    assert is_method
                    val = a.value
                    if isinstance(val, Variable):
                        a.value.is_target = True
                        self._indicate_pointer_target(args[0].value, a.value, expr)
                    else:
                        errors.report(f"{val} cannot be passed to function call as target. Please create a temporary variable.",
                                severity='error', symbol=expr)

            if None in new_expr.args:
                errors.report("Too few arguments passed in function call",
                        symbol = expr,
                        severity='error')
            elif isinstance(func, FunctionDef):
                self._check_argument_compatibility(args, func_args,
                            func, func.is_elemental)

            return new_expr

    def _sort_function_call_args(self, func_args, args):
        """
        Sort and add the missing call arguments to match the arguments in the function definition.

        We sort the call arguments by dividing them into two chunks, positional arguments and keyword arguments.
        We provide the default value of the keyword argument if the corresponding call argument is not present.

        Parameters
        ----------
        func_args : list[FunctionDefArgument]
          The arguments of the function definition.
        args : list[FunctionCallArgument]
          The arguments of the function call.

        Returns
        -------
        list[FunctionCallArgument]
            The sorted and complete call arguments.
        """
        input_args = [a for a in args if a.keyword is None]
        nargs = len(input_args)
        for ka in func_args[nargs:]:
            key = ka.name
            relevant_args = [a for a in args[nargs:] if a.keyword == key]
            n_relevant_args = len(relevant_args)
            assert n_relevant_args <= 1
            if n_relevant_args == 0 and ka.has_default:
                input_args.append(ka.default_call_arg)
            elif n_relevant_args == 1:
                input_args.append(relevant_args[0])

        return input_args

    def _annotate_the_called_function_def(self, old_func, function_call_args=None):
        """
        Annotate the called FunctionDef.

        Annotate the called FunctionDef.

        Parameters
        ----------
        old_func : FunctionDef|Interface
           The function that needs to be annotated.

        function_call_args : list[FunctionCallArgument], optional
           The list of the call arguments.

        Returns
        -------
        func: FunctionDef|Interface
            The new annotated function.
        """
        # The function call might be in a completely different scope from the FunctionDef
        # Store the current scope and go to the parent scope of the FunctionDef
        old_scope            = self._scope
        old_current_function = self._current_function
        names = []
        sc = old_func.scope if isinstance(old_func, FunctionDef) else old_func.syntactic_node.scope
        while sc.parent_scope is not None:
            sc = sc.parent_scope
            if not sc.name is None:
                names.append(sc.name)
        names.reverse()
        if names:
            self._current_function = DottedName(*names) if len(names)>1 else names[0]
        else:
            self._current_function = None

        while names:
            sc = sc.sons_scopes[names[0]]
            names = names[1:]

        # Set the Scope to the FunctionDef's parent Scope and annotate the old_func
        self._scope = sc
        self._visit_FunctionDef(old_func, function_call_args=function_call_args)
        new_name = self.scope.get_expected_name(old_func.name)
        # Retreive the annotated function
        func = self.scope.find(new_name, 'functions')
        # Add the Module of the imported function to the new function
        if old_func.is_imported:
            mod = old_func.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
            func.set_current_user_node(mod)

        # Go back to the original Scope
        self._scope = old_scope
        self._current_function = old_current_function
        # Remove the old_func from the imports dict and Assign the new annotated one
        if old_func.is_imported:
            scope = self.scope
            while new_name not in scope.imports['functions']:
                scope = scope.parent_scope
            assert old_func is scope.imports['functions'].get(new_name)
            func = func.clone(new_name, is_imported=True)
            func.set_current_user_node(mod)
            scope.imports['functions'][new_name] = func
        return func

    def _create_variable(self, name, class_type, rhs, d_lhs, *, arr_in_multirets=False,
                         insertion_scope = None):
        """
        Create a new variable.

        Create a new variable. In most cases this is just a call to
        `Variable.__init__`
        but in the case of a tuple variable it is a recursive call to
        create all elements in the tuple.
        This is done separately to _assign_lhs_variable to ensure that
        elements of a tuple do not exist in the scope.

        Parameters
        ----------
        name : str
            The name of the new variable.

        class_type : PyccelType
            The type of the new variable.

        rhs : Variable
            The value assigned to the lhs. This is required to call
            self._infer_type recursively for tuples.

        d_lhs : dict
            Dictionary of properties for the new Variable.

        arr_in_multirets : bool, default: False
            If True, the variable that will be created is an array
            in multi-values return, false otherwise.

        insertion_scope : Scope, optional
            The scope where the variable will be inserted. This is used to add any
            symbolic aliases for inhomogeneous tuples.

        Returns
        -------
        Variable
            The variable that has been created.
        """
        if isinstance(name, PyccelSymbol):
            is_temp = name.is_temp
        else:
            is_temp = False

        if insertion_scope is None:
            insertion_scope = self.scope

        if isinstance(class_type, InhomogeneousTupleType):
            if isinstance(rhs, FunctionCall):
                iterable = [self.scope.collect_tuple_element(r.var) for r in rhs.funcdef.results]
            elif isinstance(rhs, PyccelFunction):
                iterable = [IndexedElement(rhs, i)  for i in range(rhs.shape[0])]
            else:
                iterable = [self.scope.collect_tuple_element(r) for r in rhs]
            elem_vars = []
            for i,tuple_elem in enumerate(iterable):
                elem_name = self.scope.get_new_name( name + '_' + str(i) )
                elem_d_lhs = self._infer_type( tuple_elem )

                if not arr_in_multirets:
                    self._ensure_target( tuple_elem, elem_d_lhs )

                elem_type = elem_d_lhs.pop('class_type')

                var = self._create_variable(elem_name, elem_type, tuple_elem, elem_d_lhs,
                        insertion_scope = insertion_scope)
                elem_vars.append(var)

            if any(v.is_alias for v in elem_vars):
                d_lhs['memory_handling'] = 'alias'

            lhs = Variable(class_type, name, **d_lhs, is_temp=is_temp)

            for i, v in enumerate(elem_vars):
                insertion_scope.insert_symbolic_alias(IndexedElement(lhs, i), v)

        else:
            lhs = Variable(class_type, name, **d_lhs, is_temp=is_temp)

        return lhs

    def _ensure_target(self, rhs, d_lhs):
        """
        Function using data about the new lhs.

        Function using data about the new lhs to determine
        whether the lhs is an alias and the rhs is a target.

        Parameters
        ----------
        rhs : TypedAstNode
            The value assigned to the lhs.

        d_lhs : dict
            Dictionary of properties for the new Variable.
        """

        if isinstance(rhs, NumpyTranspose) and rhs.internal_var.on_heap:
            d_lhs['memory_handling'] = 'alias'
            rhs.internal_var.is_target = True

        if isinstance(rhs, Variable) and (rhs.is_ndarray or isinstance(rhs.class_type, CustomDataType)):
            d_lhs['memory_handling'] = 'alias'
            rhs.is_target = not rhs.is_alias

        if isinstance(rhs, IndexedElement) and rhs.rank > 0 and \
                (getattr(rhs.base, 'is_ndarray', False) or getattr(rhs.base, 'is_alias', False)):
            d_lhs['memory_handling'] = 'alias'
            rhs.base.is_target = not rhs.base.is_alias

    def _assign_lhs_variable(self, lhs, d_var, rhs, new_expressions, is_augassign,arr_in_multirets=False):
        """
        Create a variable from the left-hand side (lhs) of an assignment.
        
        Create a lhs based on the information in d_var, if the lhs already exists
        then check that it has the expected properties.

        Parameters
        ----------
        lhs : PyccelSymbol (or DottedName of PyccelSymbols)
            The representation of the lhs provided by the SyntacticParser.

        d_var : dict
            Dictionary of expected lhs properties.

        rhs : Variable / expression
            The representation of the rhs provided by the SemanticParser.
            This is necessary in order to set the rhs 'is_target' property
            if necessary.

        new_expressions : list
            A list which allows collection of any additional expressions
            resulting from this operation (e.g. Allocation).

        is_augassign : bool
            Indicates whether this is an assign ( = ) or an augassign ( += / -= / etc )
            This is necessary as the restrictions on the dtype are less strict in this
            case.

        arr_in_multirets : bool
            If True, rhs has an array in its results, otherwise, it should be set to False.
            It helps when we don't need lhs to be a pointer in case of a returned array in
            a tuple of results.

        Returns
        -------
        pyccel.ast.variable.Variable
            The representation of the lhs provided by the SemanticParser.
        """

        if isinstance(lhs, IndexedElement):
            lhs = self._visit(lhs)
        elif isinstance(lhs, (PyccelSymbol, DottedName)):

            name = lhs
            if lhs == '_':
                name = self.scope.get_new_name()
            class_type = d_var.pop('class_type')

            d_lhs = d_var.copy()
            # ISSUES #177: lhs must be a pointer when rhs is heap array
            if not arr_in_multirets:
                self._ensure_target(rhs, d_lhs)

            if isinstance(lhs, DottedName):
                prefix = self.get_class_prefix(lhs)
                class_def = prefix.cls_base
                attr_name = lhs.name[-1]
                attribute = class_def.scope.find(attr_name) if class_def else None
                if attribute:
                    var = attribute.clone(attribute.name, new_class = DottedVariable, lhs = prefix)
                else:
                    var = None
            else:
                symbolic_var = self.scope.find(lhs, 'symbolic_alias')
                if symbolic_var:
                    errors.report(f"{lhs} variable represents a symbolic concept. Its value cannot be changed.",
                            severity='fatal')
                var = self.scope.find(lhs)

            # Variable not yet declared (hence array not yet allocated)
            if var is None:

                if isinstance(lhs, DottedName):
                    prefix_parts = lhs.name[:-1]
                    syntactic_prefix = prefix_parts[0] if len(prefix_parts) == 1 else DottedName(*prefix_parts)
                    prefix = self._visit(syntactic_prefix)
                    class_def = prefix.cls_base
                    if prefix.name == 'self':
                        var = self.get_variable('self')

                        # Collect the name that should be used in the generated code
                        attribute_name = lhs.name[-1]
                        new_name = class_def.scope.get_expected_name(attribute_name)
                        # Create the attribute
                        member = self._create_variable(new_name, class_type, rhs, d_lhs,
                                insertion_scope = class_def.scope)

                        # Insert the attribute to the class scope
                        # Passing the original name ensures that the attribute can be found under this name
                        class_def.scope.insert_variable(member, attribute_name)

                        lhs = self.insert_attribute_to_class(class_def, var, member)
                    else:
                        errors.report(f"{lhs.name[0]} should be named : self", symbol=lhs, severity='fatal')
                # Update variable's dictionary with information from function decorators
                decorators = self.scope.decorators
                if decorators:
                    if 'stack_array' in decorators:
                        if name in decorators['stack_array']:
                            d_lhs.update(memory_handling='stack')
                    if 'allow_negative_index' in decorators:
                        if lhs in decorators['allow_negative_index']:
                            d_lhs.update(allows_negative_indexes=True)

                # We cannot allow the definition of a stack array from a shape which
                # is unknown at the declaration
                if not isinstance(class_type, StringType) and class_type.rank > 0 and d_lhs.get('memory_handling', None) == 'stack':
                    for a in d_lhs['shape']:
                        if (isinstance(a, FunctionCall) and not a.funcdef.is_pure) or \
                                any(not f.funcdef.is_pure for f in a.get_attribute_nodes(FunctionCall)):
                            errors.report(STACK_ARRAY_SHAPE_UNPURE_FUNC, symbol=a.funcdef.name,
                            severity='error',
                            bounding_box=(self.current_ast_node.lineno,
                                self.current_ast_node.col_offset))
                        if (isinstance(a, Variable) and not a.is_argument) \
                                or not all(b.is_argument for b in a.get_attribute_nodes(Variable)):
                            errors.report(STACK_ARRAY_UNKNOWN_SHAPE, symbol=name,
                            severity='error',
                            bounding_box=(self.current_ast_node.lineno,
                                self.current_ast_node.col_offset))

                if not isinstance(lhs, DottedVariable):
                    new_name = self.scope.get_expected_name(name)
                    # Create new variable
                    lhs = self._create_variable(new_name, class_type, rhs, d_lhs, arr_in_multirets=arr_in_multirets)

                    # Add variable to scope
                    self.scope.insert_variable(lhs, name)

                # ...
                # Add memory allocation if needed
                array_declared_in_function = (isinstance(rhs, FunctionCall) and not isinstance(rhs.funcdef, PyccelFunctionDef) \
                                            and not getattr(rhs.funcdef, 'is_elemental', False) and not isinstance(lhs.class_type, HomogeneousTupleType)) or arr_in_multirets
                if not isinstance(lhs.class_type, StringType) and lhs.on_heap and not array_declared_in_function:
                    if self.scope.is_loop:
                        # Array defined in a loop may need reallocation at every cycle
                        errors.report(ARRAY_DEFINITION_IN_LOOP, symbol=name,
                            severity='warning',
                            bounding_box=(self.current_ast_node.lineno,
                                self.current_ast_node.col_offset))
                        status='unknown'
                    else:
                        # Array defined outside of a loop will be allocated only once
                        status='unallocated'

                    # Create Allocate node
                    if isinstance(lhs.class_type, InhomogeneousTupleType):
                        args = [self.scope.collect_tuple_element(v) for v in lhs if v.rank>0]
                        new_args = []
                        while len(args) > 0:
                            for a in args:
                                if isinstance(a.class_type, InhomogeneousTupleType):
                                    new_args.extend(self.scope.collect_tuple_element(v) for v in a if v.rank>0)
                                else:
                                    new_expressions.append(Allocate(a,
                                        shape=a.alloc_shape, status=status))
                            args = new_args
                            new_args = []
                    else:
                        new_expressions.append(Allocate(lhs, shape=lhs.alloc_shape, status=status))
                # ...

                # ...
                # Add memory deallocation
                if isinstance(lhs.class_type, CustomDataType) or (not lhs.on_stack and not isinstance(lhs.class_type, StringType)):
                    if isinstance(lhs.class_type, InhomogeneousTupleType):
                        args = [self.scope.collect_tuple_element(v) for v in lhs if v.rank>0]
                        new_args = []
                        while len(args) > 0:
                            for a in args:
                                if isinstance(a.class_type, InhomogeneousTupleType):
                                    new_args.extend(self.scope.collect_tuple_element(v) for v in a if v.rank>0)
                                else:
                                    self._allocs[-1].add(a)
                            args = new_args
                            new_args = []
                    else:
                        self._allocs[-1].add(lhs)
                # ...

                # We cannot allow the definition of a stack array in a loop
                if lhs.is_stack_array and self.scope.is_loop:
                    errors.report(STACK_ARRAY_DEFINITION_IN_LOOP, symbol=name,
                        severity='error',
                        bounding_box=(self.current_ast_node.lineno,
                            self.current_ast_node.col_offset))

                # Not yet supported for arrays: x=y+z, x=b[:]
                # Because we cannot infer shape of right-hand side yet
                know_lhs_shape = (lhs.rank == 0) or all(sh is not None for sh in lhs.alloc_shape) \
                        or isinstance(lhs.class_type, StringType)

                if not know_lhs_shape:
                    msg = f"Cannot infer shape of right-hand side for expression {lhs} = {rhs}"
                    errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg,
                        bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                        severity='fatal')

            # Variable already exists
            else:

                self._ensure_inferred_type_matches_existing(class_type, d_var, var, is_augassign, new_expressions, rhs)

                # in the case of elemental, lhs is not of the same class_type as
                # var.
                # TODO d_lhs must be consistent with var!
                # the following is a small fix, since lhs must be already
                # declared
                if isinstance(lhs, DottedName):
                    lhs = var.clone(var.name, new_class = DottedVariable, lhs = self._visit(lhs.name[0]))
                else:
                    lhs = var
        else:
            lhs_type = str(type(lhs))
            raise NotImplementedError(f"_assign_lhs_variable does not handle {lhs_type}")

        return lhs

    def _ensure_inferred_type_matches_existing(self, class_type, d_var, var, is_augassign, new_expressions, rhs):
        """
        Ensure that the inferred type matches the existing variable.

        Ensure that the inferred type of the new variable, matches the existing variable (which has the
        same name). If this is not the case then errors are raised preventing pyccel reaching the codegen
        stage.
        This function also handles any reallocations caused by differing shapes between the two objects.
        These allocations/deallocations are saved in the list new_expressions

        Parameters
        ----------
        class_type : PyccelType
            The inferred PyccelType.
        d_var : dict
            The inferred information about the variable. Usually created by the _infer_type function.
        var : Variable
            The existing variable.
        is_augassign : bool
            A boolean indicating if the assign statement is an augassign (tests are less strict).
        new_expressions : list
            A list to which any new expressions created are appended.
        rhs : TypedAstNode
            The right hand side of the expression : lhs=rhs.
            If is_augassign is False, this value is not used.
        """

        # TODO improve check type compatibility
        if not isinstance(var, Variable):
            name = var.name
            message = INCOMPATIBLE_TYPES_IN_ASSIGNMENT.format(type(var), class_type)
            if var.pyccel_staging == "syntactic":
                new_name = self.scope.get_expected_name(name)
                if new_name != name:
                    message += '\nThis error may be due to object renaming to avoid name clashes (language-specific or otherwise).'
                    message += f'The conflict is with "{name}".'
                    name = new_name
            errors.report(message,
                    symbol=f'{name}={class_type}',
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='fatal')

        if not is_augassign and var.is_ndarray and var.is_target:
            errors.report(ARRAY_ALREADY_IN_USE,
                bounding_box=(self.current_ast_node.lineno,
                    self.current_ast_node.col_offset),
                        severity='error', symbol=var.name)

        elif not is_augassign and var.is_ndarray and isinstance(rhs, (Variable, IndexedElement)) and var.on_heap:
            errors.report(ASSIGN_ARRAYS_ONE_ANOTHER,
                bounding_box=(self.current_ast_node.lineno,
                    self.current_ast_node.col_offset),
                        severity='error', symbol=var)

        elif var.is_ndarray and var.is_alias and isinstance(rhs, NumpyNewArray):
            errors.report(INVALID_POINTER_REASSIGN,
                bounding_box=(self.current_ast_node.lineno,
                    self.current_ast_node.col_offset),
                        severity='error', symbol=var.name)

        elif var.is_ndarray and var.is_alias and not is_augassign:
            # we allow pointers to be reassigned multiple times
            # pointers reassigning need to call free_pointer func
            # to remove memory leaks
            new_expressions.append(Deallocate(var))

        elif class_type != var.class_type:
            if is_augassign:
                tmp_result = PyccelAdd(var, rhs)
                result_type = tmp_result.class_type
                raise_error = var.class_type != result_type
            else:
                raise_error = True

            if raise_error:
                name = var.name
                rhs_str = str(rhs)
                errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT.format(var.class_type, class_type),
                    symbol=f'{name}={rhs_str}',
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='error')

        elif not is_augassign and not isinstance(var.class_type, StringType):

            shape = var.shape

            # Get previous allocation calls
            previous_allocations = var.get_direct_user_nodes(lambda p: isinstance(p, Allocate))

            if len(previous_allocations) == 0:
                var.set_init_shape(d_var['shape'])

            if d_var['shape'] != shape:

                if var.is_argument:
                    errors.report(ARRAY_IS_ARG, symbol=var,
                        severity='error',
                        bounding_box=(self.current_ast_node.lineno,
                            self.current_ast_node.col_offset))

                elif var.is_stack_array:
                    if var.get_direct_user_nodes(lambda a: isinstance(a, Assign) and a.lhs is var):
                        errors.report(INCOMPATIBLE_REDEFINITION_STACK_ARRAY, symbol=var.name,
                            severity='error',
                            bounding_box=(self.current_ast_node.lineno,
                                self.current_ast_node.col_offset))

                else:
                    if previous_allocations:
                        var.set_changeable_shape()
                        last_allocation = previous_allocations[-1]

                        # Find outermost IfSection of last allocation
                        last_alloc_ifsection = last_allocation.get_user_nodes(IfSection)
                        alloc_ifsection = last_alloc_ifsection[-1] if last_alloc_ifsection else None
                        while len(last_alloc_ifsection)>0:
                            alloc_ifsection = last_alloc_ifsection[-1]
                            last_alloc_ifsection = alloc_ifsection.get_user_nodes(IfSection)

                        ifsection_has_if = len(alloc_ifsection.get_direct_user_nodes(
                                                            lambda x: isinstance(x,If))) == 1 \
                                        if alloc_ifsection else False

                        if alloc_ifsection and not ifsection_has_if:
                            status = last_allocation.status
                        elif last_allocation.get_user_nodes((If, For, While)):
                            status='unknown'
                        else:
                            status='allocated'
                    else:
                        status = 'unallocated'

                    new_expressions.append(Allocate(var, shape=d_var['shape'], status=status))

                    if status == 'unallocated':
                        self._allocs[-1].add(var)
                    else:
                        errors.report(ARRAY_REALLOCATION, symbol=var.name,
                            severity='warning',
                            bounding_box=(self.current_ast_node.lineno,
                                self.current_ast_node.col_offset))
            elif previous_allocations and previous_allocations[-1].get_user_nodes(IfSection) \
                        and not previous_allocations[-1].get_user_nodes((If)):
                # If previously allocated in If still under construction
                status = previous_allocations[-1].status

                new_expressions.append(Allocate(var, shape=d_var['shape'], status=status))
            elif isinstance(var.class_type, CustomDataType) and not var.is_alias:
                new_expressions.append(Deallocate(var))

    def _assign_GeneratorComprehension(self, lhs_name, expr):
        """
        Visit the GeneratorComprehension node.

        Create all necessary expressions for the
        GeneratorComprehension node definition.

        Parameters
        ----------
        lhs_name : str
                    The name to which the expression is assigned.
        expr : GeneratorComprehension
                The GeneratorComprehension node.

        Returns
        -------
        pyccel.ast.functionalexpr.GeneratorComprehension
                CodeBlock containing the semantic version of the GeneratorComprehension node.
        """
        result   = expr.expr

        loop = expr.loops
        nlevels = 0
        # Create throw-away variable to help obtain result type
        index   = Variable(PythonNativeInt(),self.scope.get_new_name('to_delete'), is_temp=True)
        self.scope.insert_variable(index)
        new_expr = []
        while isinstance(loop, For):
            nlevels+=1
            iterable = Iterable(self._visit(loop.iterable))
            n_index = max(1, iterable.num_loop_counters_required)
            # Set dummy indices to iterable object in order to be able to
            # obtain a target with a deducible dtype
            iterable.set_loop_counter(*[index]*n_index)

            iterator = loop.target

            # Collect a target with a deducible dtype
            iterator_rhs = iterable.get_target_from_range()
            # Use _visit_Assign to create the requested iterator with the correct type
            # The result of this operation is not stored, it is just used to declare
            # iterator with the correct dtype to allow correct dtype deductions later
            pyccel_stage.set_stage('syntactic')
            syntactic_assign = Assign(iterator, iterator_rhs, python_ast=expr.python_ast)
            pyccel_stage.set_stage('semantic')
            self._visit(syntactic_assign)

            loop_elem = loop.body.body[0]

            if isinstance(loop_elem, Assign):
                # If the result contains a GeneratorComprehension, treat it and replace
                # it with it's lhs variable before continuing
                gens = set(loop_elem.get_attribute_nodes(GeneratorComprehension))
                if len(gens)==1:
                    gen = gens.pop()
                    pyccel_stage.set_stage('syntactic')
                    assert isinstance(gen.lhs, PyccelSymbol) and gen.lhs.is_temp
                    gen_lhs = self.scope.get_new_name() if gen.lhs.is_temp else gen.lhs
                    syntactic_assign = Assign(gen_lhs, gen, python_ast=gen.python_ast)
                    pyccel_stage.set_stage('semantic')
                    assign = self._visit(syntactic_assign)

                    new_expr.append(assign)
                    loop.substitute(gen, assign.lhs)
                    loop_elem = loop.body.body[0]
            loop = loop_elem
        # Remove the throw-away variable from the scope
        self.scope.remove_variable(index)

        # Visit result expression (correctly defined as iterator
        # objects exist in the scope despite not being defined)
        result = self._visit(result)
        if isinstance(result, CodeBlock):
            result = result.body[-1]

        # Create start value
        if isinstance(expr, FunctionalSum):
            dtype = result.dtype
            if isinstance(dtype, PythonNativeBool):
                val = LiteralInteger(0, dtype)
            else:
                val = convert_to_literal(0, dtype)
            d_var = self._infer_type(PyccelAdd(result, val))
        elif isinstance(expr, FunctionalMin):
            val = math_constants['inf']
            d_var = self._infer_type(result)
        elif isinstance(expr, FunctionalMax):
            val = PyccelUnarySub(math_constants['inf'])
            d_var = self._infer_type(result)

        # Infer the final dtype of the expression
        class_type = d_var.pop('class_type')
        d_var['is_temp'] = expr.lhs.is_temp

        lhs  = self.check_for_variable(lhs_name)
        if lhs:
            self._ensure_inferred_type_matches_existing(class_type, d_var, lhs, False, new_expr, None)
        else:
            lhs_name = self.scope.get_expected_name(lhs_name)
            lhs = Variable(class_type, lhs_name, **d_var)
            self.scope.insert_variable(lhs)

        # Iterate over the loops
        # This provides the definitions of iterators as well
        # as the central expression
        loops = [self._visit(expr.loops)]

        # If necessary add additional expressions corresponding
        # to nested GeneratorComprehensions
        if new_expr:
            loop = loops[0]
            for _ in range(nlevels-1):
                loop = loop.body.body[0]
            for e in new_expr:
                loop.body.insert2body(e, back=False)
                e.loops[-1].scope.update_parent_scope(loop.scope, is_loop = True)

        # Initialise result with correct initial value
        stmt = Assign(lhs, val)
        stmt.set_current_ast(expr.python_ast)
        loops.insert(0, stmt)

        indices = [self._visit(i) for i in expr.indices]

        if isinstance(expr, FunctionalSum):
            expr_new = FunctionalSum(loops, lhs=lhs, indices = indices)
        elif isinstance(expr, FunctionalMin):
            expr_new = FunctionalMin(loops, lhs=lhs, indices = indices)
        elif isinstance(expr, FunctionalMax):
            expr_new = FunctionalMax(loops, lhs=lhs, indices = indices)
        expr_new.set_current_ast(expr.python_ast)
        return expr_new

    def _find_superclasses(self, expr):
        """
        Find all the superclasses in the scope.

        From a syntactic ClassDef, extract the names of the superclasses and
        search through the scope to find their definitions. If there is no
        definition then an error is raised.

        Parameters
        ----------
        expr : ClassDef
            The class whose superclasses we wish to find.

        Returns
        -------
        list
            An iterable containing the definitions of all the superclasses.

        Raises
        ------
        PyccelSemanticError
            A `PyccelSemanticError` is reported and will be raised after the
            semantic stage is complete.
        """
        parent = {s: self.scope.find(s, 'classes') for s in expr.superclasses}
        if any(c is None for c in parent.values()):
            for s,c in parent.items():
                if c is None:
                    errors.report(f"Couldn't find class {s} in scope", symbol=expr,
                            severity='error')
            parent = {s:c for s,c in parent.items() if c is not None}

        return list(parent.values())

    def _convert_syntactic_object_to_type_annotation(self, syntactic_annotation):
        """
        Convert an arbitrary syntactic object to a type annotation.

        Convert an arbitrary syntactic object to a type annotation. This means that
        the syntactic object is wrapped in a SyntacticTypeAnnotation (if necessary).
        This ensures that a type annotation is obtained instead of e.g. a function.

        Parameters
        ----------
        syntactic_annotation : PyccelAstNode
            A syntactic object that needs to be visited as a type annotation.

        Returns
        -------
        SyntacticTypeAnnotation
            A syntactic object that will be recognised as a type annotation.
        """
        if not isinstance(syntactic_annotation, SyntacticTypeAnnotation):
            pyccel_stage.set_stage('syntactic')
            syntactic_annotation = SyntacticTypeAnnotation(dtype=syntactic_annotation)
            pyccel_stage.set_stage('semantic')
        return syntactic_annotation

    def _get_indexed_type(self, base, args, expr):
        """
        Extract a type annotation from an IndexedElement.

        Extract a type annotation from an IndexedElement. This may be a type indexed with
        slices (indicating a NumPy array), or a class type such as tuple/list/etc which is
        indexed with the datatype.

        Parameters
        ----------
        base : type deriving from PyccelAstNode
            The object being indexed.
        args : tuple of PyccelAstNode
            The indices being used to access the base.
        expr : PyccelAstNode
            The annotation, used for error printing.

        Returns
        -------
        UnionTypeAnnotation
            The type annotation described by this object.
        """
        if isinstance(base, PyccelFunctionDef) and base.cls_name is TypingFinal:
            syntactic_annotation = args[0]
            if not isinstance(syntactic_annotation, SyntacticTypeAnnotation):
                pyccel_stage.set_stage('syntactic')
                syntactic_annotation = SyntacticTypeAnnotation(dtype=syntactic_annotation)
                pyccel_stage.set_stage('semantic')
            annotation = self._visit(syntactic_annotation)
            for t in annotation.type_list:
                t.is_const = True
            return annotation
        elif isinstance(base, UnionTypeAnnotation):
            return UnionTypeAnnotation(*[self._get_indexed_type(t, args, expr) for t in base.type_list])

        if all(isinstance(a, Slice) for a in args):
            rank = len(args)
            order = None if rank < 2 else 'C'
            if isinstance(base, VariableTypeAnnotation):
                dtype = base.class_type
                if dtype.rank != 0:
                    raise errors.report("NumPy element must be a scalar type", severity='fatal', symbol=expr)
                class_type = NumpyNDArrayType(numpy_process_dtype(dtype), rank, order)
            elif isinstance(base, PyccelFunctionDef):
                dtype_cls = base.cls_name
                dtype = numpy_process_dtype(dtype_cls.static_type())
                class_type = NumpyNDArrayType(dtype, rank, order)
            return VariableTypeAnnotation(class_type)

        if not any(isinstance(a, Slice) for a in args):
            if isinstance(base, PyccelFunctionDef):
                dtype_cls = base.cls_name
            else:
                raise errors.report(f"Unknown annotation base {base}\n"+PYCCEL_RESTRICTION_TODO,
                        severity='fatal', symbol=expr)
            if (len(args) == 2 and args[1] is LiteralEllipsis()) or len(args) == 1:
                syntactic_annotation = self._convert_syntactic_object_to_type_annotation(args[0])
                internal_datatypes = self._visit(syntactic_annotation)
                if dtype_cls in type_container:
                    class_type = type_container[dtype_cls]
                else:
                    raise errors.report(f"Unknown annotation base {base}\n"+PYCCEL_RESTRICTION_TODO,
                            severity='fatal', symbol=expr)
                type_annotations = [VariableTypeAnnotation(class_type(u.class_type), u.is_const)
                                    for u in internal_datatypes.type_list]
                return UnionTypeAnnotation(*type_annotations)
            elif len(args) == 2 and dtype_cls is PythonDictFunction:
                syntactic_key_annotation = self._convert_syntactic_object_to_type_annotation(args[0])
                syntactic_val_annotation = self._convert_syntactic_object_to_type_annotation(args[1])
                key_types = self._visit(syntactic_key_annotation)
                val_types = self._visit(syntactic_val_annotation)
                type_annotations = [VariableTypeAnnotation(DictType(k.class_type, v.class_type)) \
                                    for k,v in zip(key_types.type_list, val_types.type_list)]
                return UnionTypeAnnotation(*type_annotations)
            else:
                raise errors.report("Cannot handle non-homogenous type index\n"+PYCCEL_RESTRICTION_TODO,
                        severity='fatal', symbol=expr)

        raise errors.report("Unrecognised type slice",
                severity='fatal', symbol=expr)

    def insert_attribute_to_class(self, class_def, self_var, attrib):
        """
        Insert a new attribute into an existing class.

        Insert a new attribute into an existing class definition. In order to do this a dotted
        variable must be created. If the new attribute is an inhomogeneous tuple then this
        function is called recursively to insert each variable comprising the tuple into the
        class definition.

        Parameters
        ----------
        class_def : ClassDef
            The class defintion to which the attribute should be added.
        self_var : Variable
            The variable representing the 'self' variable of the class instance.
        attrib : Variable
            The attribute which should be inserted into the class defintion.

        Returns
        -------
        DottedVariable | PythonTuple
            The object that was inserted into the class definition.
        """
        # Create the local DottedVariable
        lhs = attrib.clone(attrib.name, new_class = DottedVariable, lhs = self_var)

        if isinstance(attrib.class_type, InhomogeneousTupleType):
            for v in attrib:
                self.insert_attribute_to_class(class_def, self_var, class_def.scope.collect_tuple_element(v))
        else:
            # update the attributes of the class and push it to the scope
            class_def.add_new_attribute(lhs)

        return lhs

    #====================================================
    #                 _visit functions
    #====================================================


    def _visit(self, expr):
        """
        Annotate the AST.

        The annotation is done by finding the appropriate function _visit_X
        for the object expr. X is the type of the object expr. If this function
        does not exist then the method resolution order is used to search for
        other compatible _visit_X functions. If none are found then an error is
        raised.
        
        Parameters
        ----------
        expr : pyccel.ast.basic.PyccelAstNode | PyccelSymbol
            Object to visit of type X.
        
        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            AST object which is the semantic equivalent of expr.
        """
        if getattr(expr, 'pyccel_staging', 'syntactic') == 'semantic':
            return expr

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors
        current_ast = self.current_ast_node

        if getattr(expr,'python_ast', None) is not None:
            self._current_ast_node = expr.python_ast

        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                obj = getattr(self, annotation_method)(expr)
                if isinstance(obj, PyccelAstNode) and self.current_ast_node:
                    obj.set_current_ast(self.current_ast_node)
                self._current_ast_node = current_ast
                return obj

        # Unknown object, we raise an error.
        return errors.report(PYCCEL_RESTRICTION_TODO, symbol=type(expr),
            bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
            severity='fatal')

    def _visit_Module(self, expr):
        imports = [self._visit(i) for i in expr.imports]
        init_func_body = [i for i in imports if not isinstance(i, EmptyNode)]

        for f in expr.funcs:
            self.insert_function(f)

        # Avoid conflicts with symbols from Program
        if expr.program:
            self.scope.insert_symbols(expr.program.scope.all_used_symbols)

        for c in expr.classes:
            self._visit(c)

        init_func_body += self._visit(expr.init_func).body
        mod_name = self.metavars.get('module_name', None)
        if mod_name is None:
            mod_name = expr.name
        else:
            self.scope.insert_symbol(mod_name)
        self._mod_name = mod_name
        if isinstance(expr.name, AsName):
            name_suffix = expr.name.name
        else:
            name_suffix = expr.name

        if expr.program:
            prog_name = 'prog_'+name_suffix
            prog_name = self.scope.get_new_name(prog_name)
            self._allocs.append(set())
            self._pointer_targets.append({})

            mod_scope = self.scope
            prog_syntactic_scope = expr.program.scope
            self.scope = mod_scope.new_child_scope(prog_name,
                    used_symbols = prog_syntactic_scope.local_used_symbols.copy(),
                    original_symbols = prog_syntactic_scope.python_names.copy())
            prog_scope = self.scope

            imports = [self._visit(i) for i in expr.program.imports]
            body = [i for i in imports if not isinstance(i, EmptyNode)]

            body += self._visit(expr.program.body).body

            program_body = CodeBlock(body)

            # Calling the Garbage collecting,
            # it will add the necessary Deallocate nodes
            # to the ast
            program_body.insert2body(*self._garbage_collector(program_body))

            self.scope = mod_scope

        for f in self.scope.functions.copy().values():
            if not f.is_semantic and not isinstance(f, InlineFunctionDef):
                assert isinstance(f, FunctionDef)
                self._visit(f)

        variables = self.get_variables(self.scope)
        init_func = None
        free_func = None
        program   = None

        comment_types = (Header, MacroFunction, EmptyNode, Comment, CommentBlock)

        if not all(isinstance(l, comment_types) for l in init_func_body):
            # If there are any initialisation statements then create an initialisation function
            init_var = Variable(PythonNativeBool(), self.scope.get_new_name('initialised'),
                                is_private=True)
            init_func_name = self.scope.get_new_name(name_suffix+'__init')
            # Ensure that the function is correctly defined within the namespaces
            init_scope = self.create_new_function_scope(init_func_name)
            for b in init_func_body:
                if isinstance(b, ScopedAstNode):
                    b.scope.update_parent_scope(init_scope, is_loop = True)
                if isinstance(b, FunctionalFor):
                    for l in b.loops:
                        if isinstance(l, ScopedAstNode):
                            l.scope.update_parent_scope(init_scope, is_loop = True)

            self.exit_function_scope()

            # Update variable scope for temporaries
            to_remove = []
            scope_variables = list(self.scope.variables.values())
            for v in scope_variables:
                if v.is_temp:
                    self.scope.remove_variable(v)
                    init_scope.insert_variable(v)
                    to_remove.append(v)
                    variables.remove(v)

            # Get deallocations
            deallocs = self._garbage_collector(CodeBlock(init_func_body))

            # Deallocate temporaries in init function
            dealloc_vars = [d.variable for d in deallocs]
            for i,v in enumerate(dealloc_vars):
                if v in to_remove:
                    d = deallocs.pop(i)
                    init_func_body.append(d)

            init_func_body = If(IfSection(PyccelNot(init_var),
                                init_func_body+[Assign(init_var, LiteralTrue())]))

            init_func = FunctionDef(init_func_name, [], [], [init_func_body],
                    global_vars = variables, scope=init_scope)
            self.insert_function(init_func)

        if init_func:
            free_func_name = self.scope.get_new_name(name_suffix+'__free')
            pyccelised_imports = [imp for imp_name, imp in self.scope.imports['imports'].items() \
                             if imp_name in self.d_parsers]

            import_frees = [self.d_parsers[imp.source].semantic_parser.ast.free_func for imp in pyccelised_imports \
                                if imp.source in self.d_parsers]
            import_frees = [f if f.name in imp.target else \
                             f.clone(next(i.target for i in imp.target \
                                        if isinstance(i, AsName) and i.name == f.name)) \
                            for f,imp in zip(import_frees, pyccelised_imports) if f]

            if deallocs or import_frees:
                # If there is anything that needs deallocating when the module goes out of scope
                # create a deallocation function
                import_free_calls = [FunctionCall(f,[],[]) for f in import_frees if f is not None]
                free_func_body = If(IfSection(init_var,
                    import_free_calls+deallocs+[Assign(init_var, LiteralFalse())]))
                # Ensure that the function is correctly defined within the namespaces
                scope = self.create_new_function_scope(free_func_name)
                free_func = FunctionDef(free_func_name, [], [], [free_func_body],
                                    global_vars = variables, scope = scope)
                self.exit_function_scope()
                self.insert_function(free_func)

        funcs = []
        interfaces = []
        for f in self.scope.functions.values():
            if isinstance(f, FunctionDef):
                funcs.append(f)
            elif isinstance(f, Interface):
                interfaces.append(f)

        # in the case of a header file, we need to convert all headers to
        # FunctionDef etc ...

        if self.is_header_file:
            # ARA : issue-999
            is_external = self.metavars.get('external', False)
            for name, headers in self.scope.headers.items():
                if all(isinstance(v, FunctionHeader) and \
                        not isinstance(v, MethodHeader) for v in headers):
                    F = self.scope.find(name, 'functions')
                    if F is None:
                        func_defs = []
                        for v in headers:
                            types = [self._visit(d).type_list[0] for d in v.dtypes]
                            args = [Variable(t.class_type, PyccelSymbol(f'anon_{i}'),
                                shape = None, is_const = t.is_const, is_optional = False,
                                cls_base = t.class_type,
                                memory_handling = 'heap' if t.rank > 0 else 'stack') for i,t in enumerate(types)]

                            types = [self._visit(d).type_list[0] for d in v.results]
                            results = [Variable(t.class_type, PyccelSymbol(f'result_{i}'), shape = None,
                                cls_base = t.class_type,
                                is_const = t.is_const, is_optional = False,
                                memory_handling = 'heap' if t.rank > 0 else 'stack') for i,t in enumerate(types)]

                            args = [FunctionDefArgument(a) for a in args]
                            results = [FunctionDefResult(r) for r in results]
                            func_defs.append(FunctionDef(v.name, args, results, [], is_external = is_external, is_header = True))

                        if len(func_defs) == 1:
                            F = func_defs[0]
                            funcs.append(F)
                        else:
                            F = Interface(name, func_defs)
                            interfaces.append(F)
                        self.insert_function(F)
                    else:
                        errors.report(IMPORTING_EXISTING_IDENTIFIED,
                                symbol=name,
                                severity='fatal')

        for v in variables:
            if v.rank > 0 and not v.is_alias:
                v.is_target = True

        mod = Module(mod_name,
                    variables,
                    funcs,
                    init_func = init_func,
                    free_func = free_func,
                    interfaces=interfaces,
                    classes=self.scope.classes.values(),
                    imports=self.scope.imports['imports'].values(),
                    scope=self.scope)

        if expr.program:
            container = prog_scope.imports
            container['imports'][mod_name] = Import(self.scope.get_python_name(mod_name), mod)

            if init_func:
                import_init  = FunctionCall(init_func, [], [])
                program_body.insert2body(import_init, back=False)

            if free_func:
                import_free  = FunctionCall(free_func,[],[])
                program_body.insert2body(import_free)

            program = Program(prog_name,
                            self.get_variables(prog_scope),
                            program_body,
                            container['imports'].values(),
                            scope=prog_scope)

            mod.program = program
        return mod

    def _visit_PythonTuple(self, expr):
        ls = [self._visit(i) for i in expr]
        return PythonTuple(*ls)

    def _visit_PythonList(self, expr):
        ls = [self._visit(i) for i in expr]
        try:
            expr = PythonList(*ls)
        except TypeError:
            errors.report(PYCCEL_RESTRICTION_INHOMOG_LIST, symbol=expr,
                severity='fatal')
        return expr

    def _visit_PythonSet(self, expr):
        ls = [self._visit(i) for i in expr]
        try:
            expr = PythonSet(*ls)
        except TypeError as e:
            message = str(e)
            errors.report(message, symbol=expr,
                severity='fatal')
        return expr

    def _visit_PythonDict(self, expr):
        keys = [self._visit(k) for k in expr.keys]
        vals = [self._visit(v) for v in expr.values]
        try:
            expr = PythonDict(keys, vals)
        except TypeError as e:
            errors.report(str(e), symbol=expr,
                severity='fatal')
        return expr

    def _visit_FunctionCallArgument(self, expr):
        value = self._visit(expr.value)
        a = FunctionCallArgument(value, expr.keyword)
        def generate_and_assign_temp_var():
            pyccel_stage.set_stage('syntactic')
            tmp_var = self.scope.get_new_name()
            syntactic_assign = Assign(tmp_var, expr.value, python_ast = expr.value.python_ast)
            pyccel_stage.set_stage('semantic')

            assign = self._visit(syntactic_assign)
            self._additional_exprs[-1].append(assign)
            return FunctionCallArgument(self._visit(tmp_var))
        if isinstance(value, (PyccelArithmeticOperator, PyccelFunction)) and value.rank:
            a = generate_and_assign_temp_var()
        elif isinstance(value, FunctionCall) and isinstance(value.class_type, CustomDataType):
            if not value.funcdef.results[0].var.is_alias:
                a = generate_and_assign_temp_var()
        return a

    def _visit_UnionTypeAnnotation(self, expr):
        annotations = [self._visit(syntax_type_annot) for syntax_type_annot in expr.type_list]
        types = [t for a in annotations for t in (a.type_list if isinstance(a, UnionTypeAnnotation) else [a])]
        return UnionTypeAnnotation(*types)

    def _visit_FunctionTypeAnnotation(self, expr):
        arg_types = [self._visit(a)[0] for a in expr.args]
        res_types = [self._visit(r)[0] for r in expr.results]
        return UnionTypeAnnotation(FunctionTypeAnnotation(arg_types, res_types))

    def _visit_TypingFinal(self, expr):
        annotation = self._visit(expr.arg)
        for t in annotation:
            t.is_const = True
        return annotation

    def _visit_FunctionDefArgument(self, expr):
        arg = self._visit(expr.var)
        value = None if expr.value is None else self._visit(expr.value)
        kwonly = expr.is_kwonly
        is_optional = isinstance(value, Nil)
        bound_argument = expr.bound_argument

        args = []
        for v in arg:
            if isinstance(v, Variable):
                dtype = v.class_type
                if isinstance(value, Literal) and value is not Nil():
                    value = convert_to_literal(value.python_value, dtype)
                clone_var = v.clone(v.name, is_optional = is_optional, is_argument = True)
                args.append(FunctionDefArgument(clone_var, bound_argument = bound_argument,
                                        value = value, kwonly = kwonly, annotation = expr.annotation))
            else:
                args.append(FunctionDefArgument(v.clone(v.name, is_optional = is_optional,
                                is_kwonly = kwonly, is_argument = True), bound_argument = bound_argument,
                                value = value, kwonly = kwonly, annotation = expr.annotation))
        return args

    def _visit_CodeBlock(self, expr):
        ls = []
        self._additional_exprs.append([])
        for b in expr.body:

            # Save parsed code
            line = self._visit(b)
            ls.extend(self._additional_exprs[-1])
            self._additional_exprs[-1] = []
            if isinstance(line, CodeBlock):
                ls.extend(line.body)
            # ----- If block to handle VariableHeader. To be removed when headers are deprecated. ---
            elif isinstance(line, list) and isinstance(line[0], Variable):
                self.scope.insert_variable(line[0])
                if len(line) != 1:
                    errors.report(f"Variable {line[0]} cannot have multiple types",
                            severity='error', symbol=line[0])
            # ---------------------------- End of if block ------------------------------------------
            else:
                ls.append(line)
        self._additional_exprs.pop()

        return CodeBlock(ls)

    def _visit_Nil(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_Break(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_Continue(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_Comment(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_CommentBlock(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_AnnotatedComment(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_OmpAnnotatedComment(self, expr):
        code = expr._user_nodes
        code = code[-1]
        index = code.body.index(expr)
        combined_loop = expr.combined and ('for' in expr.combined or 'distribute' in expr.combined or 'taskloop' in expr.combined)

        if isinstance(expr, (OMP_Sections_Construct, OMP_Single_Construct)) \
           and expr.has_nowait:
            for node in code.body[index+1:]:
                if isinstance(node, Omp_End_Clause):
                    if node.txt.startswith(expr.name, 4):
                        node.has_nowait = True

        if isinstance(expr, (OMP_For_Loop, OMP_Simd_Construct,
                    OMP_Distribute_Construct, OMP_TaskLoop_Construct)) or combined_loop:
            index += 1
            while index < len(code.body) and isinstance(code.body[index], (Comment, CommentBlock, Pass)):
                index += 1

            if index < len(code.body) and isinstance(code.body[index], For):
                end_expr = ['!$omp', 'end', expr.name]
                if expr.combined:
                    end_expr.append(expr.combined)
                if expr.has_nowait:
                    end_expr.append('nowait')
                code.body[index].end_annotation = ' '.join(e for e in end_expr if e)+'\n'
            else:
                type_name = type(expr).__name__
                msg = f"Statement after {type_name} must be a for loop."
                errors.report(msg, symbol=expr,
                    severity='fatal')

        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_Omp_End_Clause(self, expr):
        end_loop = any(c in expr.txt for c in ['for', 'distribute', 'taskloop', 'simd'])
        if end_loop:
            errors.report("For loops do not require an end clause. This clause is ignored",
                    severity='warning', symbol=expr)
            return EmptyNode()
        else:
            expr.clear_syntactic_user_nodes()
            expr.update_pyccel_staging()
            return expr

    def _visit_Literal(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_Pass(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_Variable(self, expr):
        name = self.scope.get_python_name(expr.name)
        var = self.get_variable(name)
        return self._optional_params.get(var, var)

    def _visit_str(self, expr):
        return repr(expr)

    def _visit_Slice(self, expr):
        start = self._visit(expr.start) if expr.start is not None else None
        stop = self._visit(expr.stop) if expr.stop is not None else None
        step = self._visit(expr.step) if expr.step is not None else None

        return Slice(start, stop, step)

    def _visit_IndexedElement(self, expr):
        var = self._visit(expr.base)

        if isinstance(var, (PyccelFunctionDef, VariableTypeAnnotation, UnionTypeAnnotation)):
            return self._get_indexed_type(var, expr.indices, expr)

        # TODO check consistency of indices with shape/rank
        args = [self._visit(idx) for idx in expr.indices]

        if (len(args) == 1 and isinstance(getattr(args[0], 'class_type', None), TupleType)):
            args = args[0]

        elif any(isinstance(getattr(a, 'class_type', None), TupleType) for a in args):
            n_exprs = None
            for a in args:
                if getattr(a, 'shape', None) and isinstance(a.shape[0], LiteralInteger):
                    a_len = a.shape[0]
                    if n_exprs:
                        assert n_exprs == a_len
                    else:
                        n_exprs = a_len

            if n_exprs is not None:
                new_expr_args = [[a[i] if hasattr(a, '__getitem__') else a for a in args]
                                 for i in range(n_exprs)]
                return NumpyArray(PythonTuple(*[var[a] for a in new_expr_args]))

        return self._extract_indexed_from_var(var, args, expr)

    def _visit_PyccelSymbol(self, expr):
        name = expr

        var = self.check_for_variable(name)

        if var is None:
            var = self.scope.find(name)
        if var is None:
            var = builtin_functions_dict.get(name, None)
            if var is not None:
                var = PyccelFunctionDef(name, var)

        if var is None and self._in_annotation:
            var = numpy_funcs.get(name, None)
            if name == 'real':
                var = numpy_funcs['float']
            elif name == '*':
                return GenericType()

        if var is None:
            if name == '_':
                errors.report(UNDERSCORE_NOT_A_THROWAWAY,
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='fatal')
            else:
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='fatal')
        return self._optional_params.get(var, var)

    def _visit_AnnotatedPyccelSymbol(self, expr):
        # Check if the variable already exists
        var = self.scope.find(expr, 'variables', local_only = True)
        if var is not None:
            errors.report("Variable has been declared multiple times",
                    symbol=expr, severity='error')

        if expr.annotation is None:
            errors.report(MISSING_TYPE_ANNOTATIONS,
                    symbol=expr, severity='fatal')

        # Get the semantic type annotation (should be UnionTypeAnnotation)
        types = self._visit(expr.annotation)

        if len(types.type_list) == 0:
            errors.report(MISSING_TYPE_ANNOTATIONS,
                    symbol=expr, severity='fatal')

        python_name = expr.name
        # Get the collisionless name from the scope
        if isinstance(python_name, DottedName):
            prefix_parts = python_name.name[:-1]
            syntactic_prefix = prefix_parts[0] if len(prefix_parts) == 1 else DottedName(*prefix_parts)
            prefix = self._visit(syntactic_prefix)
            class_def = prefix.cls_base
            attribute_name = python_name.name[-1]

            name = class_def.scope.get_expected_name(attribute_name)
            var_class = DottedVariable
            kwargs = {'lhs': prefix}
        else:
            name = self.scope.get_expected_name(python_name)
            var_class = Variable
            kwargs = {}

        # Use the local decorators to define the memory and index handling
        array_memory_handling = 'heap'
        decorators = self.scope.decorators
        if decorators:
            if 'stack_array' in decorators:
                if expr.name in decorators['stack_array']:
                    array_memory_handling = 'stack'
            if 'allow_negative_index' in decorators:
                if expr.name in decorators['allow_negative_index']:
                    kwargs['allows_negative_indexes'] = True

        # For each possible data type create the necessary variables
        possible_args = []
        for t in types.type_list:
            if isinstance(t, FunctionTypeAnnotation):
                args = t.args
                results = [FunctionDefResult(r.var.clone(r.var.name, is_argument = False), annotation=r.annotation) for r in t.results]
                address = FunctionAddress(name, args, results)
                possible_args.append(address)
            elif isinstance(t, VariableTypeAnnotation):
                class_type = t.class_type
                cls_base = self.scope.find(str(class_type), 'classes') or get_cls_base(class_type)
                v = var_class(class_type, name, cls_base = cls_base,
                        shape = None,
                        is_const = t.is_const, is_optional = False,
                        memory_handling = array_memory_handling if class_type.rank > 0 else 'stack',
                        **kwargs)
                possible_args.append(v)
            else:
                errors.report(PYCCEL_RESTRICTION_TODO + '\nUnrecoginsed type annotation',
                        severity='fatal', symbol=expr)

        # An annotated variable must have a type
        assert len(possible_args) != 0

        return possible_args

    def _visit_SyntacticTypeAnnotation(self, expr):
        self._in_annotation = True
        visited_dtype = self._visit(expr.dtype)
        self._in_annotation = False
        order = expr.order

        if isinstance(visited_dtype, PyccelFunctionDef):
            dtype_cls = visited_dtype.cls_name
            class_type = dtype_cls.static_type()
            return UnionTypeAnnotation(VariableTypeAnnotation(class_type))
        elif isinstance(visited_dtype, VariableTypeAnnotation):
            if order and order != visited_dtype.class_type.order:
                visited_dtype = VariableTypeAnnotation(visited_dtype.class_type.swap_order())
            return UnionTypeAnnotation(visited_dtype)
        elif isinstance(visited_dtype, UnionTypeAnnotation):
            return visited_dtype
        elif isinstance(visited_dtype, ClassDef):
            # TODO: Improve when #1676 is merged
            dtype = self.get_class_construct(visited_dtype.name)
            return UnionTypeAnnotation(VariableTypeAnnotation(dtype))
        elif isinstance(visited_dtype, PyccelType):
            return UnionTypeAnnotation(VariableTypeAnnotation(visited_dtype))
        else:
            raise errors.report(PYCCEL_RESTRICTION_TODO + ' Could not deduce type information',
                    severity='fatal', symbol=expr)

    def _visit_VariableTypeAnnotation(self, expr):
        return expr

    def _visit_DottedName(self, expr):

        var = self.check_for_variable(_get_name(expr))
        if var:
            return var

        lhs = expr.name[0] if len(expr.name) == 2 \
                else DottedName(*expr.name[:-1])
        rhs = expr.name[-1]

        visited_lhs = self._visit(lhs)
        first = visited_lhs
        if isinstance(visited_lhs, FunctionCall):
            results = visited_lhs.funcdef.results
            if len(results) != 1:
                errors.report("Cannot get attribute of function call with multiple returns",
                        symbol=expr, severity='fatal')
            first = results[0].var
        rhs_name = _get_name(rhs)

        # Handle case of imported module
        if isinstance(first, Module):

            if rhs_name in first:
                imp = self.scope.find(_get_name(lhs), 'imports')

                new_name = rhs_name
                if imp is not None:
                    new_name = imp.find_module_target(rhs_name)
                    if new_name is None:
                        new_name = self.scope.get_new_name(rhs_name)

                        # Save the import target that has been used
                        imp.define_target(AsName(first[rhs_name], PyccelSymbol(new_name)))
                elif isinstance(rhs, FunctionCall):
                    self.scope.imports['functions'][new_name] = first[rhs_name]
                elif isinstance(rhs, ConstructorCall):
                    self.scope.imports['classes'][new_name] = first[rhs_name]
                elif isinstance(rhs, Variable):
                    self.scope.imports['variables'][new_name] = rhs

                if isinstance(rhs, FunctionCall):
                    # If object is a function
                    args  = self._handle_function_args(rhs.args)
                    func  = first[rhs_name]
                    if new_name != rhs_name:
                        if hasattr(func, 'clone') and not isinstance(func, PyccelFunctionDef):
                            func  = func.clone(new_name)
                    pyccel_stage.set_stage('syntactic')
                    syntactic_call = FunctionCall(func, args)
                    pyccel_stage.set_stage('semantic')
                    return self._handle_function(syntactic_call, func, args)
                elif isinstance(rhs, Constant):
                    var = first[rhs_name]
                    if new_name != rhs_name:
                        var.name = new_name
                    return var
                else:
                    # If object is something else (eg. dict)
                    var = first[rhs_name]
                    return var
            else:
                errors.report(UNDEFINED_IMPORT_OBJECT.format(rhs_name, str(lhs)),
                        symbol=expr, severity='fatal')
        if isinstance(first, ClassDef):
            errors.report("Static class methods are not yet supported", symbol=expr,
                    severity='fatal')

        d_var = self._infer_type(first)
        class_type = d_var['class_type']
        cls_base = get_cls_base(class_type)
        if cls_base is None:
            cls_base = self.scope.find(str(class_type), 'classes')

        # look for a class method
        if isinstance(rhs, FunctionCall):
            method = cls_base.get_method(rhs_name)
            macro = self.scope.find(rhs_name, 'macros')
            if macro is not None:
                master = macro.master
                args = rhs.args
                args = [lhs] + list(args)
                args = [self._visit(i) for i in args]
                args = macro.apply(args)
                return FunctionCall(master, args, self._current_function)

            args = [FunctionCallArgument(visited_lhs), *self._handle_function_args(rhs.args)]
            if cls_base.name == 'numpy.ndarray':
                numpy_class = method.cls_name
                self.insert_import('numpy', AsName(numpy_class, numpy_class.name))
            return self._handle_function(expr, method, args, is_method = True)

        # look for a class attribute / property
        elif isinstance(rhs, PyccelSymbol) and cls_base:
            # standard class attribute
            second = self.check_for_variable(expr)
            if second:
                return second

            # class property?
            else:
                method = cls_base.get_method(rhs_name)
                assert 'property' in method.decorators
                if cls_base.name == 'numpy.ndarray':
                    numpy_class = method.cls_name
                    self.insert_import('numpy', AsName(numpy_class, numpy_class.name))
                return self._handle_function(expr, method, [FunctionCallArgument(visited_lhs)], is_method = True)

        # look for a macro
        else:

            macro = self.scope.find(rhs_name, 'macros')

            # Macro
            if isinstance(macro, MacroVariable):
                return macro.master
            elif isinstance(macro, MacroFunction):
                args = macro.apply([visited_lhs])
                return FunctionCall(macro.master, args, self._current_function)

        # did something go wrong?
        return errors.report(f'Attribute {rhs_name} not found',
            bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
            severity='fatal')

    def _visit_PyccelOperator(self, expr):
        args     = [self._visit(a) for a in expr.args]
        return self._create_PyccelOperator(expr, args)

    def _visit_PyccelAdd(self, expr):
        args = [self._visit(a) for a in expr.args]
        arg0 = args[0]
        if isinstance(arg0.class_type, (TupleType, HomogeneousListType)):
            arg1 = args[1]
            is_homogeneous = not isinstance(arg0.class_type, InhomogeneousTupleType) and \
                                arg0.class_type == arg1.class_type
            if is_homogeneous:
                return Concatenate(*args)
            else:
                if not (isinstance(arg0.shape[0], (LiteralInteger, int)) and isinstance(arg1.shape[0], (LiteralInteger, int))):
                    errors.report("Can't create an inhomogeneous object from objects of unknown size",
                            severity='fatal', symbol=expr)

                tuple_args = [self.scope.collect_tuple_element(v) for v in arg0] + [self.scope.collect_tuple_element(v) for v in arg1]
                expr_new = PythonTuple(*tuple_args)
        else:
            expr_new = self._create_PyccelOperator(expr, args)
        return expr_new

    def _visit_PyccelMul(self, expr):
        args = [self._visit(a) for a in expr.args]
        if isinstance(args[0].class_type, (TupleType, HomogeneousListType)):
            expr_new = self._create_Duplicate(args[0], args[1])
        elif isinstance(args[1].class_type, (TupleType, HomogeneousListType)):
            expr_new = self._create_Duplicate(args[1], args[0])
        else:
            expr_new = self._create_PyccelOperator(expr, args)
        return expr_new

    def _visit_PyccelPow(self, expr):
        base, exponent = [self._visit(a) for a in expr.args]

        exp_val = exponent
        if isinstance(exponent, LiteralInteger):
            exp_val = exponent.python_value
        elif isinstance(exponent, PyccelAssociativeParenthesis):
            exp = exponent.args[0]
            # Handle (1/2)
            if isinstance(exp, PyccelDiv) and all(isinstance(a, Literal) for a in exp.args):
                exp_val = exp.args[0].python_value / exp.args[1].python_value

        if isinstance(base, (Literal, Variable)) and exp_val == 2:
            return PyccelMul(base, base)
        elif exp_val == 0.5:
            pyccel_stage.set_stage('syntactic')

            sqrt_name = self.scope.get_new_name('sqrt')
            imp_name = AsName('sqrt', sqrt_name)
            if isinstance(base.class_type.primitive_type, PrimitiveComplexType) or isinstance(exponent.class_type.primitive_type, PrimitiveComplexType):
                new_import = Import('cmath',imp_name)
            else:
                new_import = Import('math',imp_name)
            self._visit(new_import)
            if isinstance(expr.args[0], PyccelAssociativeParenthesis):
                new_call = FunctionCall(sqrt_name, [expr.args[0].args[0]])
            else:
                new_call = FunctionCall(sqrt_name, [expr.args[0]])

            pyccel_stage.set_stage('semantic')

            return self._visit(new_call)
        else:
            return PyccelPow(base, exponent)

    def _visit_Lambda(self, expr):
        errors.report("Lambda functions are not currently supported",
                symbol=expr, severity='fatal')
        expr_names = set(str(a) for a in expr.expr.get_attribute_nodes(PyccelSymbol))
        var_names = map(str, expr.variables)
        missing_vars = expr_names.difference(var_names)
        if len(missing_vars) > 0:
            errors.report(UNDEFINED_LAMBDA_VARIABLE, symbol = missing_vars,
                bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                severity='fatal')
        funcs = expr.expr.get_attribute_nodes(FunctionCall)
        for func in funcs:
            name = _get_name(func)
            f = self.scope.find(name, 'symbolic_functions')
            if f is None:
                errors.report(UNDEFINED_LAMBDA_FUNCTION, symbol=name,
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='fatal')
            else:

                f = f(*func.args)
                expr_new = expr.expr.subs(func, f)
                expr = Lambda(tuple(expr.variables), expr_new)
        return expr

    def _visit_FunctionCall(self, expr):
        name     = expr.funcdef
        try:
            name = self.scope.get_expected_name(name)
        except RuntimeError:
            pass

        func = self.scope.find(name, 'functions')

        if func is None:
            name = str(expr.funcdef)
            if name in builtin_functions_dict:
                func = PyccelFunctionDef(name, builtin_functions_dict[name])

        args = self._handle_function_args(expr.args)

        # Correct keyword names if scope is available
        # The scope is only available if the function body has been parsed
        # (i.e. not for headers or builtin functions)
        if (isinstance(func, FunctionDef) and func.scope) or isinstance(func, Interface):
            scope = func.scope if isinstance(func, FunctionDef) else func.functions[0].scope
            args = [a if a.keyword is None else \
                    FunctionCallArgument(a.value, scope.get_expected_name(a.keyword)) \
                    for a in args]
            func_args = func.arguments if isinstance(func,FunctionDef) else func.functions[0].arguments
            if not func.is_semantic:
                # Correct func_args keyword names
                func_args = [FunctionDefArgument(AnnotatedPyccelSymbol(scope.get_expected_name(a.var.name), a.annotation),
                            annotation=a.annotation, value=a.value, kwonly=a.is_kwonly, bound_argument=a.bound_argument)
                            for a in func_args]
            args      = self._sort_function_call_args(func_args, args)
            is_inline = func.is_inline if isinstance(func, FunctionDef) else func.functions[0].is_inline
            if not func.is_semantic:
                if not is_inline:
                    func = self._annotate_the_called_function_def(func)
                else:
                    func = self._annotate_the_called_function_def(func, function_call_args=args)
            elif is_inline and isinstance(func, Interface):
                is_compatible = False
                for f in func.functions:
                    fl = self._check_argument_compatibility(args, f.arguments, func, f.is_elemental, raise_error=False)
                    is_compatible |= fl
                if not is_compatible:
                    func = self._annotate_the_called_function_def(func, function_call_args=args)

        if name == 'lambdify':
            args = self.scope.find(str(expr.args[0]), 'symbolic_functions')

        if self.scope.find(name, 'cls_constructs'):

            # TODO improve the test
            # we must not invoke the scope like this

            cls = self.scope.find(name, 'classes')
            d_methods = cls.methods_as_dict
            method = d_methods.pop('__init__', None)

            if method is None:

                # TODO improve case of class with the no __init__

                errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='error')
            dtype = method.arguments[0].var.class_type
            cls_def = method.arguments[0].var.cls_base
            d_var = {'class_type' : dtype,
                    'memory_handling':'stack',
                    'shape' : None,
                    'cls_base' : cls_def,
                    }
            new_expression = []

            lhs = expr.get_user_nodes(Assign)[0].lhs
            if isinstance(lhs, AnnotatedPyccelSymbol):
                annotation = self._visit(lhs.annotation)
                if len(annotation.type_list) != 1 or annotation.type_list[0].class_type != method.arguments[0].var.class_type:
                    errors.report(f"Unexpected type annotation in creation of {cls_def.name}",
                            symbol=annotation, severity='error')
                lhs = lhs.name

            cls_variable = self._assign_lhs_variable(lhs, d_var, expr, new_expression, False)
            self._additional_exprs[-1].extend(new_expression)
            args = (FunctionCallArgument(cls_variable), *args)
            self._check_argument_compatibility(args, method.arguments,
                            method, method.is_elemental)

            new_expr = ConstructorCall(method, args, cls_variable)

            for a, f_a in zip(new_expr.args, method.arguments):
                if f_a.persistent_target:
                    val = a.value
                    if isinstance(val, Variable):
                        a.value.is_target = True
                        self._indicate_pointer_target(cls_variable, a.value, expr.get_user_nodes(Assign)[0])
                    else:
                        errors.report(f"{val} cannot be passed to class constructor call as target. Please create a temporary variable.",
                                severity='error', symbol=expr)

            self._allocs[-1].add(cls_variable)
            return new_expr
        else:

            # first we check if it is a macro, in this case, we will create
            # an appropriate FunctionCall

            macro = self.scope.find(name, 'macros')
            if macro is not None:
                func = macro.master.funcdef
                name = _get_name(func.name)
                args = macro.apply(args)

            if func is None:
                return errors.report(UNDEFINED_FUNCTION, symbol=name,
                        bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                        severity='fatal')
            else:
                return self._handle_function(expr, func, args)

    def _visit_Assign(self, expr):
        # TODO unset position at the end of this part
        new_expressions = []
        python_ast = expr.python_ast
        assert python_ast

        rhs = expr.rhs
        lhs = expr.lhs

        if isinstance(lhs, AnnotatedPyccelSymbol):
            semantic_lhs = self._visit(lhs)
            if len(semantic_lhs) != 1:
                errors.report("Cannot declare variable with multiple types",
                        symbol=expr, severity='error')
            semantic_lhs_var = semantic_lhs[0]
            if isinstance(semantic_lhs_var, DottedVariable):
                cls_def = semantic_lhs_var.lhs.cls_base
                insert_scope = cls_def.scope
                cls_def.add_new_attribute(semantic_lhs_var)
            else:
                insert_scope = self.scope

            lhs = lhs.name
            if semantic_lhs_var.class_type is TypeAlias():
                if not isinstance(rhs, SyntacticTypeAnnotation):
                    pyccel_stage.set_stage('syntactic')
                    rhs = SyntacticTypeAnnotation(rhs)
                    pyccel_stage.set_stage('semantic')
                type_annot = self._visit(rhs)
                self.scope.insert_symbolic_alias(lhs, type_annot)
                return EmptyNode()

            try:
                insert_scope.insert_variable(semantic_lhs_var)
            except RuntimeError as e:
                errors.report(e, symbol=expr, severity='error')


        # Steps before visiting
        if isinstance(rhs, GeneratorComprehension):
            rhs.substitute(rhs.lhs, lhs)
            genexp = self._assign_GeneratorComprehension(_get_name(lhs), rhs)
            if isinstance(expr, AugAssign):
                new_expressions.append(genexp)
                rhs = genexp.lhs
            elif genexp.lhs.name == lhs:
                return genexp
            else:
                new_expressions.append(genexp)
                rhs = genexp.lhs
        elif isinstance(rhs, IfTernaryOperator):
            value_true  = self._visit(rhs.value_true)
            if value_true.rank > 0 or value_true.dtype is StringType():
                # Temporarily deactivate type checks to construct syntactic assigns
                pyccel_stage.set_stage('syntactic')
                assign_true  = Assign(lhs, rhs.value_true, python_ast = python_ast)
                assign_false = Assign(lhs, rhs.value_false, python_ast = python_ast)
                pyccel_stage.set_stage('semantic')

                cond  = self._visit(rhs.cond)
                true_section  = IfSection(cond, [self._visit(assign_true)])
                false_section = IfSection(LiteralTrue(), [self._visit(assign_false)])
                return If(true_section, false_section)

        # Visit object
        if isinstance(rhs, FunctionCall):
            name = rhs.funcdef
            macro = self.scope.find(name, 'macros')
            if macro is None:
                rhs = self._visit(rhs)
                if isinstance(rhs, (PythonMap, PythonZip, PythonEnumerate, PythonRange)):
                    errors.report(f"{type(rhs)} cannot be saved to variables", symbol=expr, severity='fatal')
            else:

                # TODO check types from FunctionDef
                master = macro.master
                results = []
                args = [self._visit(i) for i in rhs.args]
                args_names = [arg.value.name for arg in args if isinstance(arg.value, Variable)]
                d_m_args = {arg.value.name:arg.value for arg in macro.master_arguments
                                  if isinstance(arg.value, Variable)}

                lhs_iter = lhs

                if not sympy_iterable(lhs_iter):
                    lhs_iter = [lhs]
                results_shapes = macro.get_results_shapes(args)
                for m_result, shape, result in zip(macro.results, results_shapes, lhs_iter):
                    if m_result in d_m_args and not result in args_names:
                        d_result = self._infer_type(d_m_args[m_result])
                        d_result['shape'] = shape
                        tmp = self._assign_lhs_variable(result, d_result, None, new_expressions, False)
                        results.append(tmp)
                    elif result in args_names:
                        _name = _get_name(result)
                        tmp = self.get_variable(_name)
                        results.append(tmp)
                    else:
                        # TODO: check for result in master_results
                        errors.report(INVALID_MACRO_COMPOSITION, symbol=result,
                            bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                            severity='error')

                expr = macro.make_necessary_copies(args, results)
                new_expressions += expr
                args = macro.apply(args, results=results)
                if isinstance(master.funcdef, FunctionDef):
                    func_call = FunctionCall(master.funcdef, args, self._current_function)
                    if new_expressions:
                        return CodeBlock([*new_expressions, func_call])
                    else:
                        return func_call
                else:
                    # TODO treate interface case
                    errors.report(PYCCEL_RESTRICTION_TODO,
                                  bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                                  severity='fatal')

        else:
            rhs = self._visit(rhs)

        if isinstance(rhs, NumpyResultType):
            errors.report("Cannot assign a datatype to a variable.",
                    symbol=expr, severity='error')

        # Checking for the result of _visit_ListExtend
        if isinstance(rhs, For) or (isinstance(rhs, CodeBlock) and
            isinstance(rhs.body[0], (ListMethod, SetMethod))):
            return rhs
        if isinstance(rhs, ConstructorCall):
            return rhs

        elif isinstance(rhs, CodeBlock) and len(rhs.body)>1 and isinstance(rhs.body[1], FunctionalFor):
            return rhs

        elif isinstance(rhs, FunctionCall):
            func = rhs.funcdef
            results = func.results
            if results:
                if len(results)==1:
                    d_var = self._infer_type(results[0].var)
                else:
                    d_var = self._infer_type(PythonTuple(*[r.var for r in results]))
            elif expr.lhs.is_temp:
                return rhs
            else:
                raise NotImplementedError("Cannot assign result of a function without a return")

            # case of elemental function
            # if the input and args of func do not have the same shape,
            # then the lhs must be already declared
            if func.is_elemental:
                # we first compare the funcdef args with the func call
                # args
                # d_var = None
                func_args = func.arguments
                call_args = rhs.args
                f_ranks = [x.var.rank for x in func_args]
                c_ranks = [x.value.rank for x in call_args]
                same_ranks = [x==y for (x,y) in zip(f_ranks, c_ranks)]
                if not all(same_ranks):
                    assert len(c_ranks) == 1
                    arg = call_args[0].value
                    d_var['shape'          ] = arg.shape
                    d_var['memory_handling'] = arg.memory_handling
                    d_var['class_type'     ] = arg.class_type
                    d_var['cls_base'       ] = arg.cls_base

        elif isinstance(rhs, NumpyTranspose):
            d_var  = self._infer_type(rhs)
            if d_var['memory_handling'] == 'alias' and not isinstance(lhs, IndexedElement):
                rhs = rhs.internal_var
        elif isinstance(rhs, PyccelFunction) and isinstance(rhs.dtype, VoidType):
            if expr.lhs.is_temp:
                return rhs
            else:
                raise NotImplementedError("Cannot assign result of a function without a return")

        else:
            d_var  = self._infer_type(rhs)
            d_list = d_var if isinstance(d_var, list) else [d_var]

            for d in d_list:
                name = d['class_type'].__class__.__name__

                if name.startswith('Pyccel'):
                    name = name[6:]
                    d['cls_base'] = self.scope.find(name, 'classes')
                    if d_var['memory_handling'] == 'alias':
                        d['memory_handling'] = 'alias'
                    else:
                        d['memory_handling'] = d_var['memory_handling'] or 'heap'

                    # TODO if we want to use pointers then we set target to true
                    # in the ConsturcterCall

                if isinstance(rhs, Variable) and rhs.is_target:
                    # case of rhs is a target variable the lhs must be a pointer
                    d['memory_handling'] = 'alias'

        if isinstance(lhs, (PyccelSymbol, DottedName)):
            if isinstance(d_var, list):
                if len(d_var) == 1:
                    d_var = d_var[0]
                else:
                    errors.report(WRONG_NUMBER_OUTPUT_ARGS, symbol=expr,
                        severity='error')
                    return None
            lhs = self._assign_lhs_variable(lhs, d_var, rhs, new_expressions, isinstance(expr, AugAssign))

        # Handle assignment to multiple variables
        elif isinstance(lhs, (PythonTuple, PythonList)):
            if isinstance(rhs, FunctionCall):
                new_lhs = []
                r_iter = [r.var for r in rhs.funcdef.results]
                for i,(l,r) in enumerate(zip(lhs,r_iter)):
                    d = self._infer_type(r)
                    new_lhs.append( self._assign_lhs_variable(l, d, r, new_expressions, isinstance(expr, AugAssign),
                                                    arr_in_multirets=r.rank>0 ) )
                lhs = PythonTuple(*new_lhs)
            else:
                if isinstance(rhs.class_type, InhomogeneousTupleType):
                    r_iter = [self.scope.collect_tuple_element(v) for v in rhs]
                else:
                    r_iter = rhs

                body = []
                for i,(l,r) in enumerate(zip(lhs,r_iter)):
                    pyccel_stage.set_stage('syntactic')
                    local_assign = Assign(l,r, python_ast = expr.python_ast)
                    pyccel_stage.set_stage('semantic')
                    body.append(self._visit(local_assign))
                return CodeBlock(body)
        else:
            lhs = self._visit(lhs)

        if not isinstance(lhs, (list, tuple)):
            lhs = [lhs]
            if isinstance(d_var,dict):
                d_var = [d_var]

        if len(lhs) == 1:
            lhs = lhs[0]

        if isinstance(lhs, Variable):
            is_pointer = lhs.is_alias
        elif isinstance(lhs, IndexedElement):
            is_pointer = False
        elif isinstance(lhs, (PythonTuple, PythonList)):
            is_pointer = any(l.is_alias for l in lhs if isinstance(lhs, Variable))

        # TODO: does is_pointer refer to any/all or last variable in list (currently last)
        is_pointer = is_pointer and isinstance(rhs, (Variable, Duplicate))
        is_pointer = is_pointer or isinstance(lhs, Variable) and lhs.is_alias

        lhs = [lhs]
        rhs = [rhs]
        # Split into multiple Assigns to ensure AliasAssign is used where necessary
        unravelling = True
        while unravelling:
            unravelling = False
            new_lhs = []
            new_rhs = []
            for l,r in zip(lhs, rhs):
                # Split assign (e.g. for a,b = 1,c)
                if isinstance(l.class_type, InhomogeneousTupleType) \
                        and not isinstance(r, (FunctionCall, PyccelFunction)):
                    new_lhs.extend(self.scope.collect_tuple_element(v) for v in l)
                    new_rhs.extend(self.scope.collect_tuple_element(v) for v in r)
                    # Repeat step to handle tuples of tuples of etc.
                    unravelling = True
                elif isinstance(l, Variable) and isinstance(l.class_type, InhomogeneousTupleType):
                    new_lhs.append(self.create_tuple_of_inhomogeneous_elements(l))
                    new_rhs.append(r)
                    # Repeat step to handle tuples of tuples of etc.
                    unravelling = True
                elif isinstance(l, Variable) and l.is_optional:
                    if l in self._optional_params:
                        # Collect temporary variable which provides
                        # allocated memory space for this optional variable
                        new_lhs.append(self._optional_params[l])
                    else:
                        # Create temporary variable to provide allocated
                        # memory space before assigning to the pointer value
                        # (may be NULL)
                        tmp_var = self.scope.get_temporary_variable(l,
                                name = l.name+'_loc', is_optional = False)
                        self._optional_params[l] = tmp_var
                        new_lhs.append(tmp_var)
                    new_rhs.append(r)
                else:
                    new_lhs.append(l)
                    new_rhs.append(r)
            lhs = new_lhs
            rhs = new_rhs

        # Examine each assign and determine assign type (Assign, AliasAssign, etc)
        for l, r in zip(lhs,rhs):
            if isinstance(l, PythonTuple):
                for li in l:
                    if li.is_const:
                        # If constant (can't use annotations on tuple assignment)
                        errors.report("Cannot modify 'const' variable",
                            bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                            symbol=li, severity='error')
            else:
                if getattr(l, 'is_const', False) and (not isinstance(expr.lhs, AnnotatedPyccelSymbol) or len(l.get_all_user_nodes()) > 0):
                    # If constant and not the initialising declaration of a constant variable
                    errors.report("Cannot modify 'const' variable",
                        bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                        symbol=l, severity='error')
            if isinstance(expr, AugAssign):
                new_expr = AugAssign(l, expr.op, r)
            else:
                is_pointer_i = l.is_alias if isinstance(l, Variable) else is_pointer
                new_expr = Assign(l, r)

                if is_pointer_i:
                    new_expr = AliasAssign(l, r)
                    if isinstance(r, FunctionCall):
                        funcdef = r.funcdef
                        target_r_idx = funcdef.result_pointer_map[funcdef.results[0].var]
                        for ti in target_r_idx:
                            self._indicate_pointer_target(l, r.args[ti].value, expr)
                    else:
                        self._indicate_pointer_target(l, r, expr)

                elif new_expr.is_symbolic_alias:
                    new_expr = SymbolicAssign(l, r)

                    # in a symbolic assign, the rhs can be a lambda expression
                    # it is then treated as a def node

                    F = self.scope.find(l, 'symbolic_functions')
                    errors.report(PYCCEL_RESTRICTION_TODO,
                                  bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                                  severity='fatal')

            new_expressions.append(new_expr)


        if (len(new_expressions)==1):
            new_expressions = new_expressions[0]

            return new_expressions
        else:
            result = CodeBlock(new_expressions)
            return result

    def _visit_For(self, expr):

        scope = self.create_new_loop_scope()

        # treatment of the index/indices
        iterable = Iterable(self._visit(expr.iterable))

        new_expr = []

        start = LiteralInteger(0)
        iterator_d_var = self._infer_type(start)

        iterator = expr.target

        if iterable.num_loop_counters_required:
            indices = [Variable(PythonNativeInt(), self.scope.get_new_name(), is_temp=True)
                        for i in range(iterable.num_loop_counters_required)]
            iterable.set_loop_counter(*indices)
        else:
            if isinstance(iterable.iterable, PythonEnumerate):
                syntactic_index = iterator[0]
            else:
                syntactic_index = iterator

            index = self.check_for_variable(syntactic_index)
            if index is None:
                index = self._assign_lhs_variable(syntactic_index, iterator_d_var,
                                rhs=start, new_expressions=new_expr,
                                is_augassign=False)
            iterable.set_loop_counter(index)

        if isinstance(iterator, PyccelSymbol):
            iterator_rhs = iterable.get_target_from_range()
            iterator_d_var = self._infer_type(iterator_rhs)

            target = self._assign_lhs_variable(iterator, iterator_d_var,
                            rhs=iterator_rhs, new_expressions=new_expr,
                            is_augassign=False)

        elif isinstance(iterator, PythonTuple):
            iterator_rhs = iterable.get_target_from_range()
            target = [self._assign_lhs_variable(it, self._infer_type(rhs),
                                rhs=rhs, new_expressions=new_expr,
                                is_augassign=False)
                        for it, rhs in zip(iterator, iterator_rhs)]
        else:

            errors.report(INVALID_FOR_ITERABLE, symbol=expr.target,
                   bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                   severity='error')

        body = self._visit(expr.body)

        self.exit_loop_scope()

        if isinstance(iterable.iterable, Product):
            for_expr = body
            scopes = self.scope.create_product_loop_scope(scope, len(target))

            for t, i, r, s in zip(target[::-1], iterable.loop_counters[::-1], iterable.get_target_from_range()[::-1], scopes[::-1]):
                # Create Variable iterable
                loop_iter = Iterable(r.base)
                loop_iter.set_loop_counter(i)

                # Create a For loop for each level of the Product
                for_expr = For(t, loop_iter, for_expr, scope=s)
                for_expr.end_annotation = expr.end_annotation
                for_expr = [for_expr]
            for_expr = for_expr[0]
        else:
            for_expr = For(target, iterable, body, scope=scope)
            for_expr.end_annotation = expr.end_annotation
        return for_expr


    def _visit_FunctionalFor(self, expr):
        old_index   = expr.index
        new_index   = self.scope.get_new_name()
        expr.substitute(old_index, new_index)

        target  = expr.expr
        index   = new_index
        indices = [self.scope.get_expected_name(i) for i in expr.indices]
        dims    = []
        body    = expr.loops[1]

        idx_subs = {}
        #scope = self.create_new_loop_scope()

        # The symbols created to represent unknown valued objects are temporary
        tmp_used_names = self.scope.all_used_symbols.copy()
        i = 0
        while isinstance(body, For):

            stop  = None
            start = LiteralInteger(0)
            step  = LiteralInteger(1)
            var   = indices[i]
            i += 1
            a     = self._visit(body.iterable)
            if isinstance(a, PythonRange):
                var   = self._create_variable(var, PythonNativeInt(), start, {})
                dvar  = self._infer_type(var)
                stop  = a.stop
                start = a.start
                step  = a.step

            elif isinstance(a, (PythonZip, PythonEnumerate)):
                dvar  = self._infer_type(a.element)
                class_type = dvar.pop('class_type')
                if class_type.rank > 0:
                    class_type = class_type.switch_rank(class_type.rank-1)
                    dvar['shape'] = (dvar['shape'])[1:]
                if class_type.rank == 0:
                    dvar['shape'] = None
                    dvar['memory_handling'] = 'stack'
                var  = Variable(class_type, var, **dvar)
                stop = a.element.shape[0]

            elif isinstance(a, Variable):
                dvar  = self._infer_type(a)
                class_type = dvar.pop('class_type')
                if class_type.rank > 0:
                    class_type = class_type.switch_rank(class_type.rank-1)
                    dvar['shape'] = (dvar['shape'])[1:]
                if class_type.rank == 0:
                    dvar['shape'] = None
                    dvar['memory_handling'] = 'stack'
                var  = Variable(class_type, var, **dvar)
                stop = a.shape[0]

            else:
                errors.report(PYCCEL_RESTRICTION_TODO,
                              bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                              severity='fatal')
            existing_var = self.scope.find(var.name, 'variables')
            if existing_var:
                if self._infer_type(existing_var) != dvar:
                    errors.report(f"Variable {var} already exists with different type",
                            symbol = expr, severity='error')
            else:
                self.scope.insert_variable(var)
            step.invalidate_node()
            step  = pyccel_to_sympy(step , idx_subs, tmp_used_names)
            start.invalidate_node()
            start = pyccel_to_sympy(start, idx_subs, tmp_used_names)
            stop.invalidate_node()
            stop  = pyccel_to_sympy(stop , idx_subs, tmp_used_names)
            size = (stop - start) / step
            if (step != 1):
                size = ceiling(size)

            body = body.body.body[0]
            dims.append((size, step, start, stop))

        # we now calculate the size of the array which will be allocated

        for idx in indices:
            var = self.get_variable(idx)
            idx_subs[idx] = var


        sp_indices  = [sp_Symbol(i) for i in indices]

        dim = sp_Integer(1)

        for i in reversed(range(len(dims))):
            size  = dims[i][0]
            step  = dims[i][1]
            start = dims[i][2]
            stop  = dims[i][3]

            # For complicated cases we must ensure that the upper bound is never smaller than the
            # lower bound as this leads to too little memory being allocated
            min_size = size
            # Collect all uses of other indices
            start_idx = [-1] + [sp_indices.index(a) for a in start.atoms(sp_Symbol) if a in sp_indices]
            stop_idx  = [-1] + [sp_indices.index(a) for a in  stop.atoms(sp_Symbol) if a in sp_indices]
            start_idx.sort()
            stop_idx.sort()

            # Find the minimum size
            while max(len(start_idx),len(stop_idx))>1:
                # Use the maximum value of the start
                if start_idx[-1] > stop_idx[-1]:
                    s = start_idx.pop()
                    min_size = min_size.subs(sp_indices[s], dims[s][3])
                # and the minimum value of the stop
                else:
                    s = stop_idx.pop()
                    min_size = min_size.subs(sp_indices[s], dims[s][2])

            # While the min_size is not a known integer, assume that the bounds are positive
            j = 0
            while not isinstance(min_size, sp_Integer) and j<=i:
                min_size = min_size.subs(dims[j][3]-dims[j][2], 1).simplify()
                j+=1
            # If the min_size is negative then the size will be wrong and an error is raised
            if isinstance(min_size, sp_Integer) and min_size < 0:
                errors.report(PYCCEL_RESTRICTION_LIST_COMPREHENSION_LIMITS.format(indices[i]),
                          bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                          severity='error')

            # sympy is necessary to carry out the summation
            dim   = dim.subs(sp_indices[i], start+step*sp_indices[i])
            dim   = Summation(dim, (sp_indices[i], 0, size-1))
            dim   = dim.doit()

        try:
            dim = sympy_to_pyccel(dim, idx_subs)
        except TypeError:
            errors.report(PYCCEL_RESTRICTION_LIST_COMPREHENSION_SIZE + f'\n Deduced size : {dim}',
                          bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                          severity='fatal')

        # TODO find a faster way to calculate dim
        # when step>1 and not isinstance(dim, Sum)
        # maybe use the c++ library of sympy

        # we annotate the target to infere the type of the list created

        target = self._visit(target)
        d_var = self._infer_type(target)

        class_type = d_var['class_type']

        if class_type is GenericType():
            errors.report(LIST_OF_TUPLES,
                          bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                          severity='fatal')

        d_var['memory_handling'] = 'heap'
        class_type = HomogeneousListType(class_type)
        d_var['class_type'] = class_type
        shape = [dim]
        if d_var['shape']:
            shape.extend(d_var['shape'])
        d_var['shape'] = shape
        d_var['cls_base'] = get_cls_base(class_type)

        # ...
        # TODO [YG, 30.10.2020]:
        #  - Check if we should allow the possibility that is_stack_array=True
        # ...
        lhs_symbol = expr.lhs.base
        ne = []
        lhs = self._assign_lhs_variable(lhs_symbol, d_var, rhs=expr, new_expressions=ne, is_augassign=False)
        lhs_alloc = ne[0]

        if isinstance(target, PythonTuple) and not target.is_homogeneous:
            errors.report(LIST_OF_TUPLES, symbol=expr, severity='error')

        target.invalidate_node()

        loops = [self._visit(i) for i in expr.loops]
        index = self._visit(index)

        l = loops[-1]
        for idx in indices:
            assert isinstance(l, For)
            # Sub in indices as defined here for coherent naming
            if idx.is_temp:
                self.scope.remove_variable(l.target)
                l.substitute(l.target, idx_subs[idx])
            l = l.body.body[-1]

        #self.exit_loop_scope()

        return CodeBlock([lhs_alloc, FunctionalFor(loops, lhs=lhs, indices=indices, index=index)])

    def _visit_GeneratorComprehension(self, expr):
        lhs = self.check_for_variable(expr.lhs)
        if lhs is None:
            pyccel_stage.set_stage('syntactic')
            if expr.lhs.is_temp:
                lhs = PyccelSymbol(self.scope.get_new_name(), is_temp=True)
            else:
                lhs = expr.lhs
            syntactic_assign = Assign(lhs, expr, python_ast=expr.python_ast)
            pyccel_stage.set_stage('semantic')

            creation = self._visit(syntactic_assign)
            self._additional_exprs[-1].append(creation)
            return self.get_variable(lhs)
        else:
            return lhs

    def _visit_While(self, expr):

        scope = self.create_new_loop_scope()
        test = self._visit(expr.test)
        body = self._visit(expr.body)
        self.exit_loop_scope()

        return While(test, body, scope=scope)

    def _visit_IfSection(self, expr):
        condition = expr.condition

        cond = self._visit(expr.condition)

        body = self._visit(expr.body)

        return IfSection(cond, body)

    def _visit_If(self, expr):
        args = [self._visit(i) for i in expr.blocks]

        conds = [b.condition for b in args]

        allocations = [arg.get_attribute_nodes(Allocate) for arg in args]

        var_shapes = [{a.variable : a.shape for a in allocs} for allocs in allocations]
        variables = [v for branch in var_shapes for v in branch]

        for v in variables:
            all_shapes_set = all(v in branch_shapes.keys() for branch_shapes in var_shapes)
            if all_shapes_set:
                shape_branch1 = var_shapes[0][v]
                same_shapes = all(shape_branch1==branch_shapes[v] \
                                for branch_shapes in var_shapes[1:])
            else:
                same_shapes = False

            if not same_shapes:
                v.set_changeable_shape()

        return If(*args)

    def _visit_IfTernaryOperator(self, expr):
        value_true  = self._visit(expr.value_true)
        if value_true.rank > 0 or value_true.dtype is StringType():
            lhs = PyccelSymbol(self.scope.get_new_name(), is_temp=True)
            # Temporarily deactivate type checks to construct syntactic assigns
            pyccel_stage.set_stage('syntactic')
            assign_true  = Assign(lhs, expr.value_true, python_ast = expr.python_ast)
            assign_false = Assign(lhs, expr.value_false, python_ast = expr.python_ast)
            pyccel_stage.set_stage('semantic')

            cond  = self._visit(expr.cond)
            true_section  = IfSection(cond, [self._visit(assign_true)])
            false_section = IfSection(LiteralTrue(), [self._visit(assign_false)])
            self._additional_exprs[-1].append(If(true_section, false_section))

            return self._visit(lhs)
        else:
            cond        = self._visit(expr.cond)
            value_false = self._visit(expr.value_false)
            return IfTernaryOperator(cond, value_true, value_false)

    def _visit_FunctionHeader(self, expr):
        warnings.warn("Support for specifying types via headers will be removed in a " +
                      "future version of Pyccel. Please use type hints. The @template " +
                      "decorator can be used to specify multiple types. See the " +
                      "documentation at " +
                      "https://github.com/pyccel/pyccel/blob/devel/docs/quickstart.md#type-annotations " +
                      "for examples.", FutureWarning)
        # TODO should we return it and keep it in the AST?
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        self.scope.insert_header(expr)
        return expr

    def _visit_Template(self, expr):
        warnings.warn("Support for specifying templates via headers will be removed in " +
                      "a future version of Pyccel. Please use the @template decorator. " +
                      "See the documentatiosn at " +
                      "https://github.com/pyccel/pyccel/blob/devel/docs/templates.md " +
                      "for examples.", FutureWarning)
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        self.scope.insert_template(expr)
        return expr

    def _visit_Return(self, expr):

        results     = expr.expr
        f_name      = self._current_function
        if isinstance(f_name, DottedName):
            f_name = f_name.name[-1]

        return_objs = self.scope.find(f_name, 'functions').results
        assigns     = []
        for o,r in zip(return_objs, results):
            v = o.var
            if not (isinstance(r, PyccelSymbol) and r == (v.name if isinstance(v, Variable) else v)):
                # Create a syntactic object to visit
                pyccel_stage.set_stage('syntactic')
                if isinstance(v, Variable):
                    v = PyccelSymbol(v.name)
                syntactic_assign = Assign(v, r, python_ast=expr.python_ast)
                pyccel_stage.set_stage('semantic')

                a = self._visit(syntactic_assign)
                assigns.append(a)
                if isinstance(a, ConstructorCall):
                    a.cls_variable.is_temp = False

        results = [self._visit(i.var) for i in return_objs]
        if any(isinstance(i.class_type, InhomogeneousTupleType) for i in results):
            # Extraction of underlying variables is not yet implemented here
            errors.report("Returning tuples is not yet implemented",
                    severity='error', symbol=expr)

        # add the Deallocate node before the Return node and eliminating the Deallocate nodes
        # the arrays that will be returned.
        self._check_pointer_targets(results)
        code = assigns + [Deallocate(i) for i in self._allocs[-1] if i not in results]
        if code:
            expr  = Return(results, CodeBlock(code))
        else:
            expr  = Return(results)
        return expr

    def _visit_FunctionDef(self, expr, function_call_args=None):
        """
        Annotate the FunctionDef if necessary.

        The FunctionDef is only annotated if the flag annotate is set to True.
        In the case of an inlined function, we always annotate the function partially,
        depending on the function call if it is an interface, otherwise we annotate it
        if the function_call argument are compatible with the FunctionDef arguments.
        In the case of non inlined function, we only pass through this method
        twice, the first time we do nothing and the second time we annotate all of functions.

        Parameter
        ---------
        expr : FunctionDef|Interface
           The node that needs to be annotated.
           If we provide an Interface, this means that the function has been annotated partially,
           and we need to continue annotating the needed ones.

        function_call_args : list[FunctionCallArgument], optional
            The list of call arguments, needed only in the case of an inlined function.
        """
        if expr.get_direct_user_nodes(lambda u: isinstance(u, CodeBlock)):
            errors.report("Functions can only be declared in modules or inside other functions.",
                    symbol=expr, severity='error')

        existing_semantic_funcs = []
        if not expr.is_semantic:
            self.scope.functions.pop(self.scope.get_expected_name(expr.name), None)
        elif isinstance(expr, Interface):
            existing_semantic_funcs = [*expr.functions]
            expr                    = expr.syntactic_node

        name               = self.scope.get_expected_name(expr.name)
        decorators         = expr.decorators
        new_semantic_funcs = []
        sub_funcs          = []
        func_interfaces    = []
        docstring          = self._visit(expr.docstring) if expr.docstring else expr.docstring
        is_pure            = expr.is_pure
        is_elemental       = expr.is_elemental
        is_private         = expr.is_private
        is_inline          = expr.is_inline

        if function_call_args is not None:
            assert is_inline
            found_func = False

        current_class = expr.get_direct_user_nodes(lambda u: isinstance(u, ClassDef))
        cls_name = current_class[0].name if current_class else None
        if cls_name:
            bound_class = self.scope.find(cls_name, 'classes', raise_if_missing = True)

        not_used = [d for d in decorators if d not in def_decorators.__all__]
        if len(not_used) >= 1:
            errors.report(UNUSED_DECORATORS, symbol=', '.join(not_used), severity='warning')

        templates = self.scope.find_all('templates')
        if decorators['template']:
            # Load templates dict from decorators dict
            templates.update(decorators['template']['template_dict'])

        for t,v in templates.items():
            templates[t] = UnionTypeAnnotation(*[self._visit(vi) for vi in v])

        def unpack(ann):
            if isinstance(ann, UnionTypeAnnotation):
                return ann.type_list
            else:
                return [ann]

        # Filter out unused templates
        templatable_args = [unpack(a.annotation) for a in expr.arguments if isinstance(a.annotation, (SyntacticTypeAnnotation, UnionTypeAnnotation))]
        arg_annotations = [annot for a in templatable_args for annot in a if isinstance(annot, SyntacticTypeAnnotation)]
        type_names = [a.dtype for a in arg_annotations]
        used_type_names = set(d.base if isinstance(d, IndexedElement) else d for d in type_names)
        templates = {t: v for t,v in templates.items() if t in used_type_names}

        # Create new temparary templates for the arguments with a Union data type.
        pyccel_stage.set_stage('syntactic')
        tmp_templates = {}
        new_expr_args = []
        for a in expr.arguments:
            if isinstance(a.annotation, UnionTypeAnnotation):
                annotation = [aa for a in a.annotation for aa in unpack(a)]
            else:
                annotation = [a.annotation]
            if len(annotation)>1:
                tmp_template_name = a.name + '_' + random_string(12)
                tmp_template_name = self.scope.get_new_name(tmp_template_name)
                tmp_templates[tmp_template_name] = UnionTypeAnnotation(*[self._visit(vi) for vi in annotation])
                dtype_symb = PyccelSymbol(tmp_template_name, is_temp=True)
                dtype_symb = SyntacticTypeAnnotation(dtype_symb)
                var_clone = AnnotatedPyccelSymbol(a.var.name, annotation=dtype_symb, is_temp=a.var.name.is_temp)
                new_expr_args.append(FunctionDefArgument(var_clone, bound_argument=a.bound_argument,
                                        value=a.value, kwonly=a.is_kwonly, annotation=dtype_symb))
            else:
                new_expr_args.append(a)
        pyccel_stage.set_stage('semantic')

        templates.update(tmp_templates)
        template_combinations = list(product(*[v.type_list for v in templates.values()]))
        template_names = list(templates.keys())
        n_templates = len(template_combinations)

        # this for the case of a function without arguments => no headers
        interface_name = name
        interface_counter = 0
        is_interface = n_templates > 1
        annotated_args = [] # collect annotated arguments to check for argument incompatibility errors
        for tmpl_idx in range(n_templates):
            if function_call_args is not None and found_func:
                break

            if is_interface:
                name, _ = self.scope.get_new_incremented_symbol(interface_name, tmpl_idx)

            scope = self.create_new_function_scope(name, decorators = decorators,
                    used_symbols = expr.scope.local_used_symbols.copy(),
                    original_symbols = expr.scope.python_names.copy())

            for n, v in zip(template_names, template_combinations[tmpl_idx]):
                self.scope.insert_symbolic_alias(n, v)
            self.scope.decorators.update(decorators)

            # Here _visit_AnnotatedPyccelSymbol always give us an list of size 1
            # so we flatten the arguments
            arguments = [i for a in new_expr_args for i in self._visit(a)]
            assert len(arguments) == len(expr.arguments)
            arg_dict  = {a.name:a.var for a in arguments}
            annotated_args.append(arguments)
            for n in template_names:
                self.scope.symbolic_alias.pop(n)

            if function_call_args is not None:
                is_compatible = self._check_argument_compatibility(function_call_args, arguments, expr, is_elemental, raise_error=False)
                if not is_compatible:
                    self.exit_function_scope()
                    # remove the new created scope and the function name
                    self.scope.sons_scopes.pop(name)
                    if is_interface:
                        self.scope.remove_symbol(name)
                    continue
                #In the case of an Interface we set found_func to True so that we don't continue
                #searching for the other functions
                found_func = True

            for a in arguments:
                a_var = a.var
                if isinstance(a_var, FunctionAddress):
                    self.insert_function(a_var)
                else:
                    self.scope.insert_variable(a_var, expr.scope.get_python_name(a.name))

            if arguments and arguments[0].bound_argument:
                if arguments[0].var.cls_base.name != cls_name:
                    errors.report('Class method self argument does not have the expected type',
                            severity='error', symbol=arguments[0])
                for s in expr.scope.dotted_symbols:
                    base = s.name[0]
                    if base in arg_dict:
                        cls_base = arg_dict[base].cls_base
                        cls_base.scope.insert_symbol(DottedName(*s.name[1:]))

            results = expr.results
            if results and results[0].annotation:
                results = [self._visit(r) for r in expr.results]

            # insert the FunctionDef into the scope
            # to handle the case of a recursive function
            # TODO improve in the case of an interface
            recursive_func_obj = FunctionDef(name, arguments, results, [])
            self.insert_function(recursive_func_obj)

            # Create a new list that store local variables for each FunctionDef to handle nested functions
            self._allocs.append(set())
            self._pointer_targets.append({})

            import_init_calls = [self._visit(i) for i in expr.imports]

            for f in expr.functions:
                self.insert_function(f)

            # we annotate the body
            body = self._visit(expr.body)
            body.insert2body(*import_init_calls, back=False)

            # Annotate the remaining functions
            sub_funcs = [i for i in self.scope.functions.values() if not i.is_header and\
                        not isinstance(i, (InlineFunctionDef, FunctionAddress)) and \
                        not i.is_semantic]
            for i in sub_funcs:
                self._visit(i)

            # Calling the Garbage collecting,
            # it will add the necessary Deallocate nodes
            # to the body of the function
            body.insert2body(*self._garbage_collector(body))
            self._check_pointer_targets(results)

            results = [self._visit(a) for a in results]

            # Determine local and global variables
            global_vars = list(self.get_variables(self.scope.parent_scope))
            global_vars = [g for g in global_vars if body.is_user_of(g)]

            # get the imports
            imports   = self.scope.imports['imports'].values()
            # Prefer dict to set to preserve order
            imports   = list({imp:None for imp in imports}.keys())

            # remove the FunctionDef from the function scope
            func_     = self.scope.functions.pop(name)
            is_recursive = False
            # check if the function is recursive if it was called on the same scope
            if func_.is_recursive and not is_inline:
                is_recursive = True
            elif func_.is_recursive and is_inline:
                errors.report("Pyccel does not support an inlined recursive function", symbol=expr,
                        severity='fatal')

            sub_funcs = [i for i in self.scope.functions.values() if not i.is_header and not isinstance(i, FunctionAddress)]

            func_args = [i for i in self.scope.functions.values() if isinstance(i, FunctionAddress)]
            if func_args:
                func_interfaces.append(Interface('', func_args, is_argument = True))

            namespace_imports = self.scope.imports
            self.exit_function_scope()

            results_names = [i.var.name for i in results]

            # Find all nodes which can modify variables
            assigns = body.get_attribute_nodes(Assign, excluded_nodes = (FunctionCall,))
            calls   = body.get_attribute_nodes(FunctionCall)

            # Collect the modified objects
            lhs_assigns   = [a.lhs for a in assigns]
            modified_args = [call_arg.value for f in calls
                                for call_arg, func_arg in zip(f.args, f.funcdef.arguments) if func_arg.inout]
            # Collect modified variables
            all_assigned = [v for a in (lhs_assigns + modified_args) for v in
                            (a.get_attribute_nodes(Variable) if not isinstance(a, Variable) else [a])]

            # ... computing inout arguments
            for a in arguments:
                if a.name not in chain(results_names, ['self']) and a.var not in all_assigned:
                    a.make_const()
            # ...
            # Raise an error if one of the return arguments is an alias.
            pointer_targets = self._pointer_targets.pop()
            result_pointer_map = {}
            for r in results:
                t = pointer_targets.get(r.var, ())
                if r.var.is_alias:
                    persistent_targets = []
                    for target, _ in t:
                        target_argument_index = next((i for i,a in enumerate(arguments) if a.var == target), -1)
                        if target_argument_index != -1:
                            persistent_targets.append(target_argument_index)
                    if not persistent_targets:
                        errors.report(UNSUPPORTED_POINTER_RETURN_VALUE,
                            symbol=r, severity='error',
                            bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset))
                    else:
                        result_pointer_map[r.var] = persistent_targets

            optional_inits = []
            for a in arguments:
                var = self._optional_params.pop(a.var, None)
                if var:
                    optional_inits.append(If(IfSection(PyccelIsNot(a.var, Nil()),
                                                       [Assign(var, a.var)])))
            body.insert2body(*optional_inits, back=False)

            func_kwargs = {
                    'global_vars':global_vars,
                    'is_pure':is_pure,
                    'is_elemental':is_elemental,
                    'is_private':is_private,
                    'imports':imports,
                    'decorators':decorators,
                    'is_recursive':is_recursive,
                    'functions': sub_funcs,
                    'interfaces': func_interfaces,
                    'result_pointer_map': result_pointer_map,
                    'docstring': docstring,
                    'scope': scope,
            }
            if is_inline:
                func_kwargs['namespace_imports'] = namespace_imports
                global_funcs = [f for f in body.get_attribute_nodes(FunctionDef) if self.scope.find(f.name, 'functions')]
                func_kwargs['global_funcs'] = global_funcs
                cls = InlineFunctionDef
            else:
                cls = FunctionDef
            func = cls(name,
                    arguments,
                    results,
                    body,
                    **func_kwargs)
            if not is_recursive:
                recursive_func_obj.invalidate_node()

            if cls_name:
                # update the class methods
                if not is_interface:
                    bound_class.add_new_method(func)

            new_semantic_funcs += [func]
            if expr.python_ast:
                func.set_current_ast(expr.python_ast)

        if function_call_args is not None and len(new_semantic_funcs) == 0:
            for args in annotated_args[:-1]:
                #raise errors if we do not find any compatible function def
                self._check_argument_compatibility(function_call_args, args, expr, is_elemental, error_type='error')
            self._check_argument_compatibility(function_call_args, annotated_args[-1], expr, is_elemental, error_type='fatal')

        if existing_semantic_funcs:
            new_semantic_funcs = existing_semantic_funcs + new_semantic_funcs

        if len(new_semantic_funcs) == 1 and not is_interface:
            new_semantic_funcs = new_semantic_funcs[0]
            self.insert_function(new_semantic_funcs)
        else:
            for f in new_semantic_funcs:
                self.insert_function(f)

            new_semantic_funcs = Interface(interface_name, new_semantic_funcs, syntactic_node=expr)
            if expr.python_ast:
                new_semantic_funcs.set_current_ast(expr.python_ast)
            if cls_name:
                bound_class.add_new_interface(new_semantic_funcs)
            self.insert_function(new_semantic_funcs)

        return EmptyNode()

    def _visit_PythonPrint(self, expr):
        args = [self._visit(i) for i in expr.expr]
        if len(args) == 0:
            return PythonPrint(args)

        def is_symbolic(var):
            return isinstance(var, Variable) \
                and isinstance(var.dtype, SymbolicType)

        if any(isinstance(a.value.class_type, InhomogeneousTupleType) for a in args):
            new_args = []
            for a in args:
                val = a.value
                if isinstance(val.class_type, InhomogeneousTupleType):
                    assert not a.has_keyword
                    if isinstance(val, FunctionCall):
                        pyccel_stage.set_stage('syntactic')
                        tmp_var = PyccelSymbol(self.scope.get_new_name())
                        assign = Assign(tmp_var, val)
                        assign.set_current_ast(expr.python_ast)
                        pyccel_stage.set_stage('semantic')
                        self._additional_exprs[-1].append(self._visit(assign))
                        val.remove_user_node(assign)
                        val = self._visit(tmp_var)
                    new_args.append(FunctionCallArgument(self.create_tuple_of_inhomogeneous_elements(val)))
                else:
                    new_args.append(a)

            args = new_args

        # TODO fix: not yet working because of mpi examples
#        if not test:
#            # TODO: Add description to parser/messages.py
#            errors.report('Either all arguments must be symbolic or none of them can be',
#                   bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
#                   severity='fatal')

        if is_symbolic(args[0]):
            _args = []
            for a in args:
                f = self.scope.find(a.name, 'symbolic_functions')
                if f is None:
                    _args.append(a)
                else:

                    # TODO improve: how can we print SymbolicAssign as  lhs = rhs

                    _args.append(f)
            return SymbolicPrint(_args)
        else:
            return PythonPrint(args)

    def _visit_ClassDef(self, expr):
        # TODO - improve the use and def of interfaces
        #      - wouldn't be better if it is done inside ClassDef?

        if expr.get_direct_user_nodes(lambda u: isinstance(u, CodeBlock)):
            errors.report("Classes can only be declared in modules.",
                    symbol=expr, severity='error')

        name = self.scope.get_expected_name(expr.name)

        #  create a new Datatype for the current class
        dtype = DataTypeFactory(name)()
        typenames_to_dtypes[name] = dtype
        self.scope.cls_constructs[name] = dtype

        parent = self._find_superclasses(expr)

        scope = self.create_new_class_scope(name, used_symbols=expr.scope.local_used_symbols,
                    original_symbols = expr.scope.python_names.copy())

        attribute_annotations = [self._visit(a) for a in expr.attributes]
        attributes = []
        for a in attribute_annotations:
            if len(a) != 1:
                errors.report(f"Couldn't determine type of {a}",
                        severity='error', symbol=a)
            else:
                v = a[0]
                scope.insert_variable(v)
                attributes.append(v)

        docstring = self._visit(expr.docstring) if expr.docstring else expr.docstring

        cls = ClassDef(name, attributes, [], superclasses=parent, scope=scope,
                docstring = docstring, class_type = dtype)
        self.scope.parent_scope.insert_class(cls)

        methods = list(expr.methods)
        init_func = None

        if not any(method.name == '__init__' for method in methods):
            argument = FunctionDefArgument(Variable(dtype, 'self', cls_base = cls), bound_argument = True)
            self.scope.insert_symbol('__init__')
            scope = self.create_new_function_scope('__init__')
            init_func = FunctionDef('__init__', [argument], (), [], cls_name=cls.name, scope=scope)
            self.exit_function_scope()
            self.insert_function(init_func)
            cls.add_new_method(init_func)
            methods.append(init_func)

        for (i, method) in enumerate(methods):
            m_name = method.name
            if m_name == '__init__':
                if init_func is None:
                    self._visit(method)
                    init_func = self.scope.functions.pop(m_name)

                if isinstance(init_func, Interface):
                    errors.report("Pyccel does not support interface constructor", symbol=method,
                        severity='fatal')
                methods.pop(i)

                # create a new attribute to check allocation
                deallocater_lhs = Variable(dtype, 'self', cls_base = cls, is_argument=True)
                deallocater = DottedVariable(lhs = deallocater_lhs, name = self.scope.get_new_name('is_freed'),
                                             class_type = PythonNativeBool(), is_private=True)
                cls.add_new_attribute(deallocater)
                deallocater_assign = Assign(deallocater, LiteralFalse())
                init_func.body.insert2body(deallocater_assign, back=False)
                break

        if not init_func:
            errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                   bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                   severity='error')

        for i in methods:
            self._visit(i)

        if not any(method.name == '__del__' for method in methods):
            argument = FunctionDefArgument(Variable(dtype, 'self', cls_base = cls), bound_argument = True)
            self.scope.insert_symbol('__del__')
            scope = self.create_new_function_scope('__del__')
            del_method = FunctionDef('__del__', [argument], (), [Pass()], scope=scope)
            self.exit_function_scope()
            self.insert_function(del_method)
            cls.add_new_method(del_method)

        for method in cls.methods:
            if method.name == '__del__':
                self._current_function = method.name
                attribute = []
                for attr in cls.attributes:
                    if not attr.on_stack:
                        attribute.append(attr)
                    elif isinstance(attr.class_type, CustomDataType) and not attr.is_alias:
                        attribute.append(attr)
                if attribute:
                    # Create a new list that store local attributes
                    self._allocs.append(set())
                    self._pointer_targets.append({})
                    self._allocs[-1].update(attribute)
                    method.body.insert2body(*self._garbage_collector(method.body))
                    self._pointer_targets.pop()
                condition = If(IfSection(PyccelNot(deallocater),
                                [method.body]+[Assign(deallocater, LiteralTrue())]))
                method.body = [condition]
                self._current_function = None
                break

        self.exit_class_scope()

        return EmptyNode()

    def _visit_Del(self, expr):

        ls = [Deallocate(self._visit(i)) for i in expr.variables]
        return Del(ls)

    def _visit_PyccelIs(self, expr):
        # Handles PyccelIs and PyccelIsNot
        IsClass = type(expr)

        # TODO ERROR wrong position ??

        var1 = self._visit(expr.lhs)
        var2 = self._visit(expr.rhs)

        if (var1 is var2) or (isinstance(var2, Nil) and isinstance(var1, Nil)):
            if IsClass == PyccelIsNot:
                return LiteralFalse()
            elif IsClass == PyccelIs:
                return LiteralTrue()

        if isinstance(var1, Nil):
            var1, var2 = var2, var1

        if isinstance(var2, Nil):
            if not isinstance(var1, Variable):
                if IsClass == PyccelIsNot:
                    return LiteralTrue()
                elif IsClass == PyccelIs:
                    return LiteralFalse()
            elif not var1.is_optional:
                errors.report(PYCCEL_RESTRICTION_OPTIONAL_NONE,
                        bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                        severity='error')
            return IsClass(var1, expr.rhs)

        if (var1.dtype != var2.dtype):
            if IsClass == PyccelIs:
                return LiteralFalse()
            elif IsClass == PyccelIsNot:
                return LiteralTrue()

        if (isinstance(var1.dtype, PythonNativeBool) and
            isinstance(var2.dtype, PythonNativeBool)):
            return IsClass(var1, var2)

        if isinstance(var1.class_type, (StringType, FixedSizeNumericType)):
            errors.report(PYCCEL_RESTRICTION_PRIMITIVE_IMMUTABLE, symbol=expr,
                severity='error')
            return IsClass(var1, var2)

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
            symbol=expr, severity='error')
        return IsClass(var1, var2)

    def _visit_Import(self, expr):

        # TODO - must have a dict where to store things that have been
        #        imported
        #      - should not use scope

        if expr.get_direct_user_nodes(lambda u: isinstance(u, CodeBlock)):
            errors.report("Imports can only be used in modules or inside functions.",
                    symbol=expr, severity='error')

        container = self.scope.imports

        result = EmptyNode()

        if isinstance(expr.source, AsName):
            source        = expr.source.name
            source_target = expr.source.local_alias
        else:
            source        = str(expr.source)
            source_target = source

        if source in pyccel_builtin_import_registry:
            imports = pyccel_builtin_import(expr)

            def _insert_obj(location, target, obj):
                F = self.scope.find(target)

                if obj is F:
                    errors.report(FOUND_DUPLICATED_IMPORT,
                                symbol=target, severity='warning')
                elif F is None or isinstance(F, dict):
                    container[location][target] = obj
                else:
                    errors.report(IMPORTING_EXISTING_IDENTIFIED, symbol=expr,
                                  severity='fatal')

            if expr.target:
                for t in expr.target:
                    t_name = t.name if isinstance(t, AsName) else t
                    if t_name not in pyccel_builtin_import_registry[source]:
                        errors.report(f"Function '{t}' from module '{source}' is not currently supported by pyccel",
                                symbol=expr,
                                severity='error')
                for (name, atom) in imports:
                    if not name is None:
                        if isinstance(atom, Decorator):
                            continue
                        elif isinstance(atom, Constant):
                            _insert_obj('variables', name, atom)
                        else:
                            _insert_obj('functions', name, atom)
            else:
                assert len(imports) == 1
                mod = imports[0][1]
                assert isinstance(mod, Module)
                _insert_obj('variables', source_target, mod)

            self.insert_import(source, [AsName(v,n) for n,v in imports], source_target)

        elif recognised_source(source):
            errors.report(f"Module {source} is not currently supported by pyccel",
                    symbol=expr,
                    severity='error')
        else:

            # we need to use str here since source has been defined
            # using repr.
            # TODO shall we improve it?

            p       = self.d_parsers[source_target]
            import_init = p.semantic_parser.ast.init_func if source_target not in container['imports'] else None
            import_free = p.semantic_parser.ast.free_func if source_target not in container['imports'] else None
            if expr.target:
                targets = {i.local_alias if isinstance(i,AsName) else i:None for i in expr.target}
                names = [i.name if isinstance(i,AsName) else i for i in expr.target]

                p_scope = p.scope
                p_imports = p_scope.imports
                entries = ['variables', 'classes', 'functions']
                direct_sons = ((e,getattr(p.scope, e)) for e in entries)
                import_sons = ((e,p_imports[e]) for e in entries)
                for entry, d_son in chain(direct_sons, import_sons):
                    for t,n in zip(targets.keys(),names):
                        if n in d_son:
                            e = d_son[n]
                            if entry == 'functions':
                                container[entry][t] = e.clone(t, is_imported=True)
                                m = e.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
                                container[entry][t].set_current_user_node(m)
                            elif entry == 'variables':
                                container[entry][t] = e.clone(t)
                            else:
                                container[entry][t] = e
                            targets[t] = e
                if None in targets.values():
                    errors.report("Import target {} could not be found",
                            severity="warning", symbol=expr)
                targets = [AsName(v,k) for k,v in targets.items() if v is not None]
            else:
                mod = p.semantic_parser.ast
                container['variables'][source_target] = mod
                targets = [AsName(mod, source_target)]

            self.scope.cls_constructs.update(p.scope.cls_constructs)
            self.scope.macros.update(p.scope.macros)

            # ... meta variables

            # in some cases (blas, lapack and openacc level-0)
            # the import should not appear in the final file
            # all metavars here, will have a prefix and suffix = __
            __ignore_at_import__ = p.metavars.get('ignore_at_import', False)

            # Indicates that the module must be imported with the syntax 'from mod import *'
            __import_all__ = p.metavars.get('import_all', False)

            # Indicates the name of the fortran module containing the functions
            __module_name__ = p.metavars.get('module_name', None)

            if source_target in container['imports']:
                targets.extend(container['imports'][source_target].target)

            if import_init:
                old_name = import_init.name
                new_name = self.scope.get_new_name(old_name)
                targets.append(AsName(import_init, new_name))

                if new_name != old_name:
                    import_init = import_init.clone(new_name)

                result  = FunctionCall(import_init,[],[])

            if import_free:
                old_name = import_free.name
                new_name = self.scope.get_new_name(old_name)
                targets.append(AsName(import_free, new_name))

                if new_name != old_name:
                    import_free = import_free.clone(new_name)

            mod = p.semantic_parser.ast

            if __import_all__:
                expr = Import(source_target, AsName(mod, __module_name__), mod=mod)
                container['imports'][source_target] = expr

            elif __module_name__:
                expr = Import(__module_name__, targets, mod=mod)
                container['imports'][source_target] = expr

            elif not __ignore_at_import__:
                expr = Import(source, targets, mod=mod)
                container['imports'][source_target] = expr

        return result



    def _visit_With(self, expr):
        scope = self.create_new_loop_scope()

        domaine = self._visit(expr.test)
        parent  = domaine.cls_base
        if not parent.is_with_construct:
            errors.report(UNDEFINED_WITH_ACCESS, symbol=expr,
                   severity='fatal')

        body = self._visit(expr.body)

        self.exit_loop_scope()
        return With(domaine, body, scope).block



    def _visit_MacroFunction(self, expr):
        # we change here the master name to its FunctionDef

        f_name = expr.master
        header = self.get_headers(f_name)
        if not header:
            func = self.scope.find(f_name, 'functions')
            if func is None:
                errors.report(MACRO_MISSING_HEADER_OR_FUNC,
                    symbol=f_name,severity='error',
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset))
        else:
            interfaces = []
            for hd in header:
                for i,_ in enumerate(hd.dtypes):
                    self.scope.insert_symbol(f'arg_{i}')
                pyccel_stage.set_stage('syntactic')
                syntactic_args = [AnnotatedPyccelSymbol(f'arg_{i}', annotation = arg) \
                        for i, arg in enumerate(hd.dtypes)]
                syntactic_results = [AnnotatedPyccelSymbol(f'out_{i}', annotation = arg) \
                        for i, arg in enumerate(hd.results)]
                pyccel_stage.set_stage('semantic')

                arguments = [FunctionDefArgument(self._visit(a)[0]) for a in syntactic_args]
                results = [FunctionDefResult(self._visit(r)[0]) for r in syntactic_results]
                interfaces.append(FunctionDef(f_name, arguments, results, []))

            # TODO -> Said: must handle interface

            func = interfaces[0]

        name = expr.name
        args = [a if isinstance(a, FunctionDefArgument) else FunctionDefArgument(a) for a in expr.arguments]

        def get_arg(func_arg, master_arg):
            if isinstance(master_arg, PyccelSymbol):
                return FunctionCallArgument(func_arg.var.clone(str(master_arg)))
            else:
                return FunctionCallArgument(master_arg)

        master_args = [get_arg(a,m) for a,m in zip(func.arguments, expr.master_arguments)]

        master = FunctionCall(func, master_args)
        macro   = MacroFunction(name, args, master, master_args,
                                results=expr.results, results_shapes=expr.results_shapes)
        self.scope.insert_macro(macro)

        return macro

    def _visit_MacroShape(self, expr):
        expr.clear_syntactic_user_nodes()
        expr.update_pyccel_staging()
        return expr

    def _visit_MacroVariable(self, expr):

        master = expr.master
        if isinstance(master, DottedName):
            errors.report(PYCCEL_RESTRICTION_TODO,
                          bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                          severity='fatal')
        header = self.get_headers(master)
        if header is None:
            var = self.get_variable(master)
        else:
            var = self.get_variable(master)

                # TODO -> Said: must handle interface

        expr = MacroVariable(expr.name, var)
        self.scope.insert_macro(expr)
        return expr

    def _visit_StarredArguments(self, expr):
        var = self._visit(expr.args_var)
        assert var.rank==1
        size = var.shape[0]
        return StarredArguments([var[i] for i in range(size)])

    def _visit_NumpyMatmul(self, expr):
        self.insert_import('numpy', AsName(NumpyMatmul, 'matmul'))
        a = self._visit(expr.a)
        b = self._visit(expr.b)
        return NumpyMatmul(a, b)

    def _visit_Assert(self, expr):
        test = self._visit(expr.test)
        return Assert(test)

    def _visit_FunctionDefResult(self, expr):
        var = self._visit(expr.var)
        if isinstance(var, list):
            n_types = len(var)
            if n_types == 0:
                errors.report("Can't deduce type for function definition result.",
                        severity = 'fatal', symbol = expr)
            elif n_types != 1:
                errors.report("The type of the result of a function definition cannot be a union of multiple types.",
                        severity = 'error', symbol = expr)
            var = var[0]
            self.scope.insert_variable(var)
        return FunctionDefResult(var, annotation = expr.annotation)

    #====================================================
    #                 _build functions
    #====================================================

    def _build_NumpyWhere(self, func_call):
        """
        Method for building the node created by a call to `numpy.where`.

        Method for building the node created by a call to `numpy.where`. If only one argument is passed to `numpy.where`
        then it is equivalent to a call to `numpy.nonzero`. The result of a call to `numpy.nonzero`
        is a complex object so there is a `_build_NumpyNonZero` function which must be called.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `numpy.nonzero.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `numpy.nonzero` function.
        """
        func_call_args = self._handle_function_args(func_call.args)
        # expr is a FunctionCall
        args = [a.value for a in func_call_args if not a.has_keyword]
        kwargs = {a.keyword: a.value for a in func_call.args if a.has_keyword}
        nargs = len(args)+len(kwargs)
        if nargs == 1:
            return self._build_NumpyNonZero(func_call)
        return NumpyWhere(*args, **kwargs)

    def _build_NumpyNonZero(self, func_call):
        """
        Method for building the node created by a call to `numpy.nonzero`.

        Method for building the node created by a call to `numpy.nonzero`. The result of a call to `numpy.nonzero`
        is a complex object (tuple of arrays) in order to ensure that the results are correctly saved into the
        correct objects it is therefore important to call `_visit` on any intermediate expressions that are required.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `numpy.nonzero.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `numpy.nonzero` function.
        """
        func_call_args = self._handle_function_args(func_call.args)
        # expr is a FunctionCall
        arg = func_call_args[0].value
        if not isinstance(arg, Variable):
            pyccel_stage.set_stage('syntactic')
            new_symbol = PyccelSymbol(self.scope.get_new_name())
            syntactic_assign = Assign(new_symbol, arg, python_ast=func_call.python_ast)
            pyccel_stage.set_stage('semantic')

            creation = self._visit(syntactic_assign)
            self._additional_exprs[-1].append(creation)
            arg = self._visit(new_symbol)
        return NumpyWhere(arg)

    def _build_ListExtend(self, expr):
        """
        Method to navigate the syntactic DottedName node of an `extend()` call.

        The purpose of this `_build` method is to construct new nodes from a syntactic 
        DottedName node. It checks the type of the iterable passed to `extend()`.
        If the iterable is an instance of `PythonList` or `PythonTuple`, it constructs 
        a CodeBlock node where its body consists of `ListAppend` objects with the 
        elements of the iterable. If not, it attempts to construct a syntactic `For` 
        loop to iterate over the iterable object and append its elements to the list 
        object. Finally, it passes to a `_visit()` call for semantic parsing.

        Parameters
        ----------
        expr : DottedName
            The syntactic DottedName node that represent the call to `.extend()`.

        Returns
        -------
        PyccelAstNode
            CodeBlock or For containing ListAppend objects.
        """
        iterable = expr.name[1].args[0].value

        if isinstance(iterable, (PythonList, PythonTuple)):
            list_variable = self._visit(expr.name[0])
            added_list = self._visit(iterable)
            try:
                store = [ListAppend(list_variable, a) for a in added_list]
            except TypeError as e:
                msg = str(e)
                errors.report(msg, symbol=expr, severity='fatal')
            return CodeBlock(store)
        else:
            pyccel_stage.set_stage('syntactic')
            for_target = self.scope.get_new_name('index')
            arg = FunctionCallArgument(for_target)
            func_call = FunctionCall('append', [arg])
            dotted = DottedName(expr.name[0], func_call)
            lhs = PyccelSymbol('_', is_temp=True)
            assign = Assign(lhs, dotted)
            assign.set_current_ast(expr.python_ast)
            body = CodeBlock([assign])
            for_obj = For(for_target, iterable, body)
            pyccel_stage.set_stage('semantic')
            return self._visit(for_obj)

    def _build_MathSqrt(self, func_call):
        """
        Method for building the node created by a call to `math.sqrt`.

        Method for building the node created by a call to `math.sqrt`. A separate method is needed for
        this because some expressions are simplified. This is notably the case for expressions such as
        `math.sqrt(a**2)`. When `a` is a complex number this expression is equivalent to a call to `math.fabs`.
        The expression is translated to this node. The associated imports therefore need to be inserted into the parser.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `cmath.sqrt`.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.sqrt` function.
        """
        func = self.scope.find(func_call.funcdef, 'functions')
        arg, = self._handle_function_args(func_call.args) #pylint: disable=unbalanced-tuple-unpacking
        if isinstance(arg.value, PyccelMul):
            mul1, mul2 = arg.value.args
            if mul1 is mul2:
                pyccel_stage.set_stage('syntactic')

                fabs_name = self.scope.get_new_name('fabs')
                imp_name = AsName('fabs', fabs_name)
                new_import = Import('math',imp_name)
                new_call = FunctionCall(fabs_name, [mul1])

                pyccel_stage.set_stage('semantic')

                self._visit(new_import)

                return self._visit(new_call)
        elif isinstance(arg.value, PyccelPow):
            base, exponent = arg.value.args
            if exponent == 2:
                pyccel_stage.set_stage('syntactic')

                fabs_name = self.scope.get_new_name('fabs')
                imp_name = AsName('fabs', fabs_name)
                new_import = Import('math',imp_name)
                new_call = FunctionCall(fabs_name, [base])

                pyccel_stage.set_stage('semantic')

                self._visit(new_import)

                return self._visit(new_call)

        return self._handle_function(func_call, func, (arg,), use_build_functions = False)

    def _build_CmathSqrt(self, func_call):
        """
        Method for building the node created by a call to `cmath.sqrt`.

        Method for building the node created by a call to `cmath.sqrt`. A separate method is needed for
        this because some expressions are simplified. This is notably the case for expressions such as
        `cmath.sqrt(a**2)`. When `a` is a complex number this expression is equivalent to a call to `cmath.fabs`.
        The expression is translated to this node. The associated imports therefore need to be inserted into the parser.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `cmath.sqrt`.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.sqrt` function.
        """
        func = self.scope.find(func_call.funcdef, 'functions')
        arg, = self._handle_function_args(func_call.args) #pylint: disable=unbalanced-tuple-unpacking
        if isinstance(arg.value, PyccelMul):
            mul1, mul2 = arg.value.args
            is_abs = False
            if isinstance(mul1, (NumpyConjugate, PythonConjugate)) and mul1.internal_var is mul2:
                is_abs = True
                abs_arg = mul2
            elif isinstance(mul2, (NumpyConjugate, PythonConjugate)) and mul1 is mul2.internal_var:
                is_abs = True
                abs_arg = mul1

            if is_abs:
                pyccel_stage.set_stage('syntactic')

                abs_name = self.scope.get_new_name('abs')
                imp_name = AsName('abs', abs_name)
                new_import = Import('numpy',imp_name)
                new_call = FunctionCall(abs_name, [abs_arg])

                pyccel_stage.set_stage('semantic')

                self._visit(new_import)

                # Cast to preserve final dtype
                return PythonComplex(self._visit(new_call))

        return self._handle_function(func_call, func, (arg,), use_build_functions = False)

    def _build_CmathPolar(self, func_call):
        """
        Method for building the node created by a call to `cmath.polar`.

        Method for building the node created by a call to `cmath.polar`. A separate method is needed for
        this because the function is translated to an expression including calls to `math.sqrt` and
        `math.atan2`. The associated imports therefore need to be inserted into the parser.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `cmath.polar`.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.polar` function.
        """
        arg, = self._handle_function_args(func_call.args) #pylint: disable=unbalanced-tuple-unpacking
        z = arg.value
        x = PythonReal(z)
        y = PythonImag(z)
        x_var = self.scope.get_temporary_variable(z, class_type=PythonNativeFloat())
        y_var = self.scope.get_temporary_variable(z, class_type=PythonNativeFloat())
        self._additional_exprs[-1].append(Assign(x_var, x))
        self._additional_exprs[-1].append(Assign(y_var, y))
        r = MathSqrt(PyccelAdd(PyccelMul(x_var,x_var), PyccelMul(y_var,y_var)))
        t = MathAtan2(y_var, x_var)
        self.insert_import('math', AsName(MathSqrt, 'sqrt'))
        self.insert_import('math', AsName(MathAtan2, 'atan2'))
        return PythonTuple(r,t)

    def _build_CmathRect(self, func_call):
        """
        Method for building the node created by a call to `cmath.rect`.

        Method for building the node created by a call to `cmath.rect`. A separate method is needed for
        this because the function is translated to an expression including calls to `math.cos` and
        `math.sin`. The associated imports therefore need to be inserted into the parser.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `cmath.rect`.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.rect` function.
        """
        arg_r, arg_phi = self._handle_function_args(func_call.args) #pylint: disable=unbalanced-tuple-unpacking
        r = arg_r.value
        phi = arg_phi.value
        x = PyccelMul(r, MathCos(phi))
        y = PyccelMul(r, MathSin(phi))
        self.insert_import('math', AsName(MathCos, 'cos'))
        self.insert_import('math', AsName(MathSin, 'sin'))
        return PyccelAdd(x, PyccelMul(y, LiteralImaginaryUnit()))

    def _build_CmathPhase(self, func_call):
        """
        Method for building the node created by a call to `cmath.phase`.

        Method for building the node created by a call to `cmath.phase`. A separate method is needed for
        this because the function is translated to a call to `math.atan2`. The associated import therefore
        needs to be inserted into the parser.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `cmath.phase`.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.phase` function.
        """
        arg, = self._handle_function_args(func_call.args) #pylint: disable=unbalanced-tuple-unpacking
        var = arg.value
        if not isinstance(var.dtype.primitive_type, PrimitiveComplexType):
            return LiteralFloat(0.0)
        else:
            self.insert_import('math', AsName(MathAtan2, 'atan2'))
            return MathAtan2(PythonImag(var), PythonReal(var))

    def _build_PythonTupleFunction(self, func_call):
        """
        Method for building the node created by a call to `tuple()`.

        Method for building the node created by a call to `tuple()`. A separate method is needed for
        this because inhomogeneous variables can be passed to this function. In order to access the
        underlying variables for the indexed elements access to the scope is required.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `tuple()`.

        Returns
        -------
        PythonTuple
            A node describing the result of a call to the `tuple()` function.
        """
        func_args = self._handle_function_args(func_call.args)
        arg = func_args[0].value
        if isinstance(arg, PythonTuple):
            return arg
        elif isinstance(arg.shape[0], LiteralInteger):
            return PythonTuple(*[self.scope.collect_tuple_element(a) for a in arg])
        else:
            raise TypeError(f"Can't unpack {arg} into a tuple")

    def _build_NumpyArray(self, expr):
        """
        Method for building the node created by a call to `numpy.array`.

        Method for building the node created by a call to `numpy.array`. A separate method is needed for
        this because inhomogeneous variables can be passed to this function. In order to access the
        underlying variables for the indexed elements access to the scope is required.

        Parameters
        ----------
        expr : FunctionCall | DottedName
            The syntactic FunctionCall describing the call to `numpy.array`.
            If `numpy.array` is called via a call to `numpy.copy` then this is a DottedName describing the call.

        Returns
        -------
        NumpyArray
            A node describing the result of a call to the `numpy.array` function.
        """
        if isinstance(expr, DottedName):
            arg = expr.name[0]
            dtype = None
            ndmin = None
            func_call = expr.name[1]
            func = func_call.funcdef
            func_call_args = func_call.args
            order = func_call_args[0].value if func_call_args else func.argument_description['order']
        else:
            func_call_args = self._handle_function_args(expr.args)
            args, kwargs = split_positional_keyword_arguments(*func_call_args)

            def unpack_args(arg, dtype = None, order = 'K', ndmin = None):
                """ Small function to reorder and get access to the named variables from args and kwargs.
                """
                return arg, dtype,  order, ndmin

            arg, dtype,  order, ndmin = unpack_args(*args, **kwargs)

        if not isinstance(arg, (PythonTuple, PythonList, Variable, IndexedElement)):
            errors.report('Unexpected object passed to numpy.array',
                    severity='fatal', symbol=expr)

        is_homogeneous_tuple = isinstance(arg.class_type, HomogeneousTupleType)
        # Inhomogeneous tuples can contain homogeneous data if it is inhomogeneous due to pointers
        if isinstance(arg.class_type, InhomogeneousTupleType):
            is_homogeneous_tuple = isinstance(arg.dtype, FixedSizeNumericType) and len(set(a.rank for a in arg))
            if not isinstance(arg, PythonTuple):
                arg = PythonTuple(*(self.scope.collect_tuple_element(a) for a in arg))

        if not (is_homogeneous_tuple or isinstance(arg.class_type, HomogeneousContainerType)):
            errors.report('Inhomogeneous type passed to numpy.array',
                    severity='fatal', symbol=expr)

        if not isinstance(order, (LiteralString, str)):
            errors.report('Order must be specified with a literal string',
                    severity='fatal', symbol=expr)
        elif isinstance(order, LiteralString):
            order = order.python_value

        if ndmin is not None:
            if not isinstance(ndmin, (LiteralInteger, int)):
                errors.report("The minimum number of dimensions must be specified explicitly with an integer.",
                        severity='fatal', symbol=expr)
            elif isinstance(ndmin, LiteralInteger):
                ndmin = ndmin.python_value


        return NumpyArray(arg, dtype, order, ndmin)

    def _build_SetUpdate(self, expr):
        """
        Method to navigate the syntactic DottedName node of an `update()` call.

        The purpose of this `_build` method is to construct new nodes from a syntactic 
        DottedName node. It checks the type of the iterable passed to `update()`.
        If the iterable is an instance of `PythonList`, `PythonSet` or `PythonTuple`, it constructs 
        a CodeBlock node where its body consists of `SetAdd` objects with the 
        elements of the iterable. If not, it attempts to construct a syntactic `For` 
        loop to iterate over the iterable object and added its elements to the set 
        object. Finally, it passes to a `_visit()` call for semantic parsing.
    
        Parameters
        ----------
        expr : DottedName
            The syntactic DottedName node that represent the call to `.update()`.

        Returns
        -------
        PyccelAstNode
            CodeBlock or For containing SetAdd objects.
        """
        iterable = expr.name[1].args[0].value
        if isinstance(iterable, (PythonList, PythonSet, PythonTuple)):
            list_variable = self._visit(expr.name[0])
            added_list = self._visit(iterable)
            try:
                store = [SetAdd(list_variable, a) for a in added_list]
            except TypeError as e:
                msg = str(e)
                errors.report(msg, symbol=expr, severity='fatal')
            return CodeBlock(store)
        else:
            pyccel_stage.set_stage('syntactic')
            for_target = self.scope.get_new_name()
            arg = FunctionCallArgument(for_target)
            func_call = FunctionCall('add', [arg])
            dotted = DottedName(expr.name[0], func_call)
            lhs = PyccelSymbol('_', is_temp=True)
            assign = Assign(lhs, dotted)
            assign.set_current_ast(expr.python_ast)
            body = CodeBlock([assign])
            for_obj = For(for_target, iterable, body)
            pyccel_stage.set_stage('semantic')
            return self._visit(for_obj)

