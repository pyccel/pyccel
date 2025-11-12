# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" File containing SemanticParser. This class handles the semantic stage of the translation.
See the developer docs for more details
"""

from itertools import chain, product
import os
from types import ModuleType, BuiltinFunctionType
import typing
import warnings

from sympy.utilities.iterables import iterable as sympy_iterable

from sympy import Sum as Summation
from sympy import Symbol as sp_Symbol
from sympy import Integer as sp_Integer
from sympy.logic.boolalg import BooleanTrue as sp_True
from sympy.logic.boolalg import BooleanFalse as sp_False
from sympy import ceiling

from textx.exceptions import TextXSyntaxError

#==============================================================================
from pyccel.ast.basic import PyccelAstNode, TypedAstNode, ScopedAstNode, iterable

from pyccel.ast.bitwise_operators import PyccelBitOr, PyccelLShift, PyccelRShift, PyccelBitAnd

from pyccel.ast.builtins import PythonPrint, PythonTupleFunction, PythonSetFunction
from pyccel.ast.builtins import PythonComplex, PythonDict, PythonListFunction
from pyccel.ast.builtins import builtin_functions_dict, PythonImag, PythonReal
from pyccel.ast.builtins import PythonList, PythonConjugate , PythonSet, VariableIterator
from pyccel.ast.builtins import PythonRange, PythonZip, PythonEnumerate, PythonTuple
from pyccel.ast.builtins import PythonMap, PythonBool

from pyccel.ast.builtin_methods.dict_methods import DictKeys
from pyccel.ast.builtin_methods.list_methods import ListAppend, ListPop, ListInsert
from pyccel.ast.builtin_methods.set_methods  import SetAdd, SetUnion, SetCopy, SetIntersectionUpdate
from pyccel.ast.builtin_methods.set_methods  import SetPop, SetDifferenceUpdate
from pyccel.ast.builtin_methods.dict_methods  import DictGetItem, DictGet, DictPop, DictPopitem

from pyccel.ast.core import Comment, CommentBlock, Pass
from pyccel.ast.core import If, IfSection
from pyccel.ast.core import Allocate, Deallocate
from pyccel.ast.core import Assign, AliasAssign
from pyccel.ast.core import AugAssign, CodeBlock
from pyccel.ast.core import Return, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core import ConstructorCall, InlineFunctionDef
from pyccel.ast.core import FunctionDef, Interface, FunctionAddress, FunctionCall, FunctionCallArgument
from pyccel.ast.core import ClassDef
from pyccel.ast.core import For
from pyccel.ast.core import Module
from pyccel.ast.core import While
from pyccel.ast.core import Del
from pyccel.ast.core import Program
from pyccel.ast.core import EmptyNode
from pyccel.ast.core import Concatenate
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import With
from pyccel.ast.core import Duplicate
from pyccel.ast.core import StarredArguments
from pyccel.ast.core import Decorator
from pyccel.ast.core import PyccelFunctionDef
from pyccel.ast.core import Assert
from pyccel.ast.core import AllDeclaration

from pyccel.ast.class_defs import get_builtin_cls_base, SetClass

from pyccel.ast.datatypes import CustomDataType, PyccelType, TupleType, VoidType, GenericType
from pyccel.ast.datatypes import PrimitiveIntegerType, StringType, SymbolicType
from pyccel.ast.datatypes import PythonNativeBool, PythonNativeInt, PythonNativeFloat
from pyccel.ast.datatypes import DataTypeFactory, HomogeneousContainerType, FinalType
from pyccel.ast.datatypes import InhomogeneousTupleType, HomogeneousTupleType, HomogeneousSetType, HomogeneousListType
from pyccel.ast.datatypes import PrimitiveComplexType, FixedSizeNumericType, DictType, TypeAlias
from pyccel.ast.datatypes import original_type_to_pyccel_type

from pyccel.ast.functionalexpr import FunctionalSum, FunctionalMax, FunctionalMin, GeneratorComprehension, FunctionalFor
from pyccel.ast.functionalexpr import MaxLimit, MinLimit

from pyccel.ast.headers import Header

from pyccel.ast.internals import PyccelFunction, Slice, PyccelSymbol, PyccelArrayShapeElement
from pyccel.ast.internals import Iterable
from pyccel.ast.itertoolsext import Product

from pyccel.ast.literals import LiteralTrue, LiteralFalse
from pyccel.ast.literals import LiteralInteger, LiteralFloat
from pyccel.ast.literals import Nil, LiteralString, LiteralImaginaryUnit
from pyccel.ast.literals import Literal, convert_to_literal, LiteralEllipsis

from pyccel.ast.low_level_tools import MemoryHandlerType, UnpackManagedMemory, ManagedMemory

from pyccel.ast.mathext  import math_constants, MathSqrt, MathAtan2, MathSin, MathCos

from pyccel.ast.numpyext import NumpyMatmul, numpy_funcs
from pyccel.ast.numpyext import NumpyWhere, NumpyArray
from pyccel.ast.numpyext import NumpyTranspose, NumpyConjugate
from pyccel.ast.numpyext import NumpyNewArray, NumpyResultType
from pyccel.ast.numpyext import process_dtype as numpy_process_dtype
from pyccel.ast.numpyext import get_shape_of_multi_level_container

from pyccel.ast.numpytypes import NumpyNDArrayType

from pyccel.ast.omp import (OMP_For_Loop, OMP_Simd_Construct, OMP_Distribute_Construct,
                            OMP_TaskLoop_Construct, OMP_Sections_Construct, Omp_End_Clause,
                            OMP_Single_Construct)

from pyccel.ast.operators import PyccelArithmeticOperator, PyccelIs, PyccelIsNot, IfTernaryOperator, PyccelUnarySub
from pyccel.ast.operators import PyccelNot, PyccelAdd, PyccelMinus, PyccelMul, PyccelPow, PyccelOr
from pyccel.ast.operators import PyccelAssociativeParenthesis, PyccelDiv, PyccelIn, PyccelOperator
from pyccel.ast.operators import PyccelAnd

from pyccel.ast.sympy_helper import sympy_to_pyccel, pyccel_to_sympy

from pyccel.ast.type_annotations import VariableTypeAnnotation, UnionTypeAnnotation, SyntacticTypeAnnotation
from pyccel.ast.type_annotations import FunctionTypeAnnotation, typenames_to_dtypes

from pyccel.ast.typingext import TypingFinal, TypingTypeVar

from pyccel.ast.utilities import builtin_import as pyccel_builtin_import
from pyccel.ast.utilities import builtin_import_registry as pyccel_builtin_import_registry
from pyccel.ast.utilities import split_positional_keyword_arguments
from pyccel.ast.utilities import recognised_source, is_literal_integer, get_managed_memory_object

from pyccel.ast.variable import Constant
from pyccel.ast.variable import Variable
from pyccel.ast.variable import IndexedElement, AnnotatedPyccelSymbol
from pyccel.ast.variable import DottedName, DottedVariable

from pyccel.errors.errors import Errors, ErrorsMode, PyccelError, PyccelSemanticError

from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, UNDERSCORE_NOT_A_THROWAWAY,
        UNDEFINED_VARIABLE, IMPORTING_EXISTING_IDENTIFIED, INDEXED_TUPLE, LIST_OF_TUPLES,
        INVALID_INDICES, INCOMPATIBLE_ARGUMENT,
        UNRECOGNISED_FUNCTION_CALL, STACK_ARRAY_SHAPE_UNPURE_FUNC, STACK_ARRAY_UNKNOWN_SHAPE,
        ARRAY_DEFINITION_IN_LOOP, STACK_ARRAY_DEFINITION_IN_LOOP, MISSING_TYPE_ANNOTATIONS,
        INCOMPATIBLE_TYPES_IN_ASSIGNMENT, TARGET_ALREADY_IN_USE, ASSIGN_ARRAYS_ONE_ANOTHER,
        INVALID_POINTER_REASSIGN, ARRAY_IS_ARG,
        INCOMPATIBLE_REDEFINITION_STACK_ARRAY, ARRAY_REALLOCATION, RECURSIVE_RESULTS_REQUIRED,
        PYCCEL_RESTRICTION_INHOMOG_LIST, UNDEFINED_IMPORT_OBJECT, UNDEFINED_LAMBDA_VARIABLE,
        UNDEFINED_LAMBDA_FUNCTION, UNDEFINED_INIT_METHOD, UNDEFINED_FUNCTION,
        WRONG_NUMBER_OUTPUT_ARGS, INVALID_FOR_ITERABLE,
        PYCCEL_RESTRICTION_LIST_COMPREHENSION_LIMITS, PYCCEL_RESTRICTION_LIST_COMPREHENSION_SIZE,
        UNUSED_DECORATORS, UNSUPPORTED_POINTER_RETURN_VALUE, PYCCEL_RESTRICTION_OPTIONAL_NONE,
        PYCCEL_RESTRICTION_PRIMITIVE_IMMUTABLE, PYCCEL_RESTRICTION_IS_ISNOT,
        FOUND_DUPLICATED_IMPORT, UNDEFINED_WITH_ACCESS,
        PYCCEL_INTERNAL_ERROR)

from pyccel.parser.base      import BasicParser
from pyccel.parser.syntactic import SyntaxParser
from pyccel.parser.syntax.headers import types_meta

from pyccel.utilities.stage import PyccelStage

import pyccel.decorators as def_decorators

#==============================================================================

errors = Errors()
pyccel_stage = PyccelStage()

type_container = {
                   PythonTupleFunction : HomogeneousTupleType,
                   PythonListFunction : HomogeneousListType,
                   PythonSetFunction : HomogeneousSetType,
                   NumpyArray : NumpyNDArrayType,
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

magic_method_map = {
        PyccelAdd: '__add__',
        PyccelMinus: '__sub__',
        PyccelMul: '__mul__',
        PyccelDiv: '__truediv__',
        PyccelPow: '__pow__',
        PyccelLShift: '__lshift__',
        PyccelRShift: '__rshift__',
        PyccelBitAnd : '__and__',
        PyccelBitOr: '__or__',
        }

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

    context_dict : dict, optional
        A dictionary describing any variables in the context where the translated
        objected was defined.

    **kwargs : dict
        Additional keyword arguments for BasicParser.
    """

    def __init__(self, inputs, *, parents = (), d_parsers = (), context_dict = None, **kwargs):

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

        # provides information about the calling context to collect constants
        self._context_dict = context_dict or {}

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

    def create_new_function_scope(self, syntactic_name, semantic_name, **kwargs):
        """
        Create a new Scope object for a Python function.

        Create a new Scope object for a Python function with the given name,
        and attach any decorators' information to the scope. The new scope is
        a child of the current one, and can be accessed from the dictionary of
        its children using the function name as key.

        Before returning control to the caller, the current scope (stored in
        self._scope) is changed to the one just created, and the function's
        name is stored in self._current_function_name.

        Parameters
        ----------
        syntactic_name : str
            Function's original name in the translated code, used as a key to
            retrieve the new scope.

        semantic_name : str
            The new name of the function by which it will be known in the target
            language.

        **kwargs : dict
            Keyword arguments passed through to the new scope.

        Returns
        -------
        Scope
            The new scope for the function.
        """
        child = self.scope.new_child_scope(syntactic_name, 'function', **kwargs)
        child.local_used_symbols[syntactic_name] = semantic_name
        child.python_names[semantic_name] = syntactic_name

        self._scope = child
        self._current_function_name.append(semantic_name)

        return child

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
                class_def = self.get_cls_base(prefix.class_type)

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

    def get_cls_base(self, class_type):
        """
        Determine the base class of an object.

        From the type, determine the base class of an object.

        Parameters
        ----------
        class_type : DataType
            The Python type of the object.

        Returns
        -------
        ClassDef
            A class definition describing the base class of an object.

        Raises
        ------
        NotImplementedError
            Raised if the base class cannot be found.
        """
        # Extract type in case of qualifier (e.g. Final)
        while hasattr(class_type, 'underlying_type'):
            class_type = class_type.underlying_type

        scope_class = self.scope.find(str(class_type), 'classes')

        if scope_class:
            return scope_class
        else:
            return get_builtin_cls_base(class_type)

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
            storage_name = next((n for n, imp in self.scope.find_all('imports').items()
                                if imp.source == source), None)
        imp = self.scope.find(source, 'imports')
        found_from_import_name = False
        if imp is None:
            imp = self.scope.find(storage_name, 'imports')
            found_from_import_name = True

        if imp is not None:
            if found_from_import_name or source in (imp.source, getattr(imp.source_module, 'name', '')):
                imp.define_target(target)
            else:
                errors.report(IMPORTING_EXISTING_IDENTIFIED,
                              symbol=name,
                              bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                              severity='fatal')
        else:
            current_scope = self.scope
            while current_scope.is_loop:
                current_scope = current_scope.parent_scope
            container = current_scope.imports
            container['imports'][storage_name] = Import(source, target, True)


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
        if all(r.expr is None for r in expr.get_attribute_nodes(Return)):
            for i in self._allocs[-1]:
                if isinstance(i, DottedVariable):
                    if isinstance(i.lhs.class_type, CustomDataType) and \
                            self.current_function_name != '__del__':
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
        assert len(self._allocs) == len(self._pointer_targets)
        assert not isinstance(exceptions, Variable)
        for i in self._allocs[-1]:
            if i in exceptions:
                continue
            self._pointer_targets[-1].pop(i, None)
        targets = {t[0]:t[1] for target_list in self._pointer_targets[-1].values() for t in target_list}
        for i in self._allocs[-1]:
            if i in exceptions:
                continue
            if i in targets:
                errors.report(f"Variable {i} goes out of scope but may be the target of a pointer which is still required",
                        severity='error', symbol=targets[i])

        if self.current_function_name:
            current_func = self._current_function[-1]
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
        Indicate that a pointer is targeting a specific target.

        Indicate that a pointer is targeting a specific target by adding the pair
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
        if pointer is target:
            return

        assert pointer != target
        assert not isinstance(pointer.class_type, (StringType, FixedSizeNumericType))

        pointing_at_container_element = (isinstance(pointer.class_type, (HomogeneousSetType, HomogeneousListType)) \
                                        and (target.class_type is pointer.class_type.element_type)) or \
                                        (isinstance(pointer.class_type, DictType) \
                                        and (target.class_type is pointer.class_type.value_type))
        container_pointing_at_element = (isinstance(target.class_type, (HomogeneousSetType, HomogeneousListType)) \
                                        and (pointer.class_type is target.class_type.element_type)) or \
                                        (isinstance(target.class_type, DictType) \
                                        and (pointer.class_type is target.class_type.value_type))

        if pointing_at_container_element or container_pointing_at_element:
            managed_var = target if target.rank < pointer.rank else pointer
            if isinstance(managed_var, Variable):
                managed_mem = managed_var.get_direct_user_nodes(lambda u: isinstance(u, ManagedMemory))
                if not managed_mem:
                    mem_var = Variable(MemoryHandlerType.get_new(managed_var.class_type),
                                       self.scope.get_new_name(f'{managed_var.name}_mem'),
                                       shape=None, memory_handling='heap')
                    self.scope.insert_variable(mem_var)
                    ManagedMemory(managed_var, mem_var)

        # The class itself should also be aware of the target for freeing
        if isinstance(pointer, DottedVariable):
            self._indicate_pointer_target(pointer.lhs, target, expr)

        if isinstance(target, DottedVariable):
            self._indicate_pointer_target(pointer, target.lhs, expr)
        elif isinstance(target, IndexedElement):
            self._indicate_pointer_target(pointer, target.base, expr)
        elif isinstance(target, (DictGetItem, DictGet)):
            self._indicate_pointer_target(pointer, target.dict_obj, expr)
        elif isinstance(target, Variable):
            if target.is_alias:
                sub_targets = None
                try:
                    sub_targets = self._pointer_targets[-1][target]
                except KeyError:
                    errors.report("Pointer cannot point at a non-local pointer\n"+PYCCEL_RESTRICTION_TODO,
                        severity='error', symbol=expr)
                if sub_targets:
                    self._pointer_targets[-1].setdefault(pointer, []).extend((t[0], expr) for t in sub_targets)
            else:
                target.is_target = True
                self._pointer_targets[-1].setdefault(pointer, []).append((target, expr))
        elif isinstance(target, FunctionCall):
            if isinstance(target.funcdef, FunctionDef):
                if target.funcdef.result_pointer_map:
                    raise NotImplementedError("TODO results point at args")
        elif isinstance(target, (PythonList, PythonSet, PythonTuple)):
            if not isinstance(target.class_type.element_type, (StringType, FixedSizeNumericType)):
                for v in target:
                    self._indicate_pointer_target(pointer, v, expr)
        elif isinstance(target, (ListPop, DictPop)):
            target_var = target.list_obj if isinstance(target, ListPop) else target.dict_obj
            if target_var in self._pointer_targets[-1]:
                sub_targets = self._pointer_targets[-1][target_var]
                self._pointer_targets[-1].setdefault(pointer, []).extend((t[0], expr) for t in sub_targets)
            elif isinstance(pointer, Variable):
                self._allocs[-1].add(pointer)
            if isinstance(pointer, Variable):
                managed_mem = pointer.get_direct_user_nodes(lambda u: isinstance(u, ManagedMemory))
                if not managed_mem:
                    mem_var = Variable(MemoryHandlerType.get_new(pointer.class_type),
                                       self.scope.get_new_name(f'{pointer.name}_mem'),
                                       shape=None, memory_handling='heap')
                    self.scope.insert_variable(mem_var)
                    ManagedMemory(pointer, mem_var)
        elif isinstance(target, PythonDict):
            if not isinstance(target.class_type.value_type, (StringType, FixedSizeNumericType)):
                for v in target.values:
                    self._indicate_pointer_target(pointer, v, expr)
        elif isinstance(pointer, Variable) and pointer.is_alias:
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

        class_type = expr.class_type
        if isinstance(class_type, FinalType) and isinstance(class_type, FixedSizeNumericType):
            class_type = class_type.underlying_type

        d_var = {
                'class_type' : class_type,
                'shape'      : expr.shape,
                'cls_base'   : self.get_cls_base(class_type),
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

        elif isinstance(expr, PythonTuple):

            if isinstance(class_type, HomogeneousTupleType):
                d_var['shape'] = get_shape_of_multi_level_container(expr)
            return d_var

        elif isinstance(expr, (DictGetItem, DictGet)):

            d_var['memory_handling'] = 'alias' if not isinstance(class_type, FixedSizeNumericType) else 'stack'
            return d_var

        elif isinstance(expr, TypedAstNode):

            d_var['memory_handling'] = 'heap' if expr.rank > 0 else 'stack'
            return d_var

        else:
            type_name = type(expr).__name__
            msg = f'Type of Object : {type_name} cannot be inferred'
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

        elif isinstance(var, (PyccelFunction, FunctionCall)):
            pyccel_stage.set_stage('syntactic')
            tmp_var = PyccelSymbol(self.scope.get_new_name())
            assign = Assign(tmp_var, var)
            assign.set_current_ast(expr.python_ast)
            pyccel_stage.set_stage('semantic')
            self._additional_exprs[-1].append(self._visit(assign))
            var.remove_user_node(assign)
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
        arg1 = visited_args[0]
        if all(isinstance(a, PyccelFunctionDef) for a in visited_args):
            try:
                possible_types = [a.cls_name.static_type() for a in visited_args]
            except AttributeError:
                errors.report("Unrecognised type in type union statement",
                        severity='fatal', symbol=expr)
            return UnionTypeAnnotation(*[VariableTypeAnnotation(t) for t in possible_types])
        class_type = arg1.class_type
        class_base = self.get_cls_base(class_type)
        magic_method_name = magic_method_map.get(type(expr), None)
        magic_method = None
        if magic_method_name:
            magic_method = class_base.get_method(magic_method_name)
            if magic_method is None:
                arg2 = visited_args[1]
                class_type = arg2.class_type
                class_base = self.get_cls_base(class_type)
                magic_method_name = '__r'+magic_method_name[2:]
                magic_method = class_base.get_method(magic_method_name)
                if magic_method:
                    visited_args = [visited_args[1], visited_args[0]]
        if magic_method:
            expr_new = self._handle_function(expr, magic_method, [FunctionCallArgument(v) for v in visited_args])
        else:
            try:
                expr_new = type(expr)(*visited_args)
            except PyccelSemanticError as err:
                errors.report(str(err), symbol=expr, severity='fatal')
            except TypeError as err:
                types = ', '.join(str(a.class_type) for a in visited_args)
                errors.report(f"Operator {type(expr)} between objects of type ({types}) is not yet handled\n"
                        + PYCCEL_RESTRICTION_TODO, symbol=expr, severity='fatal',
                        traceback = err.__traceback__)
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

    def _create_class_destructor(self, expr):
        """
        Create the class destructor.

        Create the class destructor. This is important to ensure that the data in the
        class is correctly deallocated. If the class already has a destructor then the
        deallocations are added to the existing destructor. Similarly a flag is added
        to the init method to act as a guard in the class to tell if the destructor has
        been called.

        Parameters
        ----------
        expr : ClassDef
            The class that implicit __init__ and __del__ methods should be created for.
        """
        class_type = expr.class_type
        cls_scope = expr.scope

        init_func = cls_scope.functions['__init__']

        if isinstance(init_func, Interface):
            errors.report("Pyccel does not support interface constructor", symbol=init_func,
                severity='fatal')

        # create a new attribute to check allocation
        deallocater_lhs = Variable(class_type, 'self', cls_base = expr, is_argument=True)
        deallocater = DottedVariable(lhs = deallocater_lhs, name = cls_scope.get_new_name('is_freed'),
                                     class_type = PythonNativeBool(), is_private=True)
        expr.add_new_attribute(deallocater)
        deallocater_assign = Assign(deallocater, LiteralFalse())
        init_func.body.insert2body(deallocater_assign, back=False)

        del_method = expr.methods_as_dict.get('__del__', None)
        if del_method is None:
            del_name = cls_scope.insert_symbol('__del__', object_type = 'function')
            scope = self.create_new_function_scope('__del__', del_name)
            argument = FunctionDefArgument(Variable(class_type, scope.get_new_name('self'), cls_base = expr), bound_argument = True)
            scope.insert_variable(argument.var)
            self.exit_function_scope()
            del_method = FunctionDef(del_name, [argument], [Pass()], scope=scope)
            self.insert_function(del_method, cls_scope)
            expr.add_new_method(del_method)
        else:
            assert del_method.is_semantic

        # Add destructors to __del__ method
        self._current_function_name.append(del_method.name)
        attribute = []
        for attr in expr.attributes:
            if not attr.on_stack:
                attribute.append(attr)
            elif isinstance(attr.class_type, CustomDataType) and not attr.is_alias:
                attribute.append(attr)
        if attribute:
            # Create a new list that store local attributes
            self._allocs.append(set())
            self._pointer_targets.append({})
            self._allocs[-1].update(attribute)
            del_method.body.insert2body(*self._garbage_collector(del_method.body))
            self._pointer_targets.pop()
        condition = If(IfSection(PyccelNot(deallocater),
                        [del_method.body]+[Assign(deallocater, LiteralTrue())]))
        del_method.body = [condition]
        self._current_function_name.pop()

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
                semantic_func = self._annotate_the_called_function_def(val, ())
                a = FunctionCallArgument(semantic_func, keyword = a.keyword, python_ast = a.python_ast)

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
                        new_expr.set_current_ast(expr.python_ast or self._current_ast_node)
                        pyccel_stage.set_stage('semantic')
                        for u in expr.get_all_user_nodes():
                            new_expr.set_current_user_node(u)
                        expr = new_expr
                    return getattr(self, annotation_method)(expr, args)

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
            is_inline = func.is_inline if isinstance(func, FunctionDef) else False
            if is_inline:
                return self._visit_InlineFunctionDef(func, args, expr)
            elif not func.is_semantic:
                func = self._annotate_the_called_function_def(func, args)

            if self.current_function_name == func.name:
                if func.results and not isinstance(func.results.var, TypedAstNode):
                    errors.report(RECURSIVE_RESULTS_REQUIRED, symbol=func, severity="fatal")

            parent_assign = expr.get_direct_user_nodes(lambda x: isinstance(x, Assign) and not isinstance(x, AugAssign))

            func_results = func.results if isinstance(func, FunctionDef) else func.functions[0].results
            if not parent_assign and func_results.var.rank > 0:
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

            new_expr = FunctionCall(func, args, self.current_function_name)
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

            if None in args:
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
        n_funcargs = len(func_args)
        n_posonly_args = next((i for i, a in enumerate(func_args) if not a.is_posonly), n_funcargs)
        n_possible_posargs = next((i for i, a in enumerate(func_args) if a.is_vararg), n_funcargs)
        input_args = []
        kwargs = {}
        for i, a in enumerate(args):
            if a.keyword is None:
                if isinstance(a.value, StarredArguments) and not func_args[len(input_args)].is_vararg:
                    input_args.extend(FunctionCallArgument(self.scope.collect_tuple_element(av)) \
                            for av in a.value.args_var)
                else:
                    input_args.append(a)
            else:
                kwargs[a.keyword] = a
        assert len(input_args) >= n_posonly_args
        if len(input_args) > n_possible_posargs:
            if len(input_args) == n_possible_posargs + 1 and isinstance(input_args[-1].value, StarredArguments):
                vararg = FunctionCallArgument(input_args[-1].value.args_var, keyword='*args')
            else:
                remaining_input_args = [ai.value for a in input_args[n_possible_posargs:]
                                        for ai in (a.args_var if isinstance(a, StarredArguments) else (a,))]
                vararg = FunctionCallArgument(PythonTuple(*remaining_input_args), keyword='*args')
            input_args = input_args[:n_possible_posargs] + [vararg]

        for ka in func_args[len(input_args):]:
            if ka.is_vararg:
                input_args.append(FunctionCallArgument(PythonTuple(class_type = ka.var.class_type), keyword='*args'))
                continue
            if not ka.is_kwarg:
                input_args.append(kwargs.pop(ka.name, ka.default_call_arg))

        if kwargs:
            assert func_args[-1].is_kwarg
            keys = [LiteralString(k) for k in kwargs]
            values = [ka.value for ka in kwargs.values()]
            input_args.append(FunctionCallArgument(PythonDict(keys, values), keyword='**kwargs'))

        return input_args

    def _annotate_the_called_function_def(self, old_func, function_call_args):
        """
        Annotate the called FunctionDef.

        Annotate the called FunctionDef.

        Parameters
        ----------
        old_func : FunctionDef|Interface
           The function that needs to be annotated.

        function_call_args : list[FunctionCallArgument]
           The list of the call arguments.

        Returns
        -------
        func: FunctionDef|Interface
            The new annotated function.
        """
        assert not old_func.is_inline
        cls_base_syntactic = old_func.get_direct_user_nodes(lambda p: isinstance(p, ClassDef))
        if cls_base_syntactic:
            cls_name = cls_base_syntactic[0].name
            cls_base = self.scope.find(cls_name, 'classes')
            cls_scope = cls_base.scope
            syntactic_scope = cls_scope
        else:
            func_scope = old_func.scope if isinstance(old_func, FunctionDef) else old_func.syntactic_node.scope
            syntactic_scope = func_scope
        # The function call might be in a completely different scope from the FunctionDef
        # Store the current scope and go to the parent scope of the FunctionDef
        old_scope = self._scope
        old_current_function = self._current_function
        old_current_function_name = self._current_function_name

        # Walk up scope to root to find names of relevant scopes
        scope_names = []
        while syntactic_scope.parent_scope is not None:
            syntactic_scope = syntactic_scope.parent_scope
            if syntactic_scope.name is not None:
                scope_names.append(syntactic_scope.name)

        # Remove name of module scope. This is not needed to walk down the scope
        # starting from the module scope
        scope_names.pop()

        # Module scope is shared between syntactic and semantic stage
        new_scope = syntactic_scope

        # Use scope_names to find semantic scopes
        for n in scope_names[::-1]:
            new_scope = new_scope.sons_scopes[n]

        # Set the Scope to the FunctionDef's parent Scope and annotate the old_func
        self._scope = new_scope
        self._visit(old_func)

        old_name = old_func.name

        # Retrieve the annotated function
        if cls_base_syntactic:
            new_name = cls_scope.get_expected_name(old_name)
            func = cls_scope.find(old_name, 'functions')
        else:
            new_name = self.scope.get_expected_name(old_name)
            func = self.scope.find(old_name, 'functions')
        assert func is not None
        # Add the Module of the imported function to the new function
        if old_func.is_imported:
            mod = old_func.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
            func.set_current_user_node(mod)

        # Go back to the original Scope
        self._scope = old_scope
        self._current_function_name = old_current_function_name
        self._current_function = old_current_function

        return func

    def _create_variable(self, name, class_type, rhs, d_lhs, *, arr_in_multirets=False,
                         insertion_scope = None, rhs_scope = None):
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

        rhs_scope : Scope, optional
            The scope where the definition of the right hand side is found. This
            is used to locate any symbolic aliases for inhomogeneous tuples. It is
            necessary for tuples of tuples as function results.

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
            if rhs_scope is None:
                rhs_scope = self.scope

            if isinstance(rhs, FunctionCall):
                rhs_scope = rhs.funcdef.scope
                iterable = [rhs_scope.collect_tuple_element(v) for v in rhs.funcdef.results.var]
            elif isinstance(rhs, PyccelFunction):
                iterable = [IndexedElement(rhs, i)  for i in range(rhs.shape[0])]
            else:
                iterable = [rhs_scope.collect_tuple_element(r) for r in rhs]

            elem_vars = []
            for i,tuple_elem in enumerate(iterable):
                # Check if lhs element was named in the syntactic stage (this can happen for
                # results of functions)
                pyccel_stage.set_stage('syntactic')
                idx_name = IndexedElement(name, i)
                var = None
                if idx_name in self.scope.symbolic_aliases:
                    elem_name = self.scope.symbolic_aliases[idx_name]
                    var = self.check_for_variable(elem_name)
                else:
                    elem_name = insertion_scope.get_new_name( f'{name}_{i}' )
                pyccel_stage.set_stage('semantic')

                if var is None:
                    elem_d_lhs = self._infer_type( tuple_elem )

                    if not arr_in_multirets:
                        self._ensure_target( tuple_elem, elem_d_lhs )

                    elem_type = elem_d_lhs.pop('class_type')

                    var = self._create_variable(elem_name, elem_type, tuple_elem, elem_d_lhs,
                            insertion_scope = insertion_scope, rhs_scope = rhs_scope)

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

        # rhs is None in an AugAssign
        if rhs is None or isinstance(rhs, FunctionalFor):
            return

        assert rhs.pyccel_staging != 'syntactic'

        if isinstance(rhs, NumpyTranspose) and rhs.internal_var.on_heap:
            d_lhs['memory_handling'] = 'alias'
            rhs.internal_var.is_target = True

        if not isinstance(rhs.class_type, (TupleType, StringType, FixedSizeNumericType)):
            if isinstance(rhs, Variable):
                d_lhs['memory_handling'] = 'alias'
                rhs.is_target = not rhs.is_alias

            elif isinstance(rhs, IndexedElement) and \
                    isinstance(rhs.class_type, (HomogeneousTupleType, NumpyNDArrayType)):
                d_lhs['memory_handling'] = 'alias'
                rhs.base.is_target = not rhs.base.is_alias

            elif isinstance(rhs, IndexedElement) and not rhs.is_slice:
                d_lhs['memory_handling'] = 'alias'

            elif isinstance(rhs, (DictPop, DictPopitem, ListPop)):
                target_var = rhs.list_obj if isinstance(rhs, ListPop) else rhs.dict_obj
                if target_var in self._pointer_targets[-1]:
                    d_lhs['memory_handling'] = 'alias'

    def _assign_lhs_variable(self, lhs, d_var, rhs, new_expressions, is_augassign = False,
            arr_in_multirets=False):
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
            if necessary. It is also used to determine the type of allocation
            (init/resize/reserve).

        new_expressions : list
            A list which allows collection of any additional expressions
            resulting from this operation (e.g. Allocation).

        is_augassign : bool, default=False
            Indicates whether this is an assign ( = ) or an augassign ( += / -= / etc )
            This is necessary as the restrictions on the dtype are less strict in this
            case.

        arr_in_multirets : bool, default=False
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
                attribute = class_def.scope.variables.get(attr_name, None) \
                            if class_def else None
                if attribute:
                    var = attribute.clone(attribute.name, new_class = DottedVariable, lhs = prefix)
                else:
                    var = None
            else:
                symbolic_var = self.scope.find(lhs, 'symbolic_aliases')
                if symbolic_var:
                    errors.report(f"Variable '{lhs}' represents a symbolic concept. Its value cannot be changed.",
                            severity='fatal',
                            bounding_box=(self.current_ast_node.lineno,
                                self.current_ast_node.col_offset))
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
                if class_type.rank > 0 and d_lhs.get('memory_handling', None) == 'stack':
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
                                            and not getattr(rhs.funcdef, 'is_elemental', False) and \
                                            not isinstance(lhs.class_type, HomogeneousTupleType)) or arr_in_multirets or \
                                            isinstance(rhs, (ListPop, SetPop, DictPop, DictPopitem, DictGet, DictGetItem))
                if lhs.on_heap and not array_declared_in_function:
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
                                elif a.rank > 0:
                                    new_expressions.append(Allocate(a,
                                        shape=a.alloc_shape, status=status))
                            args = new_args
                            new_args = []
                    elif isinstance(lhs.class_type, (HomogeneousListType, HomogeneousSetType,DictType)):
                        if isinstance(rhs, (PythonList, PythonDict, PythonSet, FunctionCall)):
                            alloc_type = 'init'
                        elif isinstance(rhs, (IndexedElement, Duplicate)):
                            alloc_type = 'resize'
                        else:
                            alloc_type = 'reserve'
                        new_expressions.append(Allocate(lhs, shape=lhs.alloc_shape, status=status, alloc_type=alloc_type))
                    else:
                        new_expressions.append(Allocate(lhs, shape=lhs.alloc_shape, status=status))
                # ...

                # ...
                # Add memory deallocation
                if isinstance(lhs.class_type, CustomDataType) or not lhs.on_stack:
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
                if array_declared_in_function:
                    know_lhs_shape = True
                elif isinstance(lhs.dtype, StringType):
                    know_lhs_shape = (lhs.rank == 1) or all(sh is not None for sh in lhs.alloc_shape[:-1])
                else:
                    know_lhs_shape = (lhs.rank == 0) or all(sh is not None for sh in lhs.alloc_shape)

                if isinstance(class_type, (NumpyNDArrayType, HomogeneousTupleType)) and not know_lhs_shape \
                        and not array_declared_in_function:
                    msg = f"Cannot infer shape of right-hand side for expression {lhs} = {rhs}"
                    errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg,
                        bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                        severity='fatal')

            # Variable already exists
            else:
                if isinstance(var, Constant):
                    errors.report(f"Attempting to overwrite the constant {lhs}",
                                  bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                                  severity='error')

                # Try to get pre-existing DottedVariable to avoid doubles and to ensure validity of AST tree
                if isinstance(var, DottedVariable):
                    var = next((a for a in var.lhs.cls_base.attributes if var == a), var)

                self._ensure_inferred_type_matches_existing(class_type, d_lhs, var, is_augassign, new_expressions, rhs)

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

        # Check for reallocation of containers that are being used by another variable
        is_reallocatable_container = not isinstance(var.class_type, FixedSizeNumericType)
        if not is_augassign and is_reallocatable_container and var.is_target:
            pointers = [p for pt in self._pointer_targets for p, t in pt.items() \
                    if any(var in ti for ti in t)]
            if any(not p.is_temp for p in pointers):
                errors.report(TARGET_ALREADY_IN_USE,
                    bounding_box=(self.current_ast_node.lineno,
                        self.current_ast_node.col_offset),
                            severity='error', symbol=var.name)
                return

        elif not is_augassign and not var.is_alias and var.rank > 0 and \
                isinstance(rhs, (Variable, IndexedElement)) and \
                not isinstance(var.class_type, (StringType, TupleType)):
            errors.report(ASSIGN_ARRAYS_ONE_ANOTHER,
                bounding_box=(self.current_ast_node.lineno,
                    self.current_ast_node.col_offset),
                        severity='error', symbol=var)
            return

        elif var.rank > 0 and var.is_alias and isinstance(rhs, (NumpyNewArray, PythonList, PythonSet, PythonDict)):
            errors.report(INVALID_POINTER_REASSIGN,
                bounding_box=(self.current_ast_node.lineno,
                    self.current_ast_node.col_offset),
                        severity='error', symbol=var.name)
            return

        elif var.is_ndarray and var.is_alias and not is_augassign:
            # we allow pointers to be reassigned multiple times
            # pointers reassigning need to call free_pointer func
            # to remove memory leaks
            new_expressions.append(Deallocate(var))
            return

        elif class_type != var.class_type or \
                (isinstance(class_type, InhomogeneousTupleType) and class_type is not var.class_type):
            if is_augassign:
                tmp_result = PyccelAdd(var, rhs)
                result_type = tmp_result.class_type
                raise_error = var.class_type != result_type
            elif isinstance(var.class_type, InhomogeneousTupleType) and \
                    isinstance(class_type, HomogeneousTupleType):
                if d_var['shape'][0] == var.shape[0]:
                    rhs_elem = self.scope.collect_tuple_element(var[0])
                    self._ensure_inferred_type_matches_existing(class_type.element_type,
                            self._infer_type(rhs_elem), rhs_elem, is_augassign, new_expressions, rhs)
                    raise_error = False
                else:
                    raise_error = True
            elif isinstance(var.class_type, InhomogeneousTupleType) and \
                    isinstance(class_type, InhomogeneousTupleType):
                for i, element_type in enumerate(class_type):
                    rhs_elem = self.scope.collect_tuple_element(var[i])
                    self._ensure_inferred_type_matches_existing(element_type,
                            self._infer_type(rhs_elem), rhs_elem, is_augassign, new_expressions, rhs)
                raise_error = False
            elif isinstance(var.class_type, HomogeneousTupleType) and \
                    isinstance(class_type, InhomogeneousTupleType):
                # TODO: Remove isinstance(rhs, Variable) condition when tuples are saved like lists
                if isinstance(rhs, PythonTuple):
                    shape = get_shape_of_multi_level_container(rhs)
                    raise_error = len(shape) != class_type.rank or not class_type.shape_is_compatible(shape) \
                            or any(a != var.class_type.element_type for a in class_type)
                else:
                    raise_error = any(a != var.class_type.element_type for a in class_type) or \
                            not isinstance(rhs, Variable)
            else:
                raise_error = True

            if raise_error:
                name = var.name
                rhs_str = str(rhs)
                errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT.format(var.class_type, class_type),
                    symbol=f'{name}={rhs_str}',
                    bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                    severity='error')
                return

        if not is_augassign:

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
                    alloc_type = None
                    if isinstance(var.class_type, (HomogeneousListType, HomogeneousSetType,DictType)):
                        if isinstance(rhs, (PythonList, PythonDict, PythonSet, FunctionCall)):
                            alloc_type = 'init'
                        elif isinstance(rhs, (IndexedElement, Duplicate)):
                            alloc_type = 'resize'
                        else:
                            alloc_type = 'reserve'
                    if previous_allocations:
                        var.set_changeable_shape()
                        if isinstance(var, DottedVariable):
                            # DottedVariable is constructed from the variable in the class's scope
                            cls_scope = var.lhs.cls_base.scope
                            py_name = cls_scope.get_python_name(var.name)
                            attribute = var.lhs.cls_base.scope.find(py_name, 'variables')
                            attribute.set_changeable_shape()

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

                    new_expressions.append(Allocate(var, shape=d_var['shape'], status=status, alloc_type=alloc_type))

                    if status == 'unallocated':
                        self._allocs[-1].add(var)
                    elif isinstance(var.class_type, NumpyNDArrayType):
                        errors.report(ARRAY_REALLOCATION.format(class_type = var.class_type), symbol=var.name,
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
        while isinstance(loop, (For, If)):

            nlevels+=1
            self._get_for_iterators(loop.iterable, loop.target, new_expr, expr)

            loop_elem = loop.body.body[0]
            if isinstance(loop_elem, If):
                loop_elem = loop_elem.blocks[0].body.body[0]

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
            d_var = self._infer_type(result)
            val = MaxLimit(d_var['class_type'])
        elif isinstance(expr, FunctionalMax):
            d_var = self._infer_type(result)
            val = MinLimit(d_var['class_type'])

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
            expr_new = FunctionalSum(loops, lhs=lhs, indices = indices, conditions=expr.conditions)
        elif isinstance(expr, FunctionalMin):
            expr_new = FunctionalMin(loops, lhs=lhs, indices = indices, conditions=expr.conditions)
        elif isinstance(expr, FunctionalMax):
            expr_new = FunctionalMax(loops, lhs=lhs, indices = indices, conditions=expr.conditions)
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
        if isinstance(base, UnionTypeAnnotation):
            return UnionTypeAnnotation(*[self._get_indexed_type(t, args, expr) for t in base.type_list])

        if len(args) == 0:
            if not (isinstance(base, PyccelFunctionDef) and base.cls_name.static_type() is TupleType):
                errors.report("Unrecognised type", severity='fatal', symbol=expr)
            return UnionTypeAnnotation(VariableTypeAnnotation(InhomogeneousTupleType.get_new()))

        if all(isinstance(a, Slice) for a in args):
            rank = len(args)
            order = None if rank < 2 else 'C'
            if isinstance(base, VariableTypeAnnotation):
                dtype = base.class_type
                if dtype.rank != 0:
                    raise errors.report("NumPy element must be a scalar type", severity='fatal', symbol=expr)
                class_type = NumpyNDArrayType.get_new(numpy_process_dtype(dtype), rank, order)
            elif isinstance(base, PyccelFunctionDef):
                dtype_cls = base.cls_name
                try:
                    dtype = numpy_process_dtype(dtype_cls.static_type())
                except AttributeError:
                    errors.report(f"Unrecognised datatype {dtype_cls}", severity='fatal', symbol=expr)
                class_type = NumpyNDArrayType.get_new(dtype, rank, order)
            return VariableTypeAnnotation(class_type)

        if not any(isinstance(a, Slice) for a in args):
            if isinstance(base, PyccelFunctionDef):
                dtype_cls = base.cls_name.static_type()
            else:
                raise errors.report(f"Unknown annotation base {base}\n"+PYCCEL_RESTRICTION_TODO,
                        severity='fatal', symbol=expr)
            if (len(args) == 2 and args[1] is LiteralEllipsis()) or \
                    (len(args) == 1 and dtype_cls is not TupleType):
                syntactic_annotation = self._convert_syntactic_object_to_type_annotation(args[0])
                internal_datatypes = self._visit(syntactic_annotation)
                class_type = HomogeneousTupleType if dtype_cls is TupleType else dtype_cls
                type_annotations = [VariableTypeAnnotation(class_type.get_new(u.class_type))
                                    for u in internal_datatypes.type_list]
                return UnionTypeAnnotation(*type_annotations)
            elif len(args) == 2 and dtype_cls is DictType:
                syntactic_key_annotation = self._convert_syntactic_object_to_type_annotation(args[0])
                syntactic_val_annotation = self._convert_syntactic_object_to_type_annotation(args[1])
                key_types = self._visit(syntactic_key_annotation)
                val_types = self._visit(syntactic_val_annotation)
                type_annotations = [VariableTypeAnnotation(dtype_cls.get_new(k.class_type, v.class_type)) \
                                    for k,v in zip(key_types.type_list, val_types.type_list)]
                return UnionTypeAnnotation(*type_annotations)
            elif dtype_cls is TupleType:
                syntactic_annotations = [self._convert_syntactic_object_to_type_annotation(a) for a in args]
                types = [self._visit(a).type_list for a in syntactic_annotations]
                internal_datatypes = list(product(*types))
                type_annotations = [VariableTypeAnnotation(InhomogeneousTupleType.get_new(*[ui.class_type for ui in u]))
                                    for u in internal_datatypes]
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
            The class definition to which the attribute should be added.
        self_var : Variable
            The variable representing the 'self' variable of the class instance.
        attrib : Variable
            The attribute which should be inserted into the class definition.

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

    def _get_iterable(self, syntactic_iterable):
        """
        Get an Iterable object from a syntactic object that is used in an iterable context.

        Get an Iterable object from a syntactic object that is used in an iterable context.
        A typical example of an iterable context is the iterable of a for loop.

        Parameters
        ----------
        syntactic_iterable : PyccelAstNode
            The syntactic object that should be usable as an iterable.

        Returns
        -------
        Iterable
            A semantic Iterable object.
        """
        iterable = self._visit(syntactic_iterable)
        if isinstance(iterable, (Variable, IndexedElement)):
            if isinstance(iterable.class_type, DictType):
                iterable = DictKeys(iterable)
            else:
                iterable = VariableIterator(iterable)
        elif not isinstance(iterable, Iterable):
            if isinstance(iterable, TypedAstNode):
                pyccel_stage.set_stage('syntactic')
                tmp_var = self.scope.get_new_name()
                syntactic_assign = Assign(tmp_var, iterable, python_ast = iterable.python_ast)
                pyccel_stage.set_stage('semantic')
                assign = self._visit(syntactic_assign)
                self._additional_exprs[-1].append(assign)
                iterable = VariableIterator(self._visit(tmp_var))
            else:
                errors.report(f"{iterable} is not handled as the iterable of a for loop",
                        symbol=syntactic_iterable, severity='fatal')

        return iterable

    def _get_for_iterators(self, syntactic_iterable, iterator, new_expr, expr):
        """
        Get the semantic target and iterable of a for loop.

        Get the semantic target and iterable of a for loop. This method can be used to
        handle generators, comprehension expressions or basic for loops.

        Parameters
        ----------
        syntactic_iterable : TypedAstNode
            The iterable that the for loop iterates over.
        iterator : TypedAstNode
            The syntactic iterator that takes the value of the elements of the iterable.
        new_expr : list[PyccelAstNode]
            A list which allows collection of any additional expressions
            resulting from this operation (e.g. Allocation).
        expr : PyccelAstNode
            The expression being visited. This is used for error handling.

        Returns
        -------
        target : TypedAstNode
            The semantic iterator that takes the value of the elements of the iterable.
        iterable : TypedAstNode
            The semantic iterable that the for loop iterates over.
        """
        iterable = self._get_iterable(syntactic_iterable)

        if iterable.num_loop_counters_required:
            indices = [Variable(PythonNativeInt(), self.scope.get_new_name(), is_temp=True)
                        for i in range(iterable.num_loop_counters_required)]
            iterable.set_loop_counter(*indices)
        else:
            if isinstance(iterable, PythonEnumerate):
                if isinstance(iterator, PythonTuple):
                    syntactic_index = iterator[0]
                else:
                    pyccel_stage.set_stage('syntactic')
                    syntactic_index = IndexedElement(iterator,0)
                    pyccel_stage.set_stage('semantic')
            else:
                syntactic_index = iterator

            index = self.check_for_variable(syntactic_index)
            if index is None:
                start = LiteralInteger(0)
                d_var = self._infer_type(start)
                if isinstance(syntactic_index, PyccelSymbol):
                    index = self._assign_lhs_variable(syntactic_index, d_var,
                                    rhs=start, new_expressions=new_expr)
                else:
                    index = self.scope.get_temporary_variable(PythonNativeInt())
            iterable.set_loop_counter(index)

        # Collect a target with a deducible dtype
        iterator_rhs = iterable.get_python_iterable_item()

        # Use _visit_Assign to create the requested iterator with the correct type
        # The result of this operation is not stored, it is just used to declare
        # iterator with the correct dtype to allow correct dtype deductions later
        if isinstance(iterator, PyccelSymbol):
            if len(iterator_rhs) != 1:
                iterator_rhs = PythonTuple(*iterator_rhs, prefer_inhomogeneous=True)
            else:
                iterator_rhs = iterator_rhs[0]

            iterator_d_var = self._infer_type(iterator_rhs)

            target = self._assign_lhs_variable(iterator, iterator_d_var,
                            rhs=iterator_rhs, new_expressions=new_expr)

            if target.is_alias:
                self._indicate_pointer_target(target, iterator_rhs, expr.python_ast)

            if isinstance(target.class_type, InhomogeneousTupleType):
                target = [self.scope.collect_tuple_element(v) for v in target]
            else:
                target = [target]

        elif isinstance(iterator, PythonTuple):
            target = [self._assign_lhs_variable(it, self._infer_type(rhs),
                                rhs=rhs, new_expressions=new_expr)
                        for it, rhs in zip(iterator, iterator_rhs)]

            for t, rhs in zip(target, iterator_rhs):
                if t.is_alias:
                    self._indicate_pointer_target(t, rhs, expr.python_ast)
        else:
            raise errors.report(INVALID_FOR_ITERABLE, symbol=iterator,
                   bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                   severity='error')

        return target, iterable

    def env_var_to_pyccel(self, env_var, *, name = None):
        """
        Convert an environment variable to a Pyccel AST node.

        Convert an environment variable (i.e. a variable deduced from the
        context where epyccel was called) into a Pyccel AST node as though
        the object had been declared explicitly in the code.

        Parameters
        ----------
        env_var : object
            The environment variable.
        name : str, optional
            The name that was used to identify the variable.

        Returns
        -------
        PyccelAstNode
            The usable Pyccel AST node.
        """
        if env_var in original_type_to_pyccel_type:
            return VariableTypeAnnotation(original_type_to_pyccel_type[env_var])
        elif type(env_var) in original_type_to_pyccel_type:
            return convert_to_literal(env_var, dtype = original_type_to_pyccel_type[type(env_var)])
        elif env_var is typing.Final:
            return PyccelFunctionDef('Final', TypingFinal)
        elif isinstance(env_var, typing.GenericAlias):
            class_type = self.env_var_to_pyccel(typing.get_origin(env_var)).class_type.static_type()
            return VariableTypeAnnotation(class_type.get_new(*[self.env_var_to_pyccel(a).class_type for a in typing.get_args(env_var)]))
        elif isinstance(env_var, typing.TypeVar):
            constraints = [self.env_var_to_pyccel(c) for c in env_var.__constraints__]
            return TypingTypeVar(env_var.__name__, *constraints,
                                covariant = env_var.__covariant__,
                                contravariant = env_var.__contravariant__)
        elif isinstance(env_var, ModuleType):
            mod_name = env_var.__name__
            if recognised_source(mod_name):
                pyccel_stage.set_stage('syntactic')
                import_node = Import(AsName(mod_name, name))
                pyccel_stage.set_stage('semantic')
                # Insert import at global scope
                current_scope = self.scope
                scope = current_scope
                while scope.parent_scope:
                    scope = scope.parent_scope
                self.scope = scope
                self._additional_exprs[-1].append(self._visit(import_node))
                self.scope = current_scope
                return self.scope.find(name)
            else:
                errors.report(f"Unrecognised module {mod_name} imported in global scope. Please import the module locally if it was previously Pyccelised.",
                        severity='error', symbol = self.current_ast_node)
        elif isinstance(env_var, (typing.ForwardRef, str)):
            pyccel_stage.set_stage('syntactic')
            try:
                annotation = types_meta.model_from_str(getattr(env_var, '__forward_arg__', env_var))
            except TextXSyntaxError as e:
                errors.report(f"Invalid annotation. {e.message}",
                        symbol = self.current_ast_node, severity='fatal')
            annot = annotation.expr
            pyccel_stage.set_stage('semantic')
            return self._visit(annot)

        errors.report(PYCCEL_RESTRICTION_TODO,
                severity='error', symbol = self.current_ast_node)
        return None

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
            try:
                if hasattr(self, annotation_method):
                    if self._verbose > 2:
                        print(f">>>> Calling SemanticParser.{annotation_method}")
                    obj = getattr(self, annotation_method)(expr)
                    if isinstance(obj, PyccelAstNode) and self.current_ast_node:
                        obj.set_current_ast(self.current_ast_node)
                    self._current_ast_node = current_ast
                    return obj
            except PyccelError as err:
                raise err
            except NotImplementedError as error:
                errors.report(f'{error}\n'+PYCCEL_RESTRICTION_TODO,
                    symbol = self._current_ast_node, severity='fatal',
                    traceback=error.__traceback__)
            except Exception as err: #pylint: disable=broad-exception-caught
                if ErrorsMode().value == 'user':
                    errors.report(PYCCEL_INTERNAL_ERROR,
                            symbol = self._current_ast_node, severity='fatal')
                else:
                    raise err

        # Unknown object, we raise an error.
        return errors.report(PYCCEL_RESTRICTION_TODO, symbol=type(expr),
            bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
            severity='fatal')

    def _visit_Module(self, expr):
        if not self.is_stub_file:
            self.scope.insert_symbol('__init__', object_type = 'function')
            self.scope.insert_symbol('__del__', object_type = 'function')

        imports = [self._visit(i) for i in expr.imports]
        init_func_body = [i for i in imports if not isinstance(i, EmptyNode)]

        if not self.is_stub_file:
            for f in expr.funcs:
                self.insert_function(f)
        else:
            funcs = {}
            for f in expr.funcs:
                funcs.setdefault(f.name, []).append(f)

            pyccel_stage.set_stage('syntactic')
            funcs_to_insert = [f[0] if len(f) == 1 else Interface(n, f) for n, f in funcs.items()]
            pyccel_stage.set_stage('semantic')

            for f in funcs_to_insert:
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
            self.scope.insert_symbol(mod_name, 'module')
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
                    original_symbols = prog_syntactic_scope.python_names.copy(),
                    scope_type = 'program')
            prog_scope = self.scope

            imports = [self._visit(i) for i in expr.program.imports]
            body = [i for i in imports if not isinstance(i, EmptyNode)]

            body += self._visit(expr.program.body).body

            program_body = CodeBlock(body)

            # Calling the Garbage collecting,
            # it will add the necessary Deallocate nodes
            # to the ast
            program_body.insert2body(*self._garbage_collector(program_body))
            self._pointer_targets.pop()

            self.scope = mod_scope

        # Funcs to visit are collected from scope so functions that have already been visited
        # can be excluded.
        funcs_to_visit = [f for func in self.scope.functions.values()
                          for f in (func.functions if isinstance(func, Interface) else [func])]
        funcs_to_visit.extend(m for c in self.scope.classes.values() for m in c.methods)

        for f in funcs_to_visit:
            if not f.is_semantic and not isinstance(f, InlineFunctionDef):
                assert isinstance(f, FunctionDef)
                self._visit(f)

        for f in self.scope.functions.values():
            assert f.is_semantic or f.is_inline

        variables = self.get_variables(self.scope)
        init_func = None
        free_func = None
        program   = None

        comment_types = (Header, EmptyNode, Comment, CommentBlock)

        if self.is_stub_file:
            init_func = self.scope.functions.get('__init__', None)
        elif not all(isinstance(l, comment_types) for l in init_func_body):
            # If there are any initialisation statements then create an initialisation function
            init_var = Variable(PythonNativeBool(), self.scope.get_new_name('initialised'),
                                is_private=True, is_temp = True)
            syntactic_init_func_name = '__init__'
            init_func_name = self.scope.get_expected_name(syntactic_init_func_name)
            # Ensure that the function is correctly defined within the namespaces
            init_scope = self.create_new_function_scope(syntactic_init_func_name, init_func_name)
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
                    # Leave symbol to handle tuples
                    self.scope.remove_variable(v, remove_symbol = False)
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

            init_func = FunctionDef(init_func_name, [], [init_func_body],
                    global_vars = variables, scope=init_scope)
            self.insert_function(init_func)
            self.scope.insert_variable(init_var)

            # Find any remaining unused symbols in the main scope and move them to the init body scope
            # This may include temporary variables whose necessity has not yet been determined, such as
            # loop variables. They must be moved so the symbol is found when the variable is eventually
            # inserted into the scope.
            unused_symbols = [n for n in self.scope.local_used_symbols \
                              if self.scope.find(n) is None and n not in (mod_name, '__del__')]
            for s in unused_symbols:
                self.scope.remove_symbol(s)
                init_scope.insert_symbol(s)

        if init_func is None:
            self.scope.remove_symbol('__init__')

        if self.is_stub_file:
            free_func = self.scope.functions.get('__del__', None)
        elif init_func:
            syntactic_free_func_name = '__del__'
            free_func_name = self.scope.get_expected_name(syntactic_free_func_name)
            pyccelised_imports = [imp for imp_name, imp in self.scope.imports['imports'].items() \
                             if imp_name in self.d_parsers]

            import_frees = [self.d_parsers[imp.source].semantic_parser.ast.free_func for imp in pyccelised_imports \
                                if imp.source in self.d_parsers]
            import_frees = [next(t.object for t in imp.target if t.name == f.name) \
                            for f,imp in zip(import_frees, pyccelised_imports) if f]
            assert all(f.is_imported for f in import_frees)

            if deallocs or import_frees:
                # If there is anything that needs deallocating when the module goes out of scope
                # create a deallocation function
                import_free_calls = [f() for f in import_frees if f is not None]

                free_func_body = If(IfSection(init_var,
                    import_free_calls+deallocs+[Assign(init_var, LiteralFalse())]))
                # Ensure that the function is correctly defined within the namespaces
                scope = self.create_new_function_scope(syntactic_free_func_name, free_func_name)
                free_func = FunctionDef(free_func_name, [], [free_func_body],
                                    global_vars = variables, scope = scope)
                self.exit_function_scope()
                self.insert_function(free_func)

        if free_func is None:
            self.scope.remove_symbol('__del__')

        classes = self.scope.classes.values()
        if not self.is_stub_file:
            for c in classes:
                self._create_class_destructor(c)

        funcs = []
        interfaces = []
        for f in self.scope.functions.values():
            if isinstance(f, FunctionDef):
                funcs.append(f)
            elif isinstance(f, Interface):
                interfaces.append(f)

        # in the case of a stub file, we need to convert all headers to
        # FunctionDef etc ...

        if self.is_stub_file:
            if self.metavars.get('external', False):
                for f in funcs:
                    f.is_external = True
                for c in classes:
                    for m in c.methods:
                        m.is_external = True

        for v in variables:
            if v.rank > 0 and not v.is_alias:
                v.is_target = True

        mod = Module(mod_name,
                    variables,
                    funcs,
                    init_func = init_func,
                    free_func = free_func,
                    interfaces=interfaces,
                    classes=classes,
                    imports=self.scope.imports['imports'].values(),
                    scope=self.scope)

        if expr.program:
            container = prog_scope.imports
            container['imports'][mod_name] = Import(self.scope.get_python_name(mod_name), mod)

            if init_func:
                import_init  = init_func()
                program_body.insert2body(import_init, back=False)

            if free_func:
                import_free  = free_func()
                program_body.insert2body(import_free)

            imports = list(container['imports'].values())
            for i in self.scope.imports['imports'].values():
                target = []
                for t in i.target:
                    local_t = self.scope.find(t.name)
                    if local_t and program_body.is_user_of(local_t, excluded_nodes = (FunctionDef,)):
                        target.append(t)
                if target:
                    imports.append(Import(i.source, target, ignore_at_print = i.ignore, mod = i.source_module))
            program = Program(prog_name,
                            self.get_variables(prog_scope),
                            program_body,
                            imports,
                            scope=prog_scope)

            mod.program = program
        return mod

    def _visit_PythonTuple(self, expr):
        ls = [self._visit(i) for i in expr]
        prefer_inhomogeneous = False
        if expr.get_user_nodes(Return, (IndexedElement, FunctionCall, PyccelFunction, PyccelOperator)):
            func = expr.get_user_nodes(FunctionDef)[0]
            n_returns = set(r.n_explicit_results for r in func.get_attribute_nodes(Return))
            prefer_inhomogeneous = len(n_returns) == 1
        return PythonTuple(*ls, prefer_inhomogeneous = prefer_inhomogeneous)

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
            if value.funcdef.results.var and not value.funcdef.results.var.is_alias:
                a = generate_and_assign_temp_var()
        return a

    def _visit_UnionTypeAnnotation(self, expr):
        annotations = [self._visit(syntax_type_annot) for syntax_type_annot in expr.type_list]
        types = [t for a in annotations for t in (a.type_list if isinstance(a, UnionTypeAnnotation) else [a])]
        return UnionTypeAnnotation(*types)

    def _visit_FunctionTypeAnnotation(self, expr):
        arg_types = [self._visit(a)[0] for a in expr.args]
        res_type = self._visit(expr.result)
        return UnionTypeAnnotation(FunctionTypeAnnotation(arg_types, res_type))

    def _visit_FunctionDefArgument(self, expr):
        arg = self._visit(expr.var)
        value = None if expr.value is None else self._visit(expr.value)
        posonly = expr.is_posonly
        kwonly = expr.is_kwonly
        is_vararg = expr.is_vararg
        is_kwarg = expr.is_kwarg
        is_optional = isinstance(value, Nil)
        bound_argument = expr.bound_argument

        args = []
        for v in arg:
            if isinstance(v, Variable):
                dtype = v.class_type
                if isinstance(value, Literal) and value is not Nil():
                    value = convert_to_literal(value.python_value, dtype)
                if isinstance(dtype, InhomogeneousTupleType):
                    # Raise an error as elements are not yet correctly marked with is_argument.
                    # This leads to printing errors
                    errors.report("Inhomogeneous tuples are not yet supported as arguments",
                            severity='error', symbol=expr)
                if isinstance(dtype, CustomDataType) and not bound_argument:
                    cls = self.scope.find(str(dtype), 'classes')
                    if cls:
                        init_method = cls.get_method('__init__', expr)
                        if not init_method.is_semantic and not self.is_stub_file:
                            self._visit(init_method)
                clone_var = v.clone(v.name, is_optional = is_optional, is_argument = True)
                args.append(FunctionDefArgument(clone_var, bound_argument = bound_argument,
                                        value = value, posonly = posonly, kwonly = kwonly, annotation = expr.annotation,
                                        is_vararg = is_vararg, is_kwarg = is_kwarg))
            else:
                args.append(FunctionDefArgument(v.clone(v.name, is_optional = is_optional,
                                is_argument = True), posonly = posonly, kwonly = kwonly,
                                bound_argument = bound_argument,
                                value = value, annotation = expr.annotation,
                                is_vararg = is_vararg, is_kwarg = is_kwarg))
        return args

    def _visit_CodeBlock(self, expr):
        ls = []
        self._additional_exprs.append([])
        for b in expr.body:
            if isinstance(b, EmptyNode):
                continue
            if isinstance(b, InlineFunctionDef):
                self.insert_function(b)
                continue
            # Save parsed code
            line = self._visit(b)
            ls.extend(self._additional_exprs[-1])
            self._additional_exprs[-1] = []
            if isinstance(line, CodeBlock):
                ls.extend(line.body)
            elif isinstance(line, list) and isinstance(line[0], Variable):
                var = line[0]
                if isinstance(var, DottedVariable):
                    cls_def = var.lhs.cls_base
                    cls_def.add_new_attribute(var)
                    cls_def.scope.insert_variable(var.clone(var.name, new_class = Variable))
                else:
                    self.scope.insert_variable(var)
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

        class_type = var.class_type

        if isinstance(class_type, (NumpyNDArrayType, HomogeneousListType, TupleType)):
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
        else:
            cls_base = self.get_cls_base(class_type)
            method = cls_base.get_method('__getitem__')
            if method:
                class_args = self._handle_function_args([FunctionCallArgument(a) for a in expr.indices])
                args = [FunctionCallArgument(var), *class_args]
                return self._handle_function(expr, method, args)
            else:
                raise errors.report(f"No __getitem__ found for type {class_type}",
                        severity='fatal', symbol=expr)

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
            if name == '*':
                return GenericType()

        if var is None and name in self._context_dict:
            var = self.env_var_to_pyccel(self._context_dict[name], name = name)

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
        var = self.scope.find(expr.name, 'variables', local_only = True)
        if var is not None and not any(isinstance(n, FunctionDefResult) for n in var.get_all_user_nodes()):
            errors.report("Variable has been declared multiple times",
                    symbol=expr, severity='error')

        if expr.annotation is None:
            errors.report(MISSING_TYPE_ANNOTATIONS,
                    symbol=expr, severity='fatal')

        # Get the semantic type annotation (should be UnionTypeAnnotation)
        types = self._visit(expr.annotation)
        assert not isinstance(types, TypingTypeVar)

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
                scope = self.create_new_function_scope(name, name)
                if t.result.var:
                    results = FunctionDefResult(t.result.var.clone(t.result.var.name, is_argument = False),
                                    annotation=t.result.annotation)

                else:
                    results = FunctionDefResult(Nil())
                self.exit_function_scope()
                address = FunctionAddress(name, args, results, scope = scope)
                possible_args.append(address)
            elif isinstance(t, VariableTypeAnnotation):
                class_type = t.class_type
                cls_base = self.get_cls_base(class_type)
                if isinstance(class_type, InhomogeneousTupleType):
                    shape = (len(class_type),)
                elif isinstance(class_type, HomogeneousTupleType):
                    shape = (None,)*class_type.rank
                elif class_type.rank:
                    shape = (None,)*class_type.container_rank
                else:
                    shape = None
                v = var_class(class_type, name, cls_base = cls_base,
                        shape = shape, is_optional = False,
                        memory_handling = array_memory_handling if class_type.rank > 0 else 'stack',
                        **kwargs)
                possible_args.append(v)
                if isinstance(class_type, InhomogeneousTupleType):
                    for i, t in enumerate(class_type):
                        pyccel_stage.set_stage('syntactic')
                        syntactic_elem = AnnotatedPyccelSymbol(self.scope.get_new_name( f'{name}_{i}'),
                                                annotation = UnionTypeAnnotation(VariableTypeAnnotation(t)))
                        pyccel_stage.set_stage('semantic')
                        elem = self._visit(syntactic_elem)
                        self.scope.insert_symbolic_alias(IndexedElement(v, i), elem[0])
            else:
                errors.report(PYCCEL_RESTRICTION_TODO + '\nUnrecognised type annotation',
                        severity='fatal', symbol=expr)

        # An annotated variable must have a type
        assert len(possible_args) != 0

        # If var was declared in results
        if var is not None:
            new_var = possible_args[0]
            if len(possible_args) != 1 or new_var.class_type != var.class_type:
                errors.report(f"Variable was declared as the result of the function {self.current_function_name} but is now declared with a different type",
                        symbol=expr, severity='error')
            # Remove variable from scope as AnnotatedPyccelSymbol is always inserted into scope
            self.scope.remove_variable(var, remove_symbol = False)
            return [var]

        return possible_args

    def _visit_SyntacticTypeAnnotation(self, expr):
        self._in_annotation = True
        visited_dtype = self._visit(expr.dtype)
        self._in_annotation = False
        order = expr.order

        if isinstance(visited_dtype, UnionTypeAnnotation) and len(visited_dtype.type_list) == 1:
            visited_dtype = visited_dtype.type_list[0]

        if isinstance(visited_dtype, PyccelFunctionDef):
            dtype_cls = visited_dtype.cls_name
            try:
                class_type = dtype_cls.static_type()
            except AttributeError:
                errors.report(f"Unrecognised datatype {dtype_cls}", severity='fatal', symbol=expr)
            return UnionTypeAnnotation(VariableTypeAnnotation(class_type))
        elif isinstance(visited_dtype, VariableTypeAnnotation):
            if order and order != visited_dtype.class_type.order:
                visited_dtype = VariableTypeAnnotation(visited_dtype.class_type.swap_order())
            return UnionTypeAnnotation(visited_dtype)
        elif isinstance(visited_dtype, (UnionTypeAnnotation, TypingTypeVar)):
            return visited_dtype
        elif isinstance(visited_dtype, ClassDef):
            dtype = visited_dtype.class_type
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
            first = results.var
        rhs_name = _get_name(rhs)

        # Handle case of imported module
        if isinstance(first, Module):

            if rhs_name in first:
                scope = self.scope
                imp = scope.find(_get_name(lhs), 'imports')

                new_name = rhs_name
                rhs_obj = first[rhs_name]
                if imp is not None:
                    new_name = imp.find_module_target(rhs_name)
                    if new_name is None:
                        new_name = scope.get_new_name(rhs_name)

                        # Save the import target that has been used
                        imp.define_target(AsName(rhs_obj, new_name))
                        source_mod = _get_name(lhs)
                        while source_mod not in scope.imports['imports']:
                            scope = scope.parent_scope

                if isinstance(rhs_obj, PyccelFunctionDef):
                    assert new_name not in scope.imports['functions'] or scope.imports['functions'][new_name] == rhs_obj
                    scope.imports['functions'][new_name] = rhs_obj
                elif isinstance(rhs, FunctionCall):
                    if new_name not in scope.imports['functions']:
                        m = rhs_obj.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
                        rhs_obj = rhs_obj.clone(rhs_obj.name, is_imported = True)
                        scope.imports['functions'][new_name] = rhs_obj
                        rhs_obj.set_current_user_node(m)
                elif isinstance(rhs, ConstructorCall):
                    assert new_name not in scope.imports['classes']
                    scope.imports['classes'][new_name] = rhs_obj
                elif isinstance(rhs, Variable):
                    assert new_name not in scope.imports['variables']
                    scope.imports['variables'][new_name] = rhs

                if isinstance(rhs, FunctionCall):
                    # If object is a function
                    args  = self._handle_function_args(rhs.args)
                    return self._handle_function(expr, func = rhs_obj, args = args)
                elif isinstance(rhs, Constant):
                    if new_name != rhs_name:
                        rhs_obj.name = new_name
                    return rhs_obj
                else:
                    # If object is something else (eg. dict)
                    return rhs_obj
            else:
                errors.report(UNDEFINED_IMPORT_OBJECT.format(rhs_name, str(lhs)),
                        symbol=expr, severity='fatal')
        if isinstance(first, ClassDef):
            errors.report("Static class methods are not yet supported", symbol=expr,
                    severity='fatal')

        d_var = self._infer_type(first)
        class_type = d_var['class_type']
        cls_base = self.get_cls_base(class_type)

        # look for a class method
        if isinstance(rhs, FunctionCall):
            method = cls_base.get_method(rhs_name, expr)

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
                method = cls_base.get_method(rhs_name, expr)
                assert 'property' in method.decorators
                if cls_base.name == 'numpy.ndarray':
                    numpy_class = method.cls_name
                    self.insert_import('numpy', AsName(numpy_class, numpy_class.name))
                return self._handle_function(expr, method, [FunctionCallArgument(visited_lhs)], is_method = True)


        # did something go wrong?
        return errors.report(f'Attribute {rhs_name} not found',
            bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
            severity='fatal')

    def _visit_PyccelOperator(self, expr):
        args = [self._visit(a) for a in expr.args]
        return self._create_PyccelOperator(expr, args)

    def _visit_PyccelBooleanOperator(self, expr):
        args = [self._visit(a) for a in expr.args]
        args = [a if a.dtype is PythonNativeBool() else PythonBool(a) for a in args]
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

    def _visit_PyccelIn(self, expr):
        element = self._visit(expr.element)
        container = self._visit(expr.container)
        container_type = container.class_type
        if isinstance(container_type, (DictType, HomogeneousSetType, HomogeneousListType)):
            element_type = container_type.key_type if isinstance(container_type, DictType) else container_type.element_type
            if element.class_type == element_type:
                return PyccelIn(element, container)
            else:
                return LiteralFalse()

        container_base = self.get_cls_base(container_type)
        contains_method = container_base.get_method('__contains__',
                        raise_error_from = expr if isinstance(container_type, CustomDataType) else None)
        if contains_method:
            return self._handle_function(expr, contains_method, [FunctionCallArgument(container), FunctionCallArgument(element)])
        else:
            raise errors.report(f"In operator is not yet implemented for type {container_type}",
                    severity='fatal', symbol=expr)

    def _visit_FunctionCall(self, expr):
        name     = expr.funcdef
        func = self.scope.find(name, 'functions')

        if func is None:
            if name in builtin_functions_dict:
                func = PyccelFunctionDef(name, builtin_functions_dict[name])

        args = self._handle_function_args(expr.args)

        if isinstance(func, PyccelFunctionDef) and func.cls_name is TypingTypeVar:
            new_args = [args[0]]
            for a in args[1:]:
                a_val = a.value
                if isinstance(a_val, LiteralString):
                    pyccel_stage.set_stage('syntactic')
                    try:
                        syntactic_a = types_meta.model_from_str(a_val.python_value)
                    except TextXSyntaxError as e:
                        errors.report(f"Invalid annotation. {e.message}",
                                symbol = self.current_ast_node, severity='fatal')
                    annot = syntactic_a.expr
                    pyccel_stage.set_stage('semantic')
                    new_args.append(FunctionCallArgument(self._visit(annot)))
                else:
                    new_args.append(a)
            args = new_args

        # Correct keyword names if scope is available
        # The scope is only available if the function body has been parsed
        # (i.e. not for headers or builtin functions)
        if (isinstance(func, FunctionDef) and func.scope) or isinstance(func, Interface):
            scope = func.scope if isinstance(func, FunctionDef) else func.functions[0].scope
            args = [a if a.keyword is None else \
                    FunctionCallArgument(a.value, scope.local_used_symbols.get(a.keyword, a.keyword)) \
                    for a in args]
            func_args = func.arguments if isinstance(func,FunctionDef) else func.functions[0].arguments
            if not func.is_semantic:
                # Correct func_args keyword names
                func_args = [FunctionDefArgument(AnnotatedPyccelSymbol(scope.get_expected_name(a.var.name), a.annotation),
                            annotation=a.annotation, value=a.value, posonly=a.is_posonly, kwonly=a.is_kwonly,
                            bound_argument=a.bound_argument, is_vararg = a.is_vararg, is_kwarg = a.is_kwarg)
                            for a in func_args]
            args = self._sort_function_call_args(func_args, args)

        if self.scope.find(name, 'cls_constructs'):

            # TODO improve the test
            # we must not invoke the scope like this

            cls = self.scope.find(name, 'classes')
            d_methods = cls.methods_as_dict
            init_method = d_methods.pop('__init__', None)

            dtype = cls.class_type
            cls_def = cls
            d_var = {'class_type' : dtype,
                    'memory_handling':'stack',
                    'shape' : None,
                    'cls_base' : cls_def,
                    }
            new_expression = []

            assigns = expr.get_direct_user_nodes(lambda a: isinstance(a, Assign))
            if assigns:
                lhs = assigns[0].lhs
            else:
                lhs = self.scope.get_new_name()

            if init_method is not None:
                if not init_method.is_semantic:
                    if init_method.is_inline:
                        errors.report("An __init__ method cannot be inlined",
                                severity='fatal', symbol=expr)
                    init_method = self._annotate_the_called_function_def(init_method, args)

                if isinstance(lhs, AnnotatedPyccelSymbol):
                    annotation = self._visit(lhs.annotation)
                    if len(annotation.type_list) != 1 or annotation.type_list[0].class_type != init_method.arguments[0].var.class_type:
                        errors.report(f"Unexpected type annotation in creation of {cls_def.name}",
                                symbol=annotation, severity='error')
                    lhs = lhs.name

                cls_variable = self._assign_lhs_variable(lhs, d_var,
                                        rhs = init_method.results.var,
                                        new_expressions = new_expression,
                                        is_augassign = False)
                args = (FunctionCallArgument(cls_variable), *args)

                args = self._sort_function_call_args(init_method.arguments, args)
                self._check_argument_compatibility(args, init_method.arguments,
                                init_method, init_method.is_elemental)

                new_expr = ConstructorCall(init_method, args, cls_variable)

                for a, f_a in zip(new_expr.args, init_method.arguments):
                    if f_a.persistent_target:
                        val = a.value
                        if isinstance(val, Variable):
                            a.value.is_target = True
                            self._indicate_pointer_target(cls_variable, a.value, expr.get_user_nodes(Assign)[0])
                        else:
                            errors.report(f"{val} cannot be passed to class constructor call as target. Please create a temporary variable.",
                                    severity='error', symbol=expr)
            else:
                cls_variable = self._assign_lhs_variable(lhs, d_var,
                                        rhs = None,
                                        new_expressions = new_expression,
                                        is_augassign = False)
                new_expr = EmptyNode()
            self._additional_exprs[-1].extend(new_expression)
            self._allocs[-1].add(cls_variable)
            return new_expr
        else:

            if func is None and name in self._context_dict:
                env_var = self._context_dict[name]
                if isinstance(env_var, FunctionDef):
                    func = env_var
                else:
                    func = builtin_functions_dict.get(env_var.__name__, None)
                    if func is not None:
                        func = PyccelFunctionDef(env_var.__name__, func)
                    mod_name = env_var.__module__
                    recognised_mod = (mod_name is not None) and recognised_source(mod_name)

                    if func is None and recognised_mod:
                        pyccel_stage.set_stage('syntactic')
                        import_node = Import(mod_name, name)
                        pyccel_stage.set_stage('semantic')
                        self._additional_exprs[-1].append(self._visit(import_node))
                        func = self.scope.find(name)

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
                lhs_scope_name = lhs.name.name[-1]
            else:
                insert_scope = self.scope
                lhs_scope_name = lhs.name
            lhs = lhs.name

            if isinstance(semantic_lhs_var.class_type, TypeAlias):
                pyccel_stage.set_stage('syntactic')
                if isinstance(rhs, LiteralString):
                    try:
                        annotation = types_meta.model_from_str(rhs.python_value)
                    except TextXSyntaxError as e:
                        errors.report(f"Invalid header. {e.message}",
                                symbol = expr, severity = 'fatal')
                    rhs = annotation.expr
                    rhs.set_current_ast(expr.python_ast)
                elif not isinstance(rhs, (SyntacticTypeAnnotation, FunctionTypeAnnotation,
                                          VariableTypeAnnotation, UnionTypeAnnotation)):
                    rhs = SyntacticTypeAnnotation(rhs)
                pyccel_stage.set_stage('semantic')
                type_annot = self._visit(rhs)
                self.scope.insert_symbolic_alias(lhs, type_annot)
                return EmptyNode()

            try:
                insert_scope.insert_variable(semantic_lhs_var, lhs_scope_name)
            except RuntimeError as e:
                errors.report(e, symbol=expr, severity='error')

        if isinstance(rhs, (PythonTuple, PythonList)):
            assign_elems = None
            if isinstance(lhs, PythonTuple):
                # Create variables to handle swap expressions
                unsaved_vars = set()
                pyccel_stage.set_stage('syntactic')
                unsaved_vars = set(rhs.get_attribute_nodes((PyccelSymbol, DottedName, IndexedElement),
                                                            excluded_nodes = (FunctionDef,)))
                pyccel_stage.set_stage('semantic')

                # Test if the expression describes a basic swap or if the rhs contains expressions
                # (e.g. arithmetic expressions or further tuples)
                # using variables from the left-hand side.
                modified_vars = set(lhs.get_attribute_nodes((PyccelSymbol, DottedName, IndexedElement)))
                used_vars = set(rhs.get_attribute_nodes((PyccelSymbol, DottedName, IndexedElement),
                                    excluded_nodes = (FunctionDef,)))
                trivial_assign = len(modified_vars.intersection(unsaved_vars)) == 0
                all_indexed_are_simple = all(all(isinstance(idx, (PyccelSymbol, DottedName, Literal)) for idx in elem.indices)
                                             for elem in modified_vars if isinstance(elem, IndexedElement))
                if not trivial_assign and (used_vars.intersection(modified_vars).difference(unsaved_vars) or not all_indexed_are_simple):
                    errors.report("Assign statement is too complex. It seems that some of the variables used non-trivially on the right-hand side appear on the left-hand side.",
                            severity='error', symbol=expr)

                assign_elems = []
                for i, l in enumerate(lhs):
                    r = rhs[i]
                    # Get unsaved variables that are still needed
                    pyccel_stage.set_stage('syntactic')
                    tmp_rhs_tuple = PythonTuple(*rhs.args[i+1:])
                    unsaved_vars = set(tmp_rhs_tuple.get_attribute_nodes((PyccelSymbol, DottedName, IndexedElement),
                                                                         excluded_nodes = (FunctionDef,)))
                    pyccel_stage.set_stage('semantic')

                    # If the lhs element has not yet been saved to a variable create a new
                    # variable to hold this value
                    if l in unsaved_vars:
                        temp = self.scope.get_new_name()
                        pyccel_stage.set_stage('syntactic')
                        local_assign = Assign(temp, l, python_ast = expr.python_ast)
                        pyccel_stage.set_stage('semantic')
                        assign_elems.append(self._visit(local_assign))
                        # Save the variable containing the value to rhs so it can be
                        # used when it appears in the assignment
                        if isinstance(l, IndexedElement):
                            # A list is required for IndexedElements as they are not singletons
                            l_list = [r for r in rhs.get_attribute_nodes(IndexedElement) if r == l]
                        else:
                            l_list = [l]
                        for l_elem in l_list:
                            rhs.substitute(l_elem, temp)
                        if r == l:
                            r = temp

                    # Check for a replacement right-hand side if the rhs is found among the lhs variables
                    pyccel_stage.set_stage('syntactic')
                    local_assign = Assign(l, r, python_ast = expr.python_ast)
                    pyccel_stage.set_stage('semantic')
                    assign_elems.append(self._visit(local_assign))
            elif isinstance(lhs, (PyccelSymbol, DottedName)):
                semantic_lhs = self.scope.find(lhs)
                if semantic_lhs and isinstance(semantic_lhs.class_type, InhomogeneousTupleType):
                    pyccel_stage.set_stage('syntactic')
                    syntactic_assign_elems = [Assign(IndexedElement(lhs,i), r, python_ast=expr.python_ast) for i, r in enumerate(rhs)]
                    pyccel_stage.set_stage('semantic')
                    assign_elems = [self._visit(a) for a in syntactic_assign_elems]
            elif isinstance(lhs, IndexedElement):
                semantic_lhs = self._visit(lhs)
                if isinstance(semantic_lhs.class_type, InhomogeneousTupleType):
                    pyccel_stage.set_stage('syntactic')
                    syntactic_assign_elems = [Assign(IndexedElement(lhs,i), r, python_ast=expr.python_ast) for i, r in enumerate(rhs)]
                    pyccel_stage.set_stage('semantic')
                    assign_elems = [self._visit(a) for a in syntactic_assign_elems]

            if assign_elems is not None:
                return CodeBlock([l for a in assign_elems for l in (a.body if isinstance(a, CodeBlock) else [a])])

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
            rhs = self._visit(rhs)
            if isinstance(rhs, (PythonMap, PythonZip, PythonEnumerate, PythonRange)):
                errors.report(f"{type(rhs)} cannot be saved to variables", symbol=expr, severity='fatal')

        else:
            rhs = self._visit(rhs)

        if isinstance(rhs, NumpyResultType):
            errors.report("Cannot assign a datatype to a variable.",
                    symbol=expr, severity='error')

        # Checking for the result of _build_ListExtend or _build_PythonSetFunction
        if isinstance(rhs, (For, CodeBlock, ConstructorCall, EmptyNode)):
            return rhs

        elif isinstance(rhs, FunctionCall):
            func = rhs.funcdef
            results = func.results.var
            if results:
                d_var = self._infer_type(results)
            elif expr.lhs.is_temp:
                return rhs
            else:
                raise errors.report("Cannot assign result of a function without a return",
                        severity='fatal', symbol=expr)

            if isinstance(results.class_type, NumpyNDArrayType) and isinstance(lhs, IndexedElement):
                temp = self.scope.get_new_name()
                semantic_temp = self._assign_lhs_variable(temp, d_var, rhs, new_expressions)
                new_expressions.append(Assign(semantic_temp, rhs))
                rhs = semantic_temp
                errors.report((f"Saving the result of the function {func.name} to a slice requires unnecessary "
                               "data allocation and copies. This has a performance cost. Consider modifying "
                               f"{func.name} so {lhs} can be passed as an argument whose contents are modified."),
                        severity='warning', symbol=expr)

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

        elif isinstance(rhs, TypingTypeVar):
            self.scope.insert_symbolic_alias(lhs, rhs)
            return EmptyNode()

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
            lhs = self._assign_lhs_variable(lhs, d_var, rhs, new_expressions,
                    arr_in_multirets = (isinstance(rhs, FunctionCall) and \
                                        not getattr(rhs.funcdef, 'is_elemental', False)))

            # If lhs is a purely symbolic object to link tuple elements to their containing tuple
            # then no semantic object should be returned
            # This can happen when returning an inhomogeneous tuple
            if isinstance(rhs, PythonTuple) and isinstance(lhs.class_type, InhomogeneousTupleType):
                for li, ri in zip(lhs, rhs):
                    li_var = self.scope.collect_tuple_element(li)
                    if li_var == ri:
                        new_expressions = [n for n in new_expressions if not n.is_user_of(li_var)]

        # Handle assignment to multiple variables
        elif isinstance(lhs, (PythonTuple, PythonList)):
            if isinstance(rhs, FunctionCall):
                new_lhs = []
                for i,(l,r) in enumerate(zip(lhs, rhs.funcdef.results.var)):
                    d = self._infer_type(r)
                    new_lhs.append( self._assign_lhs_variable(l, d, r, new_expressions,
                                                    arr_in_multirets=r.rank>0 ) )
                if not isinstance(rhs.class_type, InhomogeneousTupleType):
                    rhs_var = self.scope.get_temporary_variable(rhs.funcdef.results.var)
                    new_expressions.append(Assign(rhs_var, rhs))
                    rhs = rhs_var
                lhs = PythonTuple(*new_lhs)
            elif isinstance(rhs, PyccelFunction):
                assert isinstance(rhs.class_type, InhomogeneousTupleType)
                r_iter = [self.scope.collect_tuple_element(v) for v in rhs]
                new_lhs = []
                for i,(l,r) in enumerate(zip(lhs, r_iter)):
                    d = self._infer_type(r)
                    new_lhs.append( self._assign_lhs_variable(l, d, r, new_expressions,
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
                    local_assign = Assign(l, r, python_ast = expr.python_ast)
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
        elif isinstance(lhs, (IndexedElement, DictGetItem)):
            is_pointer = False
        elif isinstance(lhs, (PythonTuple, PythonList)):
            is_pointer = any(l.is_alias for l in lhs if isinstance(lhs, Variable))
        else:
            raise NotImplementedError()

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
                if (isinstance(l.class_type, InhomogeneousTupleType) or isinstance(l, PythonTuple)) \
                        and not isinstance(r, (FunctionCall, PyccelFunction)):
                    new_lhs.extend(self.scope.collect_tuple_element(v) for v in l)
                    new_rhs.extend(self.scope.collect_tuple_element(r[i]) \
                            for i in range(l.shape[0]))
                    # Repeat step to handle tuples of tuples of etc.
                    unravelling = True
                elif isinstance(l, Variable) and isinstance(l.class_type, InhomogeneousTupleType):
                    new_lhs.append(PythonTuple(*self.scope.collect_all_tuple_elements(l)))
                    new_rhs.append(r)
                    # Repeat step to handle tuples of tuples of etc.
                    unravelling = True
                elif isinstance(l, Variable) and isinstance(r.class_type, InhomogeneousTupleType):
                    new_lhs.extend(l[i] for i in range(len(r.class_type)))
                    new_rhs.extend(self.scope.collect_tuple_element(ri) for ri in r)
                    # Repeat step to handle tuples of tuples of etc.
                    unravelling = True
                elif l is not r:
                    # Manage a non-tuple assignment

                    # Manage memory for optionals
                    if isinstance(l, Variable) and l.is_optional:
                        if l in self._optional_params:
                            # Collect temporary variable which provides
                            # allocated memory space for this optional variable
                            new_lhs.append(self._optional_params[l])
                        else:
                            # Create temporary variable to provide allocated
                            # memory space before assigning to the pointer value
                            # (may be NULL)
                            tmp_var = self.scope.get_temporary_variable(l,
                                    name = l.name+'_loc', is_optional = False,
                                    is_argument = False)
                            self._optional_params[l] = tmp_var
                            l = tmp_var

                    if isinstance(r, ConstructorCall):
                        # Manage a ConstructorCall in a tuple assignment.
                        # In this case a temporary variable is created which must be
                        # replaced with the tuple element.
                        cls_var = r.cls_variable
                        if cls_var.is_temp:
                            r.substitute(cls_var, l)
                            self._allocs[-1].remove(cls_var)
                            self.scope.remove_variable(cls_var)
                            self._allocs[-1].add(l)
                        new_expressions.append(r)
                    else:
                        new_lhs.append(l)
                        new_rhs.append(r)
            lhs = new_lhs
            rhs = new_rhs

        # Examine each assign and determine assign type (Assign, AliasAssign, etc)
        for l, r in zip(lhs,rhs):
            if isinstance(l, PythonTuple):
                for li in l:
                    if isinstance(li.class_type, FinalType):
                        # If constant (can't use annotations on tuple assignment)
                        errors.report("Cannot modify variable marked as Final",
                            bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                            symbol=li, severity='error')
            else:
                if isinstance(l.class_type, FinalType) and (not isinstance(expr.lhs, AnnotatedPyccelSymbol) or \
                        any(not isinstance(u, (Allocate, PyccelArrayShapeElement)) for u in l.get_all_user_nodes())):
                    # If constant and not the initialising declaration of a constant variable
                    errors.report("Cannot modify variable marked as Final",
                        bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                        symbol=l, severity='error')
            if isinstance(expr, AugAssign):
                new_expr = AugAssign(l, expr.op, r)
            else:
                is_pointer_i = l.is_alias if isinstance(l, Variable) else is_pointer
                new_expr = Assign(l, r)

                if is_pointer_i:
                    if isinstance(r, FunctionCall):
                        funcdef = r.funcdef
                        target_r_idx = funcdef.result_pointer_map[funcdef.results.var]
                        for ti in target_r_idx:
                            self._indicate_pointer_target(l, r.args[ti].value, expr)
                        l.remove_user_node(new_expr)
                        r.remove_user_node(new_expr)
                        new_expr = AliasAssign(l, r)
                    else:
                        self._indicate_pointer_target(l, r, expr)

                        if not isinstance(r.class_type, NumpyNDArrayType) and not isinstance(r, Variable):
                            mem_var = get_managed_memory_object(l)
                            new_expr = UnpackManagedMemory(l, r, mem_var)
                        else:
                            new_expr = AliasAssign(l, r)

                elif isinstance(l.class_type, SymbolicType):
                    errors.report(PYCCEL_RESTRICTION_TODO,
                                  symbol=expr,
                                  severity='fatal')
                elif isinstance(r, (PythonList, PythonSet, PythonTuple, PythonDict)):
                    self._indicate_pointer_target(l, r, expr)

            new_expressions.append(new_expr)

        if expr.lhs == '__all__':
            self.scope.remove_variable(lhs[0])
            self._allocs[-1].discard(lhs[0])
            if isinstance(lhs[0].class_type, HomogeneousListType):
                # Remove the last element of the errors (if it is a warning)
                # This will be the list of list warning
                try:
                    error_info_map = errors.error_info_map[os.path.basename(errors.target)]
                    if error_info_map[-1].severity == 'warning':
                        error_info_map.pop()
                except KeyError:
                    # There may be a KeyError if this is not the first time that this DataType
                    # of list of rank>0 is created.
                    pass
            return AllDeclaration(new_expressions[-1].rhs)

        if (len(new_expressions)==1):
            new_expressions = new_expressions[0]

            return new_expressions
        else:
            result = CodeBlock(new_expressions)
            return result

    def _visit_AugAssign(self, expr):
        lhs = self._visit(expr.lhs)
        if isinstance(lhs.class_type, FinalType):
            errors.report("Cannot modify variable marked as Final",
                bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                symbol=lhs, severity='error')
        rhs = self._visit(expr.rhs)
        operator = expr.pyccel_operator
        new_expressions = []
        try:
            test_node = operator(lhs, rhs)
        except TypeError:
            test_node = None
        if test_node:
            lhs.remove_user_node(test_node, invalidate = False)
            if test_node in rhs.get_all_user_nodes():
                rhs.remove_user_node(test_node, invalidate = False)
            else:
                assert isinstance(rhs.current_user_node, PyccelAssociativeParenthesis)
                mid = rhs.current_user_node
                rhs.remove_user_node(mid, invalidate=False)
            lhs = self._assign_lhs_variable(expr.lhs, self._infer_type(test_node), test_node,
                    new_expressions, is_augassign = True)
            lhs = self._optional_params.get(lhs, lhs)
            aug_assign = AugAssign(lhs, expr.op, rhs)
        else:
            magic_method_name = magic_method_map[operator]
            increment_magic_method_name = '__i' + magic_method_name[2:]
            class_type = lhs.class_type
            class_base = self.get_cls_base(class_type)
            increment_magic_method = class_base.get_method(increment_magic_method_name)
            args = [FunctionCallArgument(lhs), FunctionCallArgument(rhs)]
            if increment_magic_method:
                lhs = self._optional_params.get(lhs, lhs)
                return self._handle_function(expr, increment_magic_method, args)
            magic_method = class_base.get_method(magic_method_name, expr)
            operator_node = self._handle_function(expr, magic_method, args)
            lhs = self._assign_lhs_variable(expr.lhs, self._infer_type(operator_node), test_node,
                    new_expressions, is_augassign = True)
            lhs = self._optional_params.get(lhs, lhs)
            aug_assign = Assign(lhs, operator_node)
        if new_expressions:
            return CodeBlock(new_expressions + [aug_assign])
        else:
            return aug_assign

    def _visit_For(self, expr):

        scope = self.create_new_loop_scope()

        new_expr = []

        # treatment of the index/indices
        target, iterable = self._get_for_iterators(expr.iterable, expr.target, new_expr, expr)

        body = self._visit(expr.body)

        self.exit_loop_scope()

        if isinstance(iterable, Product):
            for_expr = body
            scopes = self.scope.create_product_loop_scope(scope, len(target))

            for t, i, r, s in zip(target[::-1], iterable.loop_counters[::-1], iterable.get_python_iterable_item()[::-1], scopes[::-1]):
                # Create Variable iterable
                loop_iter = VariableIterator(r.base)
                loop_iter.set_loop_counter(i)

                # Create a For loop for each level of the Product
                for_expr = For((t,), loop_iter, for_expr, scope=s)
                for_expr.end_annotation = expr.end_annotation
                for_expr = [for_expr]
            for_expr = for_expr[0]
        else:
            for_expr = For(target, iterable, body, scope=scope)
            for_expr.end_annotation = expr.end_annotation
        return for_expr

    def _visit_FunctionalFor(self, expr):
        """
        Visit and transform a FunctionalFor AST node into an equivalent code block.

        This method processes a `FunctionalFor` expression and transforms the loop structure
        into a corresponding code block.

        Parameters
        ----------
        expr : pyccel.ast.functionalexpr.FunctionalFor
            The FunctionalFor AST node.

        Returns
        -------
        pyccel.ast.basic.CodeBlock
            A code block containing the equivalent loops and necessary variable allocations for the given `FunctionalFor` expression.
        """
        target  = expr.expr
        indices = []
        dims = []
        idx_subs = {}
        tmp_used_names = self.scope.all_used_symbols.copy()
        i = 0
        loops = list(expr.loops)

        # Inner function to handle PythonNativeInt variables
        def handle_int_loop_variable(var_name, var_scope):
            indices.append(var_name)
            var = self._create_variable(var_name, PythonNativeInt(), None, {}, insertion_scope=var_scope)
            return var

        # Inner function to handle iterable variables
        def handle_iterable_variable(var_name, element, var_scope):
            indices.append(var_name)
            dvar = self._infer_type(element)
            class_type = dvar.pop('class_type')
            if class_type.rank > 0:
                class_type = class_type.switch_rank(class_type.rank - 1)
                dvar['shape'] = dvar['shape'][1:]
            if class_type.rank == 0:
                dvar['shape'] = None
                dvar['memory_handling'] = 'stack'
            var = self._create_variable(var_name, class_type, None, dvar, insertion_scope=var_scope)
            return var

        for loop, condition in zip(loops, expr.conditions):
            if condition:
                loop.insert2body(condition)

        while len(loops) > 1:
            outer_loop = loops.pop()
            inserted_into = loops[-1]
            if inserted_into.body.body:
                inserted_into.body.body[0].blocks[0].body.insert2body(outer_loop)
            else:
                inserted_into.insert2body(outer_loop)

        body = loops[0]

        while isinstance(body, (For, If)):

            if isinstance(body, If):
                body = None if not body.blocks[0].body.body else body.blocks[0].body.body[0]
                continue

            stop = None
            start = LiteralInteger(0)
            step = LiteralInteger(1)
            variables = []
            a = self._get_iterable(self._visit(body.iterable))
            if isinstance(a, PythonRange):
                var_name = self.scope.get_expected_name(expr.indices[i])
                variables.append(handle_int_loop_variable(var_name, body.scope))
                start = a.start
                stop  = a.stop
                step  = a.step

            elif isinstance(a, PythonEnumerate):
                var_name1 = self.scope.get_expected_name(expr.indices[i][0])
                var_name2 = self.scope.get_expected_name(expr.indices[i][1])
                variables.append(handle_int_loop_variable(var_name1, body.scope))
                variables.append(handle_iterable_variable(var_name2, a.element, body.scope))
                stop = a.element.shape[0]

            elif isinstance(a, PythonZip):
                for idx, arg in enumerate(a.args):
                    var = self.scope.get_expected_name(expr.indices[i][idx])
                    variables.append(handle_iterable_variable(var, arg, body.scope))
                stop = a.get_range().stop

            elif isinstance(a, VariableIterator):
                var = self.scope.get_expected_name(expr.indices[i])
                variables.append(handle_iterable_variable(var, a.variable, body.scope))
                stop = a.variable.shape[0]

            else:
                errors.report(PYCCEL_RESTRICTION_TODO,
                              bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                              severity='fatal')

            for var in variables:
                existing_var = self.scope.find(var.name, 'variables')
                if var.name == expr.lhs:
                    errors.report(f"Variable {var} has the same name as the left hand side",
                            symbol = expr, severity='fatal')
                if existing_var or var.name ==  expr.lhs:
                    if self._infer_type(existing_var)['class_type'] != var.class_type:
                        errors.report(f"Variable {var} already exists with different type",
                                symbol = expr, severity='fatal')
                else:
                    self.scope.insert_variable(var)

            step  = pyccel_to_sympy(step , idx_subs, tmp_used_names)
            start = pyccel_to_sympy(start, idx_subs, tmp_used_names)
            stop  = pyccel_to_sympy(stop , idx_subs, tmp_used_names)
            size = (stop - start) / step
            if (step != 1):
                size = ceiling(size)
            body = None if not body.body.body else body.body.body[0]
            dims.append((size, step, start, stop))
            i += 1

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
                          symbol=expr,
                          severity='fatal')

        target = self._visit(target)
        d_var = self._infer_type(target)

        class_type = d_var['class_type']

        if class_type is GenericType():
            errors.report(LIST_OF_TUPLES,
                          bounding_box=(self.current_ast_node.lineno, self.current_ast_node.col_offset),
                          severity='fatal')

        d_var['memory_handling'] = 'heap'
        target_type_name = 'list' if not expr.target_type else expr.target_type
        if isinstance(target_type_name, DottedName):
            lhs = target_type_name.name[0] if len(target_type_name.name) == 2 \
                    else DottedName(*target_type_name.name[:-1])
            first = self._visit(lhs)
            if isinstance(first, Module):
                conversion_func = first[target_type_name.name[-1]]
            else:
                conversion_func = None
        else:
            conversion_func = self.scope.find(target_type_name, 'functions')
            if conversion_func is None:
                if target_type_name in builtin_functions_dict:
                    conversion_func = PyccelFunctionDef(target_type_name,
                                            builtin_functions_dict[target_type_name])
        if conversion_func is None:
            errors.report("Unrecognised output type from functional for.\n"+PYCCEL_RESTRICTION_TODO,
                          symbol=expr,
                          severity='fatal')
        if target_type_name != 'list' and any(cond is not None for cond in  expr.conditions):
            errors.report("Cannot handle if statements in list comprehensions if lhs is a numpy array.\
                          List length cannot be calculated.\n" + PYCCEL_RESTRICTION_TODO,
                           symbol=expr, severity='error')
        try:
            class_type = type_container[conversion_func.cls_name].get_new(class_type)
        except TypeError:
            if class_type.rank > 0:
                errors.report("ND comprehension expressions cannot be saved directly to an array yet.\n"+PYCCEL_RESTRICTION_TODO,
                              symbol=expr,
                              severity='fatal')

            class_type = type_container[conversion_func.cls_name].get_new(numpy_process_dtype(class_type), rank=1, order=None)
        d_var['class_type'] = class_type
        d_var['shape'] = (dim,)
        d_var['cls_base'] = self.get_cls_base(class_type)

        # ...
        # TODO [YG, 30.10.2020]:
        #  - Check if we should allow the possibility that is_stack_array=True
        # ...
        lhs_symbol = expr.lhs
        ne = []
        lhs = self._assign_lhs_variable(lhs_symbol, d_var, rhs=expr, new_expressions=ne)
        lhs_alloc = ne[0] if ne else EmptyNode()

        if isinstance(target, PythonTuple) and not target.is_homogeneous:
            errors.report(LIST_OF_TUPLES, symbol=expr, severity='error')

        target.invalidate_node()
        operations = []
        assign = None
        target_conversion_func = self._visit(target_type_name)
        if (isinstance(target_conversion_func, PyccelFunctionDef)
                and target_conversion_func.cls_name is NumpyArray) or isinstance(lhs_alloc, EmptyNode):
            old_index   = expr.index
            new_index   = self.scope.get_new_name()
            expr.substitute(old_index, new_index, is_equivalent = lambda x,y: x is y)
            array_ops = expr.operations['numpy_array']
            for operation in array_ops:
                operation.substitute(old_index, new_index, is_equivalent = lambda x,y: x is y)
            assign = array_ops[0]
            assign = self._visit(array_ops[0])
            operations.extend(array_ops[1:])
            index = new_index
            index = self._visit(index)
        elif target_conversion_func == "'list'":
            index = None
            operations.extend(expr.operations['list'])
        else:
            errors.report("Unrecognised target for functional for.\n"+PYCCEL_RESTRICTION_TODO,
                          symbol=expr,
                          severity='fatal')

        if expr.loops[-1].body.body:
            for operation in operations:
                expr.loops[-1].body.body[0].blocks[0].body.insert2body(self._visit(operation))
        else:
            for operation in operations:
                expr.loops[-1].insert2body(self._visit(operation))


        loops = [self._visit(i) for i in loops]
        if assign:
            loops = [assign, *loops]

        l = loops[-1]
        cnt = 0
        for idx in indices:
            assert isinstance(l, For)
            if idx.is_temp:
                self.scope.remove_variable(l.target[cnt])
                l.substitute(l.target[cnt], idx_subs[idx])
            cnt += 1
            if cnt == len(l.target):
                if l.body.body:
                    if isinstance(l.body.body[0], If):
                        l = l.body.body[0].blocks[0].body.body[0]
                    else:
                        l = l.body.body[0]
                cnt = 0

        return CodeBlock([lhs_alloc, FunctionalFor(loops, lhs=lhs, index=index, indices=expr.indices, target_type=target_type_name, conditions=expr.conditions)])

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

        symbol_map = {}
        used_names = self.scope.all_used_symbols.copy()
        try:
            sympy_cond = pyccel_to_sympy(cond, symbol_map, used_names)
        except TypeError:
            sympy_cond = 'unknown'

        if sympy_cond == sp_False():
            return IfSection(LiteralFalse(), CodeBlock([]))
        elif sympy_cond == sp_True():
            cond = LiteralTrue()

        if cond.dtype is not PythonNativeBool():
            cond = PythonBool(cond)
            cond.set_current_ast(cond.python_ast or expr.python_ast)

        body = self._visit(expr.body)

        if_block = expr.get_direct_user_nodes(lambda u: isinstance(u, If))[0]
        is_last_block = expr is if_block.blocks[-1]

        def treat_condition(cond, body):
            """
            Run through the condition of the If to try to extract `if a is not None`
            conditions. These must be done in their own line in low-level languages.
            """
            is_not_conds = cond.get_attribute_nodes(PyccelIsNot)
            non_conditional_list = [c for c in is_not_conds if c.args[1] is Nil()]
            for non_conditional in non_conditional_list:
                v = non_conditional.args[0]
                var_use = v.get_direct_user_nodes(cond.is_user_of)
                # If variable is only used in `a is not None` the condition is ok
                if len(var_use) > 1:
                    # If `a is not None` is in an `and` we can split this into valid conditions
                    if isinstance(cond, PyccelAnd) and non_conditional in cond.args:
                        remaining_cond = PyccelAnd(*[a for a in cond.args if a is not non_conditional]) \
                                if len(cond.args) > 2 else next(a for a in cond.args if a is not non_conditional)
                        remaining_cond.set_current_ast(cond.python_ast)
                        cond_var = self.scope.get_temporary_variable(PythonNativeBool(),
                                                                     self.scope.get_new_name('condition'))
                        treated_remaining_cond, body = treat_condition(remaining_cond, body)
                        if is_last_block:
                            # if in the last block create an if in the current if
                            body = [If(IfSection(treated_remaining_cond, body))]
                            cond = non_conditional
                        else:
                            # Otherwise evaluate the condition before the if block
                            self._additional_exprs[-1].append(If(IfSection(non_conditional, [Assign(cond_var, treated_remaining_cond)]),
                                        IfSection(LiteralTrue(), [Assign(cond_var, LiteralFalse())])))
                            cond = cond_var
                        return cond, body
                    else:
                        errors.report("Cannot evaluate condition. Checking if a variable is present must be done before using the variable",
                                      severity='error', symbol=cond)
            return cond, body

        return IfSection(*treat_condition(cond, body))

    def _visit_If(self, expr):
        args = []

        for b in expr.blocks:
            new_b = self._visit(b)
            cond = new_b.condition
            if not isinstance(cond, LiteralFalse):
                args.append(new_b)
            if isinstance(cond, LiteralTrue):
                if len(args) == 1:
                    return new_b.body
                break

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
            if isinstance(cond, LiteralTrue):
                return value_true
            elif isinstance(cond, LiteralFalse):
                return value_false
            else:
                return IfTernaryOperator(cond, value_true, value_false)

    def _visit_Return(self, expr):

        results     = expr.expr
        f_name      = self.current_function_name
        if isinstance(f_name, DottedName):
            f_name = f_name.name[-1]

        func = self._current_function[-1]

        original_name = self.scope.get_python_name(f_name)
        if original_name.startswith('__i') and ('__'+original_name[3:]) in magic_method_map.values():
            valid_return = isinstance(expr.expr, PyccelSymbol) and expr.stmt is None and len(func.arguments) > 0
            if valid_return:
                out = self._visit(expr.expr)
                expected = func.arguments[0].var
                valid_return &= (out == expected)
            if valid_return:
                return EmptyNode()
            else:
                errors.report("Increment functions must return the class instance",
                        severity='fatal', symbol=expr)

        return_objs = func.results
        return_var = return_objs.var
        if isinstance(return_var, AnnotatedPyccelSymbol):
            return_var = return_var.name
        elif isinstance(return_var, Variable):
            return_var = self.scope.get_python_name(return_var.name)

        assigns     = []
        if return_var != results:
            # Create a syntactic object to visit
            pyccel_stage.set_stage('syntactic')
            syntactic_assign = Assign(return_var, results, python_ast=expr.python_ast)
            pyccel_stage.set_stage('semantic')

            a = self._visit(syntactic_assign)
            if not isinstance(a, Assign) or a.lhs != a.rhs:
                assigns.append(a)
                if isinstance(a, ConstructorCall):
                    a.cls_variable.is_temp = False
            else:
                a.invalidate_node()

        results = self._visit(return_var)

        # add the Deallocate node before the Return node and eliminating the Deallocate nodes
        # the arrays that will be returned.
        results_vars = self.scope.collect_all_tuple_elements(results)
        self._check_pointer_targets(results_vars)
        code = assigns + [Deallocate(i) for i in self._allocs[-1] if i not in results_vars]
        if results is Nil():
            results = None
        if code:
            expr  = Return(results, CodeBlock(code))
        else:
            expr  = Return(results)
        return expr

    def _visit_FunctionDef(self, expr):
        """
        Semantically analyse the FunctionDef.

        Analyse the FunctionDef adding all necessary semantic information.

        Parameter
        ---------
        expr : FunctionDef|Interface
           The node that needs to be annotated.
           If we provide an Interface, this means that the function has been annotated partially,
           and we need to continue annotating the needed ones.
        """
        if expr.get_direct_user_nodes(lambda u: isinstance(u, CodeBlock)):
            errors.report("Functions can only be declared in modules, classes or inside other functions.",
                    symbol=expr, severity='error')

        python_name = expr.name

        current_class = expr.get_direct_user_nodes(lambda u: isinstance(u, ClassDef))
        cls_name = current_class[0].name if current_class else None
        insertion_scope = self.scope
        if cls_name:
            bound_class = self.scope.find(cls_name, 'classes', raise_if_missing = True)
            insertion_scope = bound_class.scope

        decorators         = expr.decorators.copy()

        existing_semantic_funcs = []
        assert not expr.is_semantic

        func = insertion_scope.functions.get(python_name, None)
        if func:
            if func.is_semantic:
                if self.is_stub_file:
                    # Only Interfaces should be revisited in a stub file
                    assert isinstance(func, Interface)
                    existing_semantic_funcs = [*func.functions]
                else:
                    return EmptyNode()
            insertion_scope.remove_function(python_name)
        if 'low_level' in decorators or (self.is_stub_file and not python_name.startswith('__')):
            if 'low_level' in decorators:
                low_level_decs = decorators['low_level']
                assert len(low_level_decs) == 1
                arg = low_level_decs[0].args[0].value
                assert isinstance(arg, LiteralString)
                name = PyccelSymbol(arg.python_value)
            else:
                name = python_name
            if 'overload' not in decorators:
                insertion_scope.remove_symbol(python_name)
                insertion_scope.insert_low_level_symbol(python_name, name)
        else:
            name = expr.scope.get_expected_name(python_name)

        new_semantic_funcs = []
        sub_funcs          = []
        func_interfaces    = []
        docstring          = self._visit(expr.docstring) if expr.docstring else expr.docstring
        is_pure            = expr.is_pure
        is_elemental       = expr.is_elemental
        is_private         = expr.is_private
        is_inline          = expr.is_inline

        not_used = [d for d in decorators if d not in (*def_decorators.__all__, 'property', 'overload')]
        if len(not_used) >= 1:
            errors.report(UNUSED_DECORATORS, symbol=', '.join(not_used), severity='warning')

        if any(a.annotation is None for a in expr.arguments):
            errors.report(MISSING_TYPE_ANNOTATIONS,
                    symbol=[a for a in expr.arguments if a.annotation is None], severity='error',
                          bounding_box = (self.current_ast_node.lineno, self.current_ast_node.col_offset))
            return EmptyNode()

        available_type_vars = {n:v for n,v in self._context_dict.items() if isinstance(v, typing.TypeVar)}
        available_type_vars.update(self.scope.collect_all_type_vars())
        used_type_vars = {}

        if any(a.annotation is None for a in expr.arguments):
            errors.report(MISSING_TYPE_ANNOTATIONS,
                    symbol=expr, severity='error')
            return EmptyNode()

        for a in expr.arguments:
            used_objs = a.annotation.get_attribute_nodes(PyccelSymbol)
            for o in used_objs:
                if o in available_type_vars:
                    used_type_vars[o] = available_type_vars[o]

        for o, t in used_type_vars.items():
            if isinstance(t, typing.TypeVar):
                pyccel_type_var = self.env_var_to_pyccel(t)
                used_type_vars[o] = pyccel_type_var
                global_scope = self.scope
                while global_scope.parent_scope:
                    global_scope = global_scope.parent_scope
                global_scope.insert_symbol(o)
                global_scope.insert_symbolic_alias(o, pyccel_type_var)

        possible_combinations = list(product(*[t.type_list for t in used_type_vars.values()]))

        argument_combinations = []
        type_var_indices = []
        for i,p in enumerate(possible_combinations):
            scope = self.create_new_function_scope(expr.name, '_', decorators = decorators,
                    used_symbols = expr.scope.local_used_symbols.copy(),
                    original_symbols = expr.scope.python_names.copy(),
                    symbolic_aliases = expr.scope.symbolic_aliases)
            for n, dtype in zip(used_type_vars, p):
                self.scope.insert_symbolic_alias(n, dtype)
            args = list(product(*[self._visit(a) for a in expr.arguments]))
            argument_combinations.extend(args)
            type_var_indices.extend([i]*len(args))
            self.exit_function_scope()

        # this for the case of a function without arguments => no headers
        interface_name = expr.scope.get_expected_name(python_name)
        interface_counter = 0
        is_interface = len(argument_combinations) > 1 or 'overload' in decorators
        for interface_idx, (arguments, type_var_idx) in enumerate(zip(argument_combinations, type_var_indices)):
            if is_interface and 'low_level' not in decorators:
                name, _ = self.scope.get_new_incremented_symbol(python_name, interface_idx)

            insertion_scope.python_names[name] = python_name

            scope = self.create_new_function_scope(python_name, name, decorators = decorators,
                    used_symbols = expr.scope.local_used_symbols.copy(),
                    original_symbols = expr.scope.python_names.copy(),
                    symbolic_aliases = expr.scope.symbolic_aliases)

            for n, dtype in zip(used_type_vars, possible_combinations[type_var_idx]):
                self.scope.insert_symbolic_alias(n, dtype)

            arg_dict  = {a.name:a.var for a in arguments}

            for a in arguments:
                a_var = a.var
                if isinstance(a_var, FunctionAddress):
                    self.insert_function(a_var)
                else:
                    self.scope.insert_variable(a_var, expr.scope.get_python_name(a.name))

            if arguments and arguments[0].bound_argument:
                if arguments[0].var.cls_base is not bound_class:
                    errors.report('Class method self argument does not have the expected type',
                            severity='error', symbol=arguments[0])
                for s in expr.scope.dotted_symbols:
                    base = s.name[0]
                    if base in arg_dict:
                        cls_base = arg_dict[base].cls_base
                        cls_base.scope.insert_symbol(DottedName(*s.name[1:]))

            results = expr.results
            if results.annotation:
                results = self._visit(expr.results)

            # insert the FunctionDef into the scope
            # to handle the case of a recursive function
            # TODO improve in the case of an interface
            recursive_func_obj = FunctionDef(name, arguments, [], results, scope = scope)
            self.insert_function(recursive_func_obj, insertion_scope)

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

            results = self._visit(results)
            if isinstance(results, EmptyNode):
                results = FunctionDefResult(Nil())

            if results.var is Nil():
                results_vars = []
            else:
                results_vars = self.scope.collect_all_tuple_elements(results.var)

            self._check_pointer_targets(results_vars)

            # Calling the Garbage collecting,
            # it will add the necessary Deallocate nodes
            # to the body of the function
            body.insert2body(*self._garbage_collector(body))

            # Determine local and global variables
            global_vars = list(self.get_variables(self.scope.parent_scope))
            global_vars = [g for g in global_vars if body.is_user_of(g)]

            # get the imports
            imports   = self.scope.imports['imports'].values()
            # Prefer dict to set to preserve order
            imports   = list({imp:None for imp in imports}.keys())

            # remove the FunctionDef from the function scope
            func_ = insertion_scope.find(python_name, 'functions')
            insertion_scope.remove_function(python_name)
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

            # Raise an error if one of the return arguments is an alias.
            pointer_targets = self._pointer_targets.pop()
            result_pointer_map = {}
            for r in results_vars:
                t = pointer_targets.get(r, ())
                if r.is_alias:
                    arg_vars = [a.var for a in arguments]
                    temp_targets = [target for target, _ in t if target not in arg_vars]
                    if temp_targets:
                        errors.report(UNSUPPORTED_POINTER_RETURN_VALUE,
                            symbol=r, severity='error')
                    else:
                        result_pointer_map[r] = [next(i for i,a in enumerate(arguments) if a.var == target) for target, _ in t]

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
                global_funcs = [f for f in body.get_attribute_nodes(FunctionDef) \
                        if self.scope.find(self.scope.get_python_name(f.name), 'functions')]
                func_kwargs['global_funcs'] = global_funcs
                cls = InlineFunctionDef
            else:
                cls = FunctionDef
            func = cls(name,
                    arguments,
                    body,
                    results,
                    **func_kwargs)
            if not is_recursive:
                recursive_func_obj.invalidate_node()

            if cls_name:
                # update the class methods
                if not is_interface:
                    bound_class.update_method(expr, func)

            new_semantic_funcs += [func]
            if expr.python_ast:
                func.set_current_ast(expr.python_ast)

        if existing_semantic_funcs:
            new_semantic_funcs = existing_semantic_funcs + new_semantic_funcs

        if len(new_semantic_funcs) == 1 and not is_interface:
            new_semantic_funcs = new_semantic_funcs[0]
            self.insert_function(new_semantic_funcs, insertion_scope)
        else:
            new_semantic_funcs = Interface(interface_name, new_semantic_funcs, syntactic_node=expr)
            if expr.python_ast:
                new_semantic_funcs.set_current_ast(expr.python_ast)
            if cls_name:
                bound_class.update_interface(expr, new_semantic_funcs)
            self.insert_function(new_semantic_funcs, insertion_scope)

        return EmptyNode()

    def _visit_InlineFunctionDef(self, expr, function_call_args, function_call):
        """
        Visit an inline function definition to add the code to the calling scope.

        Visit an inline function definition to add the code to the calling scope.
        The code is inlined at this stage.

        Parameters
        ----------
        expr : InlineFunctionDef
            The inline function definition being called.
        function_call_args : list[FunctionDefArgument]
            The semantic arguments passed to the function.
        function_call : FunctionCall
            The syntactic function call being expanded to a function definition.
        """
        assign = function_call.get_direct_user_nodes(lambda a: isinstance(a, Assign) and not isinstance(a, AugAssign))
        self._current_function.append(expr)
        if assign:
            lhs = assign[-1].lhs
        else:
            lhs = self.scope.get_new_name()
        # Build the syntactic body
        replace_map = {}

        pyccel_stage.set_stage('syntactic')
        imports = list(expr.imports)
        global_scope_import_targets = {}
        if expr.is_imported:
            mod_name = expr.get_direct_user_nodes(lambda m: isinstance(m, Module))[0].name
            mod = self.d_parsers[mod_name].semantic_parser.ast

            global_symbols = set()
            to_examine = [expr.body]
            while to_examine:
                remaining_to_examine = []
                for ex in to_examine:
                    if isinstance(ex, DottedName) and isinstance(ex.name[-1], FunctionCall):
                        global_symbols.add(ex.name[0])
                        remaining = ex.name[-1].get_attribute_nodes((PyccelSymbol, DottedName))
                        remaining.remove(ex.name[-1].funcdef)
                        global_symbols.update(s for s in remaining if not isinstance(s, DottedName))
                        remaining_to_examine.extend(s for s in remaining if isinstance(s, DottedName))
                    else:
                        symbols = ex.get_attribute_nodes((PyccelSymbol, DottedName))
                        global_symbols.update(s for s in symbols if not isinstance(s, DottedName))
                        remaining_to_examine.extend(s for s in symbols if isinstance(s, DottedName))
                to_examine = remaining_to_examine

            global_symbols.difference_update(expr.scope.local_used_symbols)

            for v in global_symbols:
                import_mod_name = mod_name
                if mod.scope.find(v):
                    imported_obj = None
                    for import_type in mod.scope.imports.values():
                        if v in import_type:
                            imported_obj = import_type[v]
                            break
                    if isinstance(imported_obj, Module):
                        # Insert an imported module as a new Import object
                        new_v = v
                        if self.scope.symbol_in_use(v):
                            new_v = self.scope.get_new_name(v)
                            replace_map[v] = new_v
                        source = mod.scope.find(v, 'imports').source
                        mod_import = source if source == new_v else AsName(source, new_v)
                        imports.append(Import(mod_import))
                    else:
                        if imported_obj:
                            import_mod_name = imported_obj.get_direct_user_nodes(lambda m: isinstance(m, Module))[0].name
                        if self.scope.symbol_in_use(v):
                            new_v = self.scope.get_new_name(self.scope.get_expected_name(v))
                            replace_map[v] = new_v
                            global_scope_import_targets.setdefault(import_mod_name, []).append(AsName(v, new_v))
                        else:
                            global_scope_import_targets.setdefault(import_mod_name, []).append(v)

        # Swap in the function call arguments to replace the variables representing
        # the arguments of the inlined function
        res_vars = ()
        if expr.results:
            # Swap in the result of the function to replace the variable representing
            # the result of the inlined function
            res_var = expr.results.var
            if isinstance(res_var, AnnotatedPyccelSymbol):
                res_var = res_var.name
            if isinstance(lhs, PyccelSymbol):
                replace_map[res_var] = lhs
                res_vars = (res_var,)

        func_args = [a.var for a in expr.arguments]
        func_args = [a.name if isinstance(a, AnnotatedPyccelSymbol) else a for a in func_args]

        # Ensure local variables will be recognised and use a name that is not already in use
        for v in expr.scope.local_used_symbols:
            if v != expr.name and v not in res_vars:
                if self.scope.symbol_in_use(v):
                    new_v = self.scope.get_new_name(self.scope.get_expected_name(v))
                    replace_map[v] = new_v
                else:
                    self.scope.insert_symbol(v)

        # Map local call arguments to function arguments
        positional_call_args = [a.value for a in function_call_args if not a.has_keyword]
        for func_a, call_a in zip(func_args, positional_call_args):
            if isinstance(call_a, Variable) and not isinstance(call_a, DottedVariable) and func_a == self.scope.get_expected_name(call_a.name):
                # If call argument is a variable with the same name as the target function
                # argument then there is no need to rename
                new_func_a = replace_map.pop(func_a)
                self.scope.remove_symbol(new_func_a)
            else:
                # Otherwise the symbol used for the function arguments should be mapped
                # to the call argument
                func_a_name = replace_map.get(func_a, func_a)
                self.scope.inline_variable_definition(call_a, func_a_name)

        # Map local keyword call arguments to function arguments
        nargs = len(positional_call_args)
        kw_call_args = {a.keyword: a.value for a in function_call_args[nargs:]}
        for func_a, func_a_name in zip(expr.arguments[nargs:], func_args[nargs:]):
            call_a = kw_call_args.get(func_a_name, getattr(func_a.default_call_arg, 'value', func_a.default_call_arg))
            if isinstance(call_a, Variable) and func_a_name == self.scope.get_expected_name(call_a.name):
                # If call argument is a variable with the same name as the target function
                # argument then there is no need to rename
                new_func_a = replace_map.pop(func_a_name)
                self.scope.remove_symbol(new_func_a)
            else:
                # Otherwise the symbol used for the function arguments should be mapped
                # to the call argument
                used_func_a_name = replace_map.get(func_a_name, func_a_name)
                self.scope.inline_variable_definition(call_a, used_func_a_name)

        to_replace = list(replace_map.keys())
        local_var = list(replace_map.values())
        # Replace local syntactic variables from the inline functions with the syntactic
        # variables defined above which are sure to not cause name collisions
        expr.substitute(to_replace, local_var, invalidate = False)

        # Replace return expressions with an assign to the results
        returns = expr.body.get_attribute_nodes(Return)
        replace_return = [Assign(lhs, r.expr, python_ast = r.python_ast) \
                          if not isinstance(r.expr, PyccelSymbol) or not isinstance(lhs, PyccelSymbol) \
                          else EmptyNode() for r in returns]
        expr.body.substitute(returns, replace_return, invalidate = False)

        new_imports = [Import(m_name, targets) for m_name, targets in global_scope_import_targets.items()]
        for i in new_imports:
            i.set_current_ast(function_call.python_ast)
        imports.extend(new_imports)
        pyccel_stage.set_stage('semantic')

        import_init_calls = [self._visit(i) for i in imports]

        if expr.functions:
            errors.report("Functions in inline functions are not supported",
                    severity='error', symbol=expr)

        # Visit the body as though it appeared directly in the code
        body = self._visit(expr.body)
        body.insert2body(*import_init_calls, back=False)

        self._current_function.pop()

        pyccel_stage.set_stage('syntactic')
        # Put back the returns to create custom Assign nodes on the next visit
        expr.body.substitute(replace_return, returns)

        # Remove the symbol maps added to handle the function arguments
        # These are found in self.scope.variables but do not represent variables
        # that need to be declared.
        for func_a, call_a in zip(func_args, positional_call_args):
            func_a_name = replace_map.get(func_a, func_a)
            if not isinstance(call_a, Variable) or func_a_name != call_a.name:
                self.scope.remove_variable(call_a, func_a_name)

        for func_a, func_a_name in zip(expr.arguments[nargs:], func_args[nargs:]):
            if func_a_name in kw_call_args:
                used_func_a_name = replace_map.get(func_a_name, func_a_name)
                call_a = kw_call_args[func_a_name]
                if not isinstance(call_a, Variable) or used_func_a_name != call_a.name:
                    self.scope.remove_variable(call_a, used_func_a_name)

        # Swap the arguments back to the original version to preserve the syntactic
        # inline function definition.
        expr.substitute(local_var, to_replace)
        pyccel_stage.set_stage('semantic')

        if assign:
            return body
        else:
            assert expr.results
            self._additional_exprs[-1].append(body)
            return self._visit(lhs)

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

        return PythonPrint(args)

    def _visit_ClassDef(self, expr):
        # TODO - improve the use and def of interfaces
        #      - wouldn't be better if it is done inside ClassDef?

        if expr.get_direct_user_nodes(lambda u: isinstance(u, CodeBlock)):
            errors.report("Classes can only be declared in modules.",
                    symbol=expr, severity='error')

        decorators = expr.decorators
        not_used = [d for d in decorators if d != 'low_level']
        if len(not_used) >= 1:
            errors.report(UNUSED_DECORATORS, symbol=', '.join(not_used), severity='warning')

        if 'low_level' in decorators:
            self.scope.remove_symbol(expr.name)
            low_level_decs = decorators['low_level']
            assert len(low_level_decs) == 1
            arg = low_level_decs[0].args[0].value
            assert isinstance(arg, LiteralString)
            name = PyccelSymbol(arg.python_value)
            self.scope.insert_low_level_symbol(expr.name, name)
        elif self.is_stub_file:
            self.scope.remove_symbol(expr.name)
            name = expr.name
            self.scope.insert_low_level_symbol(expr.name, name)
        else:
            name = self.scope.get_expected_name(expr.name)

        #  create a new Datatype for the current class
        dtype = DataTypeFactory(name, self.scope.get_python_name(name))()
        typenames_to_dtypes[name] = dtype
        self.scope.insert_cls_construct(dtype)

        parent = self._find_superclasses(expr)

        cls_scope = self.create_new_class_scope(expr.name, used_symbols=expr.scope.local_used_symbols,
                    original_symbols = expr.scope.python_names.copy())

        attribute_annotations = [self._visit(a) for a in expr.attributes]
        attributes = []
        for a in attribute_annotations:
            if len(a) != 1:
                errors.report(f"Couldn't determine type of {a}",
                        severity='error', symbol=a)
            else:
                v = a[0]
                cls_scope.insert_variable(v)
                attributes.append(v)

        self.exit_class_scope()

        docstring = self._visit(expr.docstring) if expr.docstring else expr.docstring

        cls = ClassDef(name, attributes, [], superclasses=parent, scope=cls_scope,
                docstring = docstring, class_type = dtype)
        self.scope.insert_class(cls)

        methods = expr.methods
        for method in methods:
            cls.add_new_method(method)

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
            if not isinstance(var1, Variable) or not var1.is_optional:
                if IsClass == PyccelIsNot:
                    return LiteralTrue()
                elif IsClass == PyccelIs:
                    return LiteralFalse()
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

            p       = self.d_parsers[source]
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
                                if t in container[entry]:
                                    existing_e = container[entry][t]
                                    assert existing_e.is_imported
                                    assert e.get_direct_user_nodes(lambda x: isinstance(x, Module))[0] is \
                                            existing_e.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
                                else:
                                    assert t not in container[entry]
                                    container[entry][t] = e.clone(e.name, is_imported=True)
                                    m = e.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
                                    container[entry][t].set_current_user_node(m)
                            elif entry == 'variables':
                                container[entry][t] = e.clone(e.name)
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

            self.scope.imports['cls_constructs'].update(p.scope.cls_constructs)

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

                import_init = import_init.clone(import_init.name, is_imported = True)
                targets.append(AsName(import_init, new_name))
                container['functions'][new_name] = import_init

                result  = import_init()

            if import_free:
                old_name = import_free.name
                new_name = self.scope.get_new_name(old_name)

                import_free = import_free.clone(import_free.name, is_imported = True)
                targets.append(AsName(import_free, new_name))
                container['functions'][new_name] = import_free

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

    def _visit_StarredArguments(self, expr):
        var = self._visit(expr.args_var)
        return StarredArguments(var)

    def _visit_NumpyMatmul(self, expr):
        self.insert_import('numpy', AsName(NumpyMatmul, 'matmul'))
        a = self._visit(expr.a)
        b = self._visit(expr.b)
        return NumpyMatmul(a, b)

    def _visit_Assert(self, expr):
        test = self._visit(expr.test)
        return Assert(test)

    def _visit_FunctionDefResult(self, expr):
        f_name      = self.current_function_name
        if isinstance(f_name, DottedName):
            f_name = f_name.name[-1]

        # There may be no name if we are in a FunctionTypeAnnotation
        if f_name:
            original_name = self.scope.get_python_name(f_name)
            if original_name.startswith('__i') and ('__'+original_name[3:]) in magic_method_map.values():
                return EmptyNode()

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

    def _build_NumpyWhere(self, func_call, func_call_args):
        """
        Method for building the node created by a call to `numpy.where`.

        Method for building the node created by a call to `numpy.where`. If only one argument is passed to `numpy.where`
        then it is equivalent to a call to `numpy.nonzero`. The result of a call to `numpy.nonzero`
        is a complex object so there is a `_build_NumpyNonZero` function which must be called.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `numpy.nonzero.

        func_call_args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `numpy.nonzero` function.
        """
        # expr is a FunctionCall
        args = [a.value for a in func_call_args if not a.has_keyword]
        kwargs = {a.keyword: a.value for a in func_call.args if a.has_keyword}
        nargs = len(args)+len(kwargs)
        if nargs == 1:
            return self._build_NumpyNonZero(func_call, func_call_args)
        return NumpyWhere(*args, **kwargs)

    def _build_NumpyNonZero(self, func_call, func_call_args):
        """
        Method for building the node created by a call to `numpy.nonzero`.

        Method for building the node created by a call to `numpy.nonzero`. The result of a call to `numpy.nonzero`
        is a complex object (tuple of arrays) in order to ensure that the results are correctly saved into the
        correct objects it is therefore important to call `_visit` on any intermediate expressions that are required.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `numpy.nonzero.

        func_call_args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `numpy.nonzero` function.
        """
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

    def _build_ListExtend(self, expr, args):
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
            The syntactic DottedName node that represents the call to `.extend()`.

        args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

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
            if not isinstance(list_variable.class_type.element_type, (StringType, FixedSizeNumericType)):
                for a in added_list:
                    if not isinstance(a, (PythonList, PythonSet, PythonTuple, NumpyNewArray)):
                        self._indicate_pointer_target(list_variable, a, expr)
            return CodeBlock(store)
        else:
            pyccel_stage.set_stage('syntactic')
            for_target = self.scope.get_new_name('index')
            arg = FunctionCallArgument(for_target)
            func_call = FunctionCall('append', [arg])
            dotted = DottedName(expr.name[0], func_call)
            dotted.set_current_ast(expr.python_ast)
            lhs = PyccelSymbol('_', is_temp=True)
            assign = Assign(lhs, dotted)
            assign.set_current_ast(expr.python_ast)
            body = CodeBlock([assign])
            for_obj = For(for_target, iterable, body)
            pyccel_stage.set_stage('semantic')
            return self._visit(for_obj)

    def _build_MathSqrt(self, func_call, func_call_args):
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

        func_call_args : iterable[FunctionCallArgument]
            The semantic argument passed to the function.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.sqrt` function.
        """
        func = self.scope.find(func_call.funcdef, 'functions')
        arg = func_call_args[0]
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

    def _build_CmathSqrt(self, func_call, func_call_args):
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

        func_call_args : iterable[FunctionCallArgument]
            The semantic argument passed to the function.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.sqrt` function.
        """
        func = self.scope.find(func_call.funcdef, 'functions')
        arg = func_call_args[0]
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

    def _build_CmathPolar(self, func_call, func_call_args):
        """
        Method for building the node created by a call to `cmath.polar`.

        Method for building the node created by a call to `cmath.polar`. A separate method is needed for
        this because the function is translated to an expression including calls to `math.sqrt` and
        `math.atan2`. The associated imports therefore need to be inserted into the parser.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `cmath.polar`.

        func_call_args : iterable[FunctionCallArgument]
            The semantic argument passed to the function.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.polar` function.
        """
        arg = func_call_args[0]
        z = arg.value
        x = PythonReal(z)
        y = PythonImag(z)
        x_var = self.scope.get_temporary_variable(z, class_type=PythonNativeFloat(),
                    is_argument=False)
        y_var = self.scope.get_temporary_variable(z, class_type=PythonNativeFloat(),
                    is_argument=False)
        self._additional_exprs[-1].append(Assign(x_var, x))
        self._additional_exprs[-1].append(Assign(y_var, y))
        r = MathSqrt(PyccelAdd(PyccelMul(x_var,x_var), PyccelMul(y_var,y_var)))
        t = MathAtan2(y_var, x_var)
        self.insert_import('math', AsName(MathSqrt, 'sqrt'))
        self.insert_import('math', AsName(MathAtan2, 'atan2'))
        return PythonTuple(r,t)

    def _build_CmathRect(self, func_call, func_call_args):
        """
        Method for building the node created by a call to `cmath.rect`.

        Method for building the node created by a call to `cmath.rect`. A separate method is needed for
        this because the function is translated to an expression including calls to `math.cos` and
        `math.sin`. The associated imports therefore need to be inserted into the parser.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `cmath.rect`.

        func_call_args : iterable[FunctionCallArgument]
            The 2 semantic arguments passed to the function.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.rect` function.
        """
        arg_r, arg_phi = func_call_args
        r = arg_r.value
        phi = arg_phi.value
        x = PyccelMul(r, MathCos(phi))
        y = PyccelMul(r, MathSin(phi))
        self.insert_import('math', AsName(MathCos, 'cos'))
        self.insert_import('math', AsName(MathSin, 'sin'))
        return PyccelAdd(x, PyccelMul(y, LiteralImaginaryUnit()))

    def _build_CmathPhase(self, func_call, func_call_args):
        """
        Method for building the node created by a call to `cmath.phase`.

        Method for building the node created by a call to `cmath.phase`. A separate method is needed for
        this because the function is translated to a call to `math.atan2`. The associated import therefore
        needs to be inserted into the parser.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `cmath.phase`.

        func_call_args : iterable[FunctionCallArgument]
            The semantic argument passed to the function.

        Returns
        -------
        TypedAstNode
            A node describing the result of a call to the `cmath.phase` function.
        """
        arg = func_call_args[0]
        var = arg.value
        if not isinstance(var.dtype.primitive_type, PrimitiveComplexType):
            return LiteralFloat(0.0)
        else:
            self.insert_import('math', AsName(MathAtan2, 'atan2'))
            return MathAtan2(PythonImag(var), PythonReal(var))

    def _build_PythonTupleFunction(self, func_call, func_args):
        """
        Method for building the node created by a call to `tuple()`.

        Method for building the node created by a call to `tuple()`. A separate method is needed for
        this because inhomogeneous variables can be passed to this function. In order to access the
        underlying variables for the indexed elements access to the scope is required.

        Parameters
        ----------
        func_call : FunctionCall
            The syntactic FunctionCall describing the call to `tuple()`.

        func_args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

        Returns
        -------
        PythonTuple
            A node describing the result of a call to the `tuple()` function.
        """
        arg = func_args[0].value
        if isinstance(arg, PythonTuple):
            return arg
        elif isinstance(arg.shape[0], LiteralInteger):
            return PythonTuple(*[self.scope.collect_tuple_element(a) for a in arg])
        else:
            raise TypeError(f"Can't unpack {arg} into a tuple")

    def _build_NumpyArray(self, expr, func_call_args):
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

        func_call_args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

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

    def _build_SetUpdate(self, expr, args):
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
        expr : DottedName | AugAssign
            The syntactic DottedName node that represents the call to `.update()`.

        args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

        Returns
        -------
        PyccelAstNode
            CodeBlock or For containing SetAdd objects.
        """
        if isinstance(expr, DottedName):
            iterable_args = [a.value for a in expr.name[1].args]
            set_obj = expr.name[0]
        elif isinstance(expr, AugAssign):
            iterable_args = [expr.rhs]
            set_obj = expr.lhs
        else:
            raise NotImplementedError(f"Function doesn't handle {type(expr)}")

        code = []
        for iterable in iterable_args:
            if isinstance(iterable, (PythonList, PythonSet, PythonTuple)):
                list_variable = self._visit(set_obj)
                added_list = self._visit(iterable)
                try:
                    code.extend(SetAdd(list_variable, a) for a in added_list)
                except TypeError as e:
                    msg = str(e)
                    errors.report(msg, symbol=expr, severity='fatal')
            else:
                pyccel_stage.set_stage('syntactic')
                for_target = self.scope.get_new_name()
                arg = FunctionCallArgument(for_target)
                func_call = FunctionCall('add', [arg])
                dotted = DottedName(set_obj, func_call)
                lhs = PyccelSymbol('_', is_temp=True)
                assign = Assign(lhs, dotted)
                assign.set_current_ast(expr.python_ast)
                body = CodeBlock([assign])
                for_obj = For(for_target, iterable, body)
                pyccel_stage.set_stage('semantic')
                code.append(self._visit(for_obj))

        if len(code) == 1:
            return code[0]
        else:
            return CodeBlock(code)

    def _build_SetUnion(self, expr, function_call_args):
        """
        Method to navigate the syntactic DottedName node of a `set.union()` call.

        The purpose of this `_build` method is to construct new nodes from a syntactic
        DottedName node. It creates a SetUnion node if the type of the arguments matches
        the type of the original set. Otherwise it uses `set.copy` and `set.update` to
        handle iterators.

        Parameters
        ----------
        expr : DottedName
            The syntactic DottedName node that represents the call to `.union()`.

        function_call_args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

        Returns
        -------
        SetUnion | CodeBlock
            The nodes describing the union operator.
        """
        if isinstance(expr, DottedName):
            syntactic_set_obj = expr.name[0]
            syntactic_args = [a.value for a in expr.name[1].args]
        elif isinstance(expr, PyccelBitOr):
            syntactic_set_obj = expr.args[0]
            syntactic_args = expr.args[1:]
        else:
            raise NotImplementedError(f"Function doesn't handle {type(expr)}")

        args = [a.value for a in function_call_args]
        set_obj = self._visit(syntactic_set_obj)
        class_type = set_obj.class_type
        if all(a.class_type == class_type for a in args):
            return SetUnion(set_obj, *args[1:])
        else:
            element_type = class_type.element_type
            if any(a.class_type.element_type != element_type for a in args):
                errors.report(("Containers containing objects of a different type cannot be used as "
                               f"arguments to {class_type}.union"),
                        severity='fatal', symbol=expr)

            lhs = expr.get_user_nodes(Assign)[0].lhs
            pyccel_stage.set_stage('syntactic')
            body = [Assign(lhs, DottedName(syntactic_set_obj, FunctionCall('copy', ())),
                           python_ast = expr.python_ast)]
            update_calls = [DottedName(lhs, FunctionCall('update', (s_a,))) for s_a in syntactic_args]
            for c in update_calls:
                c.set_current_ast(expr.python_ast)
            body += [Assign(PyccelSymbol('_', is_temp=True), c, python_ast = expr.python_ast)
                     for c in update_calls]
            pyccel_stage.set_stage('semantic')
            return CodeBlock([self._visit(b) for b in body])

    def _build_SetIntersection(self, expr, function_call_args):
        """
        Method to visit a SetIntersection node.

        The purpose of this `_build` method is to construct multiple nodes to represent
        the single DottedName node representing the call to SetIntersection. It
        replaces the call with a call to copy followed by multiple calls to
        SetIntersectionUpdate.

        Parameters
        ----------
        expr : DottedName
            The syntactic DottedName node that represents the call to `.intersection()`.

        function_call_args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

        Returns
        -------
        CodeBlock
            CodeBlock containing SetCopy and SetIntersectionUpdate objects.
        """
        start_set = function_call_args[0].value
        set_args = [self._visit(a.value) for a in function_call_args[1:]]
        assign = expr.get_direct_user_nodes(lambda a: isinstance(a, Assign))
        if assign:
            syntactic_lhs = assign[-1].lhs
        else:
            syntactic_lhs = self.scope.get_new_name()
        d_var = self._infer_type(start_set)
        if isinstance(start_set, PythonSet):
            rhs = start_set
        else:
            rhs = SetCopy(start_set)
        body = []
        lhs = self._assign_lhs_variable(syntactic_lhs, d_var, rhs, body)
        body.append(Assign(lhs, rhs, python_ast = expr.python_ast))
        try:
            body += [SetIntersectionUpdate(lhs, s) for s in set_args]
        except TypeError as e:
            errors.report(e, symbol=expr, severity='error')
        if assign:
            return CodeBlock(body)
        else:
            self._additional_exprs[-1].extend(body)
            return lhs

    def _build_PythonLen(self, expr, function_call_args):
        """
        Method to visit a PythonLen node.

        The purpose of this `_build` method is to construct a node representing
        a call to the PythonLen function. This function returns the first element
        of the shape of a variable, or a call to a method which calculates the
        length (e.g. the `__len__` function).

        Parameters
        ----------
        expr : FunctionCall
            The syntactic node that represents the call to `len()`.

        function_call_args : iterable[FunctionCallArgument]
            The semantic argument passed to the function.

        Returns
        -------
        TypedAstNode
            The node representing an object which allows the result of the
            PythonLen function to be obtained.
        """
        arg = function_call_args[0].value
        class_type = arg.class_type
        if isinstance(arg, LiteralString):
            return LiteralInteger(len(arg.python_value))
        elif isinstance(arg.class_type, CustomDataType):
            class_base = self.get_cls_base(class_type)
            magic_method = class_base.get_method('__len__')
            if magic_method:
                return self._handle_function(expr, magic_method, function_call_args)
            else:
                raise errors.report(f"__len__ not implemented for type {class_type}",
                        severity='fatal', symbol=expr)
        elif arg.rank > 0:
            return arg.shape[0]
        else:
            raise errors.report(f"__len__ not implemented for type {class_type}",
                    severity='fatal', symbol=expr)

    def _build_PythonSetFunction(self, expr, function_call_args):
        """
        Method to visit a PythonSetFunction node.

        The purpose of this `_build` method is to construct a node representing
        a set which is built from another object. A build function is required
        as sets of unknown length must be built by calling the add function
        repeatedly. This means that the entire assignment statement must be used.

        Parameters
        ----------
        expr : FunctionCall
            The syntactic node that represents the call to `PythonSetFunction`.

        function_call_args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

        Returns
        -------
        TypedAstNode | CodeBlock
            The node representing an object which allows the set to be created.
        """
        if len(function_call_args) == 0:
            return PythonSet()

        arg = function_call_args[0].value
        class_type = arg.class_type
        if isinstance(arg, (PythonList, PythonSet, PythonTuple)):
            return PythonSet(*arg)
        elif isinstance(class_type, HomogeneousSetType):
            return SetCopy(arg)
        else:
            assigns = expr.get_direct_user_nodes(lambda a: isinstance(a, Assign))
            if not assigns:
                lhs = self.scope.get_new_name()
            else:
                assert len(assigns) == 1
                lhs = assigns[0].lhs
            d_var = {
                    'class_type' : HomogeneousSetType.get_new(class_type.element_type),
                    'shape' : arg.shape,
                    'cls_base' : SetClass,
                    'memory_handling' : 'heap'
                    }
            body = []
            lhs_semantic_var = self._assign_lhs_variable(lhs, d_var, PythonSetFunction(arg), body)
            scope = self.create_new_loop_scope()
            targets, iterable = self._get_for_iterators(arg, self.scope.get_new_name(), body, expr)
            self.exit_loop_scope()
            body.append(For(targets, iterable, [SetAdd(lhs_semantic_var, targets[0])], scope=scope))
            if assigns:
                return CodeBlock(body)
            else:
                self._additional_exprs[-1].extend(body)
                return lhs_semantic_var

    def _build_PythonIsInstance(self, expr, function_call_args):
        """
        Method to visit a PythonIsInstance node.

        The purpose of this `_build` method is to construct a literal boolean indicating
        whether or not the expression has the expected type.
        The syntactic node that represents the call to `isinstance()`.

        Parameters
        ----------
        expr : FunctionCall
            The syntactic node that represents the call to `PythonSetFunction`.

        function_call_args : iterable[FunctionCallArgument]
            The 2 semantic arguments passed to the function.

        Returns
        -------
        Literal
            A LiteralTrue or LiteralFalse node describing the result of the `isinstance`
            call.
        """
        obj = function_call_args[0].value
        class_or_tuple = function_call_args[1].value
        if isinstance(class_or_tuple, PythonTuple):
            obj_arg = function_call_args[0]
            return PyccelOr(*[self._build_PythonIsInstance(expr, [obj_arg, FunctionCallArgument(class_type)]) \
                                for class_type in class_or_tuple], simplify=True)
        elif isinstance(class_or_tuple, UnionTypeAnnotation):
            obj_arg = function_call_args[0]
            return PyccelOr(*[self._build_PythonIsInstance(expr, [obj_arg, FunctionCallArgument(var_annot)]) \
                                for var_annot in class_or_tuple.type_list], simplify=True)
        else:
            if isinstance(class_or_tuple, (VariableTypeAnnotation, ClassDef)):
                expected_type = class_or_tuple.class_type
            else:
                class_type = class_or_tuple.cls_name
                try:
                    expected_type = class_type.static_type()
                except AttributeError:
                    expected_type = None

            if isinstance(expected_type, type):
                return convert_to_literal(isinstance(obj.class_type, expected_type))

            elif expected_type:
                class_type = obj.class_type
                cls_base_to_insert = [self.get_cls_base(class_type)]
                possible_types = {class_type}
                while cls_base_to_insert:
                    cls_base = cls_base_to_insert.pop()
                    class_type = cls_base.class_type
                    possible_types.add(class_type)
                    cls_base_to_insert.extend(cls_base.superclasses)

                possible_types.discard(None)

                return convert_to_literal(expected_type in possible_types)

            else:
                errors.report(f"Type {class_or_tuple} is not handled in isinstance call.",
                        severity='error', symbol=expr)
                return LiteralTrue()

    def _build_ListAppend(self, expr, args):
        """
        Method to create the semantic ListAppend node.

        Method to create the semantic ListAppend node ensuring that pointers are
        correctly handled.

        Parameters
        ----------
        expr : DottedName
            The syntactic DottedName node that represents the call to `.append()`.

        args : iterable[FunctionCallArgument]
            An iterable containing the 1 semantic argument passed to the function.

        Returns
        -------
        ListAppend
            The semantic ListAppend object.
        """
        list_obj, append_arg = [a.value for a in args]
        try:
            semantic_node = ListAppend(list_obj, append_arg)
        except TypeError as e:
            errors.report(e, symbol=expr, severity='error')
        if not isinstance(append_arg.class_type, (StringType, FixedSizeNumericType)) \
                and not isinstance(append_arg, (PythonList, PythonSet, PythonTuple, NumpyNewArray)):
            self._indicate_pointer_target(list_obj, append_arg, expr)
        return semantic_node

    def _build_ListInsert(self, expr, args):
        """
        Method to create the semantic ListInsert node.

        Method to create the semantic ListInsert node ensuring that pointers are
        correctly handled.

        Parameters
        ----------
        expr : DottedName
            The syntactic DottedName node that represents the call to `.insert()`.

        args : iterable[FunctionCallArgument]
            The 2 semantic arguments passed to the function.

        Returns
        -------
        ListInsert
            The semantic ListInsert object.
        """
        list_obj, index, new_elem = [a.value for a in args]
        try:
            semantic_node = ListInsert(list_obj, index, new_elem)
        except TypeError as e:
            errors.report(e, symbol=expr, severity='error')
        if not isinstance(new_elem.class_type, (StringType, FixedSizeNumericType)) \
                and not isinstance(new_elem, (PythonList, PythonSet, PythonTuple, NumpyNewArray)):
            self._indicate_pointer_target(list_obj, new_elem, expr)
        return semantic_node

    def _build_SetDifference(self, expr, function_call_args):
        """
        Method to visit a SetDifference node.

        The purpose of this `_build` method is to construct multiple nodes to represent
        the single DottedName node representing the call to SetDifference. It
        replaces the call with a call to copy followed by multiple calls to
        SetDifferenceUpdate.

        Parameters
        ----------
        expr : DottedName
            The syntactic DottedName node that represents the call to `.difference()`.

        function_call_args : iterable[FunctionCallArgument]
            The semantic arguments passed to the function.

        Returns
        -------
        CodeBlock
            CodeBlock containing SetCopy and SetDifferenceUpdate objects.
        """
        start_set = function_call_args[0].value
        set_args = [self._visit(a.value) for a in function_call_args[1:]]
        assign = expr.get_direct_user_nodes(lambda a: isinstance(a, Assign))
        if assign:
            syntactic_lhs = assign[-1].lhs
        else:
            syntactic_lhs = self.scope.get_new_name()
        d_var = self._infer_type(start_set)
        if isinstance(start_set, PythonSet):
            rhs = start_set
        else:
            rhs = SetCopy(start_set)
        body = []
        lhs = self._assign_lhs_variable(syntactic_lhs, d_var, rhs, body)
        body.append(Assign(lhs, rhs, python_ast = expr.python_ast))
        try:
            body += [SetDifferenceUpdate(lhs, s) for s in set_args]
        except TypeError as e:
            errors.report(e, symbol=expr, severity='error')
        if assign:
            return CodeBlock(body)
        else:
            self._additional_exprs[-1].extend(body)
            return lhs
