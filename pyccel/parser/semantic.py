# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" File containing SemanticParser. This class handles the semantic stage of the translation.
See the developer docs for more details
"""

from itertools import chain
import warnings

from sympy.utilities.iterables import iterable as sympy_iterable

from sympy import Sum as Summation
from sympy import Symbol as sp_Symbol
from sympy import Integer as sp_Integer
from sympy import ceiling
from sympy.core import cache

#==============================================================================

from pyccel.ast.basic import Basic, PyccelAstNode, ScopedNode

from pyccel.ast.builtins import PythonPrint
from pyccel.ast.builtins import PythonInt, PythonBool, PythonFloat, PythonComplex
from pyccel.ast.builtins import python_builtin_datatype, PythonImag, PythonReal
from pyccel.ast.builtins import PythonList, PythonConjugate
from pyccel.ast.builtins import (PythonRange, PythonZip, PythonEnumerate,
                                 PythonTuple, Lambda, PythonMap)

from pyccel.ast.core import Comment, CommentBlock, Pass
from pyccel.ast.core import If, IfSection
from pyccel.ast.core import Allocate, Deallocate
from pyccel.ast.core import Assign, AliasAssign, SymbolicAssign
from pyccel.ast.core import AugAssign, CodeBlock
from pyccel.ast.core import Return, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core import ConstructorCall, InlineFunctionDef
from pyccel.ast.core import FunctionDef, Interface, FunctionAddress, FunctionCall, FunctionCallArgument
from pyccel.ast.core import DottedFunctionCall
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
from pyccel.ast.core import InProgram
from pyccel.ast.core import Decorator
from pyccel.ast.core import PyccelFunctionDef
from pyccel.ast.core import Assert

from pyccel.ast.class_defs import NumpyArrayClass, TupleClass, get_cls_base

from pyccel.ast.datatypes import NativeRange, str_dtype
from pyccel.ast.datatypes import NativeSymbol, DataTypeFactory
from pyccel.ast.datatypes import default_precision
from pyccel.ast.datatypes import (NativeInteger, NativeBool,
                                  NativeFloat, NativeString,
                                  NativeGeneric, NativeComplex,
                                  NativeVoid)

from pyccel.ast.functionalexpr import FunctionalSum, FunctionalMax, FunctionalMin, GeneratorComprehension, FunctionalFor

from pyccel.ast.headers import FunctionHeader, MethodHeader, Header
from pyccel.ast.headers import MacroFunction, MacroVariable

from pyccel.ast.internals import PyccelInternalFunction, Slice, PyccelSymbol, get_final_precision
from pyccel.ast.itertoolsext import Product

from pyccel.ast.literals import LiteralTrue, LiteralFalse
from pyccel.ast.literals import LiteralInteger, LiteralFloat
from pyccel.ast.literals import Nil, LiteralString, LiteralImaginaryUnit
from pyccel.ast.literals import Literal, convert_to_literal

from pyccel.ast.mathext  import math_constants, MathSqrt, MathAtan2, MathSin, MathCos

from pyccel.ast.numpyext import NumpyMatmul
from pyccel.ast.numpyext import NumpyBool
from pyccel.ast.numpyext import NumpyWhere, NumpyArray
from pyccel.ast.numpyext import NumpyInt, NumpyInt8, NumpyInt16, NumpyInt32, NumpyInt64
from pyccel.ast.numpyext import NumpyFloat, NumpyFloat32, NumpyFloat64
from pyccel.ast.numpyext import NumpyComplex, NumpyComplex64, NumpyComplex128
from pyccel.ast.numpyext import NumpyTranspose, NumpyConjugate
from pyccel.ast.numpyext import NumpyNewArray, NumpyNonZero, NumpyResultType
from pyccel.ast.numpyext import DtypePrecisionToCastFunction

from pyccel.ast.omp import (OMP_For_Loop, OMP_Simd_Construct, OMP_Distribute_Construct,
                            OMP_TaskLoop_Construct, OMP_Sections_Construct, Omp_End_Clause,
                            OMP_Single_Construct)

from pyccel.ast.operators import PyccelArithmeticOperator, PyccelIs, PyccelIsNot, IfTernaryOperator, PyccelUnarySub
from pyccel.ast.operators import PyccelNot, PyccelEq, PyccelAdd, PyccelMul, PyccelPow
from pyccel.ast.operators import PyccelAssociativeParenthesis, PyccelDiv

from pyccel.ast.sympy_helper import sympy_to_pyccel, pyccel_to_sympy

from pyccel.ast.utilities import builtin_function as pyccel_builtin_function
from pyccel.ast.utilities import builtin_import as pyccel_builtin_import
from pyccel.ast.utilities import builtin_import_registry as pyccel_builtin_import_registry
from pyccel.ast.utilities import split_positional_keyword_arguments
from pyccel.ast.utilities import recognised_source

from pyccel.ast.variable import Constant
from pyccel.ast.variable import Variable
from pyccel.ast.variable import TupleVariable, HomogeneousTupleVariable, InhomogeneousTupleVariable
from pyccel.ast.variable import IndexedElement
from pyccel.ast.variable import DottedName, DottedVariable

from pyccel.errors.errors import Errors
from pyccel.errors.errors import PyccelSemanticError

from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, UNDERSCORE_NOT_A_THROWAWAY,
        UNDEFINED_VARIABLE, IMPORTING_EXISTING_IDENTIFIED, INDEXED_TUPLE, LIST_OF_TUPLES,
        INVALID_INDICES, INCOMPATIBLE_ARGUMENT, INCOMPATIBLE_ORDERING,
        UNRECOGNISED_FUNCTION_CALL, STACK_ARRAY_SHAPE_UNPURE_FUNC, STACK_ARRAY_UNKNOWN_SHAPE,
        ARRAY_DEFINITION_IN_LOOP, STACK_ARRAY_DEFINITION_IN_LOOP,
        INCOMPATIBLE_TYPES_IN_ASSIGNMENT, ARRAY_ALREADY_IN_USE, ASSIGN_ARRAYS_ONE_ANOTHER,
        INVALID_POINTER_REASSIGN, INCOMPATIBLE_REDEFINITION, ARRAY_IS_ARG,
        INCOMPATIBLE_REDEFINITION_STACK_ARRAY, ARRAY_REALLOCATION, RECURSIVE_RESULTS_REQUIRED,
        PYCCEL_RESTRICTION_INHOMOG_LIST, UNDEFINED_IMPORT_OBJECT, UNDEFINED_LAMBDA_VARIABLE,
        UNDEFINED_LAMBDA_FUNCTION, UNDEFINED_INIT_METHOD, UNDEFINED_FUNCTION,
        INVALID_MACRO_COMPOSITION, WRONG_NUMBER_OUTPUT_ARGS, INVALID_FOR_ITERABLE,
        PYCCEL_RESTRICTION_LIST_COMPREHENSION_LIMITS, PYCCEL_RESTRICTION_LIST_COMPREHENSION_SIZE,
        UNUSED_DECORATORS, DUPLICATED_SIGNATURE, FUNCTION_TYPE_EXPECTED,
        UNSUPPORTED_POINTER_RETURN_VALUE, PYCCEL_RESTRICTION_OPTIONAL_NONE,
        PYCCEL_RESTRICTION_PRIMITIVE_IMMUTABLE, PYCCEL_RESTRICTION_IS_ISNOT,
        FOUND_DUPLICATED_IMPORT, UNDEFINED_WITH_ACCESS, MACRO_MISSING_HEADER_OR_FUNC,)

from pyccel.parser.base      import BasicParser
from pyccel.parser.syntactic import SyntaxParser

from pyccel.utilities.stage import PyccelStage

import pyccel.decorators as def_decorators
#==============================================================================

errors = Errors()
pyccel_stage = PyccelStage()

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
    if isinstance(var, AsName):
        return var.target
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
        self._program_namespace = self.scope.new_child_scope('__main__')

        # used to store the local variables of a code block needed for garbage collecting
        self._allocs = []

        # used to store code split into multiple lines to be reinserted in the CodeBlock
        self._additional_exprs = []

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

    @property
    def program_namespace(self):
        """
        Get the namespace relevant to the program.

        Get the namespace which describes the section of
        code which is executed as a program. In other words
        the code inside an `if __name__ == '__main__':`
        block.

        Returns
        -------
        Scope : The program namespace.
        """
        return self._program_namespace

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
        pyccel.ast.basic.Basic
            An annotated object which can be printed.
        """

        if self.semantic_done:
            print ('> semantic analysis already done')
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        errors = Errors()
        if self.filename:
            errors.set_target(self.filename, 'file')
        errors.set_parser_stage('semantic')

        # then we treat the current file

        ast = self.ast

        self._allocs.append([])
        # we add the try/except to allow the parser to find all possible errors
        pyccel_stage.set_stage('semantic')
        ast = self._visit(ast)

        self._ast = ast

        self._semantic_done = True

        return ast

    #================================================================
    #              Utility functions for scope handling
    #================================================================

    def change_to_program_scope(self):
        """
        Switch the focus to the program scope.

        Update the namespace variable so that it points at the
        program namespace (which describes the scope inside
        a `if __name__ == '__main__':` block). It is assumed that
        the current namespace is the module namespace.
        """
        self._allocs.append([])
        self._module_namespace = self.scope
        self.scope = self._program_namespace

    def change_to_module_scope(self):
        """
        Switch the focus to the module scope.

        Update the namespace variable so that it points
        at the module namespace. It is assumed that the
        current namespace is the program namespace.
        """
        self._program_namespace = self.scope
        self.scope = self._module_namespace

    def check_for_variable(self, name):
        """
        Search for a Variable object with the given name in the current scope.

        Search for a Variable object with the given name in the current scope,
        defined by the local and global Python scopes. Return None if not found.

        Parameters
        ----------
        name : str
            The object describing the variable.

        Returns
        -------
        Variable
            Returns the variable if found or None.
        """

        if isinstance(name, DottedName):
            prefix_parts = name.name[:-1]
            syntactic_prefix = prefix_parts[0] if len(prefix_parts) == 1 else DottedName(*prefix_parts)
            prefix = self._visit(syntactic_prefix)
            class_def = self.scope.find(prefix.dtype.name, 'classes')
            attr_name = name.name[-1]
            attribute = class_def.scope.find(attr_name, 'variables') if class_def else None
            if attribute:
                return attribute.clone(attribute.name, new_class = DottedVariable, lhs = prefix)
            else:
                return None
        return self.scope.find(name, 'variables')

    def get_variable(self, name):
        """ Like 'check_for_variable', but raise Pyccel error if Variable is not found.
        """
        var = self.check_for_variable(name)
        if var is None:
            if name == '_':
                errors.report(UNDERSCORE_NOT_A_THROWAWAY,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='fatal')
            else:
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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
        """Returns the class datatype for name if it exists.
        Raises an error otherwise
        """
        result = self.scope.find(name, 'cls_constructs')

        if result is None:
            msg = f'class construct {name} not found'
            return errors.report(msg,
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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
                              bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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


    #=======================================================
    #              Utility functions
    #=======================================================

    def _garbage_collector(self, expr):
        """
        Search in a CodeBlock if no trailing Return Node is present add the needed frees.
        """
        if len(expr.body)>0 and not isinstance(expr.body[-1], Return):
            deallocs = [Deallocate(i) for i in self._allocs[-1]]
        else:
            deallocs = []
        self._allocs.pop()
        return deallocs

    def _infer_type(self, expr):
        """
        Infer all relevant type information for the expression.

        Create a dictionary describing all the type information that can be
        inferred about the expression `expr`. This includes information about:
        - `datatype`
        - `precision`
        - `rank`
        - `shape`
        - `order`
        - `memory_handling`
        - `cls_base`
        - `is_target`

        Parameters
        ----------
        expr : pyccel.ast.basic.Basic
                An AST object representing an object in the code whose type
                must be determined.

        Returns
        -------
        dict
            Dictionary containing all the type information which was inferred.
        """
        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        errors = Errors()

        d_var = {}
        # TODO improve => put settings as attribut of Parser

        if expr in (PythonInt, PythonFloat, PythonComplex, PythonBool, NumpyBool, NumpyInt, NumpyInt8, NumpyInt16,
                      NumpyInt32, NumpyInt64, NumpyComplex, NumpyComplex64,
                      NumpyComplex128, NumpyFloat, NumpyFloat64, NumpyFloat32):

            d_var['datatype'   ] = '*'
            d_var['rank'       ] = 0
            d_var['precision'  ] = 0
            return d_var

        elif isinstance(expr, Variable):
            d_var['datatype'       ] = expr.dtype
            d_var['memory_handling'] = expr.memory_handling
            d_var['shape'          ] = expr.shape
            d_var['rank'           ] = expr.rank
            d_var['cls_base'       ] = expr.cls_base or self.scope.find(expr.dtype.name, 'classes')
            d_var['is_target'      ] = expr.is_target
            d_var['order'          ] = expr.order
            d_var['precision'      ] = expr.precision
            return d_var

        elif isinstance(expr, PythonTuple):
            d_var['datatype'       ] = expr.dtype
            d_var['precision'      ] = expr.precision
            d_var['memory_handling'] = 'heap'
            d_var['shape'          ] = expr.shape
            d_var['rank'           ] = expr.rank
            d_var['order'          ] = expr.order
            d_var['cls_base'       ] = TupleClass
            return d_var

        elif isinstance(expr, Concatenate):
            d_var['datatype'      ] = expr.dtype
            d_var['precision'     ] = expr.precision
            d_var['shape'         ] = expr.shape
            d_var['rank'          ] = expr.rank
            d_var['order'         ] = expr.order
            d_var['cls_base'      ] = TupleClass
            if any(getattr(a, 'on_heap', False) for a in expr.args):
                d_var['memory_handling'] = 'heap'
            else:
                d_var['memory_handling'] = 'stack'
            return d_var

        elif isinstance(expr, Duplicate):
            d = self._infer_type(expr.val)

            # TODO must check that it is consistent with pyccel's rules
            # TODO improve
            d_var['datatype'      ] = d['datatype']
            d_var['rank'          ] = expr.rank
            d_var['shape'         ] = expr.shape
            d_var['order'         ] = expr.order
            d_var['cls_base'      ] = TupleClass
            if d.get('on_stack', False) and isinstance(expr.length, LiteralInteger):
                d_var['memory_handling'] = 'stack'
            else:
                d_var['memory_handling'] = 'heap'
            return d_var

        elif isinstance(expr, NumpyNewArray):
            d_var['datatype'   ] = expr.dtype
            d_var['memory_handling'] = 'heap' if expr.rank > 0 else 'stack'
            d_var['shape'      ] = expr.shape
            d_var['rank'       ] = expr.rank
            d_var['order'      ] = expr.order
            d_var['precision'  ] = expr.precision
            d_var['cls_base'   ] = NumpyArrayClass
            return d_var

        elif isinstance(expr, NumpyTranspose):

            var = expr.internal_var

            d_var['memory_handling'] = 'alias' if isinstance(var, Variable) else 'heap'
            d_var['datatype'      ] = var.dtype
            d_var['shape'         ] = tuple(reversed(var.shape))
            d_var['rank'          ] = var.rank
            d_var['cls_base'      ] = var.cls_base
            d_var['is_target'     ] = var.is_target
            d_var['order'         ] = 'C' if var.order=='F' else 'F'
            d_var['precision'     ] = var.precision
            return d_var

        elif isinstance(expr, PyccelAstNode):

            d_var['datatype'   ] = expr.dtype
            d_var['memory_handling'] = 'heap' if expr.rank > 0 else 'stack'
            d_var['shape'      ] = expr.shape
            d_var['rank'       ] = expr.rank
            d_var['order'      ] = expr.order
            d_var['precision'  ] = expr.precision
            d_var['cls_base'   ] = get_cls_base(expr.dtype, expr.precision, expr.rank)
            return d_var

        elif isinstance(expr, PythonRange):

            d_var['datatype'   ] = NativeRange()
            d_var['memory_handling'] = 'stack' # because rank is 0 and no shape defined
            d_var['shape'      ] = None
            d_var['rank'       ] = 0
            d_var['cls_base'   ] = expr  # TODO: shall we keep it?
            return d_var

        elif isinstance(expr, Lambda):

            d_var['datatype'   ] = NativeSymbol()
            d_var['memory_handling'] = 'stack' # because rank is 0 and no shape defined
            d_var['rank'       ] = 0
            return d_var

        else:
            type_name = type(expr).__name__
            msg = f'Type of Object : {type_name} cannot be infered'
            return errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg, symbol=expr,
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='fatal')

    def _extract_indexed_from_var(self, var, indices, expr):
        """ Use indices to extract appropriate element from
        object 'var'
        This contains most of the contents of _visit_IndexedElement
        but is a separate function in order to be recursive
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
                tmp_var = PyccelSymbol(self.scope.get_new_name())
                assign = Assign(tmp_var, var)
                assign.ast = expr.ast
                self._additional_exprs[-1].append(self._visit(assign))
                var = self._visit(tmp_var)


        elif not isinstance(var, Variable):
            if hasattr(var,'__getitem__'):
                if len(indices)==1:
                    return var[indices[0]]
                else:
                    return self._visit(var[indices[0]][indices[1:]])
            else:
                var_type = type(var)
                errors.report(f"Can't index {var_type}", symbol=expr,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='fatal')

        indices = tuple(indices)

        if isinstance(var, InhomogeneousTupleVariable):

            arg = indices[0]

            if isinstance(arg, Slice):
                if ((arg.start is not None and not isinstance(arg.start, LiteralInteger)) or
                        (arg.stop is not None and not isinstance(arg.stop, LiteralInteger))):
                    errors.report(INDEXED_TUPLE, symbol=var,
                        bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                        severity='fatal')

                idx = slice(arg.start, arg.stop)
                selected_vars = var.get_var(idx)
                if len(selected_vars)==1:
                    if len(indices) == 1:
                        return selected_vars[0]
                    else:
                        var = selected_vars[0]
                        return self._extract_indexed_from_var(var, indices[1:], expr)
                elif len(selected_vars)<1:
                    return None
                elif len(indices)==1:
                    return PythonTuple(*selected_vars)
                else:
                    return PythonTuple(*[self._extract_indexed_from_var(var, indices[1:], expr) for var in selected_vars])

            elif isinstance(arg, LiteralInteger):

                if len(indices)==1:
                    return var[arg]

                var = var[arg]
                return self._extract_indexed_from_var(var, indices[1:], expr)

            else:
                errors.report(INDEXED_TUPLE, symbol=var,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='fatal')

        if isinstance(var, PythonTuple) and not var.is_homogeneous:
            errors.report(LIST_OF_TUPLES, symbol=var,
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='error')

        for arg in var[indices].indices:
            if not isinstance(arg, Slice) and not \
                (hasattr(arg, 'dtype') and isinstance(arg.dtype, NativeInteger)):
                errors.report(INVALID_INDICES, symbol=var[indices],
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='error')
        return var[indices]

    def _create_PyccelOperator(self, expr, visited_args):
        """ Called by _visit_PyccelOperator and other classes
        inheriting from PyccelOperator
        """
        try:
            expr_new = type(expr)(*visited_args)
        except PyccelSemanticError as err:
            msg = str(err)
            errors.report(msg, symbol=expr,
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='fatal')
        return expr_new

    def _create_Duplicate(self, val, length):
        """ Called by _visit_PyccelMul when a Duplicate is
        identified
        """
        # Arguments have been visited in PyccelMul

        if not isinstance(val, (TupleVariable, PythonTuple)):
            errors.report("Unexpected Duplicate", symbol=Duplicate(val, length),
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='fatal')

        if val.is_homogeneous:
            return Duplicate(val, length)
        else:
            if isinstance(length, LiteralInteger):
                length = length.python_value
            else:
                errors.report("Cannot create inhomogeneous tuple of unknown size",
                    symbol=Duplicate(val, length),
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='fatal')
            if isinstance(val, TupleVariable):
                return PythonTuple(*(val.get_vars()*length))
            else:
                return PythonTuple(*(val.args*length))

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
            if isinstance(a.value, StarredArguments):
                args.extend([FunctionCallArgument(av) for av in a.value.args_var])
            else:
                args.append(a)
        return args

    def get_type_description(self, var, include_rank = True):
        """
        Provides a text description of the type of a variable
        (useful for error messages)
        Parameters
        ----------
        var          : Variable
                       The variable to describe
        include_rank : bool
                       Indicates whether rank information should be included
                       Default : True
        """
        dtype = var.dtype
        prec  = get_final_precision(var)
        descr = f'{dtype}{(prec * 2 if isinstance(dtype, NativeComplex) else prec) * 8 if prec else ""}'
        if include_rank and var.rank>0:
            dims = ','.join(':'*var.rank)
            descr += f'[{dims}]'
        return descr

    def _check_argument_compatibility(self, input_args, func_args, expr, elemental):
        """
        Check that the provided arguments match the expected types

        Parameters
        ----------
        input_args : list
                     The arguments provided to the function
        func_args  : list
                     The arguments expected by the function
        expr       : PyccelAstNode
                     The expression where this call is found (used for error output)
        elemental  : bool
                     Indicates if the function is elemental
        """
        if elemental:
            def incompatible(i_arg, f_arg):
                return (i_arg.dtype is not f_arg.dtype or \
                        get_final_precision(i_arg) != get_final_precision(f_arg))
        else:
            def incompatible(i_arg, f_arg):
                return (i_arg.dtype is not f_arg.dtype or \
                        get_final_precision(i_arg) != get_final_precision(f_arg) or
                        i_arg.rank != f_arg.rank)

        # Compare each set of arguments
        for idx, (i_arg, f_arg) in enumerate(zip(input_args, func_args)):
            i_arg = i_arg.value
            f_arg = f_arg.var
            # Ignore types which cannot be compared
            if (i_arg is Nil()
                    or isinstance(f_arg, FunctionAddress)
                    or f_arg.dtype is NativeGeneric()):
                continue
            # Check for compatibility
            if incompatible(i_arg, f_arg):
                expected  = self.get_type_description(f_arg, not elemental)
                type_name = self.get_type_description(i_arg, not elemental)
                received  = f'{i_arg} ({type_name})'

                errors.report(INCOMPATIBLE_ARGUMENT.format(idx+1, received, expr.func_name, expected),
                        symbol = expr,
                        severity='error')
            if f_arg.rank > 1 and i_arg.order != f_arg.order:
                errors.report(INCOMPATIBLE_ORDERING.format(idx=idx+1, arg=i_arg, func=expr.func_name, order=f_arg.order),
                        symbol = expr,
                        severity='error')

    def _handle_function(self, expr, func, args, is_method = False):
        """
        Create the node representing the function call.

        Create a FunctionCall or an instance of a PyccelInternalFunction
        from the function information and arguments.

        Parameters
        ----------
        expr : PyccelAstNode
               The expression where this call is found (used for error output).

        func : FunctionDef instance, Interface instance or PyccelInternalFunction type
               The function being called.

        args : tuple
               The arguments passed to the function.

        is_method : bool
                Indicates if the function is a method (and should return a DottedFunctionCall).

        Returns
        -------
        FunctionCall/PyccelInternalFunction
            The semantic representation of the call.
        """
        if isinstance(func, PyccelFunctionDef):
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
            except TypeError:
                errors.report(UNRECOGNISED_FUNCTION_CALL,
                        symbol = expr,
                        severity = 'fatal')

            return new_expr
        else:
            if self._current_function == func.name:
                if len(func.results)>0 and not isinstance(func.results[0].var, PyccelAstNode):
                    errors.report(RECURSIVE_RESULTS_REQUIRED, symbol=func, severity="fatal")

            parent_assign = expr.get_direct_user_nodes(lambda x: isinstance(x, Assign) and not isinstance(x, AugAssign))

            func_args = func.arguments if isinstance(func, FunctionDef) else func.functions[0].arguments
            func_results = func.results if isinstance(func, FunctionDef) else func.functions[0].results

            if not parent_assign and len(func_results) == 1 and func_results[0].var.rank > 0:
                tmp_var = PyccelSymbol(self.scope.get_new_name())
                assign = Assign(tmp_var, expr)
                assign.ast = expr.ast
                self._additional_exprs[-1].append(self._visit(assign))
                return self._visit(tmp_var)

            if len(args) > len(func_args):
                errors.report("Too many arguments passed in function call",
                        symbol = expr,
                        severity='fatal')
            # Sort arguments to match the order in the function definition
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

            args = input_args

            if is_method:
                new_expr = DottedFunctionCall(func, args, current_function = self._current_function, prefix = args[0].value)
            else:
                new_expr = FunctionCall(func, args, self._current_function)

            if None in new_expr.args:
                errors.report("Too few arguments passed in function call",
                        symbol = expr,
                        severity='error')
            elif isinstance(func, FunctionDef):
                self._check_argument_compatibility(args, func_args,
                            expr, func.is_elemental)
            return new_expr

    def _create_variable(self, name, dtype, rhs, d_lhs, arr_in_multirets=False):
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

        dtype : DataType
            The data type of the new variable.

        rhs : Variable
            The value assigned to the lhs. This is required to call
            self._infer_type recursively for tuples.

        d_lhs : dict
            Dictionary of properties for the new Variable.

        arr_in_multirets : bool, default: False
            If True, the variable that will be created is an array
            in multi-values return, false otherwise.

        Returns
        -------
        Variable
            The variable that has been created.
        """
        if isinstance(name, PyccelSymbol):
            is_temp = name.is_temp
        else:
            is_temp = False

        if isinstance(rhs, (PythonTuple, InhomogeneousTupleVariable, NumpyNonZero)) or \
                ((isinstance(rhs, FunctionCall) and rhs.pyccel_staging != 'syntactic') and len(rhs.funcdef.results)>1):
            if isinstance(rhs, FunctionCall):
                iterable = [r.var for r in rhs.funcdef.results]
            else:
                iterable = rhs
            elem_vars = []
            is_homogeneous = True
            elem_d_lhs_ref = None
            for i,r in enumerate(iterable):
                elem_name = self.scope.get_new_name( name + '_' + str(i) )
                elem_d_lhs = self._infer_type( r )

                if not arr_in_multirets:
                    self._ensure_target( r, elem_d_lhs )
                if elem_d_lhs_ref is None:
                    elem_d_lhs_ref = elem_d_lhs.copy()
                    is_homogeneous = elem_d_lhs['datatype'] is not NativeGeneric()
                elif elem_d_lhs != elem_d_lhs_ref:
                    is_homogeneous = False

                elem_dtype = elem_d_lhs.pop('datatype')

                var = self._create_variable(elem_name, elem_dtype, r, elem_d_lhs)
                elem_vars.append(var)

            if any(v.is_alias for v in elem_vars):
                d_lhs['memory_handling'] = 'alias'
            else:
                d_lhs['memory_handling'] = d_lhs.get('memory_handling', False) or 'heap'

            if is_homogeneous and not (d_lhs['memory_handling'] == 'alias' and isinstance(rhs, PythonTuple)):
                lhs = HomogeneousTupleVariable(dtype, name, **d_lhs, is_temp=is_temp)
            else:
                lhs = InhomogeneousTupleVariable(elem_vars, dtype, name, **d_lhs, is_temp=is_temp)

        else:
            new_type = HomogeneousTupleVariable \
                    if isinstance(rhs, (HomogeneousTupleVariable, Concatenate, Duplicate)) \
                    else Variable
            lhs = new_type(dtype, name, **d_lhs, is_temp=is_temp)

        return lhs

    def _ensure_target(self, rhs, d_lhs):
        """ Function using data about the new lhs to determine
        whether the lhs is an alias and the rhs is a target
        """

        if isinstance(rhs, NumpyTranspose) and rhs.internal_var.on_heap:
            d_lhs['memory_handling'] = 'alias'
            rhs.internal_var.is_target = True

        if isinstance(rhs, Variable) and rhs.is_ndarray:
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

            name = str(lhs)
            if lhs == '_':
                name = self.scope.get_new_name()
            dtype = d_var.pop('datatype')

            d_lhs = d_var.copy()
            # ISSUES #177: lhs must be a pointer when rhs is heap array
            if not arr_in_multirets:
                self._ensure_target(rhs, d_lhs)

            var = self.check_for_variable(lhs)

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
                        member = self._create_variable(new_name, dtype, rhs, d_lhs)

                        # Insert the attribute to the class scope
                        # Passing the original name ensures that the attribute can be found under this name
                        class_def.scope.insert_variable(member, attribute_name)

                        # Create the local DottedVariable
                        lhs = member.clone(member.name, new_class = DottedVariable, lhs = var)

                        # update the attributes of the class and push it to the scope
                        class_def.add_new_attribute(member)

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
                if d_lhs['rank'] > 0 and d_lhs.get('memory_handling', None) == 'stack':
                    for a in d_lhs['shape']:
                        if (isinstance(a, FunctionCall) and not a.funcdef.is_pure) or \
                                any(not f.funcdef.is_pure for f in a.get_attribute_nodes(FunctionCall)):
                            errors.report(STACK_ARRAY_SHAPE_UNPURE_FUNC, symbol=a.funcdef.name,
                            severity='error',
                            bounding_box=(self._current_ast_node.lineno,
                                self._current_ast_node.col_offset))
                        if (isinstance(a, Variable) and not a.is_argument) \
                                or not all(b.is_argument for b in a.get_attribute_nodes(Variable)):
                            errors.report(STACK_ARRAY_UNKNOWN_SHAPE, symbol=name,
                            severity='error',
                            bounding_box=(self._current_ast_node.lineno,
                                self._current_ast_node.col_offset))

                if not isinstance(lhs, DottedVariable):
                    new_name = self.scope.get_expected_name(name)
                    # Create new variable
                    lhs = self._create_variable(new_name, dtype, rhs, d_lhs, arr_in_multirets=arr_in_multirets)

                    # Add variable to scope
                    self.scope.insert_variable(lhs, name)

                # ...
                # Add memory allocation if needed
                array_declared_in_function = (isinstance(rhs, FunctionCall) and not isinstance(rhs.funcdef, PyccelFunctionDef) \
                                            and not getattr(rhs.funcdef, 'is_elemental', False) and not isinstance(lhs, HomogeneousTupleVariable)) or arr_in_multirets
                if lhs.on_heap and not array_declared_in_function:
                    if self.scope.is_loop:
                        # Array defined in a loop may need reallocation at every cycle
                        errors.report(ARRAY_DEFINITION_IN_LOOP, symbol=name,
                            severity='warning',
                            bounding_box=(self._current_ast_node.lineno,
                                self._current_ast_node.col_offset))
                        status='unknown'
                    else:
                        # Array defined outside of a loop will be allocated only once
                        status='unallocated'

                    # Create Allocate node
                    if isinstance(lhs, InhomogeneousTupleVariable):
                        args = [v for v in lhs.get_vars() if v.rank>0]
                        new_args = []
                        while len(args) > 0:
                            for a in args:
                                if isinstance(a, InhomogeneousTupleVariable):
                                    new_args.extend(v for v in a.get_vars() if v.rank>0)
                                else:
                                    new_expressions.append(Allocate(a,
                                        shape=a.alloc_shape, order=a.order, status=status))
                                    # Add memory deallocation for array variables
                                    self._allocs[-1].append(a)
                            args = new_args
                    else:
                        new_expressions.append(Allocate(lhs, shape=lhs.alloc_shape, order=lhs.order, status=status))
                # ...

                # ...
                # Add memory deallocation for array variables
                if lhs.is_ndarray and not lhs.on_stack:
                    # Create Deallocate node
                    self._allocs[-1].append(lhs)
                # ...

                # We cannot allow the definition of a stack array in a loop
                if lhs.is_stack_array and self.scope.is_loop:
                    errors.report(STACK_ARRAY_DEFINITION_IN_LOOP, symbol=name,
                        severity='error',
                        bounding_box=(self._current_ast_node.lineno,
                            self._current_ast_node.col_offset))

                # Not yet supported for arrays: x=y+z, x=b[:]
                # Because we cannot infer shape of right-hand side yet
                know_lhs_shape = (lhs.rank == 0) or all(sh is not None for sh in lhs.alloc_shape)

                if not know_lhs_shape:
                    msg = f"Cannot infer shape of right-hand side for expression {lhs} = {rhs}"
                    errors.report(PYCCEL_RESTRICTION_TODO+'\n'+msg,
                        bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                        severity='fatal')

            # Variable already exists
            else:

                self._ensure_inferred_type_matches_existing(dtype, d_var, var, is_augassign, new_expressions, rhs)

                # in the case of elemental, lhs is not of the same dtype as
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

    def _ensure_inferred_type_matches_existing(self, dtype, d_var, var, is_augassign, new_expressions, rhs):
        """
        Ensure that the inferred type of the new variable, matches the existing variable (which has the
        same name). If this is not the case then errors are raised preventing pyccel reaching the codegen
        stage.
        This function also handles any reallocations caused by differing shapes between the two objects.
        These allocations/deallocations are saved in the list new_expressions

        Parameters
        ----------
        dtype : DataType
            The inferred DataType.
        d_var : dict
            The inferred information about the variable. Usually created by the _infer_type function.
        var : Variable
            The existing variable.
        is_augassign : bool
            A boolean indicating if the assign statement is an augassign (tests are less strict).
        new_expressions : list
            A list to which any new expressions created are appended.
        rhs : PyccelAstNode
            The right hand side of the expression : lhs=rhs.
            If is_augassign is False, this value is not used.
        """
        precision = d_var.get('precision',None)
        internal_precision = default_precision[str(dtype)] if precision == -1 else precision

        # TODO improve check type compatibility
        if not hasattr(var, 'dtype'):
            name = var.name
            errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT.format('<module>', dtype),
                    symbol=f'{name}={dtype}',
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='fatal')

        elif not is_augassign and var.is_ndarray and var.is_target:
            errors.report(ARRAY_ALREADY_IN_USE,
                bounding_box=(self._current_ast_node.lineno,
                    self._current_ast_node.col_offset),
                        severity='error', symbol=var.name)

        elif not is_augassign and var.is_ndarray and isinstance(rhs, (Variable, IndexedElement)) and var.on_heap:
            errors.report(ASSIGN_ARRAYS_ONE_ANOTHER,
                bounding_box=(self._current_ast_node.lineno,
                    self._current_ast_node.col_offset),
                        severity='error', symbol=var)

        elif var.is_ndarray and var.is_alias and isinstance(rhs, NumpyNewArray):
            errors.report(INVALID_POINTER_REASSIGN,
                bounding_box=(self._current_ast_node.lineno,
                    self._current_ast_node.col_offset),
                        severity='error', symbol=var.name)

        elif var.is_ndarray and var.is_alias and not is_augassign:
            # we allow pointers to be reassigned multiple times
            # pointers reassigning need to call free_pointer func
            # to remove memory leaks
            new_expressions.append(Deallocate(var))

        elif str(dtype) != str(var.dtype) or \
                internal_precision != get_final_precision(var):
            if is_augassign:
                tmp_result = PyccelAdd(var, rhs)
                result_dtype = str(tmp_result.dtype)
                result_precision = get_final_precision(tmp_result)
                raise_error = (str(var.dtype) != result_dtype or \
                        get_final_precision(var) != result_precision)
            else:
                raise_error = True

            if raise_error:
                # Get type name from cast function (handles precision implicitly)
                try:
                    d1 = DtypePrecisionToCastFunction[var.dtype.name][var.precision].name
                except KeyError:
                    d1 = str(var.dtype)
                try:
                    d2 = DtypePrecisionToCastFunction[dtype.name][precision].name
                except KeyError:
                    d2 = str(var.dtype)

                name = var.name
                rhs_str = str(rhs)
                errors.report(INCOMPATIBLE_TYPES_IN_ASSIGNMENT.format(d1, d2),
                    symbol=f'{name}={rhs_str}',
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='error')

        elif not is_augassign:

            rank  = getattr(var, 'rank' , 'None')
            order = getattr(var, 'order', 'None')
            shape = getattr(var, 'shape', 'None')

            if (d_var['rank'] != rank) or (rank > 1 and d_var['order'] != order):

                txt = '|{name}| {dtype}{old} <-> {dtype}{new}'
                def format_shape(s):
                    return "" if s is None else s
                txt = txt.format(name=var.name, dtype=dtype, old=format_shape(var.shape),
                    new=format_shape(d_var['shape']))
                errors.report(INCOMPATIBLE_REDEFINITION, symbol=txt,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='error')

            elif d_var['shape'] != shape:

                if var.is_argument:
                    errors.report(ARRAY_IS_ARG, symbol=var,
                        severity='error',
                        bounding_box=(self._current_ast_node.lineno,
                            self._current_ast_node.col_offset))

                elif var.is_stack_array:
                    errors.report(INCOMPATIBLE_REDEFINITION_STACK_ARRAY, symbol=var.name,
                        severity='error',
                        bounding_box=(self._current_ast_node.lineno,
                            self._current_ast_node.col_offset))

                else:
                    var.set_changeable_shape()
                    previous_allocations = var.get_direct_user_nodes(lambda p: isinstance(p, Allocate))
                    if not previous_allocations:
                        errors.report("PYCCEL INTERNAL ERROR : Variable exists already, but it has never been allocated",
                                symbol=var, severity='fatal')

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
                    new_expressions.append(Allocate(var,
                        shape=d_var['shape'], order=d_var['order'],
                        status=status))

                    if status != 'unallocated':
                        errors.report(ARRAY_REALLOCATION, symbol=var.name,
                            severity='warning',
                            bounding_box=(self._current_ast_node.lineno,
                                self._current_ast_node.col_offset))
            else:
                # Same shape as before
                previous_allocations = var.get_direct_user_nodes(lambda p: isinstance(p, Allocate))

                if previous_allocations and previous_allocations[-1].get_user_nodes(IfSection) \
                        and not previous_allocations[-1].get_user_nodes((If)):
                    # If previously allocated in If still under construction
                    status = previous_allocations[-1].status

                    new_expressions.append(Allocate(var,
                        shape=d_var['shape'], order=d_var['order'],
                        status=status))

        if var.precision == -1 and precision != var.precision:
            var.use_exact_precision()

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
        index   = Variable('int',self.scope.get_new_name('to_delete'), is_temp=True)
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
            self._visit(Assign(iterator, iterator_rhs, ast=expr.ast))

            loop_elem = loop.body.body[0]

            if isinstance(loop_elem, Assign):
                # If the result contains a GeneratorComprehension, treat it and replace
                # it with it's lhs variable before continuing
                gens = set(loop_elem.get_attribute_nodes(GeneratorComprehension))
                if len(gens)==1:
                    gen = gens.pop()
                    assert isinstance(gen.lhs, PyccelSymbol) and gen.lhs.is_temp
                    gen_lhs = self.scope.get_new_name() if gen.lhs.is_temp else gen.lhs
                    assign = self._visit(Assign(gen_lhs, gen, ast=gen.ast))
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

        # Infer the final dtype of the expression
        d_var = self._infer_type(result)
        dtype = d_var.pop('datatype')
        d_var['is_temp'] = expr.lhs.is_temp

        lhs  = self.check_for_variable(lhs_name)
        if lhs:
            self._ensure_inferred_type_matches_existing(dtype, d_var, lhs, False, new_expr, None)
        else:
            lhs_name = self.scope.get_expected_name(lhs_name)
            lhs = Variable(dtype, lhs_name, **d_var)
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

        if isinstance(expr, FunctionalSum):
            val = LiteralInteger(0)
            if str_dtype(dtype) in ['float', 'complex']:
                val = LiteralFloat(0.0)
        elif isinstance(expr, FunctionalMin):
            val = math_constants['inf']
        elif isinstance(expr, FunctionalMax):
            val = PyccelUnarySub(math_constants['inf'])

        # Initialise result with correct initial value
        stmt = Assign(lhs, val)
        stmt.ast = expr.ast
        loops.insert(0, stmt)

        indices = [self._visit(i) for i in expr.indices]

        if isinstance(expr, FunctionalSum):
            expr_new = FunctionalSum(loops, lhs=lhs, indices = indices)
        elif isinstance(expr, FunctionalMin):
            expr_new = FunctionalMin(loops, lhs=lhs, indices = indices)
        elif isinstance(expr, FunctionalMax):
            expr_new = FunctionalMax(loops, lhs=lhs, indices = indices)
        expr_new.ast = expr.ast
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
        expr : pyccel.ast.basic.Basic
            Object to visit of type X.
        
        Returns
        -------
        pyccel.ast.basic.Basic
            AST object which is the semantic equivalent of expr.
        """

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors
        current_ast = self._current_ast_node

        if getattr(expr,'ast', None) is not None:
            self._current_ast_node = expr.ast

        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                obj = getattr(self, annotation_method)(expr)
                if isinstance(obj, Basic) and self._current_ast_node:
                    obj.ast = self._current_ast_node
                self._current_ast_node = current_ast
                return obj

        # Unknown object, we raise an error.
        return errors.report(PYCCEL_RESTRICTION_TODO, symbol=type(expr),
            bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
            severity='fatal')

    def _visit_Module(self, expr):
        body = self._visit(expr.program).body
        program_body      = []
        init_func_body    = []
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
        prog_name = 'prog_'+name_suffix
        prog_name = self.scope.get_new_name(prog_name)

        for b in body:
            if isinstance(b, If):
                if any(isinstance(i.condition, InProgram) for i in b.blocks):
                    for i in b.blocks:
                        if isinstance(i.condition, InProgram):
                            program_body.extend(i.body.body)
                        else:
                            init_func_body.append(i.body.body)
                else:
                    init_func_body.append(b)
            elif isinstance(b, CodeBlock):
                init_func_body.extend(b.body)
            else:
                init_func_body.append(b)

        variables = self.get_variables(self.scope)
        init_func = None
        free_func = None
        program   = None

        comment_types = (Header, MacroFunction, EmptyNode, Comment, CommentBlock)

        if not all(isinstance(l, comment_types) for l in init_func_body):
            # If there are any initialisation statements then create an initialisation function
            init_var = Variable(NativeBool(), self.scope.get_new_name('initialised'),
                                is_private=True)
            init_func_name = self.scope.get_new_name(name_suffix+'__init')
            # Ensure that the function is correctly defined within the namespaces
            init_scope = self.create_new_function_scope(init_func_name)
            for b in init_func_body:
                if isinstance(b, ScopedNode):
                    b.scope.update_parent_scope(init_scope, is_loop = True)
                if isinstance(b, FunctionalFor):
                    for l in b.loops:
                        if isinstance(l, ScopedNode):
                            l.scope.update_parent_scope(init_scope, is_loop = True)

            self.exit_function_scope()

            # Update variable scope for temporaries
            to_remove = []
            for v in self.scope.variables.values():
                if v.is_temp:
                    init_scope.insert_variable(v)
                    to_remove.append(v)

            # Remove in a second loop so the dictionary doesn't change during iteration
            for v in to_remove:
                self.scope.remove_variable(v)
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
                        func_defs = [vi for v in headers for vi in v.create_definition(is_external=is_external)]
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
        container = self._program_namespace.imports
        container['imports'][mod_name] = Import(mod_name, mod)

        if program_body:
            if init_func:
                import_init  = FunctionCall(init_func,[],[])
                program_body = [import_init, *program_body]
            if free_func:
                import_free  = FunctionCall(free_func,[],[])
                program_body = [*program_body, import_free]
            container = self._program_namespace
            program = Program(prog_name,
                            self.get_variables(container),
                            program_body,
                            container.imports['imports'].values(),
                            scope=self._program_namespace)

            mod.program = program

        return mod

    def _visit_PythonTuple(self, expr):
        ls = [self._visit(i) for i in expr]
        return PythonTuple(*ls)

    def _visit_PythonList(self, expr):
        ls = [self._visit(i) for i in expr]
        expr = PythonList(*ls)

        if not expr.is_homogeneous:
            errors.report(PYCCEL_RESTRICTION_INHOMOG_LIST, symbol=expr,
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='fatal')
        return expr

    def _visit_FunctionCallArgument(self, expr):
        value = self._visit(expr.value)
        a = FunctionCallArgument(value, expr.keyword)
        if isinstance(value, (PyccelArithmeticOperator, PyccelInternalFunction)) and value.rank:
            tmp_var = self.scope.get_new_name()
            assign = self._visit(Assign(tmp_var, expr.value, ast = expr.value.ast))
            self._additional_exprs[-1].append(assign)
            a = FunctionCallArgument(self._visit(tmp_var))
        return a

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
        return self.get_variable(name)

    def _visit_str(self, expr):
        return repr(expr)

    def _visit_Slice(self, expr):
        start = self._visit(expr.start) if expr.start is not None else None
        stop = self._visit(expr.stop) if expr.stop is not None else None
        step = self._visit(expr.step) if expr.step is not None else None

        return Slice(start, stop, step)

    def _visit_IndexedElement(self, expr):
        var = self._visit(expr.base)
        # TODO check consistency of indices with shape/rank
        args = [self._visit(idx) for idx in expr.indices]

        if (len(args) == 1 and isinstance(args[0], (TupleVariable, PythonTuple))):
            args = args[0]

        elif any(isinstance(a, (TupleVariable, PythonTuple)) for a in args):
            n_exprs = None
            for a in args:
                if hasattr(a, '__len__'):
                    if n_exprs:
                        assert n_exprs == len(a)
                    else:
                        n_exprs = len(a)
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
            var = python_builtin_datatype(name)

        if var is None:
            if name == '_':
                errors.report(UNDERSCORE_NOT_A_THROWAWAY,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='fatal')
            else:
                errors.report(UNDEFINED_VARIABLE, symbol=name,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='fatal')
        return var


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
                        symbol=expr,
                        bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                        severity='fatal')
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
                    return self._handle_function(expr, func, args)
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
                        symbol=expr,
                        bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                        severity='fatal')
        if isinstance(first, ClassDef):
            errors.report("Static class methods are not yet supported", symbol=expr,
                    severity='fatal')

        d_var = self._infer_type(first)
        if d_var.get('cls_base', None) is None:
            errors.report(f'Attribute {rhs_name} not found',
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='fatal')

        cls_base = d_var['cls_base']

        # look for a class method
        if isinstance(rhs, FunctionCall):
            macro = self.scope.find(rhs_name, 'macros')
            if macro is not None:
                master = macro.master
                args = rhs.args
                args = [lhs] + list(args)
                args = [self._visit(i) for i in args]
                args = macro.apply(args)
                return FunctionCall(master, args, self._current_function)

            args = [FunctionCallArgument(visited_lhs), *self._handle_function_args(rhs.args)]
            method = cls_base.get_method(rhs_name)
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
            bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
            severity='fatal')

    def _visit_PyccelOperator(self, expr):
        args     = [self._visit(a) for a in expr.args]
        return self._create_PyccelOperator(expr, args)

    def _visit_PyccelAdd(self, expr):
        args = [self._visit(a) for a in expr.args]
        if isinstance(args[0], (TupleVariable, PythonTuple, Concatenate, Duplicate)):
            is_homogeneous = all((isinstance(a, (TupleVariable, PythonTuple)) and a.is_homogeneous) \
                                or isinstance(a, (Concatenate, Duplicate)) for a in args)
            if is_homogeneous:
                return Concatenate(*args)
            else:
                def get_vars(a):
                    if isinstance(a, InhomogeneousTupleVariable):
                        return a.get_vars()
                    elif isinstance(a, PythonTuple):
                        return a.args
                    elif isinstance(a, HomogeneousTupleVariable):
                        n_vars = len(a)
                        if not isinstance(len(a), (LiteralInteger, int)):
                            errors.report("Can't create an inhomogeneous tuple using a homogeneous tuple of unknown size",
                                    symbol=expr, severity='fatal')
                        return [a[i] for i in range(n_vars)]
                    else:
                        a_type = type(a)
                        raise NotImplementedError(f"Unexpected type {a_type} in tuple addition")
                tuple_args = [ai for a in args for ai in get_vars(a)]
                expr_new = PythonTuple(*tuple_args)
        else:
            expr_new = self._create_PyccelOperator(expr, args)
        return expr_new

    def _visit_PyccelMul(self, expr):
        args = [self._visit(a) for a in expr.args]
        if isinstance(args[0], (TupleVariable, PythonTuple, PythonList)):
            expr_new = self._create_Duplicate(args[0], args[1])
        elif isinstance(args[1], (TupleVariable, PythonTuple, PythonList)):
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

    def _visit_MathSqrt(self, expr):
        func = self.scope.find(expr.funcdef, 'functions')
        arg, = self._handle_function_args(expr.args) #pylint: disable=unbalanced-tuple-unpacking
        if isinstance(arg.value, PyccelMul):
            mul1, mul2 = arg.value.args
            mul1_syn, mul2_syn = expr.args[0].value.args
            is_abs = False
            if mul1 is mul2 and mul1.dtype in (NativeInteger(), NativeFloat()):
                pyccel_stage.set_stage('syntactic')

                fabs_name = self.scope.get_new_name('fabs')
                imp_name = AsName('fabs', fabs_name)
                new_import = Import('math',imp_name)
                self._visit(new_import)
                new_call = FunctionCall(fabs_name, [mul1_syn])

                pyccel_stage.set_stage('semantic')

                return self._visit(new_call)
            elif isinstance(mul1, (NumpyConjugate, PythonConjugate)) and mul1.internal_var is mul2:
                is_abs = True
                abs_arg = mul2_syn
            elif isinstance(mul2, (NumpyConjugate, PythonConjugate)) and mul1 is mul2.internal_var:
                is_abs = True
                abs_arg = mul1_syn

            if is_abs:
                pyccel_stage.set_stage('syntactic')

                abs_name = self.scope.get_new_name('abs')
                imp_name = AsName('abs', abs_name)
                new_import = Import('numpy',imp_name)
                self._visit(new_import)
                new_call = FunctionCall(abs_name, [abs_arg])

                pyccel_stage.set_stage('semantic')

                # Cast to preserve final dtype
                return PythonComplex(self._visit(new_call))
        elif isinstance(arg.value, PyccelPow):
            base, exponent = arg.value.args
            base_syn, _ = expr.args[0].value.args
            if exponent == 2 and base.dtype in (NativeInteger(), NativeFloat()):
                pyccel_stage.set_stage('syntactic')

                fabs_name = self.scope.get_new_name('fabs')
                imp_name = AsName('fabs', fabs_name)
                new_import = Import('math',imp_name)
                self._visit(new_import)
                new_call = FunctionCall(fabs_name, [base_syn])

                pyccel_stage.set_stage('semantic')

                return self._visit(new_call)

        return self._handle_function(expr, func, (arg,))

    def _visit_CmathPolar(self, expr):
        arg, = self._handle_function_args(expr.args) #pylint: disable=unbalanced-tuple-unpacking
        z = arg.value
        x = PythonReal(z)
        y = PythonImag(z)
        x_var = self.scope.get_temporary_variable(z, dtype=NativeFloat())
        y_var = self.scope.get_temporary_variable(z, dtype=NativeFloat())
        self._additional_exprs[-1].append(Assign(x_var, x))
        self._additional_exprs[-1].append(Assign(y_var, y))
        r = MathSqrt(PyccelAdd(PyccelMul(x_var,x_var), PyccelMul(y_var,y_var)))
        t = MathAtan2(y_var, x_var)
        self.insert_import('math', AsName(MathSqrt, 'sqrt'))
        self.insert_import('math', AsName(MathAtan2, 'atan2'))
        return PythonTuple(r,t)

    def _visit_CmathRect(self, expr):
        arg_r, arg_phi = self._handle_function_args(expr.args) #pylint: disable=unbalanced-tuple-unpacking
        r = arg_r.value
        phi = arg_phi.value
        x = PyccelMul(r, MathCos(phi))
        y = PyccelMul(r, MathSin(phi))
        self.insert_import('math', AsName(MathCos, 'cos'))
        self.insert_import('math', AsName(MathSin, 'sin'))
        return PyccelAdd(x, PyccelMul(y, LiteralImaginaryUnit()))

    def _visit_CmathPhase(self, expr):
        arg, = self._handle_function_args(expr.args) #pylint: disable=unbalanced-tuple-unpacking
        var = arg.value
        if var.dtype is not NativeComplex():
            return LiteralFloat(0.0)
        else:
            self.insert_import('math', AsName(MathAtan2, 'atan2'))
            return MathAtan2(PythonImag(var), PythonReal(var))

    def _visit_Lambda(self, expr):
        expr_names = set(str(a) for a in expr.expr.get_attribute_nodes(PyccelSymbol))
        var_names = map(str, expr.variables)
        missing_vars = expr_names.difference(var_names)
        if len(missing_vars) > 0:
            errors.report(UNDEFINED_LAMBDA_VARIABLE, symbol = missing_vars,
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='fatal')
        funcs = expr.expr.get_attribute_nodes(FunctionCall)
        for func in funcs:
            name = _get_name(func)
            f = self.scope.find(name, 'symbolic_functions')
            if f is None:
                errors.report(UNDEFINED_LAMBDA_FUNCTION, symbol=name,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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

        func     = self.scope.find(name, 'functions')

        # Check for specialised method
        if isinstance(func, PyccelFunctionDef):
            annotation_method = '_visit_' + func.cls_name.__name__
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(expr)

        args = self._handle_function_args(expr.args)
        # Correct keyword names if scope is available
        # The scope is only available if the function body has been parsed
        # (i.e. not for headers or builtin functions)
        if isinstance(func, FunctionDef) and func.scope:
            args = [a if a.keyword is None else \
                    FunctionCallArgument(a.value, func.scope.get_expected_name(a.keyword)) \
                    for a in args]


        if name == 'lambdify':
            args = self.scope.find(str(expr.args[0]), 'symbolic_functions')
        F = pyccel_builtin_function(expr, args)

        if F is not None:
            return F

        elif self.scope.find(name, 'cls_constructs'):

            # TODO improve the test
            # we must not invoke the scope like this

            cls = self.scope.find(name, 'classes')
            d_methods = cls.methods_as_dict
            method = d_methods.pop('__init__', None)

            if method is None:

                # TODO improve case of class with the no __init__

                errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='error')
            d_var = {'datatype': self.get_class_construct(method.cls_name)(),
                    'memory_handling':'stack',
                    'shape' : None,
                    'rank' : 0,
                    'is_target' : False,
                    'cls_base' : self.scope.find(method.cls_name, 'classes')}
            cls_variable = self._assign_lhs_variable(expr.current_user_node.lhs, d_var, expr, [], True)
            args = (FunctionCallArgument(cls_variable), *args)
            # TODO check compatibility
            # TODO treat parametrized arguments.

            expr = ConstructorCall(method, args, cls_variable)
            #if len(stmts) > 0:
            #    stmts.append(expr)
            #    return CodeBlock(stmts)
            return expr
        else:

            # first we check if it is a macro, in this case, we will create
            # an appropriate FunctionCall

            macro = self.scope.find(name, 'macros')
            if macro is not None:
                func = macro.master.funcdef
                name = _get_name(func.name)
                args = macro.apply(args)
            else:
                func = self.scope.find(name, 'functions')
            if func is None:
                return errors.report(UNDEFINED_FUNCTION, symbol=name,
                        bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                        severity='fatal')
            else:
                return self._handle_function(expr, func, args)

    def _visit_Expr(self, expr):
        errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
            bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
            severity='fatal')


    def _visit_Assign(self, expr):
        # TODO unset position at the end of this part
        new_expressions = []
        ast = expr.ast
        assert(ast)

        rhs = expr.rhs
        lhs = expr.lhs

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
            if value_true.rank > 0 or value_true.dtype is NativeString():
                # Temporarily deactivate type checks to construct syntactic assigns
                pyccel_stage.set_stage('syntactic')
                assign_true  = Assign(lhs, rhs.value_true, ast = ast)
                assign_false = Assign(lhs, rhs.value_false, ast = ast)
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

                if not sympy_iterable(lhs):
                    lhs = [lhs]
                results_shapes = macro.get_results_shapes(args)
                for m_result, shape, result in zip(macro.results, results_shapes, lhs):
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
                        bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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
                                  bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                                  severity='fatal')

        else:
            rhs = self._visit(rhs)

        if isinstance(rhs, NumpyResultType):
            errors.report("Cannot assign a datatype to a variable.",
                    symbol=expr, severity='error')

        if isinstance(rhs, ConstructorCall):
            return rhs
        elif isinstance(rhs, FunctionDef):

            # case of lambdify

            rhs = rhs.rename(expr.lhs.name)
            for i in rhs.body:
                i.ast = ast
            return rhs

        elif isinstance(rhs, CodeBlock):
            if len(rhs.body)>1 and isinstance(rhs.body[1], FunctionalFor):
                return rhs

            # case of complex stmt
            # that needs to be splitted
            # into a list of stmts
            stmts = rhs.body
            stmt  = stmts[-1]
            lhs   = expr.lhs
            if isinstance(lhs, PyccelSymbol):
                name = lhs
                if self.check_for_variable(name) is None:
                    d_var = self._infer_type(stmt)
                    dtype = d_var.pop('datatype')
                    lhs = Variable(dtype, name , **d_var, is_temp = lhs.is_temp)
                    self.scope.insert_variable(lhs)

            if isinstance(expr, Assign):
                stmt = Assign(lhs, stmt)
            elif isinstance(expr, AugAssign):
                stmt = AugAssign(lhs, expr.op, stmt)
            stmt.ast = ast
            stmts[-1] = stmt
            return CodeBlock(stmts)

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
                    assert(len(c_ranks) == 1)
                    arg = call_args[0].value
                    d_var['shape'          ] = arg.shape
                    d_var['rank'           ] = arg.rank
                    d_var['memory_handling'] = arg.memory_handling
                    d_var['order'          ] = arg.order

        elif isinstance(rhs, NumpyTranspose):
            d_var  = self._infer_type(rhs)
            if d_var['memory_handling'] == 'alias' and not isinstance(lhs, IndexedElement):
                rhs = rhs.internal_var
        elif isinstance(rhs, PyccelInternalFunction) and isinstance(rhs.dtype, NativeVoid):
            if expr.lhs.is_temp:
                return rhs
            else:
                raise NotImplementedError("Cannot assign result of a function without a return")

        else:
            d_var  = self._infer_type(rhs)
            d_list = d_var if isinstance(d_var, list) else [d_var]

            for d in d_list:
                name = d['datatype'].__class__.__name__

                if name.startswith('Pyccel'):
                    name = name[6:]
                    d['cls_base'] = self.scope.find(name, 'classes')
                    #TODO: Avoid writing the default variables here
                    if d_var.get('is_target', False) or d_var.get('memory_handling', False) == 'alias':
                        d['memory_handling'] = 'alias'
                    else:
                        d['memory_handling'] = d_var.get('memory_handling', False) or 'heap'

                    # TODO if we want to use pointers then we set target to true
                    # in the ConsturcterCall

                if isinstance(rhs, Variable) and rhs.is_target:
                    # case of rhs is a target variable the lhs must be a pointer
                    d['is_target' ] = False
                    d['memory_handling'] = 'alias'

        lhs = expr.lhs
        if isinstance(lhs, (PyccelSymbol, DottedName)):
            if isinstance(d_var, list):
                if len(d_var) == 1:
                    d_var = d_var[0]
                else:
                    errors.report(WRONG_NUMBER_OUTPUT_ARGS, symbol=expr,
                        bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                        severity='error')
                    return None
            lhs = self._assign_lhs_variable(lhs, d_var, rhs, new_expressions, isinstance(expr, AugAssign))
        elif isinstance(lhs, PythonTuple):
            n = len(lhs)
            if isinstance(rhs, (PythonTuple, InhomogeneousTupleVariable, FunctionCall)):
                if isinstance(rhs, FunctionCall):
                    r_iter = [r.var for r in rhs.funcdef.results]
                else:
                    r_iter = rhs
                new_lhs = []
                for i,(l,r) in enumerate(zip(lhs,r_iter)):
                    d = self._infer_type(r)
                    new_lhs.append( self._assign_lhs_variable(l, d, r, new_expressions, isinstance(expr, AugAssign),arr_in_multirets=r.rank>0 ) )
                lhs = PythonTuple(*new_lhs)

            elif isinstance(rhs, HomogeneousTupleVariable):
                new_lhs = []
                d_var = self._infer_type(rhs[0])
                new_rhs = []
                for i,l in enumerate(lhs):
                    new_lhs.append( self._assign_lhs_variable(l, d_var.copy(),
                        rhs[i], new_expressions, isinstance(expr, AugAssign)) )
                    new_rhs.append(rhs[i])
                rhs = PythonTuple(*new_rhs)
                d_var = [d_var]
                lhs = PythonTuple(*new_lhs)

            elif isinstance(d_var, list) and len(d_var)== n:
                new_lhs = []
                if hasattr(rhs,'__getitem__'):
                    for i,l in enumerate(lhs):
                        new_lhs.append( self._assign_lhs_variable(l, d_var[i].copy(), rhs[i], new_expressions, isinstance(expr, AugAssign)) )
                else:
                    for i,l in enumerate(lhs):
                        new_lhs.append( self._assign_lhs_variable(l, d_var[i].copy(), rhs, new_expressions, isinstance(expr, AugAssign)) )
                lhs = PythonTuple(*new_lhs)

            elif d_var['shape'][0]==n:
                new_lhs = []
                new_rhs = []

                for l, r in zip(lhs, rhs):
                    new_lhs.append( self._assign_lhs_variable(l, self._infer_type(r), r, new_expressions, isinstance(expr, AugAssign)) )
                    new_rhs.append(r)

                lhs = PythonTuple(*new_lhs)
                rhs = new_rhs
            else:
                errors.report(WRONG_NUMBER_OUTPUT_ARGS, symbol=expr,
                    bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='error')
                return None
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
                if isinstance(l, (PythonTuple, InhomogeneousTupleVariable)) \
                        and isinstance(r,(PythonTuple, TupleVariable, list)):
                    new_lhs.extend(l)
                    new_rhs.extend(r)
                    # Repeat step to handle tuples of tuples of etc.
                    unravelling = True
                else:
                    new_lhs.append(l)
                    new_rhs.append(r)
            lhs = new_lhs
            rhs = new_rhs

        # Examine each assign and determine assign type (Assign, AliasAssign, etc)
        for l, r in zip(lhs,rhs):
            if isinstance(expr, AugAssign):
                new_expr = AugAssign(l, expr.op, r)
            else:
                is_pointer_i = l.is_alias if isinstance(l, Variable) else is_pointer
                new_expr = Assign(l, r)

                if is_pointer_i:
                    new_expr = AliasAssign(l, r)

                elif new_expr.is_symbolic_alias:
                    new_expr = SymbolicAssign(l, r)

                    # in a symbolic assign, the rhs can be a lambda expression
                    # it is then treated as a def node

                    F = self.scope.find(l, 'symbolic_functions')
                    if F is None:
                        self.insert_symbolic_function(new_expr)
                    else:
                        errors.report(PYCCEL_RESTRICTION_TODO,
                                      bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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
            indices = [Variable('int', self.scope.get_new_name(), is_temp=True)
                        for i in range(iterable.num_loop_counters_required)]
            iterable.set_loop_counter(*indices)
        else:
            if isinstance(iterable.iterable, PythonEnumerate):
                syntactic_index = iterator[0]
            else:
                iterator = self.scope.get_expected_name(iterator)
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
                   bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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
                var   = self._create_variable(var, 'int', start, {})
                dvar  = self._infer_type(var)
                stop  = a.stop
                start = a.start
                step  = a.step
            elif isinstance(a, (PythonZip, PythonEnumerate)):
                dvar  = self._infer_type(a.element)
                dtype = dvar.pop('datatype')
                if dvar['rank'] > 0:
                    dvar['rank' ] -= 1
                    dvar['shape'] = (dvar['shape'])[1:]
                if dvar['rank'] == 0:
                    dvar['memory_handling'] = 'stack'
                var  = Variable(dtype, var, **dvar)
                stop = a.element.shape[0]
            elif isinstance(a, Variable):
                dvar  = self._infer_type(a)
                dtype = dvar.pop('datatype')
                if dvar['rank'] == 1:
                    dvar['rank']  = 0
                    dvar['shape'] = None
                if dvar['rank'] > 1:
                    dvar['rank'] -= 1
                    dvar['shape'] = (dvar['shape'])[1:]
                if dvar['rank'] == 0:
                    dvar['memory_handling'] = 'stack'

                var  = Variable(dtype, var, **dvar)
                stop = a.shape[0]
            else:
                errors.report(PYCCEL_RESTRICTION_TODO,
                              bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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
                          bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                          severity='error')

            # sympy is necessary to carry out the summation
            dim   = dim.subs(sp_indices[i], start+step*sp_indices[i])
            dim   = Summation(dim, (sp_indices[i], 0, size-1))
            dim   = dim.doit()

        try:
            dim = sympy_to_pyccel(dim, idx_subs)
        except TypeError:
            errors.report(PYCCEL_RESTRICTION_LIST_COMPREHENSION_SIZE + f'\n Deduced size : {dim}',
                          bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                          severity='fatal')

        # TODO find a faster way to calculate dim
        # when step>1 and not isinstance(dim, Sum)
        # maybe use the c++ library of sympy

        # we annotate the target to infere the type of the list created

        target = self._visit(target)
        d_var = self._infer_type(target)

        dtype = d_var['datatype']

        if dtype is NativeGeneric():
            errors.report(LIST_OF_TUPLES,
                          bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                          severity='fatal')

        d_var['memory_handling'] = 'heap'
        d_var['rank'] += 1
        shape = [dim]
        if d_var['rank'] != 1:
            d_var['order'] = 'C'
            shape += list(d_var['shape'])
        else:
            d_var['order'] = None
        d_var['shape'] = shape

        # ...
        # TODO [YG, 30.10.2020]:
        #  - Check if we should allow the possibility that is_stack_array=True
        # ...
        lhs_symbol = expr.lhs.base
        ne = []
        lhs = self._assign_lhs_variable(lhs_symbol, d_var, rhs=expr, new_expressions=ne, is_augassign=False)
        lhs_alloc = ne[0]

        if isinstance(target, PythonTuple) and not target.is_homogeneous:
            errors.report(LIST_OF_TUPLES, symbol=expr,
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                severity='error')

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
            if expr.lhs.is_temp:
                lhs = PyccelSymbol(self.scope.get_new_name(), is_temp=True)
            else:
                lhs = expr.lhs

            creation = self._visit(Assign(lhs, expr, ast=expr.ast))
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

        name_symbol = PyccelSymbol('__name__')
        main = LiteralString('__main__')
        prog_check = isinstance(condition, PyccelEq) \
                and all(a in (name_symbol, main) for a in condition.args)

        if prog_check:
            cond = InProgram()
            self.change_to_program_scope()
        else:
            cond = self._visit(expr.condition)
        body = self._visit(expr.body)
        if prog_check:
            # Calling the Garbage collecting,
            # it will add the necessary Deallocate nodes
            # to the ast
            body.insert2body(*self._garbage_collector(body))
            self.change_to_module_scope()

        return IfSection(cond, body)

    def _visit_If(self, expr):
        args = [self._visit(i) for i in expr.blocks]

        conds = [b.condition for b in args]
        if any(isinstance(c, InProgram) for c in conds):
            if not all(isinstance(c, (InProgram,LiteralTrue)) for c in conds):
                errors.report("Determination of main module is too complicated to handle",
                        symbol=expr, severity='error')

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
        if value_true.rank > 0 or value_true.dtype is NativeString():
            lhs = PyccelSymbol(self.scope.get_new_name(), is_temp=True)
            # Temporarily deactivate type checks to construct syntactic assigns
            pyccel_stage.set_stage('syntactic')
            assign_true  = Assign(lhs, expr.value_true, ast = expr.ast)
            assign_false = Assign(lhs, expr.value_false, ast = expr.ast)
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

    def _visit_VariableHeader(self, expr):
        warnings.warn("Support for specifying types via headers will be removed in " +
                      "a future version of Pyccel. This annotation may be unnecessary " +
                      "in your code. If you find it is necessary please open a discussion " +
                      "at https://github.com/pyccel/pyccel/discussions so we do not " +
                      "remove support until an alternative is in place.", FutureWarning)

        # TODO improve
        #      move it to the ast like create_definition for FunctionHeader?

        name  = expr.name
        d_var = expr.dtypes.copy()
        dtype = d_var.pop('datatype')
        d_var.pop('is_func')

        var = Variable(dtype, name, **d_var)
        self.scope.insert_variable(var)
        return expr

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
                a = self._visit(Assign(v, r, ast=expr.ast))
                assigns.append(a)

        results = [self._visit(i.var) for i in return_objs]

        # add the Deallocate node before the Return node and eliminating the Deallocate nodes
        # the arrays that will be returned.
        code = assigns + [Deallocate(i) for i in self._allocs[-1] if i not in results]
        if code:
            expr  = Return(results, CodeBlock(code))
        else:
            expr  = Return(results)
        return expr

    def _visit_FunctionDef(self, expr):
        name            = self.scope.get_expected_name(expr.name)
        cls_name        = expr.cls_name
        decorators      = expr.decorators
        funcs           = []
        sub_funcs       = []
        func_interfaces = []
        is_pure         = expr.is_pure
        is_elemental    = expr.is_elemental
        is_private      = expr.is_private
        is_inline       = expr.is_inline
        doc_string      = self._visit(expr.doc_string) if expr.doc_string else expr.doc_string
        headers = []

        not_used = [d for d in decorators if d not in def_decorators.__all__]
        if len(not_used) >= 1:
            errors.report(UNUSED_DECORATORS, symbol=', '.join(not_used), severity='warning')

        args_number = len(expr.arguments)
        templates = self.scope.find_all('templates')
        if decorators['template']:
            # Load templates dict from decorators dict
            templates.update(decorators['template']['template_dict'])

        tmp_headers = expr.headers
        python_name = expr.scope.get_python_name(name)
        if cls_name:
            tmp_headers += self.get_headers(cls_name + '.' + python_name)
            args_number -= 1
        else:
            tmp_headers += self.get_headers(python_name)
        for header in tmp_headers:
            if all(header.dtypes != hd.dtypes for hd in headers):
                headers.append(header)
            else:
                errors.report(DUPLICATED_SIGNATURE, symbol=header,
                        severity='warning')
        for hd in headers:
            if (args_number != len(hd.dtypes)):
                n_types = len(hd.dtypes)
                msg = f"""The number of arguments in the function {name} ({args_number}) does not match the number
                        of types in decorator/header ({n_types})."""
                if (args_number < len(hd.dtypes)):
                    errors.report(msg, symbol=expr.arguments, severity='warning')
                else:
                    errors.report(msg, symbol=expr.arguments, severity='fatal')

        interfaces = []
        if len(headers) == 0:
            # check if a header is imported from a header file
            # TODO improve in the case of multiple headers ( interface )
            func       = self.scope.find(name, 'functions')
            if func and func.is_header:
                interfaces = [func]

        if expr.arguments and not headers and not interfaces:

            # TODO ERROR wrong position

            errors.report(FUNCTION_TYPE_EXPECTED, symbol=name,
                   bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                   severity='error')

        # We construct a FunctionDef from each function header
        for hd in headers:
            interfaces += hd.create_definition(templates)

        if not interfaces:
            # this for the case of a function without arguments => no headers
            interfaces = [FunctionDef(name, [], [], [])]

#        TODO move this to codegen
#        vec_func = None
#        if 'vectorize' in decorators:
#            #TODO move to another place
#            vec_name  = 'vec_' + name
#            arg       = decorators['vectorize'][0]
#            arg       = str(arg.name)
#            args      = [str(i.name) for i in expr.arguments]
#            index_arg = args.index(arg)
#            arg       = Symbol(arg)
#            vec_arg   = arg
#            index     = self.namespace.get_new_name()
#            range_    = Function('range')(Function('len')(arg))
#            args      = symbols(args)
#            args[index_arg] = vec_arg[index]
#            body_vec        = Assign(args[index_arg], Function(name)(*args))
#            body_vec.ast = expr.ast
#            body_vec   = [For(index, range_, [body_vec])]
#            header_vec = header.vectorize(index_arg)
#            vec_func   = expr.vectorize(body_vec, header_vec)

        interface_name = name

        for i, m in enumerate(interfaces):
            args           = []
            arg_vars       = []
            results        = []
            global_vars    = []
            imports        = []
            arg            = None
            arguments      = expr.arguments
            header_results = m.results

            if len(interfaces) > 1:
                name = interface_name + '_' + str(i).zfill(2)
            scope = self.create_new_function_scope(name, decorators = decorators,
                    used_symbols = expr.scope.local_used_symbols.copy(),
                    original_symbols = expr.scope.python_names.copy())

            if cls_name and str(arguments[0].name) == 'self':
                arg       = arguments[0]
                arguments = arguments[1:]
                dt        = self.get_class_construct(cls_name)()
                cls_base  = self.scope.find(cls_name, 'classes')
                cls_base.scope.insert_symbols(expr.scope.local_used_symbols.copy())
                var       = Variable(dt, 'self', cls_base=cls_base, is_argument=True)
                self.scope.insert_variable(var)

            if arguments:
                for (a, ah) in zip(arguments, m.arguments):
                    ahv = ah.var
                    if isinstance(ahv, FunctionAddress):
                        d_var = {}
                        d_var['is_argument'] = True
                        d_var['memory_handling'] = 'alias'
                        if a.has_default:
                            # optional argument only if the value is None
                            if isinstance(a.value, Nil):
                                d_var['is_optional'] = True
                        a_new = FunctionAddress(self.scope.get_expected_name(a.name),
                                        ahv.arguments, ahv.results, [], **d_var)
                    else:
                        d_var = self._infer_type(ahv)
                        d_var['shape'] = ahv.alloc_shape
                        d_var['is_argument'] = True
                        d_var['is_const'] = ahv.is_const
                        dtype = d_var.pop('datatype')
                        if not d_var['cls_base']:
                            try:
                                d_var['cls_base'] = get_cls_base( dtype, d_var['precision'], d_var['rank'] )
                            except KeyError:
                                d_var['cls_base'] = self.scope.find( dtype, 'classes' )

                        if 'allow_negative_index' in self.scope.decorators:
                            if a.name in decorators['allow_negative_index']:
                                d_var.update(allows_negative_indexes=True)
                        if a.has_default:
                            # optional argument only if the value is None
                            if isinstance(a.value, Nil):
                                d_var['is_optional'] = True
                        a_new = Variable(dtype, self.scope.get_expected_name(a.name), **d_var)


                    value = None if a.value is None else self._visit(a.value)
                    if isinstance(value, Literal) and \
                            value.dtype is a_new.dtype and \
                            value.precision != a_new.precision:
                        value = convert_to_literal(value.python_value, a_new.dtype, a_new.precision)

                    arg_new = FunctionDefArgument(a_new,
                                value=value,
                                kwonly=a.is_kwonly,
                                annotation=ah.annotation)

                    args.append(arg_new)
                    arg_vars.append(a_new)
                    if isinstance(a_new, FunctionAddress):
                        self.insert_function(a_new)
                    else:
                        self.scope.insert_variable(a_new, a.name)
            results = expr.results
            if header_results:
                new_results = []

                for a, ah in zip(results, header_results):
                    av = a.var
                    d_var = self._infer_type(ah.var)
                    dtype = d_var.pop('datatype')
                    a_new = Variable(dtype, self.scope.get_expected_name(av),
                            **d_var, is_temp = av.is_temp)
                    self.scope.insert_variable(a_new, av)
                    new_results.append(FunctionDefResult(a_new, annotation = ah.annotation))

                results = new_results

            # insert the FunctionDef into the scope
            # to handle the case of a recursive function
            # TODO improve in the case of an interface
            recursive_func_obj = FunctionDef(name, args, results, [])
            self.insert_function(recursive_func_obj)

            # Create a new list that store local variables for each FunctionDef to handle nested functions
            self._allocs.append([])

            # we annotate the body
            body = self._visit(expr.body)

            # Calling the Garbage collecting,
            # it will add the necessary Deallocate nodes
            # to the body of the function
            body.insert2body(*self._garbage_collector(body))

            results = [self._visit(a) for a in results]

            if arg and cls_name:
                dt       = self.get_class_construct(cls_name)()
                cls_base = self.scope.find(cls_name, 'classes')
                var      = Variable(dt, 'self', cls_base=cls_base)
                args     = [FunctionDefArgument(var)] + args

            # Determine local and global variables
            global_vars = list(self.get_variables(self.scope.parent_scope))
            global_vars = [g for g in global_vars if body.is_user_of(g)]

            # get the imports
            imports   = self.scope.imports['imports'].values()
            # Prefer dict to set to preserve order
            imports   = list({imp:None for imp in imports}.keys())

            # remove the FunctionDef from the function scope
            # TODO improve func_ is None in the case of an interface
            func_     = self.scope.functions.pop(name, None)
            is_recursive = False
            # check if the function is recursive if it was called on the same scope
            if func_ and func_.is_recursive:
                is_recursive = True

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
            for a in args:
                if a.name in chain(results_names, ['self']) or a.var in all_assigned:
                    v = a.var
                    if isinstance(v, Variable) and v.is_const:
                        msg = f"Cannot modify 'const' argument ({v})"
                        errors.report(msg, bounding_box=(self._current_ast_node.lineno,
                            self._current_ast_node.col_offset),
                            severity='fatal')
                else:
                    a.make_const()
            # ...

            # Raise an error if one of the return arguments is an alias.
            for r in results:
                if r.var.is_alias:
                    errors.report(UNSUPPORTED_POINTER_RETURN_VALUE,
                    symbol=r,bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                    severity='error')

            func_kwargs = {
                    'global_vars':global_vars,
                    'cls_name':cls_name,
                    'is_pure':is_pure,
                    'is_elemental':is_elemental,
                    'is_private':is_private,
                    'imports':imports,
                    'decorators':decorators,
                    'is_recursive':is_recursive,
                    'functions': sub_funcs,
                    'interfaces': func_interfaces,
                    'doc_string': doc_string,
                    'scope': scope
                    }
            if is_inline:
                func_kwargs['namespace_imports'] = namespace_imports
                global_funcs = [f for f in body.get_attribute_nodes(FunctionDef) if self.scope.find(f.name, 'functions')]
                func_kwargs['global_funcs'] = global_funcs
                cls = InlineFunctionDef
            else:
                cls = FunctionDef
            func = cls(name,
                    args,
                    results,
                    body,
                    **func_kwargs)
            if not is_recursive:
                recursive_func_obj.invalidate_node()

            if cls_name:
                cls = self.scope.find(cls_name, 'classes')

                # update the class methods
                if expr.name == func.name:
                    cls.add_new_method(func)

            funcs += [func]

            #clear the sympy cache
            #TODO clear all variable except the global ones
            cache.clear_cache()
        if len(funcs) == 1:
            funcs = funcs[0]
            self.insert_function(funcs)

        else:
            for f in funcs:
                self.insert_function(f)

            funcs = Interface(interface_name, funcs)
            if cls_name:
                cls = self.scope.find(cls_name, 'classes')
                cls.add_new_interface(funcs)
            self.insert_function(funcs)
#        TODO move this to codegen
#        if vec_func:
#           self._visit_FunctionDef(vec_func)
#           vec_func = self.scope.functions.pop(vec_name)
#           if isinstance(funcs, Interface):
#               funcs = list(funcs.funcs)+[vec_func]
#           else:
#               self.scope.sons_scopes['sc_'+ name] = self.scope.sons_scopes[name]
#               funcs = funcs.rename('sc_'+ name)
#               funcs = [funcs, vec_func]
#           funcs = Interface(name, funcs)
#           self.insert_function(funcs)
        return EmptyNode()

    def _visit_PythonPrint(self, expr):
        args = [self._visit(i) for i in expr.expr]
        for i, arg in enumerate(args):
            rhs = arg.value
            if getattr(rhs, 'rank', 0) and isinstance(rhs, PyccelInternalFunction):
                tmp_var = self._assign_lhs_variable(self.scope.get_new_name(), self._infer_type(rhs) , rhs, self._additional_exprs[-1] , is_augassign=False)
                self._additional_exprs[-1].append(Assign(tmp_var, rhs, ast=rhs.ast))
                args[i] = FunctionCallArgument(tmp_var)
        if len(args) == 0:
            return PythonPrint(args)

        def is_symbolic(var):
            return isinstance(var, Variable) \
                and isinstance(var.dtype, NativeSymbol)

        # TODO fix: not yet working because of mpi examples
#        if not test:
#            # TODO: Add description to parser/messages.py
#            errors.report('Either all arguments must be symbolic or none of them can be',
#                   bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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

        name = expr.name

        #  create a new Datatype for the current class
        dtype = DataTypeFactory(name, '_name')
        self.scope.cls_constructs[name] = dtype

        parent = self._find_superclasses(expr)

        scope = self.create_new_class_scope(name, used_symbols=expr.scope.local_used_symbols,
                    original_symbols = expr.scope.python_names.copy())

        cls = ClassDef(name, [], [], superclasses=parent, scope=scope)
        self.scope.parent_scope.insert_class(cls)

        methods = list(expr.methods)
        init_func = None

        for (i, method) in enumerate(methods):
            m_name = method.name
            if m_name == '__init__':
                self._visit_FunctionDef(method)
                methods.pop(i)
                init_func = self.scope.functions.pop(m_name)
                break

        if not init_func:
            errors.report(UNDEFINED_INIT_METHOD, symbol=name,
                   bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                   severity='error')

        for i in methods:
            self._visit_FunctionDef(i)

        self.exit_class_scope()

        return EmptyNode()

    def _visit_Del(self, expr):

        ls = [self._visit(i) for i in expr.variables]
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
                        bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                        severity='error')
            return IsClass(var1, expr.rhs)

        if (var1.dtype != var2.dtype):
            if IsClass == PyccelIs:
                return LiteralFalse()
            elif IsClass == PyccelIsNot:
                return LiteralTrue()

        if (isinstance(var1.dtype, NativeBool) and
            isinstance(var2.dtype, NativeBool)):
            return IsClass(var1, var2)

        lst = [NativeString(), NativeComplex(), NativeFloat(), NativeInteger()]
        if (var1.dtype in lst):
            errors.report(PYCCEL_RESTRICTION_PRIMITIVE_IMMUTABLE, symbol=expr,
            bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
            severity='error')
            return IsClass(var1, var2)

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
            bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
            severity='error')
        return IsClass(var1, var2)

    def _visit_Import(self, expr):

        # TODO - must have a dict where to store things that have been
        #        imported
        #      - should not use scope

        container = self.scope.imports

        result = EmptyNode()

        if isinstance(expr.source, AsName):
            source        = expr.source.name
            source_target = expr.source.target
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
                    errors.report(IMPORTING_EXISTING_IDENTIFIED,
                                  bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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
                targets = {i.target if isinstance(i,AsName) else i:None for i in expr.target}
                names = [i.name if isinstance(i,AsName) else i for i in expr.target]
                for entry in ['variables', 'classes', 'functions']:
                    d_son = getattr(p.scope, entry)
                    for t,n in zip(targets.keys(),names):
                        if n in d_son:
                            e = d_son[n]
                            if t == n:
                                container[entry][t] = e
                            else:
                                container[entry][t] = e.clone(t)
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
                targets = list(container['imports'][source_target].target.union(targets))

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
            errors.report(UNDEFINED_WITH_ACCESS,
                   bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
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
                bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset))
        else:
            interfaces = []
            for hd in header:
                interfaces += hd.create_definition()

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
                          bounding_box=(self._current_ast_node.lineno, self._current_ast_node.col_offset),
                          severity='fatal')
        header = self.get_headers(master)
        if header is None:
            var = self.get_variable(master)
        else:
            var = Variable(header.dtype, header.name)

                # TODO -> Said: must handle interface

        expr = MacroVariable(expr.name, var)
        self.scope.insert_macro(expr)
        return expr

    def _visit_StarredArguments(self, expr):
        var = self._visit(expr.args_var)
        assert(var.rank==1)
        size = var.shape[0]
        return StarredArguments([var[i] for i in range(size)])

    def _visit_NumpyMatmul(self, expr):
        if isinstance(expr, FunctionCall):
            a = self._visit(expr.args[0].value)
            b = self._visit(expr.args[1].value)
        else:
            self.insert_import('numpy', AsName(NumpyMatmul, 'matmul'))
            a = self._visit(expr.a)
            b = self._visit(expr.b)
        return NumpyMatmul(a, b)

    def _visit_Assert(self, expr):
        test = self._visit(expr.test)
        return Assert(test)

    def _visit_NumpyWhere(self, func_call):
        func_call_args = self._handle_function_args(func_call.args)
        # expr is a FunctionCall
        args = [a.value for a in func_call_args if not a.has_keyword]
        kwargs = {a.keyword: a.value for a in func_call.args if a.has_keyword}
        nargs = len(args)+len(kwargs)
        if nargs == 1:
            return self._visit_NumpyNonZero(func_call)
        return NumpyWhere(*args, **kwargs)

    def _visit_NumpyNonZero(self, func_call):
        func_call_args = self._handle_function_args(func_call.args)
        # expr is a FunctionCall
        arg = func_call_args[0].value
        if not isinstance(arg, Variable):
            new_symbol = PyccelSymbol(self.scope.get_new_name())
            creation = self._visit(Assign(new_symbol, arg, ast=func_call.ast))
            self._additional_exprs[-1].append(creation)
            arg = self._visit(new_symbol)
        return NumpyWhere(arg)

    def _visit_FunctionDefResult(self, expr):
        var = self._visit(expr.var)
        return FunctionDefResult(var, annotation = expr.annotation)
