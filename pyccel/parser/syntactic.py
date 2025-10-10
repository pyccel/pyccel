# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
import ast
import inspect
from itertools import chain
import os
import re
import types
import warnings

from textx.exceptions import TextXSyntaxError

#==============================================================================

from pyccel.ast.basic import PyccelAstNode

from pyccel.ast.core import FunctionCall, FunctionCallArgument
from pyccel.ast.core import Module, Program
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Return
from pyccel.ast.core import Pass
from pyccel.ast.core import FunctionDef, InlineFunctionDef
from pyccel.ast.core import ClassDef
from pyccel.ast.core import For
from pyccel.ast.core import If, IfSection
from pyccel.ast.core import While
from pyccel.ast.core import Del
from pyccel.ast.core import Assert
from pyccel.ast.core import Comment, EmptyNode
from pyccel.ast.core import Break, Continue
from pyccel.ast.core import FunctionDefArgument
from pyccel.ast.core import FunctionDefResult
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import CommentBlock
from pyccel.ast.core import With
from pyccel.ast.core import StarredArguments
from pyccel.ast.core import CodeBlock
from pyccel.ast.core import IndexedElement

from pyccel.ast.datatypes import TypeAlias

from pyccel.ast.bitwise_operators import PyccelRShift, PyccelLShift, PyccelBitXor, PyccelBitOr, PyccelBitAnd, PyccelInvert
from pyccel.ast.operators import PyccelPow, PyccelAdd, PyccelMul, PyccelDiv, PyccelMod, PyccelFloorDiv
from pyccel.ast.operators import PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe
from pyccel.ast.operators import PyccelAnd, PyccelOr,  PyccelNot, PyccelMinus
from pyccel.ast.operators import PyccelUnary, PyccelUnarySub
from pyccel.ast.operators import PyccelIs, PyccelIsNot, PyccelIn
from pyccel.ast.operators import IfTernaryOperator
from pyccel.ast.numpyext  import NumpyMatmul

from pyccel.ast.builtins import PythonTuple, PythonList, PythonSet, PythonDict
from pyccel.ast.builtins import PythonPrint
from pyccel.ast.headers  import MetaVariable
from pyccel.ast.literals import LiteralInteger, LiteralFloat, LiteralComplex
from pyccel.ast.literals import LiteralFalse, LiteralTrue, LiteralString
from pyccel.ast.literals import Nil, LiteralEllipsis
from pyccel.ast.functionalexpr import FunctionalSum, FunctionalMax, FunctionalMin, GeneratorComprehension, FunctionalFor
from pyccel.ast.variable  import DottedName, AnnotatedPyccelSymbol

from pyccel.ast.internals import Slice, PyccelSymbol, PyccelFunction

from pyccel.ast.type_annotations import SyntacticTypeAnnotation, UnionTypeAnnotation, VariableTypeAnnotation

from pyccel.parser.base        import BasicParser
from pyccel.parser.extend_tree import extend_tree
from pyccel.parser.scope       import Scope
from pyccel.parser.utilities   import get_default_path

from pyccel.parser.syntax.headers import parse as hdr_parse, types_meta
from pyccel.parser.syntax.openmp  import parse as omp_parse
from pyccel.parser.syntax.openacc import parse as acc_parse

from pyccel.utilities.stage import PyccelStage

from pyccel.errors.errors import Errors, ErrorsMode, PyccelError

# TODO - remove import * and only import what we need
from pyccel.errors.messages import *

def get_name(a):
    """ get the name of variable or an argument of the AST node."""
    if isinstance(a, ast.Name):
        return a.id
    elif isinstance(a, ast.arg):
        return a.arg
    elif isinstance(a, ast.FunctionDef):
        return a.name
    else:
        raise NotImplementedError()

#==============================================================================
errors = Errors()
pyccel_stage = PyccelStage()
#==============================================================================

strip_ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]|[\n\t\r]')

# use this to delete ansi_escape characters from a string
# Useful for very coarse version differentiation.

#==============================================================================

class SyntaxParser(BasicParser):
    """
    Class which handles the syntactic stage as described in the developer docs.

    This class is described in detail in developer_docs/syntactic_stage.md.
    It extracts all necessary information from the Python AST in order to create
    a representation complete enough for the semantic stage to determine types, etc
    as described in developer_docs/semantic_stage.md.

    Parameters
    ----------
    inputs : str
        A string containing code or containing the name of a file whose code
        should be read.

    context_dict : dict, optional
        A dictionary describing any variables in the context where the translated
        objected was defined.

    **kwargs : dict
        Additional keyword arguments for BasicParser.
    """

    def __init__(self, inputs, *, context_dict = None, **kwargs):
        BasicParser.__init__(self, **kwargs)

        # check if inputs is a file
        code = inputs
        if os.path.isfile(inputs):

            self._filename = inputs
            errors.set_target(self.filename)

            # we don't use is_valid_filename_py since it uses absolute path
            # file extension

            with open(inputs, 'r', encoding="utf-8") as file:
                code = file.read()

            self._scope = Scope(name = inputs.stem, scope_type = 'module')
        else:
            self._scope = Scope('', scope_type = 'module')

        self._code    = code
        self._context = []

        # provides information about the calling context to collect constants and functions
        self._context_dict = context_dict or {}

        tree                = extend_tree(code)
        self._fst           = tree
        self._in_lhs_assign = False

        self.parse()

    def parse(self):
        """
        Convert Python's AST to Pyccel's AST object.

        Convert Python's AST to Pyccel's AST object and raise errors
        for any unsupported objects.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The Pyccel AST object.
        """

        if self.syntax_done:
            return self.ast

        # TODO - add settings to Errors
        #      - filename

        pyccel_stage.set_stage('syntactic')
        ast       = self._visit(self.fst)
        self._ast = ast

        self._syntax_done = True

        return ast

    def create_new_function_scope(self, name, **kwargs):
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
        name : str
            Function's name, used as a key to retrieve the new scope.

        **kwargs : dict
            Keyword arguments passed through to the new scope.

        Returns
        -------
        Scope
            The new scope for the function.
        """
        child = self.scope.new_child_scope(name, 'function', **kwargs)

        self._scope = child
        self._current_function_name.append(name)

        return child

    def _treat_iterable(self, stmt):
        return (self._visit(i) for i in stmt)

    def _treat_comment_line(self, line, stmt):
        """
        Parse a comment line.

        Parse a comment which fits in a single line. If the comment
        begins with `#$` then it should contain a header recognised
        by Pyccel and should be parsed using textx.

        Parameters
        ----------
        line : str
            The comment line.
        stmt : ast.Ast
            The comment object in the code. This is useful for raising
            neat errors.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The treated object as a Pyccel ast node.
        """
        if line.startswith('#$'):
            env = line[2:].lstrip()
            if env.startswith('omp'):
                expr = omp_parse(stmts=line)
                try:
                    expr = omp_parse(stmts=line)
                except TextXSyntaxError as e:
                    errors.report(f"Invalid OpenMP header. {e.message}",
                            symbol = stmt, column = e.col,
                              severity='fatal')
            elif env.startswith('acc'):
                try:
                    expr = acc_parse(stmts=line)
                except TextXSyntaxError as e:
                    errors.report(f"Invalid OpenACC header. {e.message}",
                            symbol = stmt, column = e.col,
                              severity='fatal')
            elif env.startswith('header'):
                try:
                    expr = hdr_parse(stmts=line)
                except TextXSyntaxError as e:
                    errors.report(f"Invalid header. {e.message}",
                            symbol = stmt, column = e.col,
                              severity='fatal')
                if isinstance(expr, AnnotatedPyccelSymbol):
                    self.scope.insert_symbol(expr.name)
                elif isinstance(expr, MetaVariable):
                    # a metavar will not appear in the semantic stage.
                    # but can be used to modify the ast

                    self._metavars[str(expr.name)] = expr.value
                    expr = EmptyNode()
            else:
                errors.report(PYCCEL_INVALID_HEADER,
                              symbol = stmt,
                              severity='fatal')

        else:
            txt = line[1:].lstrip()
            expr = Comment(txt)

        expr.set_current_ast(stmt)
        return expr

    def _treat_type_annotation(self, stmt, annotation):
        """
        Treat an object passed as a type annotation.

        Ensure that an object that was passed as a type annotation can be
        recognised in the semantic stage by packing it into a SyntacticTypeAnnotation
        in the correct way. Also check the syntax of any string type
        annotations.

        Parameters
        ----------
        stmt : ast.Ast
            The ast node about which any errors should be raised.

        annotation : pyccel.ast.basic.PyccelAstNode
            A visited object which is describing a type annotation.

        Returns
        -------
        SyntacticTypeAnnotation | UnionTypeAnnotation
            The type annotation.
        """
        if isinstance(annotation, (tuple, list)):
            return UnionTypeAnnotation(*[self._treat_type_annotation(stmt, a) for a in annotation])
        if isinstance(annotation, (PyccelSymbol, DottedName, IndexedElement)):
            return SyntacticTypeAnnotation(dtype=annotation)
        elif isinstance(annotation, LiteralString):
            try:
                annotation = types_meta.model_from_str(annotation.python_value)
            except TextXSyntaxError as e:
                errors.report(f"Invalid header. {e.message}",
                        symbol = stmt, column = e.col,
                        severity='fatal')
            annot = annotation.expr
            if isinstance(stmt, PyccelAstNode):
                annot.set_current_ast(stmt.python_ast)
            else:
                annot.set_current_ast(stmt)
            return annot
        elif annotation is Nil():
            return None
        elif isinstance(annotation, PyccelBitOr):
            return UnionTypeAnnotation(*[self._treat_type_annotation(stmt, a) for a in annotation.args])
        else:
            errors.report('Invalid type annotation',
                        symbol = stmt, severity='error')
            return EmptyNode()

    def _get_unique_name(self, possible_names, valid_names, forbidden_names, suggestion):
        """
        Get a name for a variable amongst multiple possibilities.

        Get a name for a variable. If the name is the only possibility it is
        chosen. Otherwise a new name is constructed from the suggestions.
        This function is usually used to find the name(s) of the result of a
        FunctionDef.

        Parameters
        ----------
        possible_names : iterable[PyccelSymbol | TypedAstNode]
            The possible names found for the variable.
        valid_names : iterable[PyccelSymbol]
            The names found in the scope that can be used for this variable (this is
            important to avoid accidentally using an imported variable).
        forbidden_names : iterable[PyccelSymbol]
            The names that should not be used.
        suggestion : PyccelSymbol
            The name that may be used instead.

        Returns
        -------
        str
            The new name of the variable.
        """
        if all(isinstance(n, PythonTuple) for n in possible_names) and \
                len(set(len(n) for n in possible_names)) == 1:
            # If all possible names are iterables of the same length then find a name
            # for each element and link them to an element describing this variable
            # via IndexedElement (these are tuple elements).
            possible_names = list(zip(*[n.args for n in possible_names]))
            unique_names = [self._get_unique_name(n, valid_names, forbidden_names, f'{suggestion}_{i}') \
                            for i,n in enumerate(possible_names)]
            temp_name = self.scope.get_new_name(suggestion, is_temp = True)
            for i, n in enumerate(unique_names):
                self.scope.insert_symbolic_alias(IndexedElement(temp_name, i), n)
            return temp_name
        else:
            suggested_names = set(n for n in possible_names if isinstance(n, PyccelSymbol))
            if len(suggested_names) == 1:
                # If one name is suggested then return it unless it is forbidden
                new_name = suggested_names.pop()
                if new_name not in forbidden_names and new_name in valid_names:
                    return new_name
            return self.scope.get_new_name(suggestion, is_temp = True)

    def insert_import(self, expr):
        """
        Insert an import into the scope.

        Insert an import into the scope along with the targets that are
        needed.

        Parameters
        ----------
        expr : Import
            The import to be inserted.
        """

        assert isinstance(expr, Import)
        container = self.scope.imports['imports']

        # if source is not specified, imported things are treated as sources
        if len(expr.target) == 0:
            if isinstance(expr.source, AsName):
                name   = expr.source
            else:
                name   = str(expr.source)

            container[name] = []
        else:
            source = str(expr.source)
            if not source in container.keys():
                container[source] = []
            container[source] += expr.target

    #====================================================
    #                 _visit functions
    #====================================================

    def _visit(self, stmt):
        """
        Build the AST from the AST parsed by Python.

        The annotation is done by finding the appropriate function _visit_X
        for the object expr. X is the type of the object expr. If this function
        does not exist then the method resolution order is used to search for
        other compatible _visit_X functions. If none are found then an error is
        raised.

        Parameters
        ----------
        stmt : ast.stmt
            Object to visit of type X.

        Returns
        -------
        pyccel.ast.basic.Basic
            AST object which is the syntactic equivalent of stmt.
        """

        # TODO - add settings to Errors
        #      - line and column
        #      - blocking errors

        cls = type(stmt)
        syntax_method = '_visit_' + cls.__name__
        if hasattr(self, syntax_method):
            if self._verbose > 2:
                print(f">>>> Calling SyntaxParser.{syntax_method}")
            self._context.append(stmt)
            try:
                result = getattr(self, syntax_method)(stmt)
                if isinstance(result, PyccelAstNode) and result.python_ast is None and isinstance(stmt, ast.AST):
                    result.set_current_ast(stmt)
            except PyccelError as err:
                raise err
            except NotImplementedError as error:
                errors.report(f'{error}\n'+PYCCEL_RESTRICTION_TODO,
                    symbol = self._current_ast_node, severity='fatal',
                    traceback=error.__traceback__)
            except Exception as err: #pylint: disable=broad-exception-caught
                if ErrorsMode().value == 'user':
                    errors.report(PYCCEL_INTERNAL_ERROR,
                            symbol = self._context[-1], severity='fatal')
                else:
                    raise err
            self._context.pop()
            return result

        # Unknown object, we raise an error.
        return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX, symbol=stmt,
                      severity='error')

    def _visit_Module(self, stmt):
        """ Visits the ast and splits the result into elements relevant for the module or the program"""

        # Collect functions and classes. These must be visited last to ensure
        # all names have been collected from the parent scope
        ast_functions = [f for f in stmt.body if isinstance(f, ast.FunctionDef)]
        ast_classes = [c for c in stmt.body if isinstance(c, ast.ClassDef)]

        # Add the names of the functions and classes to the scope
        for obj in chain(ast_functions, ast_classes):
            self.scope.insert_symbol(PyccelSymbol(obj.name))

        body = [self._visit(v) for v in stmt.body
                if not isinstance(v, (ast.FunctionDef, ast.ClassDef))]

        functions = [self._visit(f) for f in ast_functions]
        classes   = [self._visit(c) for c in ast_classes]
        imports   = [i for i in body if isinstance(i, Import)]
        programs  = [p for p in body if isinstance(p, Program)]
        body      = [l for l in body if not isinstance(l, (FunctionDef, ClassDef, Import, Program))]

        if len(programs) > 1:
            errors.report("Multiple program blocks are not supported. Please group the code for the main.",
                    symbol = programs[1],
                    severity = 'error')
        program = next(iter(programs), None)

        # Define the name of the module
        # The module name allows it to be correctly referenced from an import command
        mod_name = os.path.splitext(os.path.basename(self._filename))[0]
        name = self.scope.get_new_name(mod_name, object_type = 'module')
        self.scope.python_names[name] = mod_name

        body = [b for i in body for b in (i.body if isinstance(i, CodeBlock) else [i])]
        return Module(name, [], functions, init_func = CodeBlock(body), scope = self.scope,
                classes = classes, imports = imports, program = program)

    def _visit_Expr(self, stmt):
        val = self._visit(stmt.value)
        if not isinstance(val, (CommentBlock, PythonPrint, LiteralEllipsis)):
            # Collect any results of standalone expressions
            # into a variable to avoid errors in C/Fortran
            val = Assign(PyccelSymbol('_', is_temp=True), val)
        return val

    def _visit_Tuple(self, stmt):
        return PythonTuple(*self._treat_iterable(stmt.elts))

    def _visit_List(self, stmt):
        return PythonList(*self._treat_iterable(stmt.elts))

    def _visit_Set(self, stmt):
        return PythonSet(*self._treat_iterable(stmt.elts))

    def _visit_tuple(self, stmt):
        return tuple(self._treat_iterable(stmt))

    def _visit_list(self, stmt):
        return list(self._treat_iterable(stmt))

    def _visit_alias(self, stmt):
        if not isinstance(stmt.name, str):
            raise TypeError('Expecting a string')

        old = self._visit(stmt.name)

        if stmt.asname:
            new = self._visit(stmt.asname)
            return AsName(old, new)
        else:
            return old

    def _visit_Dict(self, stmt):
        return PythonDict(self._visit(stmt.keys), self._visit(stmt.values))

    def _visit_NoneType(self, stmt):
        return Nil()

    def _visit_str(self, stmt):

        return stmt

    def _visit_Assign(self, stmt):

        rhs = self._visit(stmt.value)
        if isinstance(rhs, FunctionDef):
            return rhs

        self._in_lhs_assign = True
        lhs = self._visit(stmt.targets)
        self._in_lhs_assign = False
        if len(lhs)==1:
            lhs = lhs[0]
        else:
            lhs = PythonTuple(*lhs)

        expr = Assign(lhs, rhs)

        return expr

    def _visit_AugAssign(self, stmt):

        lhs = self._visit(stmt.target)
        rhs = self._visit(stmt.value)
        if isinstance(stmt.op, ast.Add):
            return AugAssign(lhs, '+', rhs)
        elif isinstance(stmt.op, ast.Sub):
            return AugAssign(lhs, '-', rhs)
        elif isinstance(stmt.op, ast.Mult):
            return AugAssign(lhs, '*', rhs)
        elif isinstance(stmt.op, ast.Div):
            return AugAssign(lhs, '/', rhs)
        elif isinstance(stmt.op, ast.Mod):
            return AugAssign(lhs, '%', rhs)
        elif isinstance(stmt.op, ast.BitOr):
            return AugAssign(lhs, '|', rhs)
        elif isinstance(stmt.op, ast.BitAnd):
            return AugAssign(lhs, '&', rhs)
        elif isinstance(stmt.op, ast.LShift):
            return AugAssign(lhs, '<<', rhs)
        elif isinstance(stmt.op, ast.RShift):
            return AugAssign(lhs, '>>', rhs)
        elif isinstance(stmt.op, ast.FloorDiv):
            return AugAssign(lhs, '//', rhs)
        else:
            return errors.report(PYCCEL_RESTRICTION_TODO, symbol = stmt,
                    severity='error')

    def _visit_AnnAssign(self, stmt):
        self._in_lhs_assign = True
        lhs = self._visit(stmt.target)
        self._in_lhs_assign = False

        annotation = self._treat_type_annotation(stmt, self._visit(stmt.annotation))

        annotated_lhs = AnnotatedPyccelSymbol(lhs, annotation=annotation)

        if stmt.value is None:
            return annotated_lhs
        else:
            rhs = self._visit(stmt.value)
            return Assign(annotated_lhs, rhs)

    def _visit_arguments(self, stmt):
        is_class_method = len(self._context) > 2 and isinstance(self._context[-3], ast.ClassDef)

        arguments       = []
        if stmt.posonlyargs:
            for a in stmt.posonlyargs:
                annotation=self._treat_type_annotation(a, self._visit(a.annotation))
                new_arg = FunctionDefArgument(AnnotatedPyccelSymbol(a.arg, annotation),
                                            annotation=annotation, posonly = True)
                new_arg.set_current_ast(a)
                arguments.append(new_arg)

        if stmt.args:
            n_expl = len(stmt.args)-len(stmt.defaults)

            for a in stmt.args[:n_expl]:
                annotation=self._treat_type_annotation(a, self._visit(a.annotation))
                new_arg = FunctionDefArgument(AnnotatedPyccelSymbol(a.arg, annotation),
                                            annotation=annotation)
                new_arg.set_current_ast(a)
                arguments.append(new_arg)

            for a,d in zip(stmt.args[n_expl:], stmt.defaults):
                annotation=self._treat_type_annotation(a, self._visit(a.annotation))
                new_arg = FunctionDefArgument(AnnotatedPyccelSymbol(a.arg, annotation),
                                            annotation=annotation,
                                            value = self._visit(d))
                new_arg.set_current_ast(a)
                arguments.append(new_arg)

        if stmt.vararg:
            annotation = self._treat_type_annotation(stmt.vararg, self._visit(stmt.vararg.annotation))
            tuple_annotation = IndexedElement(PyccelSymbol('tuple'), annotation, LiteralEllipsis())
            new_arg = FunctionDefArgument(AnnotatedPyccelSymbol(stmt.vararg.arg, tuple_annotation),
                                        annotation=annotation, is_vararg = True)
            new_arg.set_current_ast(stmt.vararg)
            arguments.append(new_arg)

        if is_class_method:
            expected_self_arg = arguments[0]
            if expected_self_arg.annotation is None:
                class_name = self._context[-3].name
                annotation = self._treat_type_annotation(class_name, PyccelSymbol(class_name))
                arguments[0] = FunctionDefArgument(AnnotatedPyccelSymbol(expected_self_arg.name, annotation),
                                            annotation=annotation,
                                            value = expected_self_arg.value)

        if stmt.kwonlyargs:
            for a,d in zip(stmt.kwonlyargs,stmt.kw_defaults):
                annotation=self._treat_type_annotation(a, self._visit(a.annotation))
                val = self._visit(d) if d is not None else d
                arg = FunctionDefArgument(AnnotatedPyccelSymbol(a.arg, annotation),
                            annotation=annotation,
                            value=val, kwonly=True)
                arg.set_current_ast(a)

                arguments.append(arg)

        if stmt.kwarg:
            annotation = self._treat_type_annotation(stmt.kwarg, self._visit(stmt.kwarg.annotation))
            dict_annotation = IndexedElement(PyccelSymbol('dict'), PyccelSymbol('str'), annotation)
            new_arg = FunctionDefArgument(AnnotatedPyccelSymbol(stmt.kwarg.arg, dict_annotation),
                                        annotation=annotation, is_kwarg = True)
            new_arg.set_current_ast(stmt.kwarg)
            arguments.append(new_arg)

        self.scope.insert_symbols(a.var for a in arguments)

        return arguments

    def _visit_Constant(self, stmt):
        # New in python3.8 this class contains NameConstant, Num, and String types
        if stmt.value is None:
            return Nil()

        elif stmt.value is True:
            return LiteralTrue()

        elif stmt.value is False:
            return LiteralFalse()

        elif isinstance(stmt.value, int):
            return LiteralInteger(stmt.value)

        elif isinstance(stmt.value, float):
            return LiteralFloat(stmt.value)

        elif isinstance(stmt.value, complex):
            return LiteralComplex(stmt.value.real, stmt.value.imag)

        elif isinstance(stmt.value, str):
            if isinstance(self._context[-2], ast.Expr):
                return CommentBlock(stmt.value)
            return LiteralString(stmt.value)

        elif stmt.value is Ellipsis:
            return LiteralEllipsis()

        else:
            raise NotImplementedError(f'Constant type {type(stmt.value)} not recognised')

    def _visit_Name(self, stmt):
        name = PyccelSymbol(stmt.id)
        if self._in_lhs_assign:
            self.scope.insert_symbol(name)
        return name

    def _treat_import_source(self, source, level):
        source = '.'*level + source
        if source.count('.') == 0:
            source = PyccelSymbol(source)
            self.scope.insert_symbol(source)
        else:
            source = DottedName(*source.split('.'))

        return get_default_path(source)

    def _visit_Import(self, stmt):
        expr = []
        for name in stmt.names:
            imp = self._visit(name)
            if isinstance(imp, AsName):
                source = AsName(self._treat_import_source(imp.object, 0), imp.local_alias)
                self.scope.insert_symbol(imp.local_alias)
            else:
                source = self._treat_import_source(imp, 0)
                self.scope.insert_symbol(imp)
            import_line = Import(source)
            import_line.set_current_ast(stmt)
            self.insert_import(import_line)
            expr.append(import_line)

        if len(expr)==1:
            return expr[0]
        else:
            expr = CodeBlock(expr)
            return expr

    def _visit_ImportFrom(self, stmt):

        source = self._treat_import_source(stmt.module, stmt.level)

        targets = []
        for i in stmt.names:
            s = self._visit(i)
            if s == '*':
                errors.report(PYCCEL_RESTRICTION_IMPORT_STAR,
                              bounding_box=(stmt.lineno, stmt.col_offset),
                              severity='error')

            targets.append(s)
            if isinstance(s, AsName):
                self.scope.insert_symbol(s.local_alias)
            else:
                self.scope.insert_symbol(s)

        expr = Import(source, targets)
        self.insert_import(expr)
        return expr

    def _visit_Delete(self, stmt):
        arg = self._visit(stmt.targets)
        return Del(arg)

    def _visit_UnaryOp(self, stmt):

        target = self._visit(stmt.operand)

        if isinstance(stmt.op, ast.Not):
            Func = PyccelNot

        elif isinstance(stmt.op, ast.UAdd):
            Func = PyccelUnary

        elif isinstance(stmt.op, ast.USub):
            Func = PyccelUnarySub

        elif isinstance(stmt.op, ast.Invert):
            Func = PyccelInvert
        else:
            errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                          symbol = stmt,
                          severity='error')

        return Func(target)

    def _visit_BinOp(self, stmt):

        first  = self._visit(stmt.left)
        second = self._visit(stmt.right)

        if isinstance(stmt.op, ast.Add):
            return PyccelAdd(first, second)

        elif isinstance(stmt.op, ast.Mult):
            return PyccelMul(first, second)

        elif isinstance(stmt.op, ast.Sub):
            return PyccelMinus(first, second)

        elif isinstance(stmt.op, ast.Div):
            return PyccelDiv(first, second)

        elif isinstance(stmt.op, ast.Pow):
            return PyccelPow(first, second)

        elif isinstance(stmt.op, ast.FloorDiv):
            return PyccelFloorDiv(first, second)

        elif isinstance(stmt.op, ast.Mod):
            return PyccelMod(first, second)

        elif isinstance(stmt.op, ast.RShift):
            return PyccelRShift(first, second)

        elif isinstance(stmt.op, ast.LShift):
            return PyccelLShift(first, second)

        elif isinstance(stmt.op, ast.BitXor):
            return PyccelBitXor(first, second)

        elif isinstance(stmt.op, ast.BitOr):
            return PyccelBitOr(first, second)

        elif isinstance(stmt.op, ast.BitAnd):
            return PyccelBitAnd(first, second)

        elif isinstance(stmt.op, ast.MatMult):
            return NumpyMatmul(first, second)

        else:
            return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                          symbol = stmt,
                          severity='error')

    def _visit_BoolOp(self, stmt):

        args = [self._visit(a) for a in stmt.values]

        if isinstance(stmt.op, ast.And):
            return PyccelAnd(*args)

        if isinstance(stmt.op, ast.Or):
            return PyccelOr(*args)

        return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                      symbol = stmt.op,
                      severity='error')

    def _visit_Compare(self, stmt):
        first = self._visit(stmt.left)
        comparison = None
        for comparators, op in zip(stmt.comparators, stmt.ops):
            second = self._visit(comparators)

            if isinstance(op, ast.Eq):
                expr = PyccelEq(first, second)
            elif isinstance(op, ast.NotEq):
                expr = PyccelNe(first, second)
            elif isinstance(op, ast.Lt):
                expr = PyccelLt(first, second)
            elif isinstance(op, ast.Gt):
                expr = PyccelGt(first, second)
            elif isinstance(op, ast.LtE):
                expr = PyccelLe(first, second)
            elif isinstance(op, ast.GtE):
                expr = PyccelGe(first, second)
            elif isinstance(op, ast.Is):
                expr = PyccelIs(first, second)
            elif isinstance(op, ast.IsNot):
                expr = PyccelIsNot(first, second)
            elif isinstance(op, ast.In):
                expr = PyccelIn(first, second)
            else:
                return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                              symbol = stmt,
                              severity='error')

            first = second
            comparison = PyccelAnd(comparison, expr) if comparison else expr

        return comparison

    def _visit_Return(self, stmt):
        results = self._visit(stmt.value)
        return Return(results)

    def _visit_Pass(self, stmt):
        return Pass()

    def _visit_FunctionDef(self, stmt):

        #  TODO check all inputs and which ones should be treated in stage 1 or 2

        name = PyccelSymbol(stmt.name)

        if not isinstance(self._context[-1], ast.Module):
            self.scope.insert_symbol(name, 'function')

        new_name = self.scope.get_expected_name(name)

        scope = self.create_new_function_scope(name,
                used_symbols = {name: new_name},
                original_symbols = {new_name: name})

        arguments    = self._visit(stmt.args)

        is_pure      = False
        is_elemental = False
        is_private   = False
        is_inline    = False
        docstring   = None

        decorators = {}

        for d in self._visit(stmt.decorator_list):
            tmp_var = d if isinstance(d, PyccelSymbol) else d.funcdef
            if tmp_var in decorators:
                decorators[tmp_var] += [d]
            else:
                decorators[tmp_var] = [d]

        if 'types' in decorators:
            warnings.warn("The @types decorator will be removed in version 2.0 of Pyccel. " +
                  "Please use type hints. TypeVar from Python's typing module can " +
                  "be used to specify multiple types. See the documentation at " +
                  "https://github.com/pyccel/pyccel/blob/devel/docs/quickstart.md#type-annotations"
                  "for examples.", FutureWarning)

        if 'stack_array' in decorators:
            decorators['stack_array'] = tuple(str(b.value) for a in decorators['stack_array']
                for b in a.args)

        if 'allow_negative_index' in decorators:
            decorators['allow_negative_index'] = tuple(str(b.value) for a in decorators['allow_negative_index'] for b in a.args)

        if 'pure' in decorators:
            is_pure = True

        if 'elemental' in decorators:
            is_elemental = True
            if len(arguments) > 1:
                errors.report(FORTRAN_ELEMENTAL_SINGLE_ARGUMENT,
                              symbol=decorators['elemental'],
                              bounding_box=(stmt.lineno, stmt.col_offset),
                              severity='error')

        if 'private' in decorators:
            is_private = True

        if 'inline' in decorators:
            is_inline = True


        result_annotation = self._treat_type_annotation(stmt, self._visit(stmt.returns))

        argument_names = {a.var.name for a in arguments}

        body = stmt.body
        body = self._visit(body)

        # Collect docstring
        if len(body) > 0 and isinstance(body[0], CommentBlock):
            docstring = body[0]
            docstring.header = ''
            body = body[1:]

        functions = [f for f in body if isinstance(f, FunctionDef)]
        classes   = [c for c in body if isinstance(c, ClassDef)]
        imports   = [i for i in body if isinstance(i, Import)]
        body      = [l for l in body if not isinstance(l, (FunctionDef, ClassDef, Import))]

        if classes:
            errors.report("Classes in functions are not supported.",
                    symbol=classes[0], severity='error')

        body = CodeBlock(body)

        targets = [t for target_list in self.scope.imports['imports'].values() for t in target_list]

        returns = body.get_attribute_nodes(Return,
                    excluded_nodes = (Assign, FunctionCall, PyccelFunction, FunctionDef))
        if len(returns) == 0 or all(r.expr is Nil() for r in returns):
            if result_annotation:
                results = self.scope.get_new_name('result', is_temp = True)
                results = AnnotatedPyccelSymbol(results, annotation = result_annotation)
                results = FunctionDefResult(results, annotation = result_annotation)
            else:
                results = FunctionDefResult(Nil())
        else:
            results = self._get_unique_name([r.expr for r in returns],
                                        valid_names = self.scope.local_used_symbols.keys(),
                                        forbidden_names = argument_names.union(targets),
                                        suggestion = 'result')

            if result_annotation:
                results = AnnotatedPyccelSymbol(results, annotation = result_annotation)

            results = FunctionDefResult(results, annotation = result_annotation)

        results.set_current_ast(stmt)

        self.exit_function_scope()

        cls = InlineFunctionDef if is_inline else FunctionDef
        func = cls(
               name,
               arguments,
               body,
               results,
               is_pure=is_pure,
               is_elemental=is_elemental,
               is_private=is_private,
               imports=imports,
               functions=functions,
               decorators=decorators,
               docstring=docstring,
               scope=scope)

        return func

    def _visit_ClassDef(self, stmt):

        name = stmt.name
        decorators = {}
        for d in self._visit(stmt.decorator_list):
            tmp_var = d if isinstance(d, PyccelSymbol) else d.funcdef
            decorators.setdefault(tmp_var, []).append(d)

        scope = self.create_new_class_scope(name)
        methods = []
        attributes = []
        docstring = None
        for i in stmt.body:
            visited_i = self._visit(i)
            if isinstance(visited_i, FunctionDef):
                methods.append(visited_i)
                visited_i.arguments[0].bound_argument = True
            elif isinstance(visited_i, (Pass, Comment)):
                continue
            elif isinstance(visited_i, CodeBlock) and all(isinstance(x, Comment) for x in visited_i.body):
                continue
            elif isinstance(visited_i, AnnotatedPyccelSymbol):
                attributes.append(visited_i)
            elif isinstance(visited_i, CommentBlock):
                docstring = visited_i
            else:
                errors.report(f"{type(visited_i)} not currently supported in classes",
                        severity='error', symbol=visited_i)
        parent = [p for p in (self._visit(i) for i in stmt.bases) if p != 'object']

        init_method = next((m for m in methods if m.name == '__init__'), None)
        if init_method is None:
            init_name = PyccelSymbol('__init__')
            semantic_init_name = self.scope.insert_symbol(init_name, 'function')
            annot = self._treat_type_annotation(stmt, LiteralString(name))
            init_scope = self.create_new_function_scope(init_name,
                    used_symbols = {init_name: semantic_init_name},
                    original_symbols = {semantic_init_name: init_name})
            self_arg = FunctionDefArgument(AnnotatedPyccelSymbol('self', annot),
                                           annotation=annot,
                                           kwonly=False,
                                           bound_argument = True)
            self_arg.set_current_ast(stmt)
            self.scope.insert_symbol(self_arg.var)
            self.exit_function_scope()
            methods.append(FunctionDef(init_name, (self_arg,), CodeBlock(()), FunctionDefResult(Nil()), scope=init_scope))

        self.exit_class_scope()

        expr = ClassDef(name=name, attributes=attributes,
                        methods=methods, superclasses=parent, scope=scope,
                        docstring = docstring, decorators=decorators)

        return expr

    def _visit_Subscript(self, stmt):

        args = self._visit(stmt.slice)
        if not isinstance(args, (PythonTuple, list)):
            args = (args,)
        var = self._visit(stmt.value)
        var = IndexedElement(var, *args)
        return var

    def _visit_ExtSlice(self, stmt):
        return self._visit(stmt.dims)

    def _visit_Slice(self, stmt):

        upper = self._visit(stmt.upper) if stmt.upper is not None else None
        lower = self._visit(stmt.lower) if stmt.lower is not None else None
        step = self._visit(stmt.step) if stmt.step is not None else None

        return Slice(lower, upper, step)

    def _visit_Index(self, stmt):
        return self._visit(stmt.value)

    def _visit_Attribute(self, stmt):
        val  = self._visit(stmt.value)
        attr = PyccelSymbol(stmt.attr)
        dotted = DottedName(val, attr)
        if self._in_lhs_assign:
            self.scope.insert_symbol(dotted)
        return dotted


    def _visit_Call(self, stmt):

        args = []
        if stmt.args:
            args += [FunctionCallArgument(self._visit(a), python_ast=a) for a in stmt.args]
        if stmt.keywords:
            kwargs = self._visit(stmt.keywords)
            for k, a in zip(kwargs, stmt.keywords):
                k.set_current_ast(a)

            args += kwargs

        if len(args) == 0:
            args = ()

        if len(args) == 1 and isinstance(args[0].value, (GeneratorComprehension, FunctionalFor)):
            return args[0].value

        func = self._visit(stmt.func)

        if isinstance(func, PyccelSymbol):
            if func == "print":
                func_call = PythonPrint(PythonTuple(*args))
            else:
                if func in self._context_dict and isinstance(self._context_dict[func], types.FunctionType) \
                        and not self.scope.symbol_in_use(func):
                    code_lines, _ = inspect.getsourcelines(self._context_dict[func])
                    indent_length = len(code_lines[0])-len(code_lines[0].lstrip())
                    fst = extend_tree(''.join(l[indent_length:] for l in code_lines))
                    assert len(fst.body) == 1
                    self._context_dict[func] = self._visit(fst.body[0])
                func_call = FunctionCall(func, args)
        elif isinstance(func, DottedName):
            func_attr = FunctionCall(func.name[-1], args)
            for n in func.name:
                if isinstance(n, PyccelAstNode):
                    n.clear_syntactic_user_nodes()
            func_call = DottedName(*func.name[:-1], func_attr)
        else:
            raise NotImplementedError(f' Unknown function type {type(func)}')

        return func_call

    def _visit_keyword(self, stmt):

        target = stmt.arg
        val = self._visit(stmt.value)
        return FunctionCallArgument(val, keyword=target)

    def _visit_For(self, stmt):

        scope = self.create_new_loop_scope()

        self._in_lhs_assign = True
        iterator = self._visit(stmt.target)
        self._in_lhs_assign = False
        iterable = self._visit(stmt.iter)
        body = self._visit(stmt.body)

        self.exit_loop_scope()

        expr = For(iterator, iterable, body, scope=scope)
        return expr

    def _visit_comprehension(self, stmt):

        scope = self.create_new_loop_scope()
        condition = None

        self._in_lhs_assign = True
        iterator = self._visit(stmt.target)
        self._in_lhs_assign = False
        iterable = self._visit(stmt.iter)

        if stmt.ifs:
            cond = PyccelAnd(*[self._visit(if_cond)
                                      for if_cond in stmt.ifs])
            condition = If(IfSection(cond, CodeBlock([])))

        self.exit_loop_scope()

        expr = (For(iterator, iterable, [], scope=scope), condition)
        return expr

    def _visit_ListComp(self, stmt):
        """
        Converts a list comprehension statement into a `FunctionalFor` AST object.

        This method translates the list comprehension into an equivalent `FunctionalFor`
        
        Parameters
        ----------
        stmt : ast.stmt
            Object to visit of type X.

        Returns
        -------
        pyccel.ast.functionalexpr.FunctionalFor
            AST object which is the syntactic equivalent of the list comprehension.
        """

        def create_target_operations():
            operations = {'list' : [], 'numpy_array' : []}
            index = PyccelSymbol('_', is_temp=True)
            args = [index]
            target = IndexedElement(lhs, *args)
            target = Assign(target, result)
            assign1 = Assign(index, LiteralInteger(0))
            assign1.set_current_ast(stmt)
            operations['numpy_array'].append(assign1)
            target.set_current_ast(stmt)
            operations['numpy_array'].append(target)
            assign2 = Assign(index, PyccelAdd(index, LiteralInteger(1)))
            assign2.set_current_ast(stmt)
            operations['numpy_array'].append(assign2)
            operations['list'].append(DottedName(lhs, FunctionCall('append', [FunctionCallArgument(result)])))

            return index, operations


        result = self._visit(stmt.elt)
        output_type = None
        comprehensions = list(self._visit(stmt.generators))
        generators = [c[0] for c in comprehensions]
        conditions = [c[1] for c in comprehensions]

        parent = self._context[-2]
        if isinstance(parent, ast.Call):
            output_type = self._visit(parent.func)

        success = isinstance(self._context[-2],ast.Assign)
        if not success and len(self._context) > 2:
            success = isinstance(self._context[-3],ast.Assign) and isinstance(self._context[-2],ast.Call)

        assignment = next((c for c in reversed(self._context) if isinstance(c, ast.Assign)), None)

        if not success:
            errors.report(PYCCEL_RESTRICTION_LIST_COMPREHENSION_ASSIGN,
                          symbol = stmt,
                          severity='error')
            lhs = PyccelSymbol('_', is_temp=True)
        else:
            lhs = self._visit(assignment.targets)
            if len(lhs)==1:
                lhs = lhs[0]
            else:
                raise NotImplementedError("A list comprehension cannot be unpacked")

        indices = [generator.target for generator in generators]
        index, operations = create_target_operations()

        return FunctionalFor(generators, result, lhs,indices,index=index,
                             conditions=conditions,
                             target_type=output_type, operations=operations)

    def _visit_GeneratorExp(self, stmt):
        conditions = []
        generators = []
        indices = []
        condition = None

        result = self._visit(stmt.elt)

        comprehensions = list(self._visit(stmt.generators))
        generators = [c[0] for c in comprehensions]
        conditions = [c[1] for c in comprehensions]

        parent = self._context[-2]
        if not isinstance(parent, ast.Call):
            raise NotImplementedError("GeneratorExp is not the argument of a function call")

        name = self._visit(parent.func)

        grandparent = self._context[-3]
        if isinstance(grandparent, ast.Assign):
            if len(grandparent.targets) != 1:
                raise NotImplementedError("Cannot unpack function with generator expression argument")
            lhs = self._visit(grandparent.targets[0])
        else:
            lhs = PyccelSymbol('_', is_temp=True)

        body = result
        if name == 'sum':
            body = AugAssign(lhs, '+', body)
        else:
            body = FunctionCall(name, (lhs, body))
            body = Assign(lhs, body)

        body.set_current_ast(parent)

        for loop, condition in zip(generators, conditions):
            if condition:
                loop.insert2body(condition)

        if generators[-1].body.body: # this is an If node
            if_node = generators[-1].body.body[0]
            if_node.blocks[0].body.insert2body(body)
        else:
            generators[-1].insert2body(body)

        while len(generators) > 1:
            indices.append(generators[-1].target)
            outer_loop = generators.pop()
            inserted_into = generators[-1]
            if inserted_into.body.body:
                inserted_into.body.body[0].blocks[0].body.insert2body(outer_loop)
            else:
                inserted_into.insert2body(outer_loop)
        indices.append(generators[-1].target)

        indices = indices[::-1]

        if name == 'sum':
            expr = FunctionalSum(generators[0], result, lhs, indices, conditions=conditions)
        elif name == 'min':
            expr = FunctionalMin(generators[0], result, lhs, indices, conditions=conditions)
        elif name == 'max':
            expr = FunctionalMax(generators[0], result, lhs, indices, conditions=conditions)
        else:
            expr = EmptyNode()
            errors.report(PYCCEL_RESTRICTION_TODO,
                          symbol = name,
                          bounding_box=(stmt.lineno, stmt.col_offset),
                          severity='error')

        expr.set_current_ast(parent)

        return expr

    def _visit_If(self, stmt):

        test = self._visit(stmt.test)
        orelse = self._visit(stmt.orelse)

        if isinstance(test, PyccelEq) and test.args[0] == '__name__' and test.args[1] == '__main__' \
                and isinstance(test.args[0], PyccelSymbol) and isinstance(test.args[1], LiteralString):
            if len(orelse) != 0:
                errors.report("Can't add an else condition to a program",
                        symbol = stmt,
                        severity = 'error')

            scope = self.create_new_function_scope('__main__')
            body = [self._visit(v) for v in stmt.body]
            self.exit_function_scope()

            imports = [i for i in body if isinstance(i, Import)]
            functions = [l for l in body if isinstance(l, FunctionDef) and not l.is_inline]
            classes = [l for l in body if isinstance(l, ClassDef)]
            if classes:
                errors.report("Classes should be declared in the module not in the program body",
                              symbol = stmt, severity = 'error')
            if any(not f.is_inline for f in functions):
                errors.report("Functions should be declared in the module not in the program body",
                              symbol = stmt, severity = 'error')
            body = [l for l in body if l not in chain(imports, functions, classes)]

            return Program('__main__', (), CodeBlock(body), imports=imports, scope = scope)
        else:
            body = self._visit(stmt.body)

        if len(orelse)==1 and isinstance(orelse[0],If):
            orelse = orelse[0].blocks
            return If(IfSection(test, body), *orelse)
        elif len(orelse)==0:
            return If(IfSection(test, body))
        else:
            orelse = IfSection(LiteralTrue(), orelse)
            return If(IfSection(test, body), orelse)

    def _visit_IfExp(self, stmt):

        test1 = self._visit(stmt.test)
        first = self._visit(stmt.body)
        second = self._visit(stmt.orelse)
        expr = IfTernaryOperator(test1, first, second)
        return expr

    def _visit_While(self, stmt):

        scope = self.create_new_loop_scope()

        test = self._visit(stmt.test)
        body = self._visit(stmt.body)

        self.exit_loop_scope()

        return While(test, body, scope=scope)

    def _visit_Assert(self, stmt):
        test = self._visit(stmt.test)
        return Assert(test)

    def _visit_CommentMultiLine(self, stmt):

        exprs = [self._treat_comment_line(com, stmt) for com in stmt.s.split('\n')]

        if len(exprs) == 1:
            return exprs[0]
        else:
            return CodeBlock(exprs)

    def _visit_CommentLine(self, stmt):
        return self._treat_comment_line(stmt.s, stmt)

    def _visit_Break(self, stmt):
        return Break()

    def _visit_Continue(self, stmt):
        return Continue()

    def _visit_Lambda(self, stmt):
        assign_node = self._context[-2]
        if not isinstance(assign_node, ast.Assign):
            errors.report("Lambda functions are only supported in assign statements currently.",
                          severity='fatal', symbol=stmt)
        name_lst = self._visit(assign_node.targets)

        assert len(name_lst) == 1
        name = name_lst[0]

        self.scope.insert_symbol(name, 'function')
        new_name = self.scope.get_expected_name(name)
        scope = self.create_new_function_scope(name,
                used_symbols = {name: new_name},
                original_symbols = {new_name: name})

        args = self._visit(stmt.args)
        return_expr = Return(self._visit(stmt.body))
        return_expr.set_current_ast(stmt)

        results = FunctionDefResult(self.scope.get_new_name())

        self.exit_function_scope()

        return InlineFunctionDef(name, args, CodeBlock([return_expr]), results, scope = scope)

    def _visit_withitem(self, stmt):
        # stmt.optional_vars
        context = self._visit(stmt.context_expr)
        if stmt.optional_vars:
            return AsName(context, stmt.optional_vars)
        else:
            return context

    def _visit_With(self, stmt):
        domain = self._visit(stmt.items)
        if len(domain) == 1:
            domain = domain[0]
        body = self._visit(stmt.body)

        return With(domain, body)

    def _visit_Try(self, stmt):
        # this is a blocking error, since we don't want to convert the try body
        errors.report(PYCCEL_RESTRICTION_TRY_EXCEPT_FINALLY,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_Raise(self, stmt):
        errors.report(PYCCEL_RESTRICTION_RAISE,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_Yield(self, stmt):
        errors.report(PYCCEL_RESTRICTION_YIELD,
                      bounding_box=(stmt.lineno, stmt.col_offset),
                      severity='error')

    def _visit_Starred(self, stmt):
        return StarredArguments(self._visit(stmt.value))

    def _visit_TypeAlias(self, stmt):
        if stmt.type_params:
            errors.report("Type parameters are not yet supported on a type alias expression.\n"+PYCCEL_RESTRICTION_TODO,
                    severity='error', symbol=stmt)
        self._in_lhs_assign = True
        name = self._visit(stmt.name)
        self._in_lhs_assign = False
        rhs = self._treat_type_annotation(stmt.value, self._visit(stmt.value))
        type_annotation = UnionTypeAnnotation(VariableTypeAnnotation(TypeAlias(), is_const = True))
        return Assign(AnnotatedPyccelSymbol(name, annotation=type_annotation), rhs)

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError('Expecting an argument for filename')

    parser = SyntaxParser(filename)
    print(parser.ast)

