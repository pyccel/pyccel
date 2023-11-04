# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

import os
import re

import ast
import warnings

from textx.exceptions import TextXSyntaxError

#==============================================================================

from pyccel.ast.basic import PyccelAstNode

from pyccel.ast.core import FunctionCall, FunctionCallArgument
from pyccel.ast.core import Module
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Return
from pyccel.ast.core import Pass
from pyccel.ast.core import FunctionDef, InlineFunctionDef
from pyccel.ast.core import SympyFunction
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

from pyccel.ast.bitwise_operators import PyccelRShift, PyccelLShift, PyccelBitXor, PyccelBitOr, PyccelBitAnd, PyccelInvert
from pyccel.ast.operators import PyccelPow, PyccelAdd, PyccelMul, PyccelDiv, PyccelMod, PyccelFloorDiv
from pyccel.ast.operators import PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe
from pyccel.ast.operators import PyccelAnd, PyccelOr,  PyccelNot, PyccelMinus
from pyccel.ast.operators import PyccelUnary, PyccelUnarySub
from pyccel.ast.operators import PyccelIs, PyccelIsNot
from pyccel.ast.operators import IfTernaryOperator
from pyccel.ast.numpyext  import NumpyMatmul

from pyccel.ast.builtins import PythonTuple, PythonList
from pyccel.ast.builtins import PythonPrint, Lambda
from pyccel.ast.headers  import MetaVariable, FunctionHeader, MethodHeader
from pyccel.ast.literals import LiteralInteger, LiteralFloat, LiteralComplex
from pyccel.ast.literals import LiteralFalse, LiteralTrue, LiteralString
from pyccel.ast.literals import Nil
from pyccel.ast.functionalexpr import FunctionalSum, FunctionalMax, FunctionalMin, GeneratorComprehension, FunctionalFor
from pyccel.ast.variable  import DottedName, AnnotatedPyccelSymbol

from pyccel.ast.internals import Slice, PyccelSymbol, PyccelInternalFunction

from pyccel.ast.type_annotations import SyntacticTypeAnnotation, UnionTypeAnnotation

from pyccel.parser.base        import BasicParser
from pyccel.parser.extend_tree import extend_tree
from pyccel.parser.utilities   import read_file
from pyccel.parser.utilities   import get_default_path

from pyccel.parser.syntax.headers import parse as hdr_parse, types_meta
from pyccel.parser.syntax.openmp  import parse as omp_parse
from pyccel.parser.syntax.openacc import parse as acc_parse

from pyccel.utilities.stage import PyccelStage

from pyccel.errors.errors import Errors

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

    **kwargs : dict
        Additional keyword arguments for BasicParser.
    """

    def __init__(self, inputs, **kwargs):
        BasicParser.__init__(self, **kwargs)

        # check if inputs is a file
        code = inputs
        if os.path.isfile(inputs):

            self._filename = inputs
            errors.set_target(self.filename, 'file')

            # we don't use is_valid_filename_py since it uses absolute path
            # file extension

            code = read_file(inputs)

        self._code    = code
        self._context = []

        self.load()

        tree                = extend_tree(code)
        self._fst           = tree
        self._in_lhs_assign = False

        self.parse()
        self.dump()

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
        errors.set_parser_stage('syntax')

        pyccel_stage.set_stage('syntactic')
        ast       = self._visit(self.fst)
        self._ast = ast

        self._syntax_done = True

        return ast

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
                if isinstance(expr, (MethodHeader, FunctionHeader)):
                    self.scope.insert_header(expr)
                    expr = EmptyNode()
                elif isinstance(expr, AnnotatedPyccelSymbol):
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

        expr.ast = stmt
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
        if isinstance(annotation, (PyccelSymbol, DottedName)):
            return SyntacticTypeAnnotation(dtypes=[annotation], ranks=[0], orders=[None], is_const=False)
        elif isinstance(annotation, IndexedElement):
            return SyntacticTypeAnnotation(dtypes=[annotation], ranks=[len(annotation.indices)], orders=[None], is_const=False)
        elif isinstance(annotation, LiteralString):
            try:
                annotation = types_meta.model_from_str(annotation.python_value)
            except TextXSyntaxError as e:
                errors.report(f"Invalid header. {e.message}",
                        symbol = stmt, column = e.col,
                        severity='fatal')
            annot = SyntacticTypeAnnotation.build_from_textx(annotation)
            if isinstance(stmt, PyccelAstNode):
                annot.ast = stmt.ast
            else:
                annot.ast = stmt
            return annot
        elif annotation is Nil():
            return None
        elif isinstance(annotation, PyccelBitOr):
            return UnionTypeAnnotation(*[self._treat_type_annotation(stmt, a) for a in annotation.args])
        else:
            errors.report('Invalid type annotation',
                        symbol = stmt, severity='error')
            return EmptyNode()

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
            self._context.append(stmt)
            result = getattr(self, syntax_method)(stmt)
            if isinstance(result, PyccelAstNode) and result.ast is None and isinstance(stmt, ast.AST):
                result.ast = stmt
            self._context.pop()
            return result

        # Unknown object, we raise an error.
        return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX, symbol=stmt,
                      severity='error')

    def _visit_Module(self, stmt):
        """ Visits the ast and splits the result into elements relevant for the module or the program"""
        body          = [self._visit(v) for v in stmt.body]

        # Define the name of the module
        # The module name allows it to be correctly referenced from an import command
        mod_name = os.path.splitext(os.path.basename(self._filename))[0]
        name = AsName(mod_name, self.scope.get_new_name(mod_name))

        body = [b for i in body for b in (i.body if isinstance(i, CodeBlock) else [i])]
        return Module(name, [], [], program = CodeBlock(body), scope=self.scope)

    def _visit_Expr(self, stmt):
        val = self._visit(stmt.value)
        if not isinstance(val, (CommentBlock, PythonPrint)):
            # Collect any results of standalone expressions
            # into a variable to avoid errors in C/Fortran
            val = Assign(PyccelSymbol('_', is_temp=True), val)
        return val

    def _visit_Tuple(self, stmt):
        return PythonTuple(*self._treat_iterable(stmt.elts))

    def _visit_List(self, stmt):
        return PythonList(*self._treat_iterable(stmt.elts))

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
        errors.report(PYCCEL_RESTRICTION_TODO,
                symbol=stmt, severity='error')

    def _visit_NoneType(self, stmt):
        return Nil()

    def _visit_str(self, stmt):

        return stmt

    def _visit_Str(self, stmt):
        val =  stmt.s
        if isinstance(self._context[-2], ast.Expr):
            return CommentBlock(val)
        return LiteralString(val)

    def _visit_Num(self, stmt):
        val = stmt.n

        if isinstance(val, int):
            return LiteralInteger(val)
        elif isinstance(val, float):
            return LiteralFloat(val)
        elif isinstance(val, complex):
            return LiteralComplex(val.real, val.imag)
        else:
            raise NotImplementedError(f'Num type {type(val)} not recognised')

    def _visit_Assign(self, stmt):

        self._in_lhs_assign = True
        lhs = self._visit(stmt.targets)
        self._in_lhs_assign = False
        if len(lhs)==1:
            lhs = lhs[0]
        else:
            lhs = PythonTuple(*lhs)

        rhs = self._visit(stmt.value)

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

        if stmt.vararg or stmt.kwarg:
            errors.report(VARARGS, symbol = stmt,
                    severity='error')
            return []

        arguments       = []
        if stmt.args:
            n_expl = len(stmt.args)-len(stmt.defaults)

            arguments = []
            for a in stmt.args[:n_expl]:
                annotation=self._treat_type_annotation(a, self._visit(a.annotation))
                new_arg = FunctionDefArgument(AnnotatedPyccelSymbol(a.arg, annotation),
                                            annotation=annotation)
                new_arg.ast = a
                arguments.append(new_arg)

            for a,d in zip(stmt.args[n_expl:], stmt.defaults):
                annotation=self._treat_type_annotation(a, self._visit(a.annotation))
                new_arg = FunctionDefArgument(AnnotatedPyccelSymbol(a.arg, annotation),
                                            annotation=annotation,
                                            value = self._visit(d))
                new_arg.ast = a
                arguments.append(new_arg)

        if stmt.kwonlyargs:
            for a,d in zip(stmt.kwonlyargs,stmt.kw_defaults):
                annotation=self._treat_type_annotation(a, self._visit(a.annotation))
                val = self._visit(d) if d is not None else d
                arg = FunctionDefArgument(AnnotatedPyccelSymbol(a.arg, annotation),
                            annotation=annotation,
                            value=val, kwonly=True)
                arg.ast = a

                arguments.append(arg)

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
            return self._visit_Str(stmt)

        else:
            raise NotImplementedError(f'Constant type {type(stmt.value)} not recognised')

    def _visit_NameConstant(self, stmt):
        if stmt.value is None:
            return Nil()

        elif stmt.value is True:
            return LiteralTrue()

        elif stmt.value is False:
            return LiteralFalse()

        else:
            raise NotImplementedError(f"Unknown NameConstant : {stmt.value}")


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
                source = AsName(self._treat_import_source(imp.object, 0), imp.target)
            else:
                source = self._treat_import_source(imp, 0)
            import_line = Import(source)
            import_line.ast = stmt
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
        if len(stmt.ops)>1:
            return errors.report(PYCCEL_RESTRICTION_MULTIPLE_COMPARISONS,
                      symbol = stmt,
                      severity='error')

        first = self._visit(stmt.left)
        second = self._visit(stmt.comparators[0])
        op = stmt.ops[0]

        if isinstance(op, ast.Eq):
            return PyccelEq(first, second)
        if isinstance(op, ast.NotEq):
            return PyccelNe(first, second)
        if isinstance(op, ast.Lt):
            return PyccelLt(first, second)
        if isinstance(op, ast.Gt):
            return PyccelGt(first, second)
        if isinstance(op, ast.LtE):
            return PyccelLe(first, second)
        if isinstance(op, ast.GtE):
            return PyccelGe(first, second)
        if isinstance(op, ast.Is):
            return PyccelIs(first, second)
        if isinstance(op, ast.IsNot):
            return PyccelIsNot(first, second)

        return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX,
                      symbol = stmt,
                      severity='error')

    def _visit_Return(self, stmt):
        results = self._visit(stmt.value)
        if results is Nil():
            results = []
        elif not isinstance(results, (list, PythonTuple, PythonList)):
            results = [results]
        return Return(results)

    def _visit_Pass(self, stmt):
        return Pass()

    def _visit_FunctionDef(self, stmt):

        #  TODO check all inputs and which ones should be treated in stage 1 or 2

        name = PyccelSymbol(self._visit(stmt.name))
        self.scope.insert_symbol(name)

        headers = self.scope.find(name, 'headers')

        scope = self.create_new_function_scope(name)

        arguments    = self._visit(stmt.args)

        template    = {}
        is_pure      = False
        is_elemental = False
        is_private   = False
        is_inline    = False
        imports      = []
        doc_string   = None

        decorators = {}

        for d in self._visit(stmt.decorator_list):
            tmp_var = d if isinstance(d, PyccelSymbol) else d.funcdef
            if tmp_var in decorators:
                decorators[tmp_var] += [d]
            else:
                decorators[tmp_var] = [d]

        if 'types' in decorators:
            warnings.warn("The @types decorator will be removed in a future version of Pyccel. Please use type hints. The @template decorator can be used to specify multiple types", FutureWarning)

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

        template['template_dict'] = {}
        # extract the templates
        if 'template' in decorators:
            for template_decorator in decorators['template']:
                dec_args = template_decorator.args
                if len(dec_args) != 2:
                    msg = 'Number of Arguments provided to the template decorator is not valid'
                    errors.report(msg, symbol = template_decorator,
                                    severity='error')

                if any(i.keyword not in (None, 'name', 'types') for i in dec_args):
                    errors.report('Argument provided to the template decorator is not valid',
                                    symbol = template_decorator, severity='error')

                if dec_args[0].has_keyword and dec_args[0].keyword != 'name':
                    type_name = dec_args[1].value.python_value
                    type_descriptors = dec_args[0].value
                else:
                    type_name = dec_args[0].value.python_value
                    type_descriptors = dec_args[1].value

                if not isinstance(type_descriptors, (PythonTuple, PythonList)):
                    type_descriptors = PythonTuple(type_descriptors)

                if type_name in template['template_dict']:
                    errors.report(f'The template "{type_name}" is duplicated',
                                symbol = template_decorator, severity='warning')

                possible_types = self._treat_type_annotation(template_decorator, type_descriptors.args)

                # Make templates decorator dict accessible from decorators dict
                template['template_dict'][type_name] = possible_types

            # Make template decorator list accessible from decorators dict
            template['decorator_list'] = decorators['template']
            decorators['template'] = template

        if not template['template_dict']:
            decorators['template'] = None

        argument_annotations = [a.annotation for a in arguments]
        result_annotation = self._treat_type_annotation(stmt, self._visit(stmt.returns))

        #---------------------------------------------------------------------------------------------------------
        #                   To remove when headers are deprecated
        #---------------------------------------------------------------------------------------------------------
        if headers:
            warnings.warn("Support for specifying types via headers will be removed in a " +
                          "future version of Pyccel. Please use type hints. The @template " +
                          "decorator can be used to specify multiple types. See the " +
                          "documentation at " +
                          "https://github.com/pyccel/pyccel/blob/devel/docs/quickstart.md#type-annotations " +
                          "for examples.", FutureWarning)
            if any(a is not None for a in argument_annotations):
                errors.report("Type annotations and type specification via headers should not be mixed",
                        symbol=stmt, severity='error')

            for i, _ in enumerate(argument_annotations):
                argument_annotations[i] = UnionTypeAnnotation()
            if result_annotation is not None:
                errors.report("Type annotations and type specification via headers should not be mixed",
                            symbol=stmt, severity='error')

            n_results = 0

            for head in headers:
                if len(head.dtypes) != len(argument_annotations):
                    errors.report(f"Wrong number of types in header for function {name}",
                            severity='error', symbol=stmt)
                else:
                    for i,arg in enumerate(head.dtypes):
                        argument_annotations[i].add_type(arg)
                if head.results:
                    if result_annotation is None:
                        result_annotation = head.results
                    else:
                        if len(result_annotation) != len(head.results):
                            errors.report("Different length results in headers.",
                                    severity='error', symbol=stmt)
                        else:
                            result_annotation = tuple(UnionTypeAnnotation(r, *getattr(t, 'type_list', [t])) \
                                                        for r,t in zip(head.results, result_annotation))
                    n_results += 1

            if n_results and n_results != len(headers):
                errors.report("Results have only been provided for some of the headers. The types of the results will all be chosen from the provided types. This may result in unexpected result types.",
                        severity='warning', symbol=stmt)

        # extract the types to construct a header
        if 'types' in decorators:
            if any(a is not None for a in argument_annotations):
                errors.report("Type annotations and type specification via headers should not be mixed",
                        symbol=stmt, severity='error')

            for i, _ in enumerate(argument_annotations):
                argument_annotations[i] = UnionTypeAnnotation()
            n_results = 0

            for types_decorator in decorators['types']:
                type_args = types_decorator.args

                args = [a for a in type_args if not a.has_keyword]
                kwargs = [a for a in type_args if a.has_keyword]

                if len(kwargs) > 1:
                    errors.report('Too many keyword arguments passed to @types decorator',
                                symbol = types_decorator,
                                bounding_box = (stmt.lineno, stmt.col_offset),
                                severity='error')
                elif kwargs:
                    if kwargs[0].keyword != 'results':
                        errors.report('Wrong keyword argument passed to @types decorator',
                                    symbol = types_decorator,
                                    bounding_box = (stmt.lineno, stmt.col_offset),
                                    severity='error')
                    annots = self._treat_type_annotation(kwargs[0], kwargs[0].value.args)
                    if result_annotation is None:
                        result_annotation = annots
                    else:
                        if len(result_annotation) != len(annots):
                            errors.report("Different length results in headers.",
                                    severity='error', symbol=stmt)
                        else:
                            result_annotation = tuple(UnionTypeAnnotation(r, *getattr(t, 'type_list', [t])) \
                                                        for r,t in zip(annots, result_annotation))
                    n_results += 1

                if len(args) != len(argument_annotations):
                    errors.report(f"Wrong number of types in header for function {name}",
                            severity='error', symbol=stmt)
                else:
                    for i,arg in enumerate(args):
                        argument_annotations[i].add_type(self._treat_type_annotation(arg, arg.value))

            if n_results and n_results != len(decorators['types']):
                errors.report("Results have only been provided for some of the types decorators. The types of the results will all be chosen from the provided types. This may result in unexpected result types.",
                        severity='warning', symbol=stmt)

        #---------------------------------------------------------------------------------------------------------
        #                   End of : To remove when headers are deprecated
        #---------------------------------------------------------------------------------------------------------

        # Repack AnnotatedPyccelSymbols to insert argument_annotations from headers or types decorators
        arguments = [FunctionDefArgument(AnnotatedPyccelSymbol(a.var.name, annot), annotation=annot, value=a.value, kwonly=a.is_kwonly)
                           for a, annot in zip(arguments, argument_annotations)]

        body = stmt.body

        if 'sympy' in decorators:
            # TODO maybe we should run pylint here
            stmt.decorators.pop()
            func = SympyFunction(name, arguments, [], [str(stmt)])
            func.ast = stmt
            self.insert_function(func)
            return EmptyNode()

        else:
            body = self._visit(body)

        # Collect docstring
        if len(body) > 0 and isinstance(body[0], CommentBlock):
            doc_string = body[0]
            doc_string.header = ''
            body = body[1:]

        body = CodeBlock(body)

        returns = [i.expr for i in body.get_attribute_nodes(Return,
                    excluded_nodes = (Assign, FunctionCall, PyccelInternalFunction, FunctionDef))]
        assert all(len(i) == len(returns[0]) for i in returns)
        if is_inline and len(returns)>1:
            errors.report("Inline functions cannot have multiple return statements",
                    symbol = stmt,
                    severity = 'error')
        results = []
        result_counter = 1

        local_symbols = self.scope.local_used_symbols

        if result_annotation and not isinstance(result_annotation, tuple):
            result_annotation = [result_annotation]

        for i,r in enumerate(zip(*returns)):
            r0 = r[0]

            pyccel_symbol  = isinstance(r0, PyccelSymbol)
            same_results   = all(r0 == ri for ri in r)
            name_available = all(r0 != a.name for a in arguments) and r0 in local_symbols

            if pyccel_symbol and same_results and name_available:
                result_name = r0
            else:
                result_name, result_counter = self.scope.get_new_incremented_symbol('Out', result_counter)

            if result_annotation:
                result_name = AnnotatedPyccelSymbol(result_name, annotation = result_annotation[i])

            results.append(FunctionDefResult(result_name, annotation = result_annotation))
            results[-1].ast = stmt

        self.exit_function_scope()

        cls = InlineFunctionDef if is_inline else FunctionDef
        func = cls(
               name,
               arguments,
               results,
               body,
               is_pure=is_pure,
               is_elemental=is_elemental,
               is_private=is_private,
               imports=imports,
               decorators=decorators,
               doc_string=doc_string,
               scope=scope)

        return func

    def _visit_ClassDef(self, stmt):

        name = stmt.name
        self.scope.insert_symbol(name)
        scope = self.create_new_class_scope(name)
        methods = []
        attributes = []
        for i in stmt.body:
            visited_i = self._visit(i)
            if isinstance(visited_i, FunctionDef):
                methods.append(visited_i)
            elif isinstance(visited_i, Pass):
                return errors.report(UNSUPPORTED_FEATURE_OOP_EMPTY_CLASS, symbol = stmt, severity='error')
            elif isinstance(visited_i, AnnotatedPyccelSymbol):
                attributes.append(visited_i)
            else:
                errors.report(f"{type(visited_i)} not currently supported in classes",
                        severity='error', symbol=visited_i)
        for i in methods:
            i.cls_name = name
        parent = [p for p in (self._visit(i) for i in stmt.bases) if p != 'object']
        self.exit_class_scope()
        expr = ClassDef(name=name, attributes=attributes,
                        methods=methods, superclasses=parent, scope=scope)

        return expr

    def _visit_Subscript(self, stmt):

        ch = stmt
        args = []
        while isinstance(ch, ast.Subscript):
            val = self._visit(ch.slice)
            if isinstance(val, (PythonTuple, list)):
                args += val
            else:
                args.insert(0, val)
            ch = ch.value
        args = tuple(args)
        var = self._visit(ch)
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
            args += [FunctionCallArgument(self._visit(a), ast=a) for a in stmt.args]
        if stmt.keywords:
            kwargs = self._visit(stmt.keywords)
            for k, a in zip(kwargs, stmt.keywords):
                k.ast = a

            args += kwargs

        if len(args) == 0:
            args = ()

        if len(args) == 1 and isinstance(args[0].value, GeneratorComprehension):
            return args[0].value

        func = self._visit(stmt.func)

        if isinstance(func, PyccelSymbol):
            if func == "print":
                func = PythonPrint(PythonTuple(*args))
            else:
                func = FunctionCall(func, args)
        elif isinstance(func, DottedName):
            func_attr = FunctionCall(func.name[-1], args)
            func = DottedName(*func.name[:-1], func_attr)
        else:
            raise NotImplementedError(f' Unknown function type {type(func)}')

        return func

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

        self._in_lhs_assign = True
        iterator = self._visit(stmt.target)
        self._in_lhs_assign = False
        iterable = self._visit(stmt.iter)

        self.exit_loop_scope()

        expr = For(iterator, iterable, [], scope=scope)
        return expr

    def _visit_ListComp(self, stmt):

        result = self._visit(stmt.elt)

        generators = list(self._visit(stmt.generators))

        if not isinstance(self._context[-2],ast.Assign):
            errors.report(PYCCEL_RESTRICTION_LIST_COMPREHENSION_ASSIGN,
                          symbol = stmt,
                          severity='error')
            lhs = PyccelSymbol('_', is_temp=True)
        else:
            lhs = self._visit(self._context[-2].targets)
            if len(lhs)==1:
                lhs = lhs[0]
            else:
                raise NotImplementedError("A list comprehension cannot be unpacked")

        index = PyccelSymbol('_', is_temp=True)

        args = [index]
        target = IndexedElement(lhs, *args)
        target = Assign(target, result)
        assign1 = Assign(index, LiteralInteger(0))
        assign1.ast = stmt
        target.ast = stmt
        generators[-1].insert2body(target)
        assign2 = Assign(index, PyccelAdd(index, LiteralInteger(1)))
        assign2.ast = stmt
        generators[-1].insert2body(assign2)

        indices = [generators[-1].target]
        while len(generators) > 1:
            F = generators.pop()
            generators[-1].insert2body(F)
            indices.append(generators[-1].target)
        indices = indices[::-1]

        return FunctionalFor([assign1, generators[-1]],target.rhs, target.lhs,
                             indices, index)

    def _visit_GeneratorExp(self, stmt):

        result = self._visit(stmt.elt)

        generators = self._visit(stmt.generators)
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

        body.ast = parent
        indices = []
        generators = list(generators)
        while len(generators) > 0:
            indices.append(generators[-1].target)
            generators[-1].insert2body(body)
            body = generators.pop()

        indices = indices[::-1]
        if name == 'sum':
            expr = FunctionalSum(body, result, lhs, indices)
        elif name == 'min':
            expr = FunctionalMin(body, result, lhs, indices)
        elif name == 'max':
            expr = FunctionalMax(body, result, lhs, indices)
        else:
            expr = EmptyNode()
            errors.report(PYCCEL_RESTRICTION_TODO,
                          symbol = name,
                          bounding_box=(stmt.lineno, stmt.col_offset),
                          severity='error')

        expr.ast = parent

        return expr

    def _visit_If(self, stmt):

        test = self._visit(stmt.test)
        body = self._visit(stmt.body)
        orelse = self._visit(stmt.orelse)
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

        expr = self._visit(stmt.body)
        args = self._visit(stmt.args)

        return Lambda(tuple(args), expr)

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

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError('Expecting an argument for filename')

    parser = SyntaxParser(filename)
    print(parser.ast)

