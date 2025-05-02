# pylint: disable=protected-access, missing-function-docstring
"""
Module containing the grammar rules for the OpenMP 4.5 specification,
"""
import ast
import re
from os.path import join, dirname

from textx.metamodel import metamodel_from_file
from textx import metamodel_for_language

from pyccel.ast.core import FunctionCall
from pyccel.ast.variable import Variable
from pyccel.parser.extend_tree import extend_tree
from pyccel.ast.datatypes import PythonNativeInt
from pyccel.ast.core import CodeBlock
from pyccel.errors.errors import Errors
from pyccel.plugins.Openmp.omp import OmpDirective, OmpClause, OmpEndDirective, OmpConstruct, OmpExpr, OmpList
from pyccel.plugins.Openmp.omp import OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger
from pyccel.ast.core import EmptyNode

errors = Errors()
from pyccel.errors.messages import *

class Commons:
    @classmethod
    def _helper_check_config(cls, options, method):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if cls._version != options.get('omp_version', None) or not 'openmp' in options.get('accelerators'):
                    if method:
                        return method(*args, **kwargs)
                    else:
                        return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX, symbol=args[-1],
                                             severity='error')
                return func(*args, **kwargs)
            return wrapper
        return decorator


class SyntaxParser(Commons):
    """Openmp 4.5 syntax parser"""
    _version = 4.5
    @classmethod
    def setup(cls, options, method):
        this_folder = dirname(__file__)
        # Get metamodel from language description
        grammar = join(this_folder, "grammar/openmp.tx")
        omp_classes = [OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger, OmpList]
        cls._omp_metamodel = metamodel_from_file(grammar, classes=omp_classes)

        # object processors: are registered for particular classes (grammar rules)
        # and are called when the objects of the given class is instantiated.
        # The rules OMP_X_Y are used to insert the version of the syntax used

        textx_mm = metamodel_for_language('textx')
        grammar_model = textx_mm.grammar_model_from_file(grammar)

        obj_processors = {r.name: (lambda r: lambda _: r.name.replace('_PARENT', '').lower())(r)
                          for r in grammar_model.rules if r.name.endswith('_PARENT')}

        obj_processors.update({
            'OMP_4_5': lambda _: 4.5,
            'OMP_5_0': lambda _: 5.0,
            'OMP_5_1': lambda _: 5.1,
            'TRUE': lambda _: True,
            'OMP_VERSION': lambda _: 4.5,
        })
        cls._omp_metamodel.register_obj_processors(obj_processors)
        def setup(instance, *args, **kwargs):
            if cls._version != options.get('omp_version', None) or not 'openmp' in options.get('accelerators'):
                method(instance, *args, **kwargs)
                return
            instance._skip_stmts_count = 0
            method(instance, *args, **kwargs)
        return setup

    @classmethod
    def _treat_comment_line(cls, options, method=None):

        """
        Parse a comment line.

        Parse a comment which fits in a single line if the comment
        begins with `#$omp` using textx.

        Parameters
        ----------
        line : str
            The comment line.
        expr : ast.Ast
            The comment object in the code. This is useful for raising
            errors.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The treated object as an Openmp node.
        """
        @cls._helper_check_config(options, method)
        def _treat_comment_line(instance, line, expr):
            if line.startswith('#$') and line[2:].lstrip().startswith('omp'):
                from textx.exceptions import TextXError
                try:
                    model = cls._omp_metamodel.model_from_str(line)
                    model.raw = line
                    directive = OmpDirective.from_tx_directive(model.statement)
                    directive.set_current_ast(expr)
                    return instance._visit(directive)

                except TextXError as e:
                    errors.report(e.message, severity="fatal", symbol=expr)
                    return None
            elif method:
                return method(instance, line, expr)
        return _treat_comment_line

    @classmethod
    def _visit(cls, options, method):
        @cls._helper_check_config(options, method)
        def _visit(instance, stmt):
            if instance._skip_stmts_count:
                instance._skip_stmts_count -= 1
                return EmptyNode()
            else:
                return method(instance, stmt)
        return _visit

    @classmethod
    def _visit_OmpDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpDirective(self, stmt):
            if hasattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive"):
                return getattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive")(stmt)
            clauses = [self._visit(clause) for clause in stmt.clauses]
            directive = OmpDirective.from_directive(stmt, clauses=clauses)
            if stmt.is_construct:
                body = []
                end = None
                container = None
                for el in self._context[::-1]:
                    if isinstance(el, list):
                        container = el[el.index(self._context[-2]) + 1:].copy()
                        break
                for line in container:
                    expr =  self._visit(line)
                    if isinstance(expr, OmpEndDirective) and stmt.name == expr.name:
                        end = expr
                        break
                    body.append(expr)
                if end is None:
                    errors.report(
                        f"missing `end {stmt.name}` directive",
                        symbol=stmt,
                        severity="fatal",
                    )
                self._skip_stmts_count = len(body) + 1
                body = CodeBlock(body=body)
                return OmpConstruct(start=directive, end=end, body=body)

            return directive
        return _visit_OmpDirective

    @classmethod
    def _visit_for_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_for_directive(self, stmt):
            loop = None
            for el in self._context[::-1]:
                if isinstance(el, list):
                    loop_pos = el.index(self._context[-2]) + 1
                    if len(el) < loop_pos + 1 or not isinstance(el[loop_pos], ast.For):
                        errors.report(
                            f"{stmt.name} directive should be followed by a for loop",
                            symbol=stmt,
                            severity="fatal",
                        )
                    loop = self._visit(el[loop_pos])
                    break
            clauses = [self._visit(clause) for clause in stmt.clauses]
            directive = OmpDirective.from_directive(stmt, clauses=clauses)
            self._skip_stmts_count = 1
            body = CodeBlock(body=[loop])
            return OmpConstruct(start=directive, end=None, body=body)
        return _visit_for_directive

    @classmethod
    def _visit_simd_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_simd_directive(self, expr):
            return self._visit_for_directive(expr)
        return _visit_simd_directive

    @classmethod
    def _visit_parallel_for_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_parallel_for_directive(self, expr):
            return self._visit_for_directive(expr)
        return _visit_parallel_for_directive

    @classmethod
    def _visit_OmpClause(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpClause(self, expr):
            omp_exprs = [self._visit(e) for e in expr.omp_exprs]
            return OmpClause.from_clause(expr, omp_exprs=omp_exprs)
        return _visit_OmpClause

    @classmethod
    def _visit_OmpEndDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpEndDirective(self, expr):
            clauses = [self._visit(clause) for clause in expr.clauses]
            end = OmpEndDirective.from_directive(expr, clauses=clauses)
            return end

        return _visit_OmpEndDirective

    @classmethod
    def _visit_OmpScalarExpr(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpScalarExpr(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return OmpScalarExpr.from_omp_expr(expr, value=self._visit(fst))

        return _visit_OmpScalarExpr

    @classmethod
    def _visit_OmpIntegerExpr(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpIntegerExpr(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return OmpIntegerExpr.from_omp_expr(expr, value=self._visit(fst))

        return _visit_OmpIntegerExpr

    @classmethod
    def _visit_OmpConstantPositiveInteger(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpConstantPositiveInteger(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return OmpConstantPositiveInteger.from_omp_expr(expr, value=self._visit(fst))

        return _visit_OmpConstantPositiveInteger

    @classmethod
    def _visit_OmpList(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpList(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return OmpList.from_omp_expr(expr, value=self._visit(fst))

        return _visit_OmpList

    @classmethod
    def _visit_OmpExpr(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpExpr(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return OmpExpr.from_omp_expr(expr, value=self._visit(fst))

        return _visit_OmpExpr


    
class SemanticParser(Commons):
    """Openmp 4.5 semantic parser"""
    _version = 4.5
    @classmethod
    def _visit_OmpDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpDirective(self, expr):
            if hasattr(self, f"_visit_{expr.name.replace(' ', '_')}_directive"):
                return getattr(self, f"_visit_{expr.name.replace(' ', '_')}_directive")(expr)
            clauses = [self._visit(clause) for clause in expr.clauses]
            directive = OmpDirective.from_directive(expr, clauses=clauses)
            return directive

        return _visit_OmpDirective

    @classmethod
    def _visit_OmpConstruct(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpConstruct(self, expr):
            if hasattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct"):
                return getattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct")(expr)

            body = self._visit(expr.body)
            start = self._visit(expr.start)
            end = self._visit(expr.end) if expr.end else None
            return OmpConstruct(start=start, end=end, body=body)

        return _visit_OmpConstruct

    @classmethod
    def _visit_for_construct(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_for_construct(self, expr):
            body = self._visit(expr.body)
            start = self._visit(expr.start)
            return OmpConstruct(start=start, end=None, body=body)

        return _visit_for_construct

    @classmethod
    def _visit_simd_construct(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_simd_construct(self, expr):
            return self._visit_for_construct(expr)

        return _visit_simd_construct

    @classmethod
    def _visit_parallel_for_construct(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_parallel_for_construct(self, expr):
            return self._visit_for_construct(expr)

        return _visit_parallel_for_construct

    @classmethod
    def _visit_OmpEndDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpEndDirective(self, expr):
            clauses = [self._visit(clause) for clause in expr.clauses]
            return OmpEndDirective(name=expr.name, clauses=clauses, raw=expr.raw, parent=expr.parent)

        return _visit_OmpEndDirective

    @classmethod
    def _visit_OmpScalarExpr(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpScalarExpr(self, expr):
            value = self._visit(expr.value)
            if (
                    not hasattr(value, "dtype")
                    or (isinstance(value, FunctionCall) and not value.funcdef.results)
            ):
                errors.report(
                    "expression needs to be a scalar expression",
                    symbol=self,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            return OmpScalarExpr.from_omp_expr(expr, value=value)

        return _visit_OmpScalarExpr

    @classmethod
    def _visit_OmpIntegerExpr(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpIntegerExpr(self, expr):
            value = self._visit(expr.value)
            if not hasattr(value, "dtype") or not isinstance(value.dtype, PythonNativeInt):
                errors.report(
                    "expression must be an integer expression",
                    symbol=self,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            return OmpIntegerExpr.from_omp_expr(expr, value=value)

        return _visit_OmpIntegerExpr

    @classmethod
    def _visit_OmpConstantPositiveInteger(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpConstantPositiveInteger(self, expr):
            value = self._visit(expr.value)
            return OmpConstantPositiveInteger.from_omp_expr(expr, value=value)

        return _visit_OmpConstantPositiveInteger

    @classmethod
    def _visit_OmpList(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpList(self, expr):
            items = tuple(self._visit(var) for var in expr.value)
            for i in items:
                if not isinstance(i, Variable):
                    errors.report(
                        "omp list must be a list of variables",
                        symbol=expr,
                        line=expr.line,
                        column=expr.position[0],
                        severity="fatal",
                    )
            return OmpList.from_omp_expr(expr, value=items)

        return _visit_OmpList

    @classmethod
    def _visit_OmpClause(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _visit_OmpClause(self, expr):
            omp_exprs = [self._visit(e) for e in expr.omp_exprs]
            return OmpClause.from_clause(expr, omp_exprs=omp_exprs)

        return _visit_OmpClause


class CCodePrinter(Commons):
    _version = 4.5
    @classmethod
    def _print_OmpConstruct(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpConstruct(self, expr):
            body = self._print(expr.body)
            if expr.end:
                return f"{self._print(expr.start)}\n{{\n{body}\n}}\n{self._print(expr.end)}\n"
            else:
                return f"{self._print(expr.start)}\n{body}\n"
        return _print_OmpConstruct

    @classmethod
    def _print_OmpDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpDirective(self, expr):
            return f"#pragma omp {expr.raw}\n"
        return _print_OmpDirective

    @classmethod
    def _print_OmpEndDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpEndDirective(self, expr):
            if expr.parent.start.is_construct:
                return ""
            else:
                return f"#pragma omp {expr.raw}\n"
        return _print_OmpEndDirective

class FCodePrinter(Commons):
    _version = 4.5
    @classmethod
    def _helper_delay_clauses_printing(cls, directive, clauses):
        """Transfer clauses of directive to an OmpEndDirective for printing"""
        start = directive.raw
        clauses = [c for c in directive.clauses if c.name in clauses]
        end_raw = f"{directive.name} {' '.join(c.raw for c in clauses)}"
        directive.parent.end = OmpEndDirective(name=directive.name, omp_version=directive.omp_version,
                                               raw=end_raw, parent=directive.parent)
        for c in clauses:
            start = start.replace(c.raw, '')
        return f"!$omp {start}\n"

    @classmethod
    def _print_OmpConstruct(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpConstruct(self, expr):
            if hasattr(self, f"_print_{expr.start.name.replace(' ', '_')}_construct"):
                return getattr(self, f"_print_{expr.start.name.replace(' ', '_')}_construct")(expr)
            body = self._print(expr.body)
            start = self._print(expr.start)
            if expr.end:
                end = self._print(expr.end)
                return f"{start}\n{body}\n{end}\n"
            else:
                return f"{start}\n{body}\n"

        return _print_OmpConstruct

    @classmethod
    def _print_for_construct(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_for_construct(self, expr):
            delayed = ['nowait']
            delayed = [c for c in expr.start.clauses if c.name in delayed]
            if len(delayed):
                end = f"!$omp end do {' '.join(c.raw for c in delayed)}"
            else:
                end = None

            start = re.sub(r'\bfor\b', 'do', expr.start.raw)
            for c in delayed:
                start = start.replace(c.raw, '')
            body = self._print(expr.body)
            if end:
                return f"!$omp {start}\n{body}\n{end}\n"
            else:
                return f"!$omp {start}\n{body}\n"

        return _print_for_construct

    @classmethod
    def _print_OmpDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpDirective(self, expr):
            if hasattr(self, f"_print_{expr.name.replace(' ', '_')}_directive"):
                return getattr(self, f"_print_{expr.name.replace(' ', '_')}_directive")(expr)
            return f"!$omp {expr.raw}\n"

        return _print_OmpDirective

    @classmethod
    def _print_for_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_for_directive(self, expr):
            directive = FCodePrinter._helper_delay_clauses_printing(expr, ['nowait'])
            directive = re.sub(r'\bfor\b', 'do', directive)
            return directive

        return _print_for_directive

    @classmethod
    def _print_single_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_single_directive(self, expr):
            return FCodePrinter._helper_delay_clauses_printing(expr, ['nowait', 'copyprivate'])

        return _print_single_directive

    @classmethod
    def _print_simd_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_simd_directive(self, expr):
            return self._print_for_directive(expr)

        return _print_simd_directive

    @classmethod
    def _print_parallel_for_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_parallel_for_directive(self, expr):
            return self._print_for_directive(expr)

        return _print_parallel_for_directive

    @classmethod
    def _print_parallel_for_simd_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_parallel_for_simd_directive(self, expr):
            return self._print_for_directive(expr)

        return _print_parallel_for_simd_directive

    @classmethod
    def _print_OmpEndDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpEndDirective(self, expr):
            if hasattr(self, f"_print_end_{expr.name.replace(' ', '_')}_directive"):
                return getattr(self, f"_print_end_{expr.name.replace(' ', '_')}_directive")(expr)
            return f"!$omp end {expr.raw}\n"

        return _print_OmpEndDirective

    @classmethod
    def _print_end_section_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_end_section_directive(self, expr):
            return ""

        return _print_end_section_directive

    @classmethod
    def _print_end_for_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_end_for_directive(self, expr):
            directive = f"!$omp end {expr.raw}"
            directive = re.sub(r'\bfor\b', 'do', directive)
            return directive

        return _print_end_for_directive

    @classmethod
    def _print_end_simd_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_end_simd_directive(self, expr):
            return self._print_end_for_directive(expr)

        return _print_end_simd_directive

    @classmethod
    def _print_end_parallel_for_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_end_parallel_for_directive(self, expr):
            return self._print_end_for_directive(expr)

        return _print_end_parallel_for_directive

    @classmethod
    def _print_end_parallel_for_simd_directive(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_end_parallel_for_simd_directive(self, expr):
            return self._print_end_for_directive(expr)

        return _print_end_parallel_for_simd_directive

class PythonCodePrinter(Commons):
    _version = 4.5
    @classmethod
    def _print_OmpConstruct(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpConstruct(self, expr):
            body = self._print(expr.body)
            start = self._print(expr.start)
            if expr.end:
                end = self._print(expr.end)
                return f"{start}\n{body}\n{end}\n"
            else:
                return f"{start}\n{body}\n"
        return _print_OmpConstruct

    @classmethod
    def _print_OmpDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpDirective(self, expr):
            return f"#$ omp {expr.raw}\n"
        return _print_OmpDirective

    @classmethod
    def _print_OmpEndDirective(cls, options, method=None):
        @cls._helper_check_config(options, method)
        def _print_OmpEndDirective(self, expr):
            return f"#$ omp end {expr.raw}\n"
        return _print_OmpEndDirective
