# pylint: disable=protected-access, missing-function-docstring
"""
Module containing the grammar rules for the OpenMP 4.5 specification,
"""
import ast
import re
from os.path import join, dirname
import functools
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

class ConfigMixin:
    """
    Common utilities and methods for handling OpenMP syntax and configurations.
    """
    _version = None
    _method_registry = {}
    @classmethod
    def helper_check_config(cls, func, options, method):

        """
        Check configuration for OpenMP and register methods.

        Parameters
        ----------
        func : callable
            The Openmp function to be wrapped.
        options : dict
            Configuration options for OpenMP.
        method : callable, or None
            Default method to handle unsupported expressions, could be a pyccel method,
            or another plugin's method.
        
        Returns
        -------
        callable
            Wrapped function that adheres to the configuration.
            Or original method if clear is passed in the options.
        """
        if options.get('clear', False):
            return cls._method_registry[func.__name__]
        if not func.__name__ in cls._method_registry:
            cls._method_registry[func.__name__] = method
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if cls._version != options.get('omp_version', None) or not 'openmp' in options.get('accelerators'):
                if method:
                    return method(*args, **kwargs)
                else:
                    return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX, symbol=args[-1],
                                         severity='error')
            return func(*args, **kwargs, cls=cls, method=method)
        return wrapper

class SyntaxParser(ConfigMixin):
    """Openmp 4.5 syntax parser"""
    _version = 4.5
    @classmethod
    def setup(cls, options, method=None):
        """
        Setup method for configuring the SyntaxParser Class.
        Creates a metamodel from the grammar.

        Parameters
        ----------
        options : dict
            Configuration options.
        method : callable, optional
            An other setup method from a previous plugin if existing.

        Returns
        -------
        callable
            A wrapped setup method that adheres to the configuration and initializes
            the parsing environment.
        """
        if options.get('clear', False):
            return cls._method_registry[method.__name__]
        if not method.__name__ in cls._method_registry:
            cls._method_registry[method.__name__] = method
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
        @functools.wraps(method)
        def setup(instance, *args, **kwargs):
            if cls._version != options.get('omp_version', None) or not 'openmp' in options.get('accelerators'):
                method(instance, *args, **kwargs)
                return
            instance._skip_stmts_count = 0
            method(instance, *args, **kwargs)
        return setup
    @staticmethod
    def _treat_comment_line(instance, line, expr, cls=None, method=None):
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
        if line.startswith('#$') and line[2:].lstrip().startswith('omp'):
            from textx.exceptions import TextXError
            try:
                model = cls._omp_metamodel.model_from_str(line)
                model.raw = line
                directive = OmpDirective.from_tx_directive(model.statement, cls._version)
                directive.set_current_ast(expr)
                return instance._visit(directive)
            except TextXError as e:
                errors.report(e.message, severity="fatal", symbol=expr)
                return None
        elif method:
            return method(instance, line, expr)

    @staticmethod
    def _visit(instance, stmt, cls=None, method=None):
        """Visit a statement and determine if it should be skipped."""
        if instance._skip_stmts_count:
            instance._skip_stmts_count -= 1
            return EmptyNode()
        else:
            return method(instance, stmt)

    @staticmethod
    def _visit_OmpDirective(self, stmt, cls=None, method=None):
        if hasattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive"):
            return getattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive")(stmt)
        clauses = tuple(self._visit(clause) for clause in stmt.clauses)
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

    @staticmethod
    def _visit_for_directive(self, stmt, cls=None, method=None):
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
        clauses = tuple(self._visit(clause) for clause in stmt.clauses)
        directive = OmpDirective.from_directive(stmt, clauses=clauses)
        self._skip_stmts_count = 1
        body = CodeBlock(body=[loop])
        return OmpConstruct(start=directive, end=None, body=body)

    @staticmethod
    def _visit_simd_directive(self, expr, cls=None, method=None):
        return self._visit_for_directive(expr)

    @staticmethod
    def _visit_parallel_for_directive(self, expr, cls=None, method=None):
        return self._visit_for_directive(expr)

    @staticmethod
    def _visit_parallel_for_simd_directive(self, expr, cls=None, method=None):
        return self._visit_for_directive(expr)

    @staticmethod
    def _visit_target_teams_distribute_parallel_for_directive(self, expr, cls=None, method=None):
        return self._visit_for_directive(expr)

    @staticmethod
    def _visit_OmpClause(self, expr, cls=None, method=None):
        omp_exprs = tuple(self._visit(e) for e in expr.omp_exprs)
        return OmpClause.from_clause(expr, omp_exprs=omp_exprs)

    @staticmethod
    def _visit_OmpEndDirective(self, expr, cls=None, method=None):
        clauses = [self._visit(clause) for clause in expr.clauses]
        end = OmpEndDirective.from_directive(expr, clauses=clauses)
        return end

    @classmethod
    def _helper_parse_expr(cls, expr):
        """Parses an expression and returns the equivalent node."""
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
        return fst.body[0].value

    @staticmethod
    def _visit_OmpScalarExpr(self, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpScalarExpr.from_omp_expr(expr, value=self._visit(fst))

    @staticmethod
    def _visit_OmpIntegerExpr(self, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpIntegerExpr.from_omp_expr(expr, value=self._visit(fst))

    @staticmethod
    def _visit_OmpConstantPositiveInteger(self, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpConstantPositiveInteger.from_omp_expr(expr, value=self._visit(fst))

    @staticmethod
    def _visit_OmpList(self, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpList.from_omp_expr(expr, value=self._visit(fst))

    @staticmethod
    def _visit_OmpExpr(self, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpExpr.from_omp_expr(expr, value=self._visit(fst))


class SemanticParser(ConfigMixin):
    """Openmp 4.5 semantic parser"""
    _version = 4.5
    @staticmethod
    def _visit_OmpDirective(self, expr, cls=None, method=None):
        if hasattr(self, f"_visit_{expr.name.replace(' ', '_')}_directive"):
            return getattr(self, f"_visit_{expr.name.replace(' ', '_')}_directive")(expr)
        clauses = tuple(self._visit(clause) for clause in expr.clauses)
        directive = OmpDirective.from_directive(expr, clauses=clauses)
        return directive

    @staticmethod
    def _visit_OmpConstruct(self, expr, cls=None, method=None):
        if hasattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct"):
            return getattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct")(expr)

        body = self._visit(expr.body)
        start = self._visit(expr.start)
        end = self._visit(expr.end) if expr.end else None
        return OmpConstruct(start=start, end=end, body=body)

    @staticmethod
    def _visit_for_construct(self, expr, cls=None, method=None):
        body = self._visit(expr.body)
        start = self._visit(expr.start)
        return OmpConstruct(start=start, end=None, body=body)

    @staticmethod
    def _visit_simd_construct(self, expr, cls=None, method=None):
        return self._visit_for_construct(expr)

    @staticmethod
    def _visit_parallel_for_simd_construct(self, expr, cls=None, method=None):
        return self._visit_for_construct(expr)

    @staticmethod
    def _visit_parallel_for_construct(self, expr, cls=None, method=None):
        return self._visit_for_construct(expr)

    @staticmethod
    def _visit_target_teams_distribute_parallel_for_construct(self, expr, cls=None, method=None):
        return self._visit_for_construct(expr)

    @staticmethod
    def _visit_OmpEndDirective(self, expr, cls=None, method=None):
        clauses = [self._visit(clause) for clause in expr.clauses]
        return OmpEndDirective(name=expr.name, clauses=clauses, raw=expr.raw, parent=expr.parent)

    @staticmethod
    def _visit_OmpScalarExpr(self, expr, cls=None, method=None):
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

    @staticmethod
    def _visit_OmpIntegerExpr(self, expr, cls=None, method=None):
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

    @staticmethod
    def _visit_OmpConstantPositiveInteger(self, expr, cls=None, method=None):
        value = self._visit(expr.value)
        return OmpConstantPositiveInteger.from_omp_expr(expr, value=value)

    @staticmethod
    def _visit_OmpList(self, expr, cls=None, method=None):
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

    @staticmethod
    def _visit_OmpClause(self, expr, cls=None, method=None):
        omp_exprs = tuple(self._visit(e) for e in expr.omp_exprs)
        return OmpClause.from_clause(expr, omp_exprs=omp_exprs)


class CCodePrinter(ConfigMixin):
    """Openmp 4.5 C code printer parser"""
    _version = 4.5
    @staticmethod
    def _print_OmpConstruct(self, expr, cls=None, method=None):
        body = self._print(expr.body)
        if expr.end:
            return f"{self._print(expr.start)}\n{{\n{body}\n}}\n{self._print(expr.end)}\n"
        else:
            return f"{self._print(expr.start)}\n{body}\n"

    @staticmethod
    def _print_OmpDirective(self, expr, cls=None, method=None):
        return f"#pragma omp {expr.raw}\n"

    @staticmethod
    def _print_OmpEndDirective(self, expr, cls=None, method=None):
        if expr.parent.start.is_construct:
            return ""
        else:
            return f"#pragma omp {expr.raw}\n"


class FCodePrinter(ConfigMixin):
    """Openmp 4.5 fortran code printer parser"""
    _version = 4.5
    @classmethod
    def _helper_delay_clauses_printing(cls, start, end, clauses):
        """Transfer clauses of directive to an OmpEndDirective for printing"""
        clauses = tuple(c for c in start.clauses if c.name in clauses)
        if len(clauses) or end:
            if end:
                end = f"!$omp end {end.name} {' '.join(c.raw for c in end.clauses + clauses)}"
            else:
                end = f"!$omp end {start.name} {' '.join(c.raw for c in clauses)}"
        start = start.raw
        for c in clauses:
            start = start.replace(c.raw, '', 1)
        start = f"!$omp {start}\n"
        return start, end

    @staticmethod
    def _print_OmpConstruct(self, expr, cls=None, method=None):
        if hasattr(self, f"_print_{expr.start.name.replace(' ', '_')}_construct"):
            return getattr(self, f"_print_{expr.start.name.replace(' ', '_')}_construct")(expr)
        body = self._print(expr.body)
        start = self._print(expr.start)
        if expr.end:
            end = self._print(expr.end)
            return f"{start}\n{body}\n{end}\n"
        else:
            return f"{start}\n{body}\n"

    @staticmethod
    def _print_for_construct(self, expr, cls=None, method=None):
        start, end = cls._helper_delay_clauses_printing(expr.start, expr.end, ['nowait'])
        start = re.sub(r'\bfor\b', 'do', start)
        body = self._print(expr.body)
        if end:
            end = re.sub(r'\bfor\b', 'do', end)
            return f"{start}\n{body}\n{end}\n"
        else:
            return f"{start}\n{body}\n"

    @staticmethod
    def _print_single_construct(self, expr, cls=None, method=None):
        start, end = cls._helper_delay_clauses_printing(expr.start, expr.end, ['nowait', 'copyprivate'])
        body = self._print(expr.body)
        if end:
            return f"{start}\n{body}\n{end}\n"
        else:
            return f"{start}\n{body}\n"

    @staticmethod
    def _print_simd_construct(self, expr, cls=None, method=None):
        return self._print_for_construct(expr)

    @staticmethod
    def _print_parallel_for_construct(self, expr, cls=None, method=None):
        return self._print_for_construct(expr)

    @staticmethod
    def _print_parallel_for_simd_construct(self, expr, cls=None, method=None):
        return self._print_for_construct(expr)

    @staticmethod
    def _print_target_teams_distribute_parallel_for_construct(self, expr, cls=None, method=None):
        return self._print_for_construct(expr)

    @staticmethod
    def _print_OmpDirective(self, expr, cls=None, method=None):
        if hasattr(self, f"_print_{expr.name.replace(' ', '_')}_directive"):
            return getattr(self, f"_print_{expr.name.replace(' ', '_')}_directive")(expr)
        return f"!$omp {expr.raw}\n"

    @staticmethod
    def _print_OmpEndDirective(self, expr, cls=None, method=None):
        if hasattr(self, f"_print_end_{expr.name.replace(' ', '_')}_directive"):
            return getattr(self, f"_print_end_{expr.name.replace(' ', '_')}_directive")(expr)
        return f"!$omp end {expr.raw}\n"

    @staticmethod
    def _print_end_section_directive(self, expr, cls=None, method=None):
        return ""


class PythonCodePrinter(ConfigMixin):
    """Openmp 4.5 python code printer parser"""
    _version = 4.5
    @staticmethod
    def _print_OmpConstruct(self, expr, cls=None, method=None):
        body = self._print(expr.body)
        start = self._print(expr.start)
        if expr.end:
            end = self._print(expr.end)
            return f"{start}\n{body}\n{end}\n"
        else:
            return f"{start}\n{body}\n"

    @staticmethod
    def _print_OmpDirective(self, expr, cls=None, method=None):
        return f"#$ omp {expr.raw}\n"

    @staticmethod
    def _print_OmpEndDirective(self, expr, cls=None, method=None):
        return f"#$ omp end {expr.raw}\n"
