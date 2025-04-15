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
from pyccel.extensions.Openmp.omp import OmpDirective, OmpClause, OmpEndDirective, OmpConstruct, OmpExpr, OmpList
from pyccel.extensions.Openmp.omp import OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger

errors = Errors()


class SyntaxParser:
    """Openmp 4.5 syntax parser"""

    def __init__(self):
        this_folder = dirname(__file__)
        # Get metamodel from language description
        grammar = join(this_folder, "grammar/openmp.tx")
        omp_classes = [OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger, OmpList]
        self._omp_metamodel = metamodel_from_file(grammar, classes=omp_classes)

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
        self._omp_metamodel.register_obj_processors(obj_processors)
        self._directives_stack = []
        self._structured_blocks_stack = []
        self._skip_stmts_count = 0

    @property
    def directives_stack(self):
        """Returns a list of directives that waits for an end directive"""
        return self._directives_stack


    def post_parse_checks(self):
        """Performs every post syntactic parsing checks"""
        if any(pd.require_end_directive for pd in self._directives_stack):
            errors.report(
                "directives need closing",
                symbol=self._directives_stack,
                severity="fatal",
            )
    def _treat_comment_line(self, line, expr):
        from textx.exceptions import TextXError
        try:
            model = self._omp_metamodel.model_from_str(line)
            model.raw = line
            directive = OmpDirective.from_tx_directive(model.statement)
            directive.set_current_ast(expr)
            return self._visit(directive)

        except TextXError as e:
            errors.report(e.message, severity="fatal", symbol=expr)
            return None

    def _visit_CommentLine(self, expr):
        from textx.exceptions import TextXError
        try:
            model = self._omp_metamodel.model_from_str(expr.s)
            model.raw = expr.s
            directive = OmpDirective.from_tx_directive(model.statement)
            directive.set_current_ast(expr)
            return self._visit(directive)

        except TextXError as e:
            errors.report(e.message, severity="fatal", symbol=expr)
            return None

    def _visit_OmpDirective(self, stmt):
        if hasattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive"):
            return getattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive")(stmt)
        clauses = [self._visit(clause) for clause in stmt.clauses]
        directive = OmpDirective.from_directive(stmt, clauses=clauses)
        if stmt.require_end_directive:
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

    def _visit_simd_directive(self, expr):
        return self._visit_for_directive(expr)

    def _visit_parallel_for_directive(self, expr):
        return self._visit_for_directive(expr)

    def _visit_OmpClause(self, expr):
        omp_exprs = [self._visit(e) for e in expr.omp_exprs]
        return OmpClause.from_clause(expr, omp_exprs=omp_exprs)

    def _visit_OmpEndDirective(self, expr):
        clauses = [self._visit(clause) for clause in expr.clauses]
        end = OmpEndDirective.from_directive(expr, clauses=clauses)
        return end

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

class SemanticParser:
    """Openmp 4.5 semantic parser"""

    def __init__(self):
        self._omp_reserved_nodes = []

    @property
    def omp_reserved_nodes(self):
        return self._omp_reserved_nodes

    def _visit(self, expr):
        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                obj = getattr(self, annotation_method)(expr)
                return obj
        # Unknown object, we ignore semantics, this is useful to isolate the omp parser for testing.
        return expr

    def _visit_OmpDirective(self, expr):
        if hasattr(self, f"_visit_{expr.name.replace(' ', '_')}_directive"):
            return getattr(self, f"_visit_{expr.name.replace(' ', '_')}_directive")(expr)
        clauses = [self._visit(clause) for clause in expr.clauses]
        directive = OmpDirective.from_directive(expr, clauses=clauses)
        return directive

    def _visit_OmpConstruct(self, expr):
        if hasattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct"):
            return getattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct")(expr)

        body = self._visit(expr.body)
        start = self._visit(expr.start)
        end = self._visit(expr.end) if expr.end else None
        return OmpConstruct(start=start, end=end, body=body)

    def _visit_for_construct(self, expr):
        body = self._visit(expr.body)
        start = self._visit(expr.start)
        return OmpConstruct(start=start, end=None, body=body)

    def _visit_simd_construct(self, expr):
        return self._visit_for_construct(expr)

    def _visit_parallel_for_construct(self, expr):
        return self._visit_for_construct(expr)

    def _visit_OmpEndDirective(self, expr):
        clauses = [self._visit(clause) for clause in expr.clauses]
        return OmpEndDirective(name=expr.name, clauses=clauses, raw=expr.raw, parent=expr.parent)

    def _visit_OmpScalarExpr(self, expr):
        value = self._visit(expr.value)
        if (
            not hasattr(value, "dtype")
            #or isinstance(value.dtype, NativeVoid)
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

    def _visit_OmpConstantPositiveInteger(self, expr):
        value = self._visit(expr.value)
        return OmpConstantPositiveInteger.from_omp_expr(expr, value=value)

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

    def _visit_OmpClause(self, expr):
        omp_exprs = [self._visit(e) for e in expr.omp_exprs]
        return OmpClause.from_clause(expr, omp_exprs=omp_exprs)


class CCodePrinter:

    def _print_OmpConstruct(self, expr):
        body = self._print(expr.body)
        if expr.end:
            return f"{self._print(expr.start)}\n{{\n{body}\n}}\n{self._print(expr.end)}\n"
        else:
            return f"{self._print(expr.start)}\n{body}\n"

    def _print_OmpDirective(self, expr):
        return f"#pragma omp {expr.raw}\n"

    def _print_OmpEndDirective(self, expr):
        if expr.parent.start.require_end_directive:
            return ""
        else:
            return f"#pragma omp {expr.raw}\n"

class FCodePrinter:
    def _delay_clauses_printing(self, directive, clauses):
        """Transfer clauses of directive to an OmpEndDirective for printing"""
        start = directive.raw
        clauses = [c for c in directive.clauses if c.name in clauses]
        end_raw = f"{directive.name} {' '.join(c.raw for c in clauses)}"
        directive.parent.end = OmpEndDirective(name=directive.name, omp_version=directive.omp_version,
                                               raw=end_raw, parent=directive.parent)
        for c in clauses:
            start = start.replace(c.raw, '')
        return f"!$omp {start}\n"

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


    def _print_OmpDirective(self, expr):
        if hasattr(self, f"_print_{expr.name.replace(' ', '_')}_directive"):
            return getattr(self, f"_print_{expr.name.replace(' ', '_')}_directive")(expr)
        return f"!$omp {expr.raw}\n"

    def _print_for_directive(self, expr):
        directive = self._delay_clauses_printing(expr, ['nowait'])
        directive = re.sub(r'\bfor\b', 'do', directive)
        return directive

    def _print_single_directive(self, expr):
        return self._delay_clauses_printing(expr, ['nowait', 'copyprivate'])

    def _print_simd_directive(self, expr):
        return self._print_for_directive(expr)

    def _print_parallel_for_directive(self, expr):
        return self._print_for_directive(expr)

    def _print_parallel_for_simd_directive(self, expr):
        return self._print_for_directive(expr)

    def _print_OmpEndDirective(self, expr):
        if hasattr(self, f"_print_end_{expr.name.replace(' ', '_')}_directive"):
            return getattr(self, f"_print_end_{expr.name.replace(' ', '_')}_directive")(expr)
        return f"!$omp end {expr.raw}\n"

    def _print_end_section_directive(self, expr):
        return ""

    def _print_end_for_directive(self, expr):
        directive = f"!$omp end {expr.raw}"
        directive = re.sub(r'\bfor\b', 'do', directive)
        return directive

    def _print_end_simd_directive(self, expr):
        return self._print_end_for_directive(expr)

    def _print_end_parallel_for_directive(self, expr):
        return self._print_end_for_directive(expr)

    def _print_end_parallel_for_simd_directive(self, expr):
        return self._print_end_for_directive(expr)

class PythonCodePrinter:
    def _print_OmpConstruct(self, expr):
        body = self._print(expr.body)
        start = self._print(expr.start)
        if expr.end:
            end = self._print(expr.end)
            return f"{start}\n{body}\n{end}\n"
        else:
            return f"{start}\n{body}\n"

    def _print_OmpDirective(self, expr):
        return f"#$ omp {expr.raw}\n"

    def _print_OmpEndDirective(self, expr):
        return f"#$ omp end {expr.raw}\n"