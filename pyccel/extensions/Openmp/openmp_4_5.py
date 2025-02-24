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
from pyccel.ast.core import CodeBlock, For
from pyccel.ast.core import EmptyNode
from pyccel.errors.errors import Errors
from pyccel.extensions.Openmp.ast import OmpDirective, OmpClause, OmpEndDirective, OmpConstruct, OmpExpr, OmpList
from pyccel.extensions.Openmp.ast import OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger, OmpAnnotatedComment
from pyccel.ast.basic import PyccelAstNode

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
        self._pending_directives = []
        self._pending_constructs = []
        self._bodies = []

    @property
    def pending_directives(self):
        """Returns a list of directives that waits for an end directive"""
        return self._pending_directives

    def post_parse_checks(self):
        """Performs every post syntactic parsing checks"""
        if any(pd.require_end_directive for pd in self.pending_directives):
            errors.report(
                "directives need closing",
                symbol=self._pending_directives,
                severity="fatal",
            )

    def _visit(self, stmt):
        cls = type(stmt)
        syntax_method = '_visit_' + cls.__name__
        if hasattr(self, syntax_method):
            self._context.append(stmt)
            result = getattr(self, syntax_method)(stmt)
            if isinstance(result, PyccelAstNode) and result.python_ast is None and isinstance(stmt, ast.AST):
                result.set_current_ast(stmt)
            #If Directive is a Construct the wen still  need it in the context to gather its body
            if not isinstance(result, OmpDirective) or not result.require_end_directive:
                self._context.pop()
            #Pop the construct's directive
            if isinstance(result, OmpConstruct):
                self._context.pop()
            return result
        # Unknown object, we ignore syntactic, this is useful to isolate the omp parser for testing.
        return stmt
    
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

    def _visit_OmpDirective(self, expr):
        clauses = [self._visit(clause) for clause in expr.clauses]
        directive = OmpDirective.from_directive(expr, clauses=clauses)
        self._pending_directives.append(directive)
        if expr.require_end_directive:
            self._pending_constructs.append(directive)
            self._bodies.append([])
        return directive

    def _visit_OmpClause(self, expr):
        omp_exprs = [self._visit(e) for e in expr.omp_exprs]
        return OmpClause.from_clause(expr, omp_exprs=omp_exprs)

    def _find_coresponding_directive(self, end_directive):
        """Takes an end directive and checks in the pending directives for the one that matches it"""
        cor_directive = None
        if len(self._pending_directives) == 0:
            errors.report(
                f"`end {end_directive.name}` misplaced",
                symbol=end_directive,
                severity="fatal",
            )
        if end_directive.name != self._pending_directives[-1].name:
            if not self._pending_directives[-1].require_end_directive:
                self._pending_directives.pop()
                cor_directive = self._find_coresponding_directive(end_directive)
            else:
                errors.report(
                    f"`end {end_directive.name}` misplaced",
                    symbol=end_directive,
                    severity="fatal",
                )
        else:
            cor_directive = self._pending_directives.pop()
        return cor_directive

    def _visit_OmpEndDirective(self, expr):
        directive = self._find_coresponding_directive(expr)
        clauses = [self._visit(clause) for clause in expr._clauses]
        res = OmpEndDirective(name=expr.name, clauses=clauses, coresponding_directive=directive,
                               raw=expr.raw, parent=expr.parent)

        if self._pending_constructs[-1] is directive:
            self._pending_constructs.pop()
            body = self._bodies.pop()
            body = CodeBlock(body=body)
            return OmpConstruct(start=directive, end=expr, body=body)

        return res

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
        end_directives = [dirve for dirve in expr.get_all_user_nodes() if isinstance(dirve, OmpEndDirective)]
        if len(end_directives) == 0:
           return directive
        assert len(end_directives) == 1
        end_dir = end_directives[0]
        end_dir.substitute(end_dir.coresponding_directive, EmptyNode())
        if end_dir.get_all_user_nodes() != expr.get_all_user_nodes():
            errors.report(
                f"`end {end_dir.name}` directive misplaced",
                symbol=end_dir,
                severity="fatal",
            )
        container = expr.get_all_user_nodes()[-1]
        reserved_nodes = container.body[container.body.index(expr) + 1:container.body.index(end_dir) + 1]
        end_dir = self._visit(reserved_nodes[-1])
        body = self._visit(CodeBlock(reserved_nodes[:-1]))
        self._omp_reserved_nodes = reserved_nodes
        return OmpConstruct(start=directive, end=end_dir, body=body)

    def _visit_OmpConstruct(self, expr):
        if hasattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct"):
            return getattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct")(expr)

        clauses = [self._visit(clause) for clause in expr.start.clauses]
        start = OmpDirective.from_directive(expr.start, clauses=clauses)
        expr.end.substitute(expr.start.coresponding_directive, EmptyNode())
        if expr.end.get_all_user_nodes() != expr.start.get_all_user_nodes():
            errors.report(
                f"`end {expr.end.name}` directive misplaced",
                symbol=expr.end,
                severity="fatal",
            )
        end = self._visit(expr.end)
        body = self._visit(expr.body)
        return OmpConstruct(start=start, end=end, body=body)

    def _visit_for_directive(self, expr):
        clauses = [self._visit(clause) for clause in expr._clauses]
        directive = OmpDirective.from_directive(expr, clauses=clauses)
        end_directives = [dirve for dirve in expr.get_all_user_nodes() if isinstance(dirve, OmpEndDirective)]
        container = expr.get_user_nodes(CodeBlock)[0]
        if not isinstance(container.body[container.body.index(expr) + 1], For):
            errors.report(
                f"{expr.name} directive should be followed by a for loop",
                symbol=expr,
                severity="fatal",
            )
        if len(end_directives) == 1:
            end_dir = end_directives[0]
            end_dir.substitute(end_dir.coresponding_directive, EmptyNode())
            if end_dir.get_all_user_nodes() != expr.get_all_user_nodes():
                errors.report(
                    f"`end {end_dir.name}` directive misplaced",
                    symbol=end_dir,
                    severity="fatal",
                )
            if not isinstance(container.body[container.body.index(expr) + 2], OmpEndDirective):
                errors.report(
                    f"`end {end_dir.name}` directive misplaced",
                    symbol=end_dir,
                    severity="fatal",
                )
            reserved_nodes = container.body[container.body.index(expr) + 1:container.body.index(end_dir) + 1]
            end_dir = self._visit(reserved_nodes[-1])
            body = self._visit(CodeBlock(reserved_nodes[:-1]))
            self._omp_reserved_nodes = reserved_nodes
        else:
            end_dir = None
            reserved_nodes = container.body[container.body.index(expr) + 1:container.body.index(expr) + 2]
            body = self._visit(CodeBlock(reserved_nodes))
            self._omp_reserved_nodes = reserved_nodes

        return OmpConstruct(start=directive, end=end_dir, body=body)

    def _visit_simd_directive(self, expr):
        return self._visit_for_directive(expr)

    def _visit_parallel_for_directive(self, expr):
        return self._visit_for_directive(expr)

    def _visit_OmpEndDirective(self, expr):
        if expr.coresponding_directive is None:
            errors.report(
                f"{expr.name} does not match any directive",
                symbol=expr,
                severity="fatal",
            )
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
        if expr.end and expr.start.require_end_directive:
            return f"{self._print(expr.start)}\n{{\n{body}\n}}\n{self._print(expr.end)}\n"
        elif expr.end:
            return f"{self._print(expr.start)}\n{body}\n{self._print(expr.end)}\n"
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
        body = self._print(expr.body)
        start = self._print(expr.start)
        if expr.end:
            end = self._print(expr.end)
            return f"{start}\n{body}\n{end}\n"
        else:
            return f"{start}\n{body}\n"

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