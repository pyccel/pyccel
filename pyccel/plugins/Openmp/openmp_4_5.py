# pylint: disable=protected-access
"""Contains OpenMp 4.5 parser classes"""
import ast
import functools
import re
from os.path import join, dirname

from textx import metamodel_for_language
from textx.metamodel import metamodel_from_file

from pyccel.ast.core import CodeBlock
from pyccel.ast.core import EmptyNode
from pyccel.ast.core import FunctionCall
from pyccel.ast.datatypes import PythonNativeInt
from pyccel.ast.operators import PyccelMinus, PyccelAdd
from pyccel.ast.variable import Variable
from pyccel.errors.errors import Errors
from pyccel.errors.messages import PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX
from pyccel.parser.extend_tree import extend_tree
from pyccel.plugins.Openmp.omp import OmpDirective, OmpClause, OmpEndDirective, OmpConstruct, OmpList, \
    OmpTxDirective, OmpTxEndDirective, OmpTxNode, OmpExpressionList
from pyccel.plugins.Openmp.omp import OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger

errors = Errors()

__all__ = (
    "CCodePrinter",
    "ConfigMixin",
    "FCodePrinter",
    "PythonCodePrinter",
    "SemanticParser",
    "SyntaxParser",
)


class ConfigMixin:
    """
    Common utilities and methods for handling OpenMP syntax and configurations.

    This class provides utility methods for configuring OpenMP functionality
    and handling version-specific features. It serves as a base class for
    parsers and printers that need to be aware of OpenMP configurations.

    See Also
    --------
    SyntaxParser : Parser for OpenMP syntax.
    SemanticParser : Parser for OpenMP semantics.
    CCodePrinter : Printer for OpenMP C code.
    FCodePrinter : Printer for OpenMP Fortran code.
    PythonCodePrinter : Printer for OpenMP Python code.
    """
    _version = None

    @classmethod
    def helper_check_config(cls, func, options, method):
        """
        Check configuration for OpenMP.

        This method wraps OpenMP functions to ensure they are only executed
        when the appropriate OpenMP version is configured. If the version
        doesn't match or OpenMP is not enabled, it falls back to the provided
        method or reports an error.

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

        See Also
        --------
        SyntaxParser.setup : Setup method for configuring the SyntaxParser.

        Examples
        --------
        >>> def my_openmp_func(expr, **kwargs):
        >>>     pass
        >>> options = {'omp_version': 4.5, 'accelerators': ['openmp']}
        >>> wrapped_func = ConfigMixin.helper_check_config(my_openmp_func, options, None)
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """method wrapper to check configuration"""
            if cls._version != options.get('omp_version') or not 'openmp' in options.get('accelerators', []):
                if method:
                    return method(*args[1:], **kwargs)
                else:
                    return errors.report(PYCCEL_RESTRICTION_UNSUPPORTED_SYNTAX, symbol=args[-1],
                                         severity='error')
            return func(*args, **kwargs, cls=cls, method=method)

        return wrapper


class SyntaxParser(ConfigMixin):
    """
    Openmp 4.5 syntax parser.

    This class is responsible for parsing OpenMP 4.5 directives and constructs
    from source code comments. It uses a textX grammar to parse the OpenMP syntax
    and converts it into the appropriate AST nodes.

    See Also
    --------
    ConfigMixin : Base class providing configuration utilities.
    SemanticParser : Parser for OpenMP semantics.
    pyccel.plugins.Openmp.omp.OmpDirective : Class representing an OpenMP directive.
    pyccel.plugins.Openmp.omp.OmpConstruct : Class representing an OpenMP construct.
    """
    _version = 4.5
    _omp_metamodel = None

    @classmethod
    def setup(cls, options, parser):
        """
        Configure the SyntaxParser class and create a metamodel from the grammar.

        This method initializes the OpenMP parser by setting up the metamodel from the grammar file
        and configuring the parser instance. It registers object processors for handling different
        OpenMP versions and directives and prepares the parser for processing OpenMP directives
        in the code.

        Parameters
        ----------
        options : dict
            Configuration options.
        parser : instance of pyccel SyntaxParser
            The parser instance to be configured for OpenMP parsing.

        See Also
        --------
        ConfigMixin.helper_check_config : Method for checking OpenMP configuration.
        pyccel.plugins.Openmp.omp : Module containing OpenMP AST nodes.
        """
        if not cls._omp_metamodel:
            this_folder = dirname(__file__)
            # Get metamodel from language description
            grammar = join(this_folder, "grammar/openmp.tx")
            cls._omp_metamodel = metamodel_from_file(grammar)
            # object processors: are registered for particular classes (grammar rules)
            # and are called when the objects of the given class is instantiated.
            # The rules OMP_X_Y are used to insert the version of the syntax used
            textx_mm = metamodel_for_language('textx')
            grammar_model = textx_mm.grammar_model_from_file(grammar)

            def make_parent_processor(rule):
                """returns a processor that handles allowed parent directives"""
                return lambda _: rule.name.replace('_PARENT', '').lower()

            obj_processors = {r.name: make_parent_processor(r)
                              for r in grammar_model.rules if r.name.endswith('_PARENT')}
            obj_processors.update({
                'OMP_4_5': lambda _: 4.5,
                'OMP_5_0': lambda _: 5.0,
                'OMP_5_1': lambda _: 5.1,
                'TRUE': lambda _: True,
                'OMP_VERSION': lambda _: cls._version,
            })
            cls._omp_metamodel.register_obj_processors(obj_processors)

        parser._skip_stmts_count = 0

    @classmethod
    def _helper_parse_expr(cls, expr):
        """
        Parses an expression and returns the equivalent node.

        This method takes an OpenMP expression and converts it into a Python AST node
        using the extend_tree function. It performs validation to ensure the expression
        is valid and properly structured.

        Parameters
        ----------
        expr : str
            A python expression.

        Returns
        -------
        ast.AST
            The Python AST node equivalent to the input expression.
        """
        fst = extend_tree(expr.value)
        if (
                not isinstance(fst, ast.Module)
                or len(fst.body) != 1
                or not isinstance(fst.body[0], ast.Expr)
        ):
            errors.report(
                "Invalid expression",
                symbol=expr,
                severity="fatal",
            )
        return fst.body[0].value

    @staticmethod
    def _treat_comment_line(instance, line, expr, method, cls=None):
        """
        Parse a comment line.

        Parse a comment which fits in a single line if the comment
        begins with `#$omp` using textx.

        Parameters
        ----------
        instance : object
            The parser instance that is processing the code.
        line : str
            The comment line.
        expr : ast.Ast
            The comment object in the code. This is useful for raising
            errors.
        method : callable
            The fallback method to call if the line is not an OpenMP directive.
        cls : class, optional
            Used to access the configuration and class variables, defaults to None.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The treated object as an Openmp node.

        See Also
        --------
        pyccel.plugins.Openmp.omp.OmpTxDirective : Class representing an OpenMP directive.
        pyccel.plugins.Openmp.omp.OmpTxEndDirective : Class representing an OpenMP end directive.
        """
        if line.startswith('#$') and line[2:].lstrip().startswith('omp'):
            from textx.exceptions import TextXError
            try:
                model = cls._omp_metamodel.model_from_str(line)
                directive = OmpTxEndDirective(model.statement, line, cls._version, lineno=expr.lineno,
                                              column=expr.col_offset) if model.statement.is_end_directive else OmpTxDirective(
                    model.statement, line, cls._version, lineno=expr.lineno, column=expr.col_offset)
                return instance._visit(directive)
            except TextXError as e:
                errors.report(e.message, severity="fatal", symbol=expr)
        return method(line, expr)

    @staticmethod
    def _visit(instance, stmt, method, cls=None):
        """
        Visit a statement and determine if it should be skipped.

        This method processes AST statements and handles OpenMP directives.
        It manages the skipping of statements that are part of OpenMP constructs
        and ensures proper AST node creation.

        Parameters
        ----------
        instance : object
            The parser instance that is processing the code.
        stmt : ast.AST
            The statement to visit.
        method : callable
            The method to call for visiting the statement if it's not skipped.
        cls : class, optional
            The class to use for processing, defaults to None.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The processed node, or an EmptyNode if the statement should be skipped.

        See Also
        --------
        pyccel.ast.core.EmptyNode : Node used for skipped statements.
        """
        if instance._skip_stmts_count:
            instance._skip_stmts_count -= 1
            return EmptyNode()
        else:
            res = method(stmt)
            if isinstance(stmt, OmpTxNode):
                res.set_current_ast(stmt.python_ast)
            return res

    @staticmethod
    def _visit_OmpTxDirective(instance, stmt, cls=None, method=None):
        if hasattr(instance, f"_visit_{stmt.name.replace(' ', '_')}_directive"):
            return getattr(instance, f"_visit_{stmt.name.replace(' ', '_')}_directive")(stmt)
        clauses = tuple(instance._visit(clause) for clause in stmt.clauses)
        directive = OmpDirective(clauses=clauses, **stmt.get_fixed_state())
        if stmt.is_construct:
            body = []
            end = None
            container = None
            for el in instance._context[::-1]:
                if isinstance(el, list):
                    container = el[el.index(instance._context[-2]) + 1:].copy()
                    break
            for line in container:
                expr = instance._visit(line)
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
            instance._skip_stmts_count = len(body) + 1
            body = CodeBlock(body=body)
            return OmpConstruct(start=directive, end=end, body=body)
        return directive

    @staticmethod
    def _visit_for_directive(instance, stmt, cls=None, method=None):
        loop = None
        for el in instance._context[::-1]:
            if isinstance(el, list):
                loop_pos = el.index(instance._context[-2]) + 1
                if len(el) < loop_pos + 1 or not isinstance(el[loop_pos], ast.For):
                    errors.report(
                        f"{stmt.name} directive should be followed by a for loop",
                        symbol=stmt,
                        severity="fatal",
                    )
                loop = instance._visit(el[loop_pos])
                break
        clauses = tuple(instance._visit(clause) for clause in stmt.clauses)
        directive = OmpDirective(clauses=clauses, **stmt.get_fixed_state())
        instance._skip_stmts_count = 1
        body = CodeBlock(body=[loop])
        return OmpConstruct(start=directive, end=None, body=body)

    @staticmethod
    def _visit_simd_directive(instance, expr, cls=None, method=None):
        return instance._visit_for_directive(expr)

    @staticmethod
    def _visit_parallel_for_directive(instance, expr, cls=None, method=None):
        return instance._visit_for_directive(expr)

    @staticmethod
    def _visit_parallel_for_simd_directive(instance, expr, cls=None, method=None):
        return instance._visit_for_directive(expr)

    @staticmethod
    def _visit_target_teams_distribute_parallel_for_directive(instance, expr, cls=None, method=None):
        return instance._visit_for_directive(expr)

    @staticmethod
    def _visit_OmpTxClause(instance, expr, cls=None, method=None):
        omp_exprs = tuple(instance._visit(e) for e in expr.omp_exprs)
        return OmpClause(omp_exprs=omp_exprs, **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpTxEndDirective(instance, expr, cls=None, method=None):
        clauses = tuple(instance._visit(clause) for clause in expr.clauses)
        return OmpEndDirective(clauses=clauses, **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpTxScalarExpr(instance, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpScalarExpr(value=instance._visit(fst), **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpTxIntegerExpr(instance, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpIntegerExpr(value=instance._visit(fst), **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpTxConstantPositiveInteger(instance, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpConstantPositiveInteger(value=instance._visit(fst), **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpTxList(instance, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpList(value=instance._visit(fst), **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpTxExpressionList(instance, expr, cls=None, method=None):
        fst = cls._helper_parse_expr(expr)
        return OmpExpressionList(value=instance._visit(fst), **expr.get_fixed_state())


class SemanticParser(ConfigMixin):
    """
    Openmp 4.5 semantic parser.

    This class is responsible for semantic analysis of OpenMP 4.5 directives
    and constructs. It processes the AST nodes created by the SyntaxParser
    and performs semantic validation and transformations.

    See Also
    --------
    ConfigMixin : Base class providing configuration utilities.
    SyntaxParser : Parser for OpenMP syntax.
    pyccel.plugins.Openmp.omp.OmpDirective : Class representing an OpenMP directive.
    pyccel.plugins.Openmp.omp.OmpConstruct : Class representing an OpenMP construct.
    """
    _version = 4.5

    @staticmethod
    def _visit_OmpDirective(instance, expr, cls=None, method=None):
        clauses = tuple(instance._visit(clause) for clause in expr.clauses)
        directive = OmpDirective(clauses=clauses, **expr.get_fixed_state())
        return directive

    @staticmethod
    def _visit_OmpConstruct(instance, expr, cls=None, method=None):
        if hasattr(instance, f"_visit_{expr.start.name.replace(' ', '_')}_construct"):
            return getattr(instance, f"_visit_{expr.start.name.replace(' ', '_')}_construct")(expr)

        body = instance._visit(expr.body)
        start = instance._visit(expr.start)
        end = instance._visit(expr.end) if expr.end else None
        return OmpConstruct(start=start, end=end, body=body)

    @staticmethod
    def _visit_for_construct(instance, expr, cls=None, method=None):
        body = instance._visit(expr.body)
        start = instance._visit(expr.start)
        return OmpConstruct(start=start, end=None, body=body)

    @staticmethod
    def _visit_simd_construct(instance, expr, cls=None, method=None):
        return instance._visit_for_construct(expr)

    @staticmethod
    def _visit_parallel_for_simd_construct(instance, expr, cls=None, method=None):
        return instance._visit_for_construct(expr)

    @staticmethod
    def _visit_parallel_for_construct(instance, expr, cls=None, method=None):
        return instance._visit_for_construct(expr)

    @staticmethod
    def _visit_target_teams_distribute_parallel_for_construct(instance, expr, cls=None, method=None):
        return instance._visit_for_construct(expr)

    @staticmethod
    def _visit_OmpEndDirective(instance, expr, cls=None, method=None):
        if not isinstance(expr.current_user_node, OmpConstruct) and expr.is_construct:
            errors.report(
                f"End directive `{expr.name}` doesn't belong to any openmp construct",
                symbol=expr,
                severity="error",
            )
        clauses = tuple(instance._visit(clause) for clause in expr.clauses)
        return OmpEndDirective(clauses=clauses, **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpScalarExpr(instance, expr, cls=None, method=None):
        value = instance._visit(expr.value)
        if (
                not hasattr(value, "dtype")
                or (isinstance(value, FunctionCall) and not value.funcdef.results)
        ):
            errors.report(
                "expression needs to be a scalar expression",
                symbol=instance,
                severity="fatal",
            )
        return OmpScalarExpr(value=value, **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpIntegerExpr(instance, expr, cls=None, method=None):
        value = instance._visit(expr.value)
        if not hasattr(value, "dtype") or not isinstance(value.dtype, PythonNativeInt):
            errors.report(
                "expression must be an integer expression",
                symbol=instance,
                severity="fatal",
            )
        return OmpIntegerExpr(value=value, **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpConstantPositiveInteger(instance, expr, cls=None, method=None):
        value = instance._visit(expr.value)
        return OmpConstantPositiveInteger(value=value, **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpList(instance, expr, cls=None, method=None):
        items = tuple(instance._visit(var) for var in expr.value)
        for i in items:
            if not isinstance(i, Variable):
                errors.report(
                    "omp list must be a list of variables",
                    symbol=expr,
                    severity="fatal",
                )
        return OmpList(value=items, **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpExpressionList(instance, expr, cls=None, method=None):
        items = tuple(instance._visit(var) for var in expr.value)
        for i in items:
            if not isinstance(i, Variable) and not isinstance(i, PyccelMinus) and not isinstance(i, PyccelAdd):
                errors.report(
                    "omp list must be a list of expressions",
                    symbol=expr,
                    severity="fatal",
                )
        return OmpExpressionList(value=items, **expr.get_fixed_state())

    @staticmethod
    def _visit_OmpClause(instance, expr, cls=None, method=None):
        omp_exprs = tuple(instance._visit(e) for e in expr.omp_exprs)
        return OmpClause(omp_exprs=omp_exprs, **expr.get_fixed_state())


class CCodePrinter(ConfigMixin):
    """
    Openmp 4.5 C code printer parser.

    This class is responsible for printing OpenMP 4.5 directives and constructs
    as C code. It converts the AST nodes into C syntax according to the OpenMP 4.5
    specification.

    See Also
    --------
    ConfigMixin : Base class providing configuration utilities.
    FCodePrinter : Printer for OpenMP Fortran code.
    PythonCodePrinter : Printer for OpenMP Python code.
    pyccel.plugins.Openmp.omp.OmpDirective : Class representing an OpenMP directive.
    pyccel.plugins.Openmp.omp.OmpConstruct : Class representing an OpenMP construct.
    """
    _version = 4.5

    @staticmethod
    def _print_OmpConstruct(instance, expr, cls=None, method=None):
        body = instance._print(expr.body)
        if expr.end:
            return f"{instance._print(expr.start)}\n{{\n{body}\n}}\n{instance._print(expr.end)}\n"
        else:
            return f"{instance._print(expr.start)}\n{body}\n"

    @staticmethod
    def _print_OmpDirective(instance, expr, cls=None, method=None):
        return f"#pragma omp {expr.raw}\n"

    @staticmethod
    def _print_OmpEndDirective(instance, expr, cls=None, method=None):
        if isinstance(expr.current_user_node, OmpConstruct):
            return ""
        else:
            return f"#pragma omp end {expr.raw}\n"


class FCodePrinter(ConfigMixin):
    """
    Openmp 4.5 fortran code printer parser.

    This class is responsible for printing OpenMP 4.5 directives and constructs
    as Fortran code. It converts the AST nodes into Fortran syntax according to 
    the OpenMP 4.5 specification, handling Fortran-specific features.

    See Also
    --------
    ConfigMixin : Base class providing configuration utilities.
    CCodePrinter : Printer for OpenMP C code.
    PythonCodePrinter : Printer for OpenMP Python code.
    pyccel.plugins.Openmp.omp.OmpDirective : Class representing an OpenMP directive.
    pyccel.plugins.Openmp.omp.OmpConstruct : Class representing an OpenMP construct.
    """
    _version = 4.5

    @classmethod
    def _helper_delay_clauses_printing(cls, start, end, clauses):
        """
        Transfer clauses of a directive to an OmpEndDirective for printing.

        This method handles the special case in Fortran where certain clauses
        need to be moved from the start directive to the end directive for proper
        printing. It modifies the raw representation of the directives accordingly.

        Parameters
        ----------
        start : OmpDirective
            The starting directive of an OpenMP construct.
        end : OmpEndDirective or None
            The ending directive of an OpenMP construct, or None if there is no end directive.
        clauses : list of str
            Names of clauses to be moved from the start-to-end directive.

        Returns
        -------
        tuple
            A tuple containing (modified_start_string, modified_end_string).

        See Also
        --------
        FCodePrinter._print_for_construct : Method that uses this helper.
        FCodePrinter._print_single_construct : Method that uses this helper.

        Examples
        --------
        >>> start = OmpDirective(clauses=[OmpClause(name='nowait')])
        >>> end = None
        >>> start_str, end_str = FCodePrinter._helper_delay_clauses_printing(start, end, ['nowait'])
        >>> print(start_str)
        !$omp 
        >>> print(end_str)
        !$omp end nowait
        """
        clauses = tuple(c for c in start.clauses if c.name in clauses)
        if clauses or end:
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
    def _print_OmpConstruct(instance, expr, cls=None, method=None):
        if hasattr(instance, f"_print_{expr.start.name.replace(' ', '_')}_construct"):
            return getattr(instance, f"_print_{expr.start.name.replace(' ', '_')}_construct")(expr)
        body = instance._print(expr.body)
        start = instance._print(expr.start)
        end = instance._print(expr.end)
        return f"{start}\n{body}\n{end}\n"

    @staticmethod
    def _print_for_construct(instance, expr, cls=None, method=None):
        start, end = cls._helper_delay_clauses_printing(expr.start, expr.end, ['nowait'])
        start = re.sub(r'\bfor\b', 'do', start)
        body = instance._print(expr.body)
        if end:
            end = re.sub(r'\bfor\b', 'do', end)
            return f"{start}\n{body}\n{end}\n"
        else:
            return f"{start}\n{body}\n"

    @staticmethod
    def _print_single_construct(instance, expr, cls=None, method=None):
        start, end = cls._helper_delay_clauses_printing(expr.start, expr.end, ['nowait', 'copyprivate'])
        body = instance._print(expr.body)
        return f"{start}\n{body}\n{end}\n"

    @staticmethod
    def _print_simd_construct(instance, expr, cls=None, method=None):
        return instance._print_for_construct(expr)

    @staticmethod
    def _print_parallel_for_construct(instance, expr, cls=None, method=None):
        return instance._print_for_construct(expr)

    @staticmethod
    def _print_parallel_for_simd_construct(instance, expr, cls=None, method=None):
        return instance._print_for_construct(expr)

    @staticmethod
    def _print_target_teams_distribute_parallel_for_construct(instance, expr, cls=None, method=None):
        return instance._print_for_construct(expr)

    @staticmethod
    def _print_OmpDirective(instance, expr, cls=None, method=None):
        return f"!$omp {expr.raw}\n"

    @staticmethod
    def _print_OmpEndDirective(instance, expr, cls=None, method=None):
        if hasattr(instance, f"_print_end_{expr.name.replace(' ', '_')}_directive"):
            return getattr(instance, f"_print_end_{expr.name.replace(' ', '_')}_directive")(expr)
        return f"!$omp end {expr.raw}\n"

    @staticmethod
    def _print_end_section_directive(instance, expr, cls=None, method=None):
        return ""


class PythonCodePrinter(ConfigMixin):
    """
    Openmp 4.5 python code printer parser.

    This class is responsible for printing OpenMP 4.5 directives and constructs
    as Python code comments.

    See Also
    --------
    ConfigMixin : Base class providing configuration utilities.
    CCodePrinter : Printer for OpenMP C code.
    FCodePrinter : Printer for OpenMP Fortran code.
    pyccel.plugins.Openmp.omp.OmpDirective : Class representing an OpenMP directive.
    pyccel.plugins.Openmp.omp.OmpConstruct : Class representing an OpenMP construct.
    """
    _version = 4.5

    @staticmethod
    def _print_OmpConstruct(instance, expr, cls=None, method=None):
        body = instance._print(expr.body)
        start = instance._print(expr.start)
        if expr.end:
            end = instance._print(expr.end)
            return f"{start}\n{body}\n{end}\n"
        else:
            return f"{start}\n{body}\n"

    @staticmethod
    def _print_OmpDirective(instance, expr, cls=None, method=None):
        return f"#$ omp {expr.raw}\n"

    @staticmethod
    def _print_OmpEndDirective(instance, expr, cls=None, method=None):
        return f"#$ omp end {expr.raw}\n"
