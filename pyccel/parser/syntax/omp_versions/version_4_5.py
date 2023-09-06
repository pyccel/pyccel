# pylint: disable=protected-access, missing-function-docstring
"""
Module containing the grammar rules for the OpenMP 4.5 specification,
"""
from inspect import isclass
from abc import ABC, abstractmethod
import ast as Ast
from pyccel.ast.core import FunctionCall, Return
from pyccel.ast.variable import Variable
from pyccel.parser.extend_tree import extend_tree
from pyccel.ast.datatypes import NativeInteger, NativeVoid
from pyccel.ast.core import CodeBlock
from pyccel.ast.basic  import Basic
from pyccel.ast.core import EmptyNode
from pyccel.errors.errors import Errors

__all__ = ("Openmp",)


errors = Errors()

class Openmp:
    """Class that groups all the OpenMP classes for constructs and clauses."""

    @classmethod
    def inner_classes_list(cls):
        """
        Return a list of all the classes that will be used to parse the OpenMP code.
        """
        results = []
        for attrname in dir(cls):
            obj = getattr(cls, attrname)
            if isclass(obj) and hasattr(obj, '_used_in_grammar'):
                results.append(obj)
        return results


    class OmpAnnotatedComment(Basic):

        """Represents an OpenMP Annotated Comment in the code."""

        __slots__ = ("VERSION", "DEPRECATED")
        _attribute_nodes = ()
        _current_omp_version = None

        def __init__(self, **kwargs):
            if self._current_omp_version is None:
                raise NotImplementedError(
                    "OpenMP version not set (use OmpAnnotatedComment.set_current_version)"
                )
            self.VERSION = float(kwargs.pop("VERSION", "0") or "0")
            self.DEPRECATED = float(kwargs.pop("DEPRECATED", "inf") or "inf")
            if self.version > self._current_omp_version:
                raise NotImplementedError(
                    f"Syntax not supported in OpenMP version {self._current_omp_version}"
                )
            if self.deprecated <= self._current_omp_version:
                raise NotImplementedError(
                    f"Syntax deprecated in OpenMP version {self.DEPRECATED}"
                )
            super().__init__()

        @property
        def version(self):
            """Returns the version of OpenMP syntax used."""
            return self.VERSION

        @property
        def deprecated(self):
            """Returns the deprecated version of OpenMP syntax used."""
            return self.DEPRECATED

        @classmethod
        def set_current_version(cls, version):
            """Sets the version of OpenMP syntax to support."""
            cls._current_omp_version = version

    class Omp(OmpAnnotatedComment):
        """Represents a holder for all OpenMP statements."""

        __slots__ = ("_statements",)
        _used_in_grammar = True
        def __init__(self, **kwargs):
            self._statements = kwargs.pop("statements", [])
            super().__init__(**kwargs)

        @property
        def statements(self):
            """Returns the statements of the OpenMP holder."""
            return self._statements

        def __str__(self):
            return "\n".join(str(stmt) for stmt in self.statements)

        def __repr__(self):
            return "\n".join(repr(stmt) for stmt in self.statements)

    class OmpConstruct(Basic):

        __slots__ = ("_start", "_end", "_body")
        _attribute_nodes = ("_start", "_end", "_body")

        def __init__(self, start, body, end=None):
            self._start = start
            self._end = end
            self._body = body
            super().__init__()

        @property
        def start(self):
            return self._start

        @property
        def body(self):
            return self._body

        @property
        def end(self):
            return self._end

    class OmpDirective(OmpAnnotatedComment):

        """Represents an OpenMP Construct in the code."""

        __slots__ = ("_name", "_clauses", "_require_end_directive")

        _attribute_nodes = ("_clauses",)
        _allowed_clauses = ()
        _deprecated_clauses = ()
        _used_in_grammar = True

        def __init__(self, **kwargs):
            self._name = kwargs.pop("name")
            self._clauses = kwargs.pop("clauses", [])
            self._require_end_directive = kwargs.pop("require_end_directive", False)
            super().__init__(**kwargs)

        @property
        def name(self):
            return self._name

        @property
        def clauses(self):
            return self._clauses

        @property
        def require_end_directive(self):
            return self._require_end_directive

    class OmpEndDirective(OmpDirective):
        """Represents an OpenMP End Construct."""

        __slots__ = ("_coresponding_directive",)
        _attribute_nodes = ("_coresponding_directive",)

        def __init__(self, **kwargs):
            self._coresponding_directive = kwargs.pop("coresponding_directive", None)
            super().__init__(**kwargs)

        @property
        def coresponding_directive(self):
            return self._coresponding_directive

    class OmpParallelDirective(OmpDirective):
        """Represents an OpenMP Parallel Construct for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ()

        _allowed_clauses = (
            "if",
            "num_threads",
            "default",
            "private",
            "firstprivate",
            "shared",
            "copyin",
            "reduction",
            "proc_bind",
        )
        _deprecated_clauses = ()

        def __init__(self, **kwargs):
            super().__init__(**kwargs, require_end_directive=True)


    class OmpStatement(OmpAnnotatedComment):
        """Represents an OpenMP statement."""

        __slots__ = ("_statement",)
        _used_in_grammar = True

        def __init__(self, **kwargs):
            self._statement = kwargs.pop("statement")
            super().__init__()

        @property
        def statement(self):
            """Returns the statement of the OpenMP statement."""
            return self._statement

        def __str__(self):
            return f"#$ omp {str(self.statement)}"

        def __repr__(self):
            return f"#$ omp {repr(self.statement)}"

    class OmpClause(OmpAnnotatedComment):
        """Represents an OpenMP Clause in the code."""

        __slots__ = ("_name", "_parent")

        def __init__(self, **kwargs):
            self._name = kwargs.pop("name")
            self._parent = kwargs.pop("parent")
            super().__init__(**kwargs)

        @property
        def name(self):
            """Returns the name of the clause."""
            return self._name

        def __repr__(self):
            return f"{self.name}"

    class OmpIfClause(OmpClause):
        """Represents an OpenMP If Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ('expr_ast', 'directive_name_modifier', 'expr')

        def __init__(self, **kwargs):
            self.directive_name_modifier = kwargs.pop("directive_name_modifier", None)
            self.expr = kwargs.pop("expr", None)
            self.expr_ast = None
            super().__init__(**kwargs)

    class OmpNumThreadsClause(OmpClause):
        """Represents an OpenMP NumThreads Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ('num_threads', "num_threads_ast")

        def __init__(self, **kwargs):
            self.num_threads = kwargs.pop("num_threads")
            self.num_threads_ast = None
            super().__init__(**kwargs)

    class OmpDefaultClause(OmpClause):
        """Represents an OpenMP Default Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ('attribute',)

        def __init__(self, **kwargs):
            self.attribute = kwargs.pop("attribute")
            super().__init__(**kwargs)

    class OmpPrivateClause(OmpClause):
        """Represents an OpenMP Private Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ('variables', 'variables_ast')

        def __init__(self, **kwargs):
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

    class OmpFirstPrivateClause(OmpClause):
        """Represents an OpenMP FirstPrivate Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ('variables', 'variables_ast')

        def __init__(self, **kwargs):
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

    class OmpSharedClause(OmpClause):
        """Represents an OpenMP Shared Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ('variables', 'variables_ast')

        def __init__(self, **kwargs):
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

    class OmpCopyinClause(OmpClause):
        """Represents an OpenMP Copyin Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ('variables', 'variables_ast')

        def __init__(self, **kwargs):
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

    class OmpReductionClause(OmpClause):
        """Represents an OpenMP Reduction Clause for both Pyccel AST
        and textx grammer rule
        """
        #TODO: this clause still needs more work to support 
        # both C and Fortran and also to support user defined
        # reduction operations

        __slots__ = ('modifier', 'operator', 'variables', 'variables_ast')

        def __init__(self, **kwargs):
            self.operator = kwargs.pop("operator")
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

    class OmpProcBindClause(OmpClause):
        """Represents an OpenMP ProcBind Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ('affinity_policy',)

        def __init__(self, **kwargs):
            self.affinity_policy = kwargs.pop("affinity_policy")
            super().__init__(**kwargs)

    class SemanticParser(ABC):

        def __init__(self):
            self._omp_reserved_nodes = []

        @property
        def omp_reserved_nodes(self):
            return self._omp_reserved_nodes

        @abstractmethod
        def _visit(self, stmt):
            pass

        def _visit_Omp(self, expr):
            statements = tuple(
                self._visit(stmt) for stmt in expr.statements
            )
            return statements

        def _visit_OmpDirective(self, expr):
            clauses = [self._visit(clause) for clause in expr._clauses]
            directive = Openmp.OmpDirective(name=expr.name, clauses=clauses)
            end_directives = [dirve for dirve in expr.get_all_user_nodes() if isinstance(dirve, Openmp.OmpEndDirective)]
            if len(end_directives) == 0:
               return directive
            assert len(end_directives) == 1
            end_dir = end_directives[0]
            end_dir.substitute(end_dir.coresponding_directive, EmptyNode())
            if end_dir.get_all_user_nodes() != expr.get_all_user_nodes():
                errors.report(
                    f"{expr} and {end_dir}, should be contained in the same block",
                    symbol=expr,
                    severity="fatal",
                )
            container = expr.get_all_user_nodes()[-1]
            reserved_nodes = container.body[container.body.index(expr) + 1:container.body.index(end_dir) + 1]
            end_dir = self._visit(reserved_nodes[-1])
            body = self._visit(CodeBlock(reserved_nodes[:-1]))
            self._omp_reserved_nodes = reserved_nodes
            return Openmp.OmpConstruct(start=directive, end=end_dir, body=body)

        #def _visit_OmpParallelConstruct(self, expr):
        #    """Check the validity of the clauses and the construct."""
        #    # check if the construct has an end parallel or a return before end parallel

        #    #body = self.get_body(expr)
        #    #clauses = [self._visit(clause) for clause in expr._clauses]
        #    #return  Openmp.OmpParallelConstruct(name=expr.name, clauses=clauses, body=body)
        def _visit_OmpEndDirective(self, expr):
            if expr.coresponding_directive is None:
                errors.report(
                    f"OMP END does not match any OMP {expr.name}",
                    symbol=expr,
                    severity="fatal",
                )
            clauses = [self._visit(clause) for clause in expr.clauses]
            return Openmp.OmpEndDirective(name=expr.name, clauses=clauses)

    class SyntaxParser(ABC):

        def __init__(self):
            self._pending_directives = []

        @property
        def pending_directives(self):
            return self._pending_directives

        @abstractmethod
        def _visit(self, stmt):
            pass

        def _visit_Omp(self, expr):
            statements = tuple(
                self._visit(stmt) for stmt in expr.statements
            )
            return statements

        def _find_coresponding_directive(self, end_directive):
            cor_directive = None
            if len(self._pending_directives) == 0:
                errors.report(
                   f"`{end_directive}` misplaced",
                   symbol=end_directive,
                   severity="fatal",
                   )
            if end_directive.name != self._pending_directives[-1].name:
                if not self._pending_directives[-1].require_end_directive:
                    self._pending_directives.pop()
                    cor_directive = self._find_coresponding_directive(end_directive)
                else:
                    errors.report(
                        f"`{end_directive}` misplaced",
                        symbol=end_directive,
                        severity="fatal",
                    )
            else:
                cor_directive = self._pending_directives.pop()

            return cor_directive

        def _visit_OmpDirective(self, expr):
            clauses = [self._visit(clause) for clause in expr._clauses]
            ret = Openmp.OmpDirective(name=expr.name, clauses=clauses)
            self._pending_directives.append(ret)
            return ret

        def _visit_OmpParallelDirective(self, expr):
            clauses = [self._visit(clause) for clause in expr._clauses]
            #for clause in ("if", "proc_bind", "num_threads"):
            #    if expr.clauses_count.get(clause, 0) > 1:
            #        errors.report(
            #            f"OMP PARALLEL `{clause}` clause must appear only once",
            #            symbol=expr,
            #            severity="fatal",
            #        )
            ret = Openmp.OmpParallelDirective(name=expr.name, clauses=clauses)
            self._pending_directives.append(ret)
            return ret
        def _visit_OmpEndDirective(self, expr):
            cor_directive = self._find_coresponding_directive(expr)
            clauses = [self._visit(clause) for clause in expr._clauses]
            return Openmp.OmpEndDirective(name=expr.name, clauses=clauses, coresponding_directive=cor_directive)

    class CCodePrinter:

        def _print_OmpConstruct(self, expr):
            clauses = " ".join(self._print(clause) for clause in expr.start.clauses)
            body = self._print(expr.body)
            return f"#pragma omp {expr.start.name} {clauses}\n{{\n{body}\n}}\n"

        def _print_OmpEndDirective(self, expr):
            return ""
