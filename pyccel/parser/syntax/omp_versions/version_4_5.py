# pylint: disable=protected-access, missing-function-docstring
"""
Module containing the grammar rules for the OpenMP 4.5 specification,
"""
from inspect import isclass
import ast as Ast
import pyccel.ast.omp as PreviousVersion
from pyccel.ast.core import FunctionCall, Return
from pyccel.ast.omp import (
    OmpConstruct,
    OmpEndConstruct,
)
from pyccel.ast.variable import Variable
from pyccel.parser.extend_tree import extend_tree
from pyccel.ast.datatypes import NativeInteger, NativeVoid

__all__ = ("Openmp",)


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
            if isclass(obj) and obj.__name__.startswith("Omp"):
                results.append(obj)
        return results

    @classmethod
    def _visit_syntacic_variables(cls, expr, parser, errors, variables):
        """
        Helper function to visit the syntactic of a variables list of the OpenMP clauses.
        """
        variables_ast = []
        for var in variables:
            fst = extend_tree(var)
            if (
                not isinstance(fst, Ast.Module)
                or len(fst.body) != 1
                or not isinstance(fst.body[0], Ast.Expr)
                or not isinstance(fst.body[0].value, Ast.Name)
            ):
                errors.report(
                    "OMP PRIVATE clause must be a list of variables",
                    symbol=expr,
                    severity="fatal",
                )
            fst = fst.body[0].value
            variables_ast.append(parser._visit(fst))
        return variables_ast

    @classmethod
    def _visit_semantic_variables(cls, expr, parser, errors, variables):
        """
        Helper function to visit the semantic of a variables list of the OpenMP clauses.
        """
        tmp_vars = []
        for var in variables:
            ret = parser._visit(var)
            if not isinstance(ret, Variable):
                errors.report(
                    "OMP PRIVATE clause must be a list of variables",
                    symbol=expr,
                    severity="fatal",
                )
            tmp_vars.append(ret)
        return tmp_vars

    class Omp(PreviousVersion.Omp):
        """Represents an OpenMP Construct for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ()

        def visit_syntatic(self, parser, errors):
            self._statements = [
                stmt.visit_syntatic(parser, errors) for stmt in self._statements
            ]
            return self

        def visit_semantic(self, parser, errors):
            self._statements = tuple(
                stmt.visit_semantic(parser, errors) for stmt in self._statements
            )
            return self

        def cprint(self, printer, errors):
            return "\n".join(stmt.cprint(printer, errors) for stmt in self._statements)

        def fprint(self, printer, errors):
            return "\n".join(stmt.fprint(printer, errors) for stmt in self._statements)

        def pyprint(self, printer, errors):
            return "\n".join(stmt.pyprint(printer, errors) for stmt in self._statements)

    class OmpStatement(PreviousVersion.OmpStatement):
        """Represents an OpenMP Statement for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ()

        def visit_syntatic(self, parser, errors):
            self._statement = self._statement.visit_syntatic(parser, errors)
            return self

        def visit_semantic(self, parser, errors):
            self._statement = self._statement.visit_semantic(parser, errors)
            return self

        def cprint(self, printer, errors):
            return self._statement.cprint(printer, errors)

        def fprint(self, printer, errors):
            return self._statement.fprint(printer, errors)

        def pyprint(self, printer, errors):
            return self._statement.pyprint(printer, errors)

    class OmpParallelConstruct(PreviousVersion.OmpParallelConstruct):
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
            super().__init__(**kwargs, has_closing=True)

        def visit_syntatic(self, parser, errors):
            """Check the validity of the clauses and the construct."""
            # visit the clauses
            visited_clauses = super().visit_syntatic_clauses(parser, errors)
            for clause in ("if", "proc_bind", "num_threads"):
                if self.clauses_count.get(clause, 0) > 1:
                    errors.report(
                        f"OMP PARALLEL `{clause}` clause must appear only once",
                        symbol=self,
                        severity="fatal",
                    )
            self._clauses = visited_clauses
            return self

        def visit_semantic(self, parser, errors):
            """Check the validity of the clauses and the construct."""
            # check if the construct has an end parallel or a return before end parallel
            code_block = self.current_user_node
            body = code_block.body
            index = body.index(self) + 1
            need_closing = [self._name]
            while index < len(body) and len(need_closing) > 0:
                if isinstance(body[index], OmpEndConstruct):
                    if body[index].name == need_closing[-1]:
                        need_closing.pop()
                    else:
                        errors.report(
                            f"OMP END {need_closing[-1]} does not match OMP {body[index].name}",
                            symbol=self,
                            severity="fatal",
                        )
                elif isinstance(body[index], OmpConstruct) and body[index].has_closing:
                    need_closing.append(body[index].name)
                elif isinstance(body[index], Return):
                    errors.report(
                        "OMP PARALLEL invalid branch to/from OpenMP structured block",
                        symbol=self,
                        severity="fatal",
                    )
                index += 1
            if len(need_closing) != 0:
                errors.report(
                    "OMP PARALLEL construct must be closed with OMP END PARALLEL",
                    symbol=self,
                    severity="fatal",
                )
            body[index - 1].found_opening = True

            # check the validity clauses
            self._clauses = super().visit_semantic_clauses(parser, errors)
            return self

        def cprint(self, printer, errors):
            return f"{super().cprint(printer, errors)}\n{{"

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpEndConstruct(PreviousVersion.OmpEndConstruct):
        """Represents an OpenMP End Construct for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("found_opening",)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.found_opening = False

        def visit_syntatic(self, parser, errors):
            self._clauses = super().visit_syntatic_clauses(parser, errors)
            return self

        def visit_semantic(self, parser, errors):
            if not self.found_opening:
                errors.report(
                    f"OMP END does not match any OMP {self.name}",
                    symbol=self,
                    severity="fatal",
                )
            self._clauses = super().visit_semantic_clauses(parser, errors)
            return self

        def cprint(self, printer, errors):
            return "}\n"

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def __str__(self):
            return f"$# omp end {self.name}"

        def __repr__(self):
            return f"$# omp end {self.name}"

    class OmpIfClause(PreviousVersion.OmpIfClause):
        """Represents an OpenMP If Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("directive_name_modifier", "expr", "expr_ast")

        def __init__(self, **kwargs):
            self.directive_name_modifier = kwargs.pop("directive_name_modifier", None)
            self.expr = kwargs.pop("expr", None)
            self.expr_ast = None
            super().__init__(**kwargs)

        def visit_syntatic(self, parser, errors):
            # TODO: check if directive_name_modifier matches the name if the parent construct
            fst = extend_tree(self.expr)
            if (
                not isinstance(fst, Ast.Module)
                or len(fst.body) != 1
                or not isinstance(fst.body[0], Ast.Expr)
            ):
                errors.report(
                    "OMP IF clause must be a valid expression",
                    symbol=self,
                    severity="fatal",
                )
            self.expr_ast = parser._visit(fst.body[0].value)
            return self

        def visit_semantic(self, parser, errors):
            ret = parser._visit(self.expr_ast)
            if (
                not hasattr(ret, "dtype")
                or isinstance(ret.dtype, NativeVoid)
                or (isinstance(ret, FunctionCall) and not ret.funcdef.results)
            ):
                errors.report(
                    "OMP IF clause must be a valid expression",
                    symbol=self,
                    severity="fatal",
                )
            self.expr_ast = ret
            return self

        def cprint(self, printer, errors):
            modifier = (
                f"{self.directive_name_modifier}:"
                if self.directive_name_modifier
                else ""
            )
            return f"if({modifier} {printer._print(self.expr_ast)})"

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpNumThreadsClause(PreviousVersion.OmpNumThreadsClause):
        """Represents an OpenMP NumThreads Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("num_threads", "num_threads_ast")

        def __init__(self, **kwargs):
            self.num_threads = kwargs.pop("num_threads")
            self.num_threads_ast = None
            super().__init__(**kwargs)

        def visit_syntatic(self, parser, errors):
            fst = extend_tree(self.num_threads)
            if (
                not isinstance(fst, Ast.Module)
                or len(fst.body) != 1
                or not isinstance(fst.body[0], Ast.Expr)
            ):
                errors.report(
                    "OMP NUM_THREADS clause must be an integer expression",
                    symbol=self,
                    severity="fatal",
                )
            self.num_threads_ast = parser._visit(fst.body[0].value)
            return self

        def visit_semantic(self, parser, errors):
            ret = parser._visit(self.num_threads_ast)
            if not hasattr(ret, "dtype") or not isinstance(ret.dtype, NativeInteger):
                errors.report(
                    "OMP NUM_THREADS clause must be an integer expression",
                    symbol=self,
                    severity="fatal",
                )
            self.num_threads_ast = ret
            return self

        def cprint(self, printer, errors):
            return f"num_threads({printer._print(self.num_threads_ast)})"

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpDefaultClause(PreviousVersion.OmpDefaultClause):
        """Represents an OpenMP Default Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("attribute",)

        def __init__(self, **kwargs):
            self.attribute = kwargs.pop("attribute")
            super().__init__(**kwargs)

        def visit_syntatic(self, parser, errors):
            if self._parent.clauses_count.get("default", 0) > 1:
                errors.report(
                    "OMP DEFAULT clause must be specified only once",
                    symbol=self,
                    severity="fatal",
                )
            return self

        def visit_semantic(self, parser, errors):
            return self

        def cprint(self, printer, errors):
            return f"default({self.attribute})"

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpPrivateClause(PreviousVersion.OmpPrivateClause):
        """Represents an OpenMP Private Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("variables", "variables_ast")

        def __init__(self, **kwargs):
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

        def visit_syntatic(self, parser, errors):
            self.variables_ast = Openmp._visit_syntacic_variables(
                self, parser, errors, self.variables
            )
            return self

        def visit_semantic(self, parser, errors):
            self.variables_ast = Openmp._visit_semantic_variables(
                self, parser, errors, self.variables_ast
            )
            return self

        def cprint(self, printer, errors):
            return (
                f"private({', '.join(printer._print(i) for i in self.variables_ast)})"
            )

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpFirstPrivateClause(PreviousVersion.OmpFirstPrivateClause):
        """Represents an OpenMP FirstPrivate Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("variables", "variables_ast")

        def __init__(self, **kwargs):
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

        def visit_syntatic(self, parser, errors):
            self.variables_ast = Openmp._visit_syntacic_variables(
                self, parser, errors, self.variables
            )
            return self

        def visit_semantic(self, parser, errors):
            self.variables_ast = Openmp._visit_semantic_variables(
                self, parser, errors, self.variables_ast
            )
            return self

        def cprint(self, printer, errors):
            return f"firstprivate({', '.join(printer._print(i) for i in self.variables_ast)})"

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpSharedClause(PreviousVersion.OmpSharedClause):
        """Represents an OpenMP Shared Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("variables", "variables_ast")

        def __init__(self, **kwargs):
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

        def visit_syntatic(self, parser, errors):
            self.variables_ast = Openmp._visit_syntacic_variables(
                self, parser, errors, self.variables
            )
            return self

        def visit_semantic(self, parser, errors):
            self.variables_ast = Openmp._visit_semantic_variables(
                self, parser, errors, self.variables_ast
            )
            return self

        def cprint(self, printer, errors):
            return f"shared({', '.join(printer._print(i) for i in self.variables_ast)})"

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpCopyinClause(PreviousVersion.OmpCopyinClause):
        """Represents an OpenMP Copyin Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("variables", "variables_ast")

        def __init__(self, **kwargs):
            self.variables = kwargs.pop("variables")
            self.variables_ast = None
            super().__init__(**kwargs)

        def visit_syntatic(self, parser, errors):
            self.variables_ast = Openmp._visit_syntacic_variables(
                self, parser, errors, self.variables
            )
            return self

        def visit_semantic(self, parser, errors):
            self.variables_ast = Openmp._visit_semantic_variables(
                self, parser, errors, self.variables_ast
            )
            return self

        def cprint(self, printer, errors):
            return (
                f"private({', '.join(printer._print(i) for i in self.variables_ast)})"
            )

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpReductionClause(PreviousVersion.OmpReductionClause):
        """Represents an OpenMP Reduction Clause for both Pyccel AST
        and textx grammer rule
        """

        def visit_syntatic(self, parser, errors):
            raise NotImplementedError("TODO: implement me please!")

        def visit_semantic(self, parser, errors):
            raise NotImplementedError("TODO: implement me please!")

        def cprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

    class OmpProcBindClause(PreviousVersion.OmpProcBindClause):
        """Represents an OpenMP ProcBind Clause for both Pyccel AST
        and textx grammer rule
        """

        __slots__ = ("affinity_policy",)

        def __init__(self, **kwargs):
            self.affinity_policy = kwargs.pop("affinity_policy")
            super().__init__(**kwargs)

        def visit_syntatic(self, parser, errors):
            return self

        def visit_semantic(self, parser, errors):
            return self

        def cprint(self, printer, errors):
            return f"proc_bind({self.affinity_policy})"

        def fprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")

        def pyprint(self, printer, errors):
            raise NotImplementedError("TODO: implement me please!")
