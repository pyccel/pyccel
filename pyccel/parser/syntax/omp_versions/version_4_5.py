"""
Module containing the grammar rules for the OpenMP 4.5 specification,
"""
from inspect import isclass
import pyccel.ast.omp as PreviousVersion

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

    class Omp(PreviousVersion.Omp):
        """Represents an OpenMP Construct for both Pyccel AST
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

    class OmpStatement(PreviousVersion.OmpStatement):
        """Represents an OpenMP Statement for both Pyccel AST
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

    class OmpParallelConstruct(PreviousVersion.OmpParallelConstruct):
        """Represents an OpenMP Parallel Construct for both Pyccel AST
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

    class OmpEndConstruct(PreviousVersion.OmpEndConstruct):
        """Represents an OpenMP End Construct for both Pyccel AST
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

    class OmpIfClause(PreviousVersion.OmpIfClause):
        """Represents an OpenMP If Clause for both Pyccel AST
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

    class OmpNumThreadsClause(PreviousVersion.OmpNumThreadsClause):
        """Represents an OpenMP NumThreads Clause for both Pyccel AST
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

    class OmpDefaultClause(PreviousVersion.OmpDefaultClause):
        """Represents an OpenMP Default Clause for both Pyccel AST
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

    class OmpPrivateClause(PreviousVersion.OmpPrivateClause):
        """Represents an OpenMP Private Clause for both Pyccel AST
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

    class OmpFirstPrivateClause(PreviousVersion.OmpFirstPrivateClause):
        """Represents an OpenMP FirstPrivate Clause for both Pyccel AST
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

    class OmpSharedClause(PreviousVersion.OmpSharedClause):
        """Represents an OpenMP Shared Clause for both Pyccel AST
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

    class OmpCopyinClause(PreviousVersion.OmpCopyinClause):
        """Represents an OpenMP Copyin Clause for both Pyccel AST
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
