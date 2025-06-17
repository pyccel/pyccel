"""Contains OpenMp 5.0 parser classes"""

from . import openmp_4_5

__all__ = (
    "CCodePrinter",
    "FCodePrinter",
    "PythonCodePrinter",
    "SemanticParser",
    "SyntaxParser",
)


class SyntaxParser(openmp_4_5.SyntaxParser):
    """
    Openmp 5.0 syntax parser.

    This class extends the OpenMP 4.5 syntax parser to support OpenMP 5.0 features.
    It inherits all functionality from the 4.5 parser and sets the version to 5.0,
    which affects how directives are parsed and processed.

    See Also
    --------
    openmp_4_5.SyntaxParser : Base class for OpenMP syntax parsing.
    pyccel.plugins.Openmp.omp : Module containing OpenMP AST nodes.
    """

    _version = 5.0
    _omp_metamodel = None


class SemanticParser(openmp_4_5.SemanticParser):
    """
    Openmp 5.0 semantic parser.

    This class extends the OpenMP 4.5 semantic parser to support OpenMP 5.0 features.
    It inherits all functionality from the 4.5 parser and sets the version to 5.0,
    which affects how directives are semantically analyzed and processed.

    See Also
    --------
    openmp_4_5.SemanticParser : Base class for OpenMP semantic parsing.
    pyccel.plugins.Openmp.omp.OmpDirective : Class representing an OpenMP directive.
    pyccel.plugins.Openmp.omp.OmpConstruct : Class representing an OpenMP construct.
    """

    _version = 5.0


class CCodePrinter(openmp_4_5.CCodePrinter):
    """
    Openmp 5.0 C printer.

    This class extends the OpenMP 4.5 C code printer to support OpenMP 5.0 features.
    It inherits all functionality from the 4.5 printer and sets the version to 5.0,
    which affects how directives are printed in C syntax.

    See Also
    --------
    openmp_4_5.CCodePrinter : Base class for OpenMP C code printing.
    FCodePrinter : Printer for OpenMP Fortran code.
    PythonCodePrinter : Printer for OpenMP Python code.
    """

    _version = 5.0


class FCodePrinter(openmp_4_5.FCodePrinter):
    """
    Openmp 5.0 fortran printer.

    This class extends the OpenMP 4.5 Fortran code printer to support OpenMP 5.0 features.
    It inherits all functionality from the 4.5 printer and sets the version to 5.0,
    which affects how directives are printed in Fortran syntax.

    See Also
    --------
    openmp_4_5.FCodePrinter : Base class for OpenMP Fortran code printing.
    CCodePrinter : Printer for OpenMP C code.
    PythonCodePrinter : Printer for OpenMP Python code.
    """

    _version = 5.0


class PythonCodePrinter(openmp_4_5.PythonCodePrinter):
    """
    Openmp 5.0 python printer.

    This class extends the OpenMP 4.5 Python code printer to support OpenMP 5.0 features.
    It inherits all functionality from the 4.5 printer and sets the version to 5.0,
    which affects how directives are printed in Python syntax.

    See Also
    --------
    openmp_4_5.PythonCodePrinter : Base class for OpenMP Python code printing.
    CCodePrinter : Printer for OpenMP C code.
    FCodePrinter : Printer for OpenMP Fortran code.
    """

    _version = 5.0
