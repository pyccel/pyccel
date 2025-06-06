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
    """Openmp 5.0 syntax parser"""

    _version = 5.0
    _omp_metamodel = None


class SemanticParser(openmp_4_5.SemanticParser):
    """Openmp 5.0 semantic parser"""

    _version = 5.0


class CCodePrinter(openmp_4_5.CCodePrinter):
    """Openmp 5.0 C printer"""

    _version = 5.0


class FCodePrinter(openmp_4_5.FCodePrinter):
    """Openmp 5.0 fortran printer"""

    _version = 5.0


class PythonCodePrinter(openmp_4_5.PythonCodePrinter):
    """Openmp 5.0 python printer"""

    _version = 5.0
