"""
Module containing the grammar rules for the OpenMP 5.0 specification,
"""

from .version_4_5 import Openmp as PreviosVersion


class Openmp(PreviosVersion):
    """Represents an OpenMP Parallel Construct for both Pyccel AST
    and textx grammer rule
    """

    current_version = 5.0
