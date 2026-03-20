# ------------------------------------------------------------------------- #
# This file is part of Pyccel which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE #
# for full license details.                                                 #
# ------------------------------------------------------------------------- #
"""
Module containing the `Complexity` class which provides an interface for
calculating the computational complexity of a given piece of code. This
base class may be extended to compute, e.g., the operation/time complexity,
or the memory/space complexity.
"""

import os
from pyccel.parser.parser import Parser

__all__ = ["Complexity"]


# ...
class Complexity:
    """
    Abstract class for complexity computation.

    Abstract class for complexity computation.

    Parameters
    ----------
    filename_or_text : str
        Name of the file containing the abstract grammar or input code to
        parse as a string.
    """

    def __init__(self, filename_or_text):
        pyccel = Parser(filename_or_text, output_folder=os.getcwd())
        self._ast = pyccel.parse(verbose=0)
        self._ast = pyccel.annotate(verbose=0).ast

    @property
    def ast(self):
        """Returns the Abstract Syntax Tree."""
        return self._ast

    def cost(self):
        """Computes the complexity of the given code."""
        return 0
