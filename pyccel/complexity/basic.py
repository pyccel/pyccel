# coding: utf-8

from pyccel.parser.parser import Parser


__all__ = ["Complexity"]

# ...
class Complexity(object):
    """Abstract class for complexity computation."""
    def __init__(self, filename_or_text):
        """Constructor for the Complexity class.

        filename_or_text: str
            name of the file containing the abstract grammar or input code to
            parse as a string.
        """

        pyccel = Parser(filename_or_text)
        self._ast = pyccel.parse()
        settings = {}
        self._ast = pyccel.annotate(**settings).ast

    @property
    def ast(self):
        """Returns the Abstract Syntax Tree."""
        return self._ast

    def cost(self):
        """Computes the complexity of the given code."""
        return 0





