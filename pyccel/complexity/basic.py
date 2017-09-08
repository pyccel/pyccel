# coding: utf-8

from pyccel.parser  import PyccelParser

__all__ = ["Complexity"]

# ...
class Complexity(object):
    """Abstract class for complexity computation."""
    def __init__(self, filename):
        """Constructor for the Complexity class.

        filename: str
            name of the file containing the abstract grammar.
        """
        # ... TODO improve once TextX will handle indentation
        from pyccel.codegen import clean, preprocess, make_tmp_file

        clean(filename)

        filename_tmp = make_tmp_file(filename)
        preprocess(filename, filename_tmp)
        filename = filename_tmp
        # ...

        # ...
        self._filename = filename
        # ...

        # ...
        pyccel    = PyccelParser()
        self._ast = pyccel.parse_from_file(filename)
        # ...

    @property
    def filename(self):
        """Returns the name of the file to process."""
        return self._filename

    @property
    def ast(self):
        """Returns the Abstract Syntax Tree."""
        return self._ast

    def cost(self):
        """Computes the complexity of the given code."""
        return 0
# ...
