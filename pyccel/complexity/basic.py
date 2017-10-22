# coding: utf-8

from pyccel.parser.parser  import PyccelParser
import os

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
        # ... TODO improve once TextX will handle indentation
        from pyccel.codegen.codegen import clean, preprocess, preprocess_as_str, make_tmp_file

        # ...
        if os.path.isfile(filename_or_text):
            # ...
            filename = filename_or_text

            clean(filename)

            filename_tmp = make_tmp_file(filename)
            preprocess(filename, filename_tmp)
            filename = filename_tmp
            # ...

            # ...
            pyccel    = PyccelParser()
            self._ast = pyccel.parse_from_file(filename)
            # ...
        else:
            # ...
            code = preprocess_as_str(filename_or_text)
            # ...

            # ...
            pyccel    = PyccelParser()
            self._ast = pyccel.parse(code)
            # ...
        # ...

    @property
    def ast(self):
        """Returns the Abstract Syntax Tree."""
        return self._ast

    def cost(self):
        """Computes the complexity of the given code."""
        return 0
# ...
