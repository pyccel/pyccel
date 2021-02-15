# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from collections import OrderedDict

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

        # ...
        functions = OrderedDict()
        if pyccel.namespace.functions:
            functions = pyccel.namespace.functions

        for son in pyccel.sons:
            functions.update(son.namespace.functions)

        self._functions = functions
        # ...

        self._costs = OrderedDict()
        self._symbol_map = {}
        self._used_names = set()
        self._visual = True
        self._mode = None

    @property
    def ast(self):
        """Returns the Abstract Syntax Tree."""
        return self._ast

    @property
    def functions(self):
        """Returns declared functions."""
        return self._functions

    @property
    def costs(self):
        """Returns costs of declared functions."""
        return self._costs

    @property
    def mode(self):
        return self._mode

    @property
    def visual(self):
        return self._visual

    def cost(self):
        """Computes the complexity of the given code."""
        return 0
