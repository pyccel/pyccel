# -*- coding: utf-8 -*-

from pyccel.parser.syntactic import SyntaxParser
from pyccel.parser.semantic  import SemanticParser

# TODO to be modified as a function
class Parser(object):

    def __init__(self, filename, **kwargs):

        self._filename = filename
        self._kwargs = kwargs

    def parse(self):
        parser = SyntaxParser(self._filename, **self._kwargs)
        self._syntax_parser = parser
        return parser.ast

    def annotate(self, **settings):
        parser = SemanticParser(self._syntax_parser, **settings)
        self._semantic_parser = parser
        return parser.ast

    @property
    def sons(self):
        return self._semantic_parser.sons

#==============================================================================

if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except:
        raise ValueError('Expecting an argument for filename')

    pyccel = Parser(filename)
    pyccel.parse(verbose=True)

    settings = {}
    pyccel.annotate(**settings)
