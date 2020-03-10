# -*- coding: utf-8 -*-
import os
from collections import OrderedDict

from pyccel.parser.base      import get_filename_from_import
from pyccel.parser.syntactic import SyntaxParser
from pyccel.parser.semantic  import SemanticParser

# TODO [AR, 18.11.2018] to be modified as a function
# TODO [YG, 28.01.2020] maybe pass filename to the parse method?
class Parser(object):

    def __init__(self, filename, **kwargs):

        self._filename = filename
        self._kwargs = kwargs

        # we use it to store the imports
        self._parents = []

        # a Parser can have parents, who are importing it.
        # imports are then its sons.
        self._sons = []
        self._d_parsers = OrderedDict()

        self._syntax_parser = None
        self._semantic_parser = None

        self._output_folder = kwargs.pop('output_folder', '')
        self._input_folder = os.path.dirname(filename)

    @property
    def filename(self):
        """ Python file to be parsed. """
        return self._filename

    @property
    def d_parsers(self):
        """Returns the d_parsers parser."""

        return self._d_parsers

    @property
    def parents(self):
        """Returns the parents parser."""

        return self._parents

    @property
    def sons(self):
        """Returns the sons parser."""

        return self._sons

    @property
    def metavars(self):
        if self._semantic_parser:
            return self._semantic_parser.metavars
        else:
            return self._syntax_parser.metavars

    @property
    def namespace(self):
        if self._semantic_parser:
            return self._semantic_parser.namespace
        else:
            return self._syntax_parser.namespace

    @property
    def imports(self):
        if self._semantic_parser:
            raise NotImplementedError('TODO')
        else:
            return self._syntax_parser.namespace.imports['imports']

    @property
    def fst(self):
        return self._syntax_parser.fst

    def parse(self, d_parsers=None, verbose=False):
        parser = SyntaxParser(self._filename, **self._kwargs)
        self._syntax_parser = parser

        if d_parsers is None:
            d_parsers = OrderedDict()
        self._d_parsers = self._parse_sons(d_parsers, verbose=verbose)

        return parser.ast

    def annotate(self, **settings):

        # If the semantic parser already exists, do nothing
        if self._semantic_parser:
            return self._semantic_parser

        # we first treat all sons to get imports
        verbose = settings.pop('verbose', False)
        self._annotate_sons(verbose=verbose)

        # Create a new semantic parser and store it in object
        parser = SemanticParser(self._syntax_parser,
                                d_parsers=self.d_parsers,
                                parents=self.parents,
                                **settings)
        self._semantic_parser = parser

        # Return the new semantic parser (maybe used by codegen)
        return parser

    def append_parent(self, parent):
        """."""

        # TODO check parent is not in parents

        self._parents.append(parent)

    def append_son(self, son):
        """."""

        # TODO check son is not in sons

        self._sons.append(son)

    def _parse_sons(self, d_parsers, verbose=False):
        """Recursive algorithm for syntax analysis on a given file and its
        dependencies.
        This function always terminates with an OrderedDict that contains parsers
        for all involved files.
        """

        imports = self.imports.keys()
        treated = d_parsers.keys()
        not_treated = [i for i in imports if i not in treated]

        for source in not_treated:
            if verbose:
                print ('>>> treating :: {}'.format(source))

            # get the absolute path corresponding to source

            filename = get_filename_from_import(source, self._input_folder)

            q = Parser(filename)
            q.parse(d_parsers=d_parsers)
            d_parsers[source] = q

        # link self to its sons
        for source in imports:
            d_parsers[source].append_parent(self)
            self.append_son(d_parsers[source])

        return d_parsers

    def _annotate_sons(self, **settings):

        verbose = settings.pop('verbose', False)

        # we first treat sons that have no imports

        for p in self.sons:
            if not p.sons:
                if verbose:
                    print ('>>> treating :: {}'.format(p.filename))
                p.annotate(**settings)

        # finally we treat the remaining sons recursively

        for p in self.sons:
            if p.sons:
                if verbose:
                    print ('>>> treating :: {}'.format(p.filename))
                p.annotate(**settings)

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
