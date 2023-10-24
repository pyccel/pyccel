# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module containing the Parser object
"""

import os

from pyccel.parser.base      import get_filename_from_import
from pyccel.parser.syntactic import SyntaxParser
from pyccel.parser.semantic  import SemanticParser

# TODO [AR, 18.11.2018] to be modified as a function
# TODO [YG, 28.01.2020] maybe pass filename to the parse method?
class Parser(object):
    """
    A wrapper class which handles dependencies between the syntactic and semantic parsers.

    A wrapper class which handles dependencies between the syntactic and semantic parsers.

    Parameters
    ----------
    filename : str
        The name of the file being translated.

    **kwargs : dict
        Any keyword arguments for BasicParser.
    """

    def __init__(self, filename, **kwargs):

        self._filename = filename
        self._kwargs   = kwargs

        # we use it to store the imports
        self._parents = []

        # a Parser can have parents, who are importing it.
        # imports are then its sons.
        self._sons      = []
        self._d_parsers = {}

        self._syntax_parser   = None
        self._semantic_parser = None
        self._compile_obj     = None

        self._input_folder = os.path.dirname(filename)

    @property
    def semantic_parser(self):
        """ Semantic parser """
        return self._semantic_parser

    @property
    def syntax_parser(self):
        """ Syntax parser """
        return self._syntax_parser

    @property
    def compile_obj(self):
        """ Compile object """
        return self._compile_obj

    @semantic_parser.setter
    def semantic_parser(self, parser):
        assert isinstance(parser, SemanticParser)
        self._semantic_parser = parser

    @syntax_parser.setter
    def syntax_parser(self, parser):
        assert isinstance(parser, SyntaxParser)
        self._syntax_parser = parser

    @compile_obj.setter
    def compile_obj(self, compile_obj):
        self._compile_obj = compile_obj

    @property
    def filename(self):
        """ Python file to be parsed. """
        return self._filename

    @property
    def d_parsers(self):
        """Returns the d_parsers parser."""

        return self._d_parsers

    @d_parsers.setter
    def d_parsers(self, d_parsers):
        self._d_parsers = d_parsers

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
    def scope(self):
        if self._semantic_parser:
            return self._semantic_parser.scope
        else:
            return self._syntax_parser.scope

    @property
    def imports(self):
        return self.scope.collect_all_imports()

    @property
    def fst(self):
        return self._syntax_parser.fst

    def parse(self, d_parsers_by_filename=None, verbose=False):
        """
          Parse the parent file an all its dependencies.

          Parameters
          ----------
          d_parsers_by_filename : dict
            A dictionary of parsed sons indexed by filename.

          verbose: bool
            Determine the verbosity.

          Returns
          -------
          ast: Ast
           The ast created in the syntactic stage.
          """
        if self._syntax_parser:
            return self._syntax_parser.ast

        parser             = SyntaxParser(self._filename, **self._kwargs)
        self.syntax_parser = parser
        parser.ast        = parser.ast

        if d_parsers_by_filename is None:
            d_parsers_by_filename = {}

        self._d_parsers = self.parse_sons(d_parsers_by_filename, verbose=verbose)

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

    def parse_sons(self, d_parsers_by_filename, verbose=False):
        """
        Parse the files on which this file is dependent.

        Recursive algorithm for syntax analysis on a given file and its
        dependencies.
        This function always terminates with an dict that contains parsers
        for all involved files.

        Parameters
        ----------
        d_parsers_by_filename : dict
            A dictionary of parsed sons.

        verbose : bool, default=False
            Set the verbosity.

        Returns
        -------
        dict
            The updated dictionary of parsed sons.
        """

        imports     = self.imports
        source_to_filename = {i: get_filename_from_import(i, self._input_folder) for i in imports}
        treated     = d_parsers_by_filename.keys()
        not_treated = [i for i in source_to_filename.values() if i not in treated]
        for filename in not_treated:
            if verbose:
                print ('>>> treating :: ', filename)

            # get the absolute path corresponding to source
            if filename in d_parsers_by_filename:
                q = d_parsers_by_filename[filename]
            else:
                q = Parser(filename)
            q.parse(d_parsers_by_filename=d_parsers_by_filename)
            d_parsers_by_filename[filename] = q

        d_parsers = {}
        # link self to its sons
        for source in imports:
            filename = source_to_filename[source]
            son = d_parsers_by_filename[filename]
            son.append_parent(self)
            self.append_son(son)
            d_parsers[source] = son

        return d_parsers

    def _annotate_sons(self, verbose = False):
        """
        Annotate any dependencies of the file currently being parsed.

        Annotate any dependencies of the file currently being parsed.

        Parameters
        ----------
        verbose : bool, default=False
            Indicates if the treatment should be run in verbose mode.
        """

        # we first treat sons that have no imports

        for p in self.sons:
            if not p.sons:
                if verbose:
                    print ('>>> treating :: ', p.filename)
                p.annotate()

        # finally we treat the remaining sons recursively

        for p in self.sons:
            if p.sons:
                if verbose:
                    print ('>>> treating :: ', p.filename)
                p.annotate()

#==============================================================================

if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError('Expecting an argument for filename')

    pyccel = Parser(filename)
    pyccel.parse(verbose=True)

    settings = {}
    pyccel.annotate(**settings)
