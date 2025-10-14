# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Module containing the Parser object
"""

from pathlib import Path

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
    filename : str | Path
        The absolute path to the file being translated.

    output_folder : str | Path
        The output folder for the generated file.

    context_dict : dict, optional
        A dictionary containing any variables that are available in the calling context.
        This can allow certain constants to be defined outside of the function passed to epyccel.

    original_filename : str, optional
        The name of the original Python file. This won't match the filename if the
        filename is a .pyi file in a __pyccel__ folder (i.e. a .pyi file that was
        auto-generated to describe the prototypes of the methods).

    **kwargs : dict
        Any keyword arguments for BasicParser.
    """

    def __init__(self, filename, *, output_folder, context_dict = None, original_filename = None, **kwargs):

        filename = Path(filename)
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

        self._context_dict = context_dict

        self._original_filename = Path(original_filename or filename)

        self._input_folder = self._original_filename.parent
        self._output_folder = output_folder

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
    def original_filename(self) -> Path:
        """
        The absolute path to the original Python file that was translated.

        This will be equivalent to the filename, unless the file is a dependency.
        In that case the filename will be a .pyi file while the original_filename
        will be a .py file.
        """
        return self._original_filename

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

    def parse(self, *, verbose, d_parsers_by_filename=None):
        """
        Parse the parent file and all its dependencies.

        Parse the parent file and all its dependencies.

        Parameters
        ----------
        verbose : int
            Indicates the level of verbosity.

        d_parsers_by_filename : dict[str, Parser]
            A dictionary of parsed sons indexed by filename.

        Returns
        -------
        PyccelAstNode
           The ast created in the syntactic stage.
        """
        if self._syntax_parser:
            return self._syntax_parser.ast

        if verbose:
            print ('>> Parsing :: ', self._filename)

        parser             = SyntaxParser(self._filename, verbose = verbose,
                                            context_dict = self._context_dict)
        self.syntax_parser = parser

        if d_parsers_by_filename is None:
            d_parsers_by_filename = {}

        self._d_parsers = self.parse_sons(d_parsers_by_filename, verbose=verbose)

        return parser.ast

    def annotate(self, verbose):
        """
        Annotate the AST collected from the syntactic stage.

        Use the semantic parser to annotate the AST collected from
        the syntactic stage.

        Parameters
        ----------
        verbose : int
            The level of verbosity.

        Returns
        -------
        SemanticParser
            The semantic parser that was used to annotate the AST.
        """

        # If the semantic parser already exists, do nothing
        if self._semantic_parser:
            return self._semantic_parser

        if verbose:
            print ('>> Calculating semantic annotations :: ', self._filename)

        # we first treat all sons to get imports
        self._annotate_sons(verbose=verbose)

        # Create a new semantic parser and store it in object
        parser = SemanticParser(self._syntax_parser,
                                d_parsers = self.d_parsers,
                                parents = self.parents,
                                context_dict = self._context_dict,
                                verbose = verbose)
        self._semantic_parser = parser
        parser.metavars.setdefault('printer_imports', '')
        parser.metavars['printer_imports'] += ', '.join(p.metavars['printer_imports'] for p in self.sons)
        parser.metavars['printer_imports'] = parser.metavars['printer_imports'].strip(', ')

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

    def parse_sons(self, d_parsers_by_filename, verbose):
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

        verbose : int
            Indicates the level of verbosity.

        Returns
        -------
        dict
            The updated dictionary of parsed sons.
        """

        imports     = self.imports
        source_to_filename = {i: get_filename_from_import(i, self._input_folder, self._output_folder) for i in imports}
        treated     = d_parsers_by_filename.keys()
        not_treated = [i for i in source_to_filename.values() if i not in treated]
        for filename, stashed_filename in not_treated:
            filename = str(filename)

            # get the absolute path corresponding to source
            if filename in d_parsers_by_filename:
                q = d_parsers_by_filename[filename]
            else:
                q_output_folder = stashed_filename.parent
                if stashed_filename.suffix == '.pyi' and q_output_folder.name.startswith('__pyccel__'):
                    q_output_folder = q_output_folder.parent
                q = Parser(stashed_filename, output_folder = q_output_folder, original_filename = filename)
            q.parse(d_parsers_by_filename=d_parsers_by_filename, verbose=verbose)
            d_parsers_by_filename[filename] = q

        d_parsers = {}
        # link self to its sons
        for source in imports:
            filename,_ = source_to_filename[source]
            son = d_parsers_by_filename[str(filename)]
            son.append_parent(self)
            self.append_son(son)
            d_parsers[source] = son

        return d_parsers

    def _annotate_sons(self, verbose):
        """
        Annotate any dependencies of the file currently being parsed.

        Annotate any dependencies of the file currently being parsed.

        Parameters
        ----------
        verbose : int
            Indicates the level of verbosity.
        """

        # we first treat sons that have no imports

        for p in self.sons:
            if not p.sons:
                p.annotate(verbose)

        # finally we treat the remaining sons recursively

        for p in self.sons:
            if p.sons:
                p.annotate(verbose)


