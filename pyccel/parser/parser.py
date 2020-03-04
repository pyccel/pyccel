# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
import warnings

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

        self._extracted_mod_parser = None

        self._is_extracted = kwargs.pop('is_extracted',False)

    def create_extracted_mod_parser(self):
        filename = os.path.join(self._input_folder,self._syntax_parser.ast.mod_name)+'.py'
        self._extracted_mod_parser = Parser(filename, **self._kwargs, is_extracted=True)
        self._extracted_mod_parser._parents = self._parents.copy()
        self._extracted_mod_parser._sons = self._sons.copy()
        self._extracted_mod_parser._d_parsers = self._d_parsers
        self._extracted_mod_parser._syntax_parser = self._syntax_parser
        self._extracted_mod_parser._input_folder = self._input_folder
        return self._extracted_mod_parser

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
    def imports_alias(self):
        if self._semantic_parser:
            return self._semantic_parser.namespace.imports['imports_alias']
        else:
            return self._syntax_parser.namespace.imports['imports_alias']

    @property
    def imports(self):
        if self._semantic_parser:
            return self._semantic_parser.namespace.imports['imports']
        else:
            return self._syntax_parser.namespace.imports['imports']

    @property
    def fst(self):
        return self._syntax_parser.fst

    @property
    def all_semantics(self):
        if self._extracted_mod_parser:
            return [*self._extracted_mod_parser.all_semantics, self._semantic_parser]
        else:
            return [self._semantic_parser]

    @property
    def semantics(self):
        return self._semantic_parser

    @property
    def is_extracted(self):
        return self._is_extracted

    def parse(self, d_parsers=None, verbose=False):
        if self._syntax_parser:
            return self._syntax_parser
        parser = SyntaxParser(self._filename, **self._kwargs)
        self._syntax_parser = parser

        if d_parsers is None:
            d_parsers = OrderedDict()
        if parser.ast.has_additional_module():
            # Add parser before parsing sons to avoid trying to parse self
            d_parsers[self._syntax_parser.ast.mod_name] = self.create_extracted_mod_parser()

        self._d_parsers = self._parse_sons(d_parsers, verbose=verbose)

        if parser.ast.has_additional_module():
            # Use parse_sons results for extracted module
            self._extracted_mod_parser._d_parsers = self._d_parsers
            self._extracted_mod_parser._sons.extend(self._sons)
            self._extracted_mod_parser._sons.remove(self._extracted_mod_parser)

        return parser.ast

    def annotate(self, **settings):

        # If the semantic parser already exists, do nothing
        if self._semantic_parser:
            return self._semantic_parser

        # we first treat all sons to get imports
        verbose = settings.pop('verbose', False)
        self._annotate_sons(verbose=verbose)

        # Create a new semantic parser and store it in object
        if self._is_extracted:
            parser = SemanticParser(self._syntax_parser,
                                    d_parsers=self.d_parsers,
                                    parents=self.parents,
                                    ast=self._syntax_parser.ast.module,
                                    namespace=self.namespace.copy(),
                                    filename=self.filename,
                                    **settings)
        else:
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
            ast = q.parse(d_parsers=d_parsers)
            d_parsers[source] = q
            if ast.has_additional_module():
                d_parsers[ast.mod_name] = q._extracted_mod_parser
                self.imports_alias[source] = ast.mod_name
                warnings.warn("Functions are imported from a program. This may produce compilation errors due to the random nature of the module name", SyntaxWarning)

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
