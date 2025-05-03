# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Module containing aspects of a parser which are in common over all stages.
"""

import importlib
import os
import pathlib
import re

#==============================================================================
from pyccel.version import __version__

from pyccel.ast.core import FunctionDef, Interface, FunctionAddress
from pyccel.ast.core import SympyFunction
from pyccel.ast.core import Import, AsName

from pyccel.ast.variable import DottedName

from pyccel.parser.scope     import Scope
from pyccel.parser.utilities import is_valid_filename_pyh, is_valid_filename_py

from pyccel.errors.errors   import Errors, ErrorsMode
from pyccel.errors.messages import PYCCEL_UNFOUND_IMPORTED_MODULE

#==============================================================================

errors = Errors()
error_mode = ErrorsMode()

#==============================================================================

strip_ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]|[\n\t\r]')

# use this to delete ansi_escape characters from a string
# Useful for very coarse version differentiation.

#==============================================================================
def get_filename_from_import(module_name, input_folder_name):
    """
    Get the absolute path of a module_name, searching in a given folder.

    Return a valid filename with an absolute path, that corresponds to the
    definition of module_name.
    The priority order is:
        - python files (extension == py)
        - header files (extension == pyi)

    Parameters
    ----------
    module_name : str | AsName
        Name of the module_name of interest.

    input_folder_name : str
        Relative path of the folder which should be searched for the module_name.

    Returns
    -------
    str
        Absolute path of the given module_name.

    Raises
    ------
    PyccelError
        Error raised when the module_name cannot be found.
    """

    if (isinstance(module_name, AsName)):
        module_name = str(module_name.name)

    print("importing : ", module_name)

    in_project = module_name[0] == '.'

    input_folder = pathlib.Path(input_folder_name)

    if in_project:
        project_depth = next(i for i, c in enumerate(module_name) if c != '.')
        if project_depth == 1:
            project_dir = input_folder
        else:
            project_dir = input_folder.parents[project_depth-2]
        module_path = module_name[project_depth:].split('.')
        filename_stem = project_dir.joinpath(*module_path)
    else:
        filename_stem = pathlib.Path(input_folder).joinpath(*module_name.split('.'))

    filename_py = filename_stem.with_suffix('.py')
    if filename_py.exists():
        return str(filename_py.absolute())
    elif filename_stem.with_suffix('.pyh').exists():
        return str(filename_stem.with_suffix('.pyh').absolute())
    else:
        errors = Errors()
        raise errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE, symbol=module_name,
                      severity='fatal')

#==============================================================================
class BasicParser(object):
    """
    Class for a basic parser.

    This class contains functions and properties which are common to SyntacticParser and SemanticParser.

    See Also
    --------
    SyntacticParser : A parser for Pyccel based on a context-free grammar.
    SemanticParser : A parser for Pyccel based on a context-sensitive grammar.

    Examples
    --------
    To use the BasicParser class, create an instance and call its parse() method:
    >>> parser = BasicParser()
    >>> result = parser.parse("1 + 2")
    """

    def __init__(self):
        self._code = None
        self._fst = None
        self._ast = None

        self._filename = None
        self._metavars = {}

        # represent the scope of a function
        self._scope = Scope()
        self._current_function = None

        # the following flags give us a status on the parsing stage
        self._syntax_done   = False
        self._semantic_done = False

        # current position for errors

        self._current_ast_node = None

        # flag for blocking errors. if True, an error with this flag will cause
        # Pyccel to stop
        # TODO ERROR must be passed to the Parser __init__ as argument

        self._blocking = error_mode.value == 'developer'

    @property
    def scope(self):
        """ The Scope object containing all objects defined within the current scope
        """
        return self._scope

    @scope.setter
    def scope(self, scope):
        assert isinstance(scope, Scope)
        self._scope = scope

    @property
    def filename(self):
        """
        The file being translated.

        Get the name of the file being translated.
        """
        return self._filename

    @property
    def code(self):
        """
        The original code.

        Get the original Python code which is being translated.
        """
        return self._code

    @property
    def fst(self):
        """
        Full syntax tree.

        Get the full syntax tree describing the code. This object contains `PyccelAstNode`s
        and is generated by the semantic stage. The full syntax tree is similar to the
        abstract syntax tree, but additionally contains information about the types of the
        objects etc.
        """
        return self._fst

    @property
    def ast(self):
        """
        Abstract syntax tree.

        Get the abstract syntax tree describing the code. This object contains `PyccelAstNode`s
        and is generated by the syntactic stage. The abstract syntax tree is similar to the
        full syntax tree, but only contains information about the syntax, there is no semantic
        data (e.g. the types of variables are unknown).
        """
        if self._ast is None:
            self._ast = self.parse()
        return self._ast

    @ast.setter
    def ast(self, ast):
        self._ast = ast

    @property
    def metavars(self):
        return self._metavars

    @property
    def current_function(self):
        """Name of current function, if any."""
        return self._current_function

    @property
    def syntax_done(self):
        return self._syntax_done

    @property
    def semantic_done(self):
        return self._semantic_done

    @property
    def is_header_file(self):
        """Returns True if we are treating a header file."""

        if self.filename:
            return self.filename.split(""".""")[-1] == 'pyh'
        else:
            return False

    @property
    def current_ast_node(self):
        """
        The AST for the current node.

        The AST object describing the current node. This object is never set to None
        when entering a node. Therefore if a node has no AST object (e.g. a Variable)
        the `current_ast_node` will contain the AST of the enclosing object. It is
        set in the `_visit` method of `SemanticParser`. This object is useful for
        reporting errors on objects whose context is unknown (e.g. Variables).
        """
        return self._current_ast_node

    @property
    def blocking(self):
        return self._blocking

    def insert_function(self, func):
        """
        Insert a function into the current scope.

        Insert a function into the current scope under the final name by which it
        will be known in the generated code.

        Parameters
        ----------
        func : FunctionDef | SympyFunction | Interface | FunctionAddress
            The function to be inserted into the scope.
        """

        if isinstance(func, SympyFunction):
            self.insert_symbolic_function(func)
        elif isinstance(func, (FunctionDef, Interface, FunctionAddress)):
            container = self.scope.functions
            if func.pyccel_staging == 'syntactic':
                container[self.scope.get_expected_name(func.name)] = func
            else:
                container[func.name] = func
        else:
            raise TypeError('Expected a Function definition')

    def insert_symbolic_function(self, func):
        """."""

        container = self.scope.symbolic_functions
        if isinstance(func, SympyFunction):
            container[func.name] = func
        else:
            raise TypeError('Expected a symbolic_function')

    def create_new_function_scope(self, name, **kwargs):
        """
        Create a new Scope object for a Python function with the given name,
        and attach any decorators' information to the scope. The new scope is
        a child of the current one, and can be accessed from the dictionary of
        its children using the function name as key.

        Before returning control to the caller, the current scope (stored in
        self._scope) is changed to the one just created, and the function's
        name is stored in self._current_function.

        Parameters
        ----------
        name : str
            Function's name, used as a key to retrieve the new scope.

        decorators : dict
            Decorators attached to FunctionDef object at syntactic stage.

        """
        child = self.scope.new_child_scope(name, **kwargs)

        self._scope = child
        if self._current_function:
            name = DottedName(self._current_function, name)
        self._current_function = name

        return child

    def exit_function_scope(self):
        """ Exit the function scope and return to the encasing scope
        """

        self._scope = self._scope.parent_scope
        if isinstance(self._current_function, DottedName):

            name = self._current_function.name[:-1]
            if len(name)>1:
                name = DottedName(*name)
            else:
                name = name[0]
        else:
            name = None
        self._current_function = name

    def create_new_loop_scope(self):
        """ Create a new scope describing a loop
        """
        self._scope = self._scope.create_new_loop_scope()
        return self._scope

    def exit_loop_scope(self):
        """ Exit the loop scope and return to the encasing scope
        """
        self._scope = self._scope.parent_scope

    def create_new_class_scope(self, name, **kwargs):
        """
        Create a new scope for a Python class.

        Create a new Scope object for a Python class with the given name,
        and attach any decorators' information to the scope. The new scope is
        a child of the current one, and can be accessed from the dictionary of
        its children using the function name as key.

        Before returning control to the caller, the current scope (stored in
        self._scope) is changed to the one just created, and the function's
        name is stored in self._current_function.

        Parameters
        ----------
        name : str
            Function's name, used as a key to retrieve the new scope.

        **kwargs : dict
            A dictionary containing any additional arguments of the new scope.

        Returns
        -------
        Scope
            The scope for the class.
        """
        child = self.scope.new_child_scope(name, **kwargs)
        self._scope = child

        return child

    def exit_class_scope(self):
        """ Exit the class scope and return to the encasing scope
        """
        self._scope = self._scope.parent_scope

    def copy(self, parser):
        """
        Copy the parser attributes in self

          Parameters
          ----------
          parser : BasicParser

        """
        self._fst = parser.fst
        self._ast = parser.ast

        self._metavars = parser.metavars
        self._scope    = parser.scope

        # the following flags give us a status on the parsing stage
        self._syntax_done   = parser.syntax_done
        self._semantic_done = parser.semantic_done


#==============================================================================
if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError('Expecting an argument for filename')

    parser = BasicParser(filename)
