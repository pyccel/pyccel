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
def get_filename_from_import(module, input_folder=''):
    """
    Get the absolute path of a module, searching in a given folder.

    Return a valid filename with an absolute path, that corresponds to the
    definition of module.
    The priority order is:
        - header files (extension == pyh)
        - python files (extension == py)

    Parameters
    ----------
    module : str | AsName
        Name of the module of interest.

    input_folder : str
        Relative path of the folder which should be searched for the module.

    Returns
    -------
    str
        Absolute path of the given module.

    Raises
    ------
    PyccelError
        Error raised when the module cannot be found.
    """

    if (isinstance(module, AsName)):
        module = str(module.name)

    # Remove first '.' as it doesn't represent a folder change
    if module[0] == '.':
        module = module[1:]
    filename = module.replace('.','/')

    # relative imports
    folder_above = '../'
    while filename.startswith('/'):
        filename = folder_above + filename[1:]

    filename_pyh = f'{filename}.pyh'
    filename_py  = f'{filename}.py'

    poss_filename_pyh = os.path.join( input_folder, filename_pyh )
    poss_filename_py  = os.path.join( input_folder, filename_py  )
    if is_valid_filename_pyh(poss_filename_pyh):
        return os.path.abspath(poss_filename_pyh)
    if is_valid_filename_py(poss_filename_py):
        return os.path.abspath(poss_filename_py)

    source = module
    if len(module.split(""".""")) > 1:

        # we remove the last entry, since it can be a pyh file

        source = """.""".join(i for i in module.split(""".""")[:-1])
        _module = module.split(""".""")[-1]
        filename_pyh = f'{_module}.pyh'
        filename_py  = f'{_module}.py'

    try:
        package = importlib.import_module(source)
        package_dir = str(package.__path__[0])
    except ImportError:
        errors = Errors()
        errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE, symbol=source,
                      severity='fatal')

    filename_pyh = os.path.join(package_dir, filename_pyh)
    filename_py = os.path.join(package_dir, filename_py)
    if os.path.isfile(filename_pyh):
        return filename_pyh
    elif os.path.isfile(filename_py):
        return filename_py

    errors = Errors()
    raise errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE, symbol=module,
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
        self._current_function_name = []
        self._current_function = []

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
    def current_function_name(self):
        """Name of current function, if any."""
        return self._current_function_name[-1] if self._current_function_name else None

    def enter_function(self, func):
        """Name of current function, if any."""
        assert isinstance(func, FunctionDef)
        self._current_function_name.append(func.name)

    def exit_function(self):
        """Name of current function, if any."""
        func_name = self._current_function_name.pop()
        if self._current_function and self._current_function[-1].name == func_name:
            self._current_function.pop()

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

    def insert_function(self, func, scope = None):
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
            scope = scope or self.scope
            container = scope.functions
            if func.pyccel_staging == 'syntactic':
                container[self.scope.get_expected_name(func.name)] = func
            else:
                name = func.name
                container[name] = func
                if self._current_function_name and name == self._current_function_name[-1]:
                    self._current_function.append(func)
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
        name is stored in self._current_function_name.

        Parameters
        ----------
        name : str
            Function's name, used as a key to retrieve the new scope.

        decorators : dict
            Decorators attached to FunctionDef object at syntactic stage.

        """
        child = self.scope.new_child_scope(name, **kwargs)

        self._scope = child
        self._current_function_name.append(name)

        return child

    def exit_function_scope(self):
        """ Exit the function scope and return to the encasing scope
        """

        self._scope = self._scope.parent_scope
        self.exit_function()

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
        name is stored in self._current_function_name.

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
