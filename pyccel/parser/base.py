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
import warnings

#==============================================================================
from pyccel.version import __version__

from pyccel.ast.core import FunctionDef, Interface, FunctionAddress
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
def get_filename_from_import(module_name, input_folder_name, output_folder_name):
    """
    Get the absolute path of a module_name, searching in a given folder.

    Return a valid filename with an absolute path, that corresponds to the
    definition of module_name.
    When searching for files in a folder, the order of priority is:

    - python files (extension == .py)
    - header files (extension == .pyi)

    In the Pyccel folder the priority is inverted as .py files are sometimes
    provided alongside .pyi files to spoof the functionalities so user code
    relying on these methods can be run in Python.

    Parameters
    ----------
    module_name : str | AsName
        Name of the module_name of interest.

    input_folder_name : str | Path
        Relative path of the folder which should be searched for the module_name.

    output_folder_name : str | Path
        The name of the folder where the output of the translation of the module
        from which we are searching was printed.

    Returns
    -------
    filename : pathlib.Path
        Absolute path to the Python file being imported.
    stashed_filename : pathlib.Path
        Absolute path to the .pyi version of the Python file being imported.
        If none exists then the absolute path to the Python file being imported.

    Raises
    ------
    PyccelError
        Error raised when the module_name cannot be found.
        Error raised when the file imports a file that has not been translated.
        Error raised when the file imports a file that has been changed since its last translation.
    """

    if (isinstance(module_name, AsName)):
        module_name = str(module_name.name)

    relative_project_path = module_name[0] == '.'
    in_project = '.' in module_name

    input_folder = pathlib.Path(input_folder_name)

    if relative_project_path:
        project_depth = next(i for i, c in enumerate(module_name) if c != '.')
        if project_depth == 1:
            project_dir = input_folder
        else:
            project_dir = input_folder.parents[project_depth-2]
        module_path = module_name[project_depth:].split('.')
        filename_stem = project_dir.joinpath(*module_path)
    elif in_project:
        filename_stem = input_folder.joinpath(*module_name.split('.')).with_suffix('.py')
        if not filename_stem.exists():
            module_name_parts = module_name.split('.')
            package = None
            for i in range(len(module_name_parts)):
                try:
                    package = importlib.import_module('.'.join(module_name_parts[:len(module_name_parts)-i]))
                    break
                except ImportError:
                    pass
            if package is None:
                errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE, symbol=module_name,
                                severity='fatal')
            filename_stem = pathlib.Path(package.__file__).parent / module_name.split('.')[-1]
    else:
        filename_stem = pathlib.Path(input_folder).joinpath(*module_name.split('.'))

    pyccel_folder = pathlib.Path(__file__).parent.parent
    filename_py = filename_stem.with_suffix('.py')
    filename_pyi = filename_stem.with_suffix('.pyi')
    filename_pyh = filename_stem.with_suffix('.pyh')

    # Look for .pyi or .pyh files in pyccel
    # Header files take priority in case .py files exist so files can run in Python
    if filename_pyi.exists() and pyccel_folder in filename_pyi.parents:
        abs_pyi_fname = filename_pyi.absolute()
        return abs_pyi_fname, abs_pyi_fname
    elif filename_pyh.exists() and pyccel_folder in filename_pyh.parents:
        abs_pyh_fname = filename_pyh.absolute()
        return abs_pyh_fname, abs_pyh_fname
    elif filename_py.exists() and pyccel_folder in filename_pyh.parents:
        # External files are pure Python
        abs_py_fname = filename_py.absolute()
        return abs_py_fname, abs_py_fname
    # Look for Python files which should have been translated once
    elif filename_py.exists():
        rel_path = os.path.relpath(filename_py.parent, input_folder_name)
        pyccel_output_folder = '__pyccel__' + os.environ.get('PYTEST_XDIST_WORKER', '')
        stashed_file = pathlib.Path(output_folder_name) / rel_path / pyccel_output_folder / filename_pyi.name
        if not stashed_file.exists():
            errors.report("Imported files must be pyccelised before they can be used.",
                    symbol=module_name, severity='fatal')
        if stashed_file.stat().st_mtime < filename_py.stat().st_mtime:
            errors.report(f"File {module_name} has been modified since Pyccel was last run on this file.",
                    symbol=module_name, severity='fatal')
        return filename_py.absolute(), stashed_file.resolve()
    # Look for user-defined .pyi or .pyh files
    elif filename_pyi.exists():
        abs_pyi_fname = filename_pyi.absolute()
        return abs_pyi_fname, abs_pyi_fname
    elif filename_pyh.exists():
        warnings.warn("Pyh files will be deprecated in version 2.0 of Pyccel. " +
                "Please use a .pyi file instead.", FutureWarning)
        abs_pyh_fname = filename_pyh.absolute()
        return abs_pyh_fname, abs_pyh_fname
    else:
        raise errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE, symbol=module_name,
                      severity='fatal')

#==============================================================================
class BasicParser(object):
    """
    Class for a basic parser.

    This class contains functions and properties which are common to SyntacticParser and SemanticParser.

    Parameters
    ----------
    verbose : int
        The level of verbosity.

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

    def __init__(self, verbose):
        self._code = None
        self._fst = None
        self._ast = None
        self._verbose = verbose

        self._filename = None
        self._metavars = {}

        # represent the scope of a function
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

        Get the full syntax tree describing the code. This object contains `PyccelAstNode`
        objects and is generated by the semantic stage. The full syntax tree is similar
        to the abstract syntax tree, but additionally contains information about the types
        of the objects etc.
        """
        return self._fst

    @property
    def ast(self):
        """
        Abstract syntax tree.

        Get the abstract syntax tree describing the code. This object contains `PyccelAstNode`
        objects and is generated by the syntactic stage. The abstract syntax tree is similar
        to the full syntax tree, but only contains information about the syntax, there is no
        semantic data (e.g. the types of variables are unknown).
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
        """
        The name of the function currently being visited.

        The name of the function currently being visited or None if we are not in
        a function.
        """
        return self._current_function_name[-1] if self._current_function_name else None

    @property
    def syntax_done(self):
        return self._syntax_done

    @property
    def semantic_done(self):
        return self._semantic_done

    @property
    def is_header_file(self):
        """
        Indicate if the file being translated is a header file.

        Indicate if the file being translated is a header file.
        A file is a header file if it does not include the implementation of the
        methods. This is the case for .pyi files.
        """

        if self.filename:
            return self.filename.suffix in ('.pyi', '.pyh')
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
        Insert a function into the current scope or a specified scope.

        Insert a function into a scope under the final name by which it
        will be known in the generated code. The scope is the current
        scope unless another scope is provided. This is notably the
        case when dealing with class methods which are not inserted into
        the enclosing scope.

        Parameters
        ----------
        func : FunctionDef | Interface | FunctionAddress
            The function to be inserted into the scope.

        scope : Scope, optional
            The scope where the function should be inserted.
        """

        assert isinstance(func, (FunctionDef, Interface, FunctionAddress))
        scope = scope or self.scope
        if func.pyccel_staging == 'syntactic':
            scope.insert_function(func, func.name)
        else:
            name = func.name
            scope.insert_function(func, scope.get_python_name(name))
            if self._current_function_name and name == self._current_function_name[-1]:
                self._current_function.append(func)

    def exit_function_scope(self):
        """
        Exit the function scope and return to the enclosing scope.

        Exit the function scope and return to the enclosing scope.
        """

        self._scope = self._scope.parent_scope
        func_name = self._current_function_name.pop()
        if self._current_function and self._current_function[-1].name == func_name:
            self._current_function.pop()

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
        child = self.scope.new_child_scope(name, 'class', **kwargs)
        self._scope = child

        return child

    def exit_class_scope(self):
        """ Exit the class scope and return to the encasing scope
        """
        self._scope = self._scope.parent_scope


#==============================================================================
if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError('Expecting an argument for filename')

    parser = BasicParser(filename)
