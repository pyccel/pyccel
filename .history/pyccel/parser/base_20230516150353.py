# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module containing aspects of a parser which are in common over all stages.
"""

import importlib
import os
import re
import warnings
from filelock import FileLock

#==============================================================================
from pyccel.version import __version__

from pyccel.ast.builtins import Lambda

from pyccel.ast.core import SymbolicAssign
from pyccel.ast.core import FunctionDef, Interface, FunctionAddress
from pyccel.ast.core import SympyFunction
from pyccel.ast.core import Import, AsName
from pyccel.ast.core import create_variable

from pyccel.ast.utilities import recognised_source

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


def get_filename_from_import(module,input_folder=''):
    """Returns a valid filename with absolute path, that corresponds to the
    definition of module.
    The priority order is:
        - header files (extension == pyh)
        - python files (extension == py)
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

    filename_pyh = '{}.pyh'.format(filename)
    filename_py  = '{}.py'.format(filename)
    folders = input_folder.split(""".""")
    for i in range(len(folders)):
        poss_dirname      = os.path.join( *folders[:i+1] )
        poss_filename_pyh = os.path.join( poss_dirname, filename_pyh )
        poss_filename_py  = os.path.join( poss_dirname, filename_py  )
        if is_valid_filename_pyh(poss_filename_pyh):
            return os.path.abspath(poss_filename_pyh)
        if is_valid_filename_py(poss_filename_py):
            return os.path.abspath(poss_filename_py)

    source = module
    if len(module.split(""".""")) > 1:

        # we remove the last entry, since it can be a pyh file

        source = """.""".join(i for i in module.split(""".""")[:-1])
        _module = module.split(""".""")[-1]
        filename_pyh = '{}.pyh'.format(_module)
        filename_py = '{}.py'.format(_module)

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
    errors.report(PYCCEL_UNFOUND_IMPORTED_MODULE, symbol=module,
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
        self._current_class    = None
        self._current_function = None

        # the following flags give us a status on the parsing stage
        self._syntax_done   = False
        self._semantic_done = False

        # current position for errors

        self._current_fst_node = None

        # flag for blocking errors. if True, an error with this flag will cause
        # Pyccel to stop
        # TODO ERROR must be passed to the Parser __init__ as argument

        self._blocking = error_mode.value == 'developer'


        self._created_from_pickle = False

    def __setstate__(self, state):
        copy_slots = ('_code', '_fst', '_ast', '_metavars', '_scope', '_filename',
                '_metavars', '_scope', '_current_class', '_current_function',
                '_syntax_done', '_semantic_done', '_current_fst_node')

        self.__dict__.update({s : state[s] for s in copy_slots})

        if not isinstance(self.scope, Scope):
            # self.scope as set, deprecated in PR 1089
            raise AttributeError("Scope should be a Scope (Type was previously set in syntactic parser)")

        # Error related flags. Should not be influenced by pickled file
        self._blocking = ErrorsMode().value == 'developer'

        self._created_from_pickle = True

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
        return self._filename

    @property
    def code(self):
        return self._code

    @property
    def fst(self):
        return self._fst

    @property
    def ast(self):
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
    def current_class(self):
        return self._current_class

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
    def current_fst_node(self):
        return self._current_fst_node

    @property
    def blocking(self):
        return self._blocking


    def insert_function(self, func):
        """."""

        if isinstance(func, SympyFunction):
            self.insert_symbolic_function(func)
        elif isinstance(func, (FunctionDef, Interface, FunctionAddress)):
            container = self.scope.functions
            container[func.name] = func
        else:
            raise TypeError('Expected a Function definition')

    def insert_symbolic_function(self, func):
        """."""

        container = self.scope.symbolic_functions
        if isinstance(func, SympyFunction):
            container[func.name] = func
        elif isinstance(func, SymbolicAssign) and isinstance(func.rhs,
                Lambda):
            container[func.lhs] = func.rhs
        else:
            raise TypeError('Expected a symbolic_function')

    def insert_import(self, expr):
        """."""

        # this method is only used in the syntatic stage

        if not isinstance(expr, Import):
            raise TypeError('Expecting Import expression')
        container = self.scope.imports['imports']

        # if source is not specified, imported things are treated as sources
        if len(expr.target) == 0:
            if isinstance(expr.source, AsName):
                name   = expr.source
                source = expr.source.name
            else:
                name   = str(expr.source)
                source = name

            if not recognised_source(source):
                container[name] = []
        else:
            source = str(expr.source)
            if not recognised_source(source):
                if not source in container.keys():
                    container[source] = []
                container[source] += expr.target

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

        """
        child = self.scope.new_child_scope(name, **kwargs)
        self._scope = child

        return child

    def exit_class_scope(self):
        """ Exit the class scope and return to the encasing scope
        """
        self._scope = self._scope.parent_scope

    def dump(self, filename=None):
        """
        Dump the current ast using Pickle.

          Parameters
          ----------
          filename: str
            output file name. if not given `name.pyccel` will be used and placed
            in the Pyccel directory ($HOME/.pyccel)
        """
        if self._created_from_pickle:
            return

        if not filename:
            if not self.filename:
                raise ValueError('Expecting a filename to load the ast')

            path , name  = os.path.split(self.filename)

            name, ext = os.path.splitext(name)
            if ext != '.pyh':
                return

            name     = '{}.pyccel'.format(name)
            filename = os.path.join(path, name)
        # check extension

        if os.path.splitext(filename)[1] != '.pyccel':
            raise ValueError('Expecting a .pyccel extension')

        import pickle
        import hashlib

#        print('>>> home = ', os.environ['HOME'])
        # ...

        # we are only exporting the AST.
        try:
            with FileLock(filename+'.lock'):
                try:
                    code = self.code.encode('utf-8')
                    hs   = hashlib.md5(code)
                    with open(filename, 'wb') as f:
                        pickle.dump((hs.hexdigest(), __version__, self), f)
                    print("Created pickle file : ", filename)
                except (FileNotFoundError, pickle.PickleError):
                    pass
        except PermissionError:
            warnings.warn("Can't pickle files on a read-only system. Please run `sudo pyccel-init`")

    def load(self, filename=None):
        """ Load the current ast using Pickle.

          Parameters
          ----------
          filename: str
            output file name. if not given `name.pyccel` will be used and placed
            in the Pyccel directory ($HOME/.pyccel)
        """

        # ...

        if not filename:
            if not self.filename:
                raise ValueError('Expecting a filename to load the ast')

            path , name = os.path.split(self.filename)

            name, ext = os.path.splitext(name)

            if ext != '.pyh':
                return

            name     = '{}.pyccel'.format(name)
            filename = os.path.join(path, name)

        if not filename.split(""".""")[-1] == 'pyccel':
            raise ValueError('Expecting a .pyccel extension')

        import pickle

        possible_pickle_errors = (FileNotFoundError, PermissionError,
                pickle.PickleError, AttributeError)

        try:
            with FileLock(filename+'.lock'):
                try:
                    with open(filename, 'rb') as f:
                        hs, version, parser = pickle.load(f)
                    self._created_from_pickle = True
                except possible_pickle_errors:
                    return
        except PermissionError:
            # read/write problems don't need to be avoided on a read-only system
            try:
                with open(filename, 'rb') as f:
                    hs, version, parser = pickle.load(f)
                self._created_from_pickle = True
            except possible_pickle_errors:
                return

        import hashlib
        code = self.code.encode('utf-8')
        if hashlib.md5(code).hexdigest() == hs and __version__ == version:
            self.copy(parser)

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
