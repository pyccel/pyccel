# -*- coding: utf-8 -*-

from collections import OrderedDict
import redbaron
import importlib
import os
import re

#==============================================================================

from pyccel.ast import DottedName
from pyccel.ast import SymbolicAssign
from pyccel.ast import FunctionDef, Interface
from pyccel.ast import PythonFunction, SympyFunction
from pyccel.ast import Import
from pyccel.ast import builtin_import_registery as pyccel_builtin_import_registery

from pyccel.parser.utilities import is_valid_filename_pyh, is_valid_filename_py

from pyccel.parser.errors import Errors

# TODO - remove import * and only import what we need
#      - use OrderedDict whenever it is possible

from pyccel.parser.messages import *

#==============================================================================

from sympy import Lambda

errors = Errors()
#==============================================================================

strip_ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]|[\n\t\r]')

redbaron.ipython_behavior = False

# use this to delete ansi_escape characters from a string
# Useful for very coarse version differentiation.

#==============================================================================

def is_ignored_module(name):
    if isinstance(name, DottedName):
        if str(name) in ['pyccel.decorators']:
            return True

    return False


def get_filename_from_import(module,input_folder=''):
    """Returns a valid filename with absolute path, that corresponds to the
    definition of module.
    The priority order is:
        - header files (extension == pyh)
        - python files (extension == py)
    """

    filename = module.replace('.','/')

    # relative imports
    sl   = '//'
    dots = '..'
    while sl in filename:
        filename = filename.replace(sl, dots + '/')
        sl   = sl + '/'
        dots = dots + '.'

    filename_pyh = '{}.pyh'.format(filename)
    filename_py  = '{}.py'.format(filename)

    if is_valid_filename_pyh(filename_pyh):
        return os.path.abspath(filename_pyh)
    if is_valid_filename_py(filename_py):
        return os.path.abspath(filename_py)
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
    except:
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

class Scope(object):
    """."""
    
    def __init__(self):
    
        self._imports = OrderedDict()
            
        self._imports['functions'] = OrderedDict()
        self._imports['variables'] = OrderedDict()
        self._imports['classes'  ] = OrderedDict()
        self._imports['imports'  ] = OrderedDict()
        
        self._imports['python_functions'  ] = OrderedDict()
        self._imports['symbolic_functions'] = OrderedDict()
        
        self._variables = OrderedDict()
        self._classes   = OrderedDict()
        self._functions = OrderedDict()
        self._macros    = OrderedDict()
        self._headers   = OrderedDict()
        
        # TODO use another name for headers
        #      => reserved keyword, or use __
        self.parent_scope        = None
        self._sons_scopes        = OrderedDict()
        self._static_functions   = []
        self._cls_constructs     = OrderedDict()
        self._symbolic_functions = OrderedDict()
        self._python_functions   = OrderedDict()
        
        self._is_loop = False
        # scoping for loops
        self._loops = []
        
    @property
    def imports(self):
        return self._imports
        
    @property
    def variables(self):
        return self._variables
        
    @property
    def classes(self):
        return self._classes
        
    @property
    def functions(self):
        return self._functions
        
    @property
    def macros(self):
        return self._macros
        
    @property
    def headers(self):
        return self._headers
        
    @property
    def static_functions(self):
        return self._static_functions
        
    @property
    def cls_constructs(self):
        return self._cls_constructs
        
    @property
    def sons_scopes(self):
        return self._sons_scopes
        
    @property
    def symbolic_functions(self):
        return self._symbolic_functions
        
    @property
    def python_functions(self):
        return self._python_functions
        
    @property
    def is_loop(self):
        return self._is_loop
        
    @property
    def loops(self):
        return self._loops


        

#==============================================================================

class BasicParser(object):

    """ Class for a base Parser."""

    def __init__(self,
                 debug=False,
                 headers=None,
                 static=None,
                 show_traceback=False,
                 output_folder=''):
        """Parser constructor.

        debug: bool
            True if in debug mode.

        headers: list, tuple
            list of headers to append to the namespace

        static: list/tuple
            a list of 'static' functions as strings

        show_traceback: bool
            prints Tracebacke exception if True

        """
        self._fst = None
        self._ast = None

        self._filename  = None
        self._metavars  = OrderedDict()
        self._namespace = Scope()


        self._output_folder    = output_folder

        # represent the namespace of a function

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

        self._blocking = False

        # printing exception

        self._show_traceback = show_traceback

        if headers:
            if not isinstance(headers, dict):
                raise TypeError('Expecting a dict of headers')

        
            self.namespace.headers.update(headers)


        if static:
            if not isinstance(static, (list, tuple)):
                raise TypeError('Expecting a list/tuple of static')

            for i in static:
                if not isinstance(i, str):
                    raise TypeError('Expecting str. given {}'.format(type(i)))

            self._namespace.static_functions.extend(static)

    @property
    def namespace(self):
        return self._namespace

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

    @property
    def metavars(self):
        return self._metavars

    @property
    def current_class(self):
        return self._current_class

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

    @property
    def show_traceback(self):
        return self._show_traceback

    # TODO shall we need to export the Parser too?


    def insert_function(self, func):
        """."""

        if isinstance(func, SympyFunction):
            self.insert_symbolic_function(func)
        elif isinstance(func, PythonFunction):
            self.insert_python_function(func)
        elif isinstance(func, (FunctionDef, Interface)):
            container = self.namespace.functions
            container[str(func.name)] = func
        else:
            raise TypeError('Expected a Function definition')

    def insert_symbolic_function(self, func):
        """."""
        
        container = self.namespace.symbolic_functions
        if isinstance(func, SympyFunction):
            container[str(func.name)] = func
        elif isinstance(func, SymbolicAssign) and isinstance(func.rhs,
                Lambda):
            container[str(func.lhs)] = func.rhs
        else:
            raise TypeError('Expected a symbolic_function')

    def insert_python_function(self, func):
        """."""

        container = self.namespace.python_functions
        
        if isinstance(func, PythonFunction):
            container[str(func.name)] = func
        else:
            raise TypeError('Expected a python_function')

    def insert_import(self, expr):
        """."""

        # this method is only used in the syntatic stage
        
        if not isinstance(expr, Import):
            raise TypeError('Expecting Import expression')
        container = self.namespace.imports['imports']
        
        # if source is not specified, imported things are treated as sources
        source = expr.source
        if source is None:
            for t in expr.target:
                name = str(t)
                container[name] = []
        else:
            source = str(source)
            if not source in pyccel_builtin_import_registery:
                for t in expr.target:
                    name = [str(t)]
                    if not source in container.keys():
                        container[source] = []
                    container[source] += name

    def print_namespace(self):

        # TODO improve spacing

        print ('------- namespace -------')
        for (k, v) in self.namespace.items():
            print ('{var} \t :: \t {dtype}'.format(var=k, dtype=type(v)))
        print ('-------------------------')

    def view_namespace(self, entry):

        # TODO improve

        try:
            from tabulate import tabulate

            table = []
            for (k, v) in self.namespace[entry].items():
                k_str = '{}'.format(k)
                if entry == 'imports':
                    if v is None:
                        v_str = '*'
                    else:
                        v_str = '{}'.format(v)
                elif entry == 'variables':
                    v_str = '{}'.format(type(v))
                else:
                    raise NotImplementedError('TODO')

                line = [k_str, v_str]
                table.append(line)
            headers = ['module', 'target']

#            txt = tabulate(table, headers, tablefmt="rst")

            txt = tabulate(table, tablefmt='rst')
            print (txt)
        except:

            print ('------- namespace.{} -------'.format(entry))
            for (k, v) in self.namespace[entry].items():
                print ('{var} \t :: \t {value}'.format(var=k, value=v))
            print ('-------------------------')

    def _visit(self, expr, **settings):
        raise NotImplementedError('Must be implemented by the extension')

#==============================================================================


if __name__ == '__main__':
    import sys

    try:
        filename = sys.argv[1]
    except:
        raise ValueError('Expecting an argument for filename')

    parser = BasicParser(filename)
