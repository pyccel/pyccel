#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CppToPythonWrapper
which creates an interface exposing C++ code to Python using pybind11.
"""
from pyccel.ast.core          import Import
from pyccel.ast.cwrapper      import PyccelPyObject
from pyccel.ast.cwrapper      import PyModule, PyModInitFunc
from pyccel.ast.literals      import Nil
from pyccel.ast.variable      import Variable
from pyccel.parser.scope      import Scope
from pyccel.errors.errors     import Errors
from .wrapper                 import Wrapper

errors = Errors()

class CppToPythonWrapper(Wrapper):
    """
    Class for creating a wrapper exposing C++ code to Python.

    A class which provides all necessary functions for wrapping different AST
    objects such that the resulting AST is Python-compatible.

    Parameters
    ----------
    sharedlib_dirpath : str
        The folder where the generated .so file will be located.
    verbose : int
        The level of verbosity.
    """
    target_language = 'Python'
    start_language = 'C++'

    def __init__(self, sharedlib_dirpath, verbose):
        # A map used to find the Python-compatible Variable equivalent to an object in the AST
        self._python_object_map = {}
        # The object that should be returned to indicate an error
        self._error_exit_code = Nil()

        self._sharedlib_dirpath = sharedlib_dirpath
        super().__init__(verbose)

    def _build_module_init_function(self, expr, imports):
        """
        Build the function that will be called when the module is first imported.

        Build the function that will be called when the module is first imported.
        This function must call any initialisation function of the underlying
        module and must add any variables to the module variable.

        Parameters
        ----------
        expr : Module
            The module of interest.

        imports : list of Import
            A list of any imports that will appear in the PyModule.

        Returns
        -------
        PyModInitFunc
            The initialisation function.
        """
        mod_name = expr.scope.get_python_name(expr.name)
        # Initialise the scope
        func_scope = self.scope.new_child_scope(f'PyInit_{mod_name}', 'function')
        self.scope = func_scope

        module_var = Variable(PyccelPyObject(),
                       self.scope.get_new_name("mod"))
        self.scope.insert_variable(module_var)

        body = []
        # TODO: Variables

        # Call the initialisation function
        if expr.init_func:
            init_func_clone = expr.init_func.clone(expr.init_func.name, is_imported=True)
            init_func_clone.set_current_user_node(expr)
            body.append(init_func_clone())

        # TODO: Save classes to the module variable

        # TODO: Save functions/interfaces to the module variable

        # TODO: Save module variables to the module variable

        self.exit_scope()

        return PyModInitFunc(mod_name, body, [module_var], func_scope)

    #--------------------------------------------------------------------------------------------------------------------------------------------
    # Wrap functions
    #--------------------------------------------------------------------------------------------------------------------------------------------

    def _wrap_Module(self, expr):
        """
        Build a `PyModule` from a `Module`.

        Create a `PyModule` which wraps a C-compatible `Module`.

        Parameters
        ----------
        expr : Module
            The module which can be called from C.

        Returns
        -------
        PyModule
            The module which can be called from Python.
        """
        # Define scope
        scope = expr.scope
        name = expr.name

        mod_scope = Scope(name = name, used_symbols = scope.local_used_symbols.copy(),
                          original_symbols = scope.python_names.copy(), scope_type = 'module')
        self.scope = mod_scope

        # TODO: Wrap classes

        # TODO: Wrap functions

        # TODO: Wrap interfaces

        init_func = self._build_module_init_function(expr, expr.imports)

        #API_var, import_func = self._build_module_import_function(expr)

        self.exit_scope()

        imports = [Import(mod_scope.get_python_name(expr.name), expr)]
        original_mod_name = expr.scope.get_python_name(name)
        return PyModule(original_mod_name, [], (), imports = imports,
                        interfaces = (), classes = (), scope = mod_scope,
                        init_func = init_func, import_func = None,
                        module_def_name = None)
