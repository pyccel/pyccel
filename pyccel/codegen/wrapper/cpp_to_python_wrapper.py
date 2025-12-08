#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CppToPythonWrapper
which creates an interface exposing C++ code to Python using pybind11.
"""
from pyccel.ast.core          import Import, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core          import FunctionAddress, FunctionCall, Module
from pyccel.ast.cwrapper      import PyccelPyObject
from pyccel.ast.cwrapper      import PyModule, PyModInitFunc, PyFunctionDef
from pyccel.ast.literals      import Nil
from pyccel.ast.pybind        import FunctionDeclaration
from pyccel.ast.variable      import Variable
from pyccel.parser.scope      import Scope
from pyccel.errors.errors     import Errors
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO
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
        for f in expr.funcs:
            body.append(FunctionDeclaration(self._python_object_map[f], module_var, f))

        # TODO: Save module variables to the module variable

        self.exit_scope()

        return PyModInitFunc(mod_name, body, [module_var], func_scope)

    def _get_untranslatable_function(self, name, scope, original_function, error_msg):
        """
        Create code for a function complaining about an object which cannot be wrapped.

        Certain functions are not handled in the wrapper (e.g. private),
        This creates a wrapper function which raises NotImplementedError
        exception and returns NULL.

        Parameters
        ----------
        name : str
            The name of the generated function.

        scope : Scope
            The scope of the generated function.

        original_function : FunctionDef
           The function we were trying to wrap.

        error_msg : str
            The message to be raised in the NotImplementedError.

        Returns
        -------
        PyFunctionDef
            The new function which raises the error.
        """
        current_scope = self.scope
        self.scope = scope
        func_args = [FunctionDefArgument(Variable(PyccelPyObject(),
                           self.scope.get_new_name(name),
                           is_temp=True)) for a in func_args]
        function = PyFunctionDef(name = name, arguments = func_args, results = Nil(),
                body = [PyErr_Throw(PyNotImplementedError, CStrStr(LiteralString(error_msg)))],
                scope = scope, original_function = original_function)

        self.scope = current_scope

        self.scope.insert_function(function, self.scope.get_python_name(name))

        return function

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
        funcs = [self._wrap(f) for f in expr.funcs]

        # TODO: Wrap interfaces

        init_func = self._build_module_init_function(expr, expr.imports)

        #API_var, import_func = self._build_module_import_function(expr)

        self.exit_scope()

        imports = [Import(mod_scope.get_python_name(expr.name), expr)]
        original_mod_name = expr.scope.get_python_name(name)
        return PyModule(original_mod_name, (), funcs = funcs, imports = imports,
                        interfaces = (), classes = (), scope = mod_scope,
                        init_func = init_func, import_func = None,
                        module_def_name = None)

    def _wrap_FunctionDef(self, expr):
        func_name = self.scope.get_new_name(expr.name+'_wrapper', object_type = 'wrapper')
        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope
        original_func_name = expr.scope.get_python_name(expr.name)
        func_scope.insert_symbol(func_name)

        if expr.is_private:
            self.exit_scope()
            return EmptyNode()

        # Handle un-wrappable functions
        if any(isinstance(a.var, FunctionAddress) for a in expr.arguments):
            self.exit_scope()
            warnings.warn("Functions with functions as arguments will not be callable from Python")
            return self._get_untranslatable_function(func_name,
                         func_scope, expr,
                         "Cannot pass a function as an argument")

        # Add the variables to the expected symbols in the scope
        for a in expr.arguments:
            a_var = a.var
            func_scope.insert_symbol(getattr(a_var, 'original_var', a_var).name)

        args_code = [self._extract_FunctionDefArgument(a) for a in expr.arguments]

        body = [line for a in args_code for line in a['body']]

        func_args = [a['wrapper_arg'] for a in args_code]
        call_args = [a['func_arg'] for a in args_code]

        func_results = FunctionDefResult(Nil())

        imported_expr = expr.clone(expr.name, is_imported=True)
        mod, = expr.get_direct_user_nodes(lambda m: isinstance(m, Module))
        imported_expr.set_current_user_node(mod)
        body.append(FunctionCall(imported_expr, call_args))

        self.exit_scope()

        function = PyFunctionDef(func_name, func_args, body, func_results, scope=func_scope,
                docstring = expr.docstring, original_function = expr)

        self.scope.insert_function(function, func_scope.get_python_name(func_name))
        self._python_object_map[expr] = function

        return function

    def _extract_FunctionDefArgument(self, expr):
        self.scope.insert_symbol(expr.name)

        var = expr.var

        class_type = var.class_type

        classes = type(class_type).__mro__
        for cls in classes:
            annotation_method = f'_extract_{cls.__name__}_FunctionDefArgument'
            if hasattr(self, annotation_method):
                func_def_arg_info = getattr(self, annotation_method)(var)
                func_def_arg_info['wrapper_arg'] = FunctionDefArgument(func_def_arg_info['wrapper_arg'],
                                                                       value = expr.value,
                                                                       posonly = expr.is_posonly,
                                                                       kwonly = expr.is_kwonly,
                                                                       bound_argument = expr.bound_argument,
                                                                       is_vararg = expr.is_vararg,
                                                                       is_kwarg = expr.is_kwarg)
                return func_def_arg_info


        # Unknown object, we raise an error.
        return errors.report(f"Wrapping function arguments is not implemented for type {class_type}. "+PYCCEL_RESTRICTION_TODO, symbol=var,
            severity='fatal')

    def _extract_FixedSizeType_FunctionDefArgument(self, arg_var):
        local_arg = arg_var.clone(arg_var.name)
        return {'body': [], 'wrapper_arg': arg_var, 'func_arg': arg_var}
