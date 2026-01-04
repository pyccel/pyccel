#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CppToPythonWrapper
which creates an interface exposing C++ code to Python using pybind11.
"""
from pyccel.ast.core          import Import, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core          import FunctionAddress, FunctionCall, Module, Assign
from pyccel.ast.core          import Return, EmptyNode
from pyccel.ast.datatypes     import pyccel_type_to_original_type
from pyccel.ast.cwrapper      import PyccelPyObject, PyErr_SetString
from pyccel.ast.cwrapper      import PyNotImplementedError
from pyccel.ast.cwrapper      import PyModule, PyModInitFunc, PyFunctionDef
from pyccel.ast.literals      import Nil
from pyccel.ast.pybind        import FunctionDeclaration
from pyccel.ast.variable      import Variable, IndexedElement
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
                           is_temp=True)) for _ in original_function.args]
        function = PyFunctionDef(name = name, arguments = func_args, results = Nil(),
                body = [PyErr_SetString(PyNotImplementedError, LiteralString(error_msg))],
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
        """
        Build a `PyFunctionDef` from a `FunctionDef`.

        Create a `PyFunctionDef` which wraps a C++ `FunctionDef`.

        Parameters
        ----------
        expr : FunctionDef
            The function which can be called from C++.

        Returns
        -------
        PyFunctionDef
            The function which can be called from Python.
        """
        func_name = self.scope.get_new_name(expr.name+'_wrapper', object_type = 'wrapper')
        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope
        func_scope.insert_symbol(func_name)

        if expr.is_private:
            self.exit_scope()
            return EmptyNode()

        # Handle un-wrappable functions
        if any(isinstance(a.var, FunctionAddress) for a in expr.arguments):
            self.exit_scope()
            errors.report("Functions with functions as arguments will not be callable from Python",
                          severity='warning', symbol=expr)
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

        imported_expr = expr.clone(expr.name, is_imported=True)
        mod, = set(expr.get_direct_user_nodes(lambda m: isinstance(m, Module)))
        imported_expr.set_current_user_node(mod)
        if expr.results.var is Nil():
            func_results = FunctionDefResult(Nil())
            body.append(FunctionCall(imported_expr, call_args))
        else:
            result_info = self._wrap_FunctionDefResult(expr.results)
            func_results = FunctionDefResult(result_info['wrapper_result'])
            call_out = result_info['func_result']
            body.append(Assign(call_out, FunctionCall(imported_expr, call_args)))
            self.scope.insert_variable(call_out)
            body.append(Return(call_out))

        self.exit_scope()

        function = PyFunctionDef(func_name, func_args, body, func_results,
                                 scope=func_scope, docstring = expr.docstring,
                                 original_function = expr)

        self.scope.insert_function(function, func_scope.get_python_name(func_name))
        self._python_object_map[expr] = function

        return function

    def _extract_FunctionDefArgument(self, expr):
        """
        Get a pybind11 compatible function argument.

        Get a pybind11 compatible function argument.

        Parameters
        ----------
        expr : FunctionDefArgument
            The argument of the C++ function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates
                        the C++ variable into a pybind11 compatible variable.
             - wrapper_arg : the Variable describing the argument of the wrapped function.
             - func_arg : the Variable which should be passed to call the C++ function being wrapped.
        """
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
        return errors.report(f"Wrapping function arguments is not implemented for type {class_type}. "+PYCCEL_RESTRICTION_TODO,
                             symbol=var, severity='fatal')

    def _extract_FixedSizeType_FunctionDefArgument(self, arg_var):
        """
        Extract the nodes which describe an argument of FixedSizeType.

        Extract the nodes which describe an argument of FixedSizeType.
        The nodes describe the necessary variables and any code required to
        unpack those variables.

        Parameters
        ----------
        expr : Variable
            The argument of the C++ function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates
                        the C++ variable into a pybind11 compatible variable.
             - wrapper_arg : the Variable describing the argument of the wrapped function.
             - func_arg : the Variable which should be passed to call the C++ function being wrapped.
        """
        local_arg = arg_var.clone(arg_var.name)
        return {'body': [], 'wrapper_arg': local_arg, 'func_arg': local_arg}

    def _extract_StringType_FunctionDefArgument(self, arg_var):
        """
        Extract the nodes which describe an argument of StringType.

        Extract the nodes which describe an argument of StringType.
        The nodes describe the necessary variables and any code required to
        unpack those variables.

        Parameters
        ----------
        expr : Variable
            The argument of the C++ function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates
                        the C++ variable into a pybind11 compatible variable.
             - wrapper_arg : the Variable describing the argument of the wrapped function.
             - func_arg : the Variable which should be passed to call the C++ function being wrapped.
        """
        local_arg = arg_var.clone(arg_var.name)
        return {'body': [], 'wrapper_arg': local_arg, 'func_arg': local_arg}

    def _wrap_FunctionDefResult(self, expr):
        """
        Get a pybind11 compatible function result.

        Get a pybind11 compatible function result. This function simply
        calls _extract_FunctionDefResult.

        Parameters
        ----------
        expr : FunctionDefResult
            The result of the C++ function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates
                        the C++ variable into a pybind11 compatible variable.
             - wrapper_result : the Variable describing the argument of the wrapped function.
             - func_result : the Variable which should be passed to call the C++ function being wrapped.
        """
        var = expr.var
        return self._extract_FunctionDefResult(var)

    def _extract_FunctionDefResult(self, var):
        """
        Get a pybind11 compatible function argument.

        Get a pybind11 compatible function argument. This function takes a
        Variable or an IndexedElement so it can be called from other extraction
        functions.

        Parameters
        ----------
        expr : Variable | IndexedElement
            The argument of the C++ function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates
                        the C++ variable into a pybind11 compatible variable.
             - wrapper_result : the Variable describing the argument of the wrapped function.
             - func_result : the Variable which should be passed to call the C++ function being wrapped.
        """
        if isinstance(var, Variable):
            self.scope.insert_symbol(var.name)
        class_type = var.class_type

        classes = type(class_type).__mro__
        for cls in classes:
            annotation_method = f'_extract_{cls.__name__}_FunctionDefResult'
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(var)

        # Unknown object, we raise an error.
        return errors.report(f"Wrapping function results is not implemented for type {class_type}. "+PYCCEL_RESTRICTION_TODO, symbol=var,
            severity='fatal')

    def _extract_FixedSizeType_FunctionDefResult(self, res_var):
        """
        Extract the nodes which describe a result of FixedSizeType.

        Extract the nodes which describe a result of FixedSizeType.
        The nodes describe the necessary variables and any code required to
        pack those variables.

        Parameters
        ----------
        expr : Variable
            The result of the C++ function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates
                        the C++ variable into a pybind11 compatible variable.
             - wrapper_result : the Variable describing the argument of the wrapped function.
             - func_result : the Variable which should be passed to call the C++ function being wrapped.
        """
        out_type = pyccel_type_to_original_type[res_var.class_type]
        if out_type.__module__ == 'numpy':
            errors.report("NumPy return types are not yet handled")
        if isinstance(res_var, IndexedElement):
            local_var = Variable(res_var.class_type, self.scope.get_new_name())
            self.scope.insert_symbolic_alias(res_var, local_var)
        else:
            local_var = res_var.clone(res_var.name)
        return {'wrapper_result': local_var, 'func_result': local_var, 'body': []}

    def _extract_StringType_FunctionDefResult(self, res_var):
        """
        Extract the nodes which describe a result of StringType.

        Extract the nodes which describe a result of StringType.
        The nodes describe the necessary variables and any code required to
        pack those variables.

        Parameters
        ----------
        expr : Variable
            The result of the C++ function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates
                        the C++ variable into a pybind11 compatible variable.
             - wrapper_result : the Variable describing the argument of the wrapped function.
             - func_result : the Variable which should be passed to call the C++ function being wrapped.
        """
        if isinstance(res_var, IndexedElement):
            local_var = Variable(res_var.class_type, self.scope.get_new_name())
            self.scope.insert_symbolic_alias(res_var, local_var)
        else:
            local_var = res_var.clone(res_var.name)
        return {'wrapper_result': local_var, 'func_result': local_var, 'body': []}

    def _extract_InhomogeneousTupleType_FunctionDefResult(self, res_var):
        """
        Extract the nodes which describe a result of InhomogeneousTupleType.

        Extract the nodes which describe a result of InhomogeneousTupleType.
        The nodes describe the necessary variables and any code required to
        pack those variables.

        Parameters
        ----------
        expr : Variable
            The result of the C++ function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates
                        the C++ variable into a pybind11 compatible variable.
             - wrapper_result : the Variable describing the argument of the wrapped function.
             - func_result : the Variable which should be passed to call the C++ function being wrapped.
        """
        info = [self._extract_FunctionDefResult(v) for v in res_var]
        body = [l for i in info for l in i['body']]
        local_var = res_var.clone(res_var.name, is_temp = False)
        return {'wrapper_result': local_var, 'func_result': local_var, 'body': body}
