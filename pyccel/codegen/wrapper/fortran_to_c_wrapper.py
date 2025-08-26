# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : FortranToCWrapper
which creates an interface exposing Fortran code to C.
"""
from functools import reduce
import warnings
from pyccel.ast.bind_c import BindCPointer, BindCFunctionDef, C_F_Pointer
from pyccel.ast.bind_c import CLocFunc, BindCModule, BindCModuleVariable
from pyccel.ast.bind_c import BindCArrayVariable, BindCClassDef, DeallocatePointer
from pyccel.ast.bind_c import BindCClassProperty, c_malloc, BindCSizeOf
from pyccel.ast.bind_c import BindCVariable, BindCArrayType, C_NULL_CHAR
from pyccel.ast.builtins import VariableIterator, PythonRange
from pyccel.ast.builtin_methods.list_methods import ListAppend
from pyccel.ast.builtin_methods.set_methods import SetAdd
from pyccel.ast.builtin_methods.dict_methods import DictItems
from pyccel.ast.core import Assign, FunctionCallArgument
from pyccel.ast.core import Allocate, EmptyNode, FunctionAddress
from pyccel.ast.core import If, IfSection, Import, Interface, FunctionDefArgument
from pyccel.ast.core import AsName, Module, AliasAssign, FunctionDefResult
from pyccel.ast.core import For
from pyccel.ast.datatypes import CustomDataType, FixedSizeNumericType
from pyccel.ast.datatypes import TupleType
from pyccel.ast.datatypes import PythonNativeInt, CharType
from pyccel.ast.datatypes import InhomogeneousTupleType
from pyccel.ast.internals import Slice
from pyccel.ast.literals import LiteralInteger, Nil, LiteralTrue, LiteralString
from pyccel.ast.numpytypes import NumpyNDArrayType, NumpyInt32Type
from pyccel.ast.operators import PyccelIsNot, PyccelMul, PyccelAdd
from pyccel.ast.variable import Variable, IndexedElement, DottedVariable
from pyccel.errors.errors import Errors
from pyccel.errors.messages import PYCCEL_RESTRICTION_TODO
from pyccel.parser.scope import Scope
from .wrapper import Wrapper

errors = Errors()

class FortranToCWrapper(Wrapper):
    """
    Class for creating a wrapper exposing Fortran code to C.

    A class which provides all necessary functions for wrapping different AST
    objects such that the resulting AST is C-compatible. This new AST is
    printed as an intermediary layer.

    Parameters
    ----------
    sharedlib_dirpath : str
        The folder where the generated .so file will be located.
    verbose : int
        The level of verbosity.
    """
    target_language = 'C'
    start_language = 'Fortran'

    def __init__(self, sharedlib_dirpath, verbose):
        self._additional_exprs = []
        self._wrapper_names_dict = {}
        super().__init__(verbose)

    def _get_function_def_body(self, func, wrapped_args, results, handled = ()):
        """
        Get the body of the bind c function definition.

        Get the body of the bind c function definition by inserting if blocks
        to check the presence of optional variables. Once we have ascertained
        the presence of the variables the original function is called. This
        code slices array variables to ensure the correct step.

        Parameters
        ----------
        func : FunctionDef
            The function which should be called.

        wrapped_args : list[dict]
            A list containing the dictionaries returned by _extract_FunctionDefArgument.

        results : list of Variables
            The Variables where the result of the function call will be saved.

        handled : tuple
            A list of all variables which have been handled (checked to see if they
            are present).

        Returns
        -------
        list
            A list of Basic nodes describing the body of the function.
        """
        next_optional_arg = next((a for a in wrapped_args if a['c_arg'].var.original_var.is_optional and a not in handled), None)
        if next_optional_arg:
            args = wrapped_args.copy()
            optional_var = next_optional_arg['c_arg'].var
            optional_var = getattr(optional_var, 'new_var', optional_var)
            class_type = optional_var.class_type
            if isinstance(class_type, BindCArrayType):
                optional_var = self.scope.collect_tuple_element(optional_var[0])
            elif isinstance(class_type, InhomogeneousTupleType):
                errors.report("Wrapping of optional inhomogeneous tuples is not yet implemented.\n"+PYCCEL_RESTRICTION_TODO,
                        symbol = func, severity='error')
            handled += (next_optional_arg, )
            true_section = IfSection(PyccelIsNot(optional_var, Nil()),
                                    self._get_function_def_body(func, args, results, handled))
            args.remove(next_optional_arg)
            false_section = IfSection(LiteralTrue(),
                                    self._get_function_def_body(func, args, results, handled))
            return [If(true_section, false_section)]
        else:
            args = [a['f_arg'] for a in wrapped_args]
            body = [line for a in wrapped_args for line in a['body']]

            if len(results) == 1:
                res = results[0]
                func_call = AliasAssign(res, func(*args)) if res.is_alias else \
                            Assign(res, func(*args))
            else:
                func_call = Assign(results, func(*args))
            return body + [func_call]

    def _wrap_Module(self, expr):
        """
        Create a BindCModule which is compatible with C.

        Create a BindCModule which provides an interface between C and the
        Module described by expr. This includes wrapping functions,
        interfaces, classes and module variables.

        Parameters
        ----------
        expr : pyccel.ast.core.Module
            The module to be wrapped.

        Returns
        -------
        pyccel.ast.bind_c.BindCModule
            The C-compatible module.
        """
        # Define scope
        scope = expr.scope
        mod_scope = Scope(used_symbols = scope.local_used_symbols.copy(), original_symbols = scope.python_names.copy())
        self.scope = mod_scope

        # Wrap contents
        # We only wrap the non inlined functions
        funcs_to_wrap = [f for f in expr.funcs if f.is_semantic and not f.is_inline]

        funcs = [self._wrap(f) for f in funcs_to_wrap]
        if expr.init_func:
            init_func = funcs[next(i for i,f in enumerate(funcs_to_wrap) if f == expr.init_func)]
        else:
            init_func = None
        if expr.free_func:
            free_func = funcs[next(i for i,f in enumerate(funcs_to_wrap) if f == expr.free_func)]
        else:
            free_func = None
        removed_functions = [f for f,w in zip(funcs_to_wrap, funcs) if isinstance(w, EmptyNode)]
        funcs = [f for f in funcs if not isinstance(f, EmptyNode)]
        interfaces = [self._wrap(f) for f in expr.interfaces if not f.is_inline]
        classes = [self._wrap(f) for f in expr.classes]
        variables = [self._wrap(v) for v in expr.variables if not v.is_private]
        variable_getters = [v for v in variables if isinstance(v, BindCArrayVariable)]
        # Import the module and its dependencies (in case they are used for argument types)
        imports = [Import(self.scope.get_python_name(expr.name), target = expr, mod=expr),
                   *expr.imports]

        name = mod_scope.get_new_name(f'bind_c_{expr.name}')
        self._wrapper_names_dict[expr.name] = name

        self.exit_scope()

        return BindCModule(name, variables, funcs, variable_wrappers = variable_getters,
                init_func = init_func, free_func = free_func,
                interfaces = interfaces, classes = classes,
                imports = imports, original_module = expr,
                scope = mod_scope, removed_functions = removed_functions)

    def _wrap_FunctionDef(self, expr):
        """
        Create a C-compatible function which executes the original function.

        Create a function which can be called from C which internally calls the original
        function. It does this by wrapping the arguments and the results and unrolling
        the body using self._get_function_def_body to ensure optional arguments are
        present before accessing them. With all this information a BindCFunctionDef is
        created which is C-compatible.

        Functions which cannot be wrapped raise a warning and return an EmptyNode. This
        is the case for functions with functions as arguments.

        Parameters
        ----------
        expr : FunctionDef
            The function to wrap.

        Returns
        -------
        BindCFunctionDef
            The C-compatible function.
        """
        if expr.is_private or expr.is_inline:
            return EmptyNode()

        orig_name = expr.cls_name or expr.name
        name = self.scope.get_new_name(f'bind_c_{orig_name.lower()}')
        self._wrapper_names_dict[expr.name] = name
        in_cls = expr.arguments and expr.arguments[0].bound_argument

        self._additional_exprs = []

        if any(isinstance(a.var, FunctionAddress) for a in expr.arguments):
            warnings.warn("Functions with functions as arguments cannot be wrapped by pyccel")
            return EmptyNode()

        # Create the scope
        func_scope = self.scope.new_child_scope(name)
        self.scope = func_scope

        # Wrap the arguments and collect the expressions passed as the call argument.
        wrapped_args = [self._extract_FunctionDefArgument(a, expr) for a in expr.arguments]
        func_arguments = [a['c_arg'] for a in wrapped_args]
        call_arguments = [a['f_arg'] for a in wrapped_args]
        func_to_call = {fa : ca for ca, fa in zip(call_arguments, func_arguments)}

        if expr.results.var is Nil():
            func_results = Nil()
            func_call_results = []
        else:
            result = self._extract_FunctionDefResult(expr.results.var, expr.scope)
            self._additional_exprs.extend(result['body'])
            func_results = result['c_result']
            func_call_results = self.scope.collect_all_tuple_elements(result['f_result'])

        interface = expr.get_direct_user_nodes(lambda u: isinstance(u, Interface))

        if in_cls and interface:
            body = self._get_function_def_body(interface[0], wrapped_args, func_call_results)
        else:
            body = self._get_function_def_body(expr, wrapped_args, func_call_results)

        body.extend(self._additional_exprs)
        self._additional_exprs.clear()

        if expr.scope.get_python_name(expr.name) == '__del__':
            body.append(DeallocatePointer(call_arguments[0].value))

        self.exit_scope()

        func = BindCFunctionDef(name, func_arguments, body, FunctionDefResult(func_results), scope=func_scope, original_function = expr,
                docstring = expr.docstring, result_pointer_map = expr.result_pointer_map)

        self.scope.insert_function(func, name)

        return func

    def _wrap_Interface(self, expr):
        """
        Create an interface containing only C-compatible functions.

        Create an interface containing only functions which can be called from C
        from an interface which is not necessarily C-compatible.

        Parameters
        ----------
        expr : pyccel.ast.core.Interface
            The interface to be wrapped.

        Returns
        -------
        pyccel.ast.core.Interface
            The C-compatible interface.
        """
        functions = [self._wrap(f) for f in expr.functions if not isinstance(f, EmptyNode)]
        return Interface(expr.name, functions, expr.is_argument)

    def _extract_FunctionDefArgument(self, expr, func):
        """
        Extract the C-compatible FunctionDefArgument from the Fortran FunctionDefArgument.

        Extract the C-compatible FunctionDefArgument from the Fortran FunctionDefArgument.

        The extraction is done by finding the appropriate function
        _extract_X_FunctionDefArgument for the object expr. X is the class type of the
        variable stored in the object expr. If this function does not exist then the
        method resolution order is used to search for other compatible
        _extract_X_FunctionDefArgument functions. If none are found then an error is raised.

        Parameters
        ----------
        expr : FunctionDefArgument
            An object representing the FunctionDefArgument in the Fortran code which should
            be exposed to the C code.

        func : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to access the argument.
        """
        var = expr.var
        class_type = var.class_type

        classes = type(class_type).__mro__
        for cls in classes:
            annotation_method = f'_extract_{cls.__name__}_FunctionDefArgument'
            if hasattr(self, annotation_method):
                func_def_argument_dict = getattr(self, annotation_method)(var, func)
                new_var = func_def_argument_dict['c_arg']
                func_def_argument_dict['c_arg'] = FunctionDefArgument(new_var, value = expr.value,
                    posonly = expr.is_posonly, kwonly = expr.is_kwonly, annotation = expr.annotation,
                    bound_argument = expr.bound_argument, persistent_target = expr.persistent_target,
                    is_vararg = expr.is_vararg, is_kwarg = expr.is_kwarg)
                func_def_argument_dict['f_arg'] = FunctionCallArgument(func_def_argument_dict['f_arg'],
                                                                        keyword = expr.name)
                return func_def_argument_dict

        # Unknown object, we raise an error.
        return errors.report(f"Wrapping function arguments is not implemented for type {class_type}. "
                + PYCCEL_RESTRICTION_TODO, symbol=var, severity='fatal')

    def _extract_FixedSizeNumericType_FunctionDefArgument(self, var, func):
        name = var.name
        self.scope.insert_symbol(name)
        collisionless_name = self.scope.get_expected_name(name)
        if var.is_optional:
            f_arg = var.clone(collisionless_name, new_class = Variable, is_argument = False,
                    is_optional = False, memory_handling='alias')
            new_var = Variable(BindCPointer(), self.scope.get_new_name(f'bound_{name}'),
                                is_argument = True, is_optional = False, memory_handling='alias')
            body = [C_F_Pointer(new_var, f_arg)]
        else:
            f_arg = var.clone(collisionless_name, new_class = Variable,
                        is_argument = True)
            new_var = f_arg
            body = []
        self.scope.insert_variable(f_arg)
        return {'c_arg': BindCVariable(new_var, var), 'f_arg': f_arg, 'body': body}

    def _extract_CustomDataType_FunctionDefArgument(self, var, func):
        name = var.name
        self.scope.insert_symbol(name)
        collisionless_name = self.scope.get_expected_name(name)
        f_arg = var.clone(collisionless_name, new_class = Variable, is_argument = False,
                is_optional = False, memory_handling='alias')
        new_var = Variable(BindCPointer(), self.scope.get_new_name(f'bound_{name}'),
                            is_argument = True, is_optional = False, memory_handling='alias')
        body = [C_F_Pointer(new_var, f_arg)]
        self.scope.insert_variable(f_arg)
        return {'c_arg': BindCVariable(new_var, var), 'f_arg': f_arg, 'body': body}

    def _extract_NumpyNDArrayType_FunctionDefArgument(self, var, func):
        name = var.name
        scope = self.scope
        scope.insert_symbol(name)
        collisionless_name = scope.get_expected_name(name)
        rank = var.rank
        order = var.order
        bind_var = Variable(BindCPointer(), scope.get_new_name(f'bound_{name}'),
                            is_argument = True, is_optional = False, memory_handling='alias')
        arg_var = var.clone(collisionless_name, is_argument = False, is_optional = False,
                            memory_handling = 'alias', allows_negative_indexes=False,
                            new_class = Variable)
        scope.insert_variable(arg_var)
        scope.insert_variable(bind_var)

        shape   = [scope.get_temporary_variable(PythonNativeInt(), name=f'{name}_shape_{i+1}', is_argument = True)
                   for i in range(rank)]
        stride  = [scope.get_temporary_variable(PythonNativeInt(), name=f'{name}_stride_{i+1}', is_argument = True)
                   for i in range(rank)]

        orig_size = [PyccelMul(sh,st) for sh,st in zip(shape, stride)]
        body = [C_F_Pointer(bind_var, arg_var, orig_size[::-1] if order == 'C' else orig_size)]

        c_arg_var = Variable(BindCArrayType(rank, has_strides = True),
                        scope.get_new_name(), is_argument = True,
                        shape = (LiteralInteger(rank*2+1),))

        scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(0)), bind_var)
        for i,s in enumerate(shape):
            scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(i+1)), s)
        for i,s in enumerate(stride):
            scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(i+rank+1)), s)

        start = LiteralInteger(1) # C_F_Pointer leads to default Fortran lbound
        stop = None
        indexes = [Slice(start, stop, step) for step in stride]

        f_arg = IndexedElement(arg_var, *indexes)

        return {'c_arg': BindCVariable(c_arg_var, var), 'f_arg': f_arg, 'body': body}

    def _extract_HomogeneousTupleType_FunctionDefArgument(self, var, func):
        name = var.name
        scope = self.scope
        scope.insert_symbol(name)
        collisionless_name = scope.get_expected_name(name)
        rank = var.rank
        bind_var = Variable(BindCPointer(), scope.get_new_name(f'bound_{name}'),
                            is_argument = True, is_optional = False, memory_handling='alias')
        arg_var = var.clone(collisionless_name, is_argument = False, is_optional = False,
                            memory_handling = 'alias', allows_negative_indexes=False,
                            new_class = Variable)
        scope.insert_variable(arg_var)
        scope.insert_variable(bind_var)

        shape_var = scope.get_temporary_variable(PythonNativeInt(), name=f'{name}_size', is_argument = True)

        body = [C_F_Pointer(bind_var, arg_var, (shape_var,))]

        c_arg_var = Variable(BindCArrayType(rank, has_strides = False),
                        scope.get_new_name(), is_argument = True,
                        shape = (LiteralInteger(rank+1),))

        scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(0)), bind_var)
        scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(1)), shape_var)

        return {'c_arg': BindCVariable(c_arg_var, var), 'f_arg': arg_var, 'body': body}

    def _extract_StringType_FunctionDefArgument(self, var, func):
        name = var.name
        scope = self.scope
        scope.insert_symbol(name)
        collisionless_name = scope.get_expected_name(name)
        rank = var.rank
        bind_var = Variable(BindCPointer(), scope.get_new_name(f'bound_{name}'),
                            is_argument = True, is_optional = False, memory_handling='alias',
                            is_const = True)
        arg_var = var.clone(collisionless_name, is_argument = False, is_optional = False,
                            allows_negative_indexes=False, new_class = Variable)
        array_var = Variable(NumpyNDArrayType(CharType(), 1, None), scope.get_new_name(name),
                            memory_handling='alias')
        scope.insert_variable(arg_var)
        scope.insert_variable(bind_var)
        scope.insert_variable(array_var)

        shape_var = scope.get_temporary_variable(PythonNativeInt(), name=f'{name}_size', is_argument = True)

        for_scope = scope.create_new_loop_scope()
        iterator = PythonRange(LiteralInteger(1), PyccelAdd(shape_var, LiteralInteger(1)))
        idx = Variable(PythonNativeInt(), self.scope.get_new_name())
        iterator.set_loop_counter(idx)
        self.scope.insert_variable(idx)

        # Default Fortran arrays retrieved from C_F_Pointer are 1-indexed
        # Lists are 1-indexed but Pyccel adds the shift during printing so they are
        # treated as 0-indexed here
        for_body = [Assign(arg_var, PyccelAdd(arg_var, IndexedElement(array_var, idx)))]

        body = [C_F_Pointer(bind_var, array_var, (shape_var,)),
                Assign(arg_var, LiteralString('')),
                For((idx,), iterator, for_body, scope = for_scope)]

        c_arg_var = Variable(BindCArrayType(rank, has_strides = False),
                        scope.get_new_name(), is_argument = True,
                        shape = (LiteralInteger(2),))

        scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(0)), bind_var)
        scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(1)), shape_var)

        return {'c_arg': BindCVariable(c_arg_var, var), 'f_arg': arg_var, 'body': body}

    def _extract_HomogeneousContainerType_FunctionDefArgument(self, var, func):
        name = var.name
        scope = self.scope
        scope.insert_symbol(name)
        collisionless_name = scope.get_expected_name(name)
        rank = var.rank
        element_type = var.class_type.element_type
        bind_var = Variable(BindCPointer(), scope.get_new_name(f'bound_{name}'),
                            is_argument = True, is_optional = False, memory_handling='alias')
        arg_var = var.clone(collisionless_name, is_argument = False, is_optional = False,
                            allows_negative_indexes=False, new_class = Variable)
        scope.insert_variable(arg_var)
        scope.insert_variable(bind_var)

        shape_var = scope.get_temporary_variable(PythonNativeInt(), name=f'{name}_size', is_argument = True)
        local_var = Variable(NumpyNDArrayType(element_type, 1, None), scope.get_new_name(name),
                            shape = (shape_var,), memory_handling='alias')
        scope.insert_variable(local_var)

        for_scope = scope.create_new_loop_scope()
        iterator = PythonRange(LiteralInteger(1), PyccelAdd(shape_var, LiteralInteger(1)))
        idx = Variable(PythonNativeInt(), self.scope.get_new_name())
        iterator.set_loop_counter(idx)
        self.scope.insert_variable(idx)

        insert_call = {'set': SetAdd,
                       'list': ListAppend}

        if var.class_type.name not in insert_call:
            return errors.report(f"Wrapping function arguments is not implemented for type {var.class_type}. "
                    + PYCCEL_RESTRICTION_TODO, symbol=var, severity='fatal')

        # Default Fortran arrays retrieved from C_F_Pointer are 1-indexed
        # Lists are 1-indexed but Pyccel adds the shift during printing so they are
        # treated as 0-indexed here
        for_body = [insert_call[var.class_type.name](arg_var, IndexedElement(local_var, idx))]

        body = [C_F_Pointer(bind_var, local_var, (shape_var,)),
                Allocate(arg_var, shape = (shape_var,), status = 'unallocated',
                    alloc_type = 'reserve'),
                For((idx,), iterator, for_body, scope = for_scope)]

        c_arg_var = Variable(BindCArrayType(rank, has_strides = False),
                        scope.get_new_name(), is_argument = True,
                        shape = (LiteralInteger(2),))

        scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(0)), bind_var)
        scope.insert_symbolic_alias(IndexedElement(c_arg_var, LiteralInteger(1)), shape_var)

        return {'c_arg': BindCVariable(c_arg_var, var), 'f_arg': arg_var, 'body': body}

    def _wrap_Variable(self, expr):
        """
        Create all objects necessary to expose a module variable to C.

        Create and return the objects which must be printed in the wrapping
        module in order to expose the variable to C. In the case of scalar
        numerical values nothing needs to be done so an EmptyNode is returned.
        In the case of numerical arrays a C-compatible function must be created
        which returns the array. This is necessary because built-in Fortran
        arrays are not C-compatible. In the case of classes a C-compatible
        function is also created which returns a pointer to the class object.

        Parameters
        ----------
        expr : pyccel.ast.variables.Variable
            The module variable.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The AST object describing the code which must be printed in
            the wrapping module to expose the variable.
        """
        if isinstance(expr.class_type, FixedSizeNumericType):
            return expr.clone(expr.name, new_class = BindCModuleVariable)
        elif isinstance(expr.class_type, NumpyNDArrayType):
            scope = self.scope
            func_name = scope.get_new_name('bind_c_'+expr.name.lower())
            func_scope = scope.new_child_scope(func_name)
            mod = expr.get_user_nodes(Module)[0]
            import_mod = Import(mod.name, AsName(expr,expr.name), mod=mod)
            func_scope.imports['variables'][expr.name] = expr

            # Create the data pointer
            result_wrap = self._get_bind_c_array(expr.name, expr, expr.shape,
                                            pointer_target = True)
            func = BindCFunctionDef(name = func_name,
                          body      = result_wrap['body'],
                          arguments = [],
                          results   = FunctionDefResult(result_wrap['c_result']),
                          imports   = [import_mod],
                          scope = func_scope,
                          original_function = expr)
            return expr.clone(expr.name, new_class = BindCArrayVariable, wrapper_function = func,
                                original_variable = expr)
        else:
            raise NotImplementedError(f"Objects of type {expr.class_type} cannot be wrapped yet")

    def _wrap_DottedVariable(self, expr):
        """
        Create all objects necessary to expose a class attribute to C.

        Create the getter and setter functions which expose the class attribute
        to C. Return these objects in a BindCClassProperty.

        Parameters
        ----------
        expr : DottedVariable
            The class attribute.

        Returns
        -------
        BindCClassProperty
            An object containing the getter and setter functions which expose
            the class attribute to C.
        """
        lhs = expr.lhs
        class_dtype = lhs.dtype
        # ----------------------------------------------------------------------------------
        #                        Create getter
        # ----------------------------------------------------------------------------------
        getter_name = self.scope.get_new_name(f'{class_dtype.name}_{expr.name}_getter'.lower())
        getter_scope = self.scope.new_child_scope(getter_name)
        self.scope = getter_scope
        self.scope.insert_symbol(expr.name)
        getter_result_info = self._extract_FunctionDefResult(expr, lhs.cls_base.scope)
        getter_result = getter_result_info['c_result']

        getter_arg_wrapper = self._extract_FunctionDefArgument(FunctionDefArgument(lhs, bound_argument = True), expr)
        self_obj = getter_arg_wrapper['f_arg'].value
        getter_arg = getter_arg_wrapper['c_arg']

        getter_body = getter_arg_wrapper['body']

        attrib = expr.clone(expr.name, lhs = self_obj)
        wrapped_obj = self.scope.find(expr.name)
        # Cast the C variable into a Python variable
        if expr.rank > 0 or isinstance(expr.dtype, CustomDataType):
            getter_body.append(AliasAssign(wrapped_obj, attrib))
        else:
            getter_body.append(Assign(getter_result_info['f_result'], attrib))
        getter_body.extend(getter_result_info['body'])
        self._additional_exprs.clear()
        self.exit_scope()

        getter = BindCFunctionDef(getter_name, (getter_arg,), getter_body, FunctionDefResult(getter_result),
                                original_function = expr, scope = getter_scope)

        # ----------------------------------------------------------------------------------
        #                        Create setter
        # ----------------------------------------------------------------------------------
        setter_name = self.scope.get_new_name(f'{class_dtype.name}_{expr.name}_setter'.lower())
        setter_scope = self.scope.new_child_scope(setter_name)
        self.scope = setter_scope
        self.scope.insert_symbol(expr.name)

        setter_arg_wrappers = (self._extract_FunctionDefArgument(FunctionDefArgument(lhs, bound_argument = True), expr),
                               self._extract_FunctionDefArgument(FunctionDefArgument(expr), expr))
        setter_args = (setter_arg_wrappers[0]['c_arg'],
                       setter_arg_wrappers[1]['c_arg'])
        if expr.is_alias:
            setter_args[1].persistent_target = True

        self_obj = setter_arg_wrappers[0]['f_arg'].value
        set_val = setter_arg_wrappers[1]['f_arg'].value

        setter_body = setter_arg_wrappers[0]['body'] + setter_arg_wrappers[1]['body']

        attrib = expr.clone(expr.name, lhs = self_obj)
        # Cast the C variable into a Python variable
        if expr.memory_handling == 'alias':
            setter_body.append(AliasAssign(attrib, set_val))
        else:
            setter_body.append(Assign(attrib, set_val))
        self.exit_scope()

        setter = BindCFunctionDef(setter_name, setter_args, setter_body,
                                original_function = expr, scope = setter_scope)
        return BindCClassProperty(lhs.cls_base.scope.get_python_name(expr.name),
                                  getter, setter, lhs.dtype)

    def _wrap_ClassDef(self, expr):
        """
        Create all objects necessary to expose a class definition to C.

        Create all objects necessary to expose a class definition to C.

        Parameters
        ----------
        expr : ClassDef
            The class to be wrapped.

        Returns
        -------
        BindCClassDef
            The wrapped class.
        """
        name = expr.name
        func_name = self.scope.get_new_name(f'{name}_bind_c_alloc'.lower())
        func_scope = self.scope.new_child_scope(func_name)

        # Allocatable is not returned so it must appear in local scope
        local_var = Variable(expr.class_type, func_scope.get_new_name(f'{name}_obj'),
                             cls_base = expr, memory_handling='alias')
        func_scope.insert_variable(local_var)

        # Create the C-compatible data pointer
        bind_var = Variable(BindCPointer(), func_scope.get_new_name('bound_'+name),
                            is_const=False, memory_handling='alias')
        result = BindCVariable(bind_var, local_var)

        # Define the additional steps necessary to define and fill ptr_var
        alloc = Allocate(local_var, shape=None, status='unallocated')
        c_loc = CLocFunc(local_var, bind_var)
        body = [alloc, c_loc]

        new_method = BindCFunctionDef(func_name, [], body,
                FunctionDefResult(result),
                original_function = None, scope = func_scope)

        methods = [self._wrap(m) for m in expr.methods]
        methods = [m for m in methods if not isinstance(m, EmptyNode)]
        for i in expr.interfaces:
            for f in i.functions:
                self._wrap(f)
        interfaces = [self._wrap(i) for i in expr.interfaces if not i.is_inline]

        if any(isinstance(v.class_type, TupleType) for v in expr.attributes):
            errors.report("Tuples cannot yet be exposed to Python.",
                    severity='warning',
                    symbol=expr)

        properties_getters = [BindCClassProperty(expr.scope.get_python_name(m.original_function.name),
                                                 m, None, expr.class_type, m.original_function.docstring)
                                for m in methods if 'property' in m.original_function.decorators]
        methods = [m for m in methods if m not in properties_getters
                        if 'property' not in m.original_function.decorators]

        # Pseudo-self variable is useful for pre-defined attributes which are not DottedVariables
        pseudo_self = Variable(expr.class_type, 'self', cls_base = expr)
        properties = [self._wrap(v if isinstance(v, DottedVariable) else v.clone(v.name, new_class = DottedVariable, lhs=pseudo_self)) \
                        for v in expr.attributes if not v.is_private and not isinstance(v.class_type, TupleType)]
        return BindCClassDef(expr, new_func = new_method, methods = methods,
                             interfaces = interfaces, attributes = properties_getters + properties,
                             docstring = expr.docstring, class_type = expr.class_type)

    def _extract_FunctionDefResult(self, orig_var, orig_func_scope):
        """
        Get the code and variables necessary to translate a `Variable` to a C-compatible Variable.

        Get the code and variables necessary to translate a `Variable` which is returned
        from a function to a `Variable` which can be called from C. A variable `local_var` is
        created. This variable can be retrieved using its name which matches the name of `orig_var`
        the variable that was originally returned. `local_var` should be used to retrieve the
        result of the function call. It will generally be a clone of the return variable but some
        properties (such as the memory handling) may be modified. A variable describing the
        object which should be returned from the BindCFunctionDef may also be created if necessary.
        Finally AST nodes are also created to describe any code which is needed to convert the
        `local_var` to the returned variable.

        Parameters
        ----------
        orig_var : Variable
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result:
            - c_result: The Variable which should be used in a FunctionDefResult from the wrapped
                    function.
            - body: The code which is needed to convert the local_var to the returned variable
                    saved in c_result.
            - f_result: The Variable which should be used in a FunctionCall to collect the results
                    from the Fortran function.
        """
        class_type = orig_var.class_type

        classes = type(class_type).__mro__
        for cls in classes:
            annotation_method = f'_extract_{cls.__name__}_FunctionDefResult'
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(orig_var, orig_func_scope)

        # Unknown object, we raise an error.
        return errors.report(f"Wrapping function results is not implemented for type {class_type}. "
                + PYCCEL_RESTRICTION_TODO, symbol=orig_var, severity='fatal')

    def _extract_InhomogeneousTupleType_FunctionDefResult(self, orig_var, orig_func_scope):
        name = orig_var.name
        self.scope.insert_symbol(name)
        elements = [orig_func_scope.collect_tuple_element(e) for e in orig_var]
        func_def_results = [self._extract_FunctionDefResult(e, orig_func_scope) for e in elements]
        body = [l for r in func_def_results for l in r['body']]

        # Pack C results into inhomogeneous tuple object
        element_vars = [r['c_result'] for r in func_def_results]
        class_types = [e.class_type for e in element_vars]
        local_var = Variable(InhomogeneousTupleType(*class_types), self.scope.get_expected_name(name),
                        shape = (len(class_types),))
        for i, v in enumerate(element_vars):
            self.scope.insert_symbolic_alias(IndexedElement(local_var, i), v)

        # Pack F results into inhomogeneous tuple object
        element_vars = [r['f_result'] for r in func_def_results]
        class_types = [e.class_type for e in element_vars]
        result_var = Variable(InhomogeneousTupleType(*class_types), self.scope.get_new_name('Out_'+name),
                            shape = (len(class_types),))
        for i, v in enumerate(element_vars):
            self.scope.insert_symbolic_alias(IndexedElement(result_var, i), v)
        return {'body': body, 'c_result': BindCVariable(local_var, result_var), 'f_result': result_var}

    def _extract_FixedSizeType_FunctionDefResult(self, orig_var, orig_func_scope):
        name = orig_var.name
        self.scope.insert_symbol(name)
        local_var = orig_var.clone(self.scope.get_expected_name(name), new_class = Variable)
        return {'body': [], 'c_result': BindCVariable(local_var, orig_var), 'f_result': local_var}

    def _extract_CustomDataType_FunctionDefResult(self, orig_var, orig_func_scope):
        name = orig_var.name
        scope = self.scope
        scope.insert_symbol(name)
        memory_handling = 'alias' if isinstance(orig_var, DottedVariable) else orig_var.memory_handling
        local_var = orig_var.clone(scope.get_expected_name(name), new_class = Variable,
                            memory_handling = memory_handling)
        # Allocatable is not returned so it must appear in local scope
        scope.insert_variable(local_var, name)

        # Create the C-compatible data pointer
        bind_var = Variable(BindCPointer(),
                            scope.get_new_name('bound_'+name),
                            is_const=False, memory_handling='alias')

        if isinstance(orig_var, DottedVariable):
            ptr_var = orig_var
            body = [CLocFunc(ptr_var, bind_var)]
        else:
            # Create an array variable which can be passed to CLocFunc
            ptr_var = Variable(orig_var.class_type, scope.get_new_name(name+'_ptr'), memory_handling='alias')
            scope.insert_variable(ptr_var)
            alloc = Allocate(ptr_var, shape=None, status='unallocated')
            copy = Assign(ptr_var, local_var)
            cloc = CLocFunc(ptr_var, bind_var)
            body = [alloc, copy, cloc]

        return {'body': body, 'c_result': BindCVariable(bind_var, orig_var), 'f_result': local_var}

    def _extract_NumpyNDArrayType_FunctionDefResult(self, orig_var, orig_func_scope):
        name = orig_var.name
        scope = self.scope
        scope.insert_symbol(name)
        memory_handling = 'alias' if isinstance(orig_var, DottedVariable) else orig_var.memory_handling

        # Allocatable is not returned so it must appear in local scope
        local_var = orig_var.clone(scope.get_expected_name(name), new_class = Variable,
                            memory_handling = memory_handling, shape = None)
        scope.insert_variable(local_var, name)

        if orig_var.is_alias or isinstance(orig_var, DottedVariable):
            result =  self._get_bind_c_array(name, orig_var, local_var.shape, local_var)
        else:
            result = self._get_bind_c_array(name, orig_var, local_var.shape)

            result['body'].append(Assign(result['f_array'], local_var))

        result['f_result'] = local_var

        return result

    def _extract_HomogeneousTupleType_FunctionDefResult(self, orig_var, orig_func_scope):
        return self._extract_NumpyNDArrayType_FunctionDefResult(orig_var, orig_func_scope)

    def _extract_HomogeneousSetType_FunctionDefResult(self, orig_var, orig_func_scope):
        name = orig_var.name
        scope = self.scope
        scope.insert_symbol(name)
        memory_handling = 'alias' if isinstance(orig_var, DottedVariable) else orig_var.memory_handling

        # Allocatable is not returned so it must appear in local scope
        local_var = orig_var.clone(scope.get_expected_name(name), new_class = Variable,
                            memory_handling = memory_handling)
        scope.insert_variable(local_var, name)

        result = self._get_bind_c_array(name, orig_var, local_var.shape)

        f_array = result['f_array']
        body = result['body']
        result['f_result'] = local_var

        for_scope = scope.create_new_loop_scope()
        iterator = VariableIterator(local_var)
        elem = Variable(orig_var.class_type.element_type, self.scope.get_new_name())
        idx = Variable(PythonNativeInt(), self.scope.get_new_name())
        self.scope.insert_variable(elem)
        self.scope.insert_variable(idx)

        # Default Fortran arrays retrieved from C_F_Pointer are 1-indexed
        # Lists are 1-indexed but Pyccel adds the shift during printing so they are
        # treated as 0-indexed here
        for_body = [Assign(IndexedElement(f_array, PyccelAdd(idx, LiteralInteger(1))), elem),
                    Assign(idx, PyccelAdd(idx, LiteralInteger(1)))]
        fill_for = For((elem,), iterator, for_body, scope = for_scope)
        body.extend([Assign(idx, LiteralInteger(0)), fill_for])
        return result

    def _extract_HomogeneousListType_FunctionDefResult(self, orig_var, orig_func_scope):
        name = orig_var.name
        scope = self.scope
        scope.insert_symbol(name)
        memory_handling = 'alias' if isinstance(orig_var, DottedVariable) else orig_var.memory_handling

        # Allocatable is not returned so it must appear in local scope
        local_var = orig_var.clone(scope.get_expected_name(name), new_class = Variable,
                            memory_handling = memory_handling)
        scope.insert_variable(local_var, name)

        result = self._get_bind_c_array(name, orig_var, local_var.shape)

        f_array = result['f_array']
        body = result['body']
        result['f_result'] = local_var

        for_scope = scope.create_new_loop_scope()
        iterator = VariableIterator(local_var)
        elem = Variable(orig_var.class_type.element_type, self.scope.get_new_name())
        idx = Variable(PythonNativeInt(), self.scope.get_new_name())
        self.scope.insert_variable(elem)
        iterator.set_loop_counter(idx)

        # Default Fortran arrays retrieved from C_F_Pointer are 1-indexed
        # Lists are 1-indexed but Pyccel adds the shift during printing so they are
        # treated as 0-indexed here
        for_body = [Assign(IndexedElement(f_array, PyccelAdd(idx, LiteralInteger(1))), elem)]
        body.append(For((elem,), iterator, for_body, scope = for_scope))
        return result

    def _extract_DictType_FunctionDefResult(self, orig_var, orig_func_scope):
        name = orig_var.name
        scope = self.scope
        scope.insert_symbol(name)
        memory_handling = 'alias' if isinstance(orig_var, DottedVariable) else orig_var.memory_handling

        # Allocatable is not returned so it must appear in local scope
        local_var = orig_var.clone(scope.get_expected_name(name), new_class = Variable,
                            memory_handling = memory_handling)
        scope.insert_variable(local_var, name)

        # Create the C-compatible data pointer
        key_bind_var = Variable(BindCPointer(),
                            scope.get_new_name(f'bound_{name}_key'),
                            is_const=False, memory_handling='alias')
        val_bind_var = Variable(BindCPointer(),
                            scope.get_new_name(f'bound_{name}_key'),
                            is_const=False, memory_handling='alias')
        shape_var = Variable(NumpyInt32Type(), scope.get_new_name(f'{name}_len'))
        scope.insert_variable(shape_var)

        # Check how to unpack elements
        class_type = orig_var.class_type
        key_var = Variable(class_type.key_type, scope.get_new_name(name+'_key'))
        val_var = Variable(class_type.value_type, scope.get_new_name(name+'_val'))
        key_wrap = self._extract_FunctionDefResult(key_var, orig_func_scope)
        val_wrap = self._extract_FunctionDefResult(val_var, orig_func_scope)

        # Get storage arrays
        key_c_class_type = key_wrap['c_result'].new_var.class_type
        key_ptr_var = Variable(NumpyNDArrayType(key_c_class_type, 1, None), scope.get_new_name(name+'_key_ptr'),
                            memory_handling='alias')
        key_bind_var = Variable(BindCPointer(),
                            scope.get_new_name(f'bound_{name}_key'),
                            is_const=False, memory_handling='alias')
        scope.insert_variable(key_ptr_var)
        scope.insert_variable(key_bind_var)

        val_c_class_type = val_wrap['c_result'].new_var.class_type
        val_ptr_var = Variable(NumpyNDArrayType(val_c_class_type, 1, None), scope.get_new_name(name+'_val_ptr'),
                            memory_handling='alias')
        val_bind_var = Variable(BindCPointer(),
                            scope.get_new_name(f'bound_{name}_val'),
                            is_const=False, memory_handling='alias')
        scope.insert_variable(val_ptr_var)
        scope.insert_variable(val_bind_var)

        # Define the additional steps necessary to define and fill ptr_var
        scope.insert_variable(key_wrap['c_result'])
        scope.insert_variable(val_wrap['c_result'])
        key_size = PyccelMul(BindCSizeOf(key_wrap['c_result']), shape_var)
        val_size = PyccelMul(BindCSizeOf(val_wrap['c_result']), shape_var)
        body = [Assign(shape_var, local_var.shape[0]),
                Assign(key_bind_var, c_malloc(key_size)),
                Assign(val_bind_var, c_malloc(val_size)),
                C_F_Pointer(key_bind_var, key_ptr_var, (shape_var,)),
                C_F_Pointer(val_bind_var, val_ptr_var, (shape_var,))]

        # Start to construct the result
        result = {'f_result' : local_var,
                  'body' : body}

        # Construct the loop which builds the result
        for_scope = scope.create_new_loop_scope()
        iterator = DictItems(local_var)
        idx = Variable(PythonNativeInt(), self.scope.get_new_name())
        self.scope.insert_variable(idx)
        iterator.set_loop_counter(idx)

        for_body = [*key_wrap['body'], Assign(IndexedElement(key_ptr_var, PyccelAdd(idx, LiteralInteger(1))), key_wrap['c_result']),
                    *val_wrap['body'], Assign(IndexedElement(val_ptr_var, PyccelAdd(idx, LiteralInteger(1))), val_wrap['c_result']),
                    Assign(idx, PyccelAdd(idx, LiteralInteger(1)))]
        fill_for = For((key_var, val_var), iterator, for_body, scope = for_scope)
        body.extend([Assign(idx, LiteralInteger(0)), fill_for])

        result_var = Variable(InhomogeneousTupleType(BindCPointer(), BindCPointer(), NumpyInt32Type()),
                            scope.get_new_name(), shape = (3,))
        scope.insert_symbolic_alias(IndexedElement(result_var, LiteralInteger(0)), key_bind_var)
        scope.insert_symbolic_alias(IndexedElement(result_var, LiteralInteger(1)), val_bind_var)
        scope.insert_symbolic_alias(IndexedElement(result_var, LiteralInteger(2)), shape_var)
        result['c_result'] = BindCVariable(result_var, orig_var)
        return result

    def _extract_StringType_FunctionDefResult(self, orig_var, orig_func_scope):
        name = orig_var.name
        scope = self.scope
        scope.insert_symbol(name)
        memory_handling = 'alias' if isinstance(orig_var, DottedVariable) else orig_var.memory_handling

        # Allocatable is not returned so it must appear in local scope
        local_var = orig_var.clone(scope.get_expected_name(name), new_class = Variable,
                            memory_handling = memory_handling)
        scope.insert_variable(local_var, name)

        # Create the C-compatible data pointer
        bind_var = Variable(BindCPointer(),
                            scope.get_new_name('bound_'+name),
                            is_const=False, memory_handling='alias')

        shape_var = Variable(NumpyInt32Type(), scope.get_new_name(f'{name}_len'))
        scope.insert_variable(shape_var)

        # Create an array variable which can be passed to CLocFunc
        ptr_var = Variable(NumpyNDArrayType(CharType(), 1, None), scope.get_new_name(name+'_ptr'),
                            memory_handling='alias')
        elem_var = Variable(CharType(), scope.get_new_name(name+'_elem'))
        scope.insert_variable(ptr_var)
        scope.insert_variable(elem_var)

        for_scope = scope.create_new_loop_scope()
        iterator = PythonRange(LiteralInteger(1), shape_var)
        idx = Variable(PythonNativeInt(), self.scope.get_new_name())
        iterator.set_loop_counter(idx)
        self.scope.insert_variable(idx)

        # Default Fortran arrays retrieved from C_F_Pointer are 1-indexed
        # Lists are 1-indexed but Pyccel adds the shift during printing so they are
        # treated as 0-indexed here
        for_body = [Assign(IndexedElement(ptr_var, idx), IndexedElement(local_var, idx))]

        # Define the additional steps necessary to define and fill ptr_var
        # Default Fortran arrays retrieved from C_F_Pointer are 1-indexed
        body = [Assign(shape_var, PyccelAdd(local_var.shape[0], LiteralInteger(1))),
                Assign(bind_var, c_malloc(PyccelMul(BindCSizeOf(elem_var), shape_var))),
                C_F_Pointer(bind_var, ptr_var, [shape_var]),
                For((idx,), iterator, for_body, scope = for_scope),
                Assign(IndexedElement(ptr_var, shape_var), C_NULL_CHAR())]

        return {'c_result': BindCVariable(bind_var, orig_var), 'body': body, 'f_array': ptr_var,
                'f_result': local_var}

    def _get_bind_c_array(self, name, orig_var, shape, pointer_target = False):
        """
        Get all the objects necessary to return an array from the BindCFunctionDef.

        In the case of an array, C cannot represent the array natively. Rather it is
        stored in a pointer. This function therefore creates a variable to represent
        that pointer. Additionally information about the shape and strides of the array
        are necessary.  The assignment expressions which define the shapes and strides
        are then stored in `body` along with the allocation of the pointer. The
        Fortran-accessible array is returned so that it can be filled differently
        depending on what type is described by the array (e.g. if the array describes
        an array a simple copy is required, but if the array describes a set then the
        elements need to be added one by one.

        Parameters
        ----------
        name : str
            The stem of the names of the objects that should be created.

        orig_var : Variable
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped. This is used to obtain the dtype, rank
            and order of the array that should be created.

        shape : tuple[TypedAstNode]
            A tuple describing the shape that the array should be allocated to.

        pointer_target : bool, default=False
            Indicates if the data in orig_var is a target of the pointer that will be
            created.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result:
            - c_result: The Variable which should be used in a FunctionDefResult from the wrapped
                    function.
            - body: The code which is needed to convert the local_var to the returned variable
                    saved in c_result.
            - f_array: The Fortran-accessible array that will be returned. This is where the data
                    should be copied to.
        """
        dtype = orig_var.dtype
        rank = orig_var.rank
        order = orig_var.order
        scope = self.scope
        # Create the C-compatible data pointer
        bind_var = Variable(BindCPointer(),
                            scope.get_new_name('bound_'+name),
                            is_const=False, memory_handling='alias')

        shape_vars = [Variable(NumpyInt32Type(), scope.get_new_name(f'{name}_shape_{i+1}'))
                         for i in range(rank)]

        body = [Assign(s_v, s) for s_v, s in zip(shape_vars, shape)]

        if pointer_target:
            body.append(CLocFunc(orig_var, bind_var))
            f_array = orig_var
        else:
            # Create an array variable which can be passed to CLocFunc
            ptr_var = Variable(NumpyNDArrayType(dtype, rank, order), scope.get_new_name(name+'_ptr'),
                                memory_handling='alias')
            elem_var = Variable(dtype, scope.get_new_name(name+'_elem'))
            scope.insert_variable(ptr_var)
            scope.insert_variable(elem_var)

            # Define the additional steps necessary to define and fill ptr_var
            size = reduce(PyccelMul, [BindCSizeOf(elem_var), *shape_vars])
            body += [Assign(bind_var, c_malloc(size)),
                        C_F_Pointer(bind_var, ptr_var, shape_vars if order == 'F' else shape_vars[::-1])]

            f_array = ptr_var

        result_var = Variable(BindCArrayType(rank, has_strides = False),
                        scope.get_new_name(), shape = (rank+1,))
        scope.insert_symbolic_alias(IndexedElement(result_var, LiteralInteger(0)), bind_var)
        for i,s in enumerate(shape_vars):
            scope.insert_symbolic_alias(IndexedElement(result_var, LiteralInteger(i+1)), s)

        return {'c_result': BindCVariable(result_var, orig_var), 'body': body, 'f_array': f_array}
