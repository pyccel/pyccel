# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : FortranToCWrapper
which creates an interface exposing Fortran code to C.
"""
import warnings
from pyccel.ast.bind_c import BindCFunctionDefArgument, BindCFunctionDefResult
from pyccel.ast.bind_c import BindCPointer, BindCFunctionDef, C_F_Pointer
from pyccel.ast.bind_c import CLocFunc, BindCModule
from pyccel.ast.core import Assign, FunctionCall, FunctionCallArgument
from pyccel.ast.core import Allocate, EmptyNode, FunctionAddress
from pyccel.ast.core import If, IfSection, Import, Interface
from pyccel.ast.core import AsName, Module, AliasAssign
from pyccel.ast.internals import Slice
from pyccel.ast.literals import LiteralInteger, Nil, LiteralTrue
from pyccel.ast.operators import PyccelIsNot, PyccelMul
from pyccel.ast.variable import Variable, IndexedElement
from pyccel.parser.scope import Scope
from .wrapper import Wrapper

class FortranToCWrapper(Wrapper):
    """
    Class for creating a wrapper exposing Fortran code to C.

    A class which provides all necessary functions for wrapping different AST
    objects such that the resulting AST is C-compatible. This new AST is
    printed as an intermediary layer.
    """
    def __init__(self):
        self._additional_exprs = []
        self._wrapper_names_dict = {}
        super().__init__()

    def _get_function_def_body(self, func, func_def_args, func_arg_to_call_arg, results, handled = ()):
        """
        Get the body of the bind c function definition.

        Get the body of the bind c function definition by inserting if blocks
        to check the presence of optional variables. Once we have ascertained
        the prescence of the variables the original function is called. This
        code slices array variables to ensure the correct step.

        Parameters
        ----------
        func : FunctionDef
            The function which should be called.

        func_def_args : list of FunctionDefArguments
            The arguments received by the function.

        func_arg_to_call_arg : dict
            A dictionary mapping the arguments received by the function to the arguments
            to be passed to the function call.

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
        optional = next((a for a in func_def_args if a.original_function_argument_variable.is_optional and a not in handled), None)
        if optional:
            args = func_def_args.copy()
            optional_var = optional.var
            handled += (optional, )
            true_section = IfSection(PyccelIsNot(optional_var, Nil()),
                                    self._get_function_def_body(func, args, func_arg_to_call_arg, results, handled))
            args.remove(optional)
            false_section = IfSection(LiteralTrue(),
                                    self._get_function_def_body(func, args, func_arg_to_call_arg, results, handled))
            return [If(true_section, false_section)]
        else:
            args = [FunctionCallArgument(func_arg_to_call_arg[fa],
                                         keyword = fa.original_function_argument_variable.name)
                    for fa in func_def_args]
            size = [fa.sizes[::-1] if fa.original_function_argument_variable.order == 'C' else
                    fa.sizes for fa in func_def_args]
            stride = [fa.strides[::-1] if fa.original_function_argument_variable.order == 'C' else
                      fa.strides for fa in func_def_args]
            orig_size = [[PyccelMul(l,s) for l,s in zip(sz, st)] for sz,st in zip(size,stride)]
            body = [C_F_Pointer(fa.var, func_arg_to_call_arg[fa].base, s)
                    for fa,s in zip(func_def_args, orig_size)
                    if isinstance(func_arg_to_call_arg[fa], IndexedElement)]
            body += [C_F_Pointer(fa.var, func_arg_to_call_arg[fa])
                     for fa in func_def_args
                     if not isinstance(func_arg_to_call_arg[fa], IndexedElement) \
                        and fa.original_function_argument_variable.is_optional]

            # If the function is inlined and takes an array argument create a pointer to ensure that the bounds
            # are respected
            if func.is_inline and any(isinstance(a.value, IndexedElement) for a in args):
                array_args = {a: self.scope.get_temporary_variable(a.value.base, a.keyword, memory_handling = 'alias') for a in args if isinstance(a.value, IndexedElement)}
                body += [AliasAssign(v, k.value) for k,v in array_args.items()]
                args = [FunctionCallArgument(array_args[a], keyword=a.keyword) if a in array_args else a for a in args]

            func_call = Assign(results[0], FunctionCall(func, args)) if len(results) == 1 else \
                        Assign(results, FunctionCall(func, args))
            return body + [func_call]

    def _get_call_argument(self, original_arg, bind_c_arg):
        """
        Get the argument which should be passed to the function call.

        The FunctionDefArgument passed to the function may contain additional
        information which should not be passed to the function being wrapped
        (e.g. an array with strides should not pass the strides explicitly to
        the function call, and nor should it pass the entire contiguous array).
        This function extracts the necessary information and returns the object
        which can be passed to the function call.

        Parameters
        ----------
        original_arg : Variable
            The argument to the function being wrapped.

        bind_c_arg : BindCFunctionDefArgument
            The argument to the wrapped bind_c_X function.

        Returns
        -------
        PyccelAstNode
            An object which can be passed to a function call of the function
            being wrapped.
        """
        if original_arg.is_ndarray:
            new_var = original_arg.clone(self.scope.get_new_name(original_arg.name), is_argument = False, is_optional = False,
                                memory_handling = 'alias', allows_negative_indexes=False)
            self.scope.insert_variable(new_var)
            start = LiteralInteger(1) # C_F_Pointer leads to default Fortran lbound
            stop = None
            indexes = [Slice(start, stop, step) for step in bind_c_arg.strides]
            return IndexedElement(new_var, *indexes)
        elif original_arg.is_optional:
            new_var = original_arg.clone(self.scope.get_new_name(original_arg.name), is_optional = False,
                    memory_handling = 'alias')
            self.scope.insert_variable(new_var)
            return new_var
        else:
            return bind_c_arg.var

    def _wrap_Module(self, expr):
        # Define scope
        scope = expr.scope
        mod_scope = Scope(used_symbols = scope.local_used_symbols.copy(), original_symbols = scope.python_names.copy())
        self.set_scope(mod_scope)

        # Wrap contents
        funcs_to_wrap = [f for f in expr.funcs if not f.is_private]
        funcs = [self._wrap(f) for f in funcs_to_wrap]
        if expr.init_func:
            init_func = funcs[next(i for i,f in enumerate(funcs_to_wrap) if f == expr.init_func)]
        else:
            init_func = None
        if expr.free_func:
            free_func = funcs[next(i for i,f in enumerate(funcs_to_wrap) if f == expr.free_func)]
        else:
            free_func = None
        funcs = [f for f in funcs if not isinstance(f, EmptyNode)]
        interfaces = [self._wrap(f) for f in expr.interfaces]
        classes = [self._wrap(f) for f in expr.classes]
        variable_getters = [self._wrap(v) for v in expr.variables if not v.is_private]
        variable_getters = [v for v in variable_getters if not isinstance(v, EmptyNode)]
        imports = [Import(expr.name, target = expr, mod=expr)]

        name = mod_scope.get_new_name(f'bind_c_{expr.name.target}')
        self._wrapper_names_dict[expr.name.target] = name

        self.exit_scope()

        return BindCModule(name, (), funcs, variable_wrappers = variable_getters,
                init_func = init_func, free_func = free_func,
                interfaces = interfaces, classes = classes,
                imports = imports, original_module = expr,
                scope = mod_scope)

    def _wrap_FunctionDef(self, expr):
        name = self.scope.get_new_name(f'bind_c_{expr.name.lower()}')
        self._wrapper_names_dict[expr.name] = name
        func_scope = self.scope.new_child_scope(name)
        self.set_scope(func_scope)

        self._additional_exprs = []

        if any(isinstance(a.var, FunctionAddress) for a in expr.arguments):
            warnings.warn("Functions with functions as arguments cannot be wrapped by pyccel")
            return EmptyNode()

        func_arguments = [self._wrap(a) for a in expr.arguments]
        original_arguments = [a.var for a in expr.arguments]
        call_arguments = [self._get_call_argument(oa, fa) for oa, fa in zip(original_arguments, func_arguments)]
        func_to_call = {fa : ca for ca, fa in zip(call_arguments, func_arguments)}

        func_results = [self._wrap_FunctionDefResult(r) for r in expr.results]

        func_call_results = [r.var.clone(self.scope.get_expected_name(r.var.name)) for r in expr.results]

        body = self._get_function_def_body(expr, func_arguments, func_to_call, func_call_results)

        body.extend(self._additional_exprs)
        self._additional_exprs.clear()

        self.exit_scope()

        func = BindCFunctionDef(name, func_arguments, func_results, body, scope=func_scope, original_function = expr,
                doc_string = expr.doc_string)

        self.scope.functions[name] = func

        return func

    def _wrap_Interface(self, expr):
        functions = [self.scope.functions[self._wrapper_names_dict[f.name]] for f in expr.functions]
        functions = [f for f in functions if not isinstance(f, EmptyNode)]
        return Interface(expr.name, functions, expr.is_argument)

    def _wrap_FunctionDefArgument(self, expr):
        name = expr.name
        self.scope.insert_symbol(name)
        var = expr.var
        collisionless_name = self.scope.get_expected_name(var.name)
        if var.is_ndarray or var.is_optional:
            new_var = Variable(BindCPointer(), self.scope.get_new_name(collisionless_name),
                                is_argument = True, is_optional = False, memory_handling='alias')
        else:
            new_var = var.clone(collisionless_name)
        self.scope.insert_variable(new_var)

        return BindCFunctionDefArgument(new_var, value = expr.value, original_arg_var = expr.var,
                kwonly = expr.is_kwonly, annotation = expr.annotation, scope=self.scope)

    def _wrap_FunctionDefResult(self, expr):
        var = expr.var
        name = var.name
        scope = self.scope
        # Make name availiable for later
        scope.insert_symbol(name)
        local_var = var.clone(scope.get_expected_name(name))

        if local_var.rank:
            # Allocatable is not returned so it must appear in local scope
            scope.insert_variable(local_var, name)

            # Create the data pointer
            bind_var = Variable(dtype=BindCPointer(),
                                name=scope.get_new_name('bound_'+name),
                                is_const=True, memory_handling='alias')
            scope.insert_variable(bind_var)

            result = BindCFunctionDefResult(bind_var, var, scope)

            # Save the shapes of the array
            self._additional_exprs.extend([Assign(result.sizes[i], var.shape[i]) for i in range(var.rank)])

            # Create a C-compatible array variable
            ptr_var = var.clone(scope.get_new_name(name+'_ptr'),
                                memory_handling='alias')
            scope.insert_variable(ptr_var)

            # Define the additional steps necessary to define and fill ptr_var
            alloc = Allocate(ptr_var, shape=var.shape,
                             order=var.order, status='unallocated')
            copy = Assign(ptr_var, var)
            c_loc = CLocFunc(ptr_var, bind_var)
            self._additional_exprs.extend([alloc, copy, c_loc])

            return result
        else:
            return BindCFunctionDefResult(local_var, var, scope)

    def _wrap_Variable(self, expr):
        if expr.rank == 0:
            return EmptyNode()
        else:
            scope = self.scope
            func_name = scope.get_new_name('bind_c_'+expr.name.lower())
            func_scope = scope.new_child_scope(func_name)
            mod = expr.get_user_nodes(Module)[0]
            import_mod = Import(mod.name, AsName(expr,expr.name), mod=mod)
            func_scope.imports['variables'][expr.name] = expr

            # Create the data pointer
            bind_var = Variable(dtype=BindCPointer(),
                                name=scope.get_new_name('bound_'+expr.name),
                                is_const=True, memory_handling='alias')
            func_scope.insert_variable(bind_var)

            result = BindCFunctionDefResult(bind_var, expr, func_scope)
            assigns = [Assign(result.sizes[i], expr.shape[i]) for i in range(expr.rank)]
            c_loc = CLocFunc(expr, bind_var)
            body = [*assigns, c_loc]
            func = BindCFunctionDef(name = func_name,
                          body      = body,
                          arguments = [],
                          results   = [result],
                          imports   = [import_mod],
                          scope = func_scope,
                          original_function = expr)
            return func
