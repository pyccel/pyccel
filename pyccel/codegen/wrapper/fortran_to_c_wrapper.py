# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
from ast.bind_c import BindCPointer, BindCFunctionDefArgument
from ast.core import Module, Assign, FunctionCall, FunctionDefArgument
from ast.variable import Variable
from .wrapper import Wrapper

class FortranToCWrapper(Wrapper):

    def _get_function_def_body(args, array_creation_stmts, results, handled = ()):
        optionals = [a for a in args if a.is_optional and a not in handled]
        if len(optionals) == 0:
            body = [array_creation_stmts[a] for a in args if v.is_ndarray]
            return body + [Assign(results, FunctionCall(expr, args))]
        else:
            o = optionals[0]
            true_section = IfSection(LiteralTrue(),
                                    self._get_function_def_body(args, array_creation_stmts, results, handled))
            passed_args = [a for a in args if a is not o]
            false_section = IfSection(PyccelIs(o, Nil()),
                                    self._get_function_def_body(passed_args, array_creation_stmts, results, handled + (o,)))
            return If(false_section, true_section)

    def _wrap_Module(self, expr):
        # Define scope
        scope = expr.scope
        mod_scope = scope.local_used_symbols.copy()
        self.set_scope(mod_scope)

        # Wrap contents
        funcs = [self._wrap(f) for f in expr.funcs if not f.is_private]
        interfaces = [self._wrap(f) for f in expr.interfaces]
        classes = [self._wrap(f) for f in expr.classes]
        variable_getters = [self._wrap(f) for v in expr.variables if not v.is_private]
        init_func = self._wrap(expr.init_func) if expr.init_func else None
        free_func = self._wrap(expr.free_func) if expr.free_func else None

        name = mod_scope.get_new_name(f'bind_c_{expr.name}')

        return Module(name, (), funcs + variable_getters,
                init_func = init_func, free_func = free_func, 
                interfaces = interfaces, classes = classes,
                scope = mod_scope)

    def _wrap_FunctionDef(self, expr):
        name = self.scope.get_new_name(f'bind_c_{expr.name}')
        func_scope = self.scope.new_child_scope(name)
        self.set_scope(func_scope)

        original_arguments = [a.var.clone(func_scope.get_new_name(a.name)) for a in expr.arguments]
        array_c_arguments = {v.name: Variable(BindCPointer(), func_scope.get_new_name(v.name),
                                is_pointer=True) for v in original_arguments if v.is_ndarray}

        func_arguments = {oa : BindCFunctionDefArgument(array_c_arguments[oa.name]) if oa.is_ndarray else FunctionDefArgument(oa)
                            for oa in original_arguments}

        call_arguments = [IndexedElement(oa, [Slice(LiteralInteger(0), size, stride)
                                                for size, stride in zip(oa.sizes, oa.strides)])
                          if oa.is_ndarray else oa for oa in original_arguments]

        results = [r.clone(self.scope.get_new_name(r.name)) for r in expr.results]
        func_results = [BindCFunctionDefResult(r) for r in expr.results]

        array_creation = {oa : C_F_Pointer(array_c_arguments[oa], oa, arguments[oa].sizes)
                            for oa in original_arguments if oa.is_ndarray}

        body = self._get_function_def_body(call_args, array_creation, results)

        return BindCFunctionDef(name, arguments.values(), func_results, body, scope=func_scope, original_function = expr)

    def _wrap_FunctionDefArgument(self, expr):
        var = expr.var.clone(self.scope.get_new_name(expr.name))
        if var.rank != 0 and var.dtype in NativeNumeric:
            return BindCFunctionDefArgument(var, value = expr.value,
                    kwonly = expr.kwonly, annotation = expr.annotation)
        else:
            return FunctionDefArgument(var, value = expr.value,
                    kwonly = expr.kwonly, annotation = expr.annotation)
