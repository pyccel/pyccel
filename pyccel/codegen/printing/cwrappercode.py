# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
import sys

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.bind_c     import BindCPointer
from pyccel.ast.bind_c     import BindCModule, BindCFunctionDef
from pyccel.ast.c_concepts import CStackArray, CStrStr
from pyccel.ast.core       import FunctionAddress, SeparatorComment
from pyccel.ast.core       import Import, Module, Declare
from pyccel.ast.cwrapper   import PyBuildValueNode, PyCapsule_New, PyCapsule_Import, PyModule_Create
from pyccel.ast.cwrapper   import Py_None, WrapperCustomDataType, Py_ssize_t
from pyccel.ast.cwrapper   import PyccelPyObject, PyccelPyTypeObject, PyTuple_Pack
from pyccel.ast.literals   import LiteralString, Nil, LiteralInteger
from pyccel.ast.numpy_wrapper import PyccelPyArrayObject
from pyccel.ast.c_concepts import ObjectAddress

from pyccel.errors.errors  import Errors

__all__ = ("CWrapperCodePrinter", "cwrappercode")

errors = Errors()

module_imports = [Import('numpy_version', Module('numpy_version',(),())),
            Import('numpy/arrayobject', Module('numpy/arrayobject',(),())),
            Import('cwrapper', Module('cwrapper',(),()))]


class CWrapperCodePrinter(CCodePrinter):
    """
    A printer for printing the C-Python interface.

    A printer to convert Pyccel's AST describing a translated module,
    to strings of C code which provide an interface between the module
    and Python code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    **settings : dict
            Any additional arguments which are necessary for CCodePrinter.
    """
    dtype_registry = {**CCodePrinter.dtype_registry,
                      PyccelPyObject() : 'PyObject',
                      PyccelPyArrayObject() : 'PyArrayObject',
                      PyccelPyTypeObject() : 'PyTypeObject',
                      BindCPointer()  : 'void'}

    def __init__(self, filename, **settings):
        CCodePrinter.__init__(self, filename, **settings)
        self._to_free_PyObject_list = []
        self._function_wrapper_names = dict()
        self._module_name = None

    # --------------------------------------------------------------------
    #                       Helper functions
    # --------------------------------------------------------------------

    def is_c_pointer(self, a):
        """
        Indicate whether the object is a pointer in C code.

        This function extends `CCodePrinter.is_c_pointer` to specify more objects
        which are always accessed via a C pointer.

        Parameters
        ----------
        a : TypedAstNode
            The object whose storage we are enquiring about.

        Returns
        -------
        bool
            True if a C pointer, False otherwise.

        See Also
        --------
        CCodePrinter.is_c_pointer : The extended function.
        """
        if isinstance(a.class_type, (WrapperCustomDataType, BindCPointer,
                                     CStackArray, PyTuple_Pack)):
            return True
        elif isinstance(a, (PyBuildValueNode, PyCapsule_New, PyCapsule_Import, PyModule_Create)):
            return True
        else:
            return CCodePrinter.is_c_pointer(self,a)

    def get_python_name(self, scope, obj):
        """
        Get the name of object as defined in the original python code.

        Get the name of the object as it was originally defined in the
        Python code being translated. This name may have changed before
        the printing stage in the case of name clashes or language interfaces.

        Parameters
        ----------
        scope : pyccel.parser.scope.Scope
            The scope where the object was defined.

        obj : pyccel.ast.basic.PyccelAstNode
            The object whose name we wish to identify.

        Returns
        -------
        str
            The original name of the object.
        """
        if isinstance(obj, BindCFunctionDef):
            return scope.get_python_name(obj.original_function.name)
        elif isinstance(obj, BindCModule):
            return obj.original_module.name
        else:
            return scope.get_python_name(obj.name)

    def function_signature(self, expr, print_arg_names = True):
        args = list(expr.arguments)
        if any([isinstance(a.var, FunctionAddress) for a in args]):
            # Functions with function addresses as arguments cannot be
            # exposed to python so there is no need to print their signature
            return ''
        else:
            return CCodePrinter.function_signature(self, expr, print_arg_names)

    def get_declare_type(self, expr):
        """
        Get the string which describes the type in a declaration.

        This function extends `CCodePrinter.get_declare_type` to specify types
        which are only relevant in the C-Python interface.

        Parameters
        ----------
        expr : Variable
            The variable whose type should be described.

        Returns
        -------
        str
            The code describing the type.

        Raises
        ------
        PyccelCodegenError
            If the type is not supported in the C code or the rank is too large.

        See Also
        --------
        CCodePrinter.get_declare_type : The extended function.
        """
        if expr.dtype is BindCPointer():
            if expr.is_const:
                return 'const void*'
            else:
                return 'void*'
        if expr.dtype is Py_ssize_t():
            dtype =  'Py_ssize_t*' if self.is_c_pointer(expr) else 'Py_ssize_t'
            if expr.is_const:
                return f'const {dtype}'
            else:
                return dtype
        return CCodePrinter.get_declare_type(self, expr)

    def _handle_is_operator(self, Op, expr):
        """
        Get the code to print an `is` or `is not` expression.

        Get the code to print an `is` or `is not` expression. These two operators
        function similarly so this helper function reduces code duplication.
        This function overrides CCodePrinter._handle_is_operator to add the
        handling of `Py_None`.

        Parameters
        ----------
        Op : str
            The C operator representing "is" or "is not".

        expr : PyccelIs/PyccelIsNot
            The expression being printed.

        Returns
        -------
        str
            The code describing the expression.

        Raises
        ------
        PyccelError : Raised if the comparison is poorly defined.
        """
        if expr.rhs is Py_None:
            lhs = ObjectAddress(expr.lhs)
            rhs = ObjectAddress(expr.rhs)
            lhs = self._print(lhs)
            rhs = self._print(rhs)
            return f'{lhs} {Op} {rhs}'
        else:
            return super()._handle_is_operator(Op, expr)

    #--------------------------------------------------------------------
    #                 _print_ClassName functions
    #--------------------------------------------------------------------

    def _print_DottedName(self, expr):
        names = expr.name
        return '.'.join(self._print(n) for n in names)

    def _print_PyInterface(self, expr):
        funcs_to_print = (*expr.functions, expr.type_check_func, expr.interface_func)
        return '\n'.join(self._print(f) for f in funcs_to_print)

    def _print_PyArg_ParseTupleNode(self, expr):
        name    = 'PyArg_ParseTupleAndKeywords'
        pyarg   = expr.pyarg
        pykwarg = expr.pykwarg
        flags   = expr.flags
        # All args are modified so even pointers are passed by address
        args    = ', '.join(f'&{a.name}' for a in expr.args)

        if expr.args:
            code = f'{name}({pyarg}, {pykwarg}, "{flags}", {expr.arg_names.name}, {args})'
        else :
            code =f'{name}({pyarg}, {pykwarg}, "", {expr.arg_names.name})'

        return code

    def _print_PyBuildValueNode(self, expr):
        name  = 'Py_BuildValue'
        flags = expr.flags
        args  = ', '.join(self._print(a) for a in expr.args)
        #to change for args rank 1 +
        if expr.args:
            code = f'(*{name}("{flags}", {args}))'
        else :
            code = f'(*{name}(""))'
        return code

    def _print_PyArgKeywords(self, expr):
        arg_names = ',\n'.join([f'(char*)"{a}"' for a in expr.arg_names] + [self._print(Nil())])
        return (f'static char *{expr.name}[] = {{\n'
                        f'{arg_names}\n'
                        '};\n')

    def _print_PyModule_AddObject(self, expr):
        name = self._print(expr.name)
        var  = self._print(expr.variable)
        if expr.variable.dtype is not PyccelPyObject():
            var = f'(PyObject*) {var}'
        return f'PyModule_AddObject({expr.mod_name}, {name}, {var})'

    def _print_PyCapsule_New(self, expr):
        name = expr.capsule_name
        var  = self._print(ObjectAddress(expr.API_var))
        return f'PyCapsule_New((void *){var}, "{name}", NULL)'

    def _print_PyCapsule_Import(self, expr):
        name = expr.capsule_name
        return f'(void**)PyCapsule_Import("{name}", 0)'

    def _print_PyModule_Create(self, expr):
        return f'PyModule_Create(&{expr.module_def_name})'

    def _print_ModuleHeader(self, expr):
        mod = expr.module
        self._current_module = expr.module
        name = mod.name

        # Print imports last to be sure that all additional_imports have been collected
        imports  = [*module_imports, *mod.imports]
        for i in imports:
            self.add_import(i)
        imports  = ''.join(self._print(i) for i in imports)

        function_signatures = ''.join(self.function_signature(f, print_arg_names = False) + ';\n' for f in mod.external_funcs)

        API_var = mod.variables[0]

        macro_defs = ''
        type_declarations = ''
        classes = []
        for i,c in enumerate(mod.classes):
            struct_name = c.struct_name
            type_name = c.type_name
            attributes = ''.join(self._print(Declare(a)) for a in c.attributes)
            classes.append(f"struct {struct_name} {{\n"
                    "    PyObject_HEAD\n"
                    + attributes +
                    "};\n")
            type_declarations += f'static PyTypeObject {c.type_name};\n'
            sig_methods = c.methods + (c.new_func,) + tuple(f for i in c.interfaces for f in i.functions) + \
                          tuple(i.interface_func for i in c.interfaces) + \
                          tuple(getset for p in c.properties for getset in (p.getter, p.setter) if getset) + \
                          c.magic_methods
            function_signatures += '\n'+''.join(self.function_signature(f)+';\n' for f in sig_methods)
            macro_defs += f'#define {type_name} (*(PyTypeObject*){API_var.name}[{i}])\n'

        class_code = '\n'.join(classes)

        static_import_decs = self._print(Declare(API_var, static=True))
        import_func = self._print(mod.import_func)

        self._current_module = None
        header_id = f'{name.upper()}_WRAPPER'
        header_guard = f'{header_id}_H'
        start = f"#ifndef {header_guard}\n#define {header_guard}\n"
        end = f"#endif\n#endif // {header_guard}\n"
        parts = (start, imports, class_code, f"#ifdef {header_id}\n",
                 type_declarations, function_signatures, "#else\n",
                 static_import_decs, macro_defs, import_func, end)
        return '\n'.join((p for p in parts if p))

    def _print_PyModule(self, expr):
        scope = expr.scope
        self.set_scope(scope)
        self._current_module = expr

        # Insert declared objects into scope
        variables = expr.original_module.variables if isinstance(expr, BindCModule) else expr.variables
        for f in expr.funcs:
            scope.insert_symbol(f.name.lower())
        for v in variables:
            if not v.is_private:
                scope.insert_symbol(v.name.lower())

        funcs = []

        self._module_name  = expr.name
        sep = self._print(SeparatorComment(40))

        interface_funcs = [f.name for i in expr.interfaces for f in i.functions]
        funcs += [*expr.interfaces, *(f for f in expr.funcs if f.name not in interface_funcs)]

        self._in_header = True
        decs = ''.join(self._print(d) for d in expr.declarations)
        self._in_header = False

        function_defs = '\n'.join(self._print(f) for f in funcs)

        class_defs = f"\n{sep}\n".join(self._print(c) for c in expr.classes)

        method_def_func = ''.join(('{{\n'
                                     '"{name}",\n'
                                     '(PyCFunction){wrapper_name},\n'
                                     'METH_VARARGS | METH_KEYWORDS,\n'
                                     '{docstring}\n'
                                     '}},\n').format(
                                            name = self.get_python_name(expr.scope, f.original_function),
                                            wrapper_name = f.name,
                                            docstring = self._print(CStrStr(LiteralString('\n'.join(f.docstring.comments)))) \
                                                        if f.docstring else '""')
                                     for f in funcs if not getattr(f, 'is_header', False))

        method_def_name = self.scope.get_new_name('{}_methods'.format(expr.name))
        method_def = (f'static PyMethodDef {method_def_name}[] = {{\n'
                        f'{method_def_func}'
                        '{ NULL, NULL, 0, NULL}\n'
                        '};\n')

        module_def_name = self.scope.get_new_name('{}_module'.format(expr.name))
        module_def = (f'static struct PyModuleDef {module_def_name} = {{\n'
                'PyModuleDef_HEAD_INIT,\n'
                '/* name of module */\n'
                f'"{self._module_name}",\n'
                '/* module documentation, may be NULL */\n'
                'NULL,\n' #TODO: Add documentation
                '/* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */\n'
                '0,\n'
                f'{method_def_name},\n'
                '};\n')

        init_func = self._print(expr.init_func)

        pymod_name = f'{expr.name}_wrapper'
        imports = [Import(pymod_name, Module(pymod_name,(),())), *self._additional_imports.values()]
        imports  = ''.join(self._print(i) for i in imports)

        self.exit_scope()
        self._current_module = None

        return '\n'.join(['#define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API',
                f'#define {pymod_name.upper()}\n',
                imports, decs, sep, class_defs, sep,
                function_defs, sep, method_def, sep,
                module_def, sep, init_func])

    def _print_PyClassDef(self, expr):
        struct_name = expr.struct_name
        type_name = expr.type_name
        name = self.scope.get_python_name(expr.name)
        docstring = self._print(CStrStr(LiteralString('\n'.join(expr.docstring.comments)))) \
                    if expr.docstring else '""'

        original_scope = expr.original_class.scope
        getters = tuple(p.getter for p in expr.properties)
        setters = tuple(p.setter for p in expr.properties if p.setter)
        print_methods = expr.methods + (expr.new_func,) + expr.interfaces + \
                         expr.magic_methods + getters + setters
        functions = '\n'.join(self._print(f) for f in print_methods)
        init_string = ''
        del_string = ''
        funcs = {}
        for f in expr.methods:
            py_name = self.get_python_name(original_scope, f.original_function)
            if py_name == '__init__':
                init_string = f"    .tp_init = (initproc) {f.name},\n"
            elif py_name == '__del__':
                del_string = f"    .tp_dealloc = (destructor) {f.name},\n"
            else:
                docstring = self._print(CStrStr(LiteralString('\n'.join(f.docstring.comments)))) \
                                                        if f.docstring else '""'
                funcs[py_name] = (f.name, docstring)

        for f in expr.interfaces:
            py_name = self.get_python_name(original_scope, f.original_function)
            docstring = self._print(CStrStr(LiteralString('\n'.join(f.docstring.comments)))) \
                                                    if f.docstring else '""'
            funcs[py_name] = (f.name, docstring)

        property_definitions = ''.join(''.join(('{\n',
                                        f'"{p.python_name}",\n',
                                        f'(getter) {p.getter.name},\n',
                                        f'(setter) {p.setter.name},\n' if p.setter else '(setter) NULL,\n',
                                        f'{self._print(p.docstring)},\n',
                                        'NULL\n',
                                        '},\n')) for p in expr.properties)
        property_definitions += '{ NULL }\n'

        method_def_funcs = ''.join(('{\n'
                                     f'"{name}",\n'
                                     f'(PyCFunction){wrapper_name},\n'
                                      'METH_VARARGS | METH_KEYWORDS,\n'
                                     f'{doc_string}\n'
                                     '},\n')
                                     for name, (wrapper_name, doc_string) in funcs.items())

        magic_methods = {self.get_python_name(original_scope, f.original_function): f for f in expr.magic_methods}

        number_magic_method_name = self.scope.get_new_name(f'{expr.name}_number_methods')

        number_magic_methods_def = f"static PyNumberMethods {number_magic_method_name} = {{\n"
        if '__add__' in magic_methods:
            number_magic_methods_def += f"     .nb_add = (binaryfunc){magic_methods['__add__'].name},\n"
        if '__sub__' in magic_methods:
            number_magic_methods_def += f"     .nb_subtract = (binaryfunc){magic_methods['__sub__'].name},\n"
        if '__mul__' in magic_methods:
            number_magic_methods_def += f"     .nb_multiply = (binaryfunc){magic_methods['__mul__'].name},\n"
        if '__truediv__' in magic_methods:
            number_magic_methods_def += f"     .nb_true_divide = (binaryfunc){magic_methods['__truediv__'].name},\n"
        if '__lshift__' in magic_methods:
            number_magic_methods_def += f"     .nb_lshift = (binaryfunc){magic_methods['__lshift__'].name},\n"
        if '__rshift__' in magic_methods:
            number_magic_methods_def += f"     .nb_rshift = (binaryfunc){magic_methods['__rshift__'].name},\n"
        if '__and__' in magic_methods:
            number_magic_methods_def += f"     .nb_and = (binaryfunc){magic_methods['__and__'].name},\n"
        if '__or__' in magic_methods:
            number_magic_methods_def += f"     .nb_or = (binaryfunc){magic_methods['__or__'].name},\n"
        if '__iadd__' in magic_methods:
            number_magic_methods_def += f"     .nb_inplace_add = (binaryfunc){magic_methods['__iadd__'].name},\n"
        if '__isub__' in magic_methods:
            number_magic_methods_def += f"     .nb_inplace_subtract = (binaryfunc){magic_methods['__isub__'].name},\n"
        if '__imul__' in magic_methods:
            number_magic_methods_def += f"     .nb_inplace_multiply = (binaryfunc){magic_methods['__imul__'].name},\n"
        if '__itruediv__' in magic_methods:
            number_magic_methods_def += f"     .nb_inplace_true_divide = (binaryfunc){magic_methods['__itruediv__'].name},\n"
        if '__ilshift__' in magic_methods:
            number_magic_methods_def += f"     .nb_inplace_lshift = (binaryfunc){magic_methods['__ilshift__'].name},\n"
        if '__irshift__' in magic_methods:
            number_magic_methods_def += f"     .nb_inplace_rshift = (binaryfunc){magic_methods['__irshift__'].name},\n"
        if '__iand__' in magic_methods:
            number_magic_methods_def += f"     .nb_inplace_and = (binaryfunc){magic_methods['__iand__'].name},\n"
        if '__ior__' in magic_methods:
            number_magic_methods_def += f"     .nb_inplace_or = (binaryfunc){magic_methods['__ior__'].name},\n"
        number_magic_methods_def += '};\n'

        seq_magic_method_name = self.scope.get_new_name(f'{expr.name}_sequence_methods')

        seq_magic_methods_def = f"static PySequenceMethods {seq_magic_method_name} = {{\n"
        if '__len__' in magic_methods:
            seq_magic_methods_def += f"    .sq_length = (lenfunc){magic_methods['__len__'].name},\n"
        seq_magic_methods_def += '};\n'

        map_magic_method_name = self.scope.get_new_name(f'{expr.name}_mapping_methods')
        map_magic_methods_def = f"static PyMappingMethods {map_magic_method_name} = {{\n"
        if '__len__' in magic_methods:
            map_magic_methods_def += f"    .mp_length = (lenfunc){magic_methods['__len__'].name},\n"
        if '__getitem__' in magic_methods:
            map_magic_methods_def += f"     .mp_subscript = (binaryfunc){magic_methods['__getitem__'].name},\n"
        map_magic_methods_def += '};\n'
        method_def_name = self.scope.get_new_name(f'{expr.name}_methods')
        method_def = (f'static PyMethodDef {method_def_name}[] = {{\n'
                        f'{method_def_funcs}'
                        '{ NULL, NULL, 0, NULL}\n'
                        '};\n')

        property_def_name = self.scope.get_new_name(f'{expr.name}_properties')
        property_def = (f'static PyGetSetDef {property_def_name}[] = {{\n'
                        f'{property_definitions}'
                        '};\n')

        type_code = (f"static PyTypeObject {type_name} = {{\n"
                "    PyVarObject_HEAD_INIT(NULL, 0)\n"
                f"    .tp_name = \"{self._module_name}.{name}\",\n"
                f"    .tp_as_number = &{number_magic_method_name},\n"
                f"    .tp_as_sequence = &{seq_magic_method_name},\n"
                f"    .tp_as_mapping = &{map_magic_method_name},\n"
                f"    .tp_doc = PyDoc_STR({docstring}),\n"
                f"    .tp_basicsize = sizeof(struct {struct_name}),\n"
                 "    .tp_itemsize = 0,\n"
                 "    .tp_flags = Py_TPFLAGS_DEFAULT,\n"
                f"    .tp_new = {expr.new_func.name},\n"
                f"{init_string}{del_string}"
                f"    .tp_methods = {method_def_name},\n"
                f"    .tp_getset = {property_def_name},\n"
                "};\n")

        return '\n'.join((method_def, number_magic_methods_def, seq_magic_methods_def,
                        map_magic_methods_def, property_def, type_code, functions))

    def _print_PyModInitFunc(self, expr):
        decs = ''.join(self._print(d) for d in expr.declarations)
        body = self._print(expr.body)
        return ''.join([f'PyMODINIT_FUNC {expr.name}(void)\n{{\n',
                decs,
                body,
                '}\n'])

    def _print_Allocate(self, expr):
        variable = expr.variable
        if isinstance(variable.dtype, WrapperCustomDataType):
            cls_base = variable.cls_base.original_class
            class_def = self.scope.find(cls_base.scope.get_python_name(cls_base.name), 'classes')

            type_name = class_def.type_name
            var_code = self._print(ObjectAddress(variable))
            decl_type = self.get_declare_type(variable)
            return f'{var_code} = ({decl_type}){type_name}.tp_alloc(&{type_name}, 0);\n'
        else:
            return CCodePrinter._print_Allocate(self, expr)

    def _print_Deallocate(self, expr):
        variable = expr.variable
        if isinstance(variable.dtype, WrapperCustomDataType):
            cls_base = variable.cls_base.original_class
            class_def = self.scope.find(cls_base.scope.get_python_name(cls_base.name), 'classes')

            type_name = class_def.type_name
            var_code = self._print(ObjectAddress(variable))
            return f'{type_name}.tp_free({var_code});\n'
        else:
            return CCodePrinter._print_Deallocate(self, expr)

    def _print_Declare(self, expr):
        var = expr.variable
        if isinstance(var.dtype, BindCPointer):
            declaration_type = 'void*'

            static = 'static ' if expr.static else ''
            external = 'extern ' if expr.external else ''

            variable = self._print(expr.variable.name)

            init = f' = {self._print(expr.value)}' if expr.value is not None else ''

            if var.rank == 0:
                return f'{static}{external}{declaration_type} {variable}{init};\n'

            size = var.shape[0]
            if isinstance(size, LiteralInteger):
                return f'{static}{external}{declaration_type} {variable}[{size}];\n'
            else:
                return f'{static}{external}{declaration_type}* {variable}{init};\n'
        else:
            return CCodePrinter._print_Declare(self, expr)

    def _print_IndexedElement(self, expr):
        if isinstance(expr.base.class_type, CStackArray):
            base = self._print(expr.base.name)
            idxs = ''.join(f'[{self._print(a)}]' for a in expr.indices)
            return f'{base}{idxs}'
        else:
            return CCodePrinter._print_IndexedElement(self, expr)

    def _print_Py_ssize_t_Cast(self, expr):
        var = self._print(expr.args[0])
        return f'(Py_ssize_t){var}'

    def _print_PyTuple_Pack(self, expr):
        args = expr.args
        n = len(args)
        args_code = ', '.join(self._print(a) for a in args)
        return f'(*PyTuple_Pack( {n}, {args_code} ))'

    def _print_PyList_Clear(self, expr):
        list_code = self._print(ObjectAddress(expr.list_obj))
        if sys.version_info < (3, 13):
            return f'PyList_SetSlice({list_code}, 0, PY_SSIZE_T_MAX, NULL)'
        else:
            return f'PyList_Clear({list_code})'

    def _print_PyArgumentError(self, expr):
        args = ', '.join([f'"{self._print(expr.error_msg)}"'] + \
                         [f'PyObject_Str((PyObject*)Py_TYPE({self._print(a)}))' for a in expr.args])
        return f'PyErr_SetObject({self._print(expr.error_type)}, PyUnicode_FromFormat({args}));\n'
