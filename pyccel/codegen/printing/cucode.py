# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Provide tools for generating and handling CUDA code.
This module is designed to interface Pyccel's Abstract Syntax Tree (AST) with CUDA,
enabling the direct translation of high-level Pyccel expressions into CUDA code.
"""

from pyccel.codegen.printing.ccode  import CCodePrinter

from pyccel.ast.core                import Import, Module
from pyccel.ast.literals            import Nil

from pyccel.errors.errors           import Errors
from pyccel.ast.cudatypes           import CudaArrayType
from pyccel.ast.datatypes           import HomogeneousContainerType, PythonNativeBool
from pyccel.ast.numpytypes          import numpy_precision_map
from pyccel.ast.cudaext             import CudaFull
from pyccel.ast.numpytypes          import NumpyFloat32Type, NumpyFloat64Type, NumpyComplex64Type, NumpyComplex128Type
from pyccel.ast.numpytypes          import NumpyInt8Type, NumpyInt16Type, NumpyInt32Type, NumpyInt64Type

errors = Errors()

__all__ = ["CudaCodePrinter"]

c_imports = {n : Import(n, Module(n, (), ())) for n in
                ['cuda_ndarrays',
                 'ndarrays',
                 ]}

class CudaCodePrinter(CCodePrinter):
    """
    Print code in CUDA format.

    This printer converts Pyccel's Abstract Syntax Tree (AST) into strings of CUDA code.
    Navigation through this file utilizes _print_X functions,
    as is common with all printers.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    prefix_module : str
            A prefix to be added to the name of the module.
    """
    language = "cuda"

    
    def __init__(self, filename, prefix_module = None):

        errors.set_target(filename)

        super().__init__(filename)

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        self._current_module = expr.name
        body = ''.join(self._print(i) for i in expr.body)

        global_variables = ''.join(self._print(d) for d in expr.declarations)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [Import(expr.name, Module(expr.name,(),())), *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        code = f'{imports}\n\
                 {global_variables}\n\
                 {body}\n'

        self.exit_scope()
        return code

    def function_signature(self, expr, print_arg_names = True):
        """
        Get the Cuda representation of the function signature.

        Extract from the function definition `expr` all the
        information (name, input, output) needed to create the
        function signature and return a string describing the
        function.
        This is not a declaration as the signature does not end
        with a semi-colon.

        Parameters
        ----------
        expr : FunctionDef
            The function definition for which a signature is needed.

        print_arg_names : bool, default : True
            Indicates whether argument names should be printed.

        Returns
        -------
        str
            Signature of the function.
        """
        cuda_decorator = '__global__' if 'kernel' in expr.decorators else \
        '__device__' if 'device' in expr.decorators else ''
        c_function_signature = super().function_signature(expr, print_arg_names)
        return f'{cuda_decorator} {c_function_signature}'

    def _print_KernelCall(self, expr):
        func = expr.funcdef
        args = [a.value or Nil() for a in expr.args]

        args = ', '.join(self._print(a) for a in args)
        return f"{func.name}<<<{expr.num_blocks}, {expr.tp_block}>>>({args});\n"

    def _print_CudaSynchronize(self, expr):
        return 'cudaDeviceSynchronize();\n'

    def _print_ModuleHeader(self, expr):
        self.set_scope(expr.module.scope)
        self._in_header = True
        name = expr.module.name

        funcs = ""
        cuda_headers = ""
        for f in expr.module.funcs:
            if not f.is_inline:
                if 'kernel' in f.decorators or 'device' in f.decorators:
                    cuda_headers += self.function_signature(f) + ';\n'
                else:
                    funcs += self.function_signature(f) + ';\n'
        global_variables = ''.join('extern '+self._print(d) for d in expr.module.declarations if not d.variable.is_private)
        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        self._in_header = False
        self.exit_scope()
        function_declaration = f'{cuda_headers}\n\
                    extern "C"{{\n\
                    {funcs}\
                    }}\n'
        return '\n'.join((f"#ifndef {name.upper()}_H",
                          f"#define {name.upper()}_H",
                          imports,
                          global_variables,
                          function_declaration,
                          "#endif // {name.upper()}_H\n"))
    def _print_Allocate(self, expr):
        variable = expr.variable
        if not isinstance(variable.class_type, CudaArrayType):
            return super()._print_Allocate(expr)
        shape = ", ".join(self._print(i) for i in expr.shape)
        if isinstance(variable.class_type, CudaArrayType):
            dtype = self.find_in_ndarray_type_registry(variable.dtype)
        elif isinstance(variable.class_type, HomogeneousContainerType):
            dtype = self.find_in_ndarray_type_registry(numpy_precision_map[(variable.dtype.primitive_type, variable.dtype.precision)])
        else:
            raise NotImplementedError(f"Don't know how to index {variable.class_type} type")
        shape_Assign = "int64_t shape_Assign [] = {" + shape + "};\n"
        is_view = 'false' if variable.on_heap else 'true'
        memory_location = variable.class_type.memory_location
        if memory_location in ('device', 'host'):
            memory_location = 'allocateMemoryOn' + str(memory_location).capitalize()
        else:
            memory_location = 'managedMemory'
        self.add_import(c_imports['cuda_ndarrays'])
        alloc_code = f"{self._print(expr.variable)} = cuda_array_create({variable.rank},  shape_Assign, {dtype}, {is_view},{memory_location});\n"
        return f'{shape_Assign} {alloc_code}'

    def _print_Deallocate(self, expr):
        var_code = self._print(expr.variable)

        if not isinstance(expr.variable.class_type, CudaArrayType):
            return super()._print_Deallocate(expr)

        if expr.variable.class_type.memory_location == 'host':
            return f"cuda_free_host({var_code});\n"
        else:
            return f"cuda_free({var_code});\n"
    def get_declare_type(self, expr):
        class_type = expr.class_type
        rank  = expr.rank
        if not isinstance(class_type, CudaArrayType ) or rank <= 0:
            return super().get_declare_type(expr)

        dtype = 't_cuda_ndarray'
        return dtype

    def _print_Assign(self, expr):
        rhs = expr.rhs
        if not isinstance(rhs.class_type, CudaArrayType):
                return super()._print_Assign(expr)
        if(isinstance(rhs, (CudaFull))):
            # TODO add support for CudaFull
            return " \n"

