# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CToPythonWrapper
which creates an interface exposing C code to Python.
"""
import warnings
from pyccel.ast.bind_c        import BindCFunctionDef, BindCPointer
from pyccel.ast.bind_c        import BindCModule, BindCModuleVariable, BindCVariable
from pyccel.ast.bind_c        import BindCClassDef, BindCClassProperty, BindCArrayType
from pyccel.ast.builtins      import PythonTuple, PythonRange, PythonLen, PythonSet
from pyccel.ast.builtins      import VariableIterator, PythonStr, PythonList
from pyccel.ast.builtin_methods.dict_methods import DictItems
from pyccel.ast.builtin_methods.list_methods import ListAppend
from pyccel.ast.builtin_methods.set_methods import SetAdd, SetPop
from pyccel.ast.class_defs    import StackArrayClass
from pyccel.ast.core          import Interface, If, IfSection, Return, FunctionCall
from pyccel.ast.core          import FunctionDef, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core          import Assign, AliasAssign, Deallocate, Allocate
from pyccel.ast.core          import Import, Module, AugAssign, CommentBlock, For
from pyccel.ast.core          import FunctionAddress, Declare, ClassDef, AsName
from pyccel.ast.cwrapper      import PyModule, PyccelPyObject, PyArgKeywords, PyModule_Create
from pyccel.ast.cwrapper      import PyArg_ParseTupleNode, Py_None, PyClassDef, PyModInitFunc
from pyccel.ast.cwrapper      import py_to_c_registry, check_type_registry, PyBuildValueNode
from pyccel.ast.cwrapper      import PyErr_SetString, PyTypeError, PyNotImplementedError
from pyccel.ast.cwrapper      import PyAttributeError, Py_ssize_t, Py_ssize_t_Cast
from pyccel.ast.cwrapper      import C_to_Python, PyFunctionDef, PyInterface, PyTuple_Pack
from pyccel.ast.cwrapper      import PyModule_AddObject, Py_DECREF, PyObject_TypeCheck
from pyccel.ast.cwrapper      import Py_INCREF, PyType_Ready, WrapperCustomDataType
from pyccel.ast.cwrapper      import PyList_New, PyList_Append, PyList_GetItem, PyList_SetItem
from pyccel.ast.cwrapper      import PyccelPyTypeObject, PyCapsule_New, PyCapsule_Import
from pyccel.ast.cwrapper      import PySys_GetObject, PyUnicode_FromString, PyGetSetDefElement
from pyccel.ast.cwrapper      import PyTuple_Size, PyTuple_Check, PyTuple_New
from pyccel.ast.cwrapper      import PyTuple_GetItem, PyTuple_SetItem
from pyccel.ast.cwrapper      import PySet_New, PySet_Add, PyList_Check, PyList_Size
from pyccel.ast.cwrapper      import PySet_Size, PySet_Check, PyObject_GetIter, PySet_Clear
from pyccel.ast.cwrapper      import PyIter_Next, PyList_Clear, PyArgumentError
from pyccel.ast.cwrapper      import PyDict_New, PyDict_SetItem
from pyccel.ast.cwrapper      import PyUnicode_AsUTF8, PyUnicode_Check, PyUnicode_GetLength
from pyccel.ast.c_concepts    import ObjectAddress, PointerCast, CStackArray, CNativeInt
from pyccel.ast.c_concepts    import CStrStr
from pyccel.ast.datatypes     import VoidType, PythonNativeInt, CustomDataType, DataTypeFactory
from pyccel.ast.datatypes     import FixedSizeNumericType, HomogeneousTupleType, PythonNativeBool
from pyccel.ast.datatypes     import HomogeneousSetType, HomogeneousListType
from pyccel.ast.datatypes     import HomogeneousContainerType
from pyccel.ast.datatypes     import TupleType, CharType, StringType
from pyccel.ast.internals     import Slice
from pyccel.ast.literals      import Nil, LiteralTrue, LiteralString, LiteralInteger
from pyccel.ast.literals      import LiteralFalse, convert_to_literal
from pyccel.ast.numpytypes    import NumpyNDArrayType, NumpyInt64Type, NumpyInt32Type
from pyccel.ast.numpy_wrapper import PyArray_DATA
from pyccel.ast.numpy_wrapper import get_strides_and_shape_from_numpy_array
from pyccel.ast.numpy_wrapper import pyarray_to_ndarray, PyArray_SetBaseObject, import_array
from pyccel.ast.numpy_wrapper import PyccelPyArrayObject, to_pyarray
from pyccel.ast.numpy_wrapper import numpy_dtype_registry, numpy_flag_f_contig, numpy_flag_c_contig
from pyccel.ast.numpy_wrapper import pyarray_check, is_numpy_array, no_order_check
from pyccel.ast.operators     import PyccelNot, PyccelIsNot, PyccelUnarySub, PyccelEq, PyccelIs
from pyccel.ast.operators     import PyccelLt, IfTernaryOperator, PyccelMul, PyccelAnd
from pyccel.ast.operators     import PyccelNe
from pyccel.ast.variable      import Variable, DottedVariable, IndexedElement
from pyccel.parser.scope      import Scope
from pyccel.errors.errors     import Errors
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO
from .wrapper                 import Wrapper

errors = Errors()

cwrapper_ndarray_imports = [Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ())),
                            Import('ndarrays', Module('ndarrays', (), ()))]

magic_binary_funcs = ('__add__',
                      '__sub__',
                      '__mul__',
                      '__truediv__',
                      '__pow__',
                      '__lshift__',
                      '__rshift__',
                      '__and__',
                      '__or__',
                      '__iadd__',
                      '__isub__',
                      '__imul__',
                      '__itruediv__',
                      '__ipow__',
                      '__ilshift__',
                      '__irshift__',
                      '__iand__',
                      '__ior__',
                      '__getitem__'
                      )

class CToPythonWrapper(Wrapper):
    """
    Class for creating a wrapper exposing C code to Python.

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
    start_language = 'C'

    def __init__(self, sharedlib_dirpath, verbose):
        # A map used to find the Python-compatible Variable equivalent to an object in the AST
        self._python_object_map = {}
        # The object that should be returned to indicate an error
        self._error_exit_code = Nil()

        self._sharedlib_dirpath = sharedlib_dirpath
        super().__init__(verbose)

    def get_new_PyObject(self, name, dtype = None, is_temp = False):
        """
        Create new `PyccelPyObject` `Variable` with the desired name.

        Create a new `Variable` with the datatype `PyccelPyObject` and the desired name.
        A `PyccelPyObject` datatype means that this variable can be accessed and
        manipulated from Python.

        Parameters
        ----------
        name : str
            The desired name.

        dtype : DataType, optional
            The datatype of the object which will be represented by this PyObject.
            This is not necessary unless a variable sis required which will describe
            a class.

        is_temp : bool, default=False
            Indicates if the Variable is temporary. A temporary variable may be ignored
            by the printer.

        Returns
        -------
        Variable
            The new variable.
        """
        if isinstance(dtype, CustomDataType):
            var = Variable(self._python_object_map[dtype],
                           self.scope.get_new_name(name),
                           memory_handling='alias',
                           cls_base = self.scope.find(dtype.name, 'classes', raise_if_missing = True),
                           is_temp=is_temp)
        else:
            var = Variable(PyccelPyObject(),
                           self.scope.get_new_name(name),
                           memory_handling='alias',
                           is_temp=is_temp)
        self.scope.insert_variable(var)
        return var

    def _get_python_argument_variables(self, args):
        """
        Get a new set of `PyccelPyObject` `Variable`s representing each of the arguments.

        Create a new `PyccelPyObject` variable for each argument returned in Python.
        The results are saved to the `self._python_object_map` dictionary so they can be
        discovered later.

        Parameters
        ----------
        args : iterable of FunctionDefArguments
            The arguments of the function.

        Returns
        -------
        list of Variable
            Variables which will hold the arguments in Python.
        """
        orig_args = [getattr(a.var, 'original_var', a.var) for a in args]
        is_bound = [getattr(a, 'wrapping_bound_argument', a.bound_argument) for a in args]
        collect_args = [self.get_new_PyObject(o_a.name+'_obj',
                                              dtype = o_a.dtype if b else None)
                        for a, b, o_a in zip(args, is_bound, orig_args)]
        self._python_object_map.update(dict(zip(args, collect_args)))
        return collect_args

    def _unpack_python_args(self, args, class_base = None):
        """
        Unpack the arguments received from Python into the expected Python variables.

        Create the wrapper arguments of the current `FunctionDef` (`self`, `args`, `kwargs`).
        Get a new set of `PyccelPyObject` `Variable`s representing each of the expected
        arguments. Add the code which unpacks the `args` and `kwargs` into individual
        `PyccelPyObject`s for each of the expected arguments.

        Parameters
        ----------
        args : iterable of FunctionDefArguments
            The expected arguments of the function.

        class_base : DataType, optional
            The DataType of the class which the method belongs to. In the case of a method
            defined in a module this value is None.

        Returns
        -------
        func_args : list of Variable
            The arguments of the FunctionDef.

        body : list of pyccel.ast.basic.PyccelAstNode
            The code which unpacks the arguments.

        Examples
        --------
        >>> arg = Variable('int', 'x')
        >>> func_args = (FunctionDefArgument(arg),)
        >>> wrapper_args, body = self._unpack_python_args(func_args)
        >>> wrapper_args
        [Variable('self', dtype=PyccelPyObject()), Variable('args', dtype=PyccelPyObject()), Variable('kwargs', dtype=PyccelPyObject())]
        >>> body
        [<pyccel.ast.cwrapper.PyArgKeywords object at 0x7f99ec128cc0>, <pyccel.ast.core.If object at 0x7f99ed3a5b20>]
        >>> CWrapperCodePrinter('wrapper_file.c').doprint(expr)
        static char *kwlist[] = {
            "x",
            NULL
        };
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &x_obj))
        {
            return NULL;
        }
        """
        has_bound_arg = class_base is not None
        bound_arg = args[0] if has_bound_arg else None
        args = args[int(has_bound_arg):]
        # Create necessary variables
        func_args = [self.get_new_PyObject("self", class_base)] + [self.get_new_PyObject(n) for n in ("args", "kwargs")]
        arg_vars  = self._get_python_argument_variables(args)
        keyword_list_name = self.scope.get_new_name('kwlist')

        if has_bound_arg:
            self._python_object_map[bound_arg] = func_args[0]

        # Create the list of argument names
        arg_names = ['' if a.is_posonly else getattr(a.var, 'original_var', a.var).name for a in args]
        keyword_list = PyArgKeywords(keyword_list_name, arg_names)

        # Parse arguments
        parse_node = PyArg_ParseTupleNode(*func_args[1:], args, arg_vars, keyword_list)

        # Initialise optionals
        body = [AliasAssign(py_arg, Py_None) for func_def_arg, py_arg in zip(args, arg_vars) if func_def_arg.has_default]

        body.append(keyword_list)
        body.append(If(IfSection(PyccelNot(parse_node), [Return(self._error_exit_code)])))

        return func_args, body

    def _get_python_result_variables(self, results):
        """
        Get a new set of `PyccelPyObject` `Variable`s representing each of the results.

        Create a new `PyccelPyObject` variable for each result returned in Python.
        The results are saved to the `self._python_object_map` dictionary so they can be
        discovered later.

        Parameters
        ----------
        results : iterable of FunctionDefResults
            The results of the function.

        Returns
        -------
        list of Variable
            Variables which will hold the results in Python.
        """
        collect_results = [self.get_new_PyObject(r.var.name+'_obj', getattr(r, 'original_function_result_variable', r.var).dtype) for r in results]
        self._python_object_map.update(dict(zip(results, collect_results)))
        return collect_results

    def _get_type_check_condition(self, py_obj, arg, raise_error, body, allow_empty_arrays):
        """
        Get the condition which checks if an argument has the expected type.

        Using the C-compatible description of a function argument, determine whether the Python
        object (with datatype `PyccelPyObject`) holds data which is compatible with the expected
        type. The check is returned along with any errors that may be raised depending upon the
        result and the value of `raise_error`.

        Parameters
        ----------
        py_obj : Variable
            The variable with datatype `PyccelPyObject` where the arguments is stored in Python.

        arg : Variable
            The C-compatible variable which holds all the details about the expected type.

        raise_error : bool
            True if an error should be raised in case of an unexpected type, False otherwise.

        body : list
            A list describing code where the type check will occur. This allows any necessary code
            to be inserted into the code block. E.g. code which should be run before the condition
            can be checked.

        allow_empty_arrays : bool
            A boolean indicating whether empty arrays are authorised. This is necessary as STC
            does not handle empty arrays.

        Returns
        -------
        type_check_condition : FunctionCall | Variable
            The function call which checks if the argument has the expected type or the variable
            indicating if the argument has the expected type.

        error_code : tuple of pyccel.ast.basic.PyccelAstNode
            The code which raises any necessary errors.
        """
        rank = arg.rank
        error_code = ()
        dtype = arg.dtype
        if isinstance(dtype, CustomDataType):
            python_cls_base = self.scope.find(dtype.name, 'classes', raise_if_missing = True)
            type_check_condition = PyObject_TypeCheck(py_obj, python_cls_base.type_object)
        elif isinstance(dtype, StringType):
            type_check_condition = PyccelNe(PyUnicode_Check(py_obj), LiteralInteger(0))
        elif rank == 0:
            try :
                cast_function = check_type_registry[dtype]
            except KeyError:
                errors.report(f"Can't check the type of {dtype}\n"+PYCCEL_RESTRICTION_TODO,
                        symbol=arg, severity='fatal')
            func = FunctionDef(name = cast_function,
                               body      = [],
                               arguments = [FunctionDefArgument(Variable(PyccelPyObject(), name = 'o', memory_handling='alias'))],
                               results   = FunctionDefResult(Variable(PythonNativeBool(), name = 'v')))

            type_check_condition = func(py_obj)
        elif isinstance(arg.class_type, NumpyNDArrayType):
            try :
                type_ref = numpy_dtype_registry[dtype]
            except KeyError:
                errors.report(f"Can't check the type of an array of {dtype}\n"+PYCCEL_RESTRICTION_TODO,
                        symbol=arg, severity='fatal')

            # order flag
            if rank == 1:
                flag     = no_order_check
            elif arg.order == 'F':
                flag = numpy_flag_f_contig
            else:
                flag = numpy_flag_c_contig

            allow_empty = convert_to_literal(allow_empty_arrays)

            if raise_error:
                type_check_condition = pyarray_check(CStrStr(LiteralString(arg.name)), py_obj, type_ref,
                                 LiteralInteger(rank), flag, allow_empty)
            else:
                type_check_condition = is_numpy_array(py_obj, type_ref, LiteralInteger(rank), flag, allow_empty)

        elif isinstance(arg.class_type, HomogeneousContainerType):
            # Create type check result variable
            type_check_condition = self.scope.get_temporary_variable(PythonNativeBool(), 'is_homog_set')

            check_funcs = {'set': PySet_Check,
                           'tuple': PyTuple_Check,
                           'list': PyList_Check}

            size_getter = {'set': PySet_Size,
                           'tuple': PyTuple_Size,
                           'list': PyList_Size}

            if arg.class_type.name not in check_funcs:
                return errors.report(f"Wrapping function arguments is not implemented for type {arg.class_type}. "
                        + PYCCEL_RESTRICTION_TODO, symbol=arg, severity='fatal')

            # Check if the object is a set
            type_check = PyccelNe(check_funcs[arg.class_type.name](py_obj), LiteralInteger(0))

            # If the set is an object check that the elements have the right type
            for_scope = self.scope.create_new_loop_scope()
            size_var = self.scope.get_temporary_variable(PythonNativeInt(), 'size')
            idx = self.scope.get_temporary_variable(CNativeInt())
            indexed_py_obj = self.scope.get_temporary_variable(PyccelPyObject(), memory_handling='alias')
            iter_obj = self.scope.get_temporary_variable(PyccelPyObject(), 'iter', memory_handling='alias')

            size_assign = Assign(size_var, size_getter[arg.class_type.name](py_obj))
            iter_assign = AliasAssign(iter_obj, PyObject_GetIter(py_obj))
            indexed_init = AliasAssign(indexed_py_obj, PyIter_Next(iter_obj))
            for_body = [indexed_init]
            internal_type_check_condition, _ = self._get_type_check_condition(indexed_py_obj, arg[0], False,
                                                                              for_body, allow_empty_arrays)
            for_body.append(Assign(type_check_condition, PyccelAnd(type_check_condition, internal_type_check_condition)))
            internal_type_check = For((idx,), PythonRange(size_var), for_body, scope = for_scope)

            type_checks = IfSection(type_check, [size_assign, iter_assign, Assign(type_check_condition, LiteralTrue()), internal_type_check])
            default_value = IfSection(LiteralTrue(), [Assign(type_check_condition, LiteralFalse())])
            body.append(If(type_checks, default_value))
        else:
            errors.report(f"Can't check the type of an array of {arg.class_type}\n"+PYCCEL_RESTRICTION_TODO,
                    symbol=arg, severity='fatal')

        if raise_error and not isinstance(arg.class_type, NumpyNDArrayType):
            # No error code required for arrays as the error is raised inside pyarray_check
            python_error = PyArgumentError(PyTypeError,
                                f"Expected an argument of type {arg.class_type} for argument {arg.name}. Received {{type(arg)}}",
                                arg = py_obj)
            error_code = (python_error,)

        return type_check_condition, error_code

    def _get_type_check_function(self, name, args, funcs):
        """
        Determine the flags which allow correct function to be identified from the interface.

        Each function must be identifiable by a different integer value. This value is known
        as a flag. Different parts of the flag indicate the types of different arguments.
        Take for example the following function:
        ```python
        @types('int', 'int')
        @types('float', 'float')
        def f(a, b):
            pass
        ```
        The values 0 (int) and 1 (float) would indicate the type of the argument a. In order
        to preserve this information the values which indicate the type of the argument b
        must only change the part of the flag which does not contain this information. In other
        words `flag % n_types_a = flag_a`. Therefore the values 0 (int) and 2(float) indicate
        the type of the argument b.
        We then finally have the following four options:
          1. 0 = 0 + 0 => (int,int)
          2. 1 = 1 + 0 => (float,int)
          3. 2 = 0 + 2 => (int, float)
          4. 3 = 1 + 2 => (float, float)

        of which only the first and last flags indicate acceptable arguments.

        The function returns a dictionary whose keys are the functions and whose values are
        a list of the flags which would indicate the correct types.
        In the above example we would return `{func_0 : [0,0], func_1 : [1,2]}`.
        It also returns a FunctionDef which determines the index of the chosen function.

        Parameters
        ----------
        name : str
            The name of the function to be generated.

        args : iterable of Variable
            A list containing the variables of datatype `PyccelPyObject` describing the
            arguments that were passed to the function from Python.

        funcs : list of FunctionDefs
            The functions in the Interface.

        Returns
        -------
        func : FunctionDef
            The function which determines the key identifying the relevant function.

        argument_type_flags : dict
            A dictionary whose keys are the functions and whose values are the integer keys
            which indicate that the function should be chosen.
        """
        args = [a.clone(a.name, is_argument = True) for a in args]
        func_scope = self.scope.new_child_scope(name, 'function')
        self.scope = func_scope
        orig_funcs = [getattr(func, 'original_function', func) for func in funcs]
        type_indicator = Variable(PythonNativeInt(), self.scope.get_new_name('type_indicator'))
        is_bind_c = isinstance(funcs[0], BindCFunctionDef)

        # Initialise the argument_type_flags
        argument_type_flags = {func : 0 for func in funcs}

        # Initialise type_indicator
        body = [Assign(type_indicator, LiteralInteger(0))]

        step = 1
        for i, py_arg in enumerate(args):
            # Get the relevant typed arguments from the original functions
            interface_args = [func.arguments[i].var for func in orig_funcs]
            # Get a dictionary mapping each unique type key to an example argument
            type_to_example_arg = {a.class_type : a for a in interface_args}
            # Get a list of unique keys
            possible_types = list(type_to_example_arg.keys())

            n_possible_types = len(possible_types)
            if n_possible_types != 1:
                # Update argument_type_flags with the index of the type key
                for func, a in zip(funcs, interface_args):
                    index = next(i for i, p_t in enumerate(possible_types) if p_t is a.class_type)*step
                    argument_type_flags[func] += index

                # Create the type checks and incrementation of the type_indicator
                if_blocks = []
                for index, t in enumerate(possible_types):
                    check_func_call, _ = self._get_type_check_condition(py_arg, type_to_example_arg[t], False, body,
                                allow_empty_arrays = is_bind_c)
                    if_blocks.append(IfSection(check_func_call, [AugAssign(type_indicator, '+', LiteralInteger(index*step))]))
                body.append(If(*if_blocks, IfSection(LiteralTrue(),
                            [PyArgumentError(PyTypeError, f"Unexpected type for argument {interface_args[0].name}. Received {{type(arg)}}",
                                arg = py_arg),
                             Return(PyccelUnarySub(LiteralInteger(1)))])))
            elif not orig_funcs[0].arguments[i].has_default:
                check_func_call, err_body = self._get_type_check_condition(py_arg, type_to_example_arg.popitem()[1], True, body,
                                allow_empty_arrays = is_bind_c)
                err_body = err_body + (Return(PyccelUnarySub(LiteralInteger(1))), )
                if_sec = IfSection(PyccelNot(check_func_call), err_body)
                body.append(If(if_sec))

            # Update the step to ensure unique indices for each argument
            step *= n_possible_types

        body.append(Return(type_indicator))

        self.exit_scope()

        docstring = CommentBlock("Assess the types. Raise an error for unexpected types and calculate an integer\n" +
                        "which indicates which function should be called.")

        # Build the function
        func = FunctionDef(name, [FunctionDefArgument(a) for a in args], body,
                            FunctionDefResult(type_indicator), docstring=docstring, scope=func_scope)

        return func, argument_type_flags

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
        func_args = [FunctionDefArgument(self.get_new_PyObject(n)) for n in ("self", "args", "kwargs")]
        if self._error_exit_code is Nil():
            func_results = FunctionDefResult(self.get_new_PyObject("result", is_temp=True))
        else:
            func_results = FunctionDefResult(self.scope.get_temporary_variable(self._error_exit_code.class_type, "result"))
        function = PyFunctionDef(name = name, arguments = func_args, results = func_results,
                body = [PyErr_SetString(PyNotImplementedError, CStrStr(LiteralString(error_msg))),
                        Return(self._error_exit_code)],
                scope = scope, original_function = original_function)

        self.scope = current_scope

        self.scope.insert_function(function, self.scope.get_python_name(name))

        return function

    def _save_referenced_objects(self, func, func_args):
        """
        Save any arguments passed to the wrapper which are then stored in pointers.

        If arguments are saved into pointers (e.g. inside classes) then their reference
        counter must be incremented. This prevents them being deallocated if they go
        out of scope in Python. The class must then take care to decrement their
        reference counter when it is itself deallocated to prevent a memory leak.
        The attribute `FunctionDefArgument.persistent_target` indicates whether an
        argument is a target inside the function. When it is true then additional code
        is added to the wrapper body. This code increments the reference counter for
        the argument and adds the object to a list of objects whose reference counter
        must be decremented in the class destructor.

        Parameters
        ----------
        func : FunctionDef
            The function being wrapped.
        func_args : list[FunctionDefArgument] | list[Variable]
            The arguments passed by Python to the function (self, args, kwargs).

        Returns
        -------
        list
            A list of any expressions which should be added to the wrapper body to
            add references to the arguments.
        """
        body = []
        class_arg_var = func_args[0]
        if isinstance(class_arg_var, FunctionDefArgument):
            class_arg_var = class_arg_var.var
        class_scope = class_arg_var.cls_base.scope
        for a in func.arguments:
            if a.persistent_target:
                ref_attribute = class_scope.find('referenced_objects', 'variables', raise_if_missing = True)
                ref_list = ref_attribute.clone(ref_attribute.name, new_class = DottedVariable, lhs = class_arg_var)
                python_arg = self._python_object_map[a]
                if not isinstance(python_arg.dtype, PyccelPyObject):
                    python_arg = ObjectAddress(PointerCast(python_arg, PyList_Append.arguments[1].var))
                append_call = PyList_Append(ref_list, python_arg)
                body.extend([If(IfSection(PyccelEq(append_call, PyccelUnarySub(LiteralInteger(1))),
                                          [Return(self._error_exit_code)]))])
        return body

    def _incref_return_pointer(self, ref_obj, return_var, orig_var):
        """
        Get the code necessary to return an object which references another.

        Get the code necessary to return an object which references another Python object. This is necessary when
        wrapping functions (or getters) which return pointers (e.g. attributes of a class). For these objects the
        target must not be deallocated before the returned object is no longer needed. For arrays this is achieved
        using PyArray_SetBaseObject, to save the reference. For class instances the self instance is added to the
        list of referenced objects saved in the returned class.

        Parameters
        ----------
        ref_obj : Variable
            A variable representing the class instance which must not be deallocated too early.
        return_var : Variable
            The variable which will be returned from the function.
        orig_var : Variable
            The variable which will be returned from the function as it appeared in the original code.

        Returns
        -------
        list[PyccelAstNode]
            Any nodes which must be printed to increase reference counts.
        """
        if isinstance(orig_var.class_type, NumpyNDArrayType):
            save_ref_call = PyArray_SetBaseObject(
                    ObjectAddress(PointerCast(return_var, PyArray_SetBaseObject.arguments[0].var)),
                    ObjectAddress(PointerCast(ref_obj, PyArray_SetBaseObject.arguments[1].var)))
            return [Py_INCREF(ref_obj),
                    If(IfSection(PyccelLt(save_ref_call,LiteralInteger(0, dtype=CNativeInt())),
                                      [Return(self._error_exit_code)]))]
        elif isinstance(orig_var.dtype, CustomDataType):
            ref_attribute = return_var.cls_base.scope.find('referenced_objects', 'variables', raise_if_missing = True)
            ref_list = ref_attribute.clone(ref_attribute.name, new_class = DottedVariable, lhs = return_var)
            save_ref_call = PyList_Append(ref_list, ObjectAddress(PointerCast(ref_obj, ref_list)))
            return [If(IfSection(PyccelLt(save_ref_call,LiteralInteger(0, dtype=CNativeInt())),
                                      [Return(self._error_exit_code)]))]
        elif isinstance(orig_var.class_type, FixedSizeNumericType):
            return []
        else:
            raise NotImplementedError(f"Unsure how to preserve references for attribute of type {type(orig_var.class_type)}")

    def _add_object_to_mod(self, module_var, obj, name, initialised):
        """
        Get code for adding an object to the module.

        This function creates the AST nodes necessary to add an object to
        the module. This includes the creation of the success check and
        the dereferencing of any objects used.

        Parameters
        ----------
        module_var : Variable
            The variable containing the PyObject* which describes the module.

        obj : Variable
            The variable containing the PyObject* which should be added to the module.

        name : str
            The name by which the object will be known in Pyccel.

        initialised : list[Variable]
            A list of the variables which have had their reference counter incremented
            and must therefore decrement their counter if an error is raised.

        Returns
        -------
        list[PyccelAstNode]
            The code which adds the object to the module.
        """
        add_expr = PyModule_AddObject(module_var, CStrStr(LiteralString(name)), obj)
        if_expr = If(IfSection(PyccelLt(add_expr, LiteralInteger(0)),
                        [Py_DECREF(i) for i in initialised] +
                        [Return(self._error_exit_code)]))
        initialised.append(obj)
        return [if_expr, Py_INCREF(obj)]

    def _build_module_init_function(self, expr, imports, module_def_name):
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
        mod_name = self.scope.get_python_name(getattr(expr, 'original_module', expr).name)
        # The name of the init function is compulsory for the wrapper to work
        func_name = f'PyInit_{mod_name}'
        # Initialise the scope
        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope

        for v in expr.variables:
            func_scope.insert_symbol(v.name)

        n_classes = len(expr.classes)

        # Create necessary variables
        module_var = self.get_new_PyObject("mod")
        API_var_name = self.scope.get_new_name(f'Py{mod_name}_API', object_type = 'wrapper')
        API_var = Variable(CStackArray(BindCPointer()), API_var_name, shape = (n_classes,),
                                    cls_base = StackArrayClass)
        self.scope.insert_variable(API_var)
        capsule_obj = self.get_new_PyObject(self.scope.get_new_name('c_api_object'))

        body = [AliasAssign(module_var, PyModule_Create(module_def_name)),
                If(IfSection(PyccelIs(module_var, Nil()), [Return(self._error_exit_code)]))]

        initialised = [module_var]

        # Save classes to the module variable
        for i,c in enumerate(expr.classes):
            wrapped_class = self._python_object_map[c]
            type_object = wrapped_class.type_object

            API_elem = IndexedElement(API_var, i)
            body.append(AliasAssign(API_elem, PointerCast(ObjectAddress(type_object), API_elem)))

        ok_code = LiteralInteger(0)

        # Save Capsule describing types (needed for dependent modules)
        body.append(AliasAssign(capsule_obj, PyCapsule_New(API_var, mod_name)))
        body.extend(self._add_object_to_mod(module_var, capsule_obj, '_C_API', initialised))

        body.append(import_array())
        import_funcs = [i.source_module.import_func for i in imports if isinstance(i.source_module, PyModule)]
        for i_func in import_funcs:
            body.append(If(IfSection(PyccelLt(i_func(), ok_code),
                            [Py_DECREF(i) for i in initialised] +
                            [Return(self._error_exit_code)])))

        # Call the initialisation function
        if expr.init_func:
            body.append(expr.init_func())

        # Save classes to the module variable
        for i,c in enumerate(expr.classes):
            wrapped_class = self._python_object_map[c]
            type_object = wrapped_class.type_object
            class_name = self.scope.get_python_name(wrapped_class.name)

            ready_type = PyType_Ready(type_object)
            if_expr = If(IfSection(PyccelLt(ready_type, LiteralInteger(0)),
                            [Py_DECREF(i) for i in initialised] +
                            [Return(self._error_exit_code)]))
            body.append(if_expr)

            body.extend(self._add_object_to_mod(module_var, type_object, class_name, initialised))

        # Save module variables to the module variable
        for v in expr.variables:
            if v.is_private:
                continue
            body.extend(self._wrap(v))
            wrapped_var = self._python_object_map[v]
            var_name = self.scope.get_python_name(v.name)
            body.extend(self._add_object_to_mod(module_var, wrapped_var, var_name, initialised))

        body.append(Return(module_var))

        self.exit_scope()

        return PyModInitFunc(func_name, body, [API_var], func_scope)

    def _build_module_import_function(self, expr):
        """
        Build the function that will be called in order to use the module from another module.

        Build the function that will be called when the module is first imported.
        This function must import the capsule created in the module initialisation.
        In order for this to work from any folder the `sys.path` list is modified to include
        the folder where the file is located (currently this is done by temporarily modifying
        an element of the list as the stable C-Python API doesn't contain any functions for
        reducing the size of lists).
        See <https://docs.python.org/3/extending/extending.html>
        for more details.

        Parameters
        ----------
        expr : Module
            The module of interest.

        Returns
        -------
        API_var : Variable
            The variable which contains the data extracted from the capsule.

        import_func : FunctionDef
            The import function.
        """
        mod_name = self.scope.get_python_name(getattr(expr, 'original_module', expr).name)
        # Initialise the scope
        func_name = self.scope.get_new_name(f'import')

        API_var_name = self.scope.insert_symbol(f'Py{mod_name}_API', 'wrapper')
        API_var = Variable(CStackArray(BindCPointer()), API_var_name, shape = (None,),
                                    cls_base = StackArrayClass,
                                    memory_handling = 'alias')
        self.scope.insert_variable(API_var)

        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope

        ok_code = LiteralInteger(0, dtype=CNativeInt())
        error_code = PyccelUnarySub(LiteralInteger(1, dtype=CNativeInt()))
        self._error_exit_code = error_code

        # Create variables to temporarily modify the Python path so the file will be discovered
        current_path = func_scope.get_temporary_variable(PyccelPyObject(), 'current_path', memory_handling='alias')
        stash_path = func_scope.get_temporary_variable(PyccelPyObject(), 'stash_path', memory_handling='alias')

        body = [AliasAssign(current_path, PySys_GetObject(CStrStr(LiteralString("path")))),
                AliasAssign(stash_path, PyList_GetItem(current_path, LiteralInteger(0, dtype=CNativeInt()))),
                Py_INCREF(stash_path),
                If(IfSection(PyccelEq(PyList_SetItem(current_path, LiteralInteger(0, dtype=CNativeInt()),
                                                PyUnicode_FromString(CStrStr(LiteralString(self._sharedlib_dirpath)))),
                                      PyccelUnarySub(LiteralInteger(1))),
                             [Return(self._error_exit_code)])),
                AliasAssign(API_var, PyCapsule_Import(mod_name)),
                If(IfSection(PyccelEq(PyList_SetItem(current_path, LiteralInteger(0, dtype=CNativeInt()), stash_path),
                                      PyccelUnarySub(LiteralInteger(1))),
                             [Return(self._error_exit_code)])),
                Return(IfTernaryOperator(PyccelIsNot(API_var, Nil()), ok_code, error_code))]

        result = func_scope.get_temporary_variable(CNativeInt())
        self.exit_scope()
        self._error_exit_code = Nil()
        import_func = FunctionDef(func_name, (), body, FunctionDefResult(result), is_static=True, scope = func_scope)

        return API_var, import_func

    def _allocate_class_instance(self, class_var, scope, is_alias):
        """
        Get all expressions necessary to allocate a new class description.

        Get all expressions necessary to allocate a new class description, this includes allocating
        the object itself, creating the list of referenced_objects and saving the alias status.

        Parameters
        ----------
        class_var : Variable
            The variable where the class instance is stored.

        scope : Scope
            The scope of the class (containing the class attributes).

        is_alias : bool
            A boolean indicating if an alias is being stored.

        Returns
        -------
        list[PyccelAstNode]
            A list of expressions necessary to allocate a new class description.
        """
        # Get the list of referenced objects
        ref_attribute = scope.find('referenced_objects', 'variables', raise_if_missing = True)
        ref_list = ref_attribute.clone(ref_attribute.name, new_class = DottedVariable, lhs = class_var)

        # Get alias attribute
        attribute = scope.find('is_alias', 'variables', raise_if_missing = True)
        alias_bool = attribute.clone(attribute.name, new_class = DottedVariable, lhs = class_var)

        alias_val = LiteralTrue() if is_alias else LiteralFalse()

        return [Allocate(class_var, shape=None, status='unallocated'),
                AliasAssign(ref_list, PyList_New()),
                Assign(alias_bool, alias_val)]

    def _get_class_allocator(self, class_dtype, func = None):
        """
        Create the allocator for the class.

        Create a function which will allocate the memory for the class instance. This
        is equivalent to the `__new__` function.

        Parameters
        ----------
        class_dtype : DataType
            The datatype of the class being translated.

        func : FunctionDef, optional
            The function which provides a new instance of the class.

        Returns
        -------
        PyFunctionDef
            A function that can be called to create the class instance.
        """
        if func:
            func_name = self.scope.get_new_name(f'{func.name}__wrapper', object_type = 'wrapper')
        else:
            func_name = self.scope.get_new_name(f'{class_dtype.name}__new__wrapper')
        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope

        self_var = Variable(PyccelPyTypeObject(), name=self.scope.get_new_name('self'),
                              memory_handling='alias')
        self.scope.insert_variable(self_var, 'self')
        func_args = [self_var] + [self.get_new_PyObject(n) for n in ("args", "kwargs")]
        func_args = [FunctionDefArgument(a) for a in func_args]

        func_results = FunctionDefResult(self.get_new_PyObject("result", is_temp=True))

        # Get the results of the PyFunctionDef
        python_result_var = self.get_new_PyObject('result_obj', class_dtype)
        scope = python_result_var.cls_base.scope
        attribute = scope.find('instance', 'variables', raise_if_missing = True)
        c_res = attribute.clone(attribute.name, new_class = DottedVariable, lhs = python_result_var)

        body = self._allocate_class_instance(python_result_var, scope, False)

        if func:
            body.append(AliasAssign(c_res, func()))
        else:
            result_name = self.scope.get_new_name('result')
            result = Variable(class_dtype, result_name)
            body.append(Allocate(c_res, shape=None, status='unallocated',
                         like = result))

        body.append(Return(PointerCast(python_result_var, func_results.var)))

        self.exit_scope()

        return PyFunctionDef(func_name, func_args, body, func_results,
                             scope=func_scope, original_function = None)

    def _get_class_initialiser(self, init_function, cls_dtype):
        """
        Create the constructor for the class.

        Create a function which will initialise the class. This function creates
        the `__new__` function to allocate the memory which stores the class
        instance and calls the `__init__` function.

        Parameters
        ----------
        init_function : FunctionDef
            The `__init__` function in the translated class.

        cls_dtype : DataType
            The datatype of the class being translated.

        Returns
        -------
        new_function : PyFunctionDef
            A function that can be called to create the class instance.

        init_function : PyFunctionDef
            A function that can be called to create the class instance.
        """
        original_func = getattr(init_function, 'original_function', init_function)
        func_name = self.scope.get_new_name(f'{cls_dtype.name}__init__wrapper')
        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope
        self._error_exit_code = PyccelUnarySub(LiteralInteger(1, dtype=CNativeInt()))

        is_bind_c_function_def = isinstance(init_function, BindCFunctionDef)

        # Handle un-wrappable functions
        if any(isinstance(a.var, FunctionAddress) for a in init_function.arguments):
            self.exit_scope()
            warnings.warn("Functions with functions as arguments will not be callable from Python")
            return self._get_untranslatable_function(func_name,
                         func_scope, init_function,
                         "Cannot pass a function as an argument")

        # Add the variables to the expected symbols in the scope
        for a in init_function.arguments:
            a_var = a.var
            func_scope.insert_symbol(getattr(a_var, 'original_var', a_var).name)

        # Get variables describing the arguments and results that are seen from Python
        python_args = init_function.arguments

        # Get the arguments of the PyFunctionDef
        func_args, body = self._unpack_python_args(python_args, cls_dtype)
        func_args = [FunctionDefArgument(a) for a in func_args]

        # Get the results of the PyFunctionDef
        python_result_variable = Variable(CNativeInt(), self.scope.get_new_name(), is_temp = True)

        # Get the code required to extract the C-compatible arguments from the Python arguments
        wrapped_args = [self._wrap(a) for a in python_args]
        body += [l for a in wrapped_args for l in a['body']]

        # Get the arguments and results which should be used to call the c-compatible function
        func_call_args = [ca for a in wrapped_args for ca in a['args']]

        body.extend(self._save_referenced_objects(init_function, func_args))

        # Call the C-compatible function
        body.append(init_function(*func_call_args))

        # Pack the Python compatible results of the function into one argument.
        func_results = FunctionDefResult(python_result_variable)
        body.append(Return(LiteralInteger(0, dtype=CNativeInt())))

        self.exit_scope()
        for a in python_args:
            if not a.bound_argument:
                self._python_object_map.pop(a)

        function = PyFunctionDef(func_name, func_args, body, func_results, scope=func_scope,
                docstring = init_function.docstring, original_function = original_func)

        self.scope.insert_function(function, func_scope.get_python_name(func_name))
        self._python_object_map[init_function] = function
        self._error_exit_code = Nil()

        return function

    def _get_class_destructor(self, del_function, cls_dtype, wrapper_scope):
        """
        Create the destructor for the class.

        Create a function which will act as a destructor for the class. This
        function calls the `__del__` function and frees the memory allocated
        to store the class instance.

        Parameters
        ----------
        del_function : FunctionDef
            The `__del__` function in the translated class.

        cls_dtype : DataType
            The datatype of the class being translated.

        wrapper_scope : Scope
            The scope for the wrapped version of the class.

        Returns
        -------
        PyFunctionDef
            A function that can be called to destroy the class instance.
        """
        original_func = getattr(del_function, 'original_function', del_function)
        func_name = self.scope.get_new_name(f'{cls_dtype.name}__del__wrapper')
        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope

        # Add the variables to the expected symbols in the scope
        for a in del_function.arguments:
            func_scope.insert_symbol(a.var.name)
        func_arg = self.get_new_PyObject('self', cls_dtype)

        attribute = wrapper_scope.find('instance', 'variables')
        c_obj = attribute.clone(attribute.name, new_class = DottedVariable, lhs = func_arg)

        attribute = wrapper_scope.find('is_alias', 'variables')
        is_alias = attribute.clone(attribute.name, new_class = DottedVariable, lhs = func_arg)

        if isinstance(del_function, BindCFunctionDef):
            body = [del_function(c_obj)]
        else:
            body = [del_function(c_obj),
                    Deallocate(c_obj)]
        body = [If(IfSection(PyccelNot(is_alias), body))]

        # Get the list of referenced objects
        ref_attribute = wrapper_scope.find('referenced_objects', 'variables', raise_if_missing = True)
        ref_list = ref_attribute.clone(ref_attribute.name, new_class = DottedVariable, lhs = func_arg)

        body.extend([Py_DECREF(ref_list),
                     Deallocate(func_arg)])

        self.exit_scope()

        function = PyFunctionDef(func_name, [FunctionDefArgument(func_arg)], body, scope=func_scope,
                original_function = original_func)

        self.scope.insert_function(function, func_scope.get_python_name(func_name))
        self._python_object_map[del_function] = function

        return function

    def _get_array_parts(self, orig_var, collect_arg):
        """
        Get AST nodes describing the extraction of the data pointer, shape, and strides from a Python array object.

        Get AST nodes describing the extraction of the data pointer, shape, and strides from a Python array object.
        These nodes as well as the new objects can then be packed into a structure or passed directly to a function
        depending on the target language.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefArgument being wrapped.

        collect_arg : Variable
            A variable with type PythonObject* holding the Python argument from which the
            C-compatible argument should be collected.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list containing the AST nodes which extract the data pointer, shape, and strides.
             - data : a Variable describing a pointer in which the data is stored.
             - shape : a Variable describing a stack array in which the shape information is stored.
             - strides : a Variable describing a stack array in which the strides are stored.
        """
        pyarray_collect_arg = PointerCast(collect_arg, Variable(PyccelPyArrayObject(), '_', memory_handling = 'alias'))
        data_var = Variable(VoidType(), self.scope.get_new_name(orig_var.name + '_data'),
                            memory_handling='alias')
        shape_var = Variable(CStackArray(NumpyInt64Type()), self.scope.get_new_name(orig_var.name + '_shape'),
                            shape = (orig_var.rank,))
        stride_var = Variable(CStackArray(NumpyInt64Type()), self.scope.get_new_name(orig_var.name + '_strides'),
                            shape = (orig_var.rank,))
        self.scope.insert_variable(data_var)
        self.scope.insert_variable(shape_var)
        self.scope.insert_variable(stride_var)

        get_data = AliasAssign(data_var, PyArray_DATA(ObjectAddress(pyarray_collect_arg)))
        get_strides_and_shape = get_strides_and_shape_from_numpy_array(
                                        ObjectAddress(collect_arg), shape_var, stride_var,
                                        convert_to_literal(orig_var.order != 'F'))

        body = [get_data, get_strides_and_shape]

        return {'body': body, 'data':data_var, 'shape':shape_var, 'strides':stride_var}

    def _call_wrapped_function(self, func, args, results):
        """
        Call the wrapped function.

        Call the wrapped function. The call is either a FunctionCall, an Assign or
        an AliasAssign depending on the number of results and the return type.

        Parameters
        ----------
        func : FunctionDef
            The function being wrapped.
        args : iterable[TypedAstNode]
            The arguments passed to the wrapped function.
        results : iterable[TypedAstNode]
            The results returned from the wrapped function.

        Returns
        -------
        FunctionCall | Assign | AliasAssign
            An AST node describing the function call.
        """
        n_results = len(results)
        if n_results == 0:
            return func(*args)
        elif isinstance(results, PythonTuple):
            return Assign(results, func(*args))
        elif n_results == 1:
            res = results[0]
            func_call = func(*args)
            if func_call.is_alias:
                if isinstance(res, PointerCast):
                    res = res.obj
                if isinstance(res, ObjectAddress):
                    res = res.obj
                return AliasAssign(res, func_call)
            else:
                return Assign(res, func_call)
        else:
            return Assign(results, func(*args))

    def connect_pointer_targets(self, orig_var, python_res, funcdef, is_bind_c):
        """
        Get the code to connect pointers to their targets.

        Get the code to connect pointers to their targets. The connection is done via reference
        counting to ensure that the target is not cleaned by the garbage collector before the
        pointer.

        Parameters
        ----------
        orig_var : Variable
            The result of the function being wrapped.
        python_res : Variable
            The Python accessible result of the function being wrapped.
        funcdef : FunctionDef
            The function being wrapped.
        is_bind_c : bool
            True if the code is translated from a C-compatible language. False if the
            translated code is in C.

        Returns
        -------
        list
            Any nodes which must be printed to increase reference counts.
        """
        python_args = funcdef.arguments
        arg_targets = funcdef.result_pointer_map.get(orig_var, ())
        n_targets = len(arg_targets)
        if n_targets == 1:
            collect_arg = self._python_object_map[python_args[arg_targets[0]]]
            return self._incref_return_pointer(collect_arg, python_res, orig_var)
        elif n_targets > 1:
            if isinstance(orig_var.class_type, NumpyNDArrayType):
                raise errors.report((f"Can't determine the pointer target for the return object {orig_var}. "
                            "Please avoid calling this function to prevent accidental creation of dangling pointers."),
                        symbol = getattr(funcdef, 'original_function', funcdef), severity='warning')
            else:
                body = []
                for t in arg_targets:
                    collect_arg = self._python_object_map[python_args[t]]
                    body.extend(self._incref_return_pointer(collect_arg, python_res, orig_var))
                return body
        return []

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
        original_mod = getattr(expr, 'original_module', expr)
        original_mod_name = original_mod.scope.get_python_name(original_mod.name)

        mod_scope = Scope(name = original_mod_name, used_symbols = scope.local_used_symbols.copy(),
                          original_symbols = scope.python_names.copy(), scope_type = 'module')
        self.scope = mod_scope
        init_mod_func_name = self.scope.insert_symbol(f'PyInit_{original_mod_name}', 'wrapper')

        imports = [self._wrap(i) for i in getattr(expr, 'original_module', expr).imports]
        imports = [i for i in imports if i]

        # Ensure all class types are declared
        for c in expr.classes:
            name = c.name
            python_name = c.scope.get_python_name(name)
            struct_name = self.scope.get_new_name(f'Py{python_name}Object')
            dtype = DataTypeFactory(struct_name, self.scope.get_python_name(struct_name),
                                    BaseClass=WrapperCustomDataType)()

            type_name = self.scope.get_new_name(f'Py{python_name}Type')
            wrapped_class = PyClassDef(c, struct_name, type_name, self.scope.new_child_scope(name, 'class'),
                                       docstring = c.docstring, class_type = dtype)

            orig_cls_dtype = c.scope.parent_scope.cls_constructs[python_name]
            self._python_object_map[c] = wrapped_class
            self._python_object_map[orig_cls_dtype] = dtype

            self.scope.insert_class(wrapped_class, python_name)

        # Wrap classes
        classes = [self._wrap(i) for i in expr.classes]

        # Wrap functions
        funcs_to_wrap = [f for f in expr.funcs if f not in (expr.init_func, expr.free_func)]
        funcs_to_wrap = [f for f in funcs_to_wrap if not f.is_inline]

        # Add any functions removed by the Fortran printer
        removed_functions = getattr(expr, 'removed_functions', None)
        if removed_functions:
            funcs_to_wrap.extend(removed_functions)

        funcs = [self._wrap(f) for f in funcs_to_wrap]

        # Wrap interfaces
        interfaces = [self._wrap(i) for i in expr.interfaces if not i.is_inline]

        module_def_name = self.scope.get_new_name(f'module')
        init_func = self._build_module_init_function(expr, imports, module_def_name)

        API_var, import_func = self._build_module_import_function(expr)

        self.exit_scope()

        if not isinstance(expr, BindCModule):
            imports.append(Import(mod_scope.get_python_name(expr.name), expr))
        original_mod_name = mod_scope.get_python_name(original_mod.name)
        return PyModule(original_mod_name, [API_var], funcs, imports = imports,
                        interfaces = interfaces, classes = classes, scope = mod_scope,
                        init_func = init_func, import_func = import_func,
                        module_def_name = module_def_name)

    def _wrap_BindCModule(self, expr):
        """
        Build a `PyModule` from a `BindCModule`.

        Create a `PyModule` which wraps a C-compatible `BindCModule`. This function calls the
        more general `_wrap_Module` however additional steps are required to ensure that the
        Fortran functions and variables are declared in C.

        Parameters
        ----------
        expr : Module
            The module which can be called from C.

        Returns
        -------
        PyModule
            The module which can be called from Python.
        """
        pymod = self._wrap_Module(expr)

        # Add declarations for C-compatible variables
        decs = [Declare(v.clone(v.name.lower()), module_variable=True, external = True) \
                                    for v in expr.variables if not v.is_private and isinstance(v, BindCModuleVariable)]
        pymod.declarations = decs

        external_funcs = []
        # Add external functions for functions wrapping array variables
        for v in expr.variable_wrappers:
            f = v.wrapper_function
            external_funcs.append(FunctionDef(f.name, f.arguments, [], f.results, is_header = True, scope = f.scope))

        # Add external functions for normal functions
        external_funcs.extend(FunctionDef(f.name.lower(), f.arguments, [], f.results, is_header = True, scope = f.scope)
                              for f in expr.funcs)
        external_funcs.extend(FunctionDef(f.name.lower(), f.arguments, [], f.results, is_header = True, scope = f.scope)
                              for i in expr.interfaces for f in i.functions)

        for c in expr.classes:
            m = c.new_func
            external_funcs.append(FunctionDef(m.name, m.arguments, [], m.results, is_header = True, scope = m.scope))
            for m in c.methods:
                external_funcs.append(FunctionDef(m.name, m.arguments, [], m.results, is_header = True, scope = m.scope))
            for i in c.interfaces:
                for f in i.functions:
                    external_funcs.append(FunctionDef(f.name, f.arguments, [], f.results, is_header = True, scope = f.scope))
            for a in c.attributes:
                for f in (a.getter, a.setter):
                    if f:
                        external_funcs.append(FunctionDef(f.name, f.arguments, [], f.results, is_header = True, scope = f.scope))
        pymod.external_funcs = external_funcs

        return pymod

    def _wrap_Interface(self, expr):
        """
        Build a `PyInterface` from an `Interface`.

        Create a `PyInterface` which wraps a C-compatible `Interface`. The `PyInterface`
        should take three arguments (`self`, `args`, and `kwargs`) and return a
        `PyccelPyObject`. The arguments are unpacked into multiple `PyccelPyObject`s
        which are passed to `PyFunctionDef`s describing each of the internal
        `FunctionDef` objects. The appropriate `PyFunctionDef` is chosen using an
        additional function which calculates an integer type_indicator.

        Parameters
        ----------
        expr : Interface
            The interface which can be called from C.

        Returns
        -------
        PyInterface
            The interface which can be called from Python.

        See Also
        --------
        CToPythonWrapper._get_type_check_function : The function which defines the calculation
            of the type_indicator.
        """
        # Initialise the scope
        func_name = self.scope.get_new_name(expr.name+'_wrapper', object_type = 'wrapper')
        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope
        original_funcs = expr.functions
        example_func = original_funcs[0]
        possible_class_base = expr.get_user_nodes((ClassDef,))
        if possible_class_base:
            class_dtype = possible_class_base[0].class_type
        else:
            class_dtype = None

        for f in original_funcs:
            self._wrap(f)

        # Add the variables to the expected symbols in the scope
        for a in example_func.arguments:
            func_scope.insert_symbol(a.var.name)

        # Create necessary arguments
        python_args = example_func.arguments
        func_args, body = self._unpack_python_args(python_args, class_dtype)

        # Get python arguments which will be passed to FunctionDefs
        python_arg_objs = [self._python_object_map[a] for a in python_args]

        type_indicator = Variable(PythonNativeInt(), self.scope.get_new_name('type_indicator'))
        self.scope.insert_variable(type_indicator)

        self.exit_scope()

        # Determine flags which indicate argument type
        type_check_name = self.scope.get_new_name(expr.name+'_type_check', object_type = 'wrapper')
        type_check_func, argument_type_flags = self._get_type_check_function(type_check_name, python_arg_objs, original_funcs)

        self.scope = func_scope
        # Build the body of the function
        body.append(Assign(type_indicator, type_check_func(*python_arg_objs)))

        functions = []
        if_sections = []
        for func, index in argument_type_flags.items():
            # Add an IfSection calling the appropriate function if the type_indicator matches the index
            wrapped_func = self._python_object_map[func]
            if_sections.append(IfSection(PyccelEq(type_indicator, LiteralInteger(index)),
                                [Return(wrapped_func(*python_arg_objs))]))
            functions.append(wrapped_func)
        if_sections.append(IfSection(PyccelEq(type_indicator, PyccelUnarySub(LiteralInteger(1))),
                    [Return(self._error_exit_code)]))
        if_sections.append(IfSection(LiteralTrue(),
                    [PyErr_SetString(PyTypeError, CStrStr(LiteralString("Unexpected type combination"))),
                     Return(self._error_exit_code)]))
        body.append(If(*if_sections))
        result_var = self.get_new_PyObject("result", is_temp=True)
        self.exit_scope()

        interface_func = FunctionDef(func_name,
                                     [FunctionDefArgument(a) for a in func_args],
                                     body,
                                     FunctionDefResult(result_var),
                                     scope=func_scope)
        for a in python_args:
            self._python_object_map.pop(a)

        return PyInterface(func_name, functions, interface_func, type_check_func, expr)

    def _wrap_FunctionDef(self, expr):
        """
        Build a `PyFunctionDef` from a `FunctionDef`.

        Create a `PyFunctionDef` which wraps a C-compatible `FunctionDef`.
        The `PyFunctionDef` should take three arguments (`self`, `args`,
        and `kwargs`) and return a `PyccelPyObject`. If the function is
        called from an Interface then the arguments are `PyccelPyObject`s
        describing each of the arguments of the C-compatible function.

        Parameters
        ----------
        expr : FunctionDef
            The function which can be called from C.

        Returns
        -------
        PyFunctionDef
            The function which can be called from Python.
        """
        original_func = getattr(expr, 'original_function', expr)
        func_name = self.scope.get_new_name(expr.name+'_wrapper', object_type = 'wrapper')
        func_scope = self.scope.new_child_scope(func_name, 'function')
        self.scope = func_scope
        original_func_name = original_func.scope.get_python_name(original_func.name)

        possible_class_base = expr.get_user_nodes((ClassDef,))
        if possible_class_base:
            class_dtype = possible_class_base[0].class_type
        else:
            class_dtype = None

        is_bind_c_function_def = isinstance(expr, BindCFunctionDef)

        if expr.is_private:
            self.exit_scope()
            return self._get_untranslatable_function(func_name,
                         func_scope, expr,
                         "Private functions are not accessible from python")

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

        in_interface = len(expr.get_user_nodes(Interface, excluded_nodes = (FunctionCall,))) > 0

        # Get variables describing the arguments and results that are seen from Python
        python_args = expr.arguments
        python_results = expr.results

        # Get the arguments of the PyFunctionDef
        if 'property' in original_func.decorators:
            func_args = [self.get_new_PyObject('self_obj', dtype = class_dtype),
                         func_scope.get_temporary_variable(VoidType(), memory_handling='alias')]
            self._python_object_map[python_args[0]] = func_args[0]
            func_args = [FunctionDefArgument(a) for a in func_args]
            body = []
        else:
            if in_interface or original_func_name in magic_binary_funcs or original_func_name == '__len__':
                func_args = [FunctionDefArgument(a) for a in self._get_python_argument_variables(python_args)]
                body = []
            else:
                func_args, body = self._unpack_python_args(python_args, class_dtype)
                func_args = [FunctionDefArgument(a) for a in func_args]

        # Get the code required to extract the C-compatible arguments from the Python arguments
        wrapped_args = [self._wrap(a) for a in python_args]
        body += [l for a in wrapped_args for l in a['body']]

        # Get the code required to wrap the C-compatible results into Python objects
        # This function creates variables so it must be called before extracting them from the scope.
        if original_func_name in magic_binary_funcs and original_func_name.startswith('__i'):
            res = func_args[0].var.clone(self.scope.get_new_name(func_args[0].var.name), is_argument=False)
            wrapped_results = {'c_results': [], 'py_result': res, 'body': []}
            body.append(AliasAssign(res, func_args[0].var))
            body.append(Py_INCREF(res))
        else:
            wrapped_results = self._extract_FunctionDefResult(python_results.var, is_bind_c_function_def, expr)

        # Get the arguments and results which should be used to call the c-compatible function
        func_call_args = [ca for a in wrapped_args for ca in a['args']]

        # Get the names of the results collected from the C-compatible function
        body.extend(l for l in wrapped_results.get('setup',()))
        c_results =  wrapped_results['c_results']
        python_result_variable = wrapped_results['py_result']

        if class_dtype:
            body.extend(self._save_referenced_objects(expr, func_args))

        # Call the C-compatible function
        body.append(self._call_wrapped_function(expr, func_call_args, c_results))

        # Deallocate the C equivalent of any array arguments
        # The C equivalent is the same variable that is passed to the function unless the target language is Fortran.
        # In this case known-size stack arrays are used which are automatically deallocated when they go out of scope.
        for a in python_args:
            orig_var = a.var
            if orig_var.is_ndarray:
                v = self.scope.find(orig_var.name, category='variables', raise_if_missing = True)
                if v.is_optional:
                    body.append(If( IfSection(PyccelIsNot(v, Nil()), [Deallocate(v)]) ))
                else:
                    body.append(Deallocate(v))

        if original_func_name == '__len__':
            self.scope.remove_variable(python_result_variable)
            python_result_variable = c_results[0]
        else:
            body.extend(wrapped_results['body'])
        body.extend(ai for arg in wrapped_args for ai in arg['clean_up'])

        # Pack the Python compatible results of the function into one argument.
        if python_result_variable is Py_None:
            res = Py_None
            func_results = FunctionDefResult(self.get_new_PyObject("result", is_temp=True))
            body.append(Py_INCREF(res))
        elif original_func_name == '__len__':
            res = Py_ssize_t_Cast(python_result_variable)
            func_results = FunctionDefResult(Variable(Py_ssize_t(), self.scope.get_new_name(), is_temp = True))
        else:
            res = python_result_variable
            func_results = FunctionDefResult(res)
        body.append(Return(res))

        self.exit_scope()
        for a in python_args:
            if not a.bound_argument:
                self._python_object_map.pop(a)

        function = PyFunctionDef(func_name, func_args, body, func_results, scope=func_scope,
                docstring = expr.docstring, original_function = original_func)

        self.scope.insert_function(function, func_scope.get_python_name(func_name))
        self._python_object_map[expr] = function

        if 'property' in original_func.decorators:
            python_name = original_func.scope.get_python_name(original_func.name)
            docstring = LiteralString(
                            '\n'.join(original_func.docstring.comments)
                            if original_func.docstring else f"The attribute {python_name}")
            return PyGetSetDefElement(python_name, function, None, CStrStr(docstring))
        else:
            return function

    def _wrap_FunctionDefArgument(self, expr):
        """
        Get the code which translates a Python `FunctionDefArgument` to a C-compatible `Variable`.

        Get the code necessary to transform a Variable passed as an argument in Python, from an object with
        datatype `PyccelPyObject` to a Variable that can be used in C code.

        The relevant `PyccelPyObject` is collected from `self._python_object_map`.

        The necessary steps are:
        - Create a variable to store the C-compatible result.
        - Initialise the variable to any provided default value.
        - Cast the Python object to the C object using utility functions.
        - Raise any useful errors (this is not necessary if the FunctionDef is in an interface as errors are
            raised while determining which function to call).

        Parameters
        ----------
        expr : FunctionDefArgument
            The argument of the C function.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates the `PyccelPyObject`
                        to a C-compatible variable.
             - args : a list of Variables which should be passed to call the function being wrapped.
        """

        collect_arg = self._python_object_map[expr]
        in_interface = len(expr.get_user_nodes(Interface, excluded_nodes = (FunctionCall,))) > 0
        is_bind_c_argument = isinstance(expr.var, BindCVariable)

        orig_var = getattr(expr.var, 'original_var', expr.var)
        bound_argument = expr.bound_argument

        # Collect the function which casts from a Python object to a C object
        arg_extraction = self._extract_FunctionDefArgument(orig_var, collect_arg, bound_argument, is_bind_c_argument)

        body = []
        cast = arg_extraction['body']
        arg_vars = arg_extraction['args']

        # Initialise to any default value
        if expr.has_default:
            if 'default_init' in arg_extraction:
                for i, l in enumerate(arg_extraction['default_init']):
                    body.insert(i, l)
            else:
                assert len(arg_vars) == 1
                arg_var = arg_vars[0]
                default_val = expr.value
                if isinstance(default_val, Nil):
                    body.insert(0, AliasAssign(arg_var, default_val))
                else:
                    body.insert(0, Assign(arg_var, default_val))

        # Create any necessary type checks and errors
        if expr.has_default:
            check_func, err = self._get_type_check_condition(collect_arg, orig_var, True, body,
                    allow_empty_arrays = is_bind_c_argument)
            body.append(If( IfSection(PyccelIsNot(collect_arg, Py_None), [
                                If(IfSection(check_func, cast), IfSection(LiteralTrue(), [*err, Return(self._error_exit_code)]))])))
        elif not (in_interface or bound_argument):
            check_func, err = self._get_type_check_condition(collect_arg, orig_var, True, body,
                    allow_empty_arrays = is_bind_c_argument)
            body.append(If( IfSection(PyccelNot(check_func), [*err, Return(self._error_exit_code)])))
            body.extend(cast)
        else:
            body.extend(cast)

        return {'body': body, 'args': arg_vars, 'clean_up': arg_extraction.get('clean_up', ())}

    def _wrap_Variable(self, expr):
        """
        Get the code which translates a C-compatible module variable to an object with datatype `PyccelPyObject`.

        Get the code which translates a C-compatible module variable to an object with datatype `PyccelPyObject`.
        This new object is saved into self._python_object_map. The translation is achieved using utility
        functions.

        Parameters
        ----------
        expr : Variable
            The module variable.

        Returns
        -------
        list of pyccel.ast.basic.PyccelAstNode
            The code which translates the Variable to a Python-compatible variable.
        """

        # Create the resulting Variable with datatype `PyccelPyObject`
        py_equiv = self.scope.get_temporary_variable(PyccelPyObject(), memory_handling='alias')
        # Save the Variable so it can be located later
        self._python_object_map[expr] = py_equiv

        if isinstance(expr.class_type, NumpyNDArrayType):
            # Cast the C variable into a Python variable
            typenum = numpy_dtype_registry[expr.dtype]
            data_var = DottedVariable(VoidType(), 'data', memory_handling='alias',
                        lhs=expr)
            shape_var = DottedVariable(CStackArray(NumpyInt32Type()), 'shape',
                        lhs=expr)
            release_memory = False
            return [AliasAssign(py_equiv, to_pyarray(
                             LiteralInteger(expr.rank), typenum, data_var, shape_var,
                             convert_to_literal(expr.order != 'F'),
                             convert_to_literal(release_memory)))]
        else:
            wrapper_function = C_to_Python(expr)
            return [AliasAssign(py_equiv, wrapper_function(expr))]

    def _wrap_BindCArrayVariable(self, expr):
        """
        Get the code which translates a Fortran array module variable to an object with datatype `PyccelPyObject`.

        Get the code which translates a Fortran array module variable to an object with datatype `PyccelPyObject`
        which can be used as a Python module variable. This new object is saved into self._python_object_map.
        Fortran arrays are not compatible with C, but objects of type `BindCArrayVariable` contain wrapper
        functions which can be used to retrieve C-compatible variables.

        The necessary steps are:
        - Create the variables necessary to retrieve array objects from Fortran.
        - Call the bind c wrapper function to initialise these objects.
        - Pack the results into a C-compatible `ndarray`.
        - Use `self._wrap_Variable` to get the object with datatype `PyccelPyObject`.
        - Correct the key in self._python_object_map initialised by `self._wrap_Variable`.

        Parameters
        ----------
        expr : BindCArrayVariable
            The array module variable.

        Returns
        -------
        list of pyccel.ast.basic.PyccelAstNode
            The code which translates the Variable to a Python-compatible variable.
        """
        v = expr.original_variable

        typenum = numpy_dtype_registry[v.dtype]
        # Get pointer to store raw array data
        data_var = self.scope.get_temporary_variable(dtype_or_var = VoidType(),
                name = v.name + '_data', memory_handling = 'alias')
        # Create variables to store the shape of the array
        shape_var = self.scope.get_temporary_variable(CStackArray(NumpyInt32Type()), name = v.name+'_size',
                shape = (v.rank,))
        shape = [IndexedElement(shape_var, i) for i in range(v.rank)]
        # Get the bind_c function which wraps a fortran array and returns c objects
        var_wrapper = expr.wrapper_function
        # Call bind_c function
        call = Assign(PythonTuple(ObjectAddress(data_var), *shape), var_wrapper())

        # Create the resulting Variable with datatype `PyccelPyObject`
        py_equiv = self.scope.get_temporary_variable(PyccelPyObject(), memory_handling='alias')
        self._python_object_map[expr] = py_equiv

        release_memory = False
        # Save the ndarray to vars_to_wrap to be handled as if it came from C
        return [call, AliasAssign(py_equiv, to_pyarray(LiteralInteger(v.rank), typenum,
                            data_var, shape_var, convert_to_literal(v.order != 'F'),
                            convert_to_literal(release_memory)))]

    def _wrap_DottedVariable(self, expr):
        """
        Create all objects necessary to expose a class attribute to C.

        Create the getter and setter functions which expose the class attribute
        to C. Return these objects in a PyGetSetDefElement.
        See <https://docs.python.org/3/extending/newtypes_tutorial.html#providing-finer-control-over-data-attributes>
        for more information about the necessary prototypes.

        Parameters
        ----------
        expr : DottedVariable
            The class attribute.

        Returns
        -------
        PyGetSetDefElement
            An object which contains the new getter and setter functions that should be
            described in the array of PyGetSetDef objects.
        """
        lhs = expr.lhs
        class_type = lhs.cls_base
        python_class_type = self.scope.find(self.scope.get_python_name(class_type.name),
                                            'classes', raise_if_missing = True)
        class_scope = python_class_type.scope

        class_ptr_attrib = class_scope.find('instance', 'variables', raise_if_missing = True)

        # ----------------------------------------------------------------------------------
        #                        Create getter
        # ----------------------------------------------------------------------------------
        getter_name = self.scope.get_new_name(f'{class_type.name}_{expr.name}_getter', object_type = 'wrapper')
        getter_scope = self.scope.new_child_scope(getter_name, 'function')
        self.scope = getter_scope
        getter_args = [self.get_new_PyObject('self_obj', dtype = lhs.dtype),
                       getter_scope.get_temporary_variable(VoidType(), memory_handling='alias')]
        self.scope.insert_symbol(expr.name)

        class_obj = Variable(lhs.dtype, self.scope.get_new_name('self'), memory_handling='alias')
        self.scope.insert_variable(class_obj, 'self')

        attrib = expr.clone(expr.name, lhs = class_obj)
        # Cast the C variable into a Python variable
        result_wrapping = self._extract_FunctionDefResult(expr.clone(expr.name, new_class = Variable), False)
        res_wrapper = result_wrapping['body']
        new_res_val = result_wrapping['c_results'][0]
        getter_result = result_wrapping['py_result']
        setup = result_wrapping.get('setup', ())
        if new_res_val.rank > 0:
            body = [AliasAssign(new_res_val, attrib), *res_wrapper]
        elif isinstance(expr.dtype, CustomDataType):
            if isinstance(new_res_val, PointerCast):
                new_res_val = new_res_val.obj
            body = [Allocate(getter_result, shape=None, status='unallocated'),
                    AliasAssign(new_res_val, attrib),
                    *res_wrapper]
        else:
            body = [Assign(new_res_val, attrib), *res_wrapper]

        body.extend(self._incref_return_pointer(getter_args[0], getter_result, expr))

        getter_body = [*setup, AliasAssign(class_obj, PointerCast(class_ptr_attrib.clone(class_ptr_attrib.name,
                                                                                 new_class = DottedVariable,
                                                                                lhs = getter_args[0]),
                                                          cast_type = lhs)),
                       *body,
                       Return(getter_result)]
        self.exit_scope()

        args = [FunctionDefArgument(a) for a in getter_args]
        getter = PyFunctionDef(getter_name, args, getter_body, FunctionDefResult(getter_result),
                                original_function = expr, scope = getter_scope)

        # ----------------------------------------------------------------------------------
        #                        Create setter
        # ----------------------------------------------------------------------------------
        self._error_exit_code = PyccelUnarySub(LiteralInteger(1, dtype=CNativeInt()))
        setter_name = self.scope.get_new_name(f'{class_type.name}_{expr.name}_setter', object_type = 'wrapper')
        setter_scope = self.scope.new_child_scope(setter_name, 'function')
        self.scope = setter_scope
        setter_args = [self.get_new_PyObject('self_obj', dtype = lhs.dtype),
                       self.get_new_PyObject(f'{expr.name}_obj'),
                       setter_scope.get_temporary_variable(VoidType(), memory_handling='alias')]
        setter_result = FunctionDefResult(setter_scope.get_temporary_variable(CNativeInt()))
        self.scope.insert_symbol(expr.name)
        new_set_val_arg = FunctionDefArgument(expr.clone(expr.name, new_class = Variable))
        self._python_object_map[new_set_val_arg] = setter_args[1]

        if isinstance(expr.class_type, FixedSizeNumericType) or expr.is_alias:
            class_obj = Variable(lhs.dtype, self.scope.get_new_name('self'), memory_handling='alias')
            self.scope.insert_variable(class_obj, 'self')

            attrib = expr.clone(expr.name, lhs = class_obj)
            wrap_arg = self._wrap(new_set_val_arg)
            arg_wrapper = wrap_arg['body']
            new_set_val = wrap_arg['args'][0]

            if expr.memory_handling == 'alias':
                update = AliasAssign(attrib, new_set_val)
            else:
                update = Assign(attrib, new_set_val)

            # Cast the C variable into a Python variable
            setter_body = [*arg_wrapper,
                           AliasAssign(class_obj, PointerCast(class_ptr_attrib.clone(class_ptr_attrib.name,
                                                                                     new_class = DottedVariable,
                                                                                    lhs = setter_args[0]),
                                                              cast_type = lhs)),
                           *self._incref_return_pointer(setter_args[1], setter_args[0], expr.lhs),
                           update,
                           Return(LiteralInteger(0, dtype=CNativeInt()))]
        else:
            setter_body = [PyErr_SetString(PyAttributeError,
                                        CStrStr(LiteralString("Can't reallocate memory via Python interface."))),
                        Return(self._error_exit_code)]
        self.exit_scope()

        args = [FunctionDefArgument(a) for a in setter_args]
        setter = PyFunctionDef(setter_name, args, setter_body, setter_result,
                                original_function = expr, scope = setter_scope)
        self._error_exit_code = Nil()
        self._python_object_map.pop(new_set_val_arg)
        # ----------------------------------------------------------------------------------

        python_name = class_type.scope.get_python_name(expr.name)
        return PyGetSetDefElement(python_name, getter, setter,
                                CStrStr(LiteralString(f"The attribute {python_name}")))

    def _wrap_BindCClassProperty(self, expr):
        """
        Create a PyGetSetDefElement to expose a class attribute/property to Python.

        Create getter and setter functions which are compatible with the expected prototype for
        `PyGetSetDef` and which call the getter and setter functions contained in the
        BindCClassProperty. The result is returned in a PyGetSetDefElement.
        See <https://docs.python.org/3/extending/newtypes_tutorial.html#providing-finer-control-over-data-attributes>
        for more information about the necessary prototypes.

        Parameters
        ----------
        expr : BindCClassProperty
            The object containing the getter and setter functions to be wrapped.

        Returns
        -------
        PyGetSetDefElement
            An object which contains the new getter and setter functions that should be
            described in the array of PyGetSetDef objects.
        """
        class_type = expr.class_type
        name = expr.python_name
        # ----------------------------------------------------------------------------------
        #                        Create getter
        # ----------------------------------------------------------------------------------
        getter_name = self.scope.get_new_name(f'{class_type.name}_{name}_getter', object_type = 'wrapper')
        getter_scope = self.scope.new_child_scope(getter_name, 'function')
        self.scope = getter_scope

        get_val_arg = expr.getter.arguments[0]
        self.scope.insert_symbol(get_val_arg.var.original_var.name)
        get_val_result = expr.getter.results

        getter_args = [self.get_new_PyObject('self_obj', dtype = class_type),
                       getter_scope.get_temporary_variable(VoidType(), memory_handling='alias')]

        self._python_object_map[get_val_arg] = getter_args[0]

        wrapped_args = self._wrap(get_val_arg)
        arg_code = wrapped_args['body']
        class_obj = wrapped_args['args'][0]

        # Cast the C variable into a Python variable
        get_val_result_var = getattr(get_val_result, 'original_function_result_variable', get_val_result.var)
        result_wrapping = self._extract_FunctionDefResult(get_val_result_var, True)
        res_wrapper = result_wrapping['body']
        c_results = result_wrapping['c_results']
        getter_result = result_wrapping['py_result']
        setup = result_wrapping.get('setup', ())

        call = self._call_wrapped_function(expr.getter, (class_obj,), c_results)

        if isinstance(getter_result.dtype, CustomDataType):
            arg_code.append(Allocate(getter_result, shape=None, status='unallocated'))

        if isinstance(expr.getter.original_function, DottedVariable):
            wrapped_var = expr.getter.original_function
        else:
            wrapped_var = expr.getter.original_function.results.var
        res_wrapper.extend(self._incref_return_pointer(getter_args[0], getter_result, wrapped_var))

        getter_body = [*setup,
                       *arg_code,
                       call,
                       *res_wrapper,
                       Return(getter_result)]
        self.exit_scope()

        args = [FunctionDefArgument(a) for a in getter_args]
        getter = PyFunctionDef(getter_name, args, getter_body, FunctionDefResult(getter_result),
                                original_function = expr.getter, scope = getter_scope)

        # ----------------------------------------------------------------------------------
        #                        Create setter
        # ----------------------------------------------------------------------------------
        if expr.setter:
            self._error_exit_code = PyccelUnarySub(LiteralInteger(1, dtype=CNativeInt()))
            setter_name = self.scope.get_new_name(f'{class_type.name}_{name}_setter', object_type = 'wrapper')
            setter_scope = self.scope.new_child_scope(setter_name, 'function')
            self.scope = setter_scope

            original_args = expr.setter.arguments
            f_wrapped_args = expr.setter.arguments

            self_arg = original_args[0]
            set_val_arg = original_args[1]
            for a in f_wrapped_args:
                self.scope.insert_symbol(a.var.name)
            self.scope.insert_symbol(self_arg.var.original_var.name)
            self.scope.insert_symbol(set_val_arg.var.original_var.name)

            setter_args = [self.get_new_PyObject('self_obj', dtype = class_type),
                           self.get_new_PyObject(f'{name}_obj'),
                           setter_scope.get_temporary_variable(VoidType(), memory_handling='alias')]
            setter_result = FunctionDefResult(setter_scope.get_temporary_variable(CNativeInt()))

            self._python_object_map[self_arg] = setter_args[0]
            self._python_object_map[set_val_arg] = setter_args[1]

            if isinstance(wrapped_var.class_type, FixedSizeNumericType) or wrapped_var.is_alias:
                wrapped_args = [self._wrap(a) for a in original_args]
                arg_code = [l for a in wrapped_args for l in a['body']]
                func_call_args = [ca for a in wrapped_args for ca in a['args']]

                setter_body = [*arg_code,
                               expr.setter(*func_call_args),
                               *self._save_referenced_objects(expr.setter, setter_args),
                               Return(LiteralInteger(0, dtype=CNativeInt()))]
            else:
                setter_body = [PyErr_SetString(PyAttributeError,
                                            CStrStr(LiteralString("Can't reallocate memory via Python interface."))),
                            Return(self._error_exit_code)]
            self.exit_scope()

            args = [FunctionDefArgument(a) for a in setter_args]
            setter = PyFunctionDef(setter_name, args, setter_body, setter_result,
                                    original_function = expr, scope = setter_scope)
        else:
            setter = None

        self._error_exit_code = Nil()

        docstring = LiteralString(
                        '\n'.join(expr.docstring.comments)
                        if expr.docstring else f"The attribute {expr.python_name}")
        return PyGetSetDefElement(expr.python_name, getter, setter, CStrStr(docstring))

    def _wrap_ClassDef(self, expr):
        """
        Get the code which exposes a class definition to Python.

        Get the code which exposes a class definition to Python.

        Parameters
        ----------
        expr : ClassDef
            The class definition being wrapped.

        Returns
        -------
        PyClassDef
            The wrapped class definition.
        """
        name = expr.name
        python_name = expr.scope.get_python_name(name)

        bound_class = isinstance(expr, BindCClassDef)

        orig_cls_dtype = expr.scope.parent_scope.cls_constructs[python_name]
        wrapped_class = self._python_object_map[expr]

        orig_scope = expr.scope

        for f in expr.methods:
            if f.is_inline:
                continue
            orig_f = getattr(f, 'original_function', f)
            name = orig_f.name
            python_name = orig_scope.get_python_name(name)
            if python_name == '__del__':
                wrapped_class.add_new_method(self._get_class_destructor(f, orig_cls_dtype, wrapped_class.scope))
            elif python_name == '__init__':
                wrapped_class.add_new_method(self._get_class_initialiser(f, orig_cls_dtype))
            elif python_name in (*magic_binary_funcs, '__len__'):
                wrapped_class.add_new_magic_method(self._wrap(f))
            elif 'property' in f.decorators:
                wrapped_class.add_property(self._wrap(f))
            else:
                wrapped_class.add_new_method(self._wrap(f))

        for i in expr.interfaces:
            for f in i.functions:
                self._wrap(f)
            wrapped_class.add_new_interface(self._wrap(i))

        if bound_class:
            wrapped_class.add_alloc_method(self._get_class_allocator(orig_cls_dtype, expr.new_func))
        else:
            wrapped_class.add_alloc_method(self._get_class_allocator(orig_cls_dtype))

        # Pseudo-self variable is useful for pre-defined attributes which are not DottedVariables
        pseudo_self = Variable(expr.class_type, 'self', cls_base = expr)
        for a in expr.attributes:
            if isinstance(a.class_type, TupleType):
                errors.report("Tuples cannot yet be exposed to Python.",
                        severity='warning',
                        symbol=a)
                continue

            if bound_class or not a.is_private:
                if isinstance(a, (DottedVariable, BindCClassProperty)):
                    wrapped_class.add_property(self._wrap(a))
                else:
                    wrapped_class.add_property(self._wrap(a.clone(a.name, new_class = DottedVariable,
                                                            lhs=pseudo_self)))

        return wrapped_class

    def _wrap_Import(self, expr):
        """
        Examine an Import statement and collect any relevant objects.

        Examine an Import statement used in the module being wrapped. If it imports a class
        from a module then a PyClassDef is added to the scope imports to ensure that its
        description is available for functions wishing to use this type for an argument
        or return value.

        Parameters
        ----------
        expr : Import
            The import found in the module being wrapped.

        Returns
        -------
        Import | None
            The import needed in the wrapper, or None if none is necessary.
        """
        # Imports do not use collision handling as there is not enough context available.
        # This should be fixed when stub files and proper pickling is added
        import_wrapper = False
        import_scope = None
        for as_name in expr.target:
            t = as_name.object
            if isinstance(t, ClassDef):
                if import_scope is None:
                    import_scope = Scope(name = expr.source_module.name,
                                         used_symbols = expr.source_module.scope.local_used_symbols.copy(),
                                         original_symbols = expr.source_module.scope.python_names.copy(),
                                         scope_type = 'module')
                name = t.scope.get_python_name(t.name)
                struct_name = import_scope.get_new_name(f'Py{name}Object')
                dtype = DataTypeFactory(struct_name, struct_name, BaseClass=WrapperCustomDataType)()
                type_name = import_scope.get_new_name(f'Py{name}Type')
                wrapped_class = PyClassDef(t, struct_name, type_name, Scope(name = name, scope_type = 'class'), class_type = dtype)
                self._python_object_map[t] = wrapped_class
                self._python_object_map[t.class_type] = dtype
                self.scope.imports['classes'][name] = wrapped_class
                import_wrapper = True

        if import_wrapper:
            wrapper_name = f'{expr.source}_wrapper'
            mod_spoof_scope = Scope(name=expr.source_module.name, scope_type = 'module')
            mod_import_func = FunctionDef(mod_spoof_scope.get_new_name('import'), (), (),
                       FunctionDefResult(Variable(CNativeInt(), '_', is_temp=True)))
            mod_spoof = PyModule(expr.source_module.name, (), (), scope = mod_spoof_scope,
                                 module_def_name = mod_spoof_scope.get_new_name(f'module'),
                                 import_func = mod_import_func)
            return Import(wrapper_name, AsName(mod_spoof, expr.source), mod = mod_spoof)
        else:
            return None

    def _extract_FunctionDefArgument(self, orig_var, collect_arg, bound_argument,
            is_bind_c_argument, *, arg_var = None):
        """
        Extract the C-compatible FunctionDefArgument from the PythonObject.

        Extract the C-compatible FunctionDefArgument from the PythonObject.
        The C-compatible argument is extracted from collect_arg which holds a Python
        object into arg_var.

        The extraction is done by finding the appropriate function
        _extract_X_FunctionDefArgument for the object expr. X is the class type of the
        object expr. If this function does not exist then the method resolution order
        is used to search for other compatible _extract_X_FunctionDefArgument functions.
        If none are found then an error is raised.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefArgument being wrapped.

        collect_arg : Variable
            A variable with type PythonObject* holding the Python argument from which the
            C-compatible argument should be collected.

        bound_argument : bool
            True if the argument is the self argument of a class method. False otherwise.
            This should always be False for this function.

        is_bind_c_argument : bool
            True if the argument was defined in a BindCFunctionDef. False otherwise.

        arg_var : Variable | IndexedElement, optional
            A variable or an element of the variable representing the argument that
            will be passed to the low-level function call.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to access the argument.
        """
        class_type = orig_var.class_type

        classes = type(class_type).__mro__
        for cls in classes:
            annotation_method = f'_extract_{cls.__name__}_FunctionDefArgument'
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(orig_var, collect_arg, bound_argument,
                                                is_bind_c_argument, arg_var = arg_var)

        # Unknown object, we raise an error.
        return errors.report(f"Wrapping function arguments is not implemented for type {class_type}. "+PYCCEL_RESTRICTION_TODO, symbol=orig_var,
            severity='fatal')

    def _extract_FixedSizeType_FunctionDefArgument(self, orig_var, collect_arg, bound_argument,
            is_bind_c_argument, *, arg_var = None):
        """
        Extract the C-compatible scalar FunctionDefArgument from the PythonObject.

        Extract the C-compatible scalar FunctionDefArgument from the PythonObject.
        The C-compatible argument is extracted from collect_arg which holds a Python
        object into arg_var.

        The extraction is done by calling a function from the C-Python API. These functions
        are indexed in the dictionary `py_to_c_registry`.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefArgument being wrapped.

        collect_arg : Variable
            A variable with type PythonObject* holding the Python argument from which the
            C-compatible argument should be collected.

        bound_argument : bool
            True if the argument is the self argument of a class method. False otherwise.
            This should always be False for this function.

        is_bind_c_argument : bool
            True if the argument was defined in a BindCFunctionDef. False otherwise.

        arg_var : Variable | IndexedElement
            A variable or an element of the variable representing the argument that
            will be passed to the low-level function call.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to access the argument.
        """
        assert not bound_argument
        if arg_var is None:
            arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), new_class = Variable,
                                    is_argument = False, is_const = False)
            self.scope.insert_variable(arg_var, orig_var.name)

        dtype = orig_var.dtype
        try :
            cast_function = py_to_c_registry[(dtype.primitive_type, dtype.precision)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=dtype,severity='fatal')
        cast_func = FunctionDef(name = cast_function,
                           body      = [],
                           arguments = [FunctionDefArgument(Variable(PyccelPyObject(), name = 'o', memory_handling='alias'))],
                           results   = FunctionDefResult(Variable(dtype, name = 'v')))

        body = [Assign(arg_var, cast_func(collect_arg))]

        if getattr(orig_var, 'is_optional', False):
            memory_var = self.scope.get_temporary_variable(arg_var, name = arg_var.name + '_memory', is_optional = False)
            body.insert(0, AliasAssign(arg_var, memory_var))

        return {'body': body,
                'args': [arg_var]}

    def _extract_CustomDataType_FunctionDefArgument(self, orig_var, collect_arg, bound_argument,
            is_bind_c_argument, *, arg_var = None):
        """
        Extract the C-compatible class FunctionDefArgument from the PythonObject.

        Extract the C-compatible class FunctionDefArgument from the PythonObject.
        The C-compatible argument is extracted from collect_arg which holds a Python
        object into arg_var.

        The extraction is done by accessing the pointer from the `instance` attribute of the
        Pyccel generated class definition.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefArgument being wrapped.

        collect_arg : Variable
            A variable with type PythonObject* holding the Python argument from which the
            C-compatible argument should be collected.

        bound_argument : bool
            True if the argument is the self argument of a class method. False otherwise.
            This should always be False for this function.

        is_bind_c_argument : bool
            True if the argument was defined in a BindCFunctionDef. False otherwise.

        arg_var : Variable | IndexedElement, optional
            A variable or an element of the variable representing the argument that
            will be passed to the low-level function call.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to access the argument.
        """
        if arg_var is None:
            kwargs = {'is_argument': False}
            kwargs['memory_handling']='alias'
            if is_bind_c_argument:
                kwargs['class_type'] = VoidType()

            arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), new_class = Variable,
                                    **kwargs)
            self.scope.insert_variable(arg_var, orig_var.name)

        dtype = orig_var.dtype
        python_cls_base = self.scope.find(dtype.name, 'classes', raise_if_missing = True)
        scope = python_cls_base.scope
        attribute = scope.find('instance', 'variables', raise_if_missing = True)
        if bound_argument:
            cast_type = collect_arg
            cast = []
        else:
            cast_type = Variable(self._python_object_map[dtype],
                                self.scope.get_new_name(collect_arg.name),
                                memory_handling='alias',
                                cls_base = self.scope.find(dtype.name, 'classes', raise_if_missing = True))
            self.scope.insert_variable(cast_type)
            cast = [AliasAssign(cast_type, PointerCast(collect_arg, cast_type))]
        c_res = attribute.clone(attribute.name, new_class = DottedVariable, lhs = cast_type)
        cast_c_res = PointerCast(c_res, orig_var)
        cast.append(AliasAssign(arg_var, cast_c_res))
        return {'body': cast, 'args': [arg_var]}

    def _extract_NumpyNDArrayType_FunctionDefArgument(self, orig_var, collect_arg, bound_argument,
            is_bind_c_argument, *, arg_var = None):
        """
        Extract the C-compatible NumPy array FunctionDefArgument from the PythonObject.

        Extract the C-compatible NumPy array FunctionDefArgument from the PythonObject.
        The C-compatible argument is extracted from collect_arg which holds a Python
        object into arg_var.

        The extraction is done by calling the function `pyarray_to_ndarray` from the stdlib.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefArgument being wrapped.

        collect_arg : Variable
            A variable with type PythonObject* holding the Python argument from which the
            C-compatible argument should be collected.

        bound_argument : bool
            True if the argument is the self argument of a class method. False otherwise.
            This should always be False for this function.

        is_bind_c_argument : bool
            True if the argument was defined in a BindCFunctionDef. False otherwise.

        arg_var : Variable | IndexedElement, optional
            A variable or an element of the variable representing the argument that
            will be passed to the low-level function call.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to access the argument.
        """
        assert arg_var is None
        parts = self._get_array_parts(orig_var, collect_arg)
        body = parts['body']
        shape = parts['shape']
        strides = parts['strides']
        shape_elems = [IndexedElement(shape, i) for i in range(orig_var.rank)]
        stride_elems = [IndexedElement(strides, i) for i in range(orig_var.rank)]
        args = [parts['data']] + shape_elems + stride_elems
        default_body = [AliasAssign(parts['data'], Nil())] + \
                [Assign(s, 0) for s in shape_elems] + \
                [Assign(s, 1) for s in stride_elems]

        if is_bind_c_argument:
            rank = orig_var.rank
            arg_var = Variable(BindCArrayType(rank, True), self.scope.get_new_name(orig_var.name),
                        shape = (LiteralInteger(rank*2+1),))
            self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(0)), ObjectAddress(parts['data']))
            for i,s in enumerate(shape):
                self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(i+1)), s)
            for i,s in enumerate(strides):
                self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(i+rank+1)), s)

            return {'body': body, 'args': [arg_var], 'default_init': default_body}

        arg_var = orig_var.clone(self.scope.get_new_name(orig_var.name), is_argument = False, is_optional=False,
                                memory_handling='alias', new_class = Variable, allows_negative_indexes = False,
                                is_const = False)
        self.scope.insert_variable(arg_var)
        if orig_var.is_optional:
            sliced_arg_var = orig_var.clone(self.scope.get_new_name(orig_var.name), is_argument = False,
                                    is_optional=False, memory_handling='alias', new_class = Variable,
                                    allows_negative_indexes = False, is_const = False)
            self.scope.insert_variable(sliced_arg_var)
        else:
            sliced_arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False,
                                    is_optional=False, memory_handling='alias', new_class = Variable,
                                    allows_negative_indexes = False, is_const = False)
            self.scope.insert_variable(sliced_arg_var, orig_var.name)

        original_size = tuple(PyccelMul(sh, st) for sh, st in zip(shape_elems, stride_elems))

        body.append(Allocate(arg_var, shape=original_size, status='unallocated', like=args[0]))
        body.append(AliasAssign(sliced_arg_var, IndexedElement(arg_var, *[Slice(None, None, s) for s in stride_elems])))

        collect_arg = sliced_arg_var
        if orig_var.is_optional:
            optional_arg_var = sliced_arg_var.clone(self.scope.get_expected_name(orig_var.name), is_optional = True)
            self.scope.insert_variable(optional_arg_var)
            body.append(AliasAssign(optional_arg_var, sliced_arg_var))
            default_body.append(AliasAssign(optional_arg_var, Nil()))
            collect_arg = optional_arg_var
        return {'body': body, 'args': [collect_arg], 'default_init': default_body}

    def _extract_HomogeneousTupleType_FunctionDefArgument(self, orig_var, collect_arg, bound_argument,
            is_bind_c_argument, *, arg_var = None):
        """
        Extract the C-compatible homogeneous tuple FunctionDefArgument from the PythonObject.

        Extract the C-compatible homogeneous tuple FunctionDefArgument from the PythonObject.
        The C-compatible argument is extracted from collect_arg which holds a Python
        object into arg_var.

        The extraction is done by allocating an array and filling the elements with values
        extracted from the indexed Python tuple in collect_arg.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefArgument being wrapped.

        collect_arg : Variable
            A variable with type PythonObject* holding the Python argument from which the
            C-compatible argument should be collected.

        bound_argument : bool
            True if the argument is the self argument of a class method. False otherwise.
            This should always be False for this function.

        is_bind_c_argument : bool
            True if the argument was defined in a BindCFunctionDef. False otherwise.

        arg_var : Variable | IndexedElement, optional
            A variable or an element of the variable representing the argument that
            will be passed to the low-level function call.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to access the argument.
        """
        assert arg_var is None

        if orig_var.rank > 1:
            errors.report("Wrapping multi-level tuples is not yet supported",
                    severity='fatal', symbol=orig_var)

        if orig_var.is_optional:
            errors.report("Optional tuples are not yet supported",
                    severity='fatal', symbol=orig_var)

        size_var = self.scope.get_temporary_variable(PythonNativeInt(), self.scope.get_new_name(f'{orig_var.name}_size'))

        if is_bind_c_argument:
            data_var = Variable(CStackArray(orig_var.class_type.element_type), self.scope.get_new_name(orig_var.name + '_data'),
                                memory_handling='alias')
            self.scope.insert_variable(data_var)
            arg_var = Variable(BindCArrayType(1, False), self.scope.get_new_name(orig_var.name),
                        shape = (LiteralInteger(2),))
            self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(0)), ObjectAddress(data_var))
            self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(1)), size_var)
            fill_var = data_var
            like = Variable(orig_var.class_type.element_type, '_')
        else:
            arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False,
                                    memory_handling='heap', new_class = Variable)
            self.scope.insert_variable(arg_var, orig_var.name)
            fill_var = arg_var
            like = None

        arg_vars = [arg_var]

        assert not bound_argument
        idx = self.scope.get_temporary_variable(CNativeInt())
        indexed_orig_var = IndexedElement(orig_var, idx)
        indexed_arg_var = IndexedElement(fill_var, idx)
        indexed_collect_arg = self.scope.get_temporary_variable(PyccelPyObject(), memory_handling='alias')

        body = [Assign(size_var, PyTuple_Size(collect_arg)),
                Allocate(fill_var, shape = (size_var,), status = 'unallocated', like = like)]

        for_scope = self.scope.create_new_loop_scope()
        self.scope = for_scope
        for_body = [AliasAssign(indexed_collect_arg, PyTuple_GetItem(collect_arg, idx))]
        for_body += self._extract_FunctionDefArgument(indexed_orig_var, indexed_collect_arg,
                                    bound_argument, is_bind_c_argument, arg_var = indexed_arg_var)['body']
        self.exit_scope()

        body.append(For((idx,), PythonRange(size_var), for_body, scope = for_scope))


        return {'body': body, 'args': arg_vars}

    def _extract_HomogeneousSetType_FunctionDefArgument(self, orig_var, collect_arg, bound_argument,
            is_bind_c_argument, *, arg_var = None):
        assert arg_var is None

        if orig_var.is_optional:
            errors.report("Optionals are not yet supported",
                    severity='fatal', symbol=orig_var)

        assert not bound_argument

        size_var = self.scope.get_temporary_variable(PythonNativeInt(), self.scope.get_new_name(f'{orig_var.name}_size'))
        body = [Assign(size_var, PySet_Size(collect_arg))]

        if is_bind_c_argument:
            element_type = orig_var.class_type.element_type
            #raise errors.report("Fortran set interface is not yet implemented", severity='fatal', symbol=orig_var)
            arr_var = Variable(NumpyNDArrayType(element_type, 1, None), self.scope.get_expected_name(orig_var.name),
                                shape = (size_var,), memory_handling = 'heap')
            self.scope.insert_variable(arr_var, orig_var.name)
            arg_var = Variable(BindCArrayType(1, False), self.scope.get_new_name(orig_var.name),
                        shape = (LiteralInteger(2),))
            data = DottedVariable(VoidType(), 'data', lhs=arr_var)
            self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(0)), data)
            self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(1)), size_var)
            arg_vars = [arg_var]
            body.append(Allocate(arr_var, shape = (size_var,), status='unallocated'))
        else:
            arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False,
                                    memory_handling='heap', new_class = Variable, is_const = False)
            self.scope.insert_variable(arg_var, orig_var.name)
            arg_vars = [arg_var]
            body.append(Assign(arg_var, PythonSet()))

        idx = self.scope.get_temporary_variable(CNativeInt())
        indexed_orig_var = self.scope.get_temporary_variable(orig_var.class_type.element_type)
        indexed_collect_arg = self.scope.get_temporary_variable(PyccelPyObject(), memory_handling='alias')

        iter_obj = self.scope.get_temporary_variable(PyccelPyObject(), 'iter', memory_handling='alias')

        body.append(AliasAssign(iter_obj, PyObject_GetIter(collect_arg)))

        for_scope = self.scope.create_new_loop_scope()
        self.scope = for_scope
        for_body = [AliasAssign(indexed_collect_arg, PyIter_Next(iter_obj))]
        for_body += self._extract_FunctionDefArgument(indexed_orig_var, indexed_collect_arg,
                                    bound_argument, is_bind_c_argument, arg_var = indexed_orig_var)['body']
        if is_bind_c_argument:
            for_body.append(Assign(IndexedElement(arr_var, idx), indexed_orig_var))
        else:
            for_body.append(SetAdd(arg_var, indexed_orig_var))
        self.exit_scope()

        body.append(For((idx,), PythonRange(size_var), for_body, scope = for_scope))

        clean_up = []
        if not orig_var.is_const:
            if is_bind_c_argument:
                errors.report("Python built-in containers should be passed as constant arguments when "
                              "translating to languages other than C. Any changes to the set will not "
                              "be reflected in the calling code.",
                              severity='warning', symbol=orig_var)
            else:
                element_extraction = self._extract_FunctionDefResult(IndexedElement(orig_var, idx),
                                                is_bind_c_argument, None)
                elem_set = PySet_Add(collect_arg, element_extraction['py_result'])
                for_body = [*element_extraction['body'],
                        If(IfSection(PyccelEq(elem_set, PyccelUnarySub(LiteralInteger(1))),
                                                 [Return(self._error_exit_code)]))]

                loop_iterator = VariableIterator(arg_var)
                loop_iterator.set_loop_counter(idx)
                clean_up = [If(IfSection(PyccelEq(PySet_Clear(collect_arg), PyccelUnarySub(LiteralInteger(1))),
                                                 [Return(self._error_exit_code)])),
                        For((element_extraction['c_results'][0],), loop_iterator, for_body, for_scope)]

        return {'body': body, 'args': arg_vars, 'clean_up': clean_up}

    def _extract_HomogeneousListType_FunctionDefArgument(self, orig_var, collect_arg, bound_argument,
            is_bind_c_argument, *, arg_var = None):
        assert arg_var is None

        if orig_var.is_optional:
            errors.report("Optionals are not yet supported",
                    severity='fatal', symbol=orig_var)

        assert not bound_argument

        size_var = self.scope.get_temporary_variable(PythonNativeInt(), self.scope.get_new_name(f'{orig_var.name}_size'))
        body = [Assign(size_var, PyList_Size(collect_arg))]

        if is_bind_c_argument:
            element_type = orig_var.class_type.element_type
            #raise errors.report("Fortran set interface is not yet implemented", severity='fatal', symbol=orig_var)
            arr_var = Variable(NumpyNDArrayType(element_type, 1, None), self.scope.get_expected_name(orig_var.name),
                                shape = (size_var,), memory_handling = 'heap')
            self.scope.insert_variable(arr_var, orig_var.name)
            arg_var = Variable(BindCArrayType(1, False), self.scope.get_new_name(orig_var.name),
                        shape = (LiteralInteger(2),))
            data = DottedVariable(VoidType(), 'data', lhs=arr_var)
            self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(0)), data)
            self.scope.insert_symbolic_alias(IndexedElement(arg_var, LiteralInteger(1)), size_var)
            arg_vars = [arg_var]
            body.append(Allocate(arr_var, shape = (size_var,), status='unallocated'))
        else:
            arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False,
                                    memory_handling='heap', new_class = Variable, is_const = False)
            self.scope.insert_variable(arg_var, orig_var.name)
            arg_vars = [arg_var]
            body.append(Assign(arg_var, PythonList()))

        idx = self.scope.get_temporary_variable(CNativeInt())
        indexed_orig_var = self.scope.get_temporary_variable(orig_var.class_type.element_type)
        indexed_collect_arg = self.scope.get_temporary_variable(PyccelPyObject(), memory_handling='alias')

        iter_obj = self.scope.get_temporary_variable(PyccelPyObject(), 'iter', memory_handling='alias')

        body.append(AliasAssign(iter_obj, PyObject_GetIter(collect_arg)))

        for_scope = self.scope.create_new_loop_scope()
        self.scope = for_scope
        for_body = [AliasAssign(indexed_collect_arg, PyIter_Next(iter_obj))]
        for_body += self._extract_FunctionDefArgument(indexed_orig_var, indexed_collect_arg,
                                    bound_argument, is_bind_c_argument, arg_var = indexed_orig_var)['body']
        if is_bind_c_argument:
            for_body.append(Assign(IndexedElement(arr_var, idx), indexed_orig_var))
        else:
            for_body.append(ListAppend(arg_var, indexed_orig_var))
        self.exit_scope()

        body.append(For((idx,), PythonRange(size_var), for_body, scope = for_scope))

        clean_up = []
        if not orig_var.is_const:
            if is_bind_c_argument:
                errors.report("Lists should be passed as constant arguments when translating to languages other than C." +
                              "Any changes to the list will not be reflected in the calling code.",
                              severity='warning', symbol=orig_var)
            else:
                element_extraction = self._extract_FunctionDefResult(IndexedElement(orig_var, idx),
                                                is_bind_c_argument, None)
                elem_set = PyList_Append(collect_arg, element_extraction['py_result'])
                for_body = [*element_extraction['body'],
                        If(IfSection(PyccelEq(elem_set, PyccelUnarySub(LiteralInteger(1))),
                                                 [Return(self._error_exit_code)]))]

                loop_iterator = VariableIterator(arg_var)
                loop_iterator.set_loop_counter(idx)
                clean_up = [If(IfSection(PyccelEq(PyList_Clear(collect_arg), PyccelUnarySub(LiteralInteger(1))),
                                                 [Return(self._error_exit_code)])),
                        For((element_extraction['c_results'][0],), loop_iterator, for_body, for_scope)]

        return {'body': body, 'args': arg_vars, 'clean_up': clean_up}

    def _extract_StringType_FunctionDefArgument(self, orig_var, collect_arg, bound_argument,
            is_bind_c_argument, *, arg_var = None):
        """
        Extract the C-compatible string FunctionDefArgument from the PythonObject.

        Extract the C-compatible string FunctionDefArgument from the PythonObject.
        The C-compatible argument is extracted from collect_arg which holds a Python
        object into arg_var.

        The extraction is done by allocating an array and filling the elements with values
        extracted from the indexed Python tuple in collect_arg.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefArgument being wrapped.

        collect_arg : Variable
            A variable with type PythonObject* holding the Python argument from which the
            C-compatible argument should be collected.

        bound_argument : bool
            True if the argument is the self argument of a class method. False otherwise.
            This should always be False for this function.

        is_bind_c_argument : bool
            True if the argument was saved in a BindCFunctionDefArgument. False otherwise.

        arg_var : Variable | IndexedElement, optional
            A variable or an element of the variable representing the argument that
            will be passed to the low-level function call.

        Returns
        -------
        list[PyccelAstNode]
            A list of expressions which extract the argument from collect_arg into arg_var.
        """
        assert bound_argument is False

        if is_bind_c_argument:
            if arg_var is None:
                data_var = Variable(CStackArray(CharType()), self.scope.get_expected_name(orig_var.name),
                                    shape = (None,), memory_handling='alias', is_const = True)
                size_var = Variable(PythonNativeInt(), self.scope.get_new_name(f'{data_var.name}_size'))
                arg_var = Variable(BindCArrayType(1, False), self.scope.get_new_name(orig_var.name),
                                    shape = (LiteralInteger(2),))
                self.scope.insert_variable(data_var, orig_var.name)
                self.scope.insert_variable(size_var)
                self.scope.insert_variable(arg_var, tuple_recursive = False)
                self.scope.insert_symbolic_alias(arg_var[0], ObjectAddress(data_var))
                self.scope.insert_symbolic_alias(arg_var[1], size_var)

            if getattr(orig_var, 'is_optional', False):
                body = [AliasAssign(orig_var, PyUnicode_AsUTF8(collect_arg)),
                        Assign(self.scope.collect_tuple_element(arg_var[1]), PyUnicode_GetLength(collect_arg))]
            else:
                body = [Assign(orig_var, PyUnicode_AsUTF8(collect_arg)),
                        Assign(self.scope.collect_tuple_element(arg_var[1]), PyUnicode_GetLength(collect_arg))]

            default_init = [AliasAssign(data_var, Nil()),
                            Assign(size_var, 0)]
        else:

            if arg_var is None:
                arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), new_class = Variable,
                                        is_argument = False)
                self.scope.insert_variable(arg_var, orig_var.name)

            body = [Assign(orig_var, PythonStr(PyUnicode_AsUTF8(collect_arg)))]

            default_init = [AliasAssign(arg_var, Nil())]
            if getattr(orig_var, 'is_optional', False):
                memory_var = self.scope.get_temporary_variable(arg_var, name = arg_var.name + '_memory', is_optional = False,
                                                clone_scope = self.scope)
                body.insert(0, AliasAssign(arg_var, memory_var))

        return {'body': body,
                'args': [arg_var],
                'default_init': default_init}

    def _extract_FunctionDefResult(self, orig_var, is_bind_c, funcdef = None):
        """
        Get the code which translates a C-compatible `Variable` to a Python `FunctionDefResult`.

        Get the code necessary to transform a Variable returned from a C-compatible function written in
        Fortran to an object with datatype `PyccelPyObject`.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.

        is_bind_c : bool
            True if the result comes from a C-binding from another language. False otherwise.

        funcdef : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict[str, Any]
            A dictionary with the keys:
             - body : a list of PyccelAstNodes containing the code which translates the C-compatible variable
                        to a `PyccelPyObject`.
             - c_results : a list of Variables which are returned from the function being wrapped.
             - py_result : the Variable returned to Python.
             - setup : An optional key containing a list of PyccelAstNodes with code which should be
                        run before calling the function being wrapped.
        """
        if orig_var is Nil():
            return {'c_results': [], 'py_result': Py_None, 'body': []}

        if isinstance(orig_var, BindCVariable):
            class_type = orig_var.original_var.class_type
        else:
            class_type = orig_var.class_type

        classes = type(class_type).__mro__
        for cls in classes:
            annotation_method = f'_extract_{cls.__name__}_FunctionDefResult'
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(orig_var, is_bind_c, funcdef)

        # Unknown object, we raise an error.
        return errors.report(f"Wrapping function results is not implemented for type {class_type}. " + PYCCEL_RESTRICTION_TODO, symbol=orig_var,
            severity='fatal')

    def _extract_CustomDataType_FunctionDefResult(self, wrapped_var, is_bind_c, funcdef):
        """
        Get the code which translates a `Variable` containing a class instance to a PyObject.

        Get the code which translates a `Variable` containing a class instance to a PyObject.

        Parameters
        ----------
        wrapped_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.
        is_bind_c : bool
            True if the result comes from a C-binding from another language. False otherwise.
        funcdef : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result.
        """
        orig_var = getattr(wrapped_var, 'original_var', wrapped_var)
        name = orig_var.name
        python_res = self.get_new_PyObject(f'{name}_obj', orig_var.dtype)
        setup = self._allocate_class_instance(python_res, python_res.cls_base.scope, orig_var.is_alias)
        if is_bind_c:
            c_res = orig_var.clone(self.scope.get_new_name(orig_var.name), is_argument = False,
                                    memory_handling='alias', new_class = Variable)
            self.scope.insert_variable(c_res, orig_var.name)
            scope = python_res.cls_base.scope
            attribute = scope.find('instance', 'variables', raise_if_missing = True)
            attrib_var = attribute.clone(attribute.name, new_class = DottedVariable, lhs = python_res)
            body = [AliasAssign(attrib_var, c_res)]
            result = ObjectAddress(c_res)
        else:
            scope = python_res.cls_base.scope
            attribute = scope.find('instance', 'variables', raise_if_missing = True)
            c_res = attribute.clone(attribute.name, new_class = DottedVariable, lhs = python_res)
            setup.append(Allocate(c_res, shape=None, status='unallocated', like=orig_var))
            result = PointerCast(c_res, cast_type = orig_var)
            body = []

        if funcdef:
            body.extend(self.connect_pointer_targets(orig_var, python_res, funcdef, is_bind_c))

        return {'c_results': [result], 'py_result': python_res, 'body': body, 'setup': setup}

    def _extract_FixedSizeType_FunctionDefResult(self, orig_var, is_bind_c, funcdef):
        """
        Get the code which translates a `Variable` containing a scalar to a PyObject.

        Get the code which translates a `Variable` containing a scalar to a PyObject.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.
        is_bind_c : bool
            True if the result comes from a C-binding from another language. False otherwise.
        funcdef : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result.
        """
        name = getattr(orig_var, 'name', 'tmp')
        py_res = self.get_new_PyObject(f'{name}_obj', orig_var.dtype)
        c_res = Variable(orig_var.class_type, self.scope.get_new_name(name))
        self.scope.insert_variable(c_res)

        body = [AliasAssign(py_res, FunctionCall(C_to_Python(c_res), [c_res]))]
        return {'c_results': [c_res], 'py_result': py_res, 'body': body}

    def _extract_NumpyNDArrayType_FunctionDefResult(self, orig_var, is_bind_c, funcdef):
        """
        Get the code which translates a `Variable` containing an array to a PyObject.

        Get the code which translates a `Variable` containing an array to a PyObject.

        Parameters
        ----------
        orig_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.
        is_bind_c : bool
            True if the result comes from a C-binding from another language. False otherwise.
        funcdef : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result.
        """
        if is_bind_c:
            return self._extract_BindCArrayType_FunctionDefResult(orig_var, funcdef)
        name = self.scope.get_new_name(orig_var.name)
        py_res = self.get_new_PyObject(f'{name}_obj', orig_var.dtype)
        c_res = orig_var.clone(name, is_argument = False, memory_handling='alias')
        typenum = numpy_dtype_registry[orig_var.dtype]
        data_var = DottedVariable(VoidType(), 'data', memory_handling='alias',
                    lhs=c_res)
        shape_var = DottedVariable(CStackArray(PythonNativeInt()), 'shape',
                    lhs=c_res)
        release_memory = False
        if funcdef:
            arg_targets = funcdef.result_pointer_map.get(orig_var, ())
            release_memory = len(arg_targets) == 0 and not isinstance(orig_var, DottedVariable)
        body = [AliasAssign(py_res, to_pyarray(
                         LiteralInteger(orig_var.rank), typenum, data_var, shape_var,
                         convert_to_literal(orig_var.order != 'F'),
                         convert_to_literal(release_memory)))]
        self.scope.insert_variable(c_res)
        c_result_vars = [c_res]

        if funcdef:
            body.extend(self.connect_pointer_targets(orig_var, py_res, funcdef, False))

        return {'c_results': c_result_vars, 'py_result': py_res, 'body': body}

    def _extract_BindCArrayType_FunctionDefResult(self, wrapped_var, funcdef):
        """
        Get the code which translates a `Variable` containing an array to a PyObject.

        Get the code which translates a `Variable` containing a BindCArray, which describes an
        array in Fortran, to a PyObject.

        Parameters
        ----------
        wrapped_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.
        funcdef : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result.
        """
        orig_var = wrapped_var.original_var
        name = orig_var.name
        py_res = self.get_new_PyObject(f'{name}_obj', orig_var.dtype)
        # Result of calling the bind-c function
        data_var = Variable(VoidType(), self.scope.get_new_name(name+'_data'), memory_handling='alias')
        shape_var = Variable(CStackArray(NumpyInt32Type()), self.scope.get_new_name(name+'_shape'),
                        shape = (orig_var.rank,), memory_handling='alias')
        typenum = numpy_dtype_registry[orig_var.dtype]
        # Save so we can find by iterating over func.results
        self.scope.insert_variable(data_var)
        self.scope.insert_variable(shape_var)

        release_memory = False
        if funcdef:
            arg_targets = funcdef.result_pointer_map.get(orig_var, ())
            release_memory = len(arg_targets) == 0 and not isinstance(orig_var, DottedVariable)

        body = [AliasAssign(py_res, to_pyarray(
                         LiteralInteger(orig_var.rank), typenum, data_var, shape_var,
                         convert_to_literal(orig_var.order != 'F'),
                         convert_to_literal(release_memory)))]

        shape_vars = [IndexedElement(shape_var, i) for i in range(orig_var.rank)]
        c_result_vars = [ObjectAddress(data_var)]+shape_vars

        if funcdef:
            body.extend(self.connect_pointer_targets(orig_var, py_res, funcdef, True))

        return {'c_results': c_result_vars, 'py_result': py_res, 'body': body}

    def _extract_InhomogeneousTupleType_FunctionDefResult(self, wrapped_var, is_bind_c, funcdef):
        """
        Get the code which translates a `Variable` containing an inhomogeneous tuple to a PyObject.

        Get the code which translates a `Variable` containing an inhomogeneous tuple to a PyObject.

        Parameters
        ----------
        wrapped_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.
        is_bind_c : bool
            True if the result comes from a C-binding from another language. False otherwise.
        funcdef : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result.
        """
        orig_var = getattr(wrapped_var, 'original_var', wrapped_var)
        name = orig_var.name if isinstance(orig_var, Variable) else 'Out'
        c_compatible_var = getattr(wrapped_var, 'new_var', wrapped_var)
        extract_elems = [self._extract_FunctionDefResult(funcdef.scope.collect_tuple_element(e),
                                                         is_bind_c, funcdef) for e in c_compatible_var]
        body = [l for e in extract_elems for l in e['body']]
        setup = [l for e in extract_elems for l in e.get('setup', ())]
        c_result_vars = [r for e in extract_elems for r in e['c_results']]
        py_result_vars = [e['py_result'] for e in extract_elems]
        py_res = self.get_new_PyObject(f'{name}_obj')
        body.append(AliasAssign(py_res, PyTuple_Pack(*[ObjectAddress(r) for r in py_result_vars])))
        body.extend(Py_DECREF(r) for r in py_result_vars)

        return {'c_results': PythonTuple(*c_result_vars), 'py_result': py_res, 'body': body, 'setup': setup}

    def _extract_HomogeneousContainerType_FunctionDefResult(self, wrapped_var, is_bind_c, funcdef):
        """
        Get the code which translates a `Variable` containing a homogeneous container to a PyObject.

        Get the code which translates a `Variable` containing a homogeneous container to a PyObject.
        This function handles lists, sets, and tuples.
        The current implementation for tuples is not working correctly as Pyccel does not handle
        function calls to functions returning tuples correctly.

        Parameters
        ----------
        wrapped_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.
        is_bind_c : bool
            True if the result comes from a C-binding from another language. False otherwise.
        funcdef : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result.
        """
        if isinstance(wrapped_var, PythonTuple):
            return self._extract_InhomogeneousTupleType_FunctionDefResult(wrapped_var, is_bind_c, funcdef)

        orig_var = getattr(wrapped_var, 'original_var', wrapped_var)
        name = getattr(orig_var, 'name', 'tmp')
        py_res = self.get_new_PyObject(f'{name}_obj', orig_var.dtype)
        if is_bind_c:
            result = wrapped_var.new_var
            ptr_var = funcdef.scope.collect_tuple_element(result[0])
            shape_var = funcdef.scope.collect_tuple_element(result[1])
            c_res = Variable(CStackArray(orig_var.class_type.element_type),
                             self.scope.get_new_name(ptr_var.name))
            loop_size = shape_var.clone(self.scope.get_new_name(shape_var.name), is_argument = False)
            c_results = [ObjectAddress(c_res), loop_size]
        else:
            c_res = orig_var.clone(self.scope.get_new_name(name), is_argument = False)
            c_results = [c_res]
            loop_size = Variable(PythonNativeInt(), self.scope.get_new_name(f'{name}_size'))
        idx = Variable(PythonNativeInt(), self.scope.get_new_name())
        self.scope.insert_variable(c_res)
        self.scope.insert_variable(loop_size)
        self.scope.insert_variable(idx)

        for_scope = self.scope.create_new_loop_scope()
        self.scope = for_scope
        element_extraction = self._extract_FunctionDefResult(IndexedElement(orig_var, idx), is_bind_c, funcdef)
        self.exit_scope()

        class_type = orig_var.class_type
        if isinstance(class_type, HomogeneousSetType):
            if is_bind_c:
                element = IndexedElement(c_res, idx)
            else:
                element = SetPop(c_res)
            elem_set = PySet_Add(py_res, element_extraction['py_result'])
            init = PySet_New()
        elif isinstance(class_type, HomogeneousListType):
            element = IndexedElement(c_res, idx)
            elem_set = PyList_SetItem(py_res, idx, element_extraction['py_result'])
            init = PyList_New(loop_size)
        elif isinstance(class_type, HomogeneousTupleType):
            element = IndexedElement(c_res, idx)
            elem_set = PyTuple_SetItem(py_res, idx, element_extraction['py_result'])
            init = PyTuple_New(loop_size)
        else:
            raise NotImplementedError(f"Don't know how to return an object of type {class_type}")

        for_body = [Assign(element_extraction['c_results'][0], element),
                *element_extraction['body'],
                If(IfSection(PyccelEq(elem_set, PyccelUnarySub(LiteralInteger(1))),
                                         [Return(self._error_exit_code)]))]
        body = [Assign(loop_size, PythonLen(c_res))] if not is_bind_c else []
        body += [AliasAssign(py_res, init),
                 For((idx,), PythonRange(loop_size), for_body, for_scope)]
        if is_bind_c:
            body.append(Deallocate(c_res))

        return {'c_results': c_results, 'py_result': py_res, 'body': body}

    def _extract_DictType_FunctionDefResult(self, wrapped_var, is_bind_c, funcdef):
        """
        Get the code which translates a `Variable` containing a dictionary to a PyObject.

        Get the code which translates a `Variable` containing a dictionary to a PyObject.

        Parameters
        ----------
        wrapped_var : Variable | IndexedElement
            An object representing the variable or an element of the variable from the
            FunctionDefResult being wrapped.
        is_bind_c : bool
            True if the result was saved in a BindCFunctionDefResult. False otherwise.
        funcdef : FunctionDef
            The function being wrapped.

        Returns
        -------
        dict
            A dictionary describing the objects necessary to collect the result.
        """
        orig_var = getattr(wrapped_var, 'original_var', wrapped_var)
        name = getattr(orig_var, 'name', 'tmp')
        py_res = self.get_new_PyObject(f'{name}_obj', orig_var.dtype)
        if is_bind_c:
            result = wrapped_var.new_var
            key_ptr_var = funcdef.scope.collect_tuple_element(result[0])
            val_ptr_var = funcdef.scope.collect_tuple_element(result[1])
            shape_var = funcdef.scope.collect_tuple_element(result[2])
            key_c_res = Variable(CStackArray(orig_var.class_type.key_type),
                             self.scope.get_new_name(key_ptr_var.name))
            val_c_res = Variable(CStackArray(orig_var.class_type.value_type),
                             self.scope.get_new_name(val_ptr_var.name))
            loop_size = shape_var.clone(self.scope.get_new_name(shape_var.name), is_argument = False)
            c_results = [ObjectAddress(key_c_res), ObjectAddress(val_c_res), loop_size]
            iterable = PythonRange(shape_var)
            self.scope.insert_variable(key_c_res)
            self.scope.insert_variable(val_c_res)
            self.scope.insert_variable(shape_var)
            idx = Variable(PythonNativeInt(), self.scope.get_new_name())
            self.scope.insert_variable(idx)
            iterable.set_loop_counter(idx)
            key_elem = IndexedElement(key_c_res, idx)
            val_elem = IndexedElement(val_c_res, idx)
        else:
            c_res = orig_var.clone(self.scope.get_new_name(name), is_argument = False)
            c_results = [c_res]
            iterable = DictItems(c_res)
            self.scope.insert_variable(c_res)
            iterable.set_loop_counter(Variable(PythonNativeInt(), self.scope.get_new_name()))
            key_elem, val_elem = iterable.get_python_iterable_item()


        for_scope = self.scope.create_new_loop_scope()
        self.scope = for_scope
        key_extraction = self._extract_FunctionDefResult(key_elem, is_bind_c, funcdef)
        value_extraction = self._extract_FunctionDefResult(val_elem, is_bind_c, funcdef)
        elem_set = PyDict_SetItem(py_res, key_extraction['py_result'], value_extraction['py_result'])
        for_body = [*key_extraction['body'], *value_extraction['body'],
                    If(IfSection(PyccelEq(elem_set, PyccelUnarySub(LiteralInteger(1))),
                                         [Return(self._error_exit_code)]))]
        if is_bind_c:
            for_body = [Assign(key_extraction['c_results'][0], key_elem),
                        Assign(value_extraction['c_results'][0], val_elem)] + for_body
        self.exit_scope()
        for_loop = For((key_extraction['c_results'][0], value_extraction['c_results'][0]),
                        iterable, for_body, for_scope)

        body = [AliasAssign(py_res, PyDict_New()),
                for_loop]

        return {'c_results': c_results, 'py_result': py_res, 'body': body}

    def _extract_StringType_FunctionDefResult(self, wrapped_var, is_bind_c, funcdef):
        orig_var = getattr(wrapped_var, 'original_var', wrapped_var)
        name = getattr(orig_var, 'name', 'tmp')
        py_res = self.get_new_PyObject(f'{name}_obj', orig_var.dtype)
        if is_bind_c:
            c_res = Variable(CharType(), self.scope.get_new_name(name+'_data'), memory_handling='alias')
            self.scope.insert_variable(c_res)
            char_data = ObjectAddress(c_res)
            result = [char_data]
        else:
            c_res = Variable(StringType(), self.scope.get_new_name(name), memory_handling='heap')
            self.scope.insert_variable(c_res)
            char_data = CStrStr(c_res)
            result = [c_res]

        body = [AliasAssign(py_res, PyBuildValueNode([char_data]))]
        if is_bind_c:
            body.append(Deallocate(c_res))
        return {'c_results': result, 'py_result': py_res, 'body': body}

