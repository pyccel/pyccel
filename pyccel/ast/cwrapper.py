#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module representing objects (functions/variables etc) required for the interface
between Python code and C code (using Python/C Api and cwrapper.c).
"""

from pyccel.utilities.metaclasses import Singleton

from ..errors.errors import Errors
from ..errors.messages import PYCCEL_RESTRICTION_TODO

from .basic     import PyccelAstNode

from .bind_c    import BindCPointer

from .datatypes import DataType, default_precision, CustomDataType
from .datatypes import NativeInteger, NativeFloat, NativeComplex
from .datatypes import NativeBool, NativeString, NativeVoid

from .core      import FunctionDefArgument, FunctionDefResult
from .core      import FunctionDef, ClassDef
from .core      import Module, Interface, Declare

from .internals import get_final_precision, PyccelInternalFunction

from .literals  import LiteralString, LiteralInteger

from .variable  import Variable

from .c_concepts import ObjectAddress


errors = Errors()

__all__ = (
# --------- DATATYPES -----------
    'PyccelPyObject',
    'PyccelPyClassType',
    'PyccelPyTypeObject',
    'WrapperCustomDataType',
# --------- CLASSES -----------
    'PyFunctionDef',
    'PyInterface',
    'PyClassDef',
    'PyModule',
    'PyArgKeywords',
    'PyArg_ParseTupleNode',
    'PyBuildValueNode',
    'PyCapsule_New',
    'PyCapsule_Import',
    'PyGetSetDefElement',
    'PyModule_Create',
    'PyModule_AddObject',
    'PyModInitFunc',
#--------- CONSTANTS ----------
    'Py_True',
    'Py_False',
    'Py_None',
#----- C / PYTHON FUNCTIONS ---
    'Py_INCREF',
    'Py_DECREF',
    'PyObject_TypeCheck',
    'PySys_GetObject',
    'PyUnicode_FromString',
    'PyList_GetItem',
    'PyList_SetItem',
    'PyErr_Occurred',
    'PyErr_SetString',
    'PyAttributeError',
    'PyNotImplementedError',
    'PyTypeError',
    'PyObject_TypeCheck',
)

#-------------------------------------------------------------------
#                        Python DataTypes
#-------------------------------------------------------------------
class PyccelPyObject(DataType, metaclass=Singleton):
    """
    Datatype representing a `PyObject`.

    Datatype representing a `PyObject` which is the
    class used to hold Python objects in `Python.h`.
    """
    __slots__ = ()
    _name = 'pyobject'

class PyccelPyClassType(DataType, metaclass=Singleton):
    """
    Datatype representing a subclass of `PyObject`.

    Datatype representing a subclass of `PyObject`. This is the
    datatype of a class which is compatible with Python.
    """
    __slots__ = ()
    _name = 'pyclasstype'

class PyccelPyTypeObject(DataType, metaclass=Singleton):
    """
    Datatype representing a `PyTypeObject`.

    Datatype representing a `PyTypeObject` which is the
    class used to hold Python class objects in `Python.h`.
    """
    __slots__ = ()
    _name = 'pytypeobject'

class WrapperCustomDataType(CustomDataType):
    """
    Datatype representing a subclass of `PyObject`.

    Datatype representing a subclass of `PyObject`. This is the
    datatype of a class which is compatible with Python.
    """
    __slots__ = ()
    _name = 'pycustomclasstype'

#-------------------------------------------------------------------
#                  Parsing and Building Classes
#-------------------------------------------------------------------

#TODO: Is there an equivalent to static so this can be a static list of strings?
class PyArgKeywords(PyccelAstNode):
    """
    Represents the list containing the names of all arguments to a function.
    This information allows the function to be called by keyword

    Parameters
    ----------
    name : str
        The name of the variable in which the list is stored
    arg_names : list of str
        A list of the names of the function arguments
    """
    __slots__ = ('_name','_arg_names')
    _attribute_nodes = ()
    def __init__(self, name, arg_names):
        self._name = name
        self._arg_names = arg_names
        super().__init__()

    @property
    def name(self):
        """ The name of the variable in which the list of
        all arguments to the function is stored
        """
        return self._name

    @property
    def arg_names(self):
        """ The names of the arguments to the function which are
        contained in the PyArgKeywords list
        """
        return self._arg_names

#-------------------------------------------------------------------
class PyArg_ParseTupleNode(PyccelAstNode):
    """
    Represents a call to the function `PyArg_ParseTupleNode`.

    Represents a call to the function `PyArg_ParseTupleNode` from `Python.h`.
    This function collects the expected arguments from `self`, `args`, `kwargs`
    and packs them into variables with datatype `PyccelPyObject`.

    Parameters
    ----------
    python_func_args : Variable
        Args provided to the function in Python.
    python_func_kwargs : Variable
        Kwargs provided to the function in Python.
    c_func_args : list of Variable
        List of expected arguments. This helps determine the expected output types.
    parse_args : list of Variable
        List of arguments into which the result will be collected.
    arg_names : list of str
        A list of the names of the function arguments.
    """
    __slots__ = ('_pyarg','_pykwarg','_parse_args','_arg_names','_flags')
    _attribute_nodes = ('_pyarg','_pykwarg','_parse_args','_arg_names')

    def __init__(self, python_func_args,
                        python_func_kwargs,
                        c_func_args,
                        parse_args,
                        arg_names):
        if not isinstance(python_func_args, Variable):
            raise TypeError('Python func args should be a Variable')
        if not isinstance(python_func_kwargs, Variable):
            raise TypeError('Python func kwargs should be a Variable')
        if not isinstance(parse_args, list) and any(not isinstance(c, Variable) for c in parse_args):
            raise TypeError('Parse args should be a list of Variables')
        if not isinstance(arg_names, PyArgKeywords):
            raise TypeError('Parse args should be a list of Variables')

        i = 0
        while i < len(c_func_args) and not c_func_args[i].has_default:
            i+=1
        self._flags = 'O'*i

        if i < len(c_func_args):
            self._flags += '|'
            self._flags += 'O'*(len(c_func_args)-i)

        self._pyarg      = python_func_args
        self._pykwarg    = python_func_kwargs
        self._parse_args = parse_args
        self._arg_names  = arg_names
        super().__init__()

    @property
    def pyarg(self):
        """ The  variable containing all positional arguments
        passed to the function
        """
        return self._pyarg

    @property
    def pykwarg(self):
        """ The  variable containing all keyword arguments
        passed to the function
        """
        return self._pykwarg

    @property
    def flags(self):
        """
        The flags indicating the types of the objects.

        The flags indicating the types of the objects to be collected from
        the Python arguments passed to the function.
        """
        return self._flags

    @property
    def args(self):
        """ The arguments into which the python args and kwargs
        are collected
        """
        return self._parse_args

    @property
    def arg_names(self):
        """ The PyArgKeywords object which contains all the
        names of the function's arguments
        """
        return self._arg_names

#-------------------------------------------------------------------
class PyBuildValueNode(PyccelInternalFunction):
    """
    Represents a call to the function PyBuildValueNode.

    The function PyBuildValueNode can be found in Python.h.
    It describes the creation of a new Python object based
    on a format string. More details can be found in Python's
    docs.

    Parameters
    ----------
    result_args : list of Variable
        List of arguments which the result will be built from.
    """
    __slots__ = ('_flags','_result_args')
    _attribute_nodes = ('_result_args',)
    _dtype = PyccelPyObject
    _rank = 0
    _precision = 0
    _shape = ()
    _class_type = PyccelPyObject
    _order = None

    def __init__(self, result_args = ()):
        self._flags = ''
        self._result_args = result_args
        for i in result_args:
            if isinstance(i.dtype, WrapperCustomDataType):
                self._flags += 'O'
            else:
                self._flags += pytype_parse_registry[(i.dtype, get_final_precision(i))]
        super().__init__()

    @property
    def flags(self):
        return self._flags

    @property
    def args(self):
        return self._result_args

#-------------------------------------------------------------------
class PyModule_AddObject(PyccelInternalFunction):
    """
    Represents a call to the PyModule_AddObject function.

    The PyModule_AddObject function can be found in Python.h.
    It adds a PythonObject to a module. More information about
    this function can be found in Python's documentation.

    Parameters
    ----------
    mod_name : str
                The name of the variable containing the module.
    name : str
                The name of the variable being added to the module.
    variable : Variable
                The variable containing the PythonObject.
    """
    __slots__ = ('_mod_name','_name','_var')
    _attribute_nodes = ('_name','_var')
    _dtype = NativeInteger()
    _precision = 4
    _rank = 0
    _shape = None
    _class_type = NativeInteger()

    def __init__(self, mod_name, name, variable):
        if not isinstance(name, LiteralString):
            raise TypeError("Name must be a string")
        if not isinstance(variable, Variable) or \
                variable.dtype not in (PyccelPyObject(), PyccelPyClassType()):
            raise TypeError("Variable must be a PyObject Variable")
        self._mod_name = mod_name
        self._name = name
        self._var = ObjectAddress(variable)
        super().__init__()

    @property
    def mod_name(self):
        """ The name of the variable containing the module
        """
        return self._mod_name

    @property
    def name(self):
        """ The name of the variable being added to the module
        """
        return self._name

    @property
    def variable(self):
        """ The variable containing the PythonObject
        """
        return self._var

#-------------------------------------------------------------------
class PyModule_Create(PyccelInternalFunction):
    """
    Represents a call to the PyModule_Create function.

    The PyModule_Create function can be found in Python.h.
    It acts as a constructor for a module. More information about
    this function can be found in Python's documentation.
    See <https://docs.python.org/3/c-api/module.html#c.PyModule_Create>.

    Parameters
    ----------
    module_def_name : str
        The name of the structure which defined the module.
    """
    __slots__ = ('_module_def_name',)
    _attribute_nodes = ()
    _dtype = PyccelPyObject()
    _precision = 0
    _rank = 0
    _shape = ()
    _order = None
    _class_type = PyccelPyObject()

    def __init__(self, module_def_name):
        self._module_def_name = module_def_name
        super().__init__()

    @property
    def module_def_name(self):
        """
        Get the name of the structure which defined the module.

        Get the name of the structure which defined the module.
        """
        return self._module_def_name

#-------------------------------------------------------------------
class PyCapsule_New(PyccelInternalFunction):
    """
    Represents a call to the function PyCapsule_New.

    The function PyCapsule_New can be found in Python.h. It describes
    the creation of a capsule. A capsule contains all information
    from a module which should be exposed to other modules that import
    this module.
    See <https://docs.python.org/3/extending/extending.html#using-capsules>
    for a tutorial involving capsules.
    See <https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_New>
    for the API docstrings for this method.

    Parameters
    ----------
    API_var : Variable
        The variable which contains all elements of the API which should be exposed.

    module_name : str
        The name of the module being exposed.
    """
    __slots__ = ('_capsule_name', '_API_var')
    _attribute_nodes = ('_API_var',)
    _dtype = PyccelPyObject()
    _precision = 0
    _rank = 0
    _shape = ()
    _order = None
    _class_type = PyccelPyObject()

    def __init__(self, API_var, module_name):
        self._capsule_name = f'{module_name}._C_API'
        self._API_var = API_var
        super().__init__()

    @property
    def capsule_name(self):
        """
        Get the name of the capsule being created.

        Get the name of the capsule being created.
        """
        return self._capsule_name

    @property
    def API_var(self):
        """
        Get the variable describing the API.

        Get the variable which contains all elements of the API which
        should be exposed.
        """
        return self._API_var

#-------------------------------------------------------------------
class PyCapsule_Import(PyccelInternalFunction):
    """
    Represents a call to the function PyCapsule_Import.

    The function PyCapsule_Import can be found in Python.h. It describes
    the initialisation of a capsule by importing the information from
    another module. A capsule contains all information from a module
    which should be exposed to other modules that import this module.
    See <https://docs.python.org/3/extending/extending.html#using-capsules>
    for a tutorial involving capsules.
    See <https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_Import>
    for the API docstrings for this method.

    Parameters
    ----------
    module_name : str
        The name of the module being retrieved.
    """
    __slots__ = ('_capsule_name',)
    _attribute_nodes = ()
    _dtype = BindCPointer()
    _precision = 0
    _rank = 0
    _shape = ()
    _order = None
    _class_type = BindCPointer()

    def __init__(self, module_name):
        self._capsule_name = f'{module_name}._C_API'
        super().__init__()

    @property
    def capsule_name(self):
        """
        Get the name of the capsule being retrieved.

        Get the name of the capsule being retrieved.
        """
        return self._capsule_name

#-------------------------------------------------------------------
class PyModule(Module):
    """
    Class to hold a module which is accessible from Python.

    Class to hold a module which is accessible from Python. This class
    adds external functions and external declarations to the basic
    Module. However its main utility is in order to differentiate
    itself such that a different `_print` function can be implemented
    to handle it.

    Parameters
    ----------
    name : str
        Name of the module.

    *args : tuple
        See Module.

    external_funcs : iterable of FunctionDef
        A list of external functions.

    declarations : iterable
        Any declarations of (external) variables which should be made in the module.

    init_func : FunctionDef, optional
        The function which is executed when a module is initialised.
        See: <https://docs.python.org/3/c-api/module.html#multi-phase-initialization>.

    import_func : FunctionDef, optional
        The function which allows types from this module to be imported in other
        modules.
        See: <https://docs.python.org/3/extending/extending.html>.

    **kwargs : dict
        See Module.

    See Also
    --------
    Module : The super class from which the class inherits.
    """
    __slots__ = ('_external_funcs', '_declarations', '_import_func')
    _attribute_nodes = Module._attribute_nodes + ('_external_funcs', '_declarations', '_import_func')

    def __init__(self, name, *args, external_funcs = (), declarations = (), init_func = None,
                        import_func = None, **kwargs):
        self._external_funcs = external_funcs
        self._declarations = declarations
        if import_func is None:
            self._import_func = FunctionDef(f'{name}_import', (),
                            (FunctionDefResult(Variable(NativeInteger(), '_', precision=-2, is_temp=True)),), ())
        else:
            self._import_func = import_func
        super().__init__(name, *args, init_func = init_func, **kwargs)

    @property
    def external_funcs(self):
        """
        A list of external functions.

        The external functions which should be declared at the start of the module.
        This is useful for declaring the existence of Fortran functions whose
        definition and declaration is inaccessible from C.
        """
        return self._external_funcs

    @external_funcs.setter
    def external_funcs(self, funcs):
        for f in self._external_funcs:
            f.remove_user_node(self)
        self._external_funcs = funcs
        for f in funcs:
            f.set_current_user_node(self)

    @property
    def declarations(self):
        """
        All declarations that need printing in the module.

        All declarations that need printing in the module. This usually includes
        any variables coming from a non-C language for which compatibility with C
        exists.
        """
        return self._declarations

    @declarations.setter
    def declarations(self, decs):
        for d in self._declarations:
            d.remove_user_node(self)
        self._declarations = decs
        for d in decs:
            d.set_current_user_node(self)

    @property
    def import_func(self):
        """
        The function which allows types from this module to be imported in other modules.

        The function which allows types from this module to be imported in other modules.
        See <https://docs.python.org/3/extending/extending.html> to understand how this
        is done.
        """
        return self._import_func

#-------------------------------------------------------------------
class PyFunctionDef(FunctionDef):
    """
    Class to hold a FunctionDef which is accessible from Python.

    Contains the Python-compatible version of the function which is
    used for the wrapper.
    As compared to a normal FunctionDef, this version contains
    arguments for the shape of arrays. It should be generated by
    calling `codegen.wrapper.CToPythonWrapper.wrap`.

    Parameters
    ----------
    *args : list
        See FunctionDef.

    original_function : FunctionDef
        The function from which the Python-compatible version was created.

    **kwargs : dict
        See FunctionDef.

    See Also
    --------
    pyccel.ast.core.FunctionDef
        The class from which BindCFunctionDef inherits which contains all
        details about the args and kwargs.
    """
    __slots__ = ('_original_function',)
    _attribute_nodes = (*FunctionDef._attribute_nodes, '_original_function')

    def __init__(self, *args, original_function, **kwargs):
        self._original_function = original_function
        super().__init__(*args, **kwargs, is_static = True)

    @property
    def original_function(self):
        """
        The function which is wrapped by this PyFunctionDef.

        The original function which would be printed in pure C which is not
        compatible with Python.
        """
        return self._original_function

#-------------------------------------------------------------------
class PyInterface(Interface):
    """
    Class to hold an Interface which is accessible from Python.

    A class which holds the Python-compatible Interface. It contains functions for
    determining the type of the arguments passed to the Interface and the functions
    called through the interface.

    Parameters
    ----------
    name : str
        The name of the interface. See Interface.

    functions : iterable of FunctionDef
        The functions of the interface. See Interface.

    interface_func : FunctionDef
        The function which Python will call to access the interface.

    type_check_func : FunctionDef
        The helper function which will determine the types of the arguments passed.

    original_interface : Interface
        The interface being wrapped.

    **kwargs : dict
        See Interface.

    See Also
    --------
    Interface : The super class.
    """
    __slots__ = ('_interface_func', '_type_check_func', '_original_interface')
    _attribute_nodes = Interface._attribute_nodes + ('_interface_func', '_type_check_func',
                        '_original_interface')

    def __init__(self, name, functions, interface_func, type_check_func, original_interface, **kwargs):
        self._interface_func = interface_func
        self._type_check_func = type_check_func
        self._original_interface = original_interface
        for f in functions:
            if not isinstance(f, PyFunctionDef):
                raise TypeError("PyInterface functions should be instances of the class PyFunctionDef.")
        super().__init__(name, functions, False, **kwargs)

    @property
    def interface_func(self):
        """
        The function which is exposed to Python.

        The function which receives the Python arguments `self`, `args`, and `kwargs` and calls
        the appropriate function.
        """
        return self._interface_func

    @property
    def type_check_func(self):
        """
        The function which determines the types which were passed to the Interface.

        The function which takes the arguments passed to the function and returns an integer
        indicating which function was called.
        """
        return self._type_check_func

    @property
    def original_function(self):
        """
        The Interface which is wrapped by this PyInterface.

        The original interface which would be printed in C.
        """
        return self._original_interface

#-------------------------------------------------------------------
class PyClassDef(ClassDef):
    """
    Class to hold a class definition which is accessible from Python.

    Class to hold a class definition which is accessible from Python.

    Parameters
    ----------
    original_class : ClassDef
        The original class being wrapped.

    struct_name : str
        The name of the structure which will hold the Python-compatible
        class definition.

    type_name : str
        The name of the instance of the Python-compatible class definition
        structure. This object is necessary to add the class to the module.

    scope : Scope
        The scope for the class contents.

    **kwargs : dict
        See ClassDef.

    See Also
    --------
    ClassDef
        The class from which PyClassDef inherits. This is also the object being
        wrapped.
    """
    __slots__ = ('_original_class', '_struct_name', '_type_name', '_type_object',
                 '_new_func', '_properties')

    def __init__(self, original_class, struct_name, type_name, scope, **kwargs):
        self._original_class = original_class
        self._struct_name = struct_name
        self._type_name = type_name
        self._type_object = Variable(PyccelPyClassType(), type_name)
        self._new_func = None
        self._properties = ()
        variables = [Variable(NativeVoid(), 'instance', memory_handling='alias'),
                     Variable(PyccelPyObject(), 'referenced_objects', memory_handling='alias'),
                     Variable(NativeBool(), 'is_alias')]
        scope.insert_variable(variables[0])
        scope.insert_variable(variables[1])
        scope.insert_variable(variables[2])
        super().__init__(original_class.name, variables, scope=scope, **kwargs)

    @property
    def struct_name(self):
        """
        The name of the structure which will hold the Python-compatible class definition.

        The name of the structure which will hold the Python-compatible class definition.
        """
        return self._struct_name

    @property
    def type_name(self):
        """
        The name of the Python-compatible class definition instance.

        The name of the instance of the Python-compatible class definition
        structure. This object is necessary to add the class to the module.
        """
        return self._type_name

    @property
    def type_object(self):
        """
        The Python-compatible class definition instance.

        The Variable describing the instance of the Python-compatible class definition
        structure. This object is necessary to add the class to the module.
        """
        return self._type_object

    @property
    def original_class(self):
        """
        The class which is wrapped by this PyClassDef.

        The original class which would be printed in pure C which is not
        compatible with Python.
        """
        return self._original_class

    def add_alloc_method(self, f):
        """
        Add the wrapper for `__new__` to the class definition.

        Add the wrapper for `__new__` which allocates the memory for the class instance.

        Parameters
        ----------
        f : PyFunctionDef
            The wrapper for the `__new__` function.
        """
        self._new_func = f

    @property
    def new_func(self):
        """
        Get the wrapper for `__new__`.

        Get the wrapper for `__new__` which allocates the memory for the class instance.
        """
        return self._new_func

    def add_property(self, p):
        """
        Add a class property which has been wrapped.

        Add a class property which has been wrapped.

        Parameters
        ----------
        p : PyccelAstNode
            The new wrapped property which is added to the class.
        """
        p.set_current_user_node(self)
        self._properties += (p,)

    @property
    def properties(self):
        """
        Get all wrapped class properties.

        Get all wrapped class properties.
        """
        return self._properties

#-------------------------------------------------------------------

class PyGetSetDefElement(PyccelAstNode):
    """
    A class representing a PyGetSetDef object.

    A class representing an element of the list of PyGetSetDef objects
    which are used to add attributes/properties to classes.
    See <https://docs.python.org/3/c-api/structures.html#c.PyGetSetDef>.

    Parameters
    ----------
    python_name : str
        The name of the attribute/property in the original Python code.
    getter : FunctionDef
        The function which collects the value of the class attribute.
    setter : FunctionDef
        The function which modifies the value of the class attribute.
    docstring : LiteralString
        The docstring of the property.
    """
    _attribute_nodes = ('_getter', '_setter', '_docstring')
    __slots__ = ('_python_name', '_getter', '_setter', '_docstring')
    def __init__(self, python_name, getter, setter, docstring):
        if not isinstance(getter, PyFunctionDef):
            raise TypeError("Getter should be a PyFunctionDef")
        if not isinstance(setter, PyFunctionDef):
            raise TypeError("Setter should be a PyFunctionDef")
        self._python_name = python_name
        self._getter = getter
        self._setter = setter
        self._docstring = docstring
        super().__init__()

    @property
    def python_name(self):
        """
        The name of the attribute/property in the original Python code.

        The name of the attribute/property in the original Python code.
        """
        return self._python_name

    @property
    def getter(self):
        """
        The BindCFunctionDef describing the getter function.

        The BindCFunctionDef describing the function which allows the user to collect
        the value of the property.
        """
        return self._getter

    @property
    def setter(self):
        """
        The BindCFunctionDef describing the setter function.

        The BindCFunctionDef describing the function which allows the user to modify
        the value of the property.
        """
        return self._setter

    @property
    def docstring(self):
        """
        The docstring of the property being wrapped.

        The docstring of the property being wrapped.
        """
        return self._docstring

#-------------------------------------------------------------------
class PyModInitFunc(FunctionDef):
    """
    A class representing the PyModInitFunc function def.

    A class representing the PyModInitFunc function def. This function returns the
    macro PyModInitFunc, takes no arguments and initialises a module.

    Parameters
    ----------
    name : str
        The name of the function.

    body : list[PyccelAstNode]
        The code executed in the function.

    static_vars : list[Variable]
        A list of variables which should be declared as static objects.

    scope : Scope
        The scope of the function.
    """
    __slots__ = ('_static_vars',)

    def __init__(self, name, body, static_vars, scope):
        self._static_vars = static_vars
        super().__init__(name, (), (), body, scope=scope)

    @property
    def declarations(self):
        """
        Returns the declarations of the variables.

        Returns the declarations of the variables.
        """
        return [Declare(v, static=(v in self._static_vars)) \
                for v in self.scope.variables.values()]

#-------------------------------------------------------------------
#                      Python.h Constants
#-------------------------------------------------------------------

# Python.h object  representing Booleans True and False
Py_True = Variable(PyccelPyObject(), 'Py_True', memory_handling='alias')
Py_False = Variable(PyccelPyObject(), 'Py_False', memory_handling='alias')

# Python.h object representing None
Py_None = Variable(PyccelPyObject(), 'Py_None', memory_handling='alias')

# https://docs.python.org/3/c-api/refcounting.html#c.Py_INCREF
Py_INCREF = FunctionDef(name = 'Py_INCREF',
                        body = [],
                        arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name='o', memory_handling='alias'))],
                        results = [])

# https://docs.python.org/3/c-api/refcounting.html#c.Py_DECREF
Py_DECREF = FunctionDef(name = 'Py_DECREF',
                        body = [],
                        arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name='o', memory_handling='alias'))],
                        results = [])

# https://docs.python.org/3/c-api/type.html#c.PyType_Ready
PyType_Ready = FunctionDef(name = 'PyType_Ready',
                        body = [],
                        arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name='o', memory_handling='alias'))],
                        results = [FunctionDefResult(Variable(NativeInteger(), '_'))])

# https://docs.python.org/3/c-api/sys.html#PySys_GetObject
PySys_GetObject = FunctionDef(name = 'PySys_GetObject',
                        body = [],
                        arguments = [FunctionDefArgument(Variable(dtype=NativeString(), name='_'))],
                        results = [FunctionDefResult(Variable(dtype=PyccelPyObject(), name='o', memory_handling='alias'))])

# https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_FromString
PyUnicode_FromString = FunctionDef(name = 'PyUnicode_FromString',
                        body = [],
                        arguments = [FunctionDefArgument(Variable(dtype=NativeString(), name='_'))],
                        results = [FunctionDefResult(Variable(dtype=PyccelPyObject(), name='o', memory_handling='alias'))])

# https://docs.python.org/3/c-api/list.html#c.PyList_GetItem
PyList_GetItem = FunctionDef(name = 'PyList_GetItem',
                        body = [],
                        arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name='l', memory_handling='alias')),
                                     FunctionDefArgument(Variable(dtype=NativeInteger(), name='i', precision=-2))],
                        results = [FunctionDefResult(Variable(dtype=PyccelPyObject(), name='o', memory_handling='alias'))])

# https://docs.python.org/3/c-api/list.html#c.PyList_SetItem
PyList_SetItem = FunctionDef(name = 'PyList_SetItem',
                        body = [],
                        arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name='l', memory_handling='alias')),
                                     FunctionDefArgument(Variable(dtype=NativeInteger(), name='i', precision=-2)),
                                     FunctionDefArgument(Variable(dtype=PyccelPyObject(), name='new_item', memory_handling='alias'))],
                        results = [])

#-------------------------------------------------------------------

#using the documentation of PyArg_ParseTuple() and Py_BuildValue https://docs.python.org/3/c-api/arg.html
pytype_parse_registry = {
    (NativeInteger(), 4)       : 'i',
    (NativeInteger(), 8)       : 'l',
    (NativeInteger(), 2)       : 'h',
    (NativeInteger(), 1)       : 'b',
    (NativeFloat(), 8)         : 'd',
    (NativeFloat(), 4)         : 'f',
    (NativeComplex(), 4)       : 'O',
    (NativeComplex(), 8)       : 'O',
    (NativeBool(), -1)         : 'p',
    (NativeString(), 0)        : 's',
    (PyccelPyObject(), 0)      : 'O',
    }

#-------------------------------------------------------------------
#                      cwrapper.h functions
#-------------------------------------------------------------------

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
py_to_c_registry = {
    (NativeBool(), -1)     : 'PyBool_to_Bool',
    (NativeInteger(), 1)   : 'PyInt8_to_Int8',
    (NativeInteger(), 2)   : 'PyInt16_to_Int16',
    (NativeInteger(), 4)   : 'PyInt32_to_Int32',
    (NativeInteger(), 8)   : 'PyInt64_to_Int64',
    (NativeFloat(), 4)     : 'PyFloat_to_Float',
    (NativeFloat(), 8)     : 'PyDouble_to_Double',
    (NativeComplex(), 4)   : 'PyComplex_to_Complex64',
    (NativeComplex(), 8)   : 'PyComplex_to_Complex128'}

def C_to_Python(c_object):
    """
    Create a FunctionDef responsible for casting scalar C results to Python.

    Creates a FunctionDef node which contains all the code necessary
    for casting a C object, whose characteristics match that of the object
    passed as an argument, to a PythonObject which can be used in Python code.

    Parameters
    ----------
    c_object : Variable
        The variable needed for the generation of the cast_function.

    Returns
    -------
    FunctionDef
        The function which casts the C object to Python.
    """
    if c_object.rank != 0:
        cast_function = 'ndarray_to_pyarray'
        memory_handling = 'stack'
    else:
        try :
            cast_function = c_to_py_registry[(c_object.dtype, c_object.precision)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=c_object.dtype,severity='fatal')
        memory_handling = 'alias'

    cast_func = FunctionDef(name = cast_function,
                       body      = [],
                       arguments = [FunctionDefArgument(c_object.clone('v', is_argument = True, memory_handling=memory_handling, new_class = Variable))],
                       results   = [FunctionDefResult(Variable(dtype=PyccelPyObject(), name = 'o', memory_handling='alias'))])

    return cast_func

# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
c_to_py_registry = {
    (NativeBool(), -1)     : 'Bool_to_PyBool',
    (NativeInteger(), -1)  : 'Int'+str(default_precision[NativeInteger()]*8)+'_to_PyLong',
    (NativeInteger(), 1)   : 'Int8_to_NumpyLong',
    (NativeInteger(), 2)   : 'Int16_to_NumpyLong',
    (NativeInteger(), 4)   : 'Int32_to_NumpyLong',
    (NativeInteger(), 8)   : 'Int64_to_NumpyLong',
    (NativeFloat(), 4)     : 'Float_to_NumpyDouble',
    (NativeFloat(), 8)     : 'Double_to_NumpyDouble',
    (NativeFloat(), -1)    : 'Double_to_PyDouble',
    (NativeComplex(), 4)   : 'Complex64_to_NumpyComplex',
    (NativeComplex(), 8)   : 'Complex128_to_NumpyComplex',
    (NativeComplex(), -1)  : 'Complex128_to_PyComplex'}


#-------------------------------------------------------------------
#              errors and check functions
#-------------------------------------------------------------------

# https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Occurred
PyErr_Occurred = FunctionDef(name      = 'PyErr_Occurred',
                             arguments = [],
                             results   = [FunctionDefResult(Variable(dtype = PyccelPyObject(), name = 'r', memory_handling = 'alias'))],
                             body      = [])

PyErr_SetString = FunctionDef(name = 'PyErr_SetString',
              body      = [],
              arguments = [FunctionDefArgument(Variable(dtype = PyccelPyObject(), name = 'o')),
                           FunctionDefArgument(Variable(dtype = NativeString(), name = 's'))],
              results   = [])

PyNotImplementedError = Variable(PyccelPyObject(), name = 'PyExc_NotImplementedError')
PyTypeError = Variable(PyccelPyObject(), name = 'PyExc_TypeError')
PyAttributeError = Variable(PyccelPyObject(), name = 'PyExc_AttributeError')

PyObject_TypeCheck = FunctionDef(name = 'PyObject_TypeCheck',
            arguments = [FunctionDefArgument(Variable(PyccelPyObject(), 'o', memory_handling = 'alias')),
                         FunctionDefArgument(Variable(PyccelPyClassType(), 'c_type', memory_handling='alias'))],
            results = [FunctionDefResult(Variable(NativeBool(), 'r'))],
            body = [])

PyList_New = FunctionDef(name = 'PyList_New',
                    arguments = [FunctionDefArgument(Variable(NativeInteger(), 'size'), value = LiteralInteger(0))],
                    results = [FunctionDefResult(Variable(PyccelPyObject(), 'r', memory_handling='alias'))],
                    body = [])

PyList_Append = FunctionDef(name = 'PyList_Append',
                    arguments = [FunctionDefArgument(Variable(PyccelPyObject(), 'list', memory_handling='alias')),
                                 FunctionDefArgument(Variable(PyccelPyObject(), 'item', memory_handling='alias'))],
                    results = [FunctionDefResult(Variable(NativeInteger(), 'i', precision=4))],
                    body = [])

PyList_GetItem = FunctionDef(name = 'PyList_GetItem',
                    arguments = [FunctionDefArgument(Variable(PyccelPyObject(), 'list', memory_handling='alias')),
                                 FunctionDefArgument(Variable(NativeInteger(), 'i', precision=8))],
                    results = [FunctionDefResult(Variable(PyccelPyObject(), 'item', memory_handling='alias'))],
                    body = [])

PyList_Size = FunctionDef(name = 'PyList_Size',
                    arguments = [FunctionDefArgument(Variable(PyccelPyObject(), 'list', memory_handling='alias'))],
                    results = [FunctionDefResult(Variable(NativeInteger(), 'i', precision=8))],
                    body = [])


# Functions definitions are defined in pyccel/stdlib/cwrapper/cwrapper.c
check_type_registry = {
    (NativeBool(), -1)     : 'PyIs_Bool',
    (NativeInteger(), -1)  : 'PyIs_NativeInt',
    (NativeInteger(), 1)   : 'PyIs_Int8',
    (NativeInteger(), 2)   : 'PyIs_Int16',
    (NativeInteger(), 4)   : 'PyIs_Int32',
    (NativeInteger(), 8)   : 'PyIs_Int64',
    (NativeFloat(), -1)    : 'PyIs_NativeFloat',
    (NativeFloat(), 4)     : 'PyIs_Float',
    (NativeFloat(), 8)     : 'PyIs_Double',
    (NativeComplex(), -1)  : 'PyIs_NativeComplex',
    (NativeComplex(), 4)   : 'PyIs_Complex64',
    (NativeComplex(), 8)   : 'PyIs_Complex128'}
