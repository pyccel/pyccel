# AST nodes

While translating from Python to the target language, Pyccel needs to store all of the concepts in the code in Python objects which fully describe them. These objects are called AST nodes. AST stands for Abstract Syntax Tree.

All objects in the Abstract Syntax Tree inherit from the class `pyccel.ast.basic.PyccelAstNode`. This class serves 2 roles. Firstly it provides a super class from which all our AST nodes can inherit which makes them easy to identify. Secondly it provides functionalities common to all AST nodes. For example it provides the `ast` property which allows the original code parsed by Python's `ast` module to be stored in the class. This object is important in order to report neat errors for code that cannot be handled by Pyccel. It also contains functions involving the relations between the nodes. These are explained in the section [Constructing a tree](#constructing-a-tree).

The inheritance tree for a Python AST node is often more complicated than directly inheriting from `PyccelAstNode`. In particular there are two classes which you will see in the inheritance throughout the code. These classes are `TypedAstNode` and `PyccelFunction`. These classes are explained in more detail below.

## Typed AST Node

The class `TypedAstNode` is a super class. This class should never be used directly but provides functionalities which are common to certain AST objects. These AST nodes are those which describe objects which take up space in memory in a running program. For example a Variable requires space in memory, as does the result of a function call or an arithmetic operation, however a loop or a module does not require runtime memory to store the concept. Objects which require memory must therefore contain all information necessary to declare them in the generated code. A `TypedAstNode` therefore exposes the following properties:

-   `dtype`
-   `rank`
-   `shape`
-   `order`
-   `class_type`

The contents of these types are explained in more detail below.

The class `TypedAstNode` also contains static class methods. The static class method is used for type deductions when [parsing type annotations](./type_inference.md). In type annotations we do not generally have an instance of a class, however we can get access to the class itself. The available class methods are:

-   `static_type`
-   `static_rank`
-   `static_order`

For instance let us consider the following type annotation:

```python
a : int
```

When we visit `int` in the [semantic stage](./semantic_stage.md) the `SemanticParser` will return the class `PythonInt`. This is usually used as a function (e.g to cast a variable), however here we use it to deduce the type. Following the [development conventions](./development_conventions.md#class-variables-vs-instance-variables) any attributes which will remain constant over all instances of a class should be stored in static class attributes. This means that they can be accessed via these static methods. Returning to our example, a call to the function `int` always returns a scalar object with the built-in `float` type. This means that all the properties of a `TypedAstNode` can be defined without having an instance of this class. These properties cannot be defined statically for all nodes (e.g. it would not be possible for `PyccelAdd`), however generally they can be defined for the nodes which can be used in type annotations.

### Class type

The `class_type` property represents the type reported by Python when you call the built-in function `type`. This property should return an object which inherits from `pyccel.ast.datatypes.PyccelType`.

### Datatype

Some types in Python are containers which contain elements of other types. This is the case for NumPy arrays, tuples, lists, etc. In this case, the class type does not provide enough information to write the declaration in the low-level target language. Additionally a data type is required. The data type is the type of an element of the container and can be accessed via the `dtype` property. As for the class type the object returned by this property should inherit from `pyccel.ast.datatypes.PyccelType`. If the class type is not a container then the class type and the data type will be the same, otherwise the class type will inherit from `pyccel.ast.datatypes.ContainerType` and the data type will inherit from `pyccel.ast.datatypes.FixedSizeType`.

A `FixedSizeType` represents a built-in scalar datatype which can be represented in memory. E.g. `int32`, `int64`. It is characterised by a primitive type which describes the category of datatype (integer, floating point, etc) and a precision. The precision is related to the number of bytes that the datatype takes up in memory (e.g. `float64` has precision = 8 as it takes up 8 bytes, `complex128` has precision = 8 as it is comprised of two `float64` objects). The precision is equivalent to the `kind` parameter in Fortran.

### Rank

The rank of an array is the number of dimensions in the array. This object is an integer.

### Shape

The shape of an array indicates the number of elements in each dimension of the array. This property is a tuple. The tuple contains `TypedAstNode`s. If the shape is known then these objects will be `LiteralInteger`s. If the shape is unknown then these objects will be `PyccelArrayShapeElement`s.

The length of the shape should be equal to the number of indices necessary to print this object in a low-level language. Generally this is equal to the `container_rank`, however homogeneous tuples are an exception to this rule as they are saved in multi-dimensional arrays.

### Order

The order indicates how an array is laid out in memory. This can either be row-major (C-style) ordering or column-major (Fortran-style) ordering. For more information about this, please see the [dedicated documentation](./order_docs.md).

### Static Type

The static type is the class type that would be assigned to an object created using an instance of this class as a type annotation.

## Pyccel Internal Function

The class `pyccel.ast.internals.PyccelFunction` is a super class. This class should never be used directly but provides functionalities which are common to certain AST objects. These AST nodes are those which describe functions which are supported by Pyccel. For example it is used for functions from the `math` library, the `cmath` library, the `numpy` library, etc. `PyccelFunction` inherits from `TypedAstNode`. The type information for the sub-class describes the type of the result of the function.

Instances of the subclasses of `PyccelFunction` are created from a `FunctionCall` object resulting from the syntactic stage. The `PyccelFunction` objects are initialised by passing the contents of the `FunctionCallArgument`s used in the `FunctionCall` to the constructor. The actual arguments may be either positional or keyword arguments, but the constructor of `PyccelFunction` only accepts positional arguments through the variadic `*args`. It is therefore important that any subclasses of `PyccelFunction` provide their own `__init__` with the correct function signature. I.e. a constructor whose positional and keyword arguments match the names of the positional and keyword arguments of the Python function being described.

The constructor of `PyccelFunction` takes a tuple of arguments passed to the function. The arguments can be passed through in the constructor of the sub-class to let `PyccelFunction` take care of saving the arguments, or the arguments can be saved with more useful names inside the class. However care must be taken to ensure that the argument is not saved inside the class twice. Additionally if the arguments are saved inside the sub-class then the `_attribute_nodes` static class attribute must be correctly set to ensure the tree is correctly constructed (see below). As `PyccelFunction` implements `_attribute_nodes` these classes are some of the only ones which will not raise an error during the pull request tests if this information is missing.

## Constructing a tree

The `PyccelAstNode` class is used to construct a tree of AST nodes. Each node is aware of its users and attributes. As attributes are saved inside the class instance the constructor of `PyccelAstNode` ensures that these objects are informed of their new user. Constructing a tree in this way allows a few useful utilities to be provided. The most useful of these are:

-   `pyccel.ast.basic.PyccelAstNode.get_user_nodes`
-   `pyccel.ast.basic.PyccelAstNode.get_attribute_nodes`
-   `pyccel.ast.basic.PyccelAstNode.substitute`
-   `pyccel.ast.basic.PyccelAstNode.get_direct_user_nodes`

See their documentation for more details of the use cases for each of these.

Most of the time developers will not need to worry about this, however there are two major exceptions. Firstly, in order for the constructor of `PyccelAstNode` to correctly set the attributes and users of the instance the static class attribute `_attribute_nodes` must be defined. This variable is a tuple of strings. The tuple must contain the name of any `PyccelAstNode` objects stored in the class. In most cases the Pyccel linting test in the [pull request tests](./review_process.md) will flag the missing tuple but it may be missed if the class inherits from another object which already implements this variable.

Secondly extreme care must be taken when adding or removing objects from other AST objects. It is for this reason that the constructor of `PyccelAstNode` converts any lists into tuples. As such they are harder to modify and should encourage the developer to think about why Pyccel is discouraging this behaviour. Should you need to add an object to your node you must call `set_current_user_node` on the new attribute and pass the instance of your node. Similarly if you remove an object from your node you must call `remove_user_node` on the new attribute and pass the instance of your node. If the attribute is no longer used then it will inform its attributes that they are no longer in use. This destroys the existing tree of the node, but cleans up the global tree so that any leaves which can have multiple users (e.g. Variables) only show the objects which are actually using the object at a given time. If this is not desirable (e.g. because the node is removed temporarily) then the argument `invalidate` should be set to `False`.
