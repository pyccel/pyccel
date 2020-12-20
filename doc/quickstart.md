Pyccel

 What is Pyccel?

static compiler for Python 3, using Fortran or C as backend language.
started as small open-source project in 2018 at IPP Garching.
public repository is now hosted on GitHub, freely available for download.

Python’s objects, variables, and garbage collection

 Python is an interpreted language, dynamically typed and garbage-  	       collected.

 Python object:
is created by the Python interpreter when object.__new__() is invoked (e.g. as a result of an expression).
can be either mutable or immutable, but its type never changes.
resides in memory and has a reference count.
is accessed through one or more Python variables.
is destroyed by the garbage collector when its reference count drops to zero.
For more details about Python object, see this.

 Python variable:
is a reference to a Python object in memory.
is created with an assignment operation x = expr:
if the variable x already exists, the interpreted reduces the reference count of its object
a new variable x is created, which references the value of expr.
can be destroyed with the command del x.
For more details about Python variable, see this.

Static typed languages

A language is statically-typed if the type of a variable is known at compile-time instead of at run-time. Common examples of statically-typed languages include Java, C, C++, FORTRAN, Pascal and Scala. See this and this for more details.