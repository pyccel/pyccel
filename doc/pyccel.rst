Dive into Pyccel
================

Typical processing using **Pyccel** can be splitted into 3 main stages:

1. First, we parse the *Python* file or text, and we create an *intermediate representation* (**IR**) that consists of objects described in **pyccel.parser.syntax**
2. Most of all these objects, when it makes sens, implement the property **expr**. This allows to convert the object to one or more objects from **pyccel.ast.core**. All these objects are extension of the **sympy.core.Basic** class. At the end of this stage, the **IR** will be converted into the *Abstract Syntax Tree* (AST). 
3. Using the **Codegen** classes or the user-friendly functions like **fcode**, the **AST** is converted into the target language (for example, *Fortran*)

.. note:: Always remember that **Pyccel** core is based on **sympy**. This can open a very wide range of applications and opportunities, like automatically evaluate the *computational complexity* of your code. 

.. note:: There is an intermediate step between 2 and 3, where one can walk through the AST and modify the code by applying some recursive functions (ex: mpify, openmpfy, ...)

.. tikz:: Overview of a code generation process using Fortran as a backend/target language. 

    \node[draw=black, rectangle, fill=red!40] (py)  
    at (0,0)  {Python};
    \node at (0.9,0) [color=gray,above=3mm,right=0mm] {Parser};

    \draw[black, thick, fill=blue!10] (3,0) circle [radius=0.5cm];
    \node at (3,0) [color=black] {\textsc{IR}};
    \node at (3.8,0) [color=gray,above=3mm,right=0mm,font=\fontsize{10}{10.2}] {\texttt{expr}};
    \node at (3.6,0) [color=gray,below=3mm,right=0mm,font=\fontsize{10}{10.2}] {\textit{property}};

    \draw[black, thick, fill=blue!30] (6,0) circle [radius=0.7cm];
    \node at (6,0) [color=black] {\textsc{AST}};
    \node at (6.7,0) [color=gray,above=3mm,right=0mm] {Codegen};

    \node[draw=black, rectangle, fill=green!20] (f90)  
    at (9.5,0)  {Fortran};

    \draw[->,very thick] (py) --(2.5,0) ;
    \draw[->,very thick] (3.5,0)--(5.3,0) ;
    \draw[->,very thick] (6.7,0)--(f90) ;

Specifications
**************

We follow `Python 3.6.2`_ specifications.

.. _Python 3.6.2: https://docs.python.org/3/reference/grammar.html

**Pyccel** grammar was derived from `ANTLR`_

.. _ANTLR: https://github.com/antlr/grammars-v4/blob/master/python3/Python3.g4

Types
^^^^^

Dynamic *vs* Static typing
__________________________

Since our aim is to generate code in a low-level language, which is in most cases of static typed, we will have to devise an alternative way to construct/find the appropriate type of a given variable. 
This can be done by including the concept of *constructors* or use specific *headers* to assist *Pyccel* in finding/infering the appropriate type.

Let's explain this more precisely; we consider the following code

.. code-block:: python

  n = 5
  x = 2.0 * n

In this example, **n** will be interprated as an **integer** while **x** will be a **double** number, so everything is fine.

The problem arises when using a function, like in the following example

.. code-block:: python

  def f(n):
    x = 2.0 * n
    return x

  n = 5
  x = f(n)

Now the question is what would be the signature of **f** if there was no call to it in the previous script?

To overcome this ambiguity, we rewrite our function as

.. code-block:: python

  #$ header f(int)
  def f(n):
    x = 2.0 * n
    return x

Such an implementation still makes sens inside *Python*. As you can see, the type of *x* is infered by analysing our *expressions*.

Built-in Types
______________

The following are the built-in types in **Pyccel**::

  int, float, double, complex, array

.. todo:: boolean and string expressions not tested yet

Built-in Functions
^^^^^^^^^^^^^^^^^^

Mathematical functions
______________________

Mathematical functions are ::

   'transpose'
   'len'
   'log'
   'exp'
   'cos'
   'sin'
   'sqrt'
   'abs'
   'sign'
   'csc'
   'sec'
   'tan'
   'cot'
   'asin'
   'acsc'
   'acos'
   'asec'
   'atan'
   'acot'
   'atan2'
   'factorial'
   'ceil'
   'pow'
   'dot'
   'min'
   'max'

.. todo:: add transpose

Built-in Constants
^^^^^^^^^^^^^^^^^^

Mathematical constants
______________________

The following constants are available::

   'pi'

Data Types
^^^^^^^^^^

.. todo:: strctures and classe are not yet available

File and Directory Access
^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo:: file and directory access is not yet available 

Importing modules
^^^^^^^^^^^^^^^^^

Importing modules is not allowed. However, you can import objects that are defined inside a given module. 

Iterators
^^^^^^^^^

There are 3 kind of iterators:

1. One that performs on groups (MPI)
   - for the moment, only **MPI_Tensor** is available

2. One that performs on teams (OpenMP, OpenACC)
   - this can be done using **prange** inside a **parallel** block

3. One that performs on atoms (sequential)


Documentation
*************

Parser
^^^^^^

.. automodule:: pyccel.parser.parser
   :members:

Syntax (IR)
^^^^^^^^^^^

.. inheritance-diagram:: pyccel.parser.syntax.core

.. automodule:: pyccel.parser.syntax.basic
   :members:

.. automodule:: pyccel.parser.syntax.core
   :members:

OpenMP
______

.. inheritance-diagram:: pyccel.parser.syntax.openmp

.. automodule:: pyccel.parser.syntax.openmp
   :members:

Codegen
^^^^^^^

.. automodule:: pyccel.codegen.codegen
   :members:

Printing
________

.. automodule:: pyccel.codegen.printing.codeprinter
   :members:

.. automodule:: pyccel.codegen.printing.fcode
   :members:

.. automodule:: pyccel.codegen.printing.ccode
   :members:

.. automodule:: pyccel.codegen.printing.luacode
   :members:

Abstract Syntax Tree (AST)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.

.. inheritance-diagram:: pyccel.ast.core

.. automodule:: pyccel.ast.core
   :members:

Parallel Computation
____________________

.. inheritance-diagram:: pyccel.ast.parallel.communicator

.. inheritance-diagram:: pyccel.ast.parallel.group

.. automodule:: pyccel.ast.parallel.basic
   :members:

.. automodule:: pyccel.ast.parallel.communicator
   :members:

.. automodule:: pyccel.ast.parallel.group
   :members:

**MPI**

.. inheritance-diagram:: pyccel.ast.parallel.mpi

.. automodule:: pyccel.ast.parallel.mpi
   :members:

**OpenMP**

.. inheritance-diagram:: pyccel.ast.parallel.openmp

.. automodule:: pyccel.ast.parallel.openmp
   :members:

**OpenACC**

.. .. inheritance-diagram:: pyccel.openmp.syntax
.. 
.. .. automodule:: pyccel.openmp.syntax
..    :members:


Calculus
^^^^^^^^

.. automodule:: pyccel.calculus.finite_differences
   :members:

Complexity
^^^^^^^^^^

Arithmetic
__________

.. automodule:: pyccel.complexity.arithmetic
   :members:

Memory
______

.. automodule:: pyccel.complexity.memory
   :members:

Basic
_____

.. automodule:: pyccel.complexity.basic
   :members:

