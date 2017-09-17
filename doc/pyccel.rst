Dive into Pyccel
================

Typical processing using **Pyccel** can be splitted into 3 main stages:

1. First, we parse the *Python* file or text, and we create an *intermediate representation* (**IR**) that consists of objects described in **pyccel.syntax**
2. Most of all these objects, when it makes sens, implement the property **expr**. This allows to convert the object to one or more objects from **pyccel.types.ast**. All these objects are extension of the **sympy.core.Basic** class. At the end of this stage, the **IR** will be converted into the *Abstract Syntax Tree* (AST). 
3. Using the **Codegen** classes or the user-friendly functions like **fcode**, the **AST** is converted into the target language (for example, *Fortran*)

.. note:: Always remember that **Pyccel** core is based on **sympy**. This can open a very wide range of applications and opportunities, like automatically evaluate the *computational complexity* of your code. 

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

Since our aim is to generate code in a low-level language, which is in most cases of static typed, we will have to *enforce* the given *Python* variables to mimic, in some sens, static types. This can be done by including the concept of *constructors* that can be implemented easily in order to still run your code inside *Python* .

Let's explain this more precisely; we consider the following code

.. code-block:: python

  n = 5
  x = 2.0 * n

In this example, **n** will be interprated as an **integer** while **x** will be a **float** number, so everything is fine.

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

  def f(n):
    n = int()
    x = 2.0 * n
    return x

Such an implementation still makes sens inside *Python*.

Built-in Types
______________

The following are the built-in types in **Pyccel**::

  int, float, double, complex, array, matrix, stencil

.. todo:: boolean expressions not tested yet

Built-in Functions
^^^^^^^^^^^^^^^^^^

Mathematical functions
______________________

Functions of one argument are ::

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

Functions of two arguments are ::

   'pow'
   'rational'
   'dot'
   'min'
   'max'

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

.. todo:: iterators are not yet available 

Parallel computing
^^^^^^^^^^^^^^^^^^

OpenMP
______

OpenACC
_______

MPI
___


Documentation
*************

Parser
^^^^^^

.. automodule:: pyccel.parser
   :members:

Syntax (IR)
^^^^^^^^^^^

.. inheritance-diagram:: pyccel.syntax

.. automodule:: pyccel.core.syntax
   :members:

.. automodule:: pyccel.syntax
   :members:

Codegen
^^^^^^^

.. automodule:: pyccel.codegen
   :members:

Abstract Syntax Tree (AST)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.

.. inheritance-diagram:: pyccel.types.ast

.. automodule:: pyccel.types.ast
   :members:

Printers
^^^^^^^^

.. automodule:: pyccel.printers.codeprinter
   :members:

.. automodule:: pyccel.printers.fcode
   :members:

.. automodule:: pyccel.printers.ccode
   :members:

.. automodule:: pyccel.printers.luacode
   :members:

Imports
^^^^^^^

.. automodule:: pyccel.imports.syntax
   :members:

.. automodule:: pyccel.imports.utilities
   :members:

Calculus
^^^^^^^^

.. automodule:: pyccel.calculus.finite_differences
   :members:

OpenMP
^^^^^^

.. inheritance-diagram:: pyccel.openmp.syntax

.. automodule:: pyccel.openmp.syntax
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

