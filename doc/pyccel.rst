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
    \node at (1,0) [color=gray,above=3mm,right=0mm] {Parser};

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


Documentation
*************

Parser
^^^^^^

.. automodule:: pyccel.parser
   :members:

Syntax
^^^^^^

.. inheritance-diagram:: pyccel.syntax

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

Patterns
^^^^^^^^

.. automodule:: pyccel.patterns.utilities
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

.. automodule:: pyccel.complexity.operation
   :members:

Memory
______

.. automodule:: pyccel.complexity.memory
   :members:

