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

  \node[draw=black, rectangle, fill=red!20, font=\fontsize{10}{10.2}] (lint)  
  at (0,2)  {Pylint};

  \node at (0.9,0) [color=gray,above=3mm,right=0mm] {Parser};

  \draw[black, thick, fill=blue!10] (3,0) circle [radius=0.5cm];
  \node at (3,0) [color=black] {\textsc{IR}};
  \node at (3.8,0) [color=gray, above=3mm, right=0mm, font=\fontsize{10}{10.2}] {\texttt{expr}};
  \node at (3.6,0) [color=gray, below=3mm, right=0mm, font=\fontsize{10}{10.2}] {\textit{property}};

  \draw[black, thick, fill=blue!30] (6,0) circle [radius=0.7cm];
  \node at (6,0) [color=black] {\textsc{AST}};
  \node at (6.7,0) [color=gray,above=3mm,right=0mm] {Codegen};

  \node[draw=black, rectangle, fill=green!20] (f90)  
  at (9.5,0)  {Fortran};

  \draw[->,very thick] (py) --(2.5,0) ;
  \draw[->,very thick] (3.5,0)--(5.3,0) ;
  \draw[->,very thick] (6.7,0)--(f90) ;
  \draw[->,very thick] (py) --(lint) ;

The idea behind **Pyccel** is to use the available tools for **Python**, without having to implement everything as it is usually done for every new language. The aim of using such high level language is to ensure a user-friendly framework for developping massively parallel codes, without having to work in a hostile environment such as *Fortran* or an obscur language like *c++*. Most of all, compared to other *DSLs* for HPC, all elements and parallel paradigms of the language are exposed.

Among the very nice tools for *Python* developpers, Pylint_ is used for **static** checking. This allows us to avoid writing a **linter** tool or having to implement an advanced tool to handle **errors**. Following the **K.I.S.S** paradigm, we want to keep it *stupid* and *simple*, hence if you are getting errors with *Pylint*, do not expect *Pyccel* to run!! We assume that you are capable of writing a **valid** *Python* code. If not, then try first to learn *Python* before trying to do fancy things!

.. _Pylint: https://www.pylint.org/



.. toctree::
   :maxdepth: 1 
   :caption: Contents:
  
   introduction
   lexsyn
   expressions
   flow
   domains
   functions
   modules
   oop
   documentation
   legacy
   io
   stdlib
   fp
