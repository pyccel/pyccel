Abstract
========

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1. Convert a *Python* code into a low-level target language (Fortran or C)

2. Accelerate a selected *Python* code, by using a pre-process, where the original code is translated into a low-level language, then compiled using **f2py** for example.

In order to achieve these tasks, in **Pyccel** we deal with the following points:

a. Implement a new *Python* parser (we do not need to cover all *Python* grammar)

b. Enrich *Python* with new statments to provide multi-threading (although some of them already exist) at the target level

c. Extends the concepts presented in **sympy** allowing for automatic code generation.  

Important remarks
*****************

1. Because of the nature of *Python*, we will have to take some conventions. We refer to the corresponding section for more details.

2. *Python* is cool because there are many useful libraries (*numpy*, *scipy* etc). This is in fact a problem if one tries to convert a *Python* code to a *Fortran* one for example. For instance, a function like **zeros** does not have an equivalent in *Fortran*. One can think of it like

  .. code-block:: fortran

    ! generated fortran code
    real, allocatable :: a(:)
    ...
    allocate(a(64))
    a = 0.0 

  However, we can't apply this solution to a function like **linspace** or **meshgrid**. We therefor will need to have their equivalent in *Fortran*. We can either write them manually, or generate them automaticaly using *Pyccel* too! That's funny, but it is very important, as you can even create the most fundamental and low level functions/subroutines of your code, for a given architecture. Therefor, the philosophy of *Pyccel* is to provide *Python* implementation of the considered functions. These functions will be generated into *Fortran* and may be available through an *import*. 
  For these reasons it will be recommend to use the following imports

  .. code-block:: python

    from pyccel.numpy import zeros, ones

  rather than

  .. code-block:: python

    from numpy import zeros, ones


.. tikz:: Overview of the Assembler object. In dashed the procedure pointers that act on the outputs. 

    \node[draw=black, rectangle, fill=red!40] (py)  
    at (0,0)  {Python};
    \node at (1,0) [color=gray,above=3mm,right=0mm] {Parser};

    \draw[black, thick, fill=blue!10] (3,0) circle [radius=0.5cm];
    \node at (3,0) [color=black] {\textsc{IR}};

    \draw[black, thick, fill=blue!30] (6,0) circle [radius=0.7cm];
    \node at (6,0) [color=black] {\textsc{AST}};
    \node at (6.7,0) [color=gray,above=3mm,right=0mm] {Codegen};

    \node[draw=black, rectangle, fill=green!20] (f90)  
    at (9.5,0)  {Fortran};

    \draw[->,very thick] (py) --(2.5,0) ;
    \draw[->,very thick] (3.5,0)--(5.3,0) ;
    \draw[->,very thick] (6.7,0)--(f90) ;

