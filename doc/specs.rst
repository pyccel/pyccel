Specifications
==============

Conventions
***********

beta version
************

1. *for* statement

2. *if/elif/else* statement 
   
3. *while* statement 
   
4. *import* statement. We will not treat *dotted* imports

5. *functions* with one or multiple returns

6. Code generation for *modules* and *programs* using **only** *Fortran*

7. Boolean expressions for conditional statements

8. arithmetic expressions

9. symbolic expressions and their associated operators will be treated in the next version.

10. static typing using *constructors*

11. arrays with *integer/variable* or list of *integers/variables* as shape

12. memory allocation is done using the builtin functions *zeros*, *ones*

13. *OpenMP* manually using the pragma *#@ omp* 

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

