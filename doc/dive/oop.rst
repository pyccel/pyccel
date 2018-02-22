Oriented Object Programming
***************************

Let's take this example; we consider the following code

.. code-block:: python
  
  from pyccel.ast.core import Variable, Assign
  from pyccel.ast.core import ClassDef, FunctionDef, Module
  from pyccel import fcode
  x = Variable('double', 'x') 
  y = Variable('double', 'y')
  z = Variable('double', 'z')
  t = Variable('double', 't')
  a = Variable('double', 'a')
  b = Variable('double', 'b')
  body = [Assign(y,x+a)]
  translate = FunctionDef('translate', [x,y,a,b], [z,t], body)
  attributs   = [x,y]
  methods     = [translate]
  Point = ClassDef('Point', attributs, methods) 
  incr = FunctionDef('incr', [x], [y], [Assign(y,x+1)])
  decr = FunctionDef('decr', [x], [y], [Assign(y,x-1)])
  module=Module('my_module', [], [incr, decr], [Point])
  code=fcode(module)
  print(code)

In this example, we created a Class *Point* that represent a point in 2d  with two functions *incr* and *decr*
The results in Fortran looks like 

.. code-block:: fortran
  
  module mod_my_module

  implicit none

  type, public :: Point
    real(kind=8) :: x
    real(kind=8) :: y
    contains
    procedure :: translate => Point_translate

  end type Point
  contains

  ! ........................................ 
  real(kind=8) function incr(x)  result(y)
  implicit none
  real(kind=8), intent(in)  :: x

  y = 1 + x

  end function
  ! ........................................ 

  ! ........................................ 
  real(kind=8) function decr(x)  result(y)
  implicit none
  real(kind=8), intent(in)  :: x

  y = -1 + x

  end function
  ! ........................................ 


  ! ........................................ 
  subroutine translate(x, y, a, b, z, t)
  implicit none
  real(kind=8), intent(in)  :: a
  real(kind=8), intent(in)  :: b
  real(kind=8), intent(out)  :: t
  real(kind=8), intent(inout)  :: y
  real(kind=8), intent(in)  :: x
  real(kind=8), intent(out)  :: z

  y = a + x

  end subroutine
  ! ........................................ 
  
  end module

Notice that in Fortran the class must be in Module that's why the class and the functions where put in a module
in the Python code.

