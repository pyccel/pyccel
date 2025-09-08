module class_overloaded_methods
  use, intrinsic :: iso_c_binding
  implicit none

  type :: Adder
  contains
    procedure :: create => create_adder
    procedure :: free => free_adder
    generic, public :: add => adder_add_int, adder_add_real
    procedure :: adder_add_int, adder_add_real
  end type Adder

contains

  subroutine create_adder(a)
    class(Adder), intent(inout) :: a
  end subroutine create_adder

  subroutine free_adder(a)
    class(Adder), intent(inout) :: a
  end subroutine free_adder

  function adder_add_int(this, x, y) result(z)
    class(Adder), intent(in) :: this
    integer(c_int64_t), intent(in) :: x, y
    integer(c_int64_t) :: z
    z = x + y
  end function adder_add_int

  function adder_add_real(this, x, y) result(z)
    class(Adder), intent(in) :: this
    real(c_double), intent(in) :: x, y
    real(c_double) :: z
    z = x + y
  end function adder_add_real

end module class_overloaded_methods

