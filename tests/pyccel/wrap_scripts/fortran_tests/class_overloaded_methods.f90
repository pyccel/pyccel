module class_overloaded_methods
  use, intrinsic :: iso_c_binding, only : C_INT64_T, C_DOUBLE
  implicit none

  type :: Adder
  contains
    procedure :: init => init_adder
    procedure :: free => free_adder
    generic, public :: add => adder_add_int, adder_add_real
    procedure :: adder_add_int, adder_add_real
  end type Adder

contains

  subroutine init_adder(a)
    class(Adder), intent(inout) :: a
  end subroutine init_adder

  subroutine free_adder(a)
    class(Adder), intent(inout) :: a
  end subroutine free_adder

  function adder_add_int(this, x, y) result(z)
    class(Adder), intent(in) :: this
    integer(C_INT64_T), intent(in) :: x, y
    integer(C_INT64_T) :: z
    z = x + y
  end function adder_add_int

  function adder_add_real(this, x, y) result(z)
    class(Adder), intent(in) :: this
    real(C_DOUBLE), intent(in) :: x, y
    real(C_DOUBLE) :: z
    z = x + y
  end function adder_add_real

end module class_overloaded_methods

