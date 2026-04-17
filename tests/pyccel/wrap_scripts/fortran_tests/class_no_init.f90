module class_no_init

  use, intrinsic :: ISO_C_Binding, only : b1 => C_BOOL , i64 => &
        C_INT64_T

  implicit none

  public :: MethodCheck

  private

  type :: MethodCheck
    integer(i64) :: my_value

    contains
    procedure :: stash_value => methodcheck_stash_value
  end type MethodCheck

  contains


  !........................................

  subroutine methodcheck_stash_value(self, val) 

    implicit none

    class(MethodCheck), intent(inout) :: self
    integer(i64), value :: val

    self%my_value = val

  end subroutine methodcheck_stash_value

  !........................................

end module class_no_init
