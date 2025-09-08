module square_mod

  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T

  implicit none

contains

  function square(x) result(y)
    integer(i64), intent(in) :: x
    integer(i64) :: y
    y = x * x
  end function square
end module square_mod

