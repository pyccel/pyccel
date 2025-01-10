module bind_c_tools

  use, intrinsic :: ISO_C_Binding, only : c32 => C_FLOAT_COMPLEX , i8 => &
        C_INT8_T , i64 => C_INT64_T , b1 => C_BOOL , i16 => C_INT16_T , &
        f64 => C_DOUBLE , c64 => C_DOUBLE_COMPLEX , i32 => C_INT32_T , &
        f32 => C_FLOAT

  implicit none

  contains

  !........................................
  subroutine free_array_i8(a, n) bind(c)

    implicit none

    integer(i64), intent(in)   :: n
    integer(i8), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_i8
  !........................................

  !........................................
  subroutine free_array_i16(a, n) bind(c)

    implicit none

    integer(i64), intent(in)    :: n
    integer(i16), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_i16
  !........................................

  !........................................
  subroutine free_array_i32(a, n) bind(c)

    implicit none

    integer(i64), intent(in)    :: n
    integer(i32), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_i32
  !........................................

  !........................................
  subroutine free_array_i64(a, n) bind(c)

    implicit none

    integer(i64), intent(in)    :: n
    integer(i64), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_i64
  !........................................

  !........................................
  subroutine free_array_f32(a, n) bind(c)

    implicit none

    integer(i64), intent(in) :: n
    real(f32), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_f32
  !........................................

  !........................................
  subroutine free_array_f64(a, n) bind(c)

    implicit none

    integer(i64), intent(in) :: n
    real(f64), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_f64
  !........................................

  !........................................
  subroutine free_array_c32(a, n) bind(c)

    implicit none

    integer(i64), intent(in)    :: n
    complex(c32), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_c32
  !........................................

  !........................................
  subroutine free_array_c64(a, n) bind(c)

    implicit none

    integer(i64), intent(in)    :: n
    complex(c64), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_c64
  !........................................

  !........................................
  subroutine free_array_b1(a, n) bind(c)

    implicit none

    integer(i64), intent(in)   :: n
    logical(b1), intent(inout) :: a(n)

    deallocate(a)

  end subroutine free_array_b1
  !........................................

end module bind_c_tools
