! -------------------------------------------------------------------------------------- !
! This file is part of Pyccel which is released under MIT License. See the LICENSE file  !
! or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. !
! -------------------------------------------------------------------------------------- !

module pyc_tools_f90

  use, intrinsic :: ISO_C_Binding, only : b1 => C_BOOL, &
         f32 => C_FLOAT, & 
         f64 => C_DOUBLE, &
         c64 => C_DOUBLE_COMPLEX, &
         c32 => C_FLOAT_COMPLEX

  implicit none

  public :: complex_comparison

  private

  interface complex_comparison
    module procedure complex_comparison_4
    module procedure complex_comparison_8
  end interface complex_comparison

  contains

  function complex_comparison_4(x, y) result(c)
    complex(c32) :: x
    complex(c32) :: y
    logical(b1) :: c
    real(f32) :: real_x
    real(f32) :: real_y

    real_x = real(x)
    real_y = real(y)

    c = merge(real_x < real_y, aimag(x) < aimag(y), real_x /= real_y)

  end function complex_comparison_4

  function complex_comparison_8(x, y) result(c)
    complex(c64) :: x
    complex(c64) :: y
    logical(b1) :: c
    real(f64) :: real_x
    real(f64) :: real_y

    real_x = real(x)
    real_y = real(y)

    c = merge(real_x < real_y, aimag(x) < aimag(y), real_x /= real_y)

  end function complex_comparison_8

end module pyc_tools_f90
