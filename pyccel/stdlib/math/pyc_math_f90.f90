! -------------------------------------------------------------------------------------- !
! This file is part of Pyccel which is released under MIT License. See the LICENSE file  !
! or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. !
! -------------------------------------------------------------------------------------- !

module pyc_math_f90

  use, intrinsic :: ISO_C_Binding, only : i32 => C_INT32_T, &
         i64 => C_INT64_T, &
         f32 => C_FLOAT, & 
         f64 => C_DOUBLE, &
         c64 => C_DOUBLE_COMPLEX, &
         c32 => C_FLOAT_COMPLEX

implicit none

real(f64), parameter, private :: pi = 4.0_f64 * DATAN(1.0_f64)

interface pyc_gcd
    module procedure pyc_gcd_4
    module procedure pyc_gcd_8
end interface

interface pyc_factorial
    module procedure pyc_factorial_4
    module procedure pyc_factorial_8
end interface pyc_factorial

interface pyc_lcm
    module procedure pyc_lcm_4
    module procedure pyc_lcm_8
end interface pyc_lcm

interface amax
    module procedure amax_4
    module procedure amax_8
end interface

interface amin
    module procedure amin_4
    module procedure amin_8
end interface

interface csgn
    module procedure numpy_v1_sign_c32
    module procedure numpy_v1_sign_c64
end interface csgn

interface csign
    module procedure numpy_v2_sign_c32
    module procedure numpy_v2_sign_c64
end interface csign

contains

! Implementation of math factorial function
pure function pyc_factorial_4(x) result(fx) ! integers with precision 4

    implicit none

    integer(i32), value      :: x
    integer(i32)             :: i
    integer(i32)             :: fx

    fx = 1_i32
    do i = 2_i32, x
        fx = fx * i
    enddo
    return

end function pyc_factorial_4

pure function pyc_factorial_8(x) result(fx) ! integers with precision 8

    implicit none

    integer(i64), value       :: x
    integer(i64)              :: fx
    integer(i64)              :: i

    fx = 1_i64
    do i = 2_i64, x
        fx = fx * i
    enddo

end function pyc_factorial_8

! Implementation of math gcd function
pure function pyc_gcd_4(a, b) result(gcd) ! integers with precision 4

    implicit none

    integer(i32), value      :: a, b
    integer(i32)             :: x, y
    integer(i32)             :: gcd

    x = MERGE(a, -a, a > 0)
    y = MERGE(b, -b, b > 0)
    do while (y > 0)
        x = MOD(x, y)
        x = IEOR(x, y)
        y = IEOR(y, x)
        x = IEOR(x, y)
    enddo
    gcd = x
    return

end function pyc_gcd_4

pure function pyc_gcd_8(a, b) result(gcd) ! integers with precision 8

    implicit none

    integer(i64), value       :: a, b
    integer(i64)              :: x, y
    integer(i64)              :: gcd

    x = MERGE(a, -a, a > 0)
    y = MERGE(b, -b, b > 0)
    do while (y > 0)
        x = MOD(x, y)
        x = IEOR(x, y)
        y = IEOR(y, x)
        x = IEOR(x, y)
    enddo
    gcd = x
    return

end function pyc_gcd_8

! Implementation of math lcm function
pure function pyc_lcm_4(a, b) result(lcm)

    implicit none

    integer(i32), value      :: a
    integer(i32), value      :: b
    integer(i32)             :: lcm

    a = MERGE(a, -a, a > 0)
    b = MERGE(b, -b, b > 0)
    lcm = a / pyc_gcd(a, b) * b
    return

end function pyc_lcm_4

pure function pyc_lcm_8(a, b) result(lcm)

    implicit none

    integer(i64), value      :: a
    integer(i64), value      :: b
    integer(i64)             :: lcm

    a = MERGE(a, -a, a > 0)
    b = MERGE(b, -b, b > 0)
    lcm = a / pyc_gcd(a, b) * b
    return

end function pyc_lcm_8

! Implementation of math radians function
pure function pyc_radians(deg) result(rad)

    implicit none

    real(f64), value     :: deg
    real(f64)            :: rad

    rad = deg * (pi / 180.0_f64)
    return

end function pyc_radians

! Implementation of math degrees function
pure function pyc_degrees(rad) result(deg)

    implicit none

    real(f64), value     :: rad
    real(f64)            :: deg

    deg = rad * (180.0_f64 / pi)
    return

end function pyc_degrees

! Implementation of numpy amax function
function amax_4(arr) result(max_value)

    implicit none

    complex(c32)                :: max_value
    complex(c32), intent(in)    :: arr(0:)
    complex(c32)                :: a
    integer(i64)                       :: current_value

    max_value = arr(0_i64)
    do current_value = 1_i64, size(arr, kind=i64) - 1_i64
      a = arr(current_value)
      if (Real(a, f32) > Real(max_value, f32) .or. (Real(a, f32) == &
            Real(max_value, f32) .and. aimag(a) > aimag(max_value &
            ))) then
        max_value = a
      end if
    end do
    return

  end function amax_4
  
  function amax_8(arr) result(max_value)

    implicit none

    complex(c64)                :: max_value
    complex(c64), intent(in)    :: arr(0:)
    complex(c64)                :: a
    integer(i64)                       :: current_value

    max_value = arr(0_i64)
    do current_value = 1_i64, size(arr, kind=i64) - 1_i64
      a = arr(current_value)
      if (Real(a, f64) > Real(max_value, f64) .or. (Real(a, f64) == &
            Real(max_value, f64) .and. aimag(a) > aimag(max_value &
            ))) then
        max_value = a
      end if
    end do
    return

  end function amax_8
  
! Implementation of numpy amin function
function amin_4(arr) result(min_value)

    implicit none

    complex(c32)                :: min_value
    complex(c32), intent(in)    :: arr(0:)
    complex(c32)                :: a
    integer(i64)                       :: current_value

    min_value = arr(0_i64)
    do current_value = 1_i64, size(arr, kind=i64) - 1_i64
      a = arr(current_value)
      if (Real(a, f32) < Real(min_value, f32) .or. (Real(a, f32) == &
            Real(min_value, f32) .and. aimag(a) < aimag(min_value &
            ))) then
        min_value = a
      end if
    end do
    return

  end function amin_4

  function amin_8(arr) result(min_value)

    implicit none

    complex(c64)                :: min_value
    complex(c64), intent(in)    :: arr(0:)
    complex(c64)                :: a
    integer(i64)                       :: current_value

    min_value = arr(0_i64)
    do current_value = 1_i64, size(arr, kind=i64) - 1_i64
      a = arr(current_value)
      if (Real(a, f64) < Real(min_value, f64) .or. (Real(a, f64) == &
            Real(min_value, f64) .and. aimag(a) < aimag(min_value &
            ))) then
        min_value = a
      end if
    end do
    return

  end function amin_8


  elemental function numpy_v1_sign_c32(x) result(Out_0001)

    implicit none

    complex(c32) :: Out_0001
    complex(c32), value :: x
    logical :: real_ne_zero ! Condition for x.real different than 0
    logical :: imag_ne_zero ! Condition for x.imag different than 0
    real(f32) :: real_sign ! np.sign(x.real)
    real(f32) :: imag_sign ! np.sign(x.imag)

    real_ne_zero = (real(x) .ne. 0._f32)
    imag_ne_zero = (aimag(x) .ne. 0._f32)
    real_sign = sign(1._f32, real(x))
    imag_sign = sign(merge(1._f32, 0._f32, imag_ne_zero), aimag(x))

    Out_0001 = merge(real_sign, imag_sign, real_ne_zero)
    return

  end function numpy_v1_sign_c32
  
  elemental function numpy_v1_sign_c64(x) result(Out_0001)

    implicit none

    complex(c64) :: Out_0001
    complex(c64), value :: x
    logical :: real_ne_zero ! Condition for x.real different than 0
    logical :: imag_ne_zero ! Condition for x.imag different than 0
    real(f64) :: real_sign ! np.sign(x.real)
    real(f64) :: imag_sign ! np.sign(x.imag)

    real_ne_zero = (real(x) .ne. 0._f64)
    imag_ne_zero = (aimag(x) .ne. 0._f64)
    real_sign = sign(1._f64, real(x))
    imag_sign = sign(merge(1._f64, 0._f64, imag_ne_zero), aimag(x))

    Out_0001 = merge(real_sign, imag_sign, real_ne_zero)
    return

  end function numpy_v1_sign_c64
  
  elemental function numpy_v2_sign_c32(x) result(Out_0001)
    implicit none

    complex(c32) :: Out_0001
    complex(c32), value :: x

    real(f32) :: abs_val

    abs_val = abs(x)
    if (abs_val == 0) then
      Out_0001 = 0._c32
    else
      Out_0001 = x / abs_val
    end if

  end function numpy_v2_sign_c32

  elemental function numpy_v2_sign_c64(x) result(Out_0001)
    implicit none

    complex(c64) :: Out_0001
    complex(c64), value :: x

    real(f64) :: abs_val

    abs_val = abs(x)
    if (abs_val == 0) then
      Out_0001 = 0._c64
    else
      Out_0001 = x / abs_val
    end if

  end function numpy_v2_sign_c64
end module pyc_math_f90
