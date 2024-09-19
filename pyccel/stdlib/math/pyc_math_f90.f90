! -------------------------------------------------------------------------------------- !
! This file is part of Pyccel which is released under MIT License. See the LICENSE file  !
! or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. !
! -------------------------------------------------------------------------------------- !

module pyc_math_f90

use ISO_C_BINDING

implicit none

real(C_DOUBLE), parameter, private :: pi = 4.0_C_DOUBLE * DATAN(1.0_C_DOUBLE)

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

contains

! Implementation of math factorial function
pure function pyc_factorial_4(x) result(fx) ! integers with precision 4

    implicit none

    integer(C_INT32_T), value      :: x
    integer(C_INT32_T)             :: i
    integer(C_INT32_T)             :: fx

    fx = 1_C_INT32_T
    do i = 2_C_INT32_T, x
        fx = fx * i
    enddo
    return

end function pyc_factorial_4

pure function pyc_factorial_8(x) result(fx) ! integers with precision 8

    implicit none

    integer(C_INT64_T), value       :: x
    integer(C_INT64_T)              :: fx
    integer(C_INT64_T)              :: i

    fx = 1_C_INT64_T
    do i = 2_C_INT64_T, x
        fx = fx * i
    enddo

end function pyc_factorial_8

! Implementation of math gcd function
pure function pyc_gcd_4(a, b) result(gcd) ! integers with precision 4

    implicit none

    integer(C_INT32_T), value      :: a, b
    integer(C_INT32_T)             :: x, y
    integer(C_INT32_T)             :: gcd

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

    integer(C_INT64_T), value       :: a, b
    integer(C_INT64_T)              :: x, y
    integer(C_INT64_T)              :: gcd

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

    integer(C_INT32_T), value      :: a
    integer(C_INT32_T), value      :: b
    integer(C_INT32_T)             :: lcm

    a = MERGE(a, -a, a > 0)
    b = MERGE(b, -b, b > 0)
    lcm = a / pyc_gcd(a, b) * b
    return

end function pyc_lcm_4

pure function pyc_lcm_8(a, b) result(lcm)

    implicit none

    integer(C_INT64_T), value      :: a
    integer(C_INT64_T), value      :: b
    integer(C_INT64_T)             :: lcm

    a = MERGE(a, -a, a > 0)
    b = MERGE(b, -b, b > 0)
    lcm = a / pyc_gcd(a, b) * b
    return

end function pyc_lcm_8

! Implementation of math radians function
pure function pyc_radians(deg) result(rad)

    implicit none

    real(C_DOUBLE), value     :: deg
    real(C_DOUBLE)            :: rad

    rad = deg * (pi / 180.0_C_DOUBLE)
    return

end function pyc_radians

! Implementation of math degrees function
pure function pyc_degrees(rad) result(deg)

    implicit none

    real(C_DOUBLE), value     :: rad
    real(C_DOUBLE)            :: deg

    deg = rad * (180.0_C_DOUBLE / pi)
    return

end function pyc_degrees

! Implementation of numpy amax function
function amax_4(arr) result(max_value)

    implicit none

    complex( C_FLOAT_COMPLEX)                :: max_value
    complex( C_FLOAT_COMPLEX), intent(in)    :: arr(0:)
    complex( C_FLOAT_COMPLEX)                :: a
    integer(C_INT64_T)                       :: current_value

    max_value = arr(0_C_INT64_T)
    do current_value = 1_C_INT64_T, size(arr, kind=C_INT64_T) - 1_C_INT64_T
      a = arr(current_value)
      if (Real(a, C_FLOAT) > Real(max_value, C_FLOAT) .or. (Real(a, C_FLOAT) == &
            Real(max_value, C_FLOAT) .and. aimag(a) > aimag(max_value &
            ))) then
        max_value = a
      end if
    end do
    return

  end function amax_4
  
  function amax_8(arr) result(max_value)

    implicit none

    complex(C_DOUBLE_COMPLEX)                :: max_value
    complex(C_DOUBLE_COMPLEX), intent(in)    :: arr(0:)
    complex(C_DOUBLE_COMPLEX)                :: a
    integer(C_INT64_T)                       :: current_value

    max_value = arr(0_C_INT64_T)
    do current_value = 1_C_INT64_T, size(arr, kind=C_INT64_T) - 1_C_INT64_T
      a = arr(current_value)
      if (Real(a, C_DOUBLE) > Real(max_value, C_DOUBLE) .or. (Real(a, C_DOUBLE) == &
            Real(max_value, C_DOUBLE) .and. aimag(a) > aimag(max_value &
            ))) then
        max_value = a
      end if
    end do
    return

  end function amax_8
  
! Implementation of numpy amin function
function amin_4(arr) result(min_value)

    implicit none

    complex( C_FLOAT_COMPLEX)                :: min_value
    complex( C_FLOAT_COMPLEX), intent(in)    :: arr(0:)
    complex( C_FLOAT_COMPLEX)                :: a
    integer(C_INT64_T)                       :: current_value

    min_value = arr(0_C_INT64_T)
    do current_value = 1_C_INT64_T, size(arr, kind=C_INT64_T) - 1_C_INT64_T
      a = arr(current_value)
      if (Real(a, C_FLOAT) < Real(min_value, C_FLOAT) .or. (Real(a, C_FLOAT) == &
            Real(min_value, C_FLOAT) .and. aimag(a) < aimag(min_value &
            ))) then
        min_value = a
      end if
    end do
    return

  end function amin_4

  function amin_8(arr) result(min_value)

    implicit none

    complex(C_DOUBLE_COMPLEX)                :: min_value
    complex(C_DOUBLE_COMPLEX), intent(in)    :: arr(0:)
    complex(C_DOUBLE_COMPLEX)                :: a
    integer(C_INT64_T)                       :: current_value

    min_value = arr(0_C_INT64_T)
    do current_value = 1_C_INT64_T, size(arr, kind=C_INT64_T) - 1_C_INT64_T
      a = arr(current_value)
      if (Real(a, C_DOUBLE) < Real(min_value, C_DOUBLE) .or. (Real(a, C_DOUBLE) == &
            Real(min_value, C_DOUBLE) .and. aimag(a) < aimag(min_value &
            ))) then
        min_value = a
      end if
    end do
    return

  end function amin_8

end module pyc_math_f90
