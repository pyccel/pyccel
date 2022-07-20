! --------------------------------------------------------------------------------------- !
! This file is part of Pyccel which is released under MIT License. See the LICENSE file   !
! or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. !
! --------------------------------------------------------------------------------------- !

module pyc_math_f90

use, intrinsic :: ISO_C_BINDING

implicit none

private

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

interface pyc_bankers_round
    module procedure pyc_bankers_round_4
    module procedure pyc_bankers_round_8
    module procedure pyc_bankers_round_int_4
    module procedure pyc_bankers_round_int_8
end interface pyc_bankers_round

public :: pyc_gcd, &
          pyc_factorial, &
          pyc_lcm, &
          pyc_radians, &
          pyc_degrees, &
          pyc_bankers_round

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

    x = a
    y = b
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

    x = a
    y = b
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

    lcm = a / pyc_gcd(a, b) * b
    return

end function pyc_lcm_4

pure function pyc_lcm_8(a, b) result(lcm)

    implicit none

    integer(C_INT64_T), value      :: a
    integer(C_INT64_T), value      :: b
    integer(C_INT64_T)             :: lcm

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

pure function pyc_bankers_round_4(arg, ndigits) result(rnd)

    implicit none

    real(C_DOUBLE), value     :: arg
    integer(C_INT32_T), value :: ndigits
    real(C_DOUBLE)            :: rnd

    real(C_DOUBLE) :: diff

    arg = arg * 10._C_DOUBLE**ndigits

    rnd = nint(arg, kind=C_INT64_T)

    diff = arg - rnd

    if (ndigits <= 0 .and. (diff == 0.5_C_DOUBLE .or. diff == -0.5_C_DOUBLE)) then
        rnd = nint(arg*0.5_C_DOUBLE, kind=C_INT64_T)*2_C_INT64_T
    end if

    rnd = rnd * 10._C_DOUBLE**(-ndigits)

end function pyc_bankers_round_4

pure function pyc_bankers_round_8(arg, ndigits) result(rnd)

    implicit none

    real(C_DOUBLE), value     :: arg
    integer(C_INT64_T), value :: ndigits
    real(C_DOUBLE)            :: rnd

    real(C_DOUBLE) :: diff

    arg = arg * 10._C_DOUBLE**ndigits

    rnd = nint(arg, kind=C_INT64_T)

    diff = arg - rnd

    if (ndigits <= 0 .and. (diff == 0.5_C_DOUBLE .or. diff == -0.5_C_DOUBLE)) then
        rnd = nint(arg*0.5_C_DOUBLE, kind=C_INT64_T)*2_C_INT64_T
    end if

    rnd = rnd * 10._C_DOUBLE**(-ndigits)

end function pyc_bankers_round_8

pure function pyc_bankers_round_int_4(arg, ndigits) result(rnd)

    implicit none

    integer(C_INT32_T), value :: arg
    integer(C_INT32_T), value :: ndigits
    integer(C_INT32_T)        :: rnd

    real(C_DOUBLE) :: val

    val = arg * 10._C_DOUBLE**ndigits

    rnd = nint(val, kind=C_INT64_T)

    rnd = rnd * 10._C_DOUBLE**(-ndigits)

end function pyc_bankers_round_int_4

pure function pyc_bankers_round_int_8(arg, ndigits) result(rnd)

    implicit none

    integer(C_INT64_T), value :: arg
    integer(C_INT64_T), value :: ndigits
    integer(C_INT64_T)        :: rnd

    real(C_DOUBLE) :: val

    val = arg * 10._C_DOUBLE**ndigits

    rnd = nint(val, kind=C_INT64_T)

    rnd = rnd * 10._C_DOUBLE**(-ndigits)

end function pyc_bankers_round_int_8

end module pyc_math_f90
