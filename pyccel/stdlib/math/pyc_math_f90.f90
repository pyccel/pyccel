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

public :: pyc_gcd, &
          pyc_factorial, &
          pyc_lcm, &
          pyc_radians, &
          pyc_degrees, &
          amax, &
          amin, &
          csign, &
          pyc_bankers_round, &
          pyc_floor_div, &
          expm1

private

real(f64), parameter :: pi = 4.0_f64 * DATAN(1.0_f64)

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

interface csign
    module procedure sign_c32
    module procedure sign_c64
end interface csign

interface pyc_bankers_round
    module procedure pyc_bankers_round_float
    module procedure pyc_bankers_round_int
end interface pyc_bankers_round

interface pyc_floor_div
    module procedure pyc_floor_div_i8
    module procedure pyc_floor_div_i16
    module procedure pyc_floor_div_i32
    module procedure pyc_floor_div_i64
end interface pyc_floor_div

interface
   real(f64) pure function c_expm1(x) bind(c, name='expm1')
     use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE
     real(f64), intent(in), value :: x
   end function c_expm1
end interface

interface expm1
    module procedure pyc_expm1_f64
    module procedure pyc_expm1_c64
end interface expm1

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
  
  elemental function sign_c32(x) result(Out_0001)
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

  end function sign_c32

  elemental function sign_c64(x) result(Out_0001)
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

  end function sign_c64

pure function pyc_bankers_round_float(arg, ndigits) result(rnd)

    implicit none

    real(f64), value     :: arg
    integer(i64), value :: ndigits
    real(f64)            :: rnd

    real(f64) :: diff

    arg = arg * 10._f64**ndigits

    rnd = nint(arg, kind=i64)

    diff = arg - rnd

    if (ndigits <= 0 .and. (diff == 0.5_f64 .or. diff == -0.5_f64)) then
        rnd = nint(arg*0.5_f64, kind=i64)*2_i64
    end if

    rnd = rnd * 10._f64**(-ndigits)

end function pyc_bankers_round_float

pure function pyc_bankers_round_int(arg, ndigits) result(rnd)

    implicit none

    integer(i64), value :: arg
    integer(i64), value :: ndigits
    integer(i64)        :: rnd

    integer(i64) :: val
    integer(i64) :: mul_fact
    integer(i64) :: pivot_point
    integer(i64) :: remainder

    if (ndigits >= 0) then
        rnd = arg
    else
        mul_fact = 10_i64**(-ndigits)
        pivot_point = sign(5_i64*10_i64**(-ndigits-1_i64), arg)
        remainder = modulo(arg, mul_fact)
        if ( remainder == pivot_point ) then
            val = (mul_fact - remainder) / mul_fact
            rnd = (val + IAND(val, 1_i64)) * mul_fact
        else
            rnd = ((arg + pivot_point) / mul_fact) * mul_fact
        endif
    endif

end function pyc_bankers_round_int

elemental pure integer(kind=1) function pyc_floor_div_i8(x, y) result(res)
  implicit none
  integer(kind=1), intent(in) :: x, y
  res = x / y - merge(1, 0, mod(x, y) /= 0 .and. ((x < 0) .neqv. (y < 0)))
end function pyc_floor_div_i8

elemental pure integer(kind=2) function pyc_floor_div_i16(x, y) result(res)
  implicit none
  integer(kind=2), intent(in) :: x, y
  res = x / y - merge(1, 0, mod(x, y) /= 0 .and. ((x < 0) .neqv. (y < 0)))
end function pyc_floor_div_i16

elemental pure integer(kind=4) function pyc_floor_div_i32(x, y) result(res)
  implicit none
  integer(kind=4), intent(in) :: x, y
  res = x / y - merge(1, 0, mod(x, y) /= 0 .and. ((x < 0) .neqv. (y < 0)))
end function pyc_floor_div_i32

elemental pure integer(kind=8) function pyc_floor_div_i64(x, y) result(res)
  implicit none
  integer(kind=8), intent(in) :: x, y
  res = x / y - merge(1, 0, mod(x, y) /= 0 .and. ((x < 0) .neqv. (y < 0)))
end function pyc_floor_div_i64

elemental pure function pyc_expm1_f64(x) result(Out_0001)
    implicit none
    real(f64) :: Out_0001
    real(f64), value :: x

    Out_0001 = c_expm1(x)

end function pyc_expm1_f64

elemental pure function pyc_expm1_c64(x) result(Out_0001)
    implicit none

    complex(c64) :: Out_0001
    complex(c64), value :: x

    real(f64) :: a

    a = sin(aimag(x) * 0.5_f64)
    Out_0001 = (c_expm1(real(x)) * cos(aimag(x)) - 2._f64 * a * a) + &
               (exp(real(x)) * sin(aimag(x))) * cmplx(0,1, kind = c64)

end function pyc_expm1_c64

end module pyc_math_f90
