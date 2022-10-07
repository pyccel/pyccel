! ! --------------------------------------------------------------------------------------- !
! ! This file is part of Pyccel which is released under MIT License. See the LICENSE file   !
! ! or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. !
! ! --------------------------------------------------------------------------------------- !

module numpy_f90


    use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f32 => &
          C_FLOAT , i32 => C_INT32_T , f64 => C_DOUBLE , i8 => C_INT8_T , &
          c64 => C_DOUBLE_COMPLEX , i16 => C_INT16_T , c32 => &
          C_FLOAT_COMPLEX
    implicit none

    private

    public :: numpy_sign
  
    interface numpy_sign
      module procedure numpy_sign_i8
      module procedure numpy_sign_i16
      module procedure numpy_sign_i32
      module procedure numpy_sign_i64
      module procedure numpy_sign_f32
      module procedure numpy_sign_f64
      module procedure numpy_sign_c32
      module procedure numpy_sign_c64
    end interface
  
    contains
  
    !........................................
    elemental function numpy_sign_i8(x) result(Out_0001)
  
      implicit none
  
      integer(i8) :: Out_0001
      integer(i8), value :: x
  
      Out_0001 = merge(0_i8, (merge(1_i8, -1_i8, x > 0_i8)), x == 0_i8)
      return
  
    end function numpy_sign_i8
    !........................................
  
    !........................................
    elemental function numpy_sign_i16(x) result(Out_0001)
  
      implicit none
  
      integer(i16) :: Out_0001
      integer(i16), value :: x
  
      Out_0001 = merge(0_i16, (merge(1_i16, -1_i16, x > 0_i16)), x == 0_i16)
      return
  
    end function numpy_sign_i16
    !........................................
  
    !........................................
    elemental function numpy_sign_i32(x) result(Out_0001)
  
      implicit none
  
      integer(i32) :: Out_0001
      integer(i32), value :: x
  
      Out_0001 = merge(0_i32, (merge(1_i32, -1_i32, x > 0_i32)), x == 0_i32)
      return
  
    end function numpy_sign_i32
    !........................................
  
    !........................................
    elemental function numpy_sign_i64(x) result(Out_0001)
  
      implicit none
  
      integer(i64) :: Out_0001
      integer(i64), value :: x
  
      Out_0001 = merge(0_i64, (merge(1_i64, -1_i64, x > 0_i64)), x == 0_i64)
      return
  
    end function numpy_sign_i64
    !........................................
  
    !........................................
    elemental function numpy_sign_f32(x) result(Out_0001)
  
      implicit none
  
      real(f32) :: Out_0001
      real(f32), value :: x
  
      Out_0001 = merge(0_f32, (merge(1_f32, -1_f32, x > 0_f32)), x == 0_f32)
      return
  
    end function numpy_sign_f32
    !........................................
  
    !........................................
    elemental function numpy_sign_f64(x) result(Out_0001)
  
      implicit none
  
      real(f64) :: Out_0001
      real(f64), value :: x
  
      Out_0001 = merge(0_f64, (merge(1_f64, -1_f64, x > 0_f64)), x == 0_f64)
      return
  
    end function numpy_sign_f64
    !........................................
  
    !........................................
    elemental function numpy_sign_c32(x) result(Out_0001)
  
      implicit none

      complex(c32) :: Out_0001
      complex(c32), value :: x
      logical :: x_ne_zero ! Condition for x different than 0
      logical :: x_lt_zero ! Condition for x less than 0
  
      x_ne_zero = (REALPART(x) .ne. 0_f32) .or. (IMAGPART(x) .ne. 0_f32)
      x_lt_zero = ((REALPART(x) .eq. 0_f32) .and. IMAGPART(x) .lt. 0_f32) &
              .or. (REALPART(x) .lt. 0_f32)
  
      Out_0001 = merge(merge(-1_c32, 1_c32, x_lt_zero), 0_c32, x_ne_zero)
      return
  
    end function numpy_sign_c32
    !........................................
  
    !........................................
    elemental function numpy_sign_c64(x) result(Out_0001)
  
      implicit none
  
      complex(c64) :: Out_0001
      complex(c64), value :: x
      logical :: x_ne_zero ! Condition for x different than 0
      logical :: x_lt_zero ! Condition for x less than 0

      x_ne_zero = (REALPART(x) .ne. 0_f64) .or. (IMAGPART(x) .ne. 0_f64)
      x_lt_zero = ((REALPART(x) .eq. 0_f64) .and. IMAGPART(x) .lt. 0_f64) &
              .or. (REALPART(x) .lt. 0_f64)

      Out_0001 = merge(merge(-1_c64, 1_c64, x_lt_zero), 0_c64, x_ne_zero)
      return

    end function numpy_sign_c64
    !........................................
  
  end module numpy_f90
