! -------------------------------------------------------------------------------------- !
! This file is part of Pyccel which is released under MIT License. See the LICENSE file  !
! or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. !
! -------------------------------------------------------------------------------------- !

module numpy_f90


    use, intrinsic :: ISO_C_Binding, only : f32 => C_FLOAT , &
        f64 => C_DOUBLE , c64 => C_DOUBLE_COMPLEX , c32 => &
          C_FLOAT_COMPLEX
    implicit none

    private

    public :: numpy_sign
  
    interface numpy_sign
      module procedure numpy_sign_c32
      module procedure numpy_sign_c64
    end interface
  
    contains
  
    !........................................
    elemental function numpy_sign_c32(x) result(Out_0001)
  
      implicit none

      complex(c32) :: Out_0001
      complex(c32), value :: x
      logical :: real_ne_zero ! Condition for x.real different than 0
      logical :: imag_ne_zero ! Condition for x.imag different than 0
      real(c32) :: real_sign ! np.sign(x.real)
      real(c32) :: imag_sign ! np.sign(x.imag)
  
      real_ne_zero = (real(x) .ne. 0._f32)
      imag_ne_zero = (aimag(x) .ne. 0._f32)
      real_sign = sign(1._f32, real(x))
      imag_sign = sign(merge(1._f32, 0._f32, imag_ne_zero), aimag(x))
  
      Out_0001 = merge(real_sign, imag_sign, real_ne_zero)
      return
  
    end function numpy_sign_c32
    !........................................
  
    !........................................
    elemental function numpy_sign_c64(x) result(Out_0001)
  
      implicit none
  
      complex(c64) :: Out_0001
      complex(c64), value :: x
      logical :: real_ne_zero ! Condition for x.real different than 0
      logical :: imag_ne_zero ! Condition for x.imag different than 0
      real(c64) :: real_sign ! np.sign(x.real)
      real(c64) :: imag_sign ! np.sign(x.imag)

      real_ne_zero = (real(x) .ne. 0._f64)
      imag_ne_zero = (aimag(x) .ne. 0._f64)
      real_sign = sign(1._f64, real(x))
      imag_sign = sign(merge(1._f64, 0._f64, imag_ne_zero), aimag(x))
  
      Out_0001 = merge(real_sign, imag_sign, real_ne_zero)
      return

    end function numpy_sign_c64
    !........................................
  
  end module numpy_f90
