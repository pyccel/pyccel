! --------------------------------------------------------------------------------------- !
! This file is part of Pyccel which is released under MIT License. See the LICENSE file   !
! or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. !
! --------------------------------------------------------------------------------------- !

module numpy_f90

    use ISO_C_BINDING
    
    implicit none
    
    interface numpy_sign
        module procedure real_sign
        module procedure cmplx_float_sign
        module procedure cmplx_double_sign
    end interface numpy_sign
    
    contains


    ! Implementation of numpy.sign function for real numbers
    elemental function real_sign(x) result(y)
    
        implicit none
    
        real(C_DOUBLE), value       :: x
        real(C_DOUBLE)              :: y
        real(C_DOUBLE)              :: merge_ne_zero
        real(C_DOUBLE)              :: zero
    
        zero = 0
        merge_ne_zero = merge(-1., 1., x .lt. 0.)
        y = merge(merge_ne_zero, zero, x .ne. 0.)
        return

    end function real_sign


    ! Implementation of numpy.sign function for complex numbers
    elemental function cmplx_float_sign(x) result(y)

        implicit none

        Complex(C_FLOAT_COMPLEX), value    :: x
        Complex(C_FLOAT_COMPLEX)           :: y
        Complex(C_FLOAT_COMPLEX)           :: merge_ne_zero
        Complex(C_FLOAT_COMPLEX)           :: zero
        logical                            :: x_ne_zero ! Condition for x different than 0
        logical                            :: x_lt_zero ! Condition for x less than 0

        zero = 0
        x_ne_zero = (REALPART(x) .ne. 0) .or. (IMAGPART(x) .ne. 0)
        x_lt_zero = ((REALPART(x) .eq. 0) .and. IMAGPART(x) .lt. 0) .or. (REALPART(x) .lt. 0)

        merge_ne_zero = merge(-1, 1, x_lt_zero)
        
        y = merge(merge_ne_zero, zero, x_ne_zero)
        return

    end function cmplx_float_sign


    ! Implementation of numpy.sign function for complex numbers
    elemental function cmplx_double_sign(x) result(y)

        implicit none

        Complex(C_DOUBLE_COMPLEX), value    :: x
        Complex(C_DOUBLE_COMPLEX)           :: y
        Complex(C_DOUBLE_COMPLEX)           :: merge_ne_zero
        Complex(C_DOUBLE_COMPLEX)           :: zero
        logical                            :: x_ne_zero ! Condition for x different than 0
        logical                            :: x_lt_zero ! Condition for x less than 0

        zero = 0
        x_ne_zero = (REALPART(x) .ne. 0) .or. (IMAGPART(x) .ne. 0)
        x_lt_zero = ((REALPART(x) .eq. 0) .and. IMAGPART(x) .lt. 0) .or. (REALPART(x) .lt. 0)

        merge_ne_zero = merge(-1, 1, x_lt_zero)
        
        y = merge(merge_ne_zero, zero, x_ne_zero)
        return

    end function cmplx_double_sign

end module numpy_f90