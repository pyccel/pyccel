! --------------------------------------------------------------------------------------- !
! This file is part of Pyccel which is released under MIT License. See the LICENSE file   !
! or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. !
! --------------------------------------------------------------------------------------- !

module numpy_f90

    use ISO_C_BINDING
    
    implicit none
    
    private

    public :: numpy_sign

    interface numpy_sign
        module procedure i8_sign
        module procedure i16_sign
        module procedure i32_sign
        module procedure i64_sign
        module procedure float_sign
        module procedure double_sign
        module procedure cmplx_float_sign
        module procedure cmplx_double_sign
    end interface numpy_sign
    
    contains

    ! Implementation of numpy.sign function for real numbers
    elemental function i8_sign(x) result(y)

        implicit none

        INTEGER(C_INT8_T), value :: x
        INTEGER(C_INT8_T)        :: y
        
        y = merge(merge(-1, 1, x .lt. 0), 0, x .ne. 0)

    end function i8_sign

    elemental function i16_sign(x) result(y)

        implicit none

        INTEGER(C_INT16_T), value :: x
        INTEGER(C_INT16_T)        :: y

        y = merge(merge(-1, 1, x .lt. 0), 0, x .ne. 0)

    end function i16_sign

    elemental function i32_sign(x) result(y)

        implicit none

        INTEGER, value :: x
        INTEGER        :: y

        y = merge(merge(-1, 1, x .lt. 0), 0, x .ne. 0)

    end function i32_sign


    elemental function i64_sign(x) result(y)

        implicit none

        INTEGER(C_INT64_T), value :: x
        INTEGER(C_INT64_T)        :: y

        y = merge(merge(-1, 1, x .lt. 0), 0, x .ne. 0)

    end function i64_sign


    ! Implementation of numpy.sign function for real numbers
    elemental function float_sign(x) result(y)
    
        implicit none
    
        real(C_FLOAT), value       :: x
        real(C_FLOAT)              :: y
        
        y = merge(merge(-1., 1., x .lt. 0.), 0., x .ne. 0.)

    end function float_sign

    elemental function double_sign(x) result(y)
    
        implicit none
    
        real(C_DOUBLE), value       :: x
        real(C_DOUBLE)              :: y

        y = merge(merge(-1., 1., x .lt. 0.), 0., x .ne. 0.)

    end function double_sign


    ! Implementation of numpy.sign function for complex numbers
    elemental function cmplx_float_sign(x) result(y)

        implicit none

        complex(C_FLOAT_COMPLEX), value    :: x
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

    end function cmplx_double_sign

end module numpy_f90