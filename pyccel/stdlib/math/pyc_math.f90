module pyc_math

    implicit none

        real(kind=8), parameter, private :: pi = 4.D0 * DATAN(1.D0)

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
    contains
        ! Implementation of math factorial function
        function pyc_factorial_4(x) result(fx) ! integers with precision 4

            implicit none
            integer(kind=4), value  :: x
            integer(kind=4)         :: fx

            fx = 1
            do while (x > 0)
                fx = fx * x
                x = x - 1
            enddo
            return

        end function pyc_factorial_4

        function pyc_factorial_8(x) result(fx) ! integers with precision 8

            implicit none
            integer(kind=8), value  :: x
            integer(kind=8)         :: fx

            fx = 1
            do while (x > 0)
                fx = fx * x
                x = x - 1
            enddo
            return

        end function pyc_factorial_8

        ! Implementation of math gcd function
        function pyc_gcd_4(a, b) result(gcd) ! integers with precision 4

            implicit none
            integer(kind=4), value  :: a
            integer(kind=4), value  :: b
            integer(kind=4)         :: gcd

            do while (b > 0)
                a = MOD(a, b)
                a = XOR(a, b)
                b = XOR(b, a)
                a = XOR(a, b)
            enddo
            gcd = a
            return

        end function pyc_gcd_4

        function pyc_gcd_8(a, b) result(gcd) ! integers with precision 8

            implicit none
            integer(kind=8), value  :: a
            integer(kind=8), value  :: b
            integer(kind=8)         :: gcd

            do while (b > 0)
                a = MOD(a, b)
                a = XOR(a, b)
                b = XOR(b, a)
                a = XOR(a, b)
            enddo
            gcd = a
            return

        end function pyc_gcd_8

        ! Implementation of math lcm function
        function pyc_lcm_4(a, b) result(lcm)

            implicit none
            integer(kind=4), value :: a
            integer(kind=4), value :: b
            integer(kind=4)        :: lcm

            lcm = a / pyc_gcd(a, b) * b
            return

        end function pyc_lcm_4

        function pyc_lcm_8(a, b) result(lcm)

            implicit none
            integer(kind=8), value :: a
            integer(kind=8), value :: b
            integer(kind=8)        :: lcm

            lcm = a / pyc_gcd(a, b) * b
            return

        end function pyc_lcm_8

        ! Implementation of math radians function
        function pyc_radians(deg) result(rad)

            implicit none
            real(kind=8), value     :: deg
            real(kind=8)            :: rad

            rad = deg * (pi / 180.0)
            return

        end function pyc_radians

        ! Implementation of math degrees function
        function pyc_degrees(rad) result(deg)

            implicit none
            real(kind=8), value     :: rad
            real(kind=8)            :: deg

            deg = rad * (180.0 / pi)
            return

        end function pyc_degrees

end module pyc_math
