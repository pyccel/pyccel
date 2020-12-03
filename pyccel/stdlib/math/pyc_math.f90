module pyc_math

    implicit none

        real(kind=8), parameter, private :: pi = 4.D0 * DATAN(1.D0)

    contains

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
