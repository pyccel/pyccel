module math
    implicit none

        real, parameter,private :: pi = 3.1415926536
        real, parameter, private :: e = 2.7182818285

    contains
        function factorial(x)result(fx)
        implicit none
            integer::x
            integer::fx
            fx = 1
            do while (x > 0)
                fx = fx * x
                x = x - 1
            enddo
        end function factorial

end module math
