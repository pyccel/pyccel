# pylint: disable=missing-function-docstring, missing-module-docstring/
# ------------------------------- Strings ------------------------------------

from numpy import int32, int64
if __name__ == '__main__':
    print(0)
    print(00)
    print(1)
    print(-1)
    print(-0)
    print(10000)
    print(-10000)
    print(2147483647)
    print(int64(2147483648))
    print(int64(9223372036854775807))
    print(int32(0))
    print(int32(00))
    print(int32(1))
    print(int32(-1))
    print(int32(-0))
    print(int32(10000))
    print(int32(-10000))
    print(int32(2147483647))
    # Fortran on windows doesn't compile with literal -2147483648
    # print(int64(-2147483648))
    # print(int32(-2147483648))
