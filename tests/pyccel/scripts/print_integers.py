# pylint: disable=missing-function-docstring, missing-module-docstring
# ------------------------------- Strings ------------------------------------

from numpy import int32, int64, int16, int8
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

    print(int16(0))
    print(int16(-10))
    print(int16(1))
    print(int16(32767))
    print(int16(-32768))

    print(int8(0))
    print(int8(-10))
    print(int8(1))
    print(int8(127))
    print(int8(-128))
