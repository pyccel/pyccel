# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

def f(a: int):
    # 1
    if a:
        # 2
        b = (a + 2)
        # 3
    elif (a + 1):
        # 4
        b = (a + 1)
        # 5
    else:
        # 6
        b = a
        # 7
    # 8
    c = (b * 3)
    # 9
    return c

def g(b : int):
    # 10
    while b:
        # 11
        b -= 1
        # 12
    # 13
    for i in range(10):
        # 14
        for _ in range(5):
            # 15
            print(i)
            # 16
            if (i == 4):
                # 17
                k = 3
                # 18
            else:
                # 19
                k = 4
                # 20
            # 21
            print(k)
            # 22
        # 23
    # 24
#25

