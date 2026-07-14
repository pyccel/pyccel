# pylint: disable=missing-function-docstring, missing-module-docstring


def f1(x: "int"):
    a = 5 if x < 5 else x
    return a


def f2(x: "int"):
    a = 5.5 if x < 5 else x
    return a


def f3(x: "int"):
    a = x if x < 5 else 5 + 2
    return a


def f3wp(x: "int"):
    a = (x if x < 5 else 5) + 2
    return a


def f4(x: "int"):
    a = x if x < 5 else 5 >> 2
    return a


def f4wp(x: "int"):
    a = (x if x < 5 else 5) >> 2
    return a


def f5(x: "int"):
    a = x if x < 5 else 5 if x == 5 else 5.5
    return a


def f5wp(x: "int"):
    a = x if x < 5 else (5 if x == 5 else 5.5)
    return a


def f6(x: "int"):
    # a = x if x < 0 else (1 if x < 5 else (complex(0, 1) if x == 5 else 6.5))
    a = x if x < 0 else 1 if x < 5 else complex(0, 1) if x == 5 else 6.5
    return a


def f6wp(x: "int"):
    a = x if x < 0 else (1 if x < 5 else (complex(0, 1) if x == 5 else 6.5))
    return a


def f8(x: "int"):
    a = (1 + 0j, 2 + 0j) if x < 5 else (complex(5, 1), complex(2, 2))
    return a[0]


def f8wp(x: "int"):
    a = (
        (1 + 0j, 2 + 0j)
        if x < 5
        else (
            (complex(5, 1), complex(2, 2)) if x > 5 else (complex(7, 2), complex(3, 3))
        )
    )
    return a[0]


def f9(x: "int"):
    a = 1 + 2 if x < 5 else 3
    return a


def f9wp1(x: "int"):
    a = 1 + (2 if x < 5 else 3)
    return a


def f9wp2(x: "int"):
    a = (1 + 2) if x < 5 else 3
    return a


def f10(x: "int"):
    a = 2 if x < 5 else 3 + 1
    return a


def f10wp1(x: "int"):
    a = (2 if x < 5 else 3) + 1
    return a


def f10wp2(x: "int"):
    a = 2 if x < 5 else (3 + 1)
    return a


def f11(x: "int"):
    a = 2 if (x + 2) * 5 < 5 else 3
    return a


def f12(x: "int"):
    a = [1.0, 2.0, 3.0, 4.0] if x < 5 else [1.5, 6.5, 7.5]
    return a[0]


def f12wp(x: "int"):
    a = (
        [1.0, 2.0, 3.0]
        if x < 5
        else ([1.5, 6.5, 7.5] if x > 5 else [3.1, 9.5, 2.8, 2.9])
    )
    return a[0]


def f13(b: bool):
    a = "hello" if b else "world!"
    return a


def f13wp(b1: bool, b2: bool):
    a = "hello" if b1 else ("world" if b2 else "hello world")
    return a
