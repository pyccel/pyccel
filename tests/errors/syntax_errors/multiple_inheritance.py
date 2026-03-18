# Multiple inheritance is not supported
# pylint: disable=missing-class-docstring, missing-module-docstring


class A:
    pass


class B:
    pass


class C(A, B):
    pass
