# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
class A:
    def f(self) -> int:
        return 1


class B(A):
    def f(self) -> int:
        return super().f() + 1
