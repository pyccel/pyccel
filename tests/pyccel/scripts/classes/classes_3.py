# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
# coding: utf-8

class Point1:
    def __init__(self : Point1, x : float):
        self.x = x

class Point2:
    def __init__(self : Point2, y : float):
        self.y = y

    def test_func(self : Point2):
        p = Point1(self.y)
        print(p.x)

if __name__ == '__main__':
    j = Point2(2.2)
    j.test_func()
