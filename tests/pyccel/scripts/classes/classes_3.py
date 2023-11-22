# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
# coding: utf-8

class Point:
    def __init__(self, x : float, y : float):
        self.x = x
        self.X = y

    def set_coordinates(self, x : float, y : float):
        self.x = x
        self.X = y

    def get_coordinates(self):
        return self.x, self.X

class Point1:
    def __init__(self, x : float):
        self.x = x

class Point2:
    def __init__(self, y : float):
        self.y = y

    def test_func(self):
        p = Point1(self.y)
        print(p.x)

if __name__ == '__main__':
    j = Point2(2.2)
    j.test_func()

    j = Point2(6.5)
    j.test_func()

    p = Point(3.5, 0.1)
    p.set_coordinates(2.3, 5.1)
    print(p.get_coordinates())
