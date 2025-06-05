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
        return p.x
