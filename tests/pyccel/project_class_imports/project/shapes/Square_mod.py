# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring, relative-beyond-top-level
from ..basics.Point_mod import Point

class Square:
    def __init__(self, a : Point, b : Point, c : Point, d : Point):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def get_corner_1(self):
        x,y = self.a.get_val()
        return x,y

    def get_corner_2(self):
        x,y = self.b.get_val()
        return x,y

    def get_corner_3(self):
        x,y = self.c.get_val()
        return x,y

    def get_corner_4(self):
        x,y = self.d.get_val()
        return x,y
