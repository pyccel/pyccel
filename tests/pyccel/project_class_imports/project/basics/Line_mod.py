# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from .Point_mod import Point

class Line:
    def __init__(self, start : Point, end : Point):
        self.start = start
        self.end = end

    def get_start(self):
        x,y = self.start.get_val()
        return x,y

    def get_end(self):
        x,y = self.end.get_val()
        return x,y
