# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring, relative-beyond-top-level
import numpy as np
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

    def longer_than(self, length : 'int | float'):
        s_x, s_y = self.get_start()
        e_x, e_y = self.get_end()
        my_length = np.sqrt((e_x-s_x)**2+(e_y-s_y)**2)
        return my_length > length
