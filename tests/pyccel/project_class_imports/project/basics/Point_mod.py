# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class Point:
    def __init__(self, x : float, y : float):
        self.x = x
        self.y = y

    def get_val(self):
        return self.x, self.y
