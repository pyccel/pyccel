from ..basics.Point_mod import Point

class Square:
    def __init__(self, a : Point, b : Point, c : Point, d : Point):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def get_corners(self):
        return self.a.get_val(), self.b.get_val(), self.c.get_val(), self.d.get_val()
