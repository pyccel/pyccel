from .Point_mod import Point

class Line:
    def __init__(self, start : Point, end : Point):
        self.start = start
        self.end = end

    def get_start(self):
        return self.start.get_val()

    def get_end(self):
        return self.end.get_val()
