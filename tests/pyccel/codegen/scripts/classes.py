#$ header class Point(public)
#$ header method __init__(Point, double, double)
#$ header method __del__(Point)
#$ header method translate(Point, double, double)

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __del__(self):
        pass

    def translate(self, a, b):
        self.x = self.x + a
        self.y = self.y + b

#p = Point(0.0, 0.0)
x = 5
y = 0

while y < 3:
    y = y + 1

while y < 3 and x > 2:
    y = y + 1
    x = x - 1
