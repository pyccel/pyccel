from pyccel.decorators import types


#$ header class Point(public)
class Point(object):

    @types('Point', 'float', 'float')
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @types('Point')
    def __del__(self):
        pass

    @types('Point', 'float', 'float')
    def translate(self, a, b):
        self.x = self.x + a
        self.y = self.y + b

p = Point(0.0, 0.0)

p.translate(1.0, 2.0)

print(p.x, p.y)

print(a)

del p
