# coding: utf-8

#$ header class StopIteration(public, hide)
#$ header method __init__(StopIteration)
#$ header method __del__(StopIteration)
class StopIteration(object):

    def __init__(self):
        pass

    def __del__(self):
        pass

#$ header class Tensor(public, iterable)
#$ header method __init__(Tensor, int, int, int, int, int, int)
#$ header method __del__(Tensor)
#$ header method __iter__(Tensor)
#$ header method __next__(Tensor)
class Tensor(object):

    def __init__(self, sx, sy, ex, ey, dx, dy):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.dx = dx
        self.dy = dy

        self.i = sx
        self.j = sy

    def __del__(self):
        print('> free')

    def __iter__(self):
        self.i = self.sx
        self.j = self.sy

    # to improve
    def __next__(self):
        if (self.i < self.ex) and (self.j < self.ey):
            i = self.i
            self.i = self.i + 1
        else:
            raise StopIteration()

p = Tensor(-2, -6, 3, 5, 1, 1)

for i,j in p:
    print(i,j)
