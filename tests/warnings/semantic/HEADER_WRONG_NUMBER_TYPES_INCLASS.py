#$ header class Point(public)
#$ header method __init__(Point, int, int)
#$ header method incr(Point, int, int)

class Point(object):
    def __init__(self, x):
        return x

    def incr(self, x):
        x   =  x + 1
