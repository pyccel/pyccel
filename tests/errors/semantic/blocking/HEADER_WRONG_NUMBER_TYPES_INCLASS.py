# pylint: disable=missing-function-docstring, missing-module-docstring
#$ header class Point(public)
#$ header method __init__(Point)
#$ header method __del__(Point)

class Point(object):
    def __init__(self, x):
        self.x = x

    def __del__(self):
        pass
