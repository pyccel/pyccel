# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class Point(object):
    def __init__(self : 'Point', x):
        self.x = x

    def __del__(self : 'Point'):
        pass
