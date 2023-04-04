# pylint: disable=missing-function-docstring, missing-module-docstring
class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def translate(self, a, b):
        self.x = self.x + a
        self.y = self.y + b


class Point2(object):
    import inside_class

    def __init__(self, x):
        self.x = x
