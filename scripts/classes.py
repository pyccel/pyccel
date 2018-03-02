class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def translate(self, x, y, a, b):
        self.x = self.x + a
        self.y = self.y + b
