# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
#An example of a class
#$ header class Shape(public)
#$ header method __init__(Shape, double, double)
#$ header method area(Shape) results(double)
#$ header method perimeter(Shape) results(double)
#$ header method describe(Shape,str)
#$ header method authorName(Shape,str)
#$ header method scaleSize(Shape, double)

class Shape:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.description = "This shape has not been described yet"
        self.author = "Nobody has claimed to make this shape yet"

    @property
    def area(self):
        y = self.x * self.y
        return y

    @property
    def perimeter(self):
        x = 2 * self.x + 2 * self.y
        return x

    def describe(self, text):
        self.description = text

    def authorName(self, text):
        self.author = text

    def scaleSize(self, scale):
        self.x = self.x * scale
        self.y = self.y * scale

rectangle = Shape(100., 45.)
#finding the area of your rectangle:
print(rectangle.area)

#$ header function f(int)
#@inline
#def f(t):
#    x = 5*t
#    return x
# y = f(6)

#$ header function g(int)
@vectorize(z)
def g(z):
    x= 5+z
    return x



