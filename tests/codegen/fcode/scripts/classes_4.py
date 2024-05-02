# pylint: disable=missing-function-docstring, missing-module-docstring/
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

    def area(self):
        y = self.x * self.y
        return y

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
print(rectangle.area())

#finding the perimeter of your rectangle:
print(rectangle.perimeter())

#describing the rectangle
rectangle.describe("A wide rectangle, more than twice as wide as it is tall")

#making the rectangle 50% smaller
rectangle.scaleSize(0.5)

#re-printing the new area of the rectangle
print(rectangle.area())
