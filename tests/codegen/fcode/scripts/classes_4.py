# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
#An example of a class

class Shape:

    def __init__(self : Shape, x : float, y : float):
        self.x = x
        self.y = y
        self.description = "This shape has not been described yet"
        self.author = "Nobody has claimed to make this shape yet"

    def area(self : Shape) -> float:
        y = self.x * self.y
        return y

    def perimeter(self : Shape) -> float:
        x = 2 * self.x + 2 * self.y
        return x

    def describe(self : Shape, text : str):
        self.description = text

    def authorName(self : Shape, text : str):
        self.author = text

    def scaleSize(self : Shape, scale : float):
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
