# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from pyccel.decorators import inline

#An example of a class
class Shape:

    def __init__(self : 'Shape', x : float, y : float):
        self.x = x
        self.y = y
        self.description = "This shape has not been described yet"
        self.author = "Nobody has claimed to make this shape yet"

    @property
    def area(self : 'Shape') -> float:
        y = self.x * self.y
        return y

    @property
    def perimeter(self : 'Shape') -> float:
        x = 2 * self.x + 2 * self.y
        return x

    def describe(self : 'Shape', text : str):
        self.description = text

    def authorName(self : 'Shape', text : str):
        self.author = text

    def scaleSize(self : 'Shape', scale : float):
        self.x = self.x * scale
        self.y = self.y * scale

rectangle = Shape(100., 45.)
#finding the area of your rectangle:
print(rectangle.area)

@inline
def f(t : int):
    x = 5*t
    return x

y = f(6)

def g(z : int):
    x= 5+z
    return x

