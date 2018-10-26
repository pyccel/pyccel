@elemental
@types(float)
def square(x):
    s = x*x
    return s

a = 2.0
b = square(a)
print(b)

x = [1., 2., 3.]
print(square(x))
