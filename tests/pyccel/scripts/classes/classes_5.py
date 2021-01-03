# pylint: disable=missing-class-docstring,  disable=missing-function-docstring, missing-module-docstring/
#$ header class A(public)
#$ header method __init__(A, int)
#$ header method __del__(A)
#$ header method f(A)

class A(object):
    def __init__(self, x):
        self.x = x

    def __del__(self):
        pass

    def f(self):
        return self.x+2

def get_A():
    a_cls = A(3)
    return a_cls

b = get_A().x
c = get_A().f()

print(b)
print(c)
