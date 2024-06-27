# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import kernel
from pyccel import cuda

@kernel
def say_hello(its_morning : bool):
    if(its_morning):
        print("Hello and Good morning")
    else:
        print("Hello and Good afternoon")

def f():
    its_morning = True
    say_hello[1,1](its_morning)
    cuda.synchronize()

if __name__ == '__main__':
    f()

