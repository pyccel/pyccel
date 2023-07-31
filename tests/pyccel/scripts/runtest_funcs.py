# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import pure
@pure
def add2(x : 'int', y : 'int'):
    return x+y

@pure
def sum_to_n(n : 'int'):
    result=0
    for i in range(1,n+1):
        result = add2(result,i)
    return result

if __name__ == '__main__':
    print(sum_to_n(4))
