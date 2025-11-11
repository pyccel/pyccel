# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring
import numpy as np
from classes_array_property import A

def main():
    my_a = A(5)
    y = my_a.x
    print(y)

if __name__ == '__main__':
    main()
