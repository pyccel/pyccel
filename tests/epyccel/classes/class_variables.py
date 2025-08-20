# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import numpy as np

class A:
    x : int
    def __init__(self : 'A', b : bool):
        self.x = 3
        self.z : float = 10.0
        if b:
            self.y = np.ones(5, dtype=int)
        else:
            self.y = np.ones(7, dtype=int)

    def get_4(self : 'A'):
        return 4

    def get_y_len(self):
        return len(self.y)

if __name__ == '__main__':
    myA : 'A' = A(True)

    print(myA.x)
