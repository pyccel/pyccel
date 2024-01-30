import numpy as np

class A:
    def __init__(self, n : int):
        self.x = np.ones(n)

    def get_x(self):
        return self.x
