class A:
    def __init__(self, x : int):
        self.x = x

    def __add__(self, other : int):
        return A(self.x+other)

    def __mul__(self, other : int):
        return A(self.x*other)

    def __iadd__(self, other : int):
        self.x += other
        return self

