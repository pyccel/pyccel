# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
#$ header class Point(public)
#$ header method __init__(Point)

class Point(NonExistantSuperClass): # pylint: disable=undefined-variable
    def __init__(self):
        pass
