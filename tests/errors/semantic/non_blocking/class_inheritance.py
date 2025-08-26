# Couldn't find class NonExistentSuperClass in scope
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

class Point(NonExistentSuperClass): # pylint: disable=undefined-variable
    def __init__(self : 'Point'):
        pass
