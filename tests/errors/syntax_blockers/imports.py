# pylint: disable=missing-function-docstring, missing-module-docstring

# should not pass
import foo.bar.baz as stuff
import numpy as np

def g():
    import blabla

for i in [0,1,2]:
    import inside_loop

if True:
    import inside_if

while True:
    import inside_while

class Point(object):
    import inside_class

    def __init__(self, x):
        self.x = x

from foo import *
