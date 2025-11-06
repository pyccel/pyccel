# An object cannot be both a pointer to an object stored elsewhere and a stack allocated object.
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from typing import Annotated
import numpy as np

a : Annotated['int[:]', 'stack', 'pointer']

