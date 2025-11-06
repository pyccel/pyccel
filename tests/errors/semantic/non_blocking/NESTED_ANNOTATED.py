# Nested Annotated[] type modifiers are not handled.
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from typing import Annotated
import numpy as np

a : Annotated[Annotated['int[:]', 'stack'], '>0']
