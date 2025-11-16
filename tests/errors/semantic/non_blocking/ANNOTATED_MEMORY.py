# An object cannot be both an alias to an object stored elsewhere and a stack allocated object.
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from typing import Annotated

a : "Annotated[int[:], 'stack', 'alias']"

