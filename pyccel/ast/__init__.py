# -*- coding: UTF-8 -*-

# TODO [YG, 12.03.2020] Avoid!
# These imports currently cause an undefined behavior because:
#   1. Most modules here do not define the __all__ variable;
#   2. Multiple classes have the same name.

from .basic          import *
from .builtins       import *
from .datatypes      import *
from .core           import *
from .macros         import *
from .headers        import *
from .numpyext       import *
from .f2py           import *
from .fortran        import *
from .functionalexpr import *
from .utilities      import *
from .parallel       import *

