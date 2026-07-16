# pylint: disable=missing-function-docstring, missing-module-docstring
import sys

import numpy as np
from numpy import finfo, iinfo

min_int8 = iinfo(np.int8).min
max_int8 = iinfo(np.int8).max

min_int16 = iinfo(np.int16).min
max_int16 = iinfo(np.int16).max

# Use int32 for Windows compatibility
min_int = iinfo(np.int32).min
max_int = iinfo(np.int32).max

min_int32 = iinfo(np.int32).min
max_int32 = iinfo(np.int32).max

min_int64 = iinfo(np.int64).min
max_int64 = iinfo(np.int64).max

min_abs_float = sys.float_info.min
min_float = finfo(float).min
max_float = finfo(float).max

min_float32 = finfo(np.float32).min / 2
max_float32 = finfo(np.float32).max / 2

min_float64 = finfo(np.float64).min / 2
max_float64 = finfo(np.float64).max / 2

# Relative and absolute tolerances for array comparisons in the form
# numpy.isclose(a, b, rtol, atol). Windows has larger round-off errors.
if sys.platform == "win32":
    RTOL = 1e-13
    ATOL = 1e-14
else:
    RTOL = 2e-14
    ATOL = 1e-15

RTOL32 = 1e-5
ATOL32 = 1e-6
