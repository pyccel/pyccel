# pylint: disable=missing-module-docstring
import sys

import numpy as np
from numpy import finfo, iinfo

min_int8 = iinfo("int8").min
max_int8 = iinfo("int8").max

min_int16 = iinfo("int16").min
max_int16 = iinfo("int16").max

# Use int32 for Windows compatibility
min_int = iinfo(np.int32).min
max_int = iinfo(np.int32).max

min_int32 = iinfo("int32").min
max_int32 = iinfo("int32").max

min_int64 = iinfo("int64").min
max_int64 = iinfo("int64").max

min_float = finfo("float").min
max_float = finfo("float").max

min_float32 = finfo("float32").min / 2
max_float32 = finfo("float32").max / 2

min_float64 = finfo("float64").min / 2
max_float64 = finfo("float64").max / 2

default_numpy_int = np.array([1]).dtype

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
