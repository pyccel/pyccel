from .version_4_5 import Openmp as Openmp_v4_5
from .version_5_0 import Openmp as Openmp_v5_0
from .version_5_1 import Openmp as Openmp_v5_1

__all__ = (
    'openmp_versions',
)

openmp_versions = {
    '4.5': Openmp_v4_5,
    '5.0': Openmp_v5_0,
    '5.1': Openmp_v5_1,
}
