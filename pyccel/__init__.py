"""
Pyccel : write Python code, get Fortran speed

Find documentation at pyccel.github.io/pyccel/
"""
from .version import __version__
from .commands.epyccel import epyccel
from .commands.lambdify import lambdify

__all__ = (
    "__version__",
    "epyccel",
    "lambdify",
)
