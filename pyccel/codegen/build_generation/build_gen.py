"""
Module describing things that are in common between different build system handlers.
"""

class BuildSystemHandler:
    """
    The superclass from which build system handlers should inherit.

    The superclass from which build system handlers should inherit.
    This class exists to ensure that all build system handlers have
    the same constructor.

    Parameters
    ----------
    pyccel_dir : Path
        The directory where generated files should be outputted.
    root_dir : Path
        The directory from which the pyccel-make command was called.
    verbose : int
        The level of verbosity.
    debug_mode : bool
        Indicates if we should compile in debug mode.
    compiler : Compiler
        The compiler that should be used to compile the code.
    accelerators : iterable[str]
        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
    """
    def __init__(self, pyccel_dir, root_dir, verbose, debug_mode, compiler, accelerators):
        self._pyccel_dir = pyccel_dir
        self._root_dir = root_dir
        self._verbose = verbose
        self._debug_mode = debug_mode
        self._accelerators = accelerators
