import os
from fffi import FortranLibrary, FortranModule

def create_shared_library(codegen,
                          pyccel_dirpath,
                          compiler,
                          mpi_compiler,
                          accelerator,
                          dep_mods,
                          extra_args='',
                          sharedlib_modname=None):

    module_name = codegen.name

    object_files = ['{}.o'.format(m) for m in dep_mods]
    if sharedlib_modname is None:
        sharedlib_modname = module_name


    fffi_dir = os.path.join(pyccel_dirpath, '__fffi__')
    lib = FortranLibrary(
        sharedlib_modname, compiler={'name': 'gfortran', 'version': 9})
    mod = FortranModule(lib, module_name)

    # TODO: do some Fortran definitions based on structure
    mod.fdef(
        """\
            subroutine f(x)
                integer(kind=4), intent(in)  :: x
            end subroutine f
        """
    )
    lib.compile(
        tmpdir=fffi_dir, verbose=1, skiplib=True, extra_objects=object_files)

    return os.path.join(fffi_dir, '_' + lib.name + '.so')
