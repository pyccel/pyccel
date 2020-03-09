import os
from fffi import FortranWrapper

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
    wrapper = FortranWrapper(sharedlib_modname, "extern void _f(int);",
                             object_files)
    wrapper.compile(tmpdir=fffi_dir, verbose=1)

    return os.path.join(fffi_dir, wrapper.target)
