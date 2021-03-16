import os
import shutil
import sysconfig

ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

def pyccel_clean(path_dir = None, recursive = True):
    if path_dir is None:
        path_dir = os.getcwd()

    files = os.listdir(path_dir)
    for f in files:
        file_name = os.path.join(path_dir,f)
        if f in  ("__pyccel__", "__epyccel__"):
            shutil.rmtree( file_name, ignore_errors=True)
        elif not os.path.isfile(file_name) and recursive:
            pyccel_clean(file_name)
        elif f.endswith(ext_suffix):
            os.remove(file_name)
