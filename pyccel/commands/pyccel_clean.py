# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing scripts to remove pyccel generated objects
"""
import os
import shutil
import sysconfig

ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

def pyccel_clean(path_dir = None, recursive = True, remove_shared_libs = False):
    """ Remove __pyccel__ and __epyccel__ folders as well
    as any python shared libraries from the directory path_dir

    Parameters
    ----------
    path_dir  : str
                The path to the folder which should be cleaned
                Default : current working directory
    recursive : bool
                Indicates whether the function should recurse
                into sub-folders
                Default : True
    remove_shared_libs : bool
                Indicates whether shared libraries generated
                by python should also be removed from the
                directory path_dir
    """
    if path_dir is None:
        path_dir = os.getcwd()

    files = os.listdir(path_dir)
    for f in files:
        file_name = os.path.join(path_dir,f)
        if f in  ("__pyccel__", "__epyccel__"):
            shutil.rmtree( file_name, ignore_errors=True)
        elif not os.path.isfile(file_name) and recursive:
            pyccel_clean(file_name, recursive, remove_shared_libs)
        elif remove_shared_libs and f.endswith(ext_suffix):
            os.remove(file_name)
