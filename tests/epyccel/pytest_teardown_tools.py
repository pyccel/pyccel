""" Tools useful to decrease memory footprint by tearing down over multiple processes
"""

import os
from itertools import chain
from pyccel.epyccel import epyccel

generated_file_stems = []

def run_epyccel(func_or_mod, **kwargs):
    """
    Run epyccel and cache generated file name.

    Run the epyccel command and cache the name of the
    generated file name so it can be deleted later.

    Parameters
    ----------
    func_or_mod : function or module
        The function or module to be pyccelised.

    **kwargs : dict
        Any additional arguments to be passed to epyccel.

    Returns
    -------
    function or module
        The pyccelised version of func_or_mod.
    """
    f = epyccel(func_or_mod, verbose=True, **kwargs)
    mod_name = getattr(f, '__module__', f.__name__)
    print(mod_name)
    generated_file_stems.append(mod_name)
    return f

def clean_test():
    """
    Delete generated files.

    Reduce memory by deleting any files generated using the
    run_epyccel command. The names of the stems of these files
    can be found in the module variable generated_file_stems.
    """
    pass
    #path_dir = os.getcwd()
    #folder_e = os.path.join(path_dir,"__epyccel__")
    #folder_p = os.path.join(folder_e,"__pyccel__")
    #files_e = [(folder_e,f) for f in os.listdir(folder_e) if not os.path.isdir(f)]
    #files_p = [(folder_p,f) for f in os.listdir(folder_p) if not os.path.isdir(f)]
    #to_remove = []
    #print(len(generated_file_stems), len(set(generated_file_stems)))
    #for (folder, filename) in chain(files_e, files_p):
    #    stem = filename.split('.')[0]
    #    if stem in generated_file_stems:
    #        full_file = os.path.join(folder, filename)
    #        print(os.path.getsize(full_file))
    #        os.remove(full_file)
    #        to_remove.append(stem)
    #    elif stem.endswith('_wrapper') and stem[:-8] in generated_file_stems:
    #        full_file = os.path.join(folder, filename)
    #        print(os.path.getsize(full_file))
    #        os.remove(os.path.join(folder, filename))

    #for r in set(to_remove):
    #    generated_file_stems.remove(r)
