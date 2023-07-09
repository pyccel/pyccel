""" File containing functionalities which help output the CI annotations easily
"""
import importlib
import os

def print_to_string(*args, text):
    """
    Print to a string and to terminal.

    Wrapper around the printing function to print the same text to both
    the terminal and an output string.

    Parameters
    ----------
    *args : tuple
        Positional arguments to print function.

    *kwargs : dict
        Key word arguments to print function.

    text : list
        A list of strings where the output should also be saved.
    """
    print(*args)
    text.append(' '.join(args)+'\n')

def get_code_file_and_lines(obj, base_folder, mod_name = None):
    """
    Get the file and the relevant lines for the object.

    From a string (or two strings) describing an object and its module,
    get the file and the relvant lines where annotations should be
    printed.

    Parameters
    ----------
    obj : str
        The name of the object being examined. If mod_name is provided
        then this is the name of the object inside the module, otherwise
        it may contain the module path.

    base_folder : str
        The name of the folder to which paths are relative.

    mod_name : str, optional
        The python name of the module (relative to the base folder).

    Returns
    -------
    str
        The name of the file where the object is defined relative to the
        base folder.
    int
        The first line of relevant code.
    int
        The last line of relevant code.
    """
    obj_parts = obj.split('.')
    if mod_name is None:
        idx = len(obj_parts)
        filename = os.path.join(base_folder, '/'.join(obj_parts[:idx])+'.py')
        while idx > 0 and not os.path.isfile(filename):
            idx -= 1
            filename = os.path.join(base_folder, '/'.join(obj_parts[:idx])+'.py')
        assert idx != 0
        mod_name = '.'.join(obj_parts[:idx])
        obj_parts = obj_parts[idx:]

    mod = importlib.import_module(mod_name)
    file = os.path.relpath(mod.__file__, base_folder)

    if obj_parts:
        # Get the object
        obj = mod
        for o in obj_parts:
            obj = getattr(obj, o)

        # If the object is a class property, get the underlying function
        method = getattr(method, 'fget', method)

        source, start_line = inspect.getsourcelines(method)
        length = len(source)
        return file, start_line, start_line+length-1
    else:
        # Module
        return file, 1, 1

