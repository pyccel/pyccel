""" File containing functionalities which help output the CI annotations easily
"""
import importlib
import inspect
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

    text : list
        A list of strings where the output should also be saved.
    """
    print(*args)
    text.append(' '.join(args)+'\n')

def get_code_file_and_lines(obj, pyccel_folder, mod_name = None):
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

    pyccel_folder : str
        The folder containing the pyccel repo.

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
    if not pyccel_folder:
        pyccel_folder = os.getcwd()

    obj_parts = obj.split('.')
    if mod_name is None:
        idx = len(obj_parts)
        print(pyccel_folder, obj)
        filename = os.path.join(pyccel_folder, '/'.join(obj_parts[:idx])+'.py')
        while idx > 0 and not os.path.isfile(filename):
            idx -= 1
            filename = os.path.join(pyccel_folder, '/'.join(obj_parts[:idx])+'.py')
        assert idx != 0
        mod_name = '.'.join(obj_parts[:idx])
        obj_parts = obj_parts[idx:]

    mod = importlib.import_module(mod_name)
    filename = mod.__file__.split('/')
    file = os.path.relpath(mod.__file__, pyccel_folder)

    if obj_parts:
        # Get the object
        obj = mod
        for o in obj_parts:
            obj = getattr(obj, o)

        # If the object is a class property, get the underlying function
        obj = getattr(obj, 'fget', obj)

        source, start_line = inspect.getsourcelines(obj)
        length = len(source)
        return file, start_line, start_line+length-1
    else:
        # Module
        return file, 1, 1

def locate_code_blocks(lines):
    """
    Find all code blocks in a markdown file.

    From a list of strings describing the lines in a markdown
    file, find a list of the start and end line numbers for
    each of the code blocks.

    Parameters
    ----------
    lines : list of str
        A list of the lines in the file.

    Returns
    -------
    list of 2-tuple of ints
        A list of the start and end line numbers for each of the
        code blocks.
    """
    stripped_lines = [l.strip() for l in lines]
    code_block_indexes = [i for i, l in enumerate(stripped_lines,1) if l.startswith('```')]
    nblock_indexes = len(code_block_indexes)
    assert nblock_indexes % 2 == 0
    nblocks = nblock_indexes // 2
    return [b for b in zip(code_block_indexes[::2], code_block_indexes[1::2])]

def is_text(line, start, end, line_number, code_blocks):
    """
    Determine if a word in a string is text which should be checked.

    When checking spelling it is important to filter out code snippets
    and urls. This function takes a line, information about the position
    of code blocks in the file and the indices of the word being
    investigated and determines whether the word is relevant or not.

    Parameters
    ----------
    line : str
        The line of interest.

    start : int
        The index of the start of the word being investigated.

    end : int
        The index of the end of the word being investigated.

    line_number : int
        The line number.

    code_blocks : list of 2-tuple of ints
        The result of a call to locate_code_blocks.

    Returns
    -------
    bool
        True if the word is in text, False if it is in a url or a code snippet.
    """
    if any(c[0] <= line_number <= c[1] for c in code_blocks):
        return False
    else:
        n = len(line)
        idx = -1
        last_block_was_text = False
        in_link = False
        in_url = False
        while idx < start:
            if in_link:
                link_idx = line[idx+1:].find(')')
                assert link_idx != -1
                code_idx = n
                url_idx = n
            elif in_url:
                url_idx = line[idx+1:].find('>')
                assert url_idx != -1
                code_idx = n
                link_idx = n
            else:
                code_idx = line[idx+1:].find('`')
                link_idx = line[idx+1:].find('](')
                url_idx = line[idx+1:].find('<')
                if code_idx == -1:
                    code_idx = n
                if link_idx == -1:
                    link_idx = n
                if url_idx == -1:
                    url_idx = n

            nearest_match = min(code_idx, link_idx, url_idx)

            if nearest_match == url_idx:
                in_url = not in_url
            elif nearest_match == link_idx:
                in_link = not in_link
            idx += nearest_match+1
            last_block_was_text = not last_block_was_text

        return last_block_was_text
