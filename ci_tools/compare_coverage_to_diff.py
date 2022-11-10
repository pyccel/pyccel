import os
import xml.etree.ElementTree as ET

def get_untested_lines(coverage_filename):
    """
    Parse a coverage xml file and return a dictionary containing the files and lines
    which are untested

    Parameters
    ----------
    coverage_filename : str
        The name of the xml file containing the coverage information

    Result
    ------
    coverage : dict
            A dictionary whose keys are the files in pyccel
            and whose values are lists containing the line numbers
            where coverage is lacking in that file
    """
    tree = ET.parse(coverage_filename)
    root = tree.getroot()
    directories = root.findall('packages')[0].findall('package')

    changes ={}
    current_file_name = None
    current_file_lines = []

    for f in root.findall('.//class'):
        filename = f.attrib['filename']
        lines = f.findall('lines')[0].findall('line')
        untested_lines = [int(l.attrib['number']) for l in lines if l.attrib['hits'] == "0"]
        changes[os.path.join('pyccel',filename)] = untested_lines

    return changes

def compare_coverage_to_diff(coverage, diff):
    """
    Compare dictionaries containing coverage information and git
    diff information to find untested lines which have been added
    to the code base

    Parameters
    ----------
    coverage : dict
            A dictionary whose keys are the files in pyccel
            and whose values are lists containing the line numbers
            where coverage is lacking in that file
    diff     : dict
            A dictionary whose keys are files which have been
            changed in this branch and whose values are a dictionary.
            The dictionary must contain a key 'addition' whose value
            is a list containing the line numbers of lines which have
            been changed/added

    Result
    ------
    untested : dict
            A dictionary whose keys are the files in pyccel with
            untested lines which have been added in this branch
            and whose values are lists containing the line numbers
            where coverage is lacking in that file
    """
    untested = {}
    for f,line_info in diff.items():
        if not f.startswith('pyccel/'):
            continue
        new_lines = line_info['addition']
        untested_lines = coverage[f]
        if any(n in untested_lines for n in new_lines):
            untested[f] = [n for n in new_lines if n in untested_lines]
    return untested

def allow_untested_error_calls(untested):
    """
    Takes a dictionary describing untested lines and returns an
    equivalent dictionary without lines designed to raise errors

    Parameter
    ---------
    untested : dict
            A dictionary whose keys are the files in pyccel with
            untested lines which have been added in this branch
            and whose values are lists containing the line numbers
            where coverage is lacking in that file
    """
    reduced_untested = {}
    for f,line_nums in untested.items():
        with open(f) as filename:
            f_lines = filename.readlines()
        lines = [f_lines[i-1].strip() for i in line_nums]
        lines = [l for l in lines if not l.startswith('raise ')]
        if len(lines):
            reduced_untested[f] = lines

    return reduced_untested
