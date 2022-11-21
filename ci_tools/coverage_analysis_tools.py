""" Functions for comparing coverage output and git diff output
"""
import os
import sys
import defusedxml.ElementTree as ET

def get_untested_lines(coverage_filename):
    """
    Parse a coverage xml file and return a dictionary containing the files and lines
    which are untested

    Parameters
    ----------
    coverage_filename : str
        The name of the xml file containing the coverage information

    Returns
    -------
    no_coverage : dict
            A dictionary whose keys are the files in pyccel
            and whose values are lists containing the line numbers
            where coverage is lacking in that file
    content_lines : dict
            A dictionary whose keys are the files in pyccel
            and whose values are lists containing the line numbers
            where a python command starts (this excludes comments,
            empty lines, and lines which are continuations of
            previous lines)
    """
    tree = ET.parse(coverage_filename)
    root = tree.getroot()

    content_lines = {}
    no_coverage = {}

    for f in root.findall('.//class'):
        filename = f.attrib['filename']
        lines = f.findall('lines')[0].findall('line')
        all_lines = [int(l.attrib['number']) for l in lines]
        untested_lines = [int(l.attrib['number']) for l in lines if l.attrib['hits'] == "0"]
        no_coverage[os.path.join('pyccel',filename)] = untested_lines
        content_lines[os.path.join('pyccel',filename)] = all_lines

    return no_coverage, content_lines

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

    Returns
    -------
    untested : dict
            A dictionary whose keys are the files in pyccel with
            untested lines which have been added in this branch
            and whose values are lists containing the line numbers
            where coverage is lacking in that file
    """
    untested = {}
    for f,line_info in diff.items():
        if f not in coverage:
            # Ignore non-python files or files in other directories
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

    Returns
    -------
    reduced_untested : dict
            A dictionary which is a copy of the input dictionary
            without the lines which express raise statements
    """
    reduced_untested = {}
    for f,line_nums in untested.items():
        with open(f, encoding="utf-8") as filename:
            f_lines = filename.readlines()
        untested_lines = [f_lines[i-1].strip() for i in line_nums]
        lines = [l for l in untested_lines if not (l.startswith('raise ') or l.startswith('errors.report(') or l.startswith('return errors.report('))]
        if len(lines):
            reduced_untested[f] = lines

    return reduced_untested

def print_markdown_summary(untested, content_lines, commit, output):
    """
    Print the results neatly in markdown in a provided file

    Parameters
    ----------
    untested : dict
        Dictionary whose keys are the files in pyccel with untested
        lines which have been added in this branch and whose values
        are lists containing the line numbers where coverage is
        lacking in that file
    content_lines : dict
        Dictionary whose keys are the files in pyccel and whose
        values are lists containing the line numbers where python
        commands begin
    commit : str
        The commit being tested
    output : str
        The file where the markdown summary should be printed
    """
    if len(untested) == 0:
        md_string = "## Congratulations! All new python code in the pyccel package is fully tested! :tada:"
    else:
        md_string = "## Warning! The new code is not run\n"
        for f, lines in untested.items():
            md_string += f"### {f}\n"
            line_indices = content_lines[f]
            n_code_lines = len(line_indices)
            n_untested = len(lines)
            i = 0
            while i < n_untested:
                start_line = lines[i]
                j = line_indices.index(start_line)
                while j < n_code_lines and i < n_untested and lines[i] == line_indices[j]:
                    i+=1
                    j+=1
                if j < n_code_lines-1:
                    end_line = line_indices[j]-1
                else:
                    end_line = line_indices[j]
                md_string += "https://github.com/pyccel/pyccel/blob/"+commit+"/"+f+f"#L{start_line}-L{end_line}"

    with open(output, "a", encoding="utf-8") as out_file:
        print(md_string, file=out_file)

def show_results(untested):
    """
    Print the results and fail if coverage is lacking

    Parameters
    ----------
    untested : dict
        Dictionary whose keys are the files in pyccel with untested
        lines which have been added in this branch and whose values
        are lists containing the line numbers where coverage is
        lacking in that file
    """
    for f, lines in untested.items():
        print(f"In file {f} the following lines are untested : {lines}")

    if len(untested) != 0:
        sys.exit(1)
