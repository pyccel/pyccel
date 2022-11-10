import os
import defusedxml.ElementTree as ET

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

    content_lines = {}
    changes = {}

    for f in root.findall('.//class'):
        filename = f.attrib['filename']
        lines = f.findall('lines')[0].findall('line')
        all_lines = [int(l.attrib['number']) for l in lines]
        untested_lines = [int(l.attrib['number']) for l in lines if l.attrib['hits'] == "0"]
        changes[os.path.join('pyccel',filename)] = untested_lines
        content_lines[os.path.join('pyccel',filename)] = all_lines

    return changes, content_lines

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
        with open(f, encoding="ascii") as filename:
            f_lines = filename.readlines()
        lines = [f_lines[i-1].strip() for i in line_nums]
        lines = [l for l in lines if not l.startswith('raise ')]
        if len(lines):
            reduced_untested[f] = lines

    return reduced_untested

def print_markdown_summary(untested, content_lines, commit, output):
    if len(untested) == 0:
        return "## All new python code in the pyccel package is fully tested! :tada:"
    else:
        md_string = "## The new code is not fully tested\n"
        for f, lines in untested:
            md_string += f"### {f}\n"
            line_indices = content_lines[f]
            n_code_lines = len(line_indices)
            i = 0
            while i<len(lines):
                start_line = lines[i]
                j = line_indices.find(start_line)
                while lines[i] == line_indices[j]:
                    i+=1
                    j+=1
                if j < n_code_lines-1:
                    end_line = line_indices[j+1]-1
                else:
                    end_line = line_indices[j]
                md_string += "https://github.com/pyccel/pyccel/blob/"+commit+f+f"#L{start_line}-{end_line}"

    print(md_string, file=open(output, "w"))

def show_results(untested):
    for f, lines in untested:
        print(f"In file {f} the following lines are untested : {lines}")

    assert len(untested) == 0
