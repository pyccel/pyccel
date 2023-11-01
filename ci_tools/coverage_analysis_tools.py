""" Functions for comparing coverage output and git diff output
"""
import sys
import defusedxml.ElementTree as ET

def get_untested_lines(coverage_filename):
    """
    Get all untested lines from a coverage output.

    Parse a coverage xml file and return a dictionary containing the files and lines
    which are untested.

    Parameters
    ----------
    coverage_filename : str
        The name of the xml file containing the coverage information.

    Returns
    -------
    no_coverage : dict
            A dictionary whose keys are the files in pyccel
            and whose values are lists containing the line numbers
            where coverage is lacking in that file.
    content_lines : dict
            A dictionary whose keys are the files in pyccel
            and whose values are lists containing the line numbers
            where a python command starts (this excludes comments,
            empty lines, and lines which are continuations of
            previous lines).
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
        no_coverage[filename] = untested_lines
        content_lines[filename] = all_lines

    return no_coverage, content_lines

def compare_coverage_to_diff(coverage, diff):
    """
    Compare coverage to git diff.

    Compare dictionaries containing coverage information and git
    diff information to find untested lines which have been added
    to the code base.

    Parameters
    ----------
    coverage : dict
            A dictionary whose keys are the files in pyccel
            and whose values are lists containing the line numbers
            where coverage is lacking in that file.
    diff : dict
            A dictionary whose keys are files which have been
            changed in this branch and whose values are a dictionary.
            The dictionary must contain a key 'addition' whose value
            is a list containing the line numbers of lines which have
            been changed/added.

    Returns
    -------
    dict
            A dictionary whose keys are the files in pyccel with
            untested lines which have been added in this branch
            and whose values are lists containing the line numbers
            where coverage is lacking in that file.
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

def allow_untested_debug_code(untested):
    """
    Remove `str` and `repr` functions from dictionary of untested lines.

    Takes a dictionary describing untested lines and returns an
    equivalent dictionary without lines designed to print a class
    (should only be used for debugging).

    Parameters
    ----------
    untested : dict
            A dictionary whose keys are the files in pyccel with
            untested lines which have been added in this branch
            and whose values are lists containing the line numbers
            where coverage is lacking in that file.

    Returns
    -------
    dict
            A dictionary which is a copy of the input dictionary
            without the lines which express raise statements.
    """
    reduced_untested = {}
    for f,line_nums in untested.items():
        with open(f, encoding="utf-8") as filename:
            f_lines = filename.readlines()
        for l in line_nums:
            line = f_lines[l-1]
            n = len(line)-len(line.lstrip())
            i = l-1
            func_found = ''
            while i >= 0 and not func_found:
                line = f_lines[i]
                strip_line = line.lstrip()
                n_indent = len(line)-len(strip_line)
                if n_indent < n and strip_line.startswith('def '):
                    func_found = strip_line.split()[1].split('(')[0]
                else:
                    if n_indent < n and strip_line!='':
                        n = n_indent
                    i-=1
            if func_found not in ('__repr__', '__str__'):
                reduced_untested.setdefault(f, []).append(l)

    return reduced_untested

def allow_untested_error_calls(untested):
    """
    Remove error calls from untested lines.

    Takes a dictionary describing untested lines and returns an
    equivalent dictionary without lines designed to raise errors.

    Parameters
    ----------
    untested : dict
            A dictionary whose keys are the files in pyccel with
            untested lines which have been added in this branch
            and whose values are lists containing the line numbers
            where coverage is lacking in that file.

    Returns
    -------
    dict
            A dictionary which is a copy of the input dictionary
            without the lines which express raise statements.
    """
    reduced_untested = {}
    for f,line_nums in untested.items():
        with open(f, encoding="utf-8") as filename:
            f_lines = filename.readlines()
        untested_lines = [(i, f_lines[i-1].strip()) for i in line_nums]
        lines = [i for i,l in untested_lines if not (l.startswith('raise ') or \
                                                     l.startswith('errors.report(') or \
                                                     l.startswith('return errors.report(') or \
                                                     l.startswith('except'))]
        if len(lines):
            reduced_untested[f] = lines

    return reduced_untested

def print_markdown_summary(untested, commit, output, repo):
    """
    Print the results neatly in markdown in a provided file.

    Print the results neatly in markdown in a provided file such that they can
    be printed in a GitHub output file.

    Parameters
    ----------
    untested : list of dict
        A list of dictionaries describing all lines with unacceptable coverage.
    commit : str
        The commit being tested.
    output : str
        The file where the markdown summary should be printed.
    repo : str
        The repository where the pull request can be found.
    """
    if len(untested) == 0:
        md_string = "## Congratulations! All new python code in the pyccel package is fully tested! :tada:"
    else:
        md_string = "## Warning! The new code is not run\n"
        current_file = None
        for c in untested:
            f = c['path']
            if f!= current_file:
                md_string += f"### {f}\n"
                current_file = f
            start_line = c.get('start_line', c['line'])
            md_string += f"https://github.com/{repo}/blob/{commit}/{f}#L{start_line}-L{c['line']}\n"

    with open(output, "a", encoding="utf-8") as out_file:
        print(md_string, file=out_file)

def get_json_summary(untested, content_lines, existing_comments, diff):
    """
    Print the results neatly in json in a provided file.

    Print the results neatly in json in a provided file such that
    each error can be described by a GitHub annotation.

    Parameters
    ----------
    untested : dict
        Dictionary whose keys are the files in pyccel with untested
        lines which have been added in this branch and whose values
        are lists containing the line numbers where coverage is
        lacking in that file.
    content_lines : dict
        Dictionary whose keys are the files in pyccel and whose
        values are lists containing the line numbers where python
        commands begin.
    existing_comments : list of dict
        A list describing all comments previously left about the
        coverage results.
    diff : dict
        A dictionary whose keys are files which have been
        changed in this branch and whose values are a dictionary.
        The dictionary must contain a key 'addition' whose value
        is a list containing the line numbers of lines which have
        been changed/added.

    Returns
    -------
    old_comments : list of dict
        The coverage issues which were present before this commit and
        had already been commented on.
    new_comments : list of dict
        The coverage issues which were not present before this commit
        and have never been commented on.
    fixed_comments : list of dict
        The coverage issues which were present before this commit and
        had already been commented on but are no longer present.
    """
    message = "This code isn't tested. Please can you take a look"
    print(diff)
    print(untested)
    new_comments = []
    old_comments = []
    fixed_comments = existing_comments.copy()
    for f, lines in untested.items():
        line_indices = content_lines[f]
        n_code_lines = len(line_indices)
        n_untested = len(lines)
        print(line_indices, n_code_lines, n_untested)
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
            last_valid_line = start_line
            last_valid_line_idx = next(i for i in diff[f]['addition'] if i == last_valid_line)
            for i in diff[f]['addition'][last_valid_line_idx:]:
                if i-last_valid_line < 6:
                    last_valid_line = i
                else:
                    break
            end_line = min(end_line, last_valid_line)
            output = {'path':f, 'line':end_line, 'body':message}
            if start_line != end_line:
                output['start_line'] = start_line
            if (f,end_line) in existing_comments:
                old_comments.append(output)
                fixed_comments.pop((f,end_line))
            else:
                new_comments.append(output)

    return old_comments, new_comments, fixed_comments

def show_results(untested):
    """
    Print the results and fail if coverage is lacking.

    Print a list of all untested lines and exit the program
    with exit code 1 if the coverage is incomplete.

    Parameters
    ----------
    untested : dict
        Dictionary whose keys are the files in pyccel with untested
        lines which have been added in this branch and whose values
        are lists containing the line numbers where coverage is
        lacking in that file.
    """
    for f, lines in untested.items():
        print(f"In file {f} the following lines are untested : {lines}")

    if len(untested) != 0:
        sys.exit(1)

def check_if_coverage_ignored(comment_json, existing_comments):
    """
    Check if a reply was left indicating that the coverage can be ignored.

    For a given coverage problem discovered on a previous run, check if a
    reply to the comment was left indicating that the coverage can be
    ignored.

    Parameters
    ----------
    comment_json : dict
        A dictionary describing a coverage problem which was found on a
        previous run.

    existing_comments : dict
        A dictionary whose keys are a tuple containing the file and the
        relevant line in the current version of the code, and whose values
        are a list of comments and replies previously left on this blob.

    Returns
    -------
    bool
        True if the coverage issue should be ignored, false otherwise.
    """
    key = (comment_json['path'], comment_json['line'])
    comment = existing_comments[key]
    print(comment)
    return any('/bot accept' in c['body'] for c in comment)

def evaluate_success(old_comments, new_comments, existing_comments):
    """
    Determine if the coverage check was successful.

    Use the old comments, the new comments, and any replies to old comments
    to determine whether the coverage check passes. The check passes if
    there are no coverage issues, or if all the coverage issues are handled
    via replies beginning with "/bot accept"

    Parameters
    ----------
    old_comments : list of dict
        A list of dictionaries describing any coverage problems which were
        already found on a previous run.

    new_comments : list of dict
        A list of dictionaries describing any coverage problems which were
        discovered during this run.

    existing_comments : dict
        A dictionary whose keys are a tuple containing the file and the
        relevant line in the current version of the code, and whose values
        are a list of comments and replies previously left on this blob.

    Returns
    -------
    bool
        True if the test succeeded, false otherwise.
    """
    if new_comments:
        return False

    if len(old_comments) == 0:
        return True

    return all(check_if_coverage_ignored(r, existing_comments) for r in old_comments)
