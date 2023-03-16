""" Tools to help examine git information and interact with git
"""
from collections import namedtuple
from datetime import datetime
import json
import shutil
import subprocess

github_cli = shutil.which('gh')

ReviewComment = namedtuple('ReviewComment', ['state', 'date'])

def get_diff_as_json(filename):
    """
    A function which converts the output of a reduced git diff call
    to a dictionary that can be exported using json.
    The diff call should use the argument `--unified=0`

    Parameters
    ----------
    filename : str
            The file where the diff was printed

    Returns
    -------
    changes : dict
            A dictionary whose keys are files which have been
            changed in this branch and whose values are a dictionary.
            The dictionary is itself a dictionary with the keys 'addition'
            and 'deletion' whose values are lists containing the line
            numbers of lines which have been changed/added (addition) or
            changed/deleted (deletion)
    """
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]
    changes ={}
    i = 0
    n = len(lines)

    current_file_name = None
    current_file_additions = []
    current_file_deletions = []

    while i < n:
        l = lines[i]
        if l.startswith("diff "):
            if current_file_name:
                changes[current_file_name] = {}
                changes[current_file_name]['addition'] = current_file_additions
                changes[current_file_name]['deletion'] = current_file_deletions
                current_file_additions = []
                current_file_deletions = []
            current_file_name = l.split(' ')[3][2:]
            i+=1
        elif l.startswith('@@'):
            line_info = l.split('@@')[1].split()
            for info in line_info:
                key = info[0]
                info = info[1:]
                if ',' in info:
                    line_num, n_lines = [int(li) for li in info.split(',')]
                else:
                    n_lines = 1
                    line_num = int(info)
                if key == '+':
                    insert_index = line_num
                    n_append = n_lines
                elif key == '-':
                    delete_index = line_num
                    n_delete = n_lines
            i+=1
            j=0
            while j<n_delete and lines[i].startswith('-'):
                current_file_deletions.append(delete_index+j)
                j+=1
                i+=1
            assert n_delete == j
            while i<n and lines[i].startswith('\\'):
                i+=1
            j=0
            while j<n_append and lines[i].startswith('+'):
                current_file_additions.append(insert_index+j)
                j+=1
                i+=1
            assert n_append == j
        else:
            print(lines[i])
            i+=1

    if current_file_name:
        changes[current_file_name] = {}
        changes[current_file_name]['addition'] = current_file_additions
        changes[current_file_name]['deletion'] = current_file_deletions

    return changes


def get_pr_number():
    """
    Check if this branch has exactly 1 related PR.

    Use GitHub's command-line tool to get the PR number for
    all PRs related to this branch. Output a message to clarify
    which PR is considered relevant.

    Results
    -------
    bool : Indicates whether this branch has exactly 1 related PR.
    """
    cmds = [github_cli, 'pr', 'status', '--json', 'number']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    output = json.loads(result)

    if 'currentBranch' not in output:
        return []
    else:
        PRs = output['currentBranch']['number']
        if isinstance(PRs, list):
            return PRs
        else:
            return PRs



def get_labels():
    """
    Get the labels associated with the PR.

    Use GitHub's command-line tool to get all labels for
    all PRs related to this branch.

    Results
    -------
    list : A list of the names of all the labels.
    """
    cmds = [github_cli, 'pr', 'status', '--json', 'labels']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    label_json = json.loads(result)['currentBranch']['labels']
    current_labels = [l['name'] for l in label_json]
    return current_labels



def is_draft():
    """
    Get the draft status of the PR.

    Use GitHub's command-line tool to get the draft status for
    the first PR related to this branch.

    Results
    -------
    bool : The draft status of the PR.
    """
    cmds = [github_cli, 'pr', 'status', '--json', 'isDraft']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    return json.loads(result)['currentBranch']['isDraft']



def get_review_status():
    """
    Get the reviews left on the PR.

    Get a dictionary describing the current state of the reviews
    on the PR. 

    Results
    -------
    dict : Keys are authors of reviews, values are the state of their review.
    """
    cmds = [github_cli, 'pr', 'status', '--json', 'reviews']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    reviews = json.loads(result)['currentBranch']['reviews']

    cmds = [github_cli, 'pr', 'status', '--json', 'reviewRequests']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    requests = json.loads(result)['currentBranch']['reviewRequests']
    requested_authors = [r["login"] for r in requests]

    review_status = {}
    for r in reviews:
        author = r['author']['login']
        date = datetime.fromisoformat(r['submittedAt'].strip('Z'))
        state = r['state']
        if author not in review_status:
            review_status[author] = ReviewComment(state, date)
        elif state != 'COMMENTED' and review_status[author].date < date:
            review_status[author] = ReviewComment(state, date)
    for a in review_status:
        if a in requested_authors:
            review_status[a] = ReviewComment('REVIEW_REQUESTED', review_status[a].date)
    for a in requests:
        if a not in review_status:
            review_status[a] = ReviewComment('UNRESPONSIVE', None)
    return review_status, requested_authors


def check_passing():
    """
    Check if tests are passing for this PR.

    Results
    -------
    bool : Indicates if tests are passed
    """

    # Wait till tests have finished
    cmds = [github_cli, 'pr', 'checks', '--required', '--watch']

    p = subprocess.Popen(cmds)
    p.communicate()


    # Collect results
    cmds = [github_cli, 'pr', 'status', '--json', 'statusCheckRollup']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    checks = json.loads(result)['currentBranch']['statusCheckRollup']
    passing = all(c['conclusion'] == 'SUCCESS' for c in checks if c['name'] not in ('CoverageChecker', 'Check labels', 'Welcome'))

    return passing



def leave_comment(number, comment, allow_duplicate):
    """
    Leave a comment on the PR.

    Use GitHub's command-line interface to leave a comment
    on the request PR related to this branch.

    Parameters
    ----------
    number : int
        The number of the PR.

    comment : str
        The comment which should be left on the PR.

    allow_duplicate : bool
        Allow the bot to post the same message twice in a row.
    """
    if not allow_duplicate:
        previous_comments, last_comment, last_date = check_previous_comments()
        ok_to_print = last_comment != comment
    else:
        ok_to_print = allow_duplicate

    if ok_to_print:
        cmds = [github_cli, 'pr', 'comment', str(number), '-b', f'"{comment}"']

        p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
        result, err = p.communicate()
    else:
        print("Not duplicating comment:")
        print(comment)


def remove_labels(number, labels):
    """
    Remove the specified labels from the PR.

    Use GitHub's command-line interface to remove all the specified
    labels from a PR.

    Parameters
    ----------
    number : int
        The number of the PR.

    labels : list
        A list of the labels to be removed.
    """

    cmds = [github_cli, 'pr', 'edit', str(number)]
    for lab in labels:
        cmds += ['--remove-label', lab]

    p = subprocess.Popen(cmds)
    p.communicate()


def set_draft(number):
    """
    Set PR to draft.

    Use GitHub's command-line interface to set the PR to a draft.

    Parameters
    ----------
    number : int
        The number of the PR.
    """
    cmds = [github_cli, 'pr', 'ready', str(number), '--undo']

    p = subprocess.Popen(cmds)
    p.communicate()
