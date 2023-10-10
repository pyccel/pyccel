""" Tools to help examine git information and interact with git
"""
from collections import namedtuple
from datetime import datetime
import json
import shutil
import subprocess
import time

__all__ = ('github_cli',
           'ReviewComment',
           'get_diff_as_json',
           'get_previous_pr_comments',
           'check_previous_comments',
           'get_labels',
           'get_review_status',
           'leave_comment',
           'add_labels',
           'remove_labels',
           'set_draft',
           'set_ready',
           'get_job_information',
           'check_previous_contributions'
           )

github_cli = shutil.which('gh')

ReviewComment = namedtuple('ReviewComment', ['state', 'date', 'author'])
Comment = namedtuple('Comment', ['body', 'date', 'author'])

def get_status_json(pr_id, tags):
    """
    Get the tagged status of a PR in json format.

    Use the GitHub CLI to investigate a certain property,
    passed as a tag, of a PR. Multiple tags can also be
    provided. In this case the result is a dictionary with
    the tags appearing as keys.
    Otherwise the value is provided directly (but this may itself
    be a dictionary for certain tags). See the [GitHub CLI docs](https://cli.github.com/manual/gh_pr_view)
    for more details.

    Parameters
    ----------
    pr_id : int
        The number of the PR.

    tags : str
        The tag of the information we are enquiring about.
        Multiple tags must be separated by a comma.

    Results
    -------
    dict/str : The output of the request.
    """
    # Check status of PR
    cmds = [github_cli, 'pr', 'view', str(pr_id), '--json', tags]
    with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
        result, err = p.communicate()
    print(err)

    data = json.loads(result)

    if ',' in tags:
        return data
    else:
        return data[tags]

def get_diff_as_json(filename):
    """
    Get a dictionary describing the changes.

    A function which converts the output of a reduced git diff call
    to a dictionary that can be exported using json.
    The diff call should use the argument `--unified=0`
    The result is a dictionary whose keys are files which have been
    changed in this branch and whose values are a dictionary.
    The dictionary is itself a dictionary with the keys 'addition'
    and 'deletion' whose values are lists containing the line
    numbers of lines which have been changed/added (addition) or
    changed/deleted (deletion).

    Parameters
    ----------
    filename : str
        The file where the diff was printed.

    Returns
    -------
    changes : dict
        Dictionary describing changes to the files.
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
        elif l.startswith('++') and '/dev/null' in l:
            current_file_name = None
            i += 1
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

def get_previous_pr_comments(pr_id):
    """
    Get all previous comments left on the PR.

    Get a list of all comments left on the PR as reported by the GitHub
    CLI.

    Parameters
    ----------
    pr_id : int
        The number of the PR.

    Results
    -------
    list : A list of all the messages left on the PR.
    """
    relevant_comments = get_status_json(pr_id, 'comments')

    comments = [Comment(c["body"], datetime.fromisoformat(c['createdAt'].strip('Z')), c['author']['login'])
                        for c in relevant_comments]
    return comments

def check_previous_comments(pr_id):
    """
    Get information about previous comments made by the bot.

    Get a list of all comments left by the bot as well as its most recent comment
    to avoid it repeating itself unnecessarily

    Parameters
    ----------
    pr_id : int
        The number of the PR.

    Results
    -------
    list : A list of all messages left by the bot on this PR.

    str : The last message left by the bot.

    datetime : The last time the bot commented.
    """
    comments = get_previous_pr_comments(pr_id)
    my_comments = [c for c in comments if c.author == 'github-actions']

    if len(my_comments) == 0:
        return [], '', None
    else:
        last_messages = {}
        final_message = my_comments[0].body
        final_date = my_comments[0].date

        for c in my_comments:
            if c.body not in last_messages:
                last_messages[c.body] = c.date
            elif last_messages[c.body] < c.date:
                last_messages[c.body] = c.date

            if final_date < c.date:
                final_message = c.body
                final_date = c.date

        return last_messages, final_message, final_date


def get_labels(pr_id):
    """
    Get the labels associated with the PR.

    Use GitHub's command-line tool to get all labels for
    all PRs related to this branch.

    Results
    -------
    list : A list of the names of all the labels.
    """
    label_json = get_status_json(pr_id, 'labels')
    current_labels = [l['name'] for l in label_json]
    return current_labels



def get_review_status(pr_id):
    """
    Get the reviews left on the PR.

    Get a dictionary describing the current state of the reviews
    on the PR. 

    Results
    -------
    dict : Keys are authors of reviews, values are the state of their review.
    """
    reviews = get_status_json(pr_id, 'reviews')
    requests = get_status_json(pr_id, 'reviewRequests')

    requested_authors = [r["login"] for r in requests]

    review_status = {}
    for r in reviews:
        author = r['author']['login']
        date = datetime.fromisoformat(r['submittedAt'].strip('Z'))
        state = r['state']
        if author not in review_status:
            review_status[author] = ReviewComment(state, date, author)
        elif state != 'COMMENTED' and review_status[author].date < date:
            review_status[author] = ReviewComment(state, date, author)
    for a in review_status:
        if a in requested_authors:
            review_status[a] = ReviewComment('REVIEW_REQUESTED', review_status[a].date, a)
    for a in requested_authors:
        if a not in review_status:
            review_status[a] = ReviewComment('UNRESPONSIVE', None, a)
    return review_status, requested_authors


def get_job_information(run_id):
    """
    Get information about the jobs run.

    Use the GitHub CLI to check all information about the jobs
    related to a workflow.

    Results
    -------
    bool : Indicates if tests are passed
    """
    cmd = [github_cli, 'run', 'view', str(run_id), '--json', 'jobs']
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        result, err = p.communicate()
    print(err)
    return json.loads(result)['jobs']


def leave_comment(number, comment, edit = False):
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

    edit : bool, default: False
        Indicates whether the bot should edit the last comment
        or just write a new comment.
    """
    cmds = [github_cli, 'pr', 'comment', str(number), '-b', comment]

    if edit:
        cmds.append('--edit-last')

    with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
        _,err = p.communicate()
    print(err)



def add_labels(number, labels):
    """
    Add the specified labels to the PR.

    Use GitHub's command-line interface to add all the specified
    labels from a PR.

    Parameters
    ----------
    number : int
        The number of the PR.

    labels : list
        A list of the labels to be added.
    """

    cmds = [github_cli, 'pr', 'edit', str(number)]
    for lab in labels:
        cmds += ['--add-label', lab]

    with subprocess.Popen(cmds) as p:
        _, err = p.communicate()
    print(err)


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

    with subprocess.Popen(cmds) as p:
        _, err = p.communicate()
    print(err)


def set_ready(number):
    """
    Remove draft status from PR.

    Use GitHub's command-line interface to remove the draft status of
    the PR.

    Parameters
    ----------
    number : int
        The number of the PR.
    """
    cmds = [github_cli, 'pr', 'ready', str(number)]

    with subprocess.Popen(cmds) as p:
        _, err = p.communicate()
    print(err)


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

    with subprocess.Popen(cmds) as p:
        _, err = p.communicate()
    print(err)

def check_previous_contributions(repo, author):
    """
    Get a list of previous PRs made on this repository by the author.

    Use the GitHub CLI to get a dictionary listing the number and state
    of all the PRs previously opened by this author on the repository.

    Parameters
    ----------
    repo : str
        The name of the repository (e.g. 'pyccel/pyccel')

    author : str
        The username of the author
    """
    cmds = [github_cli, 'search', 'prs', '--author', author, '--repo', repo, '--json', 'number,state']

    with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
        result, err = p.communicate()
        returncode = p.returncode
    print(err)
    print(returncode)
    ntries = 1
    if returncode:
        while returncode and ntries < 10:
            ntries += 1
            time.sleep(10)
            with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
                result, err = p.communicate()
                returncode = p.returncode
            print("New returncode : ", returncode)


    return json.loads(result)
