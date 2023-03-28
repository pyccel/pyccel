""" Tools to help examine git information and interact with git
"""
from collections import namedtuple
from datetime import datetime
import json
import shutil
import subprocess

__all__ = ('github_cli',
           'ReviewComment',
           'get_diff_as_json',
           'get_previous_pr_comments',
           'check_previous_comments',
           'get_pr_number',
           'get_labels',
           'is_draft',
           'get_review_status',
           'check_passing',
           'leave_comment',
           'remove_labels',
           'set_draft',
           'get_head_ref',
           'trigger_test'
           )

github_cli = shutil.which('gh')

ReviewComment = namedtuple('ReviewComment', ['state', 'date', 'author'])
Comment = namedtuple('Comment', ['body', 'date', 'author'])

def get_status_json(pr_id, tags):
    # Change to PR to have access to relevant status
    cmds = [github_cli, 'pr', 'checkout', str(pr_id)]
    with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
        p.communicate()
    # Check status of PR
    cmds = [github_cli, 'pr', 'status', '--json', f'{tags},number']
    with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
        result, _ = p.communicate()
        print(result, pr_id)
    # Return to master branch
    cmds = ['git', 'checkout', 'master']
    with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
        p.communicate()

    data = json.loads(result)['currentBranch']
    if isinstance(data, list):
        relevant_data = [d for d in data if d['number'] == pr_id][0]
    else:
        assert data['number'] == pr_id
        relevant_data = data

    if ',' in tags:
        return relevant_data
    else:
        return relevant_data[tags]

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
        final_message = my_comments[0].state
        final_date = my_comments[0].date

        for c in my_comments:
            if c.state not in last_messages:
                last_messages[c.state] = c.date
            elif last_messages[c.state] < c.date:
                last_messages[c.state] = c.date

            if final_date < c.date:
                final_message = c.state
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



def is_draft():
    """
    Get the draft status of the PR.

    Use GitHub's command-line tool to get the draft status for
    the first PR related to this branch.

    Results
    -------
    bool : The draft status of the PR.
    """
    return get_status_json(pr_id, 'isDraft')



def get_review_status():
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
    for a in requests:
        if a not in review_status:
            review_status[a] = ReviewComment('UNRESPONSIVE', None, a)
    return review_status, requested_authors


def check_passing():
    """
    Check if tests are passing for this PR.

    Results
    -------
    bool : Indicates if tests are passed
    """

    # Wait till tests have finished (refresh every minute)
    cmds = [github_cli, 'pr', 'checks', '--required', '--watch', '-i', '60']

    with subprocess.Popen(cmds) as p:
        p.communicate()


    # Collect results
    checks = get_status_json(pr_id, 'statusCheckRollup')
    passing = all(c['conclusion'] == 'SUCCESS' for c in checks if c['name'] not in ('CoverageChecker', 'Check labels', 'Welcome'))

    return passing


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
        p.communicate()


def leave_non_repeat_comment(number, comment, allow_duplicate):
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
        _, last_comment, _ = check_previous_comments()
        ok_to_print = last_comment != comment
    else:
        ok_to_print = allow_duplicate

    if ok_to_print:
        cmds = [github_cli, 'pr', 'comment', str(number), '-b', comment]

        with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
            p.communicate()
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

    with subprocess.Popen(cmds) as p:
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

    with subprocess.Popen(cmds) as p:
        p.communicate()

def get_head_ref(number):
    return get_status_json(number, 'headRefName')

def trigger_test(number, workflow_name, head_ref):
    """
    Trigger the requested workflow.

    Use GitHub's command-line interface to trigger the named
    workflow via the workflow_dispatch trigger.

    Parameters
    ----------
    number : int
        The number of the PR.

    workflow_name : str
        The name of the workflow to be triggered.
    """
    print(workflow_name, head_ref)
    cmds = [github_cli, 'workflow', 'run', workflow_name, 'comments', '--ref', head_ref]

    with subprocess.Popen(cmds, stdout=subprocess.PIPE) as p:
        p.communicate()
